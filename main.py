import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import concurrent.futures
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(1024)  # Check if it's a text file
    except UnicodeDecodeError:
        return False
    return True

def read_file_contents(file_path):
    """Read and return the contents of a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return file_path, f.read()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")

def read_files_from_directory(directory, extensions):
    """Use ThreadPoolExecutor to read files in parallel."""
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                if is_text_file(file_path):
                    file_paths.append(file_path)
                else:
                    logger.warning(f"Skipping non-text file: {file_path}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        file_contents = list(tqdm(executor.map(read_file_contents, file_paths), total=len(file_paths), desc="Reading files"))

    return (item for item in file_contents if item is not None)

def batch(iterable, n=1):
    """Yield successive n-sized chunks from an iterable."""
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]

def analyze_code(file_contents, tokenizer, model, device, batch_size):
    predictions = []
    for file_path, content in tqdm(file_contents, desc="Analyzing code"):
        lines = [line for line in content.split('\n') if line.strip()]
        for line_batch in batch(lines, batch_size):
            inputs = tokenizer(line_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                predictions_batch = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()
                predictions.extend([(file_path, line, prediction) for line, prediction in zip(line_batch, predictions_batch)])
    return predictions

def interpret_predictions(predictions):
    for file_path, line, prediction in predictions:
        if prediction[1] > prediction[0]:  # Assuming index 1 indicates "insecure"
            logger.warning(f"Insecure code detected in {file_path}: {line.strip()}")
            logger.info("Explanation: This line may contain security vulnerabilities.")

def train_and_evaluate_model(train_dataset, val_dataset, model, device, output_dir, epochs, learning_rate):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}, Loss: {avg_loss}")

        # Evaluation step
        model.eval()
        correct_predictions, total_predictions = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)

                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        accuracy = correct_predictions / total_predictions
        logger.info(f"Validation Accuracy: {accuracy}")

    model.save_pretrained(os.path.join(output_dir, "model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

def accumulate_labeled_data(predictions):
    return [(line, 1) for _, line, prediction in predictions if prediction[1] > prediction[0]]

def main(directory_to_analyze, model_dir, epochs, model_name, tokenizer_name, extensions, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if os.path.isdir(model_name) and os.path.isdir(tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=model_dir)
    model.to(device)

    file_contents = read_files_from_directory(directory_to_analyze, extensions)
    predictions = analyze_code(file_contents, tokenizer, model, device, batch_size)

    interpret_predictions(predictions)
    labeled_data = accumulate_labeled_data(predictions)

    if labeled_data:
        logger.info("Accumulated new labeled data. Training the model...")
        train_dataset = TensorDataset(torch.tensor(labeled_data[0]), torch.tensor(labeled_data[1]))
        val_dataset = TensorDataset(torch.tensor(labeled_data[0]), torch.tensor(labeled_data[1]))  # Placeholder for actual validation data
        train_and_evaluate_model(train_dataset, val_dataset, model, device, model_dir, epochs, learning_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze code files for security issues and retrain the model with new findings.")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing code files.")
    parser.add_argument("--model_dir", type=str, default="output", help="Directory to save retrained model and tokenizer.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for analysis.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training.")
    parser.add_argument("--model", type=str, default="mrm8488/codebert-base-finetuned-detect-insecure-code", help="Model name or path.")
    parser.add_argument("--tokenizer", type=str, default="mrm8488/codebert-base-finetuned-detect-insecure-code", help="Tokenizer name or path.")
    parser.add_argument("--extensions", nargs="+", default=['.py', '.java', '.js', '.php', '.ts', '.tsx', '.jsx'], help="List of file extensions to analyze.")

    args = parser.parse_args()

    main(args.directory, args.model_dir, args.epochs, args.model, args.tokenizer, args.extensions, args.batch_size, args.learning_rate)
