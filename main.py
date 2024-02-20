import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def read_files_from_directory(directory, extensions):
    file_contents = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line_num, line in enumerate(lines, start=1):
                            if line.strip():  # Skip empty lines
                                file_contents.append((file_path, line_num, line))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return file_contents

def analyze_code(file_contents, tokenizer, model, device):
    predictions = []
    for file_path, line_num, line in file_contents:
        inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
            predictions.append((file_path, line_num, line, prediction))
    return predictions

def interpret_predictions(predictions):
    for file_path, line_num, line, prediction in predictions:
        if prediction[1] > prediction[0]:  # Assuming index 1 corresponds to insecure
            print(f"Insecure Code Detected in {file_path} on line {line_num}: {line.strip()}")
            print(f"Explanation: This line of code may contain security vulnerabilities or bad practices.\n")

def retrain_model(labeled_data, tokenizer, model, device):
    inputs = tokenizer([data[0] for data in labeled_data], padding=True, truncation=True, return_tensors="pt", max_length=512)
    labels = torch.tensor([data[1] for data in labeled_data])
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = labels.to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(3):  # Adjust the number of epochs as needed
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Save the retrained model
    model.save_pretrained("/models/pretrained")
    tokenizer.save_pretrained("/tokenizer/pretrained")

def accumulate_labeled_data(predictions):
    labeled_data = []
    for file_path, line_num, line, prediction in predictions:
        # Example: Automated labeling based on model's confidence
        if prediction[1] > prediction[0]:  # If model predicts insecure code with high confidence
            labeled_data.append((line, 1))  # Label as insecure
        # You can add more criteria for labeling, such as manual review by experts
    return labeled_data

def main(directory_to_analyze):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("mrm8488/codebert-base-finetuned-detect-insecure-code")
    model = AutoModelForSequenceClassification.from_pretrained("mrm8488/codebert-base-finetuned-detect-insecure-code").to(device)

    print(f"Analyzing code in directory: {directory_to_analyze}")
    extensions = ['.py', '.java', '.js', '.php', '.ts', '.tsx', '.jsx']
    file_contents = read_files_from_directory(directory_to_analyze, extensions)

    if file_contents:
        predictions = analyze_code(file_contents, tokenizer, model, device)
        interpret_predictions(predictions)

        # Accumulate labeled data for retraining
        labeled_data = accumulate_labeled_data(predictions)

        # Example: Retrain the model with new labeled data periodically
        if len(labeled_data) > 0:
            print("Accumulated new labeled data. Retraining the model...")
            retrain_model(labeled_data, tokenizer, model, device)
    else:
        print("No code files found in the directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a directory of code files for potential security issues.")
    parser.add_argument("--directory", type=str, required=True, help="Path to the directory containing code files to analyze.")
    args = parser.parse_args()

    main(args.directory)
