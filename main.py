import os
import json
import argparse
import torch
import torchhd
import concurrent.futures
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
import ast
from datetime import datetime
from networkx import Graph, draw
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set dimensionality for hypervectors
D = 10000

def load_patterns(json_path="patterns.json"):
    """Load known vulnerability patterns from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading patterns: {e}")
        return []

def is_text_file(file_path):
    """Check if the file is a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(1024)
    except UnicodeDecodeError:
        return False
    return True

def read_file_contents(file_path):
    """Read contents of a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return file_path, f.read()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")

def read_files_from_directory(directory, extensions):
    """Read files in parallel using multiple threads."""
    file_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if any(file.endswith(ext) for ext in extensions) and is_text_file(os.path.join(root, file))
    ]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        file_contents = list(tqdm(executor.map(read_file_contents, file_paths), total=len(file_paths), desc="Reading files"))
    return (item for item in file_contents if item is not None)

def parse_code_structure(code):
    """Parse code using AST to capture symbolic relations and bindings."""
    parsed_code = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.Call, ast.Assign, ast.Import, ast.ImportFrom)):
                parsed_code.append(ast.dump(node))
    except SyntaxError as e:
        logger.warning(f"Syntax error during AST parsing: {e}")
    return parsed_code

def encode_line(line, D):
    """Encode a line of code into a hypervector by combining token hypervectors."""
    tokens = line.strip().split()
    token_vectors = [torchhd.random(1, D).squeeze(0) for _ in tokens]
    if token_vectors:
        line_hv = token_vectors[0]
        for token_hv in token_vectors[1:]:
            line_hv = torchhd.bundle(line_hv, token_hv)
    else:
        line_hv = torchhd.random(1, D).squeeze(0)
    return line_hv

def create_knowledge_graph(known_patterns):
    """Create a neuro-symbolic knowledge graph with insecure patterns as hypervectors."""
    pattern_vectors = [encode_line(pattern['pattern'], D) for pattern in known_patterns]
    knowledge_graph = torch.stack(pattern_vectors)
    return knowledge_graph

def analyze_code(file_contents, knowledge_graph, patterns, threshold=0.5):
    """Analyze code for similarity to insecure patterns."""
    predictions = []
    for file_path, content in tqdm(file_contents, desc="Analyzing code"):
        parsed_structure = parse_code_structure(content)
        for line in parsed_structure:
            line_vector = encode_line(line, D)
            similarity = torchhd.cosine_similarity(line_vector.unsqueeze(0), knowledge_graph)
            max_similarity, idx = torch.max(similarity, dim=1)
            if max_similarity.item() > threshold:
                matched_pattern = patterns[idx.item()]
                predictions.append({
                    "file": file_path,
                    "line": line,
                    "pattern": matched_pattern['description'],
                    "severity": matched_pattern.get("severity", "medium"),
                    "similarity": max_similarity.item(),
                    "timestamp": datetime.now().isoformat()
                })
                logger.warning(f"Insecure code detected in {file_path}: {line.strip()}")
                logger.info(f"Matched pattern '{matched_pattern['description']}' with similarity {max_similarity.item()}")
    return predictions

def visualize_knowledge_graph(predictions, output_path="knowledge_graph.png"):
    """Visualize detected vulnerabilities in a knowledge graph."""
    graph = Graph()
    for pred in predictions:
        graph.add_node(pred["file"])
        graph.add_edge(pred["file"], pred["pattern"])
    plt.figure(figsize=(10, 8))
    if graph.nodes:
        draw(graph, with_labels=True, node_color='skyblue', edge_color='gray')
    else:
        plt.text(0.5, 0.5, "No Vulnerabilities Detected", ha="center", va="center", fontsize=12)
    plt.savefig(output_path)
    logger.info(f"Knowledge graph saved to {output_path}")

def interpret_predictions(predictions):
    """Log detailed explanations of flagged lines."""
    for pred in predictions:
        logger.warning(f"Insecure code detected in {pred['file']}: {pred['line'].strip()}")
        logger.info(f"Explanation: Pattern matched - '{pred['pattern']}' with similarity {pred['similarity']}")

def accumulate_labeled_data(predictions):
    """Accumulate labeled data for training from predictions."""
    lines = [pred["line"] for pred in predictions]
    labels = [1 for _ in predictions]
    return lines, labels

def train_and_evaluate_model(train_data, model, tokenizer, device, output_dir, epochs, learning_rate):
    """Train and evaluate the model using new labeled data."""
    inputs = tokenizer(train_data[0], return_tensors="pt", padding=True, truncation=True, max_length=512)
    labels = torch.tensor(train_data[1])
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

    model.save_pretrained(os.path.join(output_dir, "model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    logger.info(f"Model saved to {output_dir}")

def generate_report(predictions, report_path="analysis_report.json"):
    """Generate a JSON report with detected vulnerabilities and explanations."""
    report = {"timestamp": datetime.now().isoformat(), "predictions": predictions}
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    logger.info(f"Report generated: {report_path}")

def main(directory_to_analyze, model_dir, epochs, pattern_path, extensions, batch_size, learning_rate, threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load patterns and create knowledge graph
    patterns = load_patterns(pattern_path)
    if not patterns:
        logger.error("No patterns loaded. Exiting.")
        return

    knowledge_graph = create_knowledge_graph(patterns).to(device)

    # Initialize tokenizer and model for retraining
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to(device)

    # Read files and analyze code
    file_contents = read_files_from_directory(directory_to_analyze, extensions)
    predictions = analyze_code(file_contents, knowledge_graph, patterns, threshold)

    interpret_predictions(predictions)
    lines, labels = accumulate_labeled_data(predictions)

    if lines:
        logger.info("Accumulated new labeled data. Training the model...")
        train_data = (lines, labels)
        train_and_evaluate_model(train_data, model, tokenizer, device, model_dir, epochs, learning_rate)

    # Generate report and visualize knowledge graph
    generate_report(predictions)
    visualize_knowledge_graph(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze code files for security issues using Torchhd and neuro-symbolic knowledge graphs.")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing code files.")
    parser.add_argument("--model_dir", type=str, default="output", help="Directory to save retrained model and tokenizer.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for analysis.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training.")
    parser.add_argument("--pattern_path", type=str, default="patterns.json", help="Path to JSON file with vulnerability patterns.")
    parser.add_argument("--extensions", nargs="+", default=['.py', '.java', '.js', '.php', '.ts', '.tsx', '.jsx'], help="List of file extensions to analyze.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold for pattern matching.")

    args = parser.parse_args()
    main(args.directory, args.model_dir, args.epochs, args.pattern_path, args.extensions, args.batch_size, args.learning_rate, args.threshold)
