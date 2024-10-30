# AI-CodeSentry
AI-CodeSentry is an advanced static code analysis tool designed for automated detection of security vulnerabilities, potential threats and insecure patterns in codebases. It combines machine learning, hyperdimensional computing (Torchhd) and NLP-based embeddings to deliver powerful and contextual analysis across multiple programming languages.

# Features
- Automated Vulnerability Detection: Scans code files within specified directories, identifying insecure patterns and highlighting high-risk code.
- Neuro-Symbolic Knowledge Graph: Utilizes a neuro-symbolic knowledge graph with hyperdimensional computing to detect vulnerabilities with contextual understanding.
- NLP-Based Contextual Embedding: Leverages NLP (spaCy) to enhance pattern recognition, embedding contextual insights from docstrings, comments, and code structure.
- Pattern Visualization: Generates an interactive knowledge graph, visualizing code dependencies and vulnerability relationships.
- Adaptive Learning and Retraining: Accumulates labeled data for periodic retraining, enabling continuous improvement with new findings.
- Cross-Language Support: Parses Abstract Syntax Trees (AST) for multiple languages, identifying patterns in languages like Python, Java, JavaScript, and PHP.
- Comprehensive Reporting: Generates JSON-based risk reports, including severity levels, vulnerability details, and remediation suggestions.
- GPU Acceleration and Scalability: Supports GPU processing for faster analysis, ensuring high scalability.
- Flexible CLI and API-Ready: Command-line interface and modular structure enable integration into CI/CD workflows, with REST API readiness.

# Installation
Clone the repository:

```
git clone https://github.com/yourusername/AI-CodeSentry.git
cd AI-CodeSentry
```

# Install dependencies:

```
pip install -r requirements.txt
```

# Download NLP Model:

```
python -m spacy download en_core_web_sm
```
# Usage
To run AI-CodeSentry, specify the directory to analyze:

```
python main.py --directory <path_to_code_directory>
```

Replace <path_to_code_directory> with the directory containing the code files you want to scan.

# Example Usage:

```
python main.py --directory ./my_project --pattern_path ./patterns.json --epochs 5 --batch_size 16 --learning_rate 1e-5
```

# Optional Arguments:
--pattern_path: Path to the JSON file with custom vulnerability patterns.
--model_dir: Directory to save the retrained model and tokenizer.
--epochs: Number of epochs for retraining.
--batch_size: Batch size for analysis.
--learning_rate: Learning rate for retraining.
--extensions: File extensions to analyze (e.g., .py, .java, .js).

# Outputs
- JSON Report: A detailed report (analysis_report.json) is generated, documenting flagged vulnerabilities, severity levels, and explanation summaries.
- Knowledge Graph: Generates a visualization (knowledge_graph.png) illustrating detected vulnerabilities and code relationships.

# Contributing
Contributions are welcome! If you have ideas, bug fixes, or suggestions, please open an issue or submit a pull request.

# License
This project is licensed under the MIT License.

