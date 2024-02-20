# AI-CodeSentry
AI-CodeSentry is a powerful tool for automated detection of security vulnerabilities and potential threats in codebases. It leverages machine learning techniques and pre-trained models to provide robust analysis capabilities across various programming languages.

## Features
- **Automated Scanning:** Quickly scan code files in specified directories for potential security issues.
- **Machine Learning Integration:** Utilizes pre-trained models for sequence classification to identify insecure code patterns.
- **Interpretation:** Interpret model predictions to highlight suspicious code lines with detailed explanations.
- **Retraining:** Accumulate labeled data for periodic retraining of the model to enhance accuracy.
- **Scalability:** Supports GPU acceleration for faster analysis, ensuring scalability.
- **Easy Integration:** Flexible command-line interface for seamless integration into existing workflows.

## Installation
To install AI-CodeSentry and its dependencies, run:

```
pip install -r requirements.txt
```

## Usage
```
python main.py --directory <path_to_code_directory>
```
Replace <path_to_code_directory> with the directory containing the code files you want to analyze.

## Contributing
Contributions are welcome! If you have any ideas, bug fixes, or suggestions, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
