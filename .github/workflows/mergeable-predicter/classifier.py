import sys
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    """Runs the text classification task from command line arguments.

    Expects the filename of the text file as an argument.

    Raises:
        SystemExit: If an incorrect number of arguments is provided or the file is not found.
    """
    warnings.filterwarnings('ignore', category=UserWarning)

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <text_string>")
        sys.exit(1)

    # Get the text string from the argument
    text_string = sys.argv[1]

    # Load tokenizer from local path
    tokenizer = AutoTokenizer.from_pretrained("./local_model/")

    # Load model from local path
    model = AutoModelForSequenceClassification.from_pretrained(
        "./local_model/",  # Directory containing the model and config.json
        trust_remote_code=True  # Enable this if the model uses custom layers
    )

    # Tokenize input
    inputs = tokenizer(text_string, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    outputs = model(**inputs)
    logits = outputs.logits

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Get the predicted class
    predicted_class = torch.argmax(probs).item()

    # Output prediction
    if predicted_class == 1:  # Assuming "1" corresponds to "Merged"
        print("True")
    else:
        print("False")

if __name__ == "__main__":
    main()
