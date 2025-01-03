import re
import joblib
import sys
import warnings
from nltk import download as nltk_download
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Text normalization function
def normalize_text(text):
    """Normalizes text for machine learning processing.

    Args:
        str: The text to be normalized.

    Returns:
        str: The normalized text.
    """

    # Preprocessing the text
    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    normalized_text = ' '.join(tokens)

    return normalized_text


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

    # Preprocess the text
    new_text = normalize_text(text_string)

    # Convert the text into a list (single sample)
    new_text_list = [new_text]

    # Load the saved model using joblib
    loaded_pipeline = joblib.load('./.github/workflows/mergeable-predicter/my_model.pkl')

    # Use the loaded pipeline to predict the class label
    prediction = loaded_pipeline.predict(new_text_list)[0]
    
    # print(prediction)
    # return prediction
    if prediction == "Merged":
        print("True")
    else:
        print("False")


if __name__ == "__main__":
    main()