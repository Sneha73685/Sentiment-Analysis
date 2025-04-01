import re
import string

def clean_text(text):
    """Clean text by removing punctuation, special characters, and converting to lowercase."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"\@\w+|\#", '', text)  # Remove mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.strip()
    return text
