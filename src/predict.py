import joblib

# Load trained model & vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def predict_sentiment(text):
    """Predict sentiment for a given text input."""
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[prediction]

# Example usage
if __name__ == "__main__":
    sample_text = "I absolutely love this product!"
    print(f"Predicted Sentiment: {predict_sentiment(sample_text)}")
