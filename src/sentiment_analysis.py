import pandas as pd
import joblib
from utils import clean_text

# Load trained model & vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Load dataset
df = pd.read_csv("data/raw_data.csv")  # Update file name if needed

# Clean text data
df["cleaned_text"] = df["text"].apply(clean_text)

# Transform text to TF-IDF format
X_tfidf = vectorizer.transform(df["cleaned_text"])

# Predict sentiment
df["predicted_sentiment"] = model.predict(X_tfidf)

# Map numeric labels to sentiment labels
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
df["sentiment_label"] = df["predicted_sentiment"].map(sentiment_map)

# Save results
df.to_csv("data/sentiment_results.csv", index=False)
print("Sentiment analysis completed! Results saved in 'data/sentiment_results.csv'.")
