import streamlit as st
import joblib
from utils import clean_text

# Load trained model & vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Streamlit UI
st.title("📝 Social Media Sentiment Analyzer")
st.subheader("Enter a sentence to analyze sentiment")

# Input text box
user_input = st.text_area("Enter your text here...")

if st.button("Analyze Sentiment"):
    if user_input:
        cleaned_text = clean_text(user_input)
        text_tfidf = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_tfidf)[0]
        sentiment_map = {0: "😡 Negative", 1: "😐 Neutral", 2: "😊 Positive"}
        
        st.write(f"**Predicted Sentiment:** {sentiment_map[prediction]}")
    else:
        st.write("Please enter some text to analyze.")

