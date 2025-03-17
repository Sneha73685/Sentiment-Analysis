import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
with open("models/sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("models/tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocess input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Streamlit App UI
st.title("Social Media Sentiment Analysis")
st.write("Enter a text snippet to analyze its sentiment:")

user_input = st.text_area("Enter text here:")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        processed_text = preprocess_text(user_input)
        text_vector = vectorizer.transform([processed_text])
        prediction = model.predict(text_vector)[0]
        
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        st.write(f"### Sentiment: {sentiment_map[prediction]}")
    else:
        st.warning("Please enter some text to analyze.")

# Run the app using: streamlit run app.py
