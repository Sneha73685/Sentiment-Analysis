import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/cleaned_data.csv")

# Features (X) and Labels (y)
X = df["cleaned_text"]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical format (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naïve Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model & vectorizer
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Model training complete! Model saved in models/ folder.")
