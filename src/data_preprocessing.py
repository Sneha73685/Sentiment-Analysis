import pandas as pd
import re
import string
from utils import clean_text

# Load raw data
df = pd.read_csv("data/raw_data.csv")

# Apply text cleaning
df["cleaned_text"] = df["text"].apply(clean_text)

# Save cleaned data
df.to_csv("data/cleaned_data.csv", index=False)
print("Data preprocessing completed and saved as cleaned_data.csv!")
