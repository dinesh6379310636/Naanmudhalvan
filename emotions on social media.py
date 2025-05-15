import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Function to preprocess text data
def preprocess_text(text):
    original_text = text
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'<.*?>|http\S+', '', text)  # Remove HTML tags and URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    text = ' '.join(text.split())  # Remove extra whitespace
    if not text.strip():
        st.write(f"Warning: Text became empty after preprocessing: {original_text}")
    return text

# Load your dataset (adjust based on your actual files)
data = []
files = ['val.txt']  # Add your other files here: ['val.txt', 'file2.txt', 'file3.txt']
for file_name in files:
    try:
        with open(file_name, 'r') as file:
            for line in file:
                if line.strip():
                    text, emotion = line.strip().split(';')
                    data.append({'text': text, 'label': emotion})
    except Exception as e:
        st.error(f"Error loading file {file_name}: {str(e)}")

df = pd.DataFrame(data)
st.write("Initial DataFrame:")
st.write(df.head())
st.write("Number of rows:", len(df))

# Step 1: Clean the data
df['text'] = df['text'].fillna("")
df['text'] = df['text'].apply(preprocess_text)
df = df[df['text'].str.strip() != ""].reset_index(drop=True)

# Display cleaned data
st.write("Cleaned DataFrame:")
st.write(df.head())
st.write("Number of rows after cleaning:", len(df))

# Step 2: Check if DataFrame is empty
if df.empty:
    st.error("Error: No valid text data remains after preprocessing. Please check your input data.")
else:
    # Step 3: Test vectorization on a small subset first
    st.write("Testing vectorization on first 5 rows...")
    small_subset = df['text'].head(5)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=None, min_df=1, max_df=1.0)
    try:
        X_small = vectorizer.fit_transform(small_subset)
        st.write("Small subset vectorization successful. Shape:", X_small.shape)
    except ValueError as e:
        st.error(f"Error during small subset vectorization: {str(e)}")

    # Step 4: Vectorize the full dataset
    st.write("Vectorizing full dataset...")
    try:
        X = vectorizer.fit_transform(df['text'])
        st.write("Shape of transformed data (X):", X.shape)
        st.write("Vocabulary size:", len(vectorizer.vocabulary_))
        st.write("Sample feature names:", vectorizer.get_feature_names_out()[:10])

        # Step 5: Proceed with labels (if needed)
        y = df['label']
        st.write("Labels:", y.head())
    except ValueError as e:
        st.error(f"Error during full dataset vectorization: {str(e)}")
        # Identify problematic rows
        st.write("Identifying problematic rows...")
        for idx, text in enumerate(df['text']):
            try:
                vectorizer.fit_transform([text])
            except ValueError as e:
                st.error(f"Error at row {idx}: {text}")
                st.error(f"Error message: {str(e)}")
