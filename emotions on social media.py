import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Function to preprocess text data
def preprocess_text(text):
    # Handle non-string values
    if not isinstance(text, str):
        text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags and URLs
    text = re.sub(r'<.*?>|http\S+', '', text)
    # Remove special characters and numbers, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Load your dataset (adjust this based on how you're loading your data)
# For example, if loading from "val.txt", you might need to parse it manually
# Here, I'll assume df is already loaded with 'text' and 'label' columns
# df = pd.read_csv('your_file.csv')  # Uncomment and adjust as needed

# For demonstration, let's simulate loading "val.txt" data
# If you have a different loading mechanism, replace this part
data = []
with open('val.txt', 'r') as file:  # Adjust path as needed
    for line in file:
        # Assuming each line is in the format: "text;emotion"
        if line.strip():
            text, emotion = line.strip().split(';')
            data.append({'text': text, 'label': emotion})
df = pd.DataFrame(data)

# Display initial data for debugging (Streamlit)
st.write("Initial DataFrame:")
st.write(df.head())
st.write("Number of rows:", len(df))

# Step 1: Clean the data
# Replace NaN with empty strings
df['text'] = df['text'].fillna("")
# Convert all entries to strings and preprocess
df['text'] = df['text'].apply(preprocess_text)
# Remove rows where text is empty after preprocessing
df = df[df['text'].str.strip() != ""].reset_index(drop=True)

# Display cleaned data for debugging
st.write("Cleaned DataFrame:")
st.write(df.head())
st.write("Number of rows after cleaning:", len(df))

# Step 2: Check if DataFrame is empty after cleaning
if df.empty:
    st.error("Error: No valid text data remains after preprocessing. Please check your input data.")
else:
    # Step 3: Initialize the vectorizer
    vectorizer = TfidfVectorizer(
        stop_words='english',  # Remove common English stop words
        max_features=5000,     # Limit the number of features to prevent memory issues
        min_df=1,              # Include words that appear in at least 1 document
        max_df=0.95            # Ignore words that appear in >95% of documents
    )

    # Step 4: Transform the text data
    try:
        # Transform the text into a sparse matrix (avoid .toarray() unless necessary)
        X = vectorizer.fit_transform(df['text'])
        st.write("Shape of transformed data (X):", X.shape)
        st.write("Vocabulary size:", len(vectorizer.vocabulary_))

        # If you need a dense array (e.g., for certain models), uncomment this:
        # X = X.toarray()
        # st.write("Shape of dense array:", X.shape)

        # Example: Display the feature names (vocabulary)
        st.write("Sample feature names:", vectorizer.get_feature_names_out()[:10])

        # Step 5: Proceed with your analysis (e.g., model training)
        # Example: If you have labels, you can proceed with classification
        y = df['label']  # Assuming 'label' column contains the emotions
        st.write("Labels:", y.head())

    except ValueError as e:
        st.error(f"Error during vectorization: {str(e)}")
        st.write("Check your data for empty or invalid entries.")
