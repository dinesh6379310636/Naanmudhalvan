import streamlit as st

# Dependency check
required_libraries = {
    'pandas': 'pandas',
    'nltk': 'nltk',
    'sklearn': 'scikit-learn',
    'streamlit': 'streamlit',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'joblib': 'joblib',
    'requests': 'requests'
}

for lib_name, pkg_name in required_libraries.items():
    try:
        __import__(lib_name)
    except ImportError:
        st.error(f"Required library '{pkg_name}' is not installed.")
        st.write("Please install all required dependencies by running:")
        st.code("pip install pandas nltk scikit-learn streamlit matplotlib seaborn numpy scipy joblib requests")
        st.stop()

# Import libraries after dependency check
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import requests
import os

# Pre-download NLTK resources with error handling
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    st.write("NLTK resources loaded successfully.")
except Exception as e:
    st.error(f"Failed to download NLTK resources: {str(e)}. Please ensure you have an internet connection.")
    st.stop()

# Function to preprocess text
def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
        return ' '.join(tokens) if tokens else text
    except Exception as e:
        st.error(f"Error preprocessing text: {str(e)}")
        return text

# Function to fetch dataset from GitHub
def fetch_dataset_from_github(github_url):
    try:
        response = requests.get(github_url, timeout=10)
        response.raise_for_status()
        st.write("Successfully fetched dataset from GitHub.")
        return response.content
    except requests.exceptions.RequestException as e:
        st.warning(f"Failed to fetch dataset from GitHub: {str(e)}")
        return None

# Function to load dataset with configurable delimiter and encoding
def load_dataset(file_path=None, file_content=None, delimiter=';'):
    try:
        data = []
        encodings = ['utf-8', 'latin-1', 'utf-16']
        lines = None

        if file_path:
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        lines = file.readlines()
                    st.write(f"Successfully read file with {encoding} encoding.")
                    break
                except UnicodeDecodeError:
                    continue
            if lines is None:
                st.warning(f"Failed to read file {file_path} with tried encodings.")
                return None
        elif file_content:
            for encoding in encodings:
                try:
                    lines = file_content.decode(encoding).splitlines()
                    st.write(f"Successfully decoded dataset with {encoding} encoding.")
                    break
                except UnicodeDecodeError:
                    continue
            if lines is None:
                st.warning(f"Failed to decode dataset with tried encodings.")
                return None
        else:
            st.warning("No dataset content provided.")
            return None

        if not lines:
            st.warning("Dataset is empty.")
            return None

        st.write("First 5 lines of the dataset (for debugging):")
        st.write(lines[:5])

        invalid_lines = 0
        for line in lines:
            if line.strip():
                if delimiter not in line:
                    st.warning(f"Skipping line due to missing delimiter '{delimiter}': {line.strip()}")
                    invalid_lines += 1
                    continue
                parts = line.strip().split(delimiter)
                if len(parts) != 2:
                    st.warning(f"Skipping invalid line (expected 2 parts, got {len(parts)}): {line.strip()}")
                    invalid_lines += 1
                    continue
                text, emotion = parts
                if not text or not emotion:
                    st.warning(f"Skipping line with empty text or emotion: {line.strip()}")
                    invalid_lines += 1
                    continue
                data.append([text, emotion])

        if invalid_lines > 0:
            st.write(f"Skipped {invalid_lines} invalid lines.")

        df = pd.DataFrame(data, columns=['text', 'emotion'])
        if df.empty:
            st.warning("Dataset contains no valid data after parsing.")
            return None
        return df
    except Exception as e:
        st.warning(f"Error loading dataset: {str(e)}")
        return None

# Default small dataset
default_dataset = [
    "I feel so happy today;joy",
    "This is making me sad;sadness",
    "I am so angry right now;anger",
    "I love this so much;love",
    "This is surprising;surprise"
]

# Dataset loading logic
data_dir = './data/'
dataset_file = os.path.join(data_dir, 'emotions_data.txt')

# GitHub URL, file upload, and delimiter inputs
github_url = st.text_input("Enter the raw GitHub URL for your dataset (optional)", value="", help="Leave empty to use default dataset or local file.")
uploaded_file = st.file_uploader("Upload your dataset (optional)", type=['txt'])
delimiter = st.text_input("Delimiter for dataset (default is ';')", value=';')

# Load dataset
df = None

# Try GitHub first if URL is provided
if github_url:
    dataset_content = fetch_dataset_from_github(github_url)
    if dataset_content:
        df = load_dataset(file_content=dataset_content, delimiter=delimiter)

# Try local file if GitHub fails or no URL
if df is None and os.path.exists(dataset_file):
    df = load_dataset(file_path=dataset_file, delimiter=delimiter)

# Try uploaded file if local file fails
if df is None and uploaded_file:
    df = load_dataset(file_content=uploaded_file.read(), delimiter=delimiter)

# Use default dataset if all else fails
if df is None:
    st.write("Using default dataset with 5 samples.")
    df = load_dataset(file_content='\n'.join(default_dataset).encode('utf-8'), delimiter=delimiter)

if df is None:
    st.error("Failed to load any dataset. Please check your GitHub URL, local file, or uploaded file.")
    st.stop()

# Display dataset info
st.write("Dataset loaded successfully. Total samples:", len(df))
st.write("Sample data:", df.head())

# Preprocess text
try:
    df['processed_text'] = df['text'].apply(preprocess_text)
    st.write("Text preprocessing complete. Sample processed text:", df['processed_text'].iloc[0])
except Exception as e:
    st.error(f"Error during text preprocessing: {str(e)}")
    st.stop()

# Map emotions to sentiment polarity
emotion_to_sentiment = {
    'sadness': 'negative',
    'anger': 'negative',
    'fear': 'negative',
    'joy': 'positive',
    'love': 'positive',
    'surprise': 'neutral'
}
df['sentiment'] = df['emotion'].map(emotion_to_sentiment)

# Load or train model
vectorizer_path = 'vectorizer.pkl'
emotion_model_path = 'emotion_model.pkl'
sentiment_model_path = 'sentiment_model.pkl'

try:
    if os.path.exists(vectorizer_path) and os.path.exists(emotion_model_path) and os.path.exists(sentiment_model_path):
        vectorizer = joblib.load(vectorizer_path)
        emotion_model = joblib.load(emotion_model_path)
        sentiment_model = joblib.load(sentiment_model_path)
        X_tfidf = vectorizer.transform(df['processed_text'])
        st.write("Loaded pre-trained models and vectorizer.")
    else:
        X = df['processed_text']
        y_emotion = df['emotion']
        y_sentiment = df['sentiment']

        vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = vectorizer.fit_transform(X)
        st.write("TF-IDF vectorization complete. Shape of X_tfidf:", X_tfidf.shape)

        X_train, X_test, y_train_emotion, y_test_emotion = train_test_split(X_tfidf, y_emotion, test_size=0.2, random_state=42)
        _, _, y_train_sentiment, y_test_sentiment = train_test_split(X_tfidf, y_sentiment, test_size=0.2, random_state=42)

        emotion_model = LogisticRegression(max_iter=1000)
        emotion_model.fit(X_train, y_train_emotion)
        st.write("Emotion model trained successfully.")

        sentiment_model = LogisticRegression(max_iter=1000)
        sentiment_model.fit(X_train, y_train_sentiment)
        st.write("Sentiment model trained successfully.")

        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(emotion_model, emotion_model_path)
        joblib.dump(sentiment_model, sentiment_model_path)
        st.write("Models and vectorizer saved for future use.")
except Exception as e:
    st.error(f"Error during model training or loading: {str(e)}")
    st.stop()

# Streamlit App
st.title("Decoding Emotions in Social Media Conversations")
st.write("Enter a social media post to analyze its emotion and sentiment.")

user_input = st.text_area("Enter your text:", "I feel so happy today!")

if st.button("Analyze"):
    try:
        if not user_input.strip():
            st.error("Please enter some text to analyze.")
            st.stop()

        processed_input = preprocess_text(user_input)
        st.write("Processed user input:", processed_input)

        if not processed_input.strip():
            st.error("Processed text is empty after preprocessing. Please try a different input.")
            st.stop()

        input_tfidf = vectorizer.transform([processed_input])

        predicted_emotion = emotion_model.predict(input_tfidf)[0]
        predicted_sentiment = sentiment_model.predict(input_tfidf)[0]

        st.subheader("Analysis Results")
        st.write(f"**Predicted Emotion**: {predicted_emotion}")
        st.write(f"**Predicted Sentiment**: {predicted_sentiment}")

        emotion_probs = emotion_model.predict_proba(input_tfidf)[0]
        emotion_labels = emotion_model.classes_
        st.subheader("Emotion Confidence Scores")
        for label, prob in zip(emotion_labels, emotion_probs):
            st.write(f"{label}: {prob:.2%}")
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")

st.subheader("Dataset Emotion Distribution")
try:
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='emotion', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.clf()
except Exception as e:
    st.error(f"Error rendering visualization: {str(e)}")
