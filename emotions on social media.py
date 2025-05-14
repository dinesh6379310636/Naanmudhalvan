import streamlit as st

# Dependency check
required_libraries = {
    'pandas': 'pandas',
    'sklearn': 'scikit-learn',
    'streamlit': 'streamlit',
    'requests': 'requests',
    'joblib': 'joblib'
}

for lib_name, pkg_name in required_libraries.items():
    try:
        __import__(lib_name)
    except ImportError:
        st.error(f"Required library '{pkg_name}' is not installed.")
        st.write("Please install all required dependencies by running:")
        st.code("pip install pandas scikit-learn streamlit requests joblib")
        st.stop()

# Import libraries
import pandas as pd
import re
import requests
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Function to preprocess text (basic, no NLTK)
def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = re.sub(r'\@w+|\#', '', text)  # Remove mentions and hashtags
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        words = text.split()
        # Basic stopword removal
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}
        words = [word for word in words if word not in stop_words and len(word) > 1]
        return ' '.join(words) if words else text
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
        st.error(f"Failed to fetch dataset from GitHub: {str(e)}")
        st.write("Please ensure the URL is correct and the repository is public.")
        st.stop()

# Function to load dataset
def load_dataset(file_content, delimiter=';'):
    try:
        data = []
        lines = file_content.decode('utf-8').splitlines()

        if not lines:
            st.error("Dataset is empty.")
            st.stop()

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
                if not text.strip() or not emotion.strip():
                    st.warning(f"Skipping line with empty text or emotion: {line.strip()}")
                    invalid_lines += 1
                    continue
                data.append([text.strip(), emotion.strip()])

        if invalid_lines > 0:
            st.write(f"Skipped {invalid_lines} invalid lines.")

        df = pd.DataFrame(data, columns=['text', 'emotion'])
        if df.empty:
            st.error("Dataset contains no valid data after parsing.")
            st.stop()
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

# GitHub URL and delimiter inputs
github_url = st.text_input("Enter the raw GitHub URL for your dataset (e.g., https://raw.githubusercontent.com/yourusername/your-repo/main/data/emotions_data.txt)", value="https://raw.githubusercontent.com/yourusername/your-repo/main/data/emotions_data.txt")
delimiter = st.text_input("Delimiter for dataset (default is ';')", value=';')

# Load dataset
if not github_url or "yourusername" in github_url:
    st.error("Please provide the correct raw GitHub URL for your dataset.")
    st.write("To get the raw GitHub URL:")
    st.write("1. Go to your GitHub repository.")
    st.write("2. Navigate to the file (e.g., data/emotions_data.txt).")
    st.write("3. Click the 'Raw' button.")
    st.write("4. Copy the URL.")
    st.stop()

dataset_content = fetch_dataset_from_github(github_url)
df = load_dataset(dataset_content, delimiter=delimiter)

# Display dataset info
st.write("Dataset loaded successfully. Total samples:", len(df))
st.write("Sample data:", df.head())

# Preprocess text
df['processed_text'] = df['text'].apply(preprocess_text)

# Check for empty processed text
df = df[df['processed_text'].str.strip() != '']
if df.empty:
    st.error("No valid data after preprocessing. All processed texts are empty. Please check the dataset content.")
    st.stop()

st.write("Text preprocessing complete. Sample processed text:", df['processed_text'].iloc[0])

# Map emotions to sentiments for sentiment analysis
emotion_to_sentiment = {
    'sadness': 'negative',
    'anger': 'negative',
    'fear': 'negative',
    'joy': 'positive',
    'love': 'positive',
    'surprise': 'neutral'
}
df['sentiment'] = df['emotion'].map(emotion_to_sentiment)

# Load or train models
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

        if len(df) < 2:
            st.error("Not enough data to train the model. At least 2 samples are required.")
            st.stop()

        vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = vectorizer.fit_transform(X)
        st.write("TF-IDF vectorization complete. Shape of X_tfidf:", X_tfidf.shape)

        X_train, X_test, y_train_emotion, y_test_emotion = train_test_split(X_tfidf, y_emotion, test_size=0.2, random_state=42)
        _, _, y_train_sentiment, y_test_sentiment = train_test_split(X_tfidf, y_sentiment, test_size=0.2, random_state=42)

        if X_train.shape[0] < 1 or X_test.shape[0] < 1:
            st.error("Not enough data for train-test split. Please provide more valid samples.")
            st.stop()

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
st.title("Decoding Emotions through Sentiment Analysis")
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
