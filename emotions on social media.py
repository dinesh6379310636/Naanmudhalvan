import streamlit as st

# Dependency check (removed kaggle)
required_libraries = {
    'pandas': 'pandas',
    'nltk': 'nltk',
    'sklearn': 'scikit-learn',
    'streamlit': 'streamlit',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'joblib': 'joblib'
}

for lib_name, pkg_name in required_libraries.items():
    try:
        __import__(lib_name)
    except ImportError:
        st.error(f"Required library '{pkg_name}' is not installed.")
        st.write("Please install all required dependencies by running:")
        st.code("pip install pandas nltk scikit-learn streamlit matplotlib seaborn numpy scipy joblib")
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
import os
import joblib

# Pre-download NLTK resources with error handling
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    st.write("NLTK resources loaded successfully.")
except Exception as e:
    st.error(f"Failed to download NLTK resources: {str(e)}. Please ensure you have an internet connection and run: nltk.download('punkt'), nltk.download('stopwords') locally.")
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

# Function to load dataset from a text file
def load_dataset(file_path=None, file_content=None):
    try:
        data = []
        if file_path:  # Local file
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        elif file_content:  # Uploaded file
            lines = file_content.decode('utf-8').splitlines()
        else:
            st.error("No dataset provided. Please specify a file path or upload a dataset.")
            st.stop()

        if not lines:
            st.error("Dataset is empty.")
            st.stop()

        for line in lines:
            if line.strip():
                if ';' not in line:
                    st.warning(f"Skipping malformed line: {line.strip()}")
                    continue
                text, emotion = line.strip().split(';')
                if not text or not emotion:
                    st.warning(f"Skipping invalid line: {line.strip()}")
                    continue
                data.append([text, emotion])
        
        df = pd.DataFrame(data, columns=['text', 'emotion'])
        if df.empty:
            st.error("Dataset contains no valid data after parsing.")
            st.stop()
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

# Dataset loading logic
data_dir = './data/'
dataset_file = os.path.join(data_dir, 'emotions_data.txt')

# Add file upload option for deployment
uploaded_file = st.file_uploader("Upload your dataset (emotions_data.txt)", type=['txt'])

if uploaded_file:
    # Use uploaded file (e.g., for Streamlit Cloud)
    df = load_dataset(file_content=uploaded_file.read())
    st.write("Dataset loaded from uploaded file.")
elif os.path.exists(dataset_file):
    # Use local file
    df = load_dataset(file_path=dataset_file)
    st.write("Dataset loaded from local file: data/emotions_data.txt")
else:
    st.error("Dataset file not found at data/emotions_data.txt, and no file was uploaded.")
    st.write("Please place emotions_data.txt in the ./data/ folder, or upload it using the file uploader above.")
    st.write("Expected format: Each line should be 'text;emotion' (e.g., 'I feel so happy today;joy').")
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
