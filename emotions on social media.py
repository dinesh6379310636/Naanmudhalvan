import streamlit as st

# Dependency check
required_libraries = {
    'pandas': 'pandas',
    'nltk': 'nltk',
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
        st.code("pip install pandas nltk scikit-learn streamlit requests joblib")
        st.stop()

# Import libraries
import pandas as pd
import nltk
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import requests
import joblib

# Attempt to import NLTK resources with fallback
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words('english'))
    st.write("NLTK resources loaded successfully.")
    nltk_available = True
except Exception as e:
    st.warning(f"Failed to load NLTK resources: {str(e)}. Falling back to basic preprocessing.")
    nltk_available = False
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}

# Function to preprocess text with simplified logic
def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        if nltk_available:
            tokens = word_tokenize(text)
        else:
            tokens = text.split()
        # Less aggressive filtering
        tokens = [word for word in tokens if word not in stop_words]
        processed = ' '.join(tokens) if tokens else text
        st.write(f"Debug - Processed text: '{processed}'")
        return processed
    except Exception as e:
        st.error(f"Error preprocessing text: {str(e)}")
        return text

# Function to fetch dataset from GitHub with retries
def fetch_dataset_from_github(github_url, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = requests.get(github_url, timeout=10)
            response.raise_for_status()
            st.write(f"Successfully fetched dataset from GitHub on attempt {attempt + 1}.")
            return response.content
        except requests.exceptions.RequestException as e:
            st.warning(f"Attempt {attempt + 1} failed to fetch dataset: {str(e)}")
            if attempt < retries - 1:
                st.write(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                st.warning("Failed to fetch dataset from GitHub after all retries.")
                st.write("Possible causes:")
                st.write("- Incorrect URL: Ensure it starts with https://raw.githubusercontent.com and points to the correct file.")
                st.write("- Repository is private: You'll need a personal access token.")
                st.write("- Network issues: Check your internet connection.")
                return None

# Function to load dataset with flexible delimiter handling
def load_dataset(file_content=None, delimiter=';'):
    try:
        if not file_content:
            return None
        data = []
        lines = file_content.decode('utf-8').splitlines()

        if not lines:
            st.warning("Dataset is empty.")
            return None

        st.write("First 5 lines of the dataset (for debugging):")
        st.write(lines[:5])

        invalid_lines = 0
        for line in lines:
            if line.strip():
                # Try multiple delimiters if the specified one doesn't work
                possible_delimiters = [delimiter, ';', ',', '\t']
                parts = None
                for d in possible_delimiters:
                    if d in line:
                        parts = line.strip().split(d)
                        if len(parts) == 2:
                            break
                if not parts or len(parts) != 2:
                    st.warning(f"Skipping invalid line (expected 2 parts): {line.strip()}")
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
            st.warning("Dataset contains no valid data after parsing.")
            return None
        return df
    except Exception as e:
        st.warning(f"Error loading dataset: {str(e)}")
        return None

# Default small dataset as fallback
default_dataset = [
    "I feel so happy today;joy",
    "This is making me sad;sadness",
    "I am so angry right now;anger",
    "I love this so much;love",
    "This is surprising;surprise"
]

# GitHub URL and delimiter inputs
github_url = st.text_input("Enter the raw GitHub URL for your dataset (e.g., https://raw.githubusercontent.com/yourusername/your-repo/main/data/emotions_data.txt)", value="")
delimiter = st.text_input("Delimiter for dataset (default is ';')", value=';')

# Load dataset
df = None
if github_url:
    dataset_content = fetch_dataset_from_github(github_url)
    if dataset_content:
        df = load_dataset(dataset_content, delimiter=delimiter)

# Fallback to default dataset if GitHub fails
if df is None:
    st.write("Using default dataset with 5 samples as fallback.")
    df = load_dataset(file_content='\n'.join(default_dataset).encode('utf-8'), delimiter=delimiter)

if df is None:
    st.error("Failed to load any dataset. Please check your GitHub URL and dataset format.")
    st.stop()

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

# Load or train model
vectorizer_path = 'vectorizer.pkl'
emotion_model_path = 'emotion_model.pkl'

try:
    if os.path.exists(vectorizer_path) and os.path.exists(emotion_model_path):
        vectorizer = joblib.load(vectorizer_path)
        emotion_model = joblib.load(emotion_model_path)
        X_tfidf = vectorizer.transform(df['processed_text'])
        st.write("Loaded pre-trained model and vectorizer.")
    else:
        X = df['processed_text']
        y_emotion = df['emotion']

        if len(df) < 2:
            st.error("Not enough data to train the model. At least 2 samples are required.")
            st.stop()

        vectorizer = TfidfVectorizer(max_features=5000)
        X_tfidf = vectorizer.fit_transform(X)
        st.write("TF-IDF vectorization complete. Shape of X_tfidf:", X_tfidf.shape)

        X_train, X_test, y_train_emotion, y_test_emotion = train_test_split(X_tfidf, y_emotion, test_size=0.2, random_state=42)

        if X_train.shape[0] < 1 or X_test.shape[0] < 1:
            st.error("Not enough data for train-test split. Please provide more valid samples.")
            st.stop()

        emotion_model = LogisticRegression(max_iter=1000)
        emotion_model.fit(X_train, y_train_emotion)
        st.write("Emotion model trained successfully.")

        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(emotion_model, emotion_model_path)
        st.write("Model and vectorizer saved for future use.")
except Exception as e:
    st.error(f"Error during model training or loading: {str(e)}")
    st.stop()

# Streamlit App
st.title("Emotion Detection in Social Media Conversations")
st.write("Enter a social media post to analyze its emotion.")

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

        st.subheader("Analysis Results")
        st.write(f"**Predicted Emotion**: {predicted_emotion}")

        emotion_probs = emotion_model.predict_proba(input_tfidf)[0]
        emotion_labels = emotion_model.classes_
        st.subheader("Emotion Confidence Scores")
        for label, prob in zip(emotion_labels, emotion_probs):
            st.write(f"{label}: {prob:.2%}")
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
