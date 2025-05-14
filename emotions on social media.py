import os
import json
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from textblob import TextBlob
import joblib
from sklearn.preprocessing import LabelEncoder

# Custom CSS for styling (identical to sample)
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #34495e;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .prediction-box {
        background-color: #d3d3d3;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        margin-top: 20px;
        color: #2c3e50;
    }
    .prediction-box strong {
        color: #2c3e50;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 20px;
    }
    .metrics-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .metrics-table th, .metrics-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .metrics-table th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">Emotion Detection in Social Media</div>', unsafe_allow_html=True)

# Sidebar for user input
with st.sidebar:
    st.header("Input Social Media Post")
    st.markdown('<div class="section-header">Post Details</div>', unsafe_allow_html=True)
    user_post = st.text_area("Enter a social media post", height=150, placeholder="Type your post here...")

# Set up Kaggle API credentials
kaggle_api_token = {
    "username": st.secrets["kaggle"]["username"],
    "key": st.secrets["kaggle"]["key"]
}

os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump(kaggle_api_token, f)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# Load dataset from Kaggle
st.write("### Model Training and Evaluation")

with st.spinner("Downloading dataset..."):
    # Download the dataset using Kaggle API
    os.system("kaggle datasets download -d dair-ai/emotion --unzip -p ./data")

# Load the dataset (use train.txt, which is tab-separated)
data_path = "./data/train.txt"
df = pd.read_csv(data_path, sep='\t', names=['text', 'label'])  # Specify tab separator and column names

# Preprocess dataset
df.dropna(subset=['text', 'label'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Map numeric labels to emotion names (based on dair-ai/emotion dataset)
emotion_mapping = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}
df['emotion'] = df['label'].map(emotion_mapping)

# Encode emotion labels for training
label_encoder = LabelEncoder()
df['emotion_encoded'] = label_encoder.fit_transform(df['emotion'])

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['text']).toarray()
y = df['emotion_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Model training and evaluation
model_path = "emotion_model_xgb.pkl"
metrics_path = "emotion_metrics.json"

if not os.path.exists(model_path):
    with st.spinner("Training model..."):
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        
        y_pred = model.predict(X_test)
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1 Score': f1_score(y_test, y_pred, average='weighted')
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        st.write("Model Evaluation Results (computed during training):")
        st.write(metrics)

# Load model
model = joblib.load(model_path)

# Display evaluation metrics
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    st.write("#### Model Evaluation Metrics (XGBoost)")
    
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Metric', 'Value']
    
    metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}")

    metrics_html = metrics_df.to_html(index=False, classes='metrics-table')
    st.markdown(metrics_html, unsafe_allow_html=True)
else:
    st.write("Model evaluation metrics are not available. Please train the model first.")

# Predict emotion for user input
st.write("### Predict Emotion")
if st.button("Analyze Emotion"):
    if user_post.strip() == "":
        st.error("Please enter a social media post.")
    else:
        # Preprocess user input
        user_vector = vectorizer.transform([user_post]).toarray()
        
        # Predict emotion
        prediction = model.predict(user_vector)[0]
        predicted_emotion = label_encoder.inverse_transform([prediction])[0]
        
        # Get sentiment polarity using TextBlob
        sentiment = TextBlob(user_post).sentiment.polarity
        sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

        # Display results
        st.markdown(f"""
        <div class="prediction-box">
            <strong>Predicted Emotion:</strong> {predicted_emotion}<br>
            <strong>Sentiment:</strong> {sentiment_label}
        </div>
        """, unsafe_allow_html=True)
