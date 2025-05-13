import pandas as pd
import nltk
import re
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import zipfile
import kaggle

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Function to load dataset
def load_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                text, emotion = line.strip().split(';')
                data.append([text, emotion])
    return pd.DataFrame(data, columns=['text', 'emotion'])

# Function to download dataset using Kaggle API
def download_kaggle_dataset():
    dataset = 'praveengovi/emotions-dataset-for-nlp'
    data_dir = './data/'
    os.makedirs(data_dir, exist_ok=True)
    
    # Configure Kaggle API with secrets
    try:
        os.environ['KAGGLE_USERNAME'] = st.secrets['kaggle']['username']
        os.environ['KAGGLE_KEY'] = st.secrets['kaggle']['key']
    except KeyError:
        st.error("Kaggle API credentials not found in Streamlit secrets. Please configure secrets.toml or Streamlit Cloud secrets.")
        st.stop()
    
    # Download and unzip dataset
    kaggle.api.dataset_download_files(dataset, path=data_dir, unzip=True)

# Check if dataset files exist
data_dir = './data/'
train_file = os.path.join(data_dir, 'train.txt')
test_file = os.path.join(data_dir, 'test.txt')
val_file = os.path.join(data_dir, 'val.txt')

if not all(os.path.exists(f) for f in [train_file, test_file, val_file]):
    st.write("Dataset files not found. Downloading from Kaggle...")
    download_kaggle_dataset()
    st.write("Download complete.")

# Load and combine datasets
try:
    train_df = load_dataset(train_file)
    test_df = load_dataset(test_file)
    val_df = load_dataset(val_file)
    df = pd.concat([train_df, test_df, val_df], ignore_index=True)
except FileNotFoundError:
    st.error("Failed to load dataset files. Please ensure the dataset is downloaded correctly.")
    st.stop()

# Preprocess text
df['processed_text'] = df['text'].apply(preprocess_text)

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

# Train model
X = df['processed_text']
y_emotion = df['emotion']
y_sentiment = df['sentiment']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train_emotion, y_test_emotion = train_test_split(X_tfidf, y_emotion, test_size=0.2, random_state=42)
_, _, y_train_sentiment, y_test_sentiment = train_test_split(X_tfidf, y_sentiment, test_size=0.2, random_state=42)

# Train Logistic Regression models
emotion_model = LogisticRegression(max_iter=1000)
emotion_model.fit(X_train, y_train_emotion)

sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X_train, y_train_sentiment)

# Streamlit App
st.title("Decoding Emotions in Social Media Conversations")
st.write("Enter a social media post to analyze its emotion and sentiment.")

# User input
user_input = st.text_area("Enter your text:", "I feel so happy today!")

if st.button("Analyze"):
    # Preprocess user input
    processed_input = preprocess_text(user_input)
    input_tfidf = vectorizer.transform([processed_input])
    
    # Predict emotion and sentiment
    predicted_emotion = emotion_model.predict(input_tfidf)[0]
    predicted_sentiment = sentiment_model.predict(input_tfidf)[0]
    
    # Display results
    st.subheader("Analysis Results")
    st.write(f"**Predicted Emotion**: {predicted_emotion}")
    st.write(f"**Predicted Sentiment**: {predicted_sentiment}")
    
    # Display confidence scores
    emotion_probs = emotion_model.predict_proba(input_tfidf)[0]
    emotion_labels = emotion_model.classes_
    st.subheader("Emotion Confidence Scores")
    for label, prob in zip(emotion_labels, emotion_probs):
        st.write(f"{label}: {prob:.2%}")

# Visualization: Emotion distribution
st.subheader("Dataset Emotion Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x='emotion', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)
</x(aiArtifact>

---

### Step 4: Instructions to Run the Code
1. **Install Dependencies**:
   ```bash
   pip install pandas nltk scikit-learn streamlit matplotlib seaborn kaggle
   ```

2. **Set Up Streamlit Secrets**:
   - **Locally**:
     - Create `.streamlit/secrets.toml` in your project directory.
     - Add your Kaggle API credentials (see Step 2).
   - **For Deployment** (e.g., Streamlit Cloud):
     - Go to your app’s settings on Streamlit Cloud.
     - Add the secrets in the "Secrets" section as shown above.

3. **Save the Code**:
   - Save the code as `emotion_sentiment_analysis_with_secrets.py`.

4. **Run the Streamlit App**:
   ```bash
   streamlit run emotion_sentiment_analysis_with_secrets.py
   ```
   - The app will check for dataset files. If missing, it will download them using the Kaggle API credentials from Streamlit secrets.
   - The web interface will be available at `http://localhost:8501`.

5. **Deploy on Streamlit Cloud** (Optional):
   - Push your code to a GitHub repository.
   - Connect Streamlit Cloud to your repository.
   - Add the Kaggle API credentials in the app’s secrets settings.
   - Deploy the app.

---

### Step 5: Why Streamlit Secrets with Kaggle API is Convenient
- **Security**: Storing the Kaggle API key in Streamlit secrets avoids hardcoding sensitive information in the code, reducing the risk of accidental exposure.
- **Automation**: The dataset is downloaded automatically if missing, making the app portable and suitable for deployment.
- **Deployment-Friendly**: Streamlit Cloud supports secrets management, so the same code works locally and in production without modification.
- **Phase 3 Suitability**: This approach demonstrates proficiency in API integration, secure credential management, and web app development, aligning with a Phase 3 project’s expectations.

---

### Step 6: Potential Issues and Solutions
1. **Missing Secrets**:
   - **Error**: `KeyError: 'kaggle' not found in secrets`.
   - **Solution**: Ensure `secrets.toml` exists locally or secrets are configured in Streamlit Cloud. Verify the `[kaggle]` section includes `username` and `key`.

2. **Kaggle API Authentication Failure**:
   - **Error**: `403 Forbidden` or authentication errors.
   - **Solution**: Regenerate the Kaggle API token and update `secrets.toml`. Ensure your Kaggle account has accepted the dataset’s rules (visit the Kaggle dataset page and click "Download").

3. **Dataset Download Issues**:
   - **Error**: Files not found or corrupted.
   - **Solution**: Check internet connectivity. Manually inspect the `./data/` folder to ensure `train.txt`, `test.txt`, and `val.txt` are present. If issues persist, download manually and place files in `./data/`.

4. **Streamlit Cloud Deployment**:
   - **Issue**: App fails to download dataset.
   - **Solution**: Verify secrets are correctly set in Streamlit Cloud. Ensure the `kaggle` library is included in your `requirements.txt`:
     ```txt
     pandas
     nltk
     scikit-learn
     streamlit
     matplotlib
     seaborn
     kaggle
     ```

---

### Step 7: Additional Notes
- **Alternative Dataset**: If you prefer a tweet-specific dataset, you can use the **Sentiment140 dataset** (https://www.kaggle.com/datasets/kazanova/sentiment140). Modify the `load_dataset` function to handle CSV format:
  ```python
  def load_dataset(file_path):
      return pd.read_csv(file_path, encoding='latin-1')
  ```
  Update the dataset identifier in `download_kaggle_dataset` to `kazanova/sentiment140`.

- **Enhancements**:
  - Add a word cloud visualization for frequent words per emotion.
  - Allow users to upload a CSV file with social media posts for batch analysis.
  - Integrate a pre-trained transformer model (e.g., `distilbert-base-uncased-finetuned-sst-2-english`) for improved accuracy.

- **Testing the App**:
  - Test with inputs like:
    - "I’m so excited about my new job!" (Expected: joy, positive)
    - "Feeling really down today." (Expected: sadness, negative)
  - Verify the emotion distribution plot matches the dataset’s label distribution.

- **Project Submission**:
  - Include a `README.md` explaining how to set up secrets and run the app.
  - Document the model’s performance (e.g., add a classification report in the code).
  - Highlight the use of Streamlit secrets as a secure and modern approach.

If you need help with specific modifications (e.g., adding features, switching datasets, or troubleshooting deployment), please provide details, and I’ll assist further!
