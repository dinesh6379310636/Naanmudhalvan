import streamlit as st
import pandas as pd
import requests
from io import StringIO
from textblob import TextBlob

# Function to fetch dataset from GitHub
def load_data():
    url = "https://raw.githubusercontent.com/naanmudhalvan/data/main/emotions_data.txt"
    response = requests.get(url)
    data = StringIO(response.text)
    df = pd.read_csv(data, sep=';', names=['text', 'emotion'])
    return df

# Function to analyze sentiment and map to emotions
def analyze_emotion(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    # Mapping polarity to emotions
    if polarity < -0.5:
        return "sadness"
    elif polarity < 0:
        return "anger"
    elif polarity == 0:
        return "fear"
    elif polarity <= 0.5:
        return "love"
    else:
        return "joy"

# Streamlit app
st.title("Emotion Detection and Analysis from Social Media Conversations")

# Load dataset
df = load_data()

# Display dataset preview
st.subheader("Dataset Preview")
st.write(df.head())

# Input for custom text analysis
st.subheader("Analyze Your Own Text")
user_input = st.text_area("Enter a social media post to analyze its emotion:", "I feel happy today!")
if st.button("Analyze"):
    detected_emotion = analyze_emotion(user_input)
    st.write(f"Detected Emotion: **{detected_emotion}**")

# Analyze emotions in the dataset
st.subheader("Emotion Analysis of Dataset")
emotion_counts = {"sadness": 0, "joy": 0, "fear": 0, "anger": 0, "love": 0}
for text in df['text']:
    emotion = analyze_emotion(text)
    emotion_counts[emotion] += 1

# Display emotion counts
st.write("Emotion Distribution in Dataset:")
for emotion, count in emotion_counts.items():
    st.write(f"{emotion.capitalize()}: {count} instances")

# Display sample texts for each emotion
st.subheader("Sample Texts for Each Emotion")
for emotion in emotion_counts.keys():
    st.write(f"**{emotion.capitalize()}**")
    sample_texts = df[df['text'].apply(analyze_emotion) == emotion]['text'].head(3).tolist()
    for i, text in enumerate(sample_texts, 1):
        st.write(f"{i}. {text}")
