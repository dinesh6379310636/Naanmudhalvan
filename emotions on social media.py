import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from transformers import pipeline
import tweepy
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Streamlit app title
st.title("XPhone Launch Sentiment & Emotion Analysis Dashboard")

# Step 1: Access API key from Streamlit Secrets
try:
    api_key = st.secrets["API_KEY"]
except KeyError:
    st.error("API key not found in Streamlit Secrets. Please add 'API_KEY' in the app's secrets settings.")
    api_key = None

# Step 2: Fetch X posts via API or use fallback synthetic dataset
@st.cache_data
def fetch_x_posts(bearer_token, query="#XPhoneLaunch", max_results=15):
    if not bearer_token:
        st.warning("Using synthetic dataset due to missing API key.")
        return pd.DataFrame({
            'post_id': range(1, 16),
            'text': [
                "The new XPhone is incredible! Best tech ever! 😍 #XPhoneLaunch",
                "Why does the XPhone keep crashing? So annoyed 😡 #TechIssues",
                "Just got my XPhone, loving the camera quality! #Photography",
                "Terrible customer service for XPhone issues. #Disappointed",
                "XPhone battery life is amazing, super impressed! #Innovation",
                "Feeling sad, my XPhone arrived broken. 😢 #QualityControl",
                "Wow, XPhone’s design is sleek and modern! #LoveIt",
                "Can’t believe the XPhone price hike, not worth it. #Ripoff",
                "XPhone’s new features are a game-changer! #TechLover",
                "Struggling with XPhone setup, so frustrating! #UserExperience",
                "Shared my XPhone pics on X, got so many likes! #SocialMedia",
                "XPhone ads are everywhere, getting tired of them. #Overkill",
                "The XPhone launch event was epic, so excited! #Event",
                "XPhone signal issues are ruining my day. #Connectivity",
                "Feeling joyful with my new XPhone, it’s perfect! #Happy"
            ],
            'user_id': [f'user{i}' for i in range(1, 16)]
        })

    try:
        client = tweepy.Client(bearer_token=bearer_token)
        tweets = client.search_recent_tweets(query=query, max_results=max_results, 
                                            tweet_fields=['id', 'text', 'author_id'])
        if not tweets.data:
            st.warning("No posts found for query. Using synthetic dataset.")
            return fetch_x_posts(None)  # Fallback to synthetic

        posts_data = {
            'post_id': [tweet.id for tweet in tweets.data],
            'text': [tweet.text for tweet in tweets.data],
            'user_id': [str(tweet.author_id) for tweet in tweets.data]
        }
        return pd.DataFrame(posts_data)
    except Exception as e:
        st.error(f"Error fetching X posts: {str(e)}")
        return fetch_x_posts(None)  # Fallback to synthetic

# Fetch posts
df = fetch_x_posts(api_key)

# Display sample posts
st.subheader("Sample X Posts")
st.dataframe(df[['post_id', 'text']].head(5))

# Step 3: Initialize models
@st.cache_resource
def load_models():
    sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    emotion_analyzer = pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)
    return sentiment_analyzer, emotion_analyzer

sentiment_analyzer, emotion_analyzer = load_models()

# Step 4: Analyze sentiments and emotions
sentiments = []
emotions = []

for text in df['text']:
    # Sentiment analysis
    sentiment_result = sentiment_analyzer(text)[0]
    sentiment_label = sentiment_result['label'].capitalize()
    sentiment_score = sentiment_result['score']
    sentiments.append({'label': sentiment_label, 'score': sentiment_score})
    
    # Emotion analysis
    emotion_result = emotion_analyzer(text)[0]
    dominant_emotion = max(emotion_result, key=lambda x: x['score'])['label']
    emotions.append(dominant_emotion)

# Add results to DataFrame
df['sentiment'] = [s['label'] for s in sentiments]
df['sentiment_score'] = [s['score'] for s in sentiments]
df['emotion'] = emotions

# Step 5: Display detailed results
st.subheader("Detailed Analysis")
st.dataframe(df[['post_id', 'text', 'sentiment', 'sentiment_score', 'emotion']])

# Step 6: Dashboard-like summary
st.subheader("Dashboard: XPhone Launch Sentiment & Emotion Analysis")

# Sentiment breakdown
st.write("**Sentiment Breakdown**")
sentiment_counts = df['sentiment'].value_counts()
for sentiment, count in sentiment_counts.items():
    st.write(f"{sentiment}: {count} posts ({count/len(df)*100:.1f}%)")

# Emotion breakdown
st.write("**Emotion Breakdown**")
emotion_counts = df['emotion'].value_counts()
for emotion, count in emotion_counts.items():
    st.write(f"{emotion}: {count} posts ({count/len(df)*100:.1f}%)")

# Critical posts
critical_posts = df[(df['sentiment'] == 'Negative') & (df['sentiment_score'] > 0.9)]
st.write(f"**Critical Posts to Address ({len(critical_posts)})**")
for _, row in critical_posts.iterrows():
    st.write(f"Post {row['post_id']}: {row['text']} (Emotion: {row['emotion']})")

# Step 7: Visualize results
# Sentiment distribution (Plotly)
st.subheader("Visualizations")
fig_sentiment = px.histogram(df, x='sentiment', title='Sentiment Distribution of XPhone Posts', 
                             color='sentiment', color_discrete_sequence=px.colors.qualitative.Set2)
fig_sentiment.update_layout(xaxis_title='Sentiment', yaxis_title='Count')
st.plotly_chart(fig_sentiment)

# Emotion distribution (Plotly)
fig_emotion = px.histogram(df, x='emotion', title='Emotion Distribution of XPhone Posts', 
                           color='emotion', color_discrete_sequence=px.colors.qualitative.Pastel)
fig_emotion.update_layout(xaxis_title='Emotion', yaxis_title='Count')
st.plotly_chart(fig_emotion)

# Step 8: Sample interpretation
st.subheader("Sample Interpretation for XPhone Team")
st.write("**Insights**")
st.write("- Positive sentiment dominates, driven by joy for features like camera, battery, and design.")
st.write("- Negative sentiment linked to anger and sadness due to crashes, setup issues, and broken devices.")
st.write("**Recommendations**")
st.write("- Address critical posts: Prioritize customer support for setup and hardware issues.")
st.write("- Amplify positive feedback: Share posts about camera and design on X to boost engagement.")
st.write("- Monitor connectivity complaints: Investigate signal issues for future updates.")

# Step 9: Provide CSV download
st.subheader("Download Results")
csv = df.to_csv(index=False)
st.download_button(
    label="Download analysis as CSV",
    data=csv,
    file_name="xphone_sentiment_emotion.csv",
    mime="text/csv"
)
