---
title: Emotions On Social Media
emoji: 🌖
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Emotion Detection in Social Media Conversations This project is a Gradio app that decodes emotions in social media conversations through sentiment analysis. It uses a pre-trained Hugging Face model to analyze emotions in text data, deployed on Hugging Face Spaces as of May 15, 2025. Description The app allows users to:

View a sample dataset of social media posts. Analyze emotions in a single social media text (e.g., a tweet or comment). Upload a CSV or TXT file for batch emotion analysis.

The app uses the j-hartmann/emotion-english-distilroberta-base model, which detects seven emotions: anger, disgust, fear, joy, neutral, sadness, and surprise. Outputs are text-based, with no graphs or curves, as per the project requirements. Dataset The app uses the Social Media Sentiments Analysis Dataset by Kashish Parmar on Kaggle. This dataset contains social media posts with sentiment labels, stored in sentimentdataset.csv. It includes columns like:

Text: The social media post. Sentiment: The labeled sentiment (e.g., positive, negative, joy, anger).

The dataset is accessed via Kaggle’s API using environment variables. Setup Instructions To run this app in a Hugging Face Space:

Clone the Repository:

Create a new Hugging Face Space with the Gradio SDK and Python environment.

Set Environment Variables:

Go to the “Settings” tab in your Space. Under “Variables and Secrets,” add the following as private secrets: KAGGLE_USERNAME: Your Kaggle username (e.g., dinesh6379). KAGGLE_KEY: Your Kaggle API key (generate this from your Kaggle account settings).

Ensure there are no naming conflicts (e.g., don’t set the same variable as both public and private).

Install Dependencies:

The requirements.txt file includes:transformers pandas gradio torch regex requests

Hugging Face Spaces will automatically install these when the Space starts.

Upload Files:

Upload app.py and requirements.txt to your Space’s repository. Optionally, upload this README.md for documentation.

Restart the Space:

Restart the Space to apply changes and install dependencies.

Usage The Gradio app is organized into three card-based sections:

Explore the Social Media Dataset:

Click “Load Dataset Preview” to view the first 5 rows of the Kaggle dataset. Displays the Text and Sentiment columns.

Analyze a Single Post:

Enter a social media post (e.g., a tweet) in the text box. Click “Detect Emotions” to see the detected emotions with confidence scores (e.g., Joy: 97.72%, Sadness: 1.23%).

Analyze Multiple Posts:

Upload a CSV or TXT file in the “Analyze Multiple Posts” section. CSV files must have a text or Text column; TXT files should have one post per line. Click “Analyze File” to see the dominant emotion for each text entry.

Model Details

Model: j-hartmann/emotion-english-distilroberta-base Purpose: Emotion detection in English text, optimized for social media data. Emotions Detected: Anger, disgust, fear, joy, neutral, sadness, surprise. Performance: Achieves ~66% accuracy on a balanced dataset.

Troubleshooting

Dataset Download Fails: Ensure KAGGLE_USERNAME and KAGGLE_KEY are set correctly as private secrets. If you see a 403 Forbidden error, regenerate your Kaggle API key on Kaggle and update KAGGLE_KEY.

Configuration Error: Avoid setting the same name (e.g., KAGGLE_USERNAME) as both a public variable and a private secret to prevent naming collisions.

App Not Loading: Restart the Space to ensure dependencies are installed and environment variables are applied.

Additional Notes

Deployment: This app is deployed on Hugging Face Spaces. Public URL: [Add your Space URL here once live, e.g., https://huggingface.co/spaces/yourusername/yourspace]. Output Format: Results are text-based with no graphs or curves, as specified. Future Improvements: Add support for more datasets. Enhance text preprocessing for better accuracy. Allow users to select different emotion detection models.

Contributing Feel free to fork this project, submit issues, or contribute improvements via pull requests on the Hugging Face Space repository.
