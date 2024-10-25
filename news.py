import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from textblob import TextBlob  # For sentiment analysis
import csv
import os

# Load data
news_df = pd.read_csv("C:\\Users\\HP\\Downloads\\news.csv\\news.csv")
news_df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
news_df['content'] = news_df['title'] + ' ' + news_df['text']
news_df['label'] = news_df['label'].map({'REAL': 0, 'FAKE': 1})

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocess function
def preprocess_content(content):
    content = re.sub(r'[^a-zA-Z\s]', '', content)
    content = content.lower()
    content = ' '.join([word for word in content.split() if word not in stop_words])
    return content

news_df['content'] = news_df['content'].apply(preprocess_content)
news_df['id'] = range(1, len(news_df) + 1)

# Features and Labels
X = news_df['content'].values
y = news_df['label'].values

# Vectorize
vector = TfidfVectorizer()
X = vector.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix for additional stats
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Model performance statistics
precision = tp / (tp + fp) if (tp + fp) != 0 else 0
recall = tp / (tp + fn) if (tp + fn) != 0 else 0

# Function to save user feedback
def save_feedback(input_text, predicted_label, correct_label):
    feedback_file = 'feedback.csv'
    feedback_exists = os.path.exists(feedback_file)
    with open(feedback_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not feedback_exists:
            writer.writerow(['Article', 'Predicted Label', 'User Feedback'])
        writer.writerow([input_text, predicted_label, correct_label])

# Streamlit UI
st.markdown(
    """
    <style>
    .reportview-container {
        background: url(https://png.pngtree.com/thumb_back/fh260/background/20230415/pngtree-website-technology-line-dark-background-image_2344719.jpg);
        background-size: cover;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

st.title("Smart Fake News Detector with Sentiment Analysis and Feedback")
input_text = st.text_input('🔍 Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])  # Transform input text
    prediction = model.predict(input_data)       # Use transformed data
    return prediction[0]

# Sentiment Analysis function
def get_sentiment(input_text):
    blob = TextBlob(input_text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

if input_text:
    # Fake news prediction
    pred = prediction(input_text)
    if pred == 1:
        st.write("The news is Fake")
    else:
        st.write("The news is Real")

    # Sentiment Analysis
    sentiment = get_sentiment(input_text)
    st.write(f"Sentiment of the article: *{sentiment}*")

    # Display a sentiment score meter
    sentiment_score = TextBlob(input_text).sentiment.polarity
    st.write("Sentiment Score Meter:")
    st.progress((sentiment_score + 1) / 2)  # Normalize score to range [0,1]

    # Display performance metrics
    st.write(f"Model Accuracy: *{accuracy:.2f}*")
    st.write(f"Model Precision: *{precision:.2f}*")
    st.write(f"Model Recall: *{recall:.2f}*")

    # Example plot for model performance
    plt.figure(figsize=(10, 5))
    plt.bar(["Accuracy", "Precision", "Recall"], [accuracy * 100, precision * 100, recall * 100], color='Teal')
    plt.title("Model Performance")
    st.pyplot(plt)

    # Feedback options
    st.write("Was the prediction correct?")
    if st.button("Yes, it was correct"):
        save_feedback(input_text, pred, "Correct")
        st.write("Thank you for your feedback!")
    elif st.button("No, it was incorrect"):
        save_feedback(input_text, pred, "Incorrect")
        st.write("Thank you for your feedback! We will use this to improve the model.")

    # Reporting suspicious article
    if st.button("Report as Suspicious"):
        save_feedback(input_text, pred, "Reported as Suspicious")
        st.write("This article has been reported. Thank you!")