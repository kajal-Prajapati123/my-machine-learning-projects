# Fake News Detection Project
## Objective

The objective of this project is to build a machine learning model that can classify news articles as Real or Fake based on their headlines.
This helps in identifying misinformation using Natural Language Processing (NLP) techniques.

## Dataset
The dataset **FakeNewsNet** contains news data with the following columns:
title – news headline
url – article link
source_domain – source website
Real – target variable
0 = Fake News
1 = Real News
Since full article text was not available, the model is trained using news titles and source domain(headline-based classification).
## Model Used
Text preprocessing using NLTK
Feature extraction using TF-IDF Vectorizer
Model trained using Logistic Regression

Additionally, Naive Bayes was also tested, but Logistic Regression provided better performance and stability.

## Accuracy
Achieved approximately 80–85% accuracy on test data

## How to Run
- Install dependencies
pip install -r requirements.txt
- Run Streamlit app
streamlit run fake_news_detection_app.py
- Use the app
Enter a news headline in the input box
Click on Check
The model will classify it as Real or Fake with confidence score
## Features
Real-time prediction using Streamlit
Displays prediction confidence
Uses NLP pipeline (cleaning + TF-IDF + ML model)
## Conclusion
This project demonstrates how machine learning can be used to detect fake news.
Further improvements can be achieved by using title and source domain.

## Project Link
https://my-machine-learning-projects-kkkaxacubftochy9zauklv.streamlit.app/
