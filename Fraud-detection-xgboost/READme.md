# Credit Card Fraud Detection Project
## Objective

This project focuses on detecting fraudulent credit card transactions using machine learning techniques.
Due to the highly imbalanced nature of the dataset, special care is taken during preprocessing and model selection.

## Dataset Information
Source: Kaggle Creditcard.csv
Total Records: 284,807 transactions
Features:
V1 to V28: PCA-transformed numerical features (anonymized for confidentiality)
Amount: Transaction amount
Class:
0 → Non-Fraud
1 → Fraud
Dataset is highly imbalanced (fraud cases ≈ 0.17%)

## Data Preprocessing
Checked for missing values (none found)
Scaled Amount and Time using StandardScaler
Handled class imbalance using:
class_weight='balanced'

## Model Used
XGbClassifier

## Model Evaluation
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Special focus was given to Recall, as detecting fraud cases is more important than overall accuracy.

## Visualizations
Confusion matrix heatmap
ROC curve

## Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

The model successfully identifies fraudulent transactions despite extreme class imbalance.
This project demonstrates real-world fraud detection challenges and practical ML solutions.

## project link
https://my-machine-learning-projects-tefwxcw7d2o9mrtbvrpbws.streamlit.app/
