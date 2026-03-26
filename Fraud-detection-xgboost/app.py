import streamlit as st
import pickle 
import pandas as pd
import matplotlib.pyplot as plt


st.sidebar.header("Model Info")
st.sidebar.write("Model : XGBoost")
st.sidebar.write("Dataset: Kaggle creditcard.csv")
st.sidebar.write("Accuracy : 0.99")
st.header("Credit Card Fraud Detection System", "center")
st.markdown("An ML-powered fraud detection dashboard using XGBoost.")
col1,col2 =  st.columns(2)


with col1:
    st.subheader("Confusion Matrix")
    st.image("Fraud-detection-xgboost/Credit Card Fraud Model Confusion Matrix.png")
with col2:
    st.subheader("ROC Curve")
    st.image("Fraud-detection-xgboost/Credit Card Fraud Model ROC Curve.png")

model = pickle.load(open("Fraud-detection-xgboost/fraud_model.pkl","rb"))
scl = pickle.load(open("Fraud-detection-xgboost/scaler.pkl","rb"))

features = {}
with st.expander("Enter Transaction Features (V1–V28)"):
    for i in range(1,29):
        features[f"V{i}"] = st.number_input(f"V{i}")

features["Amount"] = st.number_input("Amount")
input_data = pd.DataFrame([features])

col1, col2, col3 = st.columns(3)

col1.metric("Fraud Precision", "0.66")
col2.metric("Fraud Recall", "0.86")
col3.metric("Fraud F1-Score", "0.74")

threshold = st.slider("Select Fraud Detection Threshold",0.0,1.0,0.5)

if st.button("Predict"):
   amount = features["Amount"]

   if amount <= 0:
       st.error("Amount is required")
   else:
        input_data["Amount"] = scl.transform(input_data[["Amount"]])
        prob = model.predict_proba(input_data)[0][1]
        prediction  = 1 if prob > threshold else 0
        if prediction == 1:
            st.error(f"Fraud Detected | Fraud Probability: {prob:.4f}")
        else:
            st.success(f"Lagitimate Transection | Probability: {1 - prob:.4f}")

