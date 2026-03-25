import streamlit as st 
import pickle 
import os

model = pickle.load(open("fake_news_detection/fake_news_detection_model.pkl","rb"))
vectorizer = pickle.load(open("fake_news_detection/vectorizer.pkl","rb"))

st.markdown("<h1 style = 'text-align: center; color: red;'>Fake News Detection App</h1>",unsafe_allow_html = True)
st.subheader("Classify news as **Real** or **Fake** by using NLP")
st.info("⚠️ This prediction is based on machine learning and may not be 100% accurate.")
st.markdown("Model Info")
st.write("""
- Model: Logistic Regression  
- Vectorizer: TF-IDF  
- Dataset: Kaggle Fake News Dataset  
- Accuracy: 84%  
""")

user_input  = st.text_area("Enter news article",)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text")

    else:
        vect = vectorizer.transform([user_input])
        pred = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0]

        if pred == 1:
            st.success("Real News")
        else :
            st.error("Fake News")
        
        st.subheader("Confidence")
        st.write(f"Fake: {proba[0]*100:.2f}%")
        st.write(f"Real: {proba[1]*100:.2f}%")
        
