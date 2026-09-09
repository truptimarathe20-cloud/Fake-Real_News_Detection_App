import streamlit as st
import pickle
import xgboost as xgb

# XGBoost model aur Vectorizer load karna
model = xgb.XGBClassifier()
model.load_model('xgb_model.json')

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Real-Time Fake News Detection System")

news_text = st.text_area("Enter your headlines or text:")

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter a text.")
    else:
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)
        
        if prediction[0] == 1 or prediction == 'Fake':
            st.error("This is a fake news.")
        else:
            st.success("This is a real news.")