import pickle
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Fake & Real News Detection System", page_icon="📰")
st.header("Real-Time Fake News Detection System")

model = xgb.XGBClassifier()
model.load_model("xgb_model.json")

with open("vectorizer.pkl", "rb") as f:
  vectorizer = pickle.load(f)

news_text = st.text_area("Enter your headlines here:")

if st.button("Predict"):
  if news_text.strip() == "":
    st.warning("Please enter text to analyze.")
  else:
    transformed_text = vectorizer.transform([news_text])
    prediction = model.predict(transformed_text)

    if prediction[0] == 1:
      st.error("This News is FAKE!")
    else:
      st.success("This News is REAL!")