import streamlit as st
import pickle
import re

# Load files
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

# Text cleaning function (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# UI
st.title("Ecommerce Product Category Predictor")

user_input = st.text_area("Enter product description:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    category = encoder.inverse_transform(prediction)

    st.success(f"Predicted Category: {category[0]}")