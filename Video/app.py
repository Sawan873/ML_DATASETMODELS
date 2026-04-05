import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile

# load model (make sure .h5 is in same folder)
model = load_model('shoplifting_model.h5')

IMG_SIZE = 128

st.title("Shoplifting Detection App")

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    frame_skip = 10   # reduced skip (more frames)
    count = 0
    predictions = []
    
    st.write("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_skip == 0:
            img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            pred = model.predict(img)[0][0]
            predictions.append(pred)
        
        count += 1
    
    cap.release()

    # ---- DEBUG INFO ----
    st.write("Sample predictions:", predictions[:10])

    # ---- LOGIC FIX ----
    high_preds = [p for p in predictions if p > 0.1]

    st.write("High prediction count:", len(high_preds))

    # ---- FINAL DECISION ----
    if len(high_preds) > 5:
        st.error("SHOPLIFTING DETECTED 🚨")
    else:
        st.success("NORMAL ACTIVITY ✅")