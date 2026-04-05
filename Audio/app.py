from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm_heartbeat_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
SR = 16000
MAX_LEN = SR * 4

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


@st.cache_resource
def load_artifacts() -> tuple[Any, Any]:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR)
    audio, _ = librosa.effects.trim(audio)

    if len(audio) > MAX_LEN:
        return audio[:MAX_LEN]
    if len(audio) < MAX_LEN:
        return np.pad(audio, (0, MAX_LEN - len(audio)))
    return audio


def extract_features(audio: np.ndarray) -> np.ndarray:
    # 122 features: 40 MFCC means + 40 delta means + 40 delta2 means + ZCR + RMS.
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    rms = librosa.feature.rms(y=audio)

    features = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.mean(delta2, axis=1),
            np.mean(zcr, axis=1),
            np.mean(rms, axis=1),
        ]
    )
    return features


def predict_audio(file_bytes: bytes, suffix: str) -> tuple[str, np.ndarray]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    audio = load_audio(temp_path)
    features = extract_features(audio).reshape(1, -1)
    model, encoder = load_artifacts()
    encoded_prediction = model.predict(features)[0]
    label = encoder.inverse_transform([encoded_prediction])[0]
    return label, features[0]


st.set_page_config(page_title="Heart Sound Classifier", page_icon="🫀", layout="wide")
st.title("Heart Sound Classification")
st.caption("Local Streamlit UI for `svm_heartbeat_model.pkl` with no API layer.")

try:
    model, encoder = load_artifacts()
except Exception as exc:
    st.error(f"Failed to load model artifacts: {exc}")
    st.stop()

st.write(
    "Upload an audio clip or record from the microphone to classify the heart sound."
)

left, right = st.columns([2, 1])
with right:
    st.subheader("Model Info")
    st.write(f"Expected features: `{getattr(model, 'n_features_in_', 'unknown')}`")
    st.write(f"Classes: `{', '.join(map(str, encoder.classes_))}`")
    st.write("Audio is trimmed, resampled to 16 kHz, and padded/truncated to 4 seconds.")

with left:
    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "ogg"])
    audio_record = st.audio_input("Or record from microphone")

source = None
file_bytes = None
suffix = ".wav"

if uploaded_file is not None:
    source = "uploaded file"
    file_bytes = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix or ".wav"
    st.audio(file_bytes)
elif audio_record is not None:
    source = "microphone"
    file_bytes = audio_record.read()
    suffix = ".wav"
    st.audio(file_bytes)

if file_bytes is not None:
    with st.spinner(f"Running prediction from {source}..."):
        try:
            label, features = predict_audio(file_bytes, suffix)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            st.success(f"Predicted class: {label}")
            st.metric("Feature vector length", len(features))
            with st.expander("Preview extracted features"):
                st.write(features.tolist())
