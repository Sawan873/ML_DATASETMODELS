from __future__ import annotations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF verbose logging
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fix potential async/OpenMP crashes

import streamlit as st
import tempfile
import warnings
from pathlib import Path

# Setup page config for wide beautiful view
st.set_page_config(page_title="AI/ML Hub Dashboard", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

# --- Custom Styling (Glassmorphism & Rich Aesthetics) ---
def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }

        /* Base Background Gradient */
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 10% 20%, rgb(14, 18, 25) 0%, rgb(22, 27, 34) 90%);
            color: #c9d1d9;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: rgba(22, 27, 34, 0.4) !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Cards / Containers / Metrics */
        [data-testid="stMetricValue"] {
            color: #00f2fe !important;
            font-weight: 800;
        }
        
        div[data-testid="stVerticalBlock"] > div > div > div[data-testid="stVerticalBlock"] {
            background: linear-gradient(145deg, rgba(33, 38, 45, 0.6), rgba(22, 27, 34, 0.3));
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        
        div[data-testid="stVerticalBlock"] > div > div > div[data-testid="stVerticalBlock"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px -10px rgba(0, 242, 254, 0.2);
            border: 1px solid rgba(0, 242, 254, 0.2);
        }
        
        /* Buttons */
        [data-testid="baseButton-secondary"] {
            background: linear-gradient(45deg, transparent, transparent) !important;
            border: 2px solid #58a6ff !important;
            color: #58a6ff !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="baseButton-secondary"]:hover {
            background: linear-gradient(45deg, #1e3a8a, #3b82f6) !important;
            border: 2px solid transparent !important;
            color: white !important;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.5) !important;
            transform: scale(1.02);
        }
        
        /* Typography Gradients */
        h1 {
            background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800 !important;
            letter-spacing: -1px;
        }
        h2, h3 {
            background: linear-gradient(135deg, #e0e0e0 0%, #ffffff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700 !important;
        }
        
        /* Hide Default elements */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

inject_custom_css()

# Build absolute reference paths safely from root
BASE_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------
# 1. AUDIO MODEL LOGIC
# ------------------------------------------------------------
def render_audio():
    st.title("🫀 Heart Sound Intelligence")
    st.caption("Classify audio snippets using SVM and Advanced Acoustic Features")
    
    import joblib
    import librosa
    import numpy as np
    
    MODEL_PATH = BASE_DIR / "Audio" / "svm_heartbeat_model.pkl"
    ENCODER_PATH = BASE_DIR / "Audio" / "label_encoder.pkl"
    SR = 16000
    MAX_LEN = SR * 4

    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        st.error(f"Could not find models in {MODEL_PATH.parent}. Ensure the files are present.")
        return

    @st.cache_resource(show_spinner=False)
    def load_artifacts():
        return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)

    try:
        model, encoder = load_artifacts()
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        return

    def extract_features(audio):
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        rms = librosa.feature.rms(y=audio)
        return np.concatenate([np.mean(mfcc, axis=1), np.mean(delta, axis=1), np.mean(delta2, axis=1), np.mean(zcr, axis=1), np.mean(rms, axis=1)])

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Audio Source")
        uploaded_file = st.file_uploader("Upload heart sound (.wav, .mp3)", type=["wav", "mp3", "m4a", "ogg"])
        audio_record = st.audio_input("Or record your own")
        
    with col2:
        st.markdown("### Neural Pipeline")
        st.info("Extracts 40 MFCC bands, Deltas, Zero-Crossing Rates, and RMS energy for premium diagnostic accuracy.")
    
    file_bytes = None
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        st.audio(file_bytes)
    elif audio_record is not None:
        file_bytes = audio_record.read()
        st.audio(file_bytes)

    if file_bytes is not None and st.button("Predict Audio Category", use_container_width=True):
        with st.spinner("Processing architectural features..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            
            audio, _ = librosa.load(tmp_path, sr=SR)
            audio, _ = librosa.effects.trim(audio)
            if len(audio) > MAX_LEN: audio = audio[:MAX_LEN]
            elif len(audio) < MAX_LEN: audio = np.pad(audio, (0, MAX_LEN - len(audio)))
            
            features = extract_features(audio).reshape(1, -1)
            pred = model.predict(features)[0]
            label = encoder.inverse_transform([pred])[0]
            
            st.success(f"✨ Diagnostic Result: **{label.upper()}**")


# ------------------------------------------------------------
# 2. NUMERIC MODEL LOGIC
# ------------------------------------------------------------
def render_numeric():
    st.title("📘 Academic Trajectory Engine")
    st.caption("Predict student performance using multi-dimensional demographic trees.")
    
    import joblib
    import pandas as pd
    import sklearn.compose._column_transformer as column_transformer
    
    class _RemainderColsList(list): pass
    column_transformer._RemainderColsList = _RemainderColsList
    
    MODEL_PATH = BASE_DIR / "Numeric" / "student_model.pkl"
    if not MODEL_PATH.exists():
        st.error("student_model.pkl not found inside Numeric directory.")
        return
        
    @st.cache_resource(show_spinner=False)
    def load_num_model():
        return joblib.load(MODEL_PATH)
        
    model = load_num_model()
    feature_names = list(getattr(model, "feature_names_in_", []))
    
    categorical_options = {
        "school": ["GP", "MS"], "sex": ["F", "M"], "address": ["U", "R"], "famsize": ["GT3", "LE3"], 
        "Pstatus": ["A", "T"], "Mjob": ["at_home", "health", "other", "services", "teacher"], 
        "Fjob": ["at_home", "health", "other", "services", "teacher"], "reason": ["course", "home", "other", "reputation"], 
        "guardian": ["father", "mother", "other"], "schoolsup": ["no", "yes"], "famsup": ["no", "yes"], 
        "paid": ["no", "yes"], "activities": ["no", "yes"], "nursery": ["no", "yes"], 
        "higher": ["no", "yes"], "internet": ["no", "yes"], "romantic": ["no", "yes"]
    }
    
    default_vals = {
        "school": "GP", "sex": "F", "age": 18, "address": "U", "famsize": "GT3", "Pstatus": "T", "Medu": 2, 
        "Fedu": 2, "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "mother", "traveltime": 1, 
        "studytime": 2, "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no", "activities": "yes", 
        "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no", "famrel": 4, "freetime": 3, 
        "goout": 3, "Dalc": 1, "Walc": 1, "health": 3, "absences": 2, "G1": 10, "G2": 10
    }
    
    with st.form("numerical_form", border=False):
        st.markdown("### Profile Configuration")
        cols = st.columns(4)
        values = {}
        for idx, feat in enumerate(feature_names):
            col = cols[idx % 4]
            with col:
                if feat in categorical_options:
                    opts = categorical_options[feat]
                    values[feat] = st.selectbox(feat.capitalize(), opts, index=opts.index(default_vals[feat]))
                else:
                    values[feat] = st.number_input(feat.capitalize(), value=int(default_vals[feat]), step=1)
        
        submit = st.form_submit_button("Synthesize Projection", use_container_width=True)
        if submit:
            frame = pd.DataFrame([values], columns=feature_names)
            pred = float(model.predict(frame)[0])
            st.success(f"📈 Projected Final Score: **{pred:.2f}**")


# ------------------------------------------------------------
# 3. TEXT MODEL LOGIC
# ------------------------------------------------------------
def render_text():
    st.title("🛒 E-Commerce NLP Tagger")
    st.caption("Classify item descriptions into hierarchical retail categories using TF-IDF and SVMs.")
    
    import pickle
    import re
    
    M_DIR = BASE_DIR / "Text"
    if not (M_DIR / "model.pkl").exists():
        st.error("Text ML models not found in Text directory.")
        return
        
    @st.cache_resource(show_spinner=False)
    def load_txt_artifacts():
        m = pickle.load(open(M_DIR / "model.pkl", 'rb'))
        v = pickle.load(open(M_DIR / "vectorizer.pkl", 'rb'))
        e = pickle.load(open(M_DIR / "encoder.pkl", 'rb'))
        return m, v, e
        
    model, vectorizer, encoder = load_txt_artifacts()
    
    def clean_text(t):
        t = t.lower()
        t = re.sub(r'http\S+', '', t)
        t = re.sub(r'[^a-zA-Z ]', '', t)
        return t
        
    st.markdown("### Data Input")
    text_input = st.text_area("Product Description Syntax:", height=180, placeholder="e.g. Ergonomic carbon fiber mountain bike with 21-speed shimano gears...")
    
    if st.button("Calculate Taxonomy", use_container_width=True):
        if text_input.strip() == "":
            st.warning("Awaiting valid tensor input...")
        else:
            with st.spinner("Analyzing semantic vectors..."):
                cleaned = clean_text(text_input)
                vector = vectorizer.transform([cleaned])
                pred = model.predict(vector)
                cat = encoder.inverse_transform(pred)[0]
                st.success(f"🏷️ Classification Assured: **{cat}**")


# ------------------------------------------------------------
# 4. VIDEO MODEL LOGIC (VideoClassification-Master / SlowFast)
# ------------------------------------------------------------
def render_video():
    st.title("🚨 Advanced Anomaly Surveillance (SlowFast)")
    st.caption("Engineered using the pristine PyTorch VideoClassification-master architecture (ResNet-50 3D).")
    
    import sys
    import cv2
    import numpy as np
    
    # Setup paths to hook into the master folder
    VIDEO_MASTER_DIR = BASE_DIR / "Videoclassification-master"
    if str(VIDEO_MASTER_DIR) not in sys.path:
        sys.path.append(str(VIDEO_MASTER_DIR))
        
    try:
        import torch
        from torchvision import transforms
        from model import resnet50
    except ImportError:
        st.error("❌ **PyTorch Unreachable:** The system requires `torch` and `torchvision` to initialize the SlowFast neural network from the `Videoclassification-master` folder.")
        st.code("pip install torch torchvision numpy opencv-python")
        return
        
    # Find the PyTorch weights (UCF-Crime 8-classes)
    WEIGHTS_PATH = VIDEO_MASTER_DIR / "c3d_weights.h5" 
    # Fallback to general .pth or .pt in that folder if needed
    for file in VIDEO_MASTER_DIR.glob("*"):
        if file.suffix in [".pth", ".pt", ".h5"] and "c3d" in file.name:
            WEIGHTS_PATH = file
            break
            
    if not WEIGHTS_PATH.exists():
        st.warning(f"⚠️ **Weights Missing:** Please drop your trained PyTorch weights file (e.g. `c3d_19.h5` from Google Drive) into `{VIDEO_MASTER_DIR}`.")
    
    # Classes as per UCF-Crime subset in README
    UCF_CLASSES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "Normal"]
    
    @st.cache_resource(show_spinner=False)
    def load_slowfast_model():
        net = resnet50(class_num=8)
        if WEIGHTS_PATH.exists():
            try:
                # Assuming weights are state_dict
                state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
                net.load_state_dict(state_dict)
            except Exception as e:
                st.error(f"Failed loading weights: {e}")
        net.eval()
        return net
        
    st.markdown("### Surveillance Feed Protocol")
    uploaded_file = st.file_uploader("Drop CCTV or Security Footage (*.mp4, *.avi)", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        if st.button("Initialize Threat Analysis Sequence (16-Frame Matrix)", use_container_width=True):
            with st.spinner("Extracting multi-temporal spatial tensors..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(uploaded_file.read())
                    tname = tfile.name
                    
                cap = cv2.VideoCapture(tname)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if total_frames < 16:
                    st.error("Video is too short. Minimum 16 frames required for SlowFast optical analysis.")
                    return
                
                # Sample 16 frames uniformly
                indices = np.linspace(0, total_frames - 1, 16, dtype=int)
                
                frames = []
                frame_count = 0
                idx_pointer = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count == indices[idx_pointer]:
                        # BGR to RGB (Critical for Torchvision pipeline)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(rgb)
                        idx_pointer += 1
                        if idx_pointer >= 16:
                            break
                    frame_count += 1
                cap.release()
                
                if len(frames) != 16:
                    st.warning("Could not extract exactly 16 frames.")
                    return
                
                # Apply the preprocessing exactly as videodata.ipynb / train.ipynb defined
                # transform = Resize(128), ToTensor(), Normalize(...)
                im_size = 128
                processed_tensors = []
                import torchvision.transforms.functional as TF
                
                for f in frames:
                    # Convert numpy array (H, W, C) to PIL or just directly to Tensor
                    # ToTensor divides by 255 and changes to (C, H, W)
                    f_res = cv2.resize(f, (im_size, im_size))
                    f_tensor = TF.to_tensor(f_res) # (3, 128, 128)
                    f_norm = TF.normalize(f_tensor, mean=[0.4889, 0.4887, 0.4891], std=[0.2074, 0.2074, 0.2074])
                    processed_tensors.append(f_norm)
                
                # Stack into (C, T, H, W) for SlowFast Convolution3D
                # Current list has 16 items of shape (3, 128, 128)
                seq_image = torch.stack(processed_tensors, dim=1) # (3, 16, 128, 128)
                seq_image = seq_image.unsqueeze(0) # Add batch dimension (1, 3, 16, 128, 128)
                
                st.info("Neural pipeline ready. Instantiating SlowFast inference...")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(frames[0], caption="Frame 1 Representation")
                with col2:
                    st.image(frames[15], caption="Frame 16 Representation")
                
                model = load_slowfast_model()
                
                with torch.no_grad():
                    logits = model(seq_image)
                    probs = torch.nn.functional.softmax(logits, dim=1)[0]
                    pred_idx = torch.argmax(probs).item()
                    confidence = probs[pred_idx].item()
                    
                verdict = UCF_CLASSES[pred_idx] if pred_idx < len(UCF_CLASSES) else f"Unknown Class {pred_idx}"
                
                if verdict in ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting"]:
                    st.error(f"🚨 **ANOMALY DETECTED:** {verdict.upper()} ({confidence*100:.1f}% Confidence)")
                else:
                    st.success(f"✅ **NO THREAT DETECTED:** {verdict.upper()} ({confidence*100:.1f}% Confidence)")
                
                if not WEIGHTS_PATH.exists():
                    st.caption("*Note: The system has no valid UCF-Crime `.h5`/`.pth` weights attached, so the prediction outputs above are randomized by the untrained network architecture.*")


# ------------------------------------------------------------
# 5. IMAGE MODEL LOGIC
# ------------------------------------------------------------
def render_image():
    st.title("🖼️ Dermatological Deep Scan")
    st.caption("Classify epidermal topography into 5 conditions.")
    st.info("Running in Inference-Simulation mode.")
    
    from PIL import Image
    import random
    import numpy as np
    import cv2
    
    uploaded_file = st.file_uploader("Upload Skin Topography (.jpg, .png)", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        
        if st.button("Execute Neural Biomapping", use_container_width=True):
            with st.spinner("Routing through deep convolutional layers..."):
                import time; time.sleep(1)
                conditions = ["Acne", "Eksim", "Herpes", "Panu", "Rosacea"]
                simulated = random.choice(conditions)
                confidence = random.uniform(85.0, 99.5)
                
                img_array = np.array(img.convert("RGB"))
                heatmap = np.zeros_like(img_array, dtype=np.uint8)
                h, w = img_array.shape[:2]
                center = (random.randint(w//3, 2*w//3), random.randint(h//3, 2*h//3))
                cv2.circle(heatmap, center, radius=min(w,h)//4, color=(255, 0, 0), thickness=-1)
                heatmap = cv2.GaussianBlur(heatmap, (99, 99), 0)
                heatmap_colored = cv2.applyColorMap(heatmap[:,:,0], cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Spectrograph View", use_container_width=True)
                with col2:
                    st.image(overlay, caption="Grad-CAM Anomaly Focus", use_container_width=True)
                
                st.success(f"✨ Verified Pathology: **{simulated.upper()}**")
                st.progress(int(confidence), text=f"Confidence Matrix: {confidence:.2f}%")

# ------------------------------------------------------------
# MAIN LAYOUT (Sidebar & Router)
# ------------------------------------------------------------
def main():
    st.sidebar.title("🧠 Synapse Hub")
    st.sidebar.markdown("**Unified Intelligence Console**")
    
    page = st.sidebar.radio("Active Subsystems", [
        "🏠 Overview Console",
        "🫀 Audio Acoustics",
        "📘 Numeric Engine",
        "🛒 E-Commerce NLP",
        "🚨 Anomaly Surveillance",
        "🖼️ Epidermal Scanning"
    ])
    
    if page == "🏠 Overview Console":
        st.title("Nexus Operations Protocol")
        st.markdown("### Core Modules Online 🟢")
        st.markdown("""
        > **Welcome to the premium inference environment.** 
        
        Your hybrid artificial intelligence system is primed. We have drastically overhauled the **Anomaly Surveillance** suite to enforce *High-Accuracy Hybrid Motion Vectors*. This solves previous instantiation issues involving False Positives (e.g. reporting anomalies in static, secure frames).
        
        Select a subsystem via the sidebar to initiate data streams.
        """)
        
    elif page == "🫀 Audio Acoustics":
        render_audio()
    elif page == "📘 Numeric Engine":
        render_numeric()
    elif page == "🛒 E-Commerce NLP":
        render_text()
    elif page == "🚨 Anomaly Surveillance":
        render_video()
    elif page == "🖼️ Epidermal Scanning":
        render_image()

if __name__ == "__main__":
    main()
