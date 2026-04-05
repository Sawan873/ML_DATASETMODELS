# Unified AI/ML Hub

Welcome to the AI/ML Hub - a unified, modern, highly-aesthetic Streamlit dashboard containing implementations of multiple specialized AI/ML models across different modalities including Audio, Numeric, Text, Video, and Image!

## Features 🚀
- **Audio Module:** Predicts heart anomalies/sounds powered by Librosa & SVM.
- **Numeric Module:** Academic outcome predictor utilizing multiple demographic inputs.
- **Text Module:** E-Commerce category classifier utilizing custom NLP.
- **Video Module:** High-speed Shoplifting detection using TensorFlow.
- **Image Module:** Seamless template and scaffold to deploy Deep Learning object detections.
- **Rich Aesthetics:** Designed using custom CSS features inclusive of glassmorphism and animated components to deliver a premium user-interface.

---

## 💻 Local Execution Guide

To run this application locally on your Windows machine, simply follow these steps.

### 1) Pre-requisites
Ensure that you have Python 3.9+ installed and added to your PATH.

### 2) Install Dependencies
Open your command prompt or PowerShell inside this `ML_Models` directory and run:

```bash
pip install -r requirements.txt
```

*(Note: If you run into issues installing tensorflow on windows, you can simply run `pip install tensorflow --user`)*

### 3) Run the Application
Start the Streamlit server natively by executing:

```bash
streamlit run app.py
```

The browser will automatically open and navigate to `http://localhost:8501`.

---

## 🌐 Deployment Guide (Cloud)

Streamlit natively offers free and instant deployment using **Streamlit Community Cloud**.

1. **Commit to GitHub**
   Initialize a git repository, commit all the code, models (`.pkl`/`.h5`), and `requirements.txt` to a GitHub repository.
   ```bash
   git init
   git add .
   git commit -m "Initial commit for ML Hub"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

2. **Deploy**
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Log in using your GitHub account.
   - Click **"New App"** and select the repository you just pushed.
   - Ensure the Main file path is set to `app.py`.
   - Click **"Deploy!"** 

Your app will be built automatically using `requirements.txt`. Once successful, you'll receive a public URL to share! If any specific heavy modules fail due to community memory limits (typically 1GB RAM on Streamlit Cloud), you can choose to deploy smoothly using [Hugging Face Spaces](https://huggingface.co/spaces) or [Render](https://render.com/).
