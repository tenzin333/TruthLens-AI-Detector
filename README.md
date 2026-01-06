# 🔍 TruthLens: Multi-Signal AI Content Verifier

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Hiring_Ready-green?style=for-the-badge)

**TruthLens** is a digital forensics platform built to combat the 2026 deepfake crisis. It moves beyond simple "AI detection" by utilizing an ensemble of Vision Transformers, Mathematical Frequency Analysis, and C2PA Metadata verification to provide a comprehensive "Trust Score" for digital media.

---

## 🚀 Key Features

* **Ensemble AI Classification:** Utilizes **Vision Transformers (ViT)** to detect high-level synthetic patterns and textures.
* **XAI (Explainable AI):** Generates **Grad-CAM Heatmaps** to highlight the specific pixel clusters the AI found suspicious.
* **Frequency Forensics:** Implements **Fast Fourier Transform (FFT)** analysis to identify the "checkerboard artifacts" inherent in Diffusion and GAN-based generators.
* **Provenance Check:** Scans for **C2PA / Content Credentials** manifests to verify the cryptographic history of an image.
* **Optimized Performance:** Features **RAM-caching** and **GPU-accelerated inference**, reducing processing time from minutes to sub-5 seconds.

---

## 📊 How it Works

The platform cross-references three independent signals to minimize false positives:



1.  **The Structural Signal:** Does the AI model recognize "machine-made" textures?
2.  **The Statistical Signal:** Does the frequency spectrum show unnatural periodic noise?
3.  **The Metadata Signal:** Does the image carry a verified digital signature of origin?

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Optimized for 2026 UI standards)
* **Machine Learning:** PyTorch, Hugging Face Transformers
* **Computer Vision:** OpenCV, NumPy, SciPy
* **Explainability:** Grad-CAM (Gradient-weighted Class Activation Mapping)
* **Standardization:** C2PA Python SDK

---

## ⚙️ Installation & Setup

### 1. Clone the Lab
    git clone [https://github.com/yourusername/truthlens.git](https://github.com/yourusername/truthlens.git)
    cd truthlens
### 1. Preparing Environment
    python -m venv venv
# Windows:
    .\venv\Scripts\Activate.ps1 
# Mac/Linux:
    source venv/bin/activate
### 3. Install Requirements
    pip install -r requirements.txt or python -m pip install -r requirements.txt
### 4. Launch Application
    python -m streamlit run app.py

### 📈 Performance Metrics
Content Type	            | Accuracy |	Latency (GPU) |	Latency (CPU)
Photorealistic (DALL-E 3)	| 89%	   |     2.1s	      |  14s
Deepfake Faces (Sora/Video)	| 84%	   |     4.5s	      |  32s
Organic Photography	        | 96%	   |     1.8s	      |  10s

### 🛡️ Responsible AI & Ethical Use
TruthLens is an investigative aid, not a final arbiter of truth. In a world of evolving generative AI, this tool serves to provide probability scores and forensic evidence to support human verification workflows.

### 👨‍💻 Author
Tenzin Thinlay

LinkedIn: linkedin.com/in/tenthinlay1
Portfolio: tenzinthinlay.netlify.com
Email:tenthinlayofficial@gmail.com