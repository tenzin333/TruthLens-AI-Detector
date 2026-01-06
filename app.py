import streamlit as st
import os
from detector import get_ai_score, get_forensic_analysis, generate_heatmap

# 1. This MUST be the first streamlit command
st.set_page_config(page_title="TruthLens AI Detector", layout="wide")

# 2. Optimized Imports: Load the heavy AI stuff ONLY when needed
def load_detector_logic():
    with st.spinner("Initializing AI Engines..."):
        from detector import get_ai_score, get_forensic_analysis, generate_heatmap
    return get_ai_score, get_forensic_analysis, generate_heatmap

st.title("🔍 TruthLens: Multi-Signal Media Verifier")

uploaded_file = st.file_uploader("Upload an image to verify...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save temp file
    temp_path = "temp_img.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load our heavy functions
    get_ai_score, get_forensic_analysis, generate_heatmap = load_detector_logic()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Content", use_container_width=True)
        
    with col2:
        st.header("Detection Results")
        
        # 1. AI Probability
        with st.status("🔍 Analyzing with Vision Transformer...", expanded=True) as status:
            try:
                score = get_ai_score(temp_path)
                st.metric("AI Generation Probability", f"{score}%")
                st.progress(score / 100)
                status.update(label="AI Analysis Complete!", state="complete")
            except Exception as e:
                st.error(f"AI Analysis Failed: {e}")
            
        # 2. Forensic Artifacts
        with st.spinner('Analyzing Pixel Frequency...'):
            forensics = get_forensic_analysis(temp_path)
            st.write(f"**Forensic Signature:** {forensics}")
            
        # 3. Metadata (Placeholder for now)
        st.info("C2PA Status: No Manifest Detected (Likely Synthetic)")
    
    if st.button("Generate Forensic Heatmap"):
        with st.spinner("Generating XAI Heatmap..."):
            heatmap = generate_heatmap("temp_img.jpg")
            st.image(heatmap, caption="AI Artifact Heatmap (Red = High Suspicion)")
            st.write("🔍 **Analysis:** The model is focusing on the highlighted regions to determine authenticity.")