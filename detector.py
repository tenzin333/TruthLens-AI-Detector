import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. Load Model (Optimized with Caching)
@st.cache_resource
def load_model():
    model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model, device

# 2. Main AI Detection Logic
def get_ai_score(image_path):
    processor, model, device = load_model()
    img = Image.open(image_path).convert("RGB")
    
    inputs = processor(images=img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Class 0: Real, Class 1: Fake (Deepfake)
    fake_prob = probs[0][1].item()
    return round(fake_prob * 100, 2)

# 3. Forensic Frequency Analysis
def get_forensic_analysis(image_path):
    img = cv2.imread(image_path, 0)
    if img is None: return "File Error"
    
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-9)
    
    h, w = magnitude_spectrum.shape
    corner_size = 30
    corners = [
        magnitude_spectrum[0:corner_size, 0:corner_size],
        magnitude_spectrum[h-corner_size:h, w-corner_size:w]
    ]
    avg_variance = np.mean([np.var(c) for c in corners])
    
    return "High Artifacts" if avg_variance > 150 else "Natural Texture"

# 4. Explainable AI Heatmap (Fixed for ViT)
def reshape_transform(tensor):
    # ViT output is (Batch, Tokens, Channels). We remove CLS token (index 0).
    # 197 tokens -> 1 CLS + 196 patches (14x14 grid)
    result = tensor[:, 1:, :].reshape(tensor.size(0), 14, 14, tensor.size(2))
    # Bring Channels to the front for PyTorch: (Batch, C, H, W)
    return result.transpose(2, 3).transpose(1, 2)

# 1. Add this wrapper class (standard for HuggingFace + GradCAM)
class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # GradCAM expects a Tensor, so we extract just the logits
        return self.model(x).logits

# 2. Update your generate_heatmap function
def generate_heatmap(image_path):
    processor, raw_model, device = load_model()
    
    # Wrap the model so it returns a Tensor instead of a dictionary
    model = HuggingfaceToTensorModelWrapper(raw_model)
    
    # Target the final normalization layer
    target_layers = [raw_model.vit.encoder.layer[-1].layernorm_before]
    
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_tensor = processor(images=img, return_tensors="pt").pixel_values.to(device)
    
    # Initialize GradCAM with the wrapper and the reshape transform
    cam = GradCAM(model=model, 
                  target_layers=target_layers, 
                  reshape_transform=reshape_transform)
    
    targets = [ClassifierOutputTarget(1)] 
    
    # Generate the heatmap
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
    
    img_array = np.array(img) / 255.0
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    return visualization