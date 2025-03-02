import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.cm as cm
import pydicom
from model import PneumoniaModel  # Ensure this is correctly imported

# Load model
@st.cache_resource
def load_model():
    model = PneumoniaModel.load_from_checkpoint("checkpoints/weights_3.ckpt", strict=False)
    model.eval()
    return model

model = load_model()

# Function to load and preprocess DICOM image
def load_dicom(file):
    dicom = pydicom.dcmread(file)
    img = dicom.pixel_array.astype(np.float32)

    # Normalize to [0,1]
    img = (img - img.min()) / (img.max() - img.min())

    # Resize for model input
    img = cv2.resize(img, (224, 224))  

    # Convert to PyTorch tensor and add batch & channel dimensions
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # Shape: (1,1,224,224)

    return img_tensor, img  # Return both tensor and numpy version for display

# Function to compute Class Activation Map (CAM)
def compute_cam(model, img):
    with torch.no_grad():
        output = model(img)  # ❌ Removed incorrect `unsqueeze(0)`, img is already (1,1,224,224)

        # Ensure model returns both prediction and feature maps
        if isinstance(output, tuple) and len(output) == 2:
            pred, features = output
        else:
            raise ValueError("Model output must include both prediction and feature maps.")

    # Extract feature maps and weights
    features = features.squeeze(0)  # Shape: (C, H, W)
    weight_params = list(model.model.fc.parameters())[0]  # FC layer weights

    # Compute CAM
    cam = torch.matmul(weight_params[0], features.reshape(features.shape[0], -1))
    cam = cam.reshape(7, 7).detach().cpu().numpy()  # Reshape to match feature map size

    # Normalize CAM to [0,1]
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    # Resize CAM to input image size (224,224)
    cam_resized = cv2.resize(cam, (224, 224))

    # Convert CAM to a heatmap
    heatmap = cm.jet(cam_resized)[:, :, :3]  # Apply colormap and keep RGB channels

    return heatmap, torch.sigmoid(pred)

# Streamlit UI
st.title("Pneumonia Detection from X-ray")
st.write("Upload a DICOM X-ray image to analyze.")

uploaded_file = st.file_uploader("Upload DICOM file", type=["dcm"])

if uploaded_file:
    img_tensor, img_display = load_dicom(uploaded_file)

    # ✅ Ensure correct shape before displaying grayscale X-ray
    st.image(img_display, caption="Input X-ray", width=300, use_container_width=True)

    activation_map, pred = compute_cam(model, img_tensor)

    # ✅ Convert grayscale X-ray to RGB for overlaying CAM
    xray_rgb = np.stack([img_display] * 3, axis=-1)  # (224,224) → (224,224,3)

    # ✅ Blend CAM heatmap with X-ray
    overlay = (0.3 * xray_rgb + 0.7 * activation_map).clip(0, 1)  

    st.image(overlay, caption="Class Activation Map", use_container_width=True)

    # Display Prediction
    pred_label = "Pneumonia" if pred.item() > 0.5 else "No Pneumonia"
    st.markdown(f"### **Prediction: {pred_label}**", unsafe_allow_html=True)
