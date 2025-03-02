import torch
import pydicom
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

MEAN, STD = 0.49, 0.248

def load_dicom_image(dcm_path):
    dcm = pydicom.dcmread(dcm_path).pixel_array
    dcm = cv2.resize(dcm, (224, 224)).astype(np.float32) / 255.0
    dcm = torch.tensor(dcm).unsqueeze(0)  # Add channel dimension
    dcm = transforms.Normalize([MEAN], [STD])(dcm)
    return dcm

def cam(model, img):
    """
    Compute class activation map
    """
    with torch.no_grad():
        pred, features = model(img.unsqueeze(0))
    b, c, h, w = features.shape

    # Reshape feature tensor
    features = features.reshape((c, h * w))
    
    # Get weights of last fully connected layer
    weight_params = list(model.model.fc.parameters())[0] 
    weight = weight_params[0].detach()

    # Compute class activation map
    cam = torch.matmul(weight, features)

    # Normalize CAM
    cam = cam - torch.min(cam)
    cam_img = cam / torch.max(cam)
    cam_img = cam_img.reshape(h, w).cpu()

    return cam_img, torch.sigmoid(pred)

def visualize(img, heatmap, pred):
    """
    Visualization function for class activation maps
    """
    img = img[0]
    heatmap = transforms.functional.resize(heatmap.unsqueeze(0), (img.shape[0], img.shape[1]))[0]
    
    fig, axis = plt.subplots(1, 2)
    axis[0].imshow(img, cmap="bone")
    axis[1].imshow(img, cmap="bone")
    axis[1].imshow(heatmap, alpha=0.5, cmap="jet")
    plt.title(f"Pneumonia: {(pred > 0.5).item()}")
    plt.show()
