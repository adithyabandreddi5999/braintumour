import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# ================================
# Load Model
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names (adjust if needed)
classes = ["Meningioma", "Glioma", "No Tumo", "Pituitary"]

# Create ResNet18 but adjust first conv to accept 1-channel input
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))

# Load trained weights
model_path = os.path.join(os.path.dirname(__file__), "brain_tumor_model_v2.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ================================
# Image Preprocessing
# ================================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 👈 ensure grayscale input
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # normalize single channel
])

def predict(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = classes[probs.argmax()]
    return pred_class, probs

# ================================
# Streamlit UI
# ================================
st.title("🧠 Brain Tumor Detection & Classification")
st.write("Upload an **axial MRI scan** to detect tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # open as RGB
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Prediction
    pred_class, probs = predict(image)

    st.markdown(f"### ✅ Prediction: **{pred_class}**")

    # Probability chart
    fig, ax = plt.subplots()
    ax.bar(classes, probs, color="skyblue")
    ax.set_ylabel("Probability")
    ax.set_title("Classification Probabilities")
    st.pyplot(fig)
