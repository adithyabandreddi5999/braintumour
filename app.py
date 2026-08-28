import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2

st.set_page_config(page_title="Brain Tumor Detection", page_icon=":brain:", layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]


@st.cache_resource
def load_model():
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    model_path = os.path.join(os.path.dirname(__file__), "brain_tumor_model_v2.pth")
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.to(device)
    m.eval()
    return m


model = load_model()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def predict(image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        pred_class = classes[pred_idx]
    return pred_class, probs, pred_idx


def generate_gradcam(image, pred_idx):
    tensor = transform(image).unsqueeze(0).to(device)

    gradients = []
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.layer4[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    inplace_layers = []
    for m in model.modules():
        if hasattr(m, 'inplace') and m.inplace:
            m.inplace = False
            inplace_layers.append(m)

    output = model(tensor)
    model.zero_grad()
    output[0, pred_idx].backward()

    for m in inplace_layers:
        m.inplace = True

    fh.remove()
    bh.remove()

    grads = gradients[0].cpu().numpy()[0]
    acts = activations[0].cpu().detach().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig_rgb = np.array(image.convert("RGB").resize((224, 224)))
    overlay = cv2.addWeighted(orig_rgb, 0.55, heatmap, 0.45, 0)

    return heatmap, overlay


st.title(":brain: Brain Tumor Detection & Classification")
st.write("Upload an **axial MRI scan** to detect tumor type and view the **Grad-CAM** attention heatmap.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Analyzing MRI scan..."):
        pred_class, probs, pred_idx = predict(image)
        heatmap, overlay = generate_gradcam(image, pred_idx)

    st.write("---")

    col1, col2, col3 = st.columns(3)
    conf = float(probs[pred_idx]) * 100
    with col1:
        if pred_class == "No Tumor":
            st.success(f":white_check_mark: **{pred_class}** ({conf:.1f}%)")
        else:
            st.error(f":warning: **{pred_class}** ({conf:.1f}%)")
    with col2:
        st.metric("Predicted Class", pred_class)
    with col3:
        st.metric("Confidence", f"{conf:.2f}%")

    st.write("---")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader("Original MRI")
        st.image(image.convert("RGB").resize((224, 224)), use_container_width=True)
    with col_b:
        st.subheader("Grad-CAM Heatmap")
        st.image(heatmap, use_container_width=True)
        st.caption(":red_circle: Red = regions driving the AI prediction")
    with col_c:
        st.subheader("Overlay")
        st.image(overlay, use_container_width=True)
        st.caption("Heatmap blended on MRI")

    st.write("---")

    st.subheader(":bar_chart: Classification Probabilities")
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#ef4444" if c == pred_class else "#60a5fa" for c in classes]
    bars = ax.barh(classes, probs * 100, color=colors)
    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, 100)
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_title("Model Confidence per Class")
    plt.tight_layout()
    st.pyplot(fig)
