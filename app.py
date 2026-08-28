import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import streamlit as st
import timm
import matplotlib.pyplot as plt

# ==============================
# Configuration
# ==============================
CLASSES = ['glioma', 'meningioma', 'pituitary', 'notumor']
CLASS_LABELS = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
IMG_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

st.set_page_config(
    page_title="Brain Tumor XAI - MobileNetV3",
    page_icon="\U0001f9e0",
    layout="wide"
)

# ==============================
# Model Definition (FedViT-CNN: MobileNetV3-Small + ViT-Tiny)
# ==============================
class FedViTCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.cnn = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=0)
        self.cnn_fc = nn.Linear(1024, 256)
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0)
        self.vit_fc = nn.Linear(192, 256)
        self.attention_fusion = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn_fc(self.cnn(x))
        vit_feat = self.vit_fc(self.vit(x))
        combined = torch.cat((cnn_feat, vit_feat), dim=1)
        attn_weights = self.attention_fusion(combined)
        attended = combined * attn_weights
        return self.classifier(attended)


# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    model = FedViTCNN(num_classes=len(CLASSES)).to(DEVICE)
    model_path = os.path.join(os.path.dirname(__file__), "best_vitcnn_mobilenet.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model
    return None


model = load_model()

# ==============================
# Image Transforms
# ==============================
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# ==============================
# Grad-CAM (hooks MobileNetV3 last block)
# ==============================
def get_gradcam(image_tensor, original_image):
    model.eval()

    cnn_gradients = []
    cnn_activations = []

    def save_gradient(module, grad_input, grad_output):
        cnn_gradients.append(grad_output[0])

    def save_activation(module, inp, out):
        cnn_activations.append(out)

    target_layer = model.cnn.blocks[-1]
    hook_f = target_layer.register_forward_hook(save_activation)
    hook_b = target_layer.register_full_backward_hook(save_gradient)

    # Disable inplace ops so backward hooks work
    inplace_layers = []
    for m in model.modules():
        if hasattr(m, 'inplace') and m.inplace:
            m.inplace = False
            inplace_layers.append(m)

    output = model(image_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    pred_idx = output.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item() * 100

    model.zero_grad()
    output[0, pred_idx].backward()

    for m in inplace_layers:
        m.inplace = True

    hook_f.remove()
    hook_b.remove()

    # Compute CAM
    grads = cnn_gradients[0].cpu().data.numpy()[0]      # (C, H, W)
    acts = cnn_activations[0].cpu().data.numpy()[0]     # (C, H, W)

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig_np = np.array(original_image.resize((IMG_SIZE, IMG_SIZE)))
    overlay = cv2.addWeighted(orig_np, 0.5, heatmap, 0.5, 0)

    return orig_np, heatmap, overlay, CLASSES[pred_idx], CLASS_LABELS[pred_idx], confidence, probs[0].detach().cpu().numpy()


# ==============================
# UI
# ==============================
st.title("\U0001f9e0 FedViT-CNN: Brain Tumor Classification + Grad-CAM")
st.markdown(
    "Upload an **axial brain MRI scan**. The model (MobileNetV3-Small + ViT-Tiny fusion) "
    "will classify the tumor type and generate a **Grad-CAM heatmap** highlighting the "
    "focal regions driving its prediction."
)

if model is None:
    st.error(
        "**Model weights not found.** "
        "Ensure `best_vitcnn_mobilenet.pth` is in the repository root."
    )
else:
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_tensor = val_transforms(image).unsqueeze(0).to(DEVICE)

        with st.spinner("Analyzing MRI and generating Grad-CAM heatmap..."):
            orig_np, heatmap_np, overlay_np, pred_key, pred_label, conf, probs = get_gradcam(img_tensor, image)

        st.write("---")

        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            if pred_key == "notumor":
                st.success(f"No tumor detected ({conf:.1f}%)")
            else:
                st.error(f"Tumor detected: **{pred_label}** ({conf:.1f}%)")
        with col2:
            st.metric("Predicted Diagnosis", pred_label)
        with col3:
            st.metric("AI Confidence", f"{conf:.2f}%")

        st.write("---")

        # Image row: Original | Heatmap | Overlay
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.subheader("Original MRI")
            st.image(orig_np, use_container_width=True)
        with col_b:
            st.subheader("Grad-CAM Heatmap")
            st.image(heatmap_np, use_container_width=True)
            st.caption("\U0001f534 Red = focal regions driving the prediction")
        with col_c:
            st.subheader("Overlay")
            st.image(overlay_np, use_container_width=True)
            st.caption("Heatmap blended on MRI (50/50)")

        st.write("---")

        # Probability bar chart
        st.subheader("\U0001f4ca Classification Probabilities")
        fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        colors = ["#ef4444" if c == pred_key else "#60a5fa" for c in CLASSES]
        bars = ax.barh(CLASS_LABELS, probs * 100, color=colors)
        ax.set_xlabel("Probability (%)", color="white")
        ax.set_xlim(0, 100)
        ax.tick_params(colors="white")
        ax.bar_label(bars, fmt="%.1f%%", padding=4, color="white")
        ax.set_title("Model Confidence per Class", color="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        plt.tight_layout()
        st.pyplot(fig)
