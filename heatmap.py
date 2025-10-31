import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import matplotlib.cm as cm

# -----------------------------
# Model Setup (4-class)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)  # same arch as training
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 output classes

state_dict = torch.load("brain_tumor_model_v2.pth", map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Define class names (update if different)
class_names = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# -----------------------------
# Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# -----------------------------
# Grad-CAM Implementation
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        gradcam_map = (weights * self.activations).sum(dim=1, keepdim=True)
        gradcam_map = torch.relu(gradcam_map)
        gradcam_map = torch.nn.functional.interpolate(
            gradcam_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        gradcam_map = gradcam_map.squeeze().cpu().numpy()
        gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-8)
        return gradcam_map, class_idx

gradcam = GradCAM(model, model.layer4[1].conv2)  # last conv layer

# -----------------------------
# GUI Setup
# -----------------------------
root = tk.Tk()
root.title("Brain Tumor Detection with Grad-CAM")
root.geometry("800x600")

panel_img = tk.Label(root)
panel_img.pack(side="left", padx=10, pady=10)

panel_cam = tk.Label(root)
panel_cam.pack(side="right", padx=10, pady=10)

label_pred = tk.Label(root, text="", font=("Arial", 16))
label_pred.pack(pady=20)

def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    # Load original image
    pil_img = Image.open(file_path).convert("L")
    img_resized = pil_img.resize((224, 224))
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Run Grad-CAM
    heatmap, class_idx = gradcam(img_tensor)
    pred_class = class_names[class_idx]

    # Overlay heatmap on image
    img_np = np.array(img_resized)
    if len(img_np.shape) == 2:  # grayscale -> RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    heatmap_colored = cm.jet(heatmap)[..., :3] * 255
    heatmap_colored = heatmap_colored.astype(np.uint8)
    overlay = cv2.addWeighted(img_np, 0.5, heatmap_colored, 0.5, 0)

    # Convert to Tkinter images
    orig_tk = ImageTk.PhotoImage(Image.fromarray(img_np))
    cam_tk = ImageTk.PhotoImage(Image.fromarray(overlay))

    panel_img.config(image=orig_tk)
    panel_img.image = orig_tk
    panel_cam.config(image=cam_tk)
    panel_cam.image = cam_tk

    label_pred.config(text=f"Prediction: {pred_class}")

btn_upload = tk.Button(root, text="Upload MRI", command=upload_and_predict, font=("Arial", 14))
btn_upload.pack(pady=10)

root.mainloop()
