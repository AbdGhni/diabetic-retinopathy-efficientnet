"""Gradio demo: diabetic retinopathy grading from a fundus photo (research prototype, not for clinical use)."""
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import gradio as gr

IMG_SIZE = 300
GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DISCLAIMER = ("Research prototype trained on the public APTOS 2019 dataset. "
              "Not a medical device. Not for clinical use.")


class DRClassifier(nn.Module):
    """Same architecture as the training notebook."""

    def __init__(self, num_classes=5, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b3", pretrained=False)
        n = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(n, 512), nn.ReLU(inplace=True),
            nn.BatchNorm1d(512), nn.Dropout(dropout * 0.75), nn.Linear(512, num_classes))

    def forward(self, x):
        return self.classifier(self.backbone(x))


def crop_circular(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 15
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    return img[max(0, ys.min() - 5):ys.max() + 5, max(0, xs.min() - 5):xs.max() + 5]


def preprocess_fundus(img_bgr):
    img = cv2.resize(crop_circular(img_bgr), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(img, (0, 0), 30)
    return cv2.addWeighted(img, 4, blur, -4, 128)  # Ben Graham


ckpt = torch.load("best_dr_model.pth", map_location="cpu", weights_only=False)
MODEL = DRClassifier(dropout=ckpt.get("config", {}).get("dropout", 0.4))
MODEL.load_state_dict(ckpt["model_state_dict"])
MODEL.eval()


def predict(image):
    if image is None:
        return {}, None, ""
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    proc = cv2.cvtColor(preprocess_fundus(bgr), cv2.COLOR_BGR2RGB)
    x = ((proc.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    with torch.no_grad():
        p = torch.softmax(MODEL(torch.from_numpy(x)), 1)[0].numpy()
    referable = p[2:].sum()
    note = f"Probability of referable DR (grade 2 or higher): {referable:.0%}"
    return {g: float(v) for g, v in zip(GRADES, p)}, proc, note


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Fundus photograph"),
    outputs=[gr.Label(label="Predicted grade"), gr.Image(label="Preprocessed input (CLAHE + Ben Graham)"),
             gr.Textbox(label="Screening view")],
    title="Diabetic retinopathy grading (EfficientNet-B3)",
    description=DISCLAIMER,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
