"""
Screening Engine  —  RETFound Glaucoma Detection
=================================================
Loads the fine-tuned RETFound ViT-Large model and runs inference
on a retinal image to classify it as normal or glaucoma.
Produces annotated images with prediction overlays.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# ── Add RETFound directory to path so we can import models_vit ────────────────
_RETFOUND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "model", "RETFound"
)
sys.path.insert(0, _RETFOUND_DIR)
import models_vit  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "Glaucoma"]
CLASS_COLORS_BGR = [
    (80, 200, 80),    # Normal  — green
    (60, 60, 220),    # Glaucoma — red
]
IMG_SIZE = 256
MODEL_PATH = os.path.join(_RETFOUND_DIR, "output_dir", "checkpoint-best.pth")

# ── Lazy-loaded model cache ──────────────────────────────────────────────────
_model = None
_device = None


def _get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _load_model():
    """Load the fine-tuned RETFound model (lazy, cached)."""
    global _model
    if _model is not None:
        return _model

    device = _get_device()
    print(f"[Screening] Loading RETFound model on {device}...")
    print(f"[Screening] Checkpoint path: {MODEL_PATH}")

    model = models_vit.RETFound_mae(
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        drop_path_rate=0.1,
        global_pool=True,
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    _model = model
    print("[Screening] Model loaded successfully.")
    return _model


# ── Preprocessing ─────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """OpenCV BGR -> PIL RGB Image."""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_disease(img_bgr: np.ndarray) -> dict:
    """
    Run the RETFound model on a BGR image.

    Returns
    -------
    dict with keys:
        class_name  : str   ("Normal" or "Glaucoma")
        class_idx   : int
        confidence  : float (0-1, confidence for the predicted class)
        probs       : list[float] (probability per class)
    """
    model = _load_model()
    device = _get_device()

    pil_img = _bgr_to_pil(img_bgr)
    tensor = _transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=-1).squeeze()

    # Ensure it's 1D
    if probs.dim() > 1:
        probs = probs.flatten()

    class_idx = int(torch.argmax(probs).item())
    confidence = float(probs[class_idx].item())
    prob_list = [float(p) for p in probs.tolist()]

    return {
        "class_name": CLASS_NAMES[class_idx],
        "class_idx": class_idx,
        "confidence": confidence,
        "probs": prob_list,
    }


# ── Annotation ────────────────────────────────────────────────────────────────

def annotate_image(img_bgr: np.ndarray, prediction: dict) -> np.ndarray:
    """
    Draw disease prediction annotation on the image.

    Adds:
      - Thick coloured border (green = normal, red = glaucoma)
      - Semi-transparent banner at the top with class name + confidence
      - Probability bars for each class
    """
    h, w = img_bgr.shape[:2]
    annotated = img_bgr.copy()

    cls_idx = prediction["class_idx"]
    cls_name = prediction["class_name"]
    confidence = prediction["confidence"]
    probs = prediction["probs"]
    color = CLASS_COLORS_BGR[cls_idx]

    # ── 1. Coloured border ────────────────────────────────────────────────────
    border_w = max(4, min(h, w) // 80)
    cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), color, border_w)

    # ── 2. Top banner ─────────────────────────────────────────────────────────
    banner_h = max(70, h // 8)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    # Status icon
    icon_radius = max(12, banner_h // 5)
    icon_cx = border_w + icon_radius + 15
    icon_cy = banner_h // 2
    cv2.circle(annotated, (icon_cx, icon_cy), icon_radius, color, -1)

    # Disease name
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_x = icon_cx + icon_radius + 15
    font_scale = max(0.6, min(h, w) / 800)
    thickness = max(1, int(font_scale * 2))

    cv2.putText(
        annotated, f"Screening Result: {cls_name}",
        (text_x, icon_cy - 5),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )
    cv2.putText(
        annotated, f"Confidence: {confidence * 100:.1f}%",
        (text_x, icon_cy + int(font_scale * 28)),
        font, font_scale * 0.7, color, max(1, thickness - 1), cv2.LINE_AA,
    )

    # ── 3. Bottom probability bars ────────────────────────────────────────────
    bar_area_h = max(55, h // 10)
    bar_y_start = h - bar_area_h
    overlay2 = annotated.copy()
    cv2.rectangle(overlay2, (0, bar_y_start), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.7, annotated, 0.3, 0, annotated)

    bar_margin = 15
    bar_h = max(10, bar_area_h // 5)
    bar_max_w = w - 2 * bar_margin - 120

    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        bar_y = bar_y_start + 12 + i * (bar_h + 10)
        label = f"{name}: {prob * 100:.1f}%"
        cv2.putText(
            annotated, label,
            (bar_margin, bar_y + bar_h - 2),
            font, font_scale * 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )
        bar_x0 = bar_margin + 110
        bar_w_actual = int(bar_max_w * prob)
        bar_color = CLASS_COLORS_BGR[i]

        # Background track
        cv2.rectangle(
            annotated, (bar_x0, bar_y), (bar_x0 + bar_max_w, bar_y + bar_h),
            (40, 40, 40), -1,
        )
        # Filled bar
        if bar_w_actual > 0:
            cv2.rectangle(
                annotated, (bar_x0, bar_y), (bar_x0 + bar_w_actual, bar_y + bar_h),
                bar_color, -1,
            )

    return annotated
