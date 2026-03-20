"""
Stitching Routes (Flask Blueprint)
===================================
Synced with new engine.py — uses correct function names:
  retina_preprocess, retina_detect_features, retina_match_features,
  retina_stitch_images
"""

import os
import uuid
import base64
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, send_from_directory

import time

from stitching.engine import (
    retina_preprocess,
    retina_detect_features,
    retina_match_features,
    retina_stitch_images,
)
from stitching.vessel_extraction import extract_vessel_map

# Blueprint
stitching_bp = Blueprint("stitching", __name__)

# Paths
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_FOLDER = os.path.join(_PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────
# BASE64 DECODER
# ─────────────────────────────────────────────
def decode_base64_image(data_url: str) -> np.ndarray:
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _load_images(raw_images: list) -> tuple:
    """Decode a list of base64 strings → list of BGR images.
    Returns (images, error_str_or_None)."""
    images = []
    for b64 in raw_images:
        img = decode_base64_image(b64)
        if img is not None:
            images.append(img)
    if len(images) < 2:
        return [], "Image decoding failed — need at least 2 valid images."
    return images, None


def _b64_encode_image(img: np.ndarray, ext: str = ".png") -> str:
    """Encode a BGR numpy image to a base64 data-URL string."""
    _, buf = cv2.imencode(ext, img)
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buf).decode('utf-8')}"


def _save_and_encode(img: np.ndarray, prefix: str) -> tuple:
    """Save image to OUTPUT_FOLDER and return (filename, b64_data_url)."""
    out_name = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    out_path = os.path.join(OUTPUT_FOLDER, out_name)
    cv2.imwrite(out_path, img)
    return out_name, _b64_encode_image(img, ".png")


# ─────────────────────────────────────────────
# NORMAL STITCH  (OpenCV built-in stitcher)
# ─────────────────────────────────────────────
@stitching_bp.route("/stitch", methods=["POST"])
def stitch():
    data = request.get_json()
    if not data or "images" not in data:
        return jsonify({"success": False, "error": "No images received."}), 400
    if len(data["images"]) < 2:
        return jsonify({"success": False, "error": "Need at least 2 images."}), 400

    images, err = _load_images(data["images"])
    if err:
        return jsonify({"success": False, "error": err}), 400

    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        return jsonify({
            "success": False,
            "error": f"OpenCV stitching failed (status code {status})"
        }), 422

    out_name, result_b64 = _save_and_encode(stitched, "stitched")
    return jsonify({"success": True, "image": result_b64, "filename": out_name})


# ─────────────────────────────────────────────
# RETINAL STITCH  (custom feather-blend pipeline)
# ─────────────────────────────────────────────
@stitching_bp.route("/stitch-retinal", methods=["POST"])
def stitch_retinal():
    data = request.get_json()
    if not data or "images" not in data:
        return jsonify({"success": False, "error": "No images received."}), 400
    if len(data["images"]) < 2:
        return jsonify({"success": False, "error": "Need at least 2 images."}), 400

    images, err = _load_images(data["images"])
    if err:
        return jsonify({"success": False, "error": err}), 400

    try:
        success, result = retina_stitch_images(images)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    if not success:
        # result is an error string when success=False
        return jsonify({"success": False, "error": result}), 422

    out_name, result_b64 = _save_and_encode(result, "retinal")
    return jsonify({"success": True, "image": result_b64, "filename": out_name})


# ─────────────────────────────────────────────
# FEATURE VISUALISATION  (debug tool)
# ─────────────────────────────────────────────
@stitching_bp.route("/visualize-features", methods=["POST"])
def visualize_features():
    data = request.get_json()
    if not data or "images" not in data:
        return jsonify({"success": False, "error": "No images received."}), 400
    if len(data["images"]) < 2:
        return jsonify({"success": False, "error": "Need at least 2 images."}), 400

    images, err = _load_images(data["images"])
    if err:
        return jsonify({"success": False, "error": err}), 400

    keypoint_images_b64 = []
    match_images_b64    = []
    keypoints_list      = []
    descriptors_list    = []

    # ── Keypoint visualisation ───────────────────────────────────────────────
    for i, img in enumerate(images):
        preprocessed = retina_preprocess(img)          # returns enhanced gray
        kp, desc     = retina_detect_features(preprocessed)  # mask=None here

        keypoints_list.append(kp   or [])
        descriptors_list.append(desc)

        vis = cv2.drawKeypoints(
            img, kp or [], None,
            color=(0, 255, 128),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        label = f"Image {i + 1}: {len(kp) if kp else 0} keypoints"
        cv2.putText(vis, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 128), 2)

        keypoint_images_b64.append(_b64_encode_image(vis, ".jpg"))

    # ── Match visualisation ──────────────────────────────────────────────────
    for i in range(len(images) - 1):
        desc1 = descriptors_list[i]
        desc2 = descriptors_list[i + 1]

        if desc1 is None or desc2 is None:
            continue

        matches = retina_match_features(desc1, desc2)
        top60   = sorted(matches, key=lambda x: x.distance)[:60]

        vis = cv2.drawMatches(
            images[i],     keypoints_list[i],
            images[i + 1], keypoints_list[i + 1],
            top60, None,
            matchColor=(0, 200, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        label = f"{len(matches)} matches (showing top 60)"
        cv2.putText(vis, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 200, 255), 2)

        match_images_b64.append(_b64_encode_image(vis, ".jpg"))

    return jsonify({
        "success":   True,
        "keypoints": keypoint_images_b64,
        "matches":   match_images_b64,
    })


# ─────────────────────────────────────────────
# VESSEL EXTRACTION
# ─────────────────────────────────────────────
@stitching_bp.route("/process-vessel", methods=["POST"])
def process_vessel():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"success": False, "error": "No image received."}), 400

    try:
        img = decode_base64_image(data["image"])
    except Exception:
        return jsonify({"success": False, "error": "Failed to decode image."}), 400

    if img is None:
        return jsonify({"success": False, "error": "Failed to decode image."}), 400

    try:
        t0 = time.time()
        result = extract_vessel_map(img, use_frangi=False)
        ms = int((time.time() - t0) * 1000)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    vessel_map_b64 = _b64_encode_image(result["vessel_map"], ".png")
    overlay_b64 = _b64_encode_image(result["overlay"], ".png")

    return jsonify({
        "success": True,
        "vessel_map": vessel_map_b64,
        "overlay": overlay_b64,
        "stats": result["stats"],
        "processing_time_ms": ms,
    })


# ─────────────────────────────────────────────
# DISEASE SCREENING  (RETFound model)
# ─────────────────────────────────────────────
@stitching_bp.route("/screen", methods=["POST"])
def screen_disease():
    """Run RETFound disease screening on a retinal image."""
    from stitching.screening import predict_disease, annotate_image

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"success": False, "error": "No image received."}), 400

    try:
        img = decode_base64_image(data["image"])
    except Exception:
        return jsonify({"success": False, "error": "Failed to decode image."}), 400

    if img is None:
        return jsonify({"success": False, "error": "Failed to decode image."}), 400

    try:
        prediction = predict_disease(img)
        annotated = annotate_image(img, prediction)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Screening error: {exc}"}), 500

    out_name, annotated_b64 = _save_and_encode(annotated, "screened")

    return jsonify({
        "success": True,
        "prediction": {
            "class": prediction["class_name"],
            "confidence": round(prediction["confidence"] * 100, 1),
            "probs": {
                name: round(p * 100, 1)
                for name, p in zip(["Normal", "Glaucoma"], prediction["probs"])
            },
            "severity": {
                "level": prediction["severity_level"],
                "label": prediction["severity_label"],
                "glaucoma_probability": round(prediction["glaucoma_prob"] * 100, 1),
            },
        },
        "annotated_image": annotated_b64,
        "filename": out_name,
    })


# ─────────────────────────────────────────────
# SERVE OUTPUT FILES
# ─────────────────────────────────────────────
@stitching_bp.route("/outputs/<path:filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)