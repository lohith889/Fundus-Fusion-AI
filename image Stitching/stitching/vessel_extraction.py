"""
Vessel Extraction Engine
========================
Blood vessel segmentation from fundus images using pure OpenCV + NumPy.
Single public function: extract_vessel_map(img_bgr, use_frangi=False)
"""

import cv2
import numpy as np


def _resize_if_needed(img):
    """Downscale if max dimension > 2000 px, preserving aspect ratio."""
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim <= 2000:
        return img, 1.0
    scale = 2000.0 / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _get_retinal_mask(img):
    """Return a binary mask of the retinal disc (non-black region)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def _segment_vessels(gray, use_frangi=False):
    """Core 7-step vessel segmentation pipeline."""
    # 1. Already have green channel as gray

    # 2. CLAHE equalisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Background normalisation via morphological opening
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bg = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_bg)
    norm = cv2.subtract(enhanced, bg)

    # 4. Gaussian blur
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)

    # 5. Adaptive threshold
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # 6. Morphological cleanup
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)   # remove noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_clean)  # fill gaps

    # 7. Optional Frangi filter
    if use_frangi:
        binary = _apply_frangi(norm, binary)

    return binary


def _apply_frangi(norm, binary):
    """Multi-scale Hessian-based Frangi vessel enhancement."""
    sigmas = [1, 2, 4]
    frangi_acc = np.zeros_like(norm, dtype=np.float64)

    for sigma in sigmas:
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(norm.astype(np.float64), (ksize, ksize), sigma)

        # Hessian components via Sobel
        dxx = cv2.Sobel(blurred, cv2.CV_64F, 2, 0, ksize=3)
        dyy = cv2.Sobel(blurred, cv2.CV_64F, 0, 2, ksize=3)
        dxy = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=3)

        # Eigenvalues of the Hessian
        tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
        lambda1 = 0.5 * ((dxx + dyy) + tmp)
        lambda2 = 0.5 * ((dxx + dyy) - tmp)

        # Vesselness: large |lambda2| with small |lambda1|
        abs_l2 = np.abs(lambda2)
        vesselness = np.where(lambda2 < 0, abs_l2, 0)
        frangi_acc = np.maximum(frangi_acc, vesselness)

    # Normalise and threshold
    if frangi_acc.max() > 0:
        frangi_acc = (frangi_acc / frangi_acc.max() * 255).astype(np.uint8)
    else:
        frangi_acc = frangi_acc.astype(np.uint8)

    _, frangi_binary = cv2.threshold(frangi_acc, 30, 255, cv2.THRESH_BINARY)

    # Combine with adaptive threshold result
    combined = cv2.bitwise_and(binary, frangi_binary)
    return combined


def _check_threshold_direction(gray, binary, retinal_mask):
    """
    If vessel_density < 1%, re-threshold with BINARY_INV and pick
    whichever gives higher (but not >40%) density.
    """
    retinal_pixels = max(np.count_nonzero(retinal_mask), 1)
    vessel_pixels = np.count_nonzero(cv2.bitwise_and(binary, retinal_mask))
    density = vessel_pixels / retinal_pixels * 100

    if density < 1.0:
        # Try inverted
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        bg = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_bg)
        norm = cv2.subtract(enhanced, bg)
        blurred = cv2.GaussianBlur(norm, (5, 5), 0)
        binary_inv = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel_clean)
        binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel_clean)

        inv_pixels = np.count_nonzero(cv2.bitwise_and(binary_inv, retinal_mask))
        inv_density = inv_pixels / retinal_pixels * 100

        if inv_density > density and inv_density <= 40.0:
            return binary_inv

    return binary


def _create_overlay(img_bgr, binary_mask):
    """Blend teal colour (0, 184, 142) at 60% opacity over vessel pixels."""
    overlay = img_bgr.copy()
    teal = np.array([142, 184, 0], dtype=np.uint8)  # BGR: (0, 184, 142) -> OpenCV BGR

    mask_bool = binary_mask == 255
    if np.any(mask_bool):
        # Create teal-coloured version
        teal_layer = np.full_like(img_bgr, teal)
        # Blend only on masked pixels: 40% original + 60% teal
        overlay[mask_bool] = cv2.addWeighted(
            img_bgr, 0.4, teal_layer, 0.6, 0
        )[mask_bool]

    return overlay


def _compute_stats(binary_mask, retinal_mask, h, w):
    """Compute vessel density, pixel count, and image size."""
    retinal_pixels = max(np.count_nonzero(retinal_mask), 1)
    vessel_pixels = int(np.count_nonzero(cv2.bitwise_and(binary_mask, retinal_mask)))
    density = round(vessel_pixels / retinal_pixels * 100, 2)

    return {
        "vessel_density": str(density),
        "vessel_pixels": str(vessel_pixels),
        "image_size": f"{w}x{h}",
    }


def extract_vessel_map(img_bgr, use_frangi=False):
    """
    Main public function for vessel segmentation.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input fundus image in BGR format.
    use_frangi : bool
        Whether to apply Frangi filter (slower, off by default).

    Returns
    -------
    dict with keys: "vessel_map", "overlay", "stats"
    Never raises — returns black images on error.
    """
    try:
        orig_h, orig_w = img_bgr.shape[:2]

        # Auto-resize guard
        small, scale = _resize_if_needed(img_bgr)
        sh, sw = small.shape[:2]

        # 1. Green channel extraction
        gray = small[:, :, 1]

        # Get retinal disc mask
        retinal_mask = _get_retinal_mask(small)

        # Run segmentation pipeline
        binary = _segment_vessels(gray, use_frangi=use_frangi)

        # Mask to retinal disc only
        binary = cv2.bitwise_and(binary, retinal_mask)

        # Check threshold direction
        binary = _check_threshold_direction(gray, binary, retinal_mask)
        binary = cv2.bitwise_and(binary, retinal_mask)

        # Upscale back to original dimensions if we downscaled
        if scale < 1.0:
            binary = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            retinal_mask = cv2.resize(retinal_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Create overlay on original image
        overlay = _create_overlay(img_bgr, binary)

        # Compute stats
        stats = _compute_stats(binary, retinal_mask, orig_h, orig_w)

        return {
            "vessel_map": binary,
            "overlay": overlay,
            "stats": stats,
        }

    except Exception:
        h, w = img_bgr.shape[:2] if img_bgr is not None else (100, 100)
        return {
            "vessel_map": np.zeros((h, w), dtype=np.uint8),
            "overlay": np.zeros((h, w, 3), dtype=np.uint8),
            "stats": {
                "vessel_density": "0",
                "vessel_pixels": "0",
                "image_size": "error",
            },
        }
