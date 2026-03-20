"""
Image Stitching Engine — Retinal Edition (Fixed)
=================================================
Key fixes over original:
  1. Circular-mask extraction  — black backgrounds are never blended, only valid
     retinal pixels participate in compositing.
  2. Feather / distance-weighted blending — smooth seams without dark rectangles.
  3. Proper alpha accumulation using float32 weight maps instead of integer counters.
  4. Vessel-enhanced SIFT kept for robust feature detection on retinal texture.
"""

import base64
import cv2
import numpy as np


# ─── Utility ─────────────────────────────────────────────────────────────────

def decode_base64_image(data_url: str) -> np.ndarray:
    """Decode a base64 data-URL into an OpenCV image (BGR)."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ─── Mask Helpers ─────────────────────────────────────────────────────────────

def _extract_retinal_mask(img: np.ndarray) -> np.ndarray:
    """
    Return a binary mask of the illuminated retinal disc, ignoring the
    black circular border that fundus cameras produce.

    Strategy:
      • Convert to grayscale, threshold out near-black pixels.
      • Find the largest connected component (the retinal disc).
      • Optionally erode slightly to avoid blending fringe edge pixels.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold: pixels brighter than ~15/255 are considered retinal content
    _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # Keep only the largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels < 2:
        return binary  # fallback: return raw threshold

    # Label 0 is background; find the largest foreground component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest_label, np.uint8(255), np.uint8(0))

    # Slight erosion to pull away from the hard circular edge
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask


def _feather_weight_map(mask: np.ndarray, feather_radius: int = 60) -> np.ndarray:
    """
    Convert a binary mask into a smooth float32 weight map via distance transform.
    Pixels at the centre of the retinal disc get weight ≈ 1.0; pixels near the
    edge taper to 0.  This eliminates seam lines at image boundaries.
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # Normalise so interior reaches 1.0; clip to [0, 1]
    weight = np.clip(dist / feather_radius, 0.0, 1.0).astype(np.float32)
    return weight


# ─── Pre-processing & Feature Detection ──────────────────────────────────────

def retina_preprocess(img: np.ndarray) -> np.ndarray:
    """CLAHE on the green channel (best contrast for retinal vessels)."""
    # Green channel carries the most vessel detail in fundus images
    green = img[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(green)


def retina_enhance_vessels(gray: np.ndarray) -> np.ndarray:
    """Top-hat morphology to highlight thin vessels against background."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    # Blend back with original for richer texture (helps SIFT find keypoints)
    enhanced = cv2.addWeighted(gray, 0.7, tophat, 0.3, 0)
    return enhanced


def retina_detect_features(img: np.ndarray, mask: np.ndarray = None):
    """SIFT feature detection; optionally restrict to the retinal mask."""
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.03, edgeThreshold=12)
    kp, desc = sift.detectAndCompute(img, mask)
    return kp, desc


def retina_match_features(desc1, desc2):
    """Lowe ratio test matching (threshold 0.75)."""
    matcher = cv2.BFMatcher()
    raw = matcher.knnMatch(desc1, desc2, k=2)
    return [m for m, n in raw if m.distance < 0.75 * n.distance]


def retina_compute_homography(kp1, kp2, matches):
    """RANSAC homography from matched keypoints."""
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    return H


# ─── Core Compositing ────────────────────────────────────────────────────────

def _warp_with_mask(img: np.ndarray,
                    mask: np.ndarray,
                    H: np.ndarray,
                    canvas_size: tuple) -> tuple:
    """
    Warp both the image and its feathered weight map to the canvas.
    Returns (warped_bgr float32, warped_weight float32).
    """
    w, h = canvas_size  # note: cv2 uses (width, height)

    warped_img = cv2.warpPerspective(
        img.astype(np.float32), H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # Build feathered weight map in source space, then warp it
    weight_src = _feather_weight_map(mask)
    warped_weight = cv2.warpPerspective(
        weight_src, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    return warped_img, warped_weight


# ─── Public Pipeline ──────────────────────────────────────────────────────────

def retina_stitch_images(images: list) -> tuple:
    """
    Full retinal stitching pipeline.

    Parameters
    ----------
    images : list of np.ndarray
        BGR images in left-to-right (or any consecutive) order.

    Returns
    -------
    (True, mosaic_bgr)  on success
    (False, error_str)  on failure
    """
    if len(images) < 2:
        return False, "Need at least 2 images."

    # ── Extract retinal masks for every image ────────────────────────────────
    masks = [_extract_retinal_mask(img) for img in images]

    # ── Step 1: Pairwise homographies between consecutive images ─────────────
    pairwise_H = []
    for i in range(len(images) - 1):
        p1 = retina_preprocess(images[i])
        p2 = retina_preprocess(images[i + 1])

        v1 = retina_enhance_vessels(p1)
        v2 = retina_enhance_vessels(p2)

        # Detect inside mask only — avoids keypoints on the black border
        kp1, desc1 = retina_detect_features(v1, masks[i])
        kp2, desc2 = retina_detect_features(v2, masks[i + 1])

        if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
            return False, f"No features detected in image pair {i + 1}–{i + 2}."

        matches = retina_match_features(desc1, desc2)

        if len(matches) < 10:
            return False, (
                f"Insufficient feature matches between image {i + 1} and {i + 2} "
                f"({len(matches)} found, need ≥ 10). Ensure sufficient overlap."
            )

        H = retina_compute_homography(kp1, kp2, matches)
        if H is None:
            return False, f"Homography estimation failed for pair {i + 1}–{i + 2}."

        pairwise_H.append(H)

    # ── Step 2: Chain to reference frame (image[0]) ──────────────────────────
    H_to_ref = [np.eye(3, dtype=np.float64)]
    for H in pairwise_H:
        H_to_ref.append(H_to_ref[-1] @ H)

    # ── Step 3: Compute full canvas bounds ───────────────────────────────────
    all_corners = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        all_corners.append(cv2.perspectiveTransform(corners, H_to_ref[i]))
    all_corners = np.concatenate(all_corners, axis=0)

    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    x_max = int(np.ceil(all_corners[:, 0, 0].max()))
    y_max = int(np.ceil(all_corners[:, 0, 1].max()))

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    if canvas_w > 12000 or canvas_h > 12000:
        return False, "Mosaic too large — check for extreme viewpoint changes."

    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0,       1]
    ], dtype=np.float64)

    # ── Step 4: Feather-blend all images into the canvas ─────────────────────
    # Accumulate weighted sum in float32 to preserve precision
    accum_color  = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    accum_weight = np.zeros((canvas_h, canvas_w),    dtype=np.float64)

    for i, img in enumerate(images):
        H_final = translation @ H_to_ref[i]
        warped_img, warped_w = _warp_with_mask(img, masks[i], H_final,
                                               (canvas_w, canvas_h))

        # Expand weight to 3 channels for broadcasting
        w3 = warped_w[:, :, np.newaxis]

        accum_color  += warped_img  * w3
        accum_weight += warped_w

    # Avoid divide-by-zero in regions with no image coverage
    denom = np.maximum(accum_weight[:, :, np.newaxis], 1e-6)
    mosaic = np.clip(accum_color / denom, 0, 255).astype(np.uint8)

    # ── Step 5: Mask out fully uncovered canvas regions (keep black) ─────────
    # Any pixel whose total weight is effectively zero stays black
    coverage = (accum_weight > 0.01).astype(np.uint8) * 255
    mosaic[coverage == 0] = 0

    # ── Step 6: Crop to tight bounding box of covered pixels ─────────────────
    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is not None:
        rx, ry, rw, rh = cv2.boundingRect(coords)
        mosaic = mosaic[ry:ry + rh, rx:rx + rw]

    return True, mosaic