import os
import cv2
import random
import numpy as np
from scipy.ndimage import gaussian_filter

img_path = "/home/will/projects/EVision/data/data_formatted2/target"
output_dir = "/home/will/projects/EVision/data/data_formatted2/input"


def make_directional_bloom(h, w, source_corner="top-right", intensity=2, spread=0.1, seed=None):
    """
    Simulates a bright light source bleeding in from a corner/edge,
    like a real overexposed camera shot.
    
    source_corner: "top-right", "top-left", "bottom-right", "bottom-left"
    intensity:     peak exposure multiplier (3-6 for heavy bloom like your example)
    spread:        0-1, how far the bloom reaches across the image (0.5-0.7 recommended)
    """
    if seed is not None:
        np.random.seed(seed)

    # Corner anchor points
    corners = {
        "top-right":    (w - 1, 0),
        "top-left":     (0, 0),
        "bottom-right": (w - 1, h - 1),
        "bottom-left":  (0, h - 1),
        "top":          (w // 2, 0),
        "bottom":       (w // 2, h - 1),
        "left":         (0, h // 2),
        "right":        (w - 1, h // 2),
        # Center
        "center":       (w // 2, h // 2),
    }
    cx, cy = corners[source_corner]

    Y, X = np.mgrid[:h, :w]

    # Distance from source corner, normalised to 0-1
    dist = np.sqrt(((X - cx) / w) ** 2 + ((Y - cy) / h) ** 2)
    dist_norm = dist / dist.max()

    # Sigma controls spread — larger = bloom reaches further into frame
    sigma = spread
    bloom = np.exp(-(dist_norm ** 2) / (2 * sigma ** 2))

    # --- Warp with smooth noise for organic, irregular edges ---
    warp_strength = min(h, w) * 0.06
    noise_x = cv2.GaussianBlur(
        np.random.randn(h, w).astype(np.float32), (0, 0), sigmaX=60
    ) * warp_strength
    noise_y = cv2.GaussianBlur(
        np.random.randn(h, w).astype(np.float32), (0, 0), sigmaX=60
    ) * warp_strength

    map_x = np.clip(X + noise_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(Y + noise_y, 0, h - 1).astype(np.float32)
    bloom = cv2.remap(bloom, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # Build final exposure map: 1.0 (unchanged) far from source, `intensity` at source
    exposure_map = 1.0 + (intensity - 1.0) * bloom
    return exposure_map.astype(np.float32)

def make_directional_shadow(h, w, source_corner="bottom-left", intensity=0.15, spread=0.65, seed=None):
    """
    Same as make_directional_bloom but intensity < 1 = darkening.
    intensity: 0.0 = fully black, 0.1-0.3 = heavy underexposure
    """
    if seed is not None:
        np.random.seed(seed)

    corners = {
        "top-right":    (w - 1, 0),
        "top-left":     (0, 0),
        "bottom-right": (w - 1, h - 1),
        "bottom-left":  (0, h - 1),
        "top":          (w // 2, 0),
        "bottom":       (w // 2, h - 1),
        "left":         (0, h // 2),
        "right":        (w - 1, h // 2),
        # Center
        "center":       (w // 2, h // 2),
    }
    cx, cy = corners[source_corner]

    Y, X = np.mgrid[:h, :w]
    dist = np.sqrt(((X - cx) / w) ** 2 + ((Y - cy) / h) ** 2)
    dist_norm = dist / dist.max()

    sigma = spread
    shadow = np.exp(-(dist_norm ** 2) / (2 * sigma ** 2))

    # Noise warp for organic edges
    warp_strength = min(h, w) * 0.06
    noise_x = cv2.GaussianBlur(
        np.random.randn(h, w).astype(np.float32), (0, 0), sigmaX=60
    ) * warp_strength
    noise_y = cv2.GaussianBlur(
        np.random.randn(h, w).astype(np.float32), (0, 0), sigmaX=60
    ) * warp_strength

    map_x = np.clip(X + noise_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(Y + noise_y, 0, h - 1).astype(np.float32)
    shadow = cv2.remap(shadow, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # 1.0 far from source, `intensity` at source (intensity < 1 = darker)
    exposure_map = 1.0 - (1.0 - intensity) * shadow
    return exposure_map.astype(np.float32)


def apply_bloom_exposure(image, exposure_map):
    """Apply a spatially varying exposure map to an image."""
    img_float = image.astype(np.float32) / 255.0
    exp_map = exposure_map[:, :, np.newaxis]
    img_exposed = np.clip(img_float * exp_map, 0, 1)
    return (img_exposed * 255).astype(np.uint8)


def add_bloom_glow(image, bloom_map, glow_strength=0.25, locality=6.0):
    """
    locality: higher = glow stays closer to the bright source (try 4-10)
    glow_strength: overall intensity of the white wash (try 0.15-0.35)
    """
    bloom_norm = (bloom_map - bloom_map.min()) / (bloom_map.max() - bloom_map.min())
    
    # Higher power = glow drops off faster away from the peak
    glow_mask = np.power(bloom_norm, locality)[:, :, np.newaxis]

    white = np.ones_like(image, dtype=np.float32) * 255
    img_float = image.astype(np.float32)
    glowed = img_float + white * glow_mask * glow_strength
    return np.clip(glowed, 0, 255).astype(np.uint8)

def add_shadow_glow(image, shadow_map, glow_strength=0.25, locality=6.0):
    """
    Adds a dark/black wash at the shadow source instead of white.
    Mirrors add_bloom_glow but blends toward black.
    """
    shadow_norm = (shadow_map - shadow_map.min()) / (shadow_map.max() - shadow_map.min())
    
    # Invert so mask peaks where shadow is strongest
    glow_mask = np.power(1.0 - shadow_norm, locality)[:, :, np.newaxis]

    black = np.zeros_like(image, dtype=np.float32)
    img_float = image.astype(np.float32)
    darkened = img_float * (1.0 - glow_mask * glow_strength)
    return np.clip(darkened, 0, 255).astype(np.uint8)

# ── Usage ──────────────────────────────────────────────────────────────────────
img_path = '/home/will/projects/EVision/data/data_formatted/target/0230.jpg'
image = cv2.imread(img_path)
h, w = image.shape[:2]

bloom_map = make_directional_bloom(h, w, source_corner="center", intensity=3, spread=0.6, seed=42)
result = apply_bloom_exposure(image, bloom_map)
resultb = add_bloom_glow(result, bloom_map, glow_strength=0.3,locality=2)

# Underexposure from bottom-left
shadow_map = make_directional_shadow(h, w, source_corner="right", intensity=0.15, spread=0.6, seed=42)
result = apply_bloom_exposure(image, shadow_map)   # reuse the same apply function
results = add_shadow_glow(result, shadow_map, glow_strength=.3, locality=2)


cv2.imshow('Directional Bloom', resultb)
cv2.imshow('Directional Shadow', results)
cv2.waitKey(0)
cv2.destroyAllWindows()

source_direction = ["top-right", "top-left", "bottom-right", "bottom-left", "top", "bottom", "left", "right", "center"]
locality = random.uniform(2, 7)
intensity_bloom = random.uniform(3, 5)

intensity_shadow = random.uniform(0.1, 0.15)
spread_bloom = random.uniform(0.4, 0.6)
spread_shadow = random.uniform(0.6, 0.8)

for file in os.listdir(img_path):

    full_path = os.path.join(img_path, file)

    # Skip non-files (folders, tmp files, etc.)
    if not os.path.isfile(full_path):
        continue

    name, ext = os.path.splitext(file)

    img = cv2.imread(full_path)
    if img is None:
        print(f"Skipping {file} (not an image)")
        continue

    # Setting random parameters for image
    source_direction = random.choice(["top-right", "top-left", "bottom-right", "bottom-left", "top", "bottom", "left", "right", "center"])
    function_choice = random.choice(["bloom", "shadow"])
    locality = random.uniform(2, 7)
    intensity_bloom = random.uniform(3, 5)
    intensity_shadow = random.uniform(0.1, 0.15)
    spread_bloom = random.uniform(0.4, 0.6)
    spread_shadow = random.uniform(0.6, 0.8)


    if function_choice == "bloom":
        bloom_map = make_directional_bloom(h, w, source_corner=source_direction, intensity=intensity_bloom, spread=spread_bloom, seed=42)
        result = apply_bloom_exposure(image, bloom_map)
        result_bloom = add_bloom_glow(result, bloom_map, glow_strength=0.3,locality=locality)
    elif function_choice == "shadow":   
        shadow_map = make_directional_shadow(h, w, source_corner="right", intensity=intensity_shadow, spread=spread_shadow, seed=42)
        result = apply_bloom_exposure(image, shadow_map)   # reuse the same apply function
        result_shadow = add_shadow_glow(result, shadow_map, glow_strength=.3, locality=locality)

    # Save with new meaningful names
    shadow_filename = f"{name}_bloom_{source_direction}_locality{locality:.2f}.jpg"
    bloom_filename = f"{name}_shadow_intensity{intensity_shadow:.2f}_spread{spread_shadow:.2f}.jpg"

    cv2.imwrite(os.path.join(output_dir, shadow_filename), result_bloom)
    cv2.imwrite(os.path.join(output_dir, bloom_filename), result_shadow)
   