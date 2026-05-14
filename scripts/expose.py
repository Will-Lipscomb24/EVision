import os
import cv2
import random
import numpy as np
import glob

def process_full_dataset(input_dir, output_dir, start_index=1):
    """
    Processes EVERY image in input_dir, applies random bloom effects,
    and saves them to output_dir with sequential, adaptive padding.
    """
    
    # 1. Gather all files and sort them to maintain sequence
    files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    total_files = len(files)
    
    if total_files == 0:
        print(f"Error: No images found in {input_dir}")
        return

    # 2. Setup adaptive padding based on the total number of files
    padding_width = len(str(start_index + total_files - 1))

    print(f"Starting batch process: {total_files} images found.")
    print(f"Output naming scheme: {{ID:0{padding_width}d}}.jpg")

    # --- Internal Exposure Helper ---
    def make_bloom_map(h, w, corner="center", intensity=3, spread=0.6):
        corners = {
            "top-right": (w-1, 0), "top-left": (0, 0), "bottom-right": (w-1, h-1), 
            "bottom-left": (0, h-1), "top": (w//2, 0), "bottom": (w//2, h-1),
            "left": (0, h//2), "right": (w-1, h//2), "center": (w//2, h//2)
        }
        cx, cy = corners[corner]
        
        Y, X = np.mgrid[:h, :w]
        dist_norm = np.sqrt(((X - cx) / w)**2 + ((Y - cy) / h)**2)
        dist_norm /= dist_norm.max()
        
        mask = np.exp(-(dist_norm**2) / (2 * spread**2))
        
        # Smooth noise warp
        warp = min(h, w) * 0.06
        nx = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32), (0,0), 60) * warp
        ny = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32), (0,0), 60) * warp
        map_x, map_y = np.clip(X + nx, 0, w-1).astype(np.float32), np.clip(Y + ny, 0, h-1).astype(np.float32)
        mask = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR)
        
        return 1.0 + (intensity - 1.0) * mask

    # 3. Processing Loop
    for i, full_path in enumerate(files):
        img = cv2.imread(full_path)
        print(img)
        if img is None:
            print(f"Warning: Could not read {full_path}. Skipping.")
            continue
        
        h, w = img.shape[:2]
        avg_brightness = img.mean()
        
        # Randomized parameters for variety
        direction = random.choice(["top-right", "top-left", "bottom-right", "bottom-left", "top", "bottom", "left", "right", "center"])
        locality = random.uniform(2, 7)
        
        # Adaptive Intensity (prevents over-blowing already bright images)
        if avg_brightness < 100: intensity = random.uniform(3, 4)
        elif avg_brightness < 175: intensity = random.uniform(2, 3)
        else: intensity = random.uniform(1.2, 1.8)
            
        # Generate and Apply Exposure
        emap = make_bloom_map(h, w, corner=direction, intensity=intensity, spread=random.uniform(0.4, 0.6))
        
        img_f = img.astype(np.float32) / 255.0
        res = np.clip(img_f * emap[:, :, np.newaxis], 0, 1)
        res = (res * 255).astype(np.uint8)
        
        # Add the white "Wash" glow
        enorm = (emap - emap.min()) / (emap.max() - emap.min())
        gmask = np.power(enorm, locality)[:, :, np.newaxis]
        glowed = res.astype(np.float32) + (np.ones_like(res) * 255 * gmask * 0.3)
        final_img = np.clip(glowed, 0, 255).astype(np.uint8)

        # Save with adaptive sequential naming
        current_id = start_index + i
        filename = f"{current_id:0{padding_width}d}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), final_img)

    print(f"\nSuccess! All {total_files} images processed and saved to {output_dir}")

