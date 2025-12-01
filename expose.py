import os
import cv2
import random
import numpy as np

img_path = ""
uo_dir = "data/under_exposed"
oo_dir = "data/over_exposed"

def adjust_exposure_and_gamma(image, exposure_factor=1.0, gamma=1.0):
    """
    Adjust image brightness/exposure and then apply gamma correction.
    exposure_factor > 1 -> brighter, < 1 -> darker
    gamma < 1 -> brighter, gamma > 1 -> darker
    """
    # Convert to float 0-1
    img_float = image.astype(np.float32) / 255.0

    # Apply exposure (multiplicative)
    img_exposed = np.clip(img_float * exposure_factor, 0, 1)

    # Avoid division by zero
    if gamma == 0:
        img_gamma = img_exposed
    else:
        img_gamma = np.power(img_exposed, 1.0 / gamma)

    return (img_gamma * 255).astype(np.uint8)

if __name__ == "__main__":

    # Create directories
    os.makedirs(uo_dir, exist_ok=True)
    os.makedirs(oo_dir, exist_ok=True)

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

        # Random values
        uexp = random.uniform(0.1, 0.5)
        ugam = 1

        oexp = random.uniform(2, 7)
        ogam = random.uniform(2, 4)

        # Transform
        u_image = adjust_exposure_and_gamma(img, exposure_factor=uexp, gamma=ugam)
        o_image = adjust_exposure_and_gamma(img, exposure_factor=oexp, gamma=ogam)

        # Save with new meaningful names
        u_filename = f"{name}_u_exp{uexp:.2f}.jpg"
        o_filename = f"{name}_o_exp{oexp:.2f}_gam{ogam:.2f}.jpg"

        cv2.imwrite(os.path.join(uo_dir, u_filename), u_image)
        cv2.imwrite(os.path.join(oo_dir, o_filename), o_image)
