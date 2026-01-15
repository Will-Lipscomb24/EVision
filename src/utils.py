import numpy as np
import torch
import torch.nn.functional as F
from metavision_core.event_io import EventDatReader
from metavision_core_ml.preprocessing.event_to_tensor_torch import event_cd_to_torch
import numpy as np
import cv2
import os 
import glob
import random


def events_to_voxel_grid(events, num_bins, width, height):
    """
    Builds a voxel grid using trilinear interpolation (voting).
    Each event distributes its polarity across 8 neighboring voxels 
    (2 in time, 4 in space).
    """
    device = events.device
    x = events[:, 0]
    y = events[:, 1]
    p = events[:, 2].float()
    t = events[:, 3]

    # --- 1. Temporal Normalization ---
    t_min, t_max = t[0], t[-1]
    dt = t_max - t_min
    
    # Map t to the range [0, num_bins - 1]
    t_norm = (t - t_min) * (num_bins - 1) / (dt + 1e-9) if dt > 0 else torch.zeros_like(t)

    # --- 2. Define 8-way Neighbor Indices ---
    # Temporal neighbors
    bin1 = torch.floor(t_norm).long().clamp(0, num_bins - 1)
    bin2 = (bin1 + 1).clamp(0, num_bins - 1)

    # --- 3. Calculate Trilinear Weights ---
    # Fractional distances
    dt_frac = t_norm - bin1.float()

    # Temporal weights
    w_t0 = 1.0 - dt_frac
    w_t1 = dt_frac

  # --- 3. Accumulate into Grid ---
    voxel_grid = torch.zeros((num_bins, height, width), dtype=torch.float32, device=device)

    # Accumulate for bin t0
    # index_put_ handles multiple events hitting the same pixel (atomic add)
    voxel_grid.index_put_((bin1, y, x), w_t0 * p, accumulate=True)
    
    # Accumulate for bin t1
    voxel_grid.index_put_((bin2, y, x), w_t1 * p, accumulate=True)  

    return voxel_grid

def vis_voxel_bin(voxel_grid, bin_index=0):
    # Convert from Torch to Numpy if necessary
    if torch.is_tensor(voxel_grid):
        # Shape is (5, H, W) -> we want one slice
        img = voxel_grid[bin_index].detach().cpu().numpy()
    else:
        img = voxel_grid[bin_index]

    # Normalize to 0-255
    # We use (img - min) / (max - min) to scale the intensities
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img_norm = (img - img_min) / (img_max - img_min) * 255
    else:
        img_norm = np.zeros_like(img)

    img_uint8 = img_norm.astype(np.uint8)

    cv2.imshow(f'Bin {bin_index}', img_uint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def over_expose(input_folder, output_folder, exposure_range=(2, 3), gamma_range=(1, 2)):
    """
    Randomly adjusts exposure and gamma, then saves the result.
    """
    files = glob.glob(os.path.join(input_folder, "*.jpg")) 

    for file in files:
        # 1. Randomly select parameters
        exposure_factor = random.uniform(*exposure_range)
        gamma = random.uniform(*gamma_range)
        image = cv2.imread(file)
        
        # 2. Convert to float 0-1
        img_float = image.astype(np.float32) / 255.0

        # 3. Apply exposure (multiplicative)
        img_exposed = np.clip(img_float * exposure_factor, 0, 1)

        # 4. Apply Gamma
        if gamma == 0:
            img_gamma = img_exposed
        else:
            img_gamma = np.power(img_exposed, 1.0 / gamma)

        # 5. Convert back to uint8
        final_image = (img_gamma * 255).astype(np.uint8)
        
        # 6. Save to specified folder
        os.makedirs(output_folder, exist_ok=True)
        
        filename = os.path.basename(file)
        # THIS LINE ENSURES THE NAME IS THE SAME
        save_path = os.path.join(output_folder, filename)
        
        cv2.imwrite(save_path, final_image)

def resize_images(input_folder_path, output_path, new_height, new_width):
    os.makedirs(output_path, exist_ok=True)
    
    files = glob.glob(os.path.join(input_folder_path, "*.jpg")) 
    
    print(f"Found {len(files)} images to resize.")

    for file in files:
        img = cv2.imread(file)
        if img is None: 
            continue
            
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        filename = os.path.basename(file)
        save_path = os.path.join(output_path, filename)
        
        cv2.imwrite(save_path, resized_img)

