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
    Builds a voxel grid with trilinear interpolation.
    
    Args:
        events: Tensor of shape (N, 4) -> [x, y, p, t]
        num_bins: B in the formula (number of temporal bins)
        width: W in the formula
        height: H in the formula
        
    Returns:
        voxel_grid: Tensor of shape (num_bins, height, width)
    """
    device = events.device
    
    x = events[:, 0]
    y = events[:, 1]
    p = events[:, 2]
    t = events[:, 3]

    # --- Step 1: Normalize Timestamp (Equation 2 in your image) ---
    # t_star = (t - t_min) / (t_max - t_min) * (B - 1)
    t_min = t[0]
    t_max = t[-1]
    
    # Avoid division by zero if all events are at the same time
    if t_max == t_min:
        t_norm = torch.zeros_like(t)
    else:
        t_norm = (t - t_min) / (t_max - t_min) * (num_bins - 1)

    # --- Step 2: Temporal Interpolation (Equation 1 in your image) ---
    # We find the two integer temporal bins: t0 (left) and t1 (right)
    # The weight is max(0, 1 - |t_n - t*|)
    
    t0 = torch.floor(t_norm).long()
    t0 = torch.clamp(t0, 0, num_bins - 1)
    
    t1 = t0 + 1
    t1 = torch.clamp(t1, 0, num_bins - 1)

    # Calculate distance |t_n - t*| for the two bins
    # For bin t0, the distance is (t_norm - t0)
    w_t0 = 1.0 - (t_norm - t0.float())  # This is max(0, 1 - |t0 - t*|)
    w_t1 = 1.0 - w_t0                   # This is max(0, 1 - |t1 - t*|)

    # --- Step 3: Spatial Interpolation (Bilinear) ---
    # Since the text says "trilinear voting", we must also split x and y.
    
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Clamp coordinates to ensure we stay inside the image [W, H]
    x0 = torch.clamp(x0, 0, width - 1)
    x1 = torch.clamp(x1, 0, width - 1)
    y0 = torch.clamp(y0, 0, height - 1)
    y1 = torch.clamp(y1, 0, height - 1)

    # Calculate spatial weights
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    # --- Step 4: Accumulate (Voting) ---
    voxel_grid = torch.zeros((num_bins, height, width), dtype=torch.float32, device=device)

    # Helper function to add values to the grid using index_put_ (fast on GPU)
    def add_votes(t_idx, y_idx, x_idx, weights):
        # We flatten the index to treat the 3D grid as a 1D array for scattering
        flat_indices = t_idx * (height * width) + y_idx * width + x_idx
        voxel_grid.put_(flat_indices, weights * p, accumulate=True)

    # We must vote into 8 locations (2 temporal * 4 spatial)
    
    # Votes for Temporal Bin t0
    add_votes(t0, y0, x0, w_t0 * wa)
    add_votes(t0, y1, x0, w_t0 * wb)
    add_votes(t0, y0, x1, w_t0 * wc)
    add_votes(t0, y1, x1, w_t0 * wd)

    # Votes for Temporal Bin t1
    add_votes(t1, y0, x0, w_t1 * wa)
    add_votes(t1, y1, x0, w_t1 * wb)
    add_votes(t1, y0, x1, w_t1 * wc)
    add_votes(t1, y1, x1, w_t1 * wd)

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

