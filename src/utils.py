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
from pathlib import Path


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

def convert_images_to_grayscale(input_folder_path, output_path):
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Grab all .jpg files (you can add .png or others if needed)
    files = glob.glob(os.path.join(input_folder_path, "*.jpg")) 
    
    print(f"Found {len(files)} images to convert to grayscale.")

    for file in files:
        img = cv2.imread(file)
        
        if img is None: 
            print(f"Skipping: {file} (could not read)")
            continue
            
        # Convert BGR image to Gray
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get original filename and save
        filename = os.path.basename(file)
        save_path = os.path.join(output_path, filename)
        
        cv2.imwrite(save_path, gray_img)
        
    print("Processing complete.")

def random_select_and_rename(input_folder, output_folder, target_count=500, start_index=1):
    os.makedirs(output_folder, exist_ok=True)
    
    files = glob.glob(os.path.join(input_folder, "*.jpg"))
    total_images = len(files)
    
    if total_images < target_count:
        target_count = total_images

    selected_files = random.sample(files, target_count)

    # --- ADAPTABLE PADDING LOGIC ---
    # Calculate how many digits the largest number will have
    # e.g., 500 -> 3 digits, 10000 -> 5 digits
    padding_width = len(str(start_index + target_count - 1))
    # -------------------------------

    print(f"Selecting {target_count} images with {padding_width}-digit padding...")

    for i, file_path in enumerate(selected_files):
        img = cv2.imread(file_path)
        if img is None:
            continue
            
        # The syntax f"{value:0{width}d}" uses nested variables
        # It says: "format 'value' with '0' padding of 'width' length"
        current_id = start_index + i
        filename = f"{current_id:0{padding_width}d}.jpg"
        
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, img)

    print(f"Done! Filenames range from {start_index:0{padding_width}d} to {start_index + target_count - 1:0{padding_width}d}")


def convert_npy_to_dat(input_path, output_path, width=1280, height=720):
    """
    Converts structured numpy events (x, y, p, t) into a Metavision .dat file.
    Fixes KeyError 65/251 by adding the correct Type 0 header and bit-packing.
    """
    # 1. Load your numpy data
    events = np.load(input_path)
    
    if len(events) == 0:
        print(f"Skipping empty file: {input_path}")
        return

    print(f"Processing {len(events):,} events: {os.path.basename(input_path)}")

    with open(output_path, 'wb') as f:
        # 2. Write the mandatory Metavision Header
        # 'Type 0' tells the reader: "This is standard CD data (x, y, p, t)"
        header = (
            f"% Data file containing CD events\n"
            f"% Version 2.0\n"
            f"% width {width}\n"
            f"% height {height}\n"
            f"% Type 0\n"
        ).encode('ascii')
        f.write(header)

        # 3. Perform Bit-Packing
        # Metavision Type 0 expects 8 bytes per event:
        # [4 bytes: uint32 timestamp] [4 bytes: packed x, y, p]
        
        # Create a specialized structured array for the output
        dat_dtype = np.dtype([('t', '<u4'), ('xyp', '<u4')])
        dat_events = np.empty(len(events), dtype=dat_dtype)

        # Timestamps (t) - Cast to 32-bit unsigned
        dat_events['t'] = events['t'].astype(np.uint32)

        # Pack x, y, p into the second 32-bit word
        # x: bits 0-13 (14 bits)
        # y: bits 14-26 (13 bits)
        # p: bit 27 (1 bit)
        x = events['x'].astype(np.uint32) & 0x3FFF
        y = (events['y'].astype(np.uint32) & 0x1FFF) << 14
        p = (events['p'].astype(np.uint32) & 0x01) << 27
        
        dat_events['xyp'] = x | y | p

        # 4. Save raw bytes
        dat_events.tofile(f)
    
    print(f"Successfully created: {output_path}")

def find_evision_root():
    # Get the absolute path of the current file
    current_path = Path(__file__).resolve()
    
    # Iterate through all parent directories
    for parent in current_path.parents:
        # Check for a unique marker of the EVision root
        if (parent / ".git").exists():
            return parent
            
    # Fallback to current directory if not found
    return current_path.parent


