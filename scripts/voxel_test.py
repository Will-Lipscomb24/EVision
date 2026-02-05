import cv2
import torch
import numpy as np
from metavision_core.event_io import EventDatReader
from metavision_core_ml.preprocessing.event_to_tensor_torch import event_cd_to_torch

# Import your utils
from src.utils import events_to_voxel_grid, vis_voxel_bin

# 1. Setup Paths
single_event_path = '/home/will/Downloads/00001.dat'
single_image_path = '/home/will/projects/EVision/data/data_formatted/target/0001.jpg'

# 2. Get Dimensions
img = cv2.imread(single_image_path)
if img is None: raise FileNotFoundError(single_image_path)
height, width = img.shape[:2] # Safe for color or gray

# 3. Load Events directly (No Dataset Class)
reader = EventDatReader(single_event_path)
raw_events = reader.load_n_events(100_000_000) # Load all events

# 4. Convert to Tensor
# event_cd_to_torch returns [Batch, x, y, p, t]
t_events = event_cd_to_torch(raw_events) 
events = t_events[:, 1:5] # Strip batch index, keep (x, y, p, t)

# 5. Create Voxel Grid
# events_to_voxel_grid(events, num_bins, width, height)
voxel_grid = events_to_voxel_grid(events, 5, 512, 512)

# 6. Visualize
# vis_voxel_bin usually expects a tensor, make sure to show it using cv2.imshow
img_preview = vis_voxel_bin(voxel_grid, 2) # Get bin 0
cv2.imshow("Voxel Bin 0", img_preview)
cv2.waitKey(0)
cv2.destroyAllWindows()