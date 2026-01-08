import numpy as np
import cv2
import torch 
from torch.utils.data import Dataset
from metavision_core.event_io import EventDatReader
from metavision_core_ml.preprocessing.event_to_tensor_torch import event_cd_to_torch
from src.utils import events_to_voxel_grid




class EventDataset(Dataset):
    def __init__(self, file_paths, num_bins, image_height, image_width):
        
        self.file_paths = file_paths
        self.num_bins = num_bins
        self.image_height = image_height
        self.image_width = image_width

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        try:
            reader = EventDatReader(file_path)
            raw_events = reader.load_n_events(100_000_000)
        except Exception as e:
            print(f'Error reading {file_path}: {e}')
            return torch.zeros((self.num_bins, 
                                self.image_height, 
                                self.image_width))

        # Load data using Metavision SDK function
        events_tensor = event_cd_to_torch(raw_events)

        # Remove batch size input (all zeros)
        events = events_tensor[:,1:5] # Event stream with x, y, p, t

        # Create necessary voxel grid formatting
        voxel_grid = events_to_voxel_grid(events, 
                                          self.num_bins, 
                                          self.image_width, 
                                          self.image_height)
        return voxel_grid

class ImageDataset(Dataset):
    def __init__(self, file_paths, image_height, image_width, to_grayscale=True):
        
        self.file_paths = file_paths
        self.image_height = image_height
        self.image_width = image_width
        self.to_grayscale = to_grayscale

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Will likely work primarily in gray scale
        if self.to_grayscale:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            # Kill training if image is corrupt
            if img is None:
                raise FileNotFoundError(f"CRITICAL ERROR: Could not load image at {file_path}. Training aborted.")
            img = cv2.resize(img, (self.image_width, self.image_height))
            img = img[np.newaxis, :, :]
        else:
            img = cv2.imread(file_path)
            if img is None:
                raise FileNotFoundError(f"CRITICAL ERROR: Could not load image at {file_path}. Training aborted.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_width, self.image_height))
            img = np.transpose(img, (2,0,1))

        img_tensor = (torch.from_numpy(img).float() / 255.0) * 2.0 - 1.0

        return img_tensor

            
        
class EnhancementDataset(Dataset):
    def __init__(self, event_paths, input_image_paths, target_image_paths, 
                 num_bins, height, width):
        
        # Initialize internal datasets
        self.event_ds = EventDataset(event_paths, num_bins, height, width)
        
        # Input images - Overexposed
        self.input_img_ds = ImageDataset(input_image_paths, height, width)
        
        # Target images - Normal MSCOCO Dataset
        self.target_img_ds = ImageDataset(target_image_paths, height, width)

    def __len__(self):
        # Return the number of samples
        return len(self.event_ds)

    def __getitem__(self, idx):
        # Load all components
        events = self.event_ds[idx]
        input_frame = self.input_img_ds[idx]
        target_frame = self.target_img_ds[idx]
        
        # Return dictionary (easier to manage multiple inputs)
        return {
            "input_events": events,
            "input_frame": input_frame,
            "target": target_frame
        }