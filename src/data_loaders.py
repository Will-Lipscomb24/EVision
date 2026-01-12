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
            # Warning: Loading 100M events might be slow/heavy for large files.
            # Ideally, ensure files are short clips.
            raw_events = reader.load_n_events(100_000_000)
            
            # Check if file was empty or load failed
            if len(raw_events) == 0:
                print(f"Warning: No events found in {file_path}")
                return None
                
        except Exception as e:
            # FIX: Return None instead of Zeros to avoid training on bad data
            print(f'Warning: Error reading events at {file_path}: {e}')
            return None

        # Load data using Metavision SDK function
        events_tensor = event_cd_to_torch(raw_events)

        # Remove batch size input (all zeros) if present
        # Ensure tensor shape is handled correctly
        events = events_tensor[:, 1:5] # Event stream with x, y, p, t

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

        # Handle loading errors gracefully
        if self.to_grayscale:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not load image at {file_path}")
                return None
            img = cv2.resize(img, (self.image_width, self.image_height))
            img = img[np.newaxis, :, :]
        else:
            img = cv2.imread(file_path)
            if img is None:
                print(f"Warning: Could not load image at {file_path}")
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_width, self.image_height))
            img = np.transpose(img, (2, 0, 1))

        # FIX: Normalize to [0, 1] range
        # This matches your model's Sigmoid output and avoids negative inputs to L1 Loss
        img_tensor = torch.from_numpy(img).float() / 255.0

        return img_tensor

class EnhancementDataset(Dataset):
    def __init__(self, event_paths, input_image_paths, target_image_paths, 
                 num_bins, height, width):
        
        self.event_ds = EventDataset(event_paths, num_bins, height, width)
        self.input_img_ds = ImageDataset(input_image_paths, height, width)
        self.target_img_ds = ImageDataset(target_image_paths, height, width)

    def __len__(self):
        return len(self.event_ds)

    def __getitem__(self, idx):
        # Load all components
        events = self.event_ds[idx]
        input_frame = self.input_img_ds[idx]
        target_frame = self.target_img_ds[idx]
        
        # FIX: Check if ANY component failed
        if events is None or input_frame is None or target_frame is None:
            return None
        
        return {
            "input_events": events,
            "input_frame": input_frame,
            "target": target_frame
        }

# --- IMPORTANT: Copy this function to your train.py ---
def collate_fn_skip_bad(batch):
    """
    Custom collate function to skip None samples (corrupted data).
    Usage: DataLoader(dataset, ..., collate_fn=collate_fn_skip_bad)
    """
    # Remove None values from the batch
    batch = list(filter(lambda x: x is not None, batch))
    
    if len(batch) == 0:
        # Handle the rare case where an entire batch is corrupted
        # Returning None here usually requires handling in the training loop,
        # or you can return an empty tensor/dict structure if preferred.
        return None 
        
    return torch.utils.data.dataloader.default_collate(batch)