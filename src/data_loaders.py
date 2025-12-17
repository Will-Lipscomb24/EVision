import numpy as np
import cv2
import torch 
from torch.utils.data import Dataset
from metavision_core.event_io import EventDatReader
from metavision_core_ml.preprocessing.event_to_tensor_torch import event_cd_to_torch
from utils import events_to_voxel_grid




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
                                self.image_width, 
                                self.image_height))

        # Load data using Metavision SDK function
        events_full = event_cd_to_torch(events)
        
        # Transform events into torch tensor using Metavision SDK function
        events_tensor = event_cd_to_torch(events_full)

        # Remove batch size input (all zeros)
        events = events_tensor[:,1:5] # Event stream with x, y, p, t

        # Create necessary voxel grid formatting
        voxel_grid = events_to_voxel_grid(self.num_bins, 
                                          self.image_width, 
                                          self.image_height)
        return voxel_grid

class ImageDataset(Dataset):
    def __init__(self, file_paths, image_height, image_width):
        
        self.file_paths = file_paths
        self.image_height = image_height
        self.image_width = image_width

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
