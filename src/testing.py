import os
import torch
import glob
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from src.utils import find_evision_root

# Imports from your src folder
from src.data_loaders import EnhancementDataset, collate_fn_skip_bad
from src.model.assembled import EventImageFusionNet
from train import WIDTH

REPO_ROOT = find_evision_root()
configs_path = REPO_ROOT / 'configs' / 'config.yaml'
with open(configs_path, 'r') as f:
    cfg = yaml.safe_load(f)

type = cfg['training']['type']
TESTING_DIR = str(REPO_ROOT / 'data' / f'{type}' / 'results')
EVENTS_DIR = REPO_ROOT / 'data' / f'{type}' / 'events'
INPUT_DIR = REPO_ROOT / 'data' / f'{type}' / 'input'
TARGET_DIR = REPO_ROOT / 'data' / f'{type}' / 'target'

MODEL_NAME = 'latest_checkpoint_epoch_75_2_25.pth' # Change this to your actual checkpoint name

def test():
    # 1. SETUP & CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initial device: {device}")
    
    os.makedirs(TESTING_DIR, exist_ok=True)

    # 2. MODEL INITIALIZATION
    model = EventImageFusionNet(
        num_bins=cfg['model']['num_bins'],
        base_channels=cfg['model']['base_channels'],
        enc_channels=cfg['model']['encoder_channels'],
        num_rfb_blocks=cfg['model']['rfb_blocks']
    )

    # Match the path from your train script
    checkpoint_path = REPO_ROOT / 'saved_model_output' / MODEL_NAME
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        # Always load to CPU first to prevent VRAM spikes
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # KEY FIX: Your train script uses 'model_state_dict', not 'state_dict'
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Successfully loaded 'model_state_dict'")
        else:
            # Fallback in case you have other versions of saved models
            model.load_state_dict(checkpoint, strict=False)
            print("Loaded checkpoint as direct state_dict (fallback)")
    else:
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        return

    model.to(device)
    model.eval()

    # 3. DATA LOADER
    event_files = sorted(glob.glob(os.path.join(EVENTS_DIR, "*.dat")))
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))
    target_files = sorted(glob.glob(os.path.join(TARGET_DIR, "*.jpg")))

    test_ds = EnhancementDataset(
        event_paths=event_files,
        input_image_paths=input_files,
        target_image_paths=target_files,
        num_bins=cfg['model']['num_bins'],
        height=cfg['model']['height'],
        width=cfg['model']['width']
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1, 
        shuffle=False,
        num_workers=0, # Keep at 0 for testing to save memory
        collate_fn=collate_fn_skip_bad
    )

    # 4. INFERENCE LOOP
    print(f"Starting inference...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            if batch is None:
                continue

            # Move data to device
            events = batch['input_events'].to(device)
            input_frame = batch['input_frame'].to(device)

            # Forward Pass
            try:
                output = model(events, input_frame)
                # Save result (convert to CPU and standard float for saving)
                save_image(output.cpu(), os.path.join(TESTING_DIR, f"result_{i:04d}.png"))
            except torch.cuda.OutOfMemoryError:
                print(f"Skipping frame {i} due to OOM. Consider reducing image size in config.")
                torch.cuda.empty_cache()
                continue

    print(f"Testing complete. Results saved to {TESTING_DIR}")

if __name__ == "__main__":
    test()