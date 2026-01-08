import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.custom_loss import custom_loss
from tqdm import tqdm
import yaml
import glob

# Imports from your src folder
from src.data_loaders import EnhancementDataset
from src.model.assembled import EventImageFusionNet

# --- CONFIG LOADING ---
with open("configs/config.yaml", 'r') as f:
    cfg = yaml.safe_load(f)


EVENTS_DIR = cfg['data']['events_dir']
INPUT_DIR = cfg['data']['input_dir']
TARGET_DIR = cfg['data']['target_dir']

BINS = cfg['model']['num_bins']
HEIGHT = cfg['model']['height']
WIDTH = cfg['model']['width']
BASE_CHANNELS = cfg['model']['base_channels']
ENCODER_CHANNELS = cfg['model']['encoder_channels']
RFB_BLOCKS = cfg['model']['rfb_blocks']

BATCH_SIZE = cfg['training']['batch_size']
SAVE_DIR = cfg['training']['save_dir']
EPOCHS = cfg['training']['epochs']
LR = cfg['training']['learning_rate']
LPIPS_NET = cfg['training']['lpips_net']
WORKERS = cfg['training']['num_workers']
SAVES = cfg['training']['num_saves']



def train():
    # 1. SETUP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    # Create directory for saving models
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 2. DATA LOADERS
    # Assuming you have lists of file paths (you can use glob to get these automatically)
    # For now, let's assume they are passed or hardcoded

    event_files = sorted(glob.glob(os.path.join(EVENTS_DIR, "*.dat")))
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))
    target_files = sorted(glob.glob(os.path.join(TARGET_DIR, "*.jpg")))

    train_ds = EnhancementDataset(
        event_paths=event_files,
        input_image_paths=input_files,
        target_image_paths=target_files,
        num_bins=BINS,
        height=HEIGHT,
        width=WIDTH
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # 3. MODEL, LOSS, OPTIMIZER
    model = EventImageFusionNet(
        num_bins=BINS,
        base_channels=BASE_CHANNELS,
        enc_channels=ENCODER_CHANNELS,
        num_rfb_blocks=RFB_BLOCKS
    ).to(device)

    # L1 Loss is standard for image reconstruction (creates sharper edges than MSE)
    criterion = custom_loss(net=LPIPS_NET, l1_weight=1.0, lpips_weight=1.0, device=device).to(device)
    criterion.lpips_fn.eval()

    optimizer = optim.Adam(model.parameters(), lr=float(LR))

    # 4. TRAINING LOOP
    num_epochs = EPOCHS
    
    for epoch in range(num_epochs):
        model.train()  # Crucial: enables Dropout and BatchNorm updates
        running_loss = 0.0
        
        # Progress bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in loop:
            # A. Get Data
            # Note: dictionary keys must match what your Dataset returns
            events = batch['input_events'].to(device)
            input_frame = batch['input_frame'].to(device)
            target = batch['target'].to(device)

            # --- ADD IT HERE ---
            with torch.autograd.set_detect_anomaly(True):
                # B. Forward Pass
                optimizer.zero_grad()
                output = model(events, input_frame)

                # C. Compute Loss
                loss = criterion(output + 1e-8, target)

                # D. Backward Pass
                # The error usually triggers exactly at this line
                loss.backward()

            # # B. Forward Pass
            # # Zero gradients from previous step
            # optimizer.zero_grad()
            
            # # Run model
            # output = model(events, input_frame)

            # # C. Compute Loss
            # loss = criterion(output, target)

            # # D. Backward Pass (Backprop)
            # loss.backward()
            
            # E. Update Weights
            optimizer.step()

            # Update progress bar
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 5. SAVE CHECKPOINT
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {epoch_loss:.6f}")

        if (epoch + 1) % EPOCHS/SAVES == 0:
            save_path = os.path.join(
                SAVE_DIR, 
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train()