import os
import glob
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

from src.custom_loss import custom_loss
from src.data_loaders import EnhancementDataset, collate_fn_skip_bad
from src.model.assembled import EventImageFusionNet
from src.utils import find_evision_root


# -------------------------
# IMPORTANT: CUDA memory fix
# -------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# -------------------------
# CONFIG
# -------------------------
REPO_ROOT = find_evision_root()
configs_path = REPO_ROOT / 'configs' / 'config.yaml'

with open(configs_path, 'r') as f:
    cfg = yaml.safe_load(f)

TYPE = cfg['training']['type']

EVENTS_DIR = REPO_ROOT / 'data' / f'{TYPE}' / 'events'
INPUT_DIR  = REPO_ROOT / 'data' / f'{TYPE}' / 'input'
TARGET_DIR = REPO_ROOT / 'data' / f'{TYPE}' / 'target'
SAVE_DIR   = REPO_ROOT / cfg['training']['save_dir']

BINS = cfg['model']['num_bins']
HEIGHT = cfg['model']['height']
WIDTH = cfg['model']['width']

BASE_CHANNELS = cfg['model']['base_channels']
ENCODER_CHANNELS = cfg['model']['encoder_channels']
RFB_BLOCKS = cfg['model']['rfb_blocks']

BATCH_SIZE = cfg['training']['batch_size']
EPOCHS = cfg['training']['epochs']
LR = cfg['training']['learning_rate']
LPIPS_NET = cfg['training']['lpips_net']
WORKERS = cfg['training']['num_workers']
SAVES = cfg['training']['num_saves']
LR_REDUCTION = cfg['training']['learning_rate_reduction']
LR_STEP = cfg['training']['learning_epochs']


# -------------------------
# TRAIN
# -------------------------
def train():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # -------------------------
    # DATA
    # -------------------------
    event_files = sorted(glob.glob(str(EVENTS_DIR / "*.dat")))
    input_files = sorted(glob.glob(str(INPUT_DIR / "*.jpg")))
    target_files = sorted(glob.glob(str(TARGET_DIR / "*.jpg")))

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
        num_workers=WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn_skip_bad
    )

    # -------------------------
    # MODEL
    # -------------------------
    model = EventImageFusionNet(
        num_bins=BINS,
        base_channels=BASE_CHANNELS,
        enc_channels=ENCODER_CHANNELS,
        num_rfb_blocks=RFB_BLOCKS
    ).to(device)

    # -------------------------
    # LOSS
    # -------------------------
    criterion = custom_loss(
        net=LPIPS_NET,
        l1_weight=1.0,
        lpips_weight=1.0,
        device=device
    ).to(device)

    criterion.lpips_fn.eval()

    # -------------------------
    # OPTIMIZER / SCHEDULER
    # -------------------------
    optimizer = optim.Adam(model.parameters(), lr=float(LR))
    scheduler = StepLR(optimizer, step_size=LR_STEP, gamma=LR_REDUCTION)

    # -------------------------
    # AMP (IMPORTANT)
    # -------------------------
    scaler = GradScaler()

    # -------------------------
    # TRAIN LOOP
    # -------------------------
    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:

            if batch is None:
                torch.cuda.empty_cache()
                continue

            events = batch['input_events'].to(device, non_blocking=True)
            input_frame = batch['input_frame'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)

            # -------------------------
            # FORWARD + LOSS (AMP)
            # -------------------------
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                output = model(events, input_frame)
                loss = criterion(output, target)

            # -------------------------
            # BACKWARD (SCALED)
            # -------------------------
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # -------------------------
            # LOGGING
            # -------------------------
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {epoch_loss:.6f}")

        # -------------------------
        # CHECKPOINTING
        # -------------------------
        save_int = max(1, EPOCHS // SAVES)

        if (epoch + 1) % save_int == 0:
            save_path = SAVE_DIR / f"checkpoint_epoch_{epoch+1}.pth"

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, save_path)

            print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    train()