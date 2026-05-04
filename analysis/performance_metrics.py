import os
import cv2
import torch
import numpy as np
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

# Image Paths
target_path = '/home/will/projects/EVision/data/validation_data/target'
inference_path = '/home/will/projects/EVision/data/validation_data/results'

# LPIPS Initialization
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lpips_metric = LearnedPerceptualImagePatchSimilarity(
    net_type='vgg',
    normalize=True
).to(device)
lpips_metric.eval()

# SSIM Initialization
ssim_metric = StructuralSimilarityIndexMeasure(
    data_range=1.0  # because images are normalized to [0,1]
).to(device)

############## PSNR ##############s
def compute_psnr(prediction, target):
    mse = torch.mean((prediction - target) ** 2)

    if mse == 0:
        return float('inf')

    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()

def compute_ssim(prediction, target):
    """
    prediction, target: tensors of shape (1,1,H,W)
    values in [0,1]
    """
    ssim_val = ssim_metric(prediction, target)
    return ssim_val.item()
# -------------------------------------------------
# Helper: Load grayscale image as (1,1,H,W)
# -------------------------------------------------
def load_grayscale_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    img = img.astype(np.float32) / 255.0  # normalize to [0,1]

    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return tensor.to(device)

# -------------------------------------------------
# Iterate + Compute LPIPS
# -------------------------------------------------
scores = []

target_images = sorted(os.listdir(target_path))[:11]

with torch.no_grad():
    for filename in target_images:

        target_file = os.path.join(target_path, filename)
        pred_file = os.path.join(inference_path, filename)

        if not os.path.exists(pred_file):
            print(f"Skipping {filename} (no prediction found)")
            continue

        target = load_grayscale_tensor(target_file)
        prediction = load_grayscale_tensor(pred_file)

        # Expand grayscale to 3 channels (no extra memory allocation)
        pred_3c   = prediction.expand(-1, 3, -1, -1)
        target_3c = target.expand(-1, 3, -1, -1)

        lpips_val = lpips_metric(pred_3c, target_3c)
        psnr_val = compute_psnr(prediction, target)
        ssim_val = compute_ssim(prediction, target)

        # Store in dictionary
        scores.append({
            "filename": filename,
            "LPIPS": lpips_val.item(),
            "PSNR": psnr_val,
            "SSIM": ssim_val
        })

        print(f"{filename}: LPIPS={lpips_val.item():.6f}, PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}")

# -------------------------------------------------
# Final Mean
# -------------------------------------------------
if scores:
    mean_lpips = sum(r["LPIPS"] for r in scores) / len(scores)
    mean_psnr  = sum(r["PSNR"]  for r in scores) / len(scores)
    mean_ssim  = sum(r["SSIM"]  for r in scores) / len(scores)

    print("\n---------------------------------")
    print(f"Dataset Mean Metrics: LPIPS={mean_lpips:.6f}, PSNR={mean_psnr:.2f}, SSIM={mean_ssim:.4f}")
    print("---------------------------------")
else:
    print("No valid image pairs found.")