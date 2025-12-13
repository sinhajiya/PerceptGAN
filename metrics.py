import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import pandas as pd

# === Paths ===
folder_path = '/data/UG/Jiya/pix2pix/results/ukan_ablation_wo_iqa/test_latest/images'

# === Metrics setup ===
loss_fn_alex = lpips.LPIPS(net='alex')
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3)
])

results = []
psnr_total = ssim_total = lpips_total = cosine_total = 0
count = 0

# === Collect all real images ===
all_files = os.listdir(folder_path)
real_files = [f for f in all_files if f.endswith('_real_B.png')]

for real_name in real_files:
    base = real_name.replace('_real_B.png', '')
    fake_name = base + '_fake_B.png'

    real_path = os.path.join(folder_path, real_name)
    fake_path = os.path.join(folder_path, fake_name)

    if not os.path.exists(fake_path):
        print(f"Fake image missing for {base}")
        continue

    real_img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
    fake_img = cv2.imread(fake_path, cv2.IMREAD_GRAYSCALE)

    if real_img is None or fake_img is None:
        print(f"Skipping unreadable image: {base}")
        continue

    # Convert to 3-channel RGB for LPIPS
    real_rgb = cv2.cvtColor(real_img, cv2.COLOR_GRAY2RGB)
    fake_rgb = cv2.cvtColor(fake_img, cv2.COLOR_GRAY2RGB)

    real_tensor = transform(real_rgb).unsqueeze(0)
    fake_tensor = transform(fake_rgb).unsqueeze(0)

    psnr_val = psnr(real_img, fake_img)
    ssim_val = ssim(real_img, fake_img, data_range=real_img.max() - real_img.min())
    lpips_val = loss_fn_alex(real_tensor, fake_tensor).item()

    real_flat = real_tensor.view(1, -1)
    fake_flat = fake_tensor.view(1, -1)
    cosine_val = cos(real_flat, fake_flat).item()


    results.append({
        "Image": base,
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "LPIPS": lpips_val,
        "CosineSimilarity": cosine_val
    })

    # === Only include in average if not excluded
    if not base.startswith('country_'):
        psnr_total += psnr_val
        ssim_total += ssim_val
        lpips_total += lpips_val
        cosine_total += cosine_val
        count += 1


# === Averages ===
if count > 0:
    results.append({
        "Image": "Average",
        "PSNR": psnr_total / count,
        "SSIM": ssim_total / count,
        "LPIPS": lpips_total / count,
        "CosineSimilarity": cosine_total / count
    })

    df = pd.DataFrame(results)
    csv_path = os.path.join(folder_path, "metrics.csv")
    df.to_csv(csv_path, index=False)

    print(folder_path)
    print(f"\n--- Averages over {count} patches ---")
    print(f"PSNR:   {psnr_total / count:.2f}")
    print(f"SSIM:   {ssim_total / count:.4f}")
    print(f"LPIPS:  {lpips_total / count:.4f}")
    print(f"Cosine: {cosine_total / count:.4f}")
    print(f"Saved results to: {csv_path}")
else:
    print("No valid image pairs found.")
