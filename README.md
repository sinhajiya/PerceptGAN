## PerceptGAN: Progressive Adversarial NIR-to-RGB Translation

PerceptGAN is a conditional GAN framework for **Near-Infrared (NIR) to RGB image translation**.
It learns complex nonlinear intensity–color mappings to generate **structurally consistent, perceptually realistic RGB images** from single-channel NIR inputs.

---

## Problem

NIR-to-RGB translation is highly ambiguous:

* NIR lacks explicit color information.
* Intensity–color relationships are nonlinear and scene-dependent.
* Standard convolutional generators tend to produce over-smoothed or desaturated outputs.

This project addresses these limitations with a stronger nonlinear generator and a progressive perceptual training strategy.

---

## Architecture

### Generator — U-KAN Based Encoder–Decoder

* U-shaped encoder–decoder backbone
* Kolmogorov–Arnold Network (KAN) layers at the bottleneck
* Patch embedding + tokenized nonlinear modeling
* Skip connections with learnable fusion weights
* Channel–Spatial SE attention modules

This design improves nonlinear approximation and preserves structure while enhancing color expressiveness.

### Discriminator — Multi-Scale PatchGAN

* Three discriminators operating at different resolutions
* Enforces:

  * Global layout consistency
  * Mid-scale structure
  * Fine texture realism
* Uses Least Squares GAN (LSGAN) objective
* Feature-matching loss stabilizes training

Together, they form a coarse-to-fine adversarial supervision framework.

---

## Training Objective (Progressive Optimization)

The generator is optimized with a staged loss:

### Early Training (Structure Focus)

* Adversarial loss (LSGAN)
* L1 reconstruction loss
* DISTS perceptual similarity loss
* Feature matching loss

### Later Training (Perceptual Refinement)

* Edge-aware loss
* IQA-based perceptual quality loss (MANIQA)
* Continues adversarial + reconstruction supervision

This shifts optimization from low-level alignment to high-level perceptual realism.

---

##  Results (Patch-Based Evaluation)

| Method           | PSNR ↑    | SSIM ↑   | LPIPS ↓  |
| ---------------- | --------- | -------- | -------- |
| Pix2Pix          | 14.53     | 0.42     | 0.37     |
| **U-KAN (Ours)** | **24.58** | **0.67** | **0.24** |

PerceptGAN significantly improves structural fidelity and perceptual quality while using fewer parameters than standard Pix2Pix.

---

## Training Details

* 200 epochs
* Batch size: 16
* Optimizer: Adam (β₁=0.5, β₂=0.999)
* Initial LR: 2e-4 (linear decay after 100 epochs)
* Hardware: NVIDIA H100 GPU
* Patch-based training: 128×128 overlapping patches

---

##  Key Features

* Strong nonlinear modeling via KAN layers
* Multi-scale adversarial supervision
* Progressive perceptual training strategy
* Patch-based efficient learning

---

##  Summary

PerceptGAN combines nonlinear function approximation, multi-scale adversarial learning, and progressive perceptual supervision to produce **color-accurate, structurally consistent, and visually realistic RGB reconstructions from NIR images**.

Designed for practical cross-spectral translation where realism and perceptual quality matter most.

---

## Acknowledgment

This implementation builds upon and adapts components from the following:

* **pytorch-CycleGAN-and-pix2pix**
  [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

* **U-KAN**
  [https://github.com/CUHK-AIM-Group/U-KAN](https://github.com/CUHK-AIM-Group/U-KAN)

