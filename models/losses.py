import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
from torch.optim import lr_scheduler
import cv2
import numpy as np
import kornia.filters as KF
import kornia as K
from piq import FSIMLoss, LPIPS
from DISTS_pytorch import DISTS
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms.functional import rgb_to_grayscale
from pytorch_msssim import SSIM, MS_SSIM


class GANLoss(nn.Module):
    """Define GAN objectives compatible with multi-scale and intermediate features."""

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """
        Initialize the GANLoss class.

        Args:
            gan_mode (str): 'vanilla', 'lsgan', or 'wgangp'.
            target_real_label (float): Label value for real images.
            target_fake_label (float): Label value for fake images.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label).float())
        self.register_buffer('fake_label', torch.tensor(target_fake_label).float())
        self.gan_mode = gan_mode

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError(f"GAN mode '{gan_mode}' is not implemented")

    def get_target_tensor(self, prediction, target_is_real):
        """
        Create target tensor of same shape as prediction.
        Args:
            prediction (Tensor): Output from discriminator.
            target_is_real (bool): True for real images, False for fake.
        Returns:
            Tensor: Target tensor with real or fake label.
        """
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, predictions, target_is_real):
        """
        Compute the GAN loss.

        Args:
            predictions (Tensor or list): Discriminator output(s).
            target_is_real (bool): True for real images, False for fake.

        Returns:
            Tensor: Computed GAN loss.
        """
        if isinstance(predictions, list):
            loss = 0
            for pred in predictions:
                # Handle intermediate features (list of layers)
                if isinstance(pred, list):
                    pred = pred[-1]  # Use final output layer only
                if self.gan_mode in ['lsgan', 'vanilla']:
                    target_tensor = self.get_target_tensor(pred, target_is_real)
                    loss += self.loss(pred, target_tensor)
                elif self.gan_mode == 'wgangp':
                    loss += -pred.mean() if target_is_real else pred.mean()
            return loss / len(predictions)
        else:
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(predictions, target_is_real)
                return self.loss(predictions, target_tensor)
            elif self.gan_mode == 'wgangp':
                return -predictions.mean() if target_is_real else predictions.mean()


    def forward(self, nir_img: torch.Tensor, generated_rgb: torch.Tensor) -> torch.Tensor:
        desc_loss = 0.0
        num_valid_images = 0
        B = nir_img.shape[0]

        for i in range(B):
            if self.keypointdetector == 'superpoint':
                nir_tensor = nir_img[i].squeeze().unsqueeze(0).unsqueeze(0).float().to(self.device)
                with torch.no_grad():
                    out_nir = self.detector({'image': nir_tensor})
                kps_nir = out_nir['keypoints'][0]
                if len(kps_nir) == 0:
                    continue

                _, desc_nir = self.descriptor.compute(nir_tensor[0, 0], kps_nir)
                if desc_nir is None or desc_nir.shape[0] == 0:
                    continue

                if self.compare_with_grayscale:
                    rgb_tensor = generated_rgb[i].unsqueeze(0).to(self.device)  # [1, 3, H, W]
                    gray_tensor = rgb_to_grayscale(rgb_tensor[0]).unsqueeze(0).to(self.device)  # [1, 1, H, W]
                    _, desc_rgb = self.descriptor.compute(gray_tensor[0, 0], kps_nir)
                    if desc_rgb is None or desc_rgb.shape[0] == 0:
                        continue

                    desc_nir, desc_rgb = self.truncate_descs(desc_nir, desc_rgb)
                    desc_nir = desc_nir.to(self.device)
                    desc_rgb = desc_rgb.to(self.device)
                    dists = torch.cdist(desc_nir, desc_rgb)
                    top2 = torch.topk(dists, k=2, dim=1, largest=False)
                    mask = top2.values[:, 0] < self.nndr_ratio * top2.values[:, 1]
                    if mask.sum() == 0:
                        continue

                    idx_q = torch.arange(desc_nir.shape[0], device=self.device)[mask]
                    idx_t = top2.indices[mask, 0]
                    loss = F.pairwise_distance(desc_nir[idx_q], desc_rgb[idx_t]).mean()
                    desc_loss += loss
                    num_valid_images += 1
                    
                else:
                    rgb_tensor = generated_rgb[i].to(self.device)  # [3, H, W]
                    channel_losses = []
                    valid_channels = 0
                    for c in range(3):
                        _, desc_rgb = self.descriptor.compute(rgb_tensor[c], kps_nir)
                        if desc_rgb is None or desc_rgb.shape[0] == 0:
                            continue

                        desc_nir_c, desc_rgb_c = self.truncate_descs(desc_nir, desc_rgb)
                        desc_nir_c = desc_nir_c.to(self.device)
                        desc_rgb_c = desc_rgb_c.to(self.device)
                        dists = torch.cdist(desc_nir_c, desc_rgb_c)
                        top2 = torch.topk(dists, k=2, dim=1, largest=False)
                        mask = top2.values[:, 0] < self.nndr_ratio * top2.values[:, 1]
                        if mask.sum() == 0:
                            continue

                        idx_q = torch.arange(desc_nir_c.shape[0], device=self.device)[mask]
                        idx_t = top2.indices[mask, 0]
                        channel_losses.append(F.pairwise_distance(desc_nir_c[idx_q], desc_rgb_c[idx_t]).mean())
                        valid_channels += 1

                    if valid_channels > 0:
                        desc_loss += sum(channel_losses) / valid_channels
                        num_valid_images += 1

            else:
                # OpenCV-based flow
                nir_np = nir_img[i].squeeze().detach().cpu().numpy()
                if nir_np.ndim == 3:
                    nir_np = nir_np[0]
                nir_np = (nir_np * 255.0).astype(np.uint8)
                rgb_np = (generated_rgb[i].detach().cpu().numpy() * 255.0).astype(np.uint8)

                channel_losses = []
                valid_channels = 0
                for c in range(3):
                    rgb_channel = rgb_np[c]
                    _, _, desc_nir, desc_rgb, matches = self.matcher.match_features_evaluation(nir_np, rgb_channel)
                    if desc_nir is None or desc_rgb is None or not matches:
                        continue

                    valid_pairs = [(m.queryIdx, m.trainIdx) for m in matches if m.queryIdx < len(desc_nir) and m.trainIdx < len(desc_rgb)]
                    if not valid_pairs:
                        continue

                    desc_nir_matched = torch.stack([desc_nir[q] for q, _ in valid_pairs]).to(self.device)
                    desc_rgb_matched = torch.stack([desc_rgb[t] for _, t in valid_pairs]).to(self.device)
                    channel_losses.append(F.pairwise_distance(desc_nir_matched, desc_rgb_matched).mean())
                    valid_channels += 1

                if valid_channels > 0:
                    desc_loss += sum(channel_losses) / valid_channels
                    num_valid_images += 1

        if num_valid_images == 0:
            print("Warning: DescriptorLoss found no valid matches!")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return desc_loss / num_valid_images

    def truncate_descs(self, desc_nir, desc_rgb, max_kp=1000):
        if desc_nir.shape[0] > max_kp:
            desc_nir = desc_nir[:max_kp]
        if desc_rgb.shape[0] > max_kp:
            desc_rgb = desc_rgb[:max_kp]
        return desc_nir, desc_rgb


class SSIM_loss(nn.Module):

    def __init__(self, gpu_ids = [], size_average: bool = True, channel: int = 3, win_size: int = 11):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
        self.ssim_module = SSIM(
            data_range=1.0,
            size_average=size_average,
            channel=channel,
            win_size=win_size
        ).to(self.device)
    
    def forward(self, generated_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:

        generated_image = (generated_image + 1) / 2
        target_image = (target_image + 1) / 2
        generated_image = generated_image.to(self.device)
        target_image = target_image.to(self.device)
        # print(f"Max: {torch.max(generated_image)}, Min: {torch.min(generated_image)}")
        ssim_score = self.ssim_module(generated_image, target_image)
        ssim_loss = 1.0 - ssim_score
        return ssim_loss

class MS_SSIM_loss(nn.Module):
    def __init__(self,  gpu_ids = [], data_range: int = 1.0, size_average: bool = True, channel: int = 3, win_size: int = 11):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
        self.ms_ssim_module = MS_SSIM(
            data_range=data_range,
            size_average=size_average,
            channel=channel,
            win_size=win_size
        ).to(self.device)
    
    def forward(self, generated_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        generated_image = (generated_image + 1) / 2
        target_image = (target_image + 1) / 2
        generated_image = generated_image.to(self.device)
        target_image = target_image.to(self.device)

        mssim_score = self.ms_ssim_module(generated_image, target_image)
        mssim_loss = 1.0 - mssim_score
        return mssim_loss
    
class DISTS_Loss(nn.Module):
    def __init__(self, gpu_ids = []):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
        self.dists_module = DISTS().to(self.device)

    def forward(self, generated_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:

        generated_image = (generated_image + 1) / 2
        target_image = (target_image + 1) / 2
        generated_image = generated_image.to(self.device)
        target_image = target_image.to(self.device)
        dists_value = self.dists_module(generated_image, target_image, require_grad=True, batch_average=True)
        return dists_value

class edge_based_loss(nn.Module):
    def __init__(self, gpu_ids=[], detector='canny', kernel_size=3, loss=['L1']):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
        self.detector = detector.lower()
        self.kernel_size = kernel_size
        self.loss = loss
        self.allowed_losses = ['L1', 'L2', 'SSIM', 'MS_SSIM']

        if 'MS_SSIM' in self.loss:
            self.ms_ssim_module = MS_SSIM(data_range=1.0, size_average=True, channel=1, win_size=11).to(self.device)
        if 'SSIM' in self.loss:
            self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=1, win_size=11).to(self.device)
        
        if self.detector == 'log':
            self.kernel_size = 5
            self.sigma = 1.0
            gaussian_kernel = self.get_gaussian_kernel(self.kernel_size, self.sigma, device=self.device)
            self.register_buffer("gaussian_kernel", gaussian_kernel)

    def rgb2gray(self, rgb: torch.Tensor) -> torch.Tensor:
        return K.color.rgb_to_grayscale(rgb)

    @staticmethod
    def laplacian_of_gaussian(x, gaussian_kernel, kernel_size=5, sigma=1.0):
        channels = x.shape[1]
        padding = kernel_size // 2
        gaussian = gaussian_kernel.to(x.device).expand(channels, 1, kernel_size, kernel_size)
        x_blur = F.conv2d(x, gaussian, padding=padding, groups=channels)

        laplacian_kernel = torch.tensor([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]], dtype=torch.float32, device=x.device).reshape(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.expand(channels, 1, 3, 3)

        edge = F.conv2d(x_blur, laplacian_kernel, padding=1, groups=channels)
        return edge.abs()

    @staticmethod
    def get_gaussian_kernel(kernel_size, sigma, device):
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        kernel_2d = torch.outer(g, g)
        return kernel_2d.view(1, 1, kernel_size, kernel_size)

    def compute_edge(self, img: torch.Tensor) -> torch.Tensor:
        if self.detector == 'sobel':
            edge = KF.Sobel()(img)

        elif self.detector == 'laplacian':
            edge = KF.Laplacian(kernel_size=self.kernel_size)(img)

        elif self.detector == 'canny':
            edge, _ = KF.Canny(low_threshold=0.1, high_threshold=0.3)(img)

        elif self.detector == 'log':
            edge = self.laplacian_of_gaussian(img, self.gaussian_kernel, kernel_size=self.kernel_size, sigma=self.sigma)

        else:
            raise ValueError(f"Unsupported detector: {self.detector}")
        
        return edge

    
    def normalize_batch(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, -1)
        min_vals = x.min(dim=1)[0].view(B, 1, 1, 1)
        max_vals = x.max(dim=1)[0].view(B, 1, 1, 1)
        return ((x.view_as(min_vals)) - min_vals) / (max_vals - min_vals + 1e-8)

    def forward(self, generated_rgb: torch.Tensor, nir_image: torch.Tensor = None, real_rgb:torch.Tensor = None) -> torch.Tensor:
        # Normalize and convert to grayscale
        generated_rgb = (generated_rgb + 1) / 2
        generated_gray = self.rgb2gray(generated_rgb)
        generated_gray = generated_gray.to(self.device)

        edge_gen = self.compute_edge(generated_gray)
        # edge_gen = K.enhance.normalize_min_max(edge_gen)
        # edge_gen = KF.gaussian_blur2d(edge_gen, (3, 3), (1.0, 1.0))  

        if nir_image is not None:
            nir_image = (nir_image + 1) / 2
            if nir_image.shape[1] == 3:
                nir_image = self.rgb2gray(nir_image)

            nir_image = nir_image.to(self.device)
            edge_comp = self.compute_edge(nir_image)

        if real_rgb is not None:
            real_rgb = (real_rgb + 1) / 2
            real_rgb_gray = self.rgb2gray(real_rgb)
            real_rgb_gray = real_rgb_gray.to(self.device)
            edge_comp = self.compute_edge(real_rgb_gray)
        
        # edge_comp = K.enhance.normalize_min_max(edge_comp)
        # edge_comp = KF.gaussian_blur2d(edge_comp, (3, 3), (1.0, 1.0)) 

        edge_loss = 0.0
        if 'L1' in self.loss:
            edge_loss += F.l1_loss(edge_gen, edge_comp)
        if 'L2' in self.loss:
            edge_loss += F.mse_loss(edge_gen, edge_comp)
        if 'SSIM' in self.loss:
            edge_loss += 1.0 - self.ssim_module(edge_gen, edge_comp)
        if 'MS_SSIM' in self.loss:
            edge_loss += 1.0 - self.ms_ssim_module(edge_gen, edge_comp)

        for l in self.loss:
            if l not in self.allowed_losses:
                raise ValueError(f"Unsupported loss type: {l}. Allowed: {self.allowed_losses}")
        
        return edge_loss



class FeatureLossMSD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_real,pred_fake, n_layers_D, num_D):
        loss_G_feat = 0.0
        feat_weights = 4.0 / (n_layers_D + 1)
        D_weights = 1.0 / num_D  
        for i in range(n_layers_D):  # for each scale
            for j in range(len(pred_fake[i]) - 1):  # skip final layer logits
                loss_G_feat += D_weights * feat_weights * F.l1_loss(pred_fake[i][j], pred_real[i][j].detach()) 
        return loss_G_feat



