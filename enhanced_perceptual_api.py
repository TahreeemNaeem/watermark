"""
Enhanced API with improved perceptual loss and image quality preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
try:
    from tqdm.auto import tqdm  # pretty progress bar
except Exception:
    # Graceful fallback if tqdm is not installed
    class _DummyTqdm:
        def __init__(self, iterable, **kwargs):
            self._iter = iterable
        def __iter__(self):
            return iter(self._iter)
        def set_postfix(self, **kwargs):
            pass
    def tqdm(iterable, **kwargs):
        return _DummyTqdm(iterable, **kwargs)
from helper import *
from model.generator import SkipEncoderDecoder, input_noise
import numpy as np


class EnhancedPerceptualLoss(nn.Module):
    """Enhanced perceptual loss with multiple components to prevent blurring"""
    
    def __init__(self, mse_weight=1.0, l1_weight=0.5, smoothness_weight=0.05, 
                 tv_weight=0.1, fft_weight=0.05, detail_weight=0.2):
        super(EnhancedPerceptualLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.smoothness_weight = smoothness_weight
        self.tv_weight = tv_weight  # Total variation for smoothness
        self.fft_weight = fft_weight  # Frequency domain preservation
        self.detail_weight = detail_weight  # Detail preservation

    def forward(self, pred, target, mask):
        # Expand mask to match prediction dimensions if needed
        if mask.dim() == 3 and pred.dim() == 4:
            mask = mask.unsqueeze(1)  # Add channel dimension
        
        # Ensure mask values are between 0 and 1
        mask = torch.clamp(mask, 0, 1)
        
        # Calculate masks for different regions
        inv_mask = 1 - mask  # Regions to preserve (non-watermarked areas)
        
        # Only compute loss in non-watermarked regions
        pred_preserve = pred * inv_mask
        target_preserve = target * inv_mask
        
        # MSE Loss for general fidelity
        mse_loss = F.mse_loss(pred_preserve, target_preserve)
        
        # L1 Loss for sharper results
        l1_loss = F.l1_loss(pred_preserve, target_preserve)
        
        # Total Variation Loss for smoothness without blurring
        tv_loss = self.compute_tv_loss(pred)
        
        # Frequency domain loss to preserve texture
        fft_loss = self.compute_fft_loss(pred_preserve, target_preserve)
        
        # Detail preservation loss (using gradient differences)
        detail_loss = self.compute_detail_loss(pred, target, inv_mask)
        
        total_loss = (self.mse_weight * mse_loss + 
                     self.l1_weight * l1_loss + 
                     self.smoothness_weight * tv_loss +
                     self.tv_weight * tv_loss +
                     self.fft_weight * fft_loss +
                     self.detail_weight * detail_loss)
        
        return total_loss
    
    def compute_tv_loss(self, img):
        """Total variation loss for smoothness"""
        # Calculate differences in x and y directions
        img_pad = F.pad(img, (0, 1, 0, 1), mode='replicate')
        tv_x = torch.mean(torch.abs(img_pad[:, :, :, :-1] - img_pad[:, :, :, 1:]))
        tv_y = torch.mean(torch.abs(img_pad[:, :, :-1, :] - img_pad[:, :, 1:, :]))
        return tv_x + tv_y
    
    def compute_fft_loss(self, pred, target):
        """Frequency domain loss to preserve texture"""
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        
        pred_fft_magnitude = torch.abs(pred_fft)
        target_fft_magnitude = torch.abs(target_fft)
        
        return F.mse_loss(pred_fft_magnitude, target_fft_magnitude)
    
    def compute_detail_loss(self, pred, target, mask):
        """Detail preservation loss to maintain sharpness"""
        # Calculate gradients for both images
        pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        
        target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        # Apply mask to preserve gradients in non-watermarked regions
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        
        grad_loss_x = F.mse_loss(pred_grad_x * mask_x, target_grad_x * mask_x)
        grad_loss_y = F.mse_loss(pred_grad_y * mask_y, target_grad_y * mask_y)
        
        return grad_loss_x + grad_loss_y


class MultiScalePerceptualLoss(nn.Module):
    """Multi-scale perceptual loss to preserve details at different scales"""
    
    def __init__(self):
        super(MultiScalePerceptualLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target, mask):
        total_loss = 0.0
        
        # Original scale
        inv_mask = 1 - mask
        original_loss = self.mse(pred * inv_mask, target * inv_mask) * 0.5 + \
                       self.l1(pred * inv_mask, target * inv_mask) * 0.5
        total_loss += original_loss
        
        # Multi-scale losses
        scales = [0.5, 0.75, 1.25]  # Different scales to preserve details
        
        for scale in scales:
            if scale < 1.0:  # Downscale
                pred_scaled = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
                mask_scaled = F.interpolate(mask, scale_factor=scale, mode='bilinear', align_corners=False)
                
                inv_mask_scaled = 1 - mask_scaled
                scaled_loss = self.mse(pred_scaled * inv_mask_scaled, target_scaled * inv_mask_scaled) * 0.25 + \
                             self.l1(pred_scaled * inv_mask_scaled, target_scaled * inv_mask_scaled) * 0.25
                total_loss += scaled_loss
            else:  # Upscale
                pred_scaled = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
                
                # Crop to original size
                h, w = target.shape[-2:]
                ph, pw = pred_scaled.shape[-2], pred_scaled.shape[-1]
                
                if ph >= h and pw >= w:
                    pred_scaled = pred_scaled[:, :, :h, :w]
                    target_scaled = target_scaled[:, :, :h, :w]
                    # Use the original mask for upscaled versions
                    scaled_loss = self.mse(pred_scaled * inv_mask, target_scaled * inv_mask) * 0.1
                    total_loss += scaled_loss
        
        return total_loss


def enhanced_remove_watermark(
    image_path,
    mask_path,
    max_dim=768,
    reg_noise=0.015,
    input_depth=64,
    lr=0.008,
    show_step=500,
    training_steps=2200,
    tqdm_length=100,
    loss_type='enhanced',
    early_stop_patience=0,
    min_delta=1e-6,
):
    """
    Enhanced watermark removal with improved loss functions to prevent blurring
    """
    DTYPE = torch.FloatTensor
    has_set_device = False
    
    if torch.cuda.is_available():
        device = 'cuda'
        has_set_device = True
        print("Setting Device to CUDA...")
    try:
        if torch.backends.mps.is_available():
            device = 'mps'
            has_set_device = True
            print("Setting Device to MPS...")
    except Exception as e:
        print(f"Your version of pytorch might be too old, which does not support MPS. Error: \n{e}")
        pass
    
    if not has_set_device:
        device = 'cpu'
        print('\nSetting device to "cpu", since torch is not built with "cuda" or "mps" support...')
        print('It is recommended to use GPU if possible...')

    image_np, mask_np = preprocess_images(image_path, mask_path, max_dim)

    print('Building the enhanced model...')
    # Enhanced model with better architecture for quality
    generator = SkipEncoderDecoder(
        input_depth,
        num_channels_down=[160, 160, 128, 128, 128],
        num_channels_up=[160, 160, 128, 128, 128],
        num_channels_skip=[32, 32, 32, 32, 32]
    ).type(DTYPE).to(device)

    # Select loss function
    if loss_type == 'enhanced':
        objective = EnhancedPerceptualLoss(
            mse_weight=1.0,
            l1_weight=0.5,
            smoothness_weight=0.05,
            tv_weight=0.1,
            fft_weight=0.05,
            detail_weight=0.2
        ).type(DTYPE).to(device)
    elif loss_type == 'multiscale':
        objective = MultiScalePerceptualLoss().type(DTYPE).to(device)
    else:
        # Original MSE loss
        objective = torch.nn.MSELoss().type(DTYPE).to(device)
    
    # Use AdamW optimizer with better parameters for quality
    optimizer = optim.AdamW(generator.parameters(), lr=lr, weight_decay=1e-6, betas=(0.9, 0.999))
    
    # Cosine annealing scheduler for learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_steps, eta_min=1e-7)

    image_var = np_to_torch_array(image_np).type(DTYPE).to(device)
    mask_var = np_to_torch_array(mask_np).type(DTYPE).to(device)

    generator_input = input_noise(input_depth, image_np.shape[1:]).type(DTYPE).to(device)
    generator_input_saved = generator_input.detach().clone()
    noise = generator_input.detach().clone()

    print('\nStarting enhanced training for watermark removal...')
    print('Using advanced loss functions to preserve image quality and reduce blurring...')
    print()

    progress_bar = tqdm(range(training_steps), desc='Progress', ncols=tqdm_length)
    best = float('inf')
    no_improve = 0

    for step in progress_bar:
        optimizer.zero_grad()
        generator_input = generator_input_saved

        # Add regularization noise but keep it low to prevent artifacts
        if reg_noise > 0:
            generator_input = generator_input_saved + (noise.normal_() * reg_noise)

        output = generator(generator_input)

        # Calculate loss using the mask (apply loss only to non-watermarked regions)
        if loss_type == 'enhanced' or loss_type == 'multiscale':
            loss = objective(output, image_var, mask_var)
        else:
            # Original loss: match masked region
            loss = objective(output * (1 - mask_var), image_var * (1 - mask_var))

        loss.backward()
        
        # Gradient clipping to prevent artifacts
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.5)

        optimizer.step()
        scheduler.step()

        if step % show_step == 0 and step > 0:
            output_image = torch_to_np_array(output)
            visualize_sample(image_np, mask_np, output_image, nrow=3, size_factor=8)
            print(f'\nStep {step}, Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.8f}')

        cur = float(loss.item())
        if cur < best - float(min_delta):
            best = cur
            no_improve = 0
        else:
            no_improve += 1

        progress_bar.set_postfix(Loss=f'{cur:.6f}')

        if early_stop_patience and no_improve >= early_stop_patience:
            # Early stop when loss plateaus
            break

    output_image = torch_to_np_array(output)
    visualize_sample(output_image, nrow=1, size_factor=10)

    # Convert to PIL image with proper clipping
    final_output = np.transpose(np.clip(output_image, 0, 1), (1, 2, 0))
    pil_image = Image.fromarray((final_output * 255.0).astype('uint8'))

    # Create output filename
    base_name = image_path.split('/')[-1].split('.')[0]
    output_path = f"{base_name}_enhanced_output.png"
    
    print(f'\nSaving final output image to: "{output_path}"\n')
    pil_image.save(output_path)
    
    return output_image


if __name__ == "__main__":
    print("Enhanced Perceptual Watermark Removal API")
    print("=" * 50)
    print("This API uses advanced perceptual loss functions to preserve image quality")
    print("and reduce blurring during watermark removal.")
