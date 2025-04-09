import torch
import torch.nn as nn
from .PositionalEncoding import PositionalEncoding
from diffusion_guidance.stable_diffusion import StableDiffusion
from Nerf.PositionalEncoding import PositionalEncoding
from Nerf.rays import SampleRays
from Nerf.volume_rendering import VolumeRendering
from torch.cuda.amp import autocast, GradScaler


scaler = torch.amp.GradScaler()

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, L=10):
        super(NeRF, self).__init__()
        self.L = L
        self.pe = PositionalEncoding(L)

        input_dim = 3 * 2 * L  
        self.input_layer = nn.Linear(input_dim, W)

        self.layers = nn.ModuleList()
        for i in range(D - 1):
            if i == 4:  
                self.layers.append(nn.Linear(W + input_dim, W))  
            else:
                self.layers.append(nn.Linear(W, W))
            self.layers.append(nn.ReLU())

        self.density_layer = nn.Linear(W, 1)  
        self.color_layer = nn.Linear(W, 3)  

    def forward(self, x):
        x_encoded = self.pe(x)  
        x_input = x_encoded  

        x = self.input_layer(x_encoded)
        x = torch.relu(x)

        for i, layer in enumerate(self.layers):
            if i == 8:  
                x = torch.cat([x, x_input], dim=-1)
            x = layer(x)

        density = self.density_layer(x)  
        color = torch.sigmoid(self.color_layer(x))  

        return density, color

def train_step(nerf, ray_o, ray_d, text_embed, sds_model):
    torch.cuda.empty_cache()
    
    nerf.train()
    
    # ↓ Reduce number of samples per ray
    num_samples = 32

    # Sample 3D points from rays
    sample_points = SampleRays.sample_points_along_ray(ray_o, ray_d, num_samples=num_samples)
    dirs = ray_d

    # NeRF Forward
    sigma, rgb = nerf(sample_points)

    num_rays = ray_o.shape[0]
    sigma = sigma.view(num_rays, num_samples)
    rgb = rgb.view(num_rays, num_samples, 3)

    # Z values for volume rendering
    z_vals = torch.linspace(0.1, 5.0, num_samples, device=ray_o.device)
    z_vals = z_vals.expand(num_rays, num_samples)

    # Volume Rendering
    rgb_map, weights = VolumeRendering.volume_render_radiance(rgb, sigma, z_vals, dirs)
    rgb_map.requires_grad_()

    H, W = 32, 32
    rgb_image = rgb_map.view(1, H, W, 3).permute(0, 3, 1, 2)

    vae_dtype = next(sds_model.vae.parameters()).dtype
    rgb_image = rgb_image.to(dtype=vae_dtype)

    with torch.no_grad():
        latents = sds_model.vae.encode(rgb_image * 0.5 + 0.5).latent_dist.sample()

    loss = sds_model.get_sds_loss_from_latents(latents, text_embed)
    loss.backward()

    return loss.item(), rgb_map.detach()
