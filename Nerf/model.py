import torch
import torch.nn as nn
from .PositionalEncoding import PositionalEncoding
from diffusion_guidance.stable_diffusion import StableDiffusion
from Nerf.PositionalEncoding import PositionalEncoding
from Nerf.rays import SampleRays
from Nerf.volume_rendering import VolumeRendering
from torch.amp import autocast, GradScaler


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

def train_step(nerf, ray_o, ray_d, text_embed, sds_model, optimizer, scaler):
    device = ray_o.device
    ray_o = ray_o.to(device)
    ray_d = ray_d.to(device)
    text_embed = text_embed.to(device)
    nerf.train()
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    num_samples = 32
    sample_points = SampleRays.sample_points_along_ray(ray_o, ray_d, num_samples=num_samples)
    dirs = ray_d

    with autocast(device_type='cuda'):
        sigma, rgb = nerf(sample_points)

        num_rays = ray_o.shape[0]
        H = W = int(num_rays ** 0.5)
        assert H * W == num_rays, f"num_rays {num_rays} must be a perfect square"

        sigma = sigma.view(num_rays, num_samples)
        rgb = rgb.view(num_rays, num_samples, 3)

        z_vals = torch.linspace(0.1, 5.0, num_samples, device=device).expand(num_rays, num_samples)
        rgb_map, weights = VolumeRendering.volume_render_radiance(rgb, sigma, z_vals, dirs)

        rgb_map = torch.clamp(rgb_map, 0.0, 1.0)

        rgb_image = rgb_map.view(1, H, W, 3).permute(0, 3, 1, 2).to(device)

        vae_dtype = next(sds_model.vae.parameters()).dtype
        rgb_image = rgb_image.to(dtype=vae_dtype)

        latents = sds_model.encode_latents(rgb_image)

        loss = sds_model.get_sds_loss(latents, text_embed)

    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    
    torch.nn.utils.clip_grad_norm_(nerf.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()

    return loss.item(), rgb_map
