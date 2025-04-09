import torch

class VolumeRendering:
    
    @staticmethod
    def volume_render_radiance(rgb, sigma, z_vals, dirs):
        
        # compute distances between consecutive z_vals
        
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[...,:1], 1e-10)], dim=-1)
        
        # account for ray direction norm to make distances in real units
        dists = dists * torch.norm(dirs[..., None, :1], dim=-1)
        
        # compute alpha and transmittance
        
        alpha = 1.0 - torch.exp(-sigma * dists)
        transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
        )[..., :-1]
        
        weights = alpha * transmittance                        
        final_rgb = torch.sum(weights[..., None] * rgb, dim=-2) 

        return final_rgb, weights
        