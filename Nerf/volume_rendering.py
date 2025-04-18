import torch

class VolumeRendering:
    
    @staticmethod
    def volume_render_radiance(rgb, sigma, z_vals, dirs):
        # Ensure dirs are normalized
        dirs = torch.nn.functional.normalize(dirs, dim=-1)

        # Distances between z_vals
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[...,:1], 1e-10)], dim=-1)
        dists = dists * torch.norm(dirs[..., None, :1], dim=-1)

        # Clamp sigma for safety
        sigma = torch.clamp(sigma, 0.0, 1e3)

        # Compute alpha
        alpha = 1.0 - torch.exp(-sigma * dists)
        alpha = torch.clamp(alpha, 0.0, 1.0)  # just in case

        # Compute transmittance
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]

        weights = alpha * transmittance
        final_rgb = torch.sum(weights[..., None] * rgb, dim=-2)

        return final_rgb, weights

        