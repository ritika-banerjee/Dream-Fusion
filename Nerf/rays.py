import torch

class SampleRays:
    
    def __init__(self):
        pass
    
    @staticmethod
    def sample_points_along_ray(ray_o, ray_d, num_samples=64, near=0.1, far=5.0):
        
        """
            Sample points along a ray between 'near' and 'far' distances.
            
            ray_o : (N, 3) -> Ray Origin (camera center)
            ray_d: (N, 3) -> Ray Direction
            
        """
        
        t_vals = torch.linspace(near, far, num_samples, device=ray_o.device) # sample depths
        t_vals = t_vals.expand(ray_o.shape[0], num_samples) # Shape: (N, num_samples)
        
        points = ray_o[..., None, :] + t_vals[..., None] * ray_d[..., None, :]
        return points.view(-1, 3) 
        