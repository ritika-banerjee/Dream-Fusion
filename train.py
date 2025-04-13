import torch
import torch.nn.functional as F
from skimage.measure import marching_cubes
import trimesh
from Nerf.model import NeRF, train_step
from diffusion_guidance.stable_diffusion import StableDiffusion
from Nerf.PositionalEncoding import PositionalEncoding
from Nerf.rays import SampleRays
from Nerf.volume_rendering import VolumeRendering
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio

# CONFIG 
prompt = "high quality photo of a futuristic city"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, W = 64, 64
num_iterations = 1000
lr = 1e-4
num_samples = 32

nerf = NeRF().to(device)
sd = StableDiffusion(device=device)
optimizer = optim.Adam(nerf.parameters(), lr=lr)
scaler = torch.amp.GradScaler()

with torch.no_grad():
    text_embeddings = sd.get_text_embeddings(prompt).detach()

def extract_density_field(nerf_model, resolution=128, device="cuda"):
    coords = torch.stack(torch.meshgrid(
        torch.linspace(-1.5, 1.5, resolution),
        torch.linspace(-1.5, 1.5, resolution),
        torch.linspace(-1.5, 1.5, resolution),
        indexing="ij"
    ), dim=-1).reshape(-1, 3).to(device)  # (res^3, 3)

    with torch.no_grad():
        sigmas = []
        chunk = 4096
        for i in range(0, coords.shape[0], chunk):
            sigma, _ = nerf_model(coords[i:i+chunk])
            sigmas.append(sigma.squeeze(-1).cpu())
        sigma_volume = torch.cat(sigmas, dim=0).reshape(resolution, resolution, resolution)
    return sigma_volume.numpy()

def generate_mesh(sigma_volume, threshold=None):
    min_val, max_val = sigma_volume.min(), sigma_volume.max()
    print(f"Sigma volume range: min={min_val:.6f}, max={max_val:.6f}")

    if threshold is None or not (min_val <= threshold <= max_val):
        threshold = 0.5 * (min_val + max_val)
        print(f"Auto-selected threshold: {threshold:.6f}")

    verts, faces, normals, _ = marching_cubes(sigma_volume, level=threshold)
    return verts, faces, normals

def save_mesh(verts, faces, filename="mesh.obj"):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(filename)

# TRAINING LOOP 
for step in range(num_iterations):
    ray_o = torch.randn(4096, 3, device=device)
    ray_d = torch.randn(4096, 3, device=device)
    ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)

    optimizer.zero_grad()
    loss, rgb_map = train_step(nerf, ray_o, ray_d, text_embeddings, sd, optimizer, scaler)
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}: Loss = {loss:.4f}")

    if step % 100 == 0:
        img = rgb_map.detach().cpu().numpy().reshape(H, W, 3)
        plt.imshow(np.clip(img, 0, 1))
        plt.axis("off")
        os.makedirs("renders", exist_ok=True)
        plt.savefig(f"renders/step_{step}.png")
        plt.close()

# OUTPUT DIR
os.makedirs("outputs", exist_ok=True)

# DENSITY FIELD EXTRACTION
print("Extracting density field...")
density = extract_density_field(nerf, resolution=128, device=device)

# DENSITY HISTOGRAM
plt.hist(density.flatten(), bins=100)
plt.title("Density Histogram")
plt.savefig("outputs/density_histogram.png")
plt.close()
print("Density histogram saved to outputs/density_histogram.png")

# MESH GENERATION
print("Generating mesh with Marching Cubes...")
verts, faces, _ = generate_mesh(density)

# SAVE MESH
print("Saving mesh...")
save_mesh(verts, faces, filename="outputs/final_mesh.obj")
print("Mesh saved to outputs/final_mesh.obj")

# FINAL RENDER SAVE
plt.imshow(np.clip(rgb_map.cpu().numpy().reshape(H, W, 3), 0, 1))
plt.axis("off")
plt.savefig("outputs/final_render.png")
plt.close()
print("Final render saved to outputs/final_render.png")

# MESH COLORIZATION
def colorize_mesh(nerf_model, verts, device="cuda"):
    with torch.no_grad():
        points = torch.tensor(verts, dtype=torch.float32).to(device)
        _, colors = nerf_model(points)
        colors = colors.cpu().numpy()
    return colors

colors = colorize_mesh(nerf, verts, device=device)
mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=(colors * 255).astype(np.uint8))
mesh.export("outputs/textured_mesh.ply")
print("Textured mesh saved to outputs/textured_mesh.ply")

# TURNTABLE RENDERING
scene = mesh.scene()
frames = []

for angle in np.linspace(0, 360, 90):
    camera_transform = trimesh.transformations.rotation_matrix(
        np.radians(angle), [0, 1, 0], point=mesh.centroid
    )
    scene.set_camera(transform=camera_transform)
    img = scene.save_image(resolution=(512, 512))
    frames.append(np.frombuffer(img, dtype=np.uint8))

imageio.mimsave("outputs/turntable.gif", frames, fps=10)
print("Turntable saved as outputs/turntable.gif")
