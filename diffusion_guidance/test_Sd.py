import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffusion_guidance.stable_diffusion import StableDiffusion
from PIL import Image

sd = StableDiffusion()
img = sd.generate_image("a futuristic city")

img.save("output2.png")
# img.show()