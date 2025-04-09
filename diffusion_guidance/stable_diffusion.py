from diffusers import StableDiffusionPipeline
import torch

class StableDiffusion:
    def __init__(self, device="cuda", precision=torch.float16):
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=precision
        ).to(device)
        
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.scheduler = self.pipe.scheduler
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder.eval() 
        
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing()
        
        
    def get_text_embeddings(self, prompt):
        tokens = self.tokenizer(prompt, return_tensors='pt').input_ids.to("cuda")
        return self.text_encoder(tokens)[0]
    

    def encode_latents(self, image):
        
        # image: [B, 3, H, W] in [-1,1]
        
        latents = self.vae.encode(image*0.5 + 0.5).latent_dist.sample()
        latents = latents * 0.18215
        return latents
    
    def get_sds_loss(self, image, text_embed):
        
        with torch.no_grad():  # This line is optional, only if you don’t want gradients through VAE
            latents = self.encode_latents(image)
    
        t = torch.randint(0, 1000, (latents.shape[0], ), device=latents.device).long()
        noise = torch.randn_like(latents)
    
        noisy_latents = self.scheduler.add_noise(latents, noise, t)
    
        noise_pred = self.unet(noisy_latents, t, encoder_hidden_states=text_embed).sample
    
        w = (1 - self.scheduler.alphas_cumprod[t]).view(-1,1,1,1)
        grad = w * (noise_pred - noise)
    
        loss = (latents * grad).mean()
        return loss
        
        
    def generate_image(self, prompt, height=512, width=512, guidance_scale=7.5, num_inference_steps=50):
        with torch.autocast("cuda"):
            image = self.pipe(prompt, height=height, width=width, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
            
        return image