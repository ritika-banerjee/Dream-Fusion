from diffusers import StableDiffusionPipeline
import torch

class StableDiffusion:
    def __init__(self, device="cuda", precision=torch.float16):
        self.device = device

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=precision
        ).to(device)
        
        self.vae = self.pipe.vae.eval().to(device)
        self.unet = self.pipe.unet.eval().to(device)
        self.scheduler = self.pipe.scheduler
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder.eval().to(device)
        
        self.pipe.enable_attention_slicing()

    def get_text_embeddings(self, prompt):
        tokens = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.pipe.device)
        return self.text_encoder(tokens)[0].to(self.pipe.device)

    def encode_latents(self, image):
        # image: [B, 3, H, W] in [-1,1]
        image = image.to(self.device)

        encoded = self.vae.encode(image * 0.5 + 0.5)
        mu = encoded.latent_dist.mean
        logvar = encoded.latent_dist.logvar
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latents = mu + eps * std        
        latents = latents * 0.18215
        return latents.to(self.device)

    def get_sds_loss(self, latents, text_embed):
        latents = latents.to(self.device)
        text_embed = text_embed.to(self.device)

        t = torch.randint(0, 1000, (latents.shape[0],), device=self.device).long()
        noise = torch.randn_like(latents, device=self.device)

        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        noise_pred = self.unet(
        noisy_latents.to(self.pipe.device),  
        t.to(self.pipe.device),              
        encoder_hidden_states=text_embed.to(self.pipe.device)  
        ).sample

        w = (1 - self.scheduler.alphas_cumprod[t]).view(-1, 1, 1, 1).to(self.device)
        grad = w * (noise_pred - noise)

        loss = (latents * grad).mean()
        return loss

    def generate_image(self, prompt, height=512, width=512, guidance_scale=7.5, num_inference_steps=50):
        with torch.autocast(self.device):
            image = self.pipe(prompt, height=height, width=width, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
        return image
