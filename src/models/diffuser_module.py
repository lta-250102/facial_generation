from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from lightning import LightningModule
from PIL import Image
import torch
import os


class DiffusionModule(LightningModule):
    def __init__(self, diffusion_pretrained: str, max_seq_len=77, fine_tune=True, learning_rate=1e-4, scheduler_prediction_type="v_prediction"):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss = torch.nn.functional.mse_loss
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type=scheduler_prediction_type)
        self.encoder = CLIPTextModel.from_pretrained(diffusion_pretrained, subfolder='text_encoder')
        self.tokenizer = CLIPTokenizer.from_pretrained(diffusion_pretrained, subfolder='tokenizer')
        self.encoder.requires_grad_(False)
        self.max_seq_len = max_seq_len
        if fine_tune:
            self.unet = UNet2DConditionModel.from_pretrained(diffusion_pretrained, subfolder='unet')
        else:
            self.unet = UNet2DConditionModel(block_out_channels=(64, 128, 256, 512), cross_attention_dim=768)
        self.vae = AutoencoderKL.from_pretrained(diffusion_pretrained, subfolder='vae')
        self.vae.requires_grad_(False)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def forward(self, noise_images, timesteps, encoded_caption):
        return self.unet(noise_images, timesteps, encoded_caption).sample

    @torch.no_grad()
    def preprocess(self, batch):
        clean_images, caption = batch['image'], batch['caption']
        batch_size = clean_images.shape[0]

        clean_images = torch.cat([self.vae.encode(clean_images[i : i + 1]).latent_dist.sample() for i in range(batch_size)], dim=0) * self.vae.config.scaling_factor
        noise = torch.randn(clean_images.shape, device=clean_images.device)

        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=clean_images.device, dtype=torch.int64)
        noise_images = self.scheduler.add_noise(clean_images, noise, timesteps)

        if isinstance(caption, str):
            caption = [caption]
        assert isinstance(caption, list), "input must be a list of strings"
        tokens = self.tokenizer(caption, padding='max_length', return_tensors="pt", max_length=self.max_seq_len, truncation=True).to(self.encoder.device)
        encoded_caption = self.encoder(**tokens).last_hidden_state.to(dtype=self.unet.dtype)

        return noise_images, timesteps, encoded_caption, noise

    def training_step(self, batch, batch_idx):
        self.unet.train()
        noise_images, timesteps, encoded_caption, noise = self.preprocess(batch)
        noise_preds = self(noise_images, timesteps, encoded_caption)
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(noise_images, noise, timesteps)    
        loss = self.loss(noise_preds.float(), target.float(), reduction="mean")
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.unet.eval()
        noise_images, timesteps, encoded_caption, noise = self.preprocess(batch)
        noise_preds = self(noise_images, timesteps, encoded_caption)
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(noise_images, noise, timesteps)    
        loss = self.loss(noise_preds.float(), target.float(), reduction="mean")
        self.log('val_loss', loss)

    def inference_step(self, prompt: list, max_timesteps=20, guidance_scale=7.5):
        tokens = self.tokenizer(prompt, padding='max_length', return_tensors="pt", max_length=self.max_seq_len, truncation=True).to(self.encoder.device)
        prompt_embeds = self.encoder(**tokens).last_hidden_state.to(dtype=self.unet.dtype) # (1, 77, 768)
        
        # free guidance: concat caption with negative prompt
        uncond_input = self.tokenizer([""] * len(prompt), padding='max_length', return_tensors="pt", max_length=self.max_seq_len, truncation=True).to(self.encoder.device)
        negative_prompt_embeds = self.encoder(**uncond_input).last_hidden_state.to(dtype=self.unet.dtype) # (1, 77, 768)
        encoded_caption = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0) # (2, 77, 768)

        self.scheduler.set_timesteps(max_timesteps)
        timesteps = self.scheduler.timesteps # (20,)

        shape = (1, 4, 512//self.vae_scale_factor, 512//self.vae_scale_factor)
        layout = torch.strided
        latents = torch.randn(shape, dtype=self.unet.dtype, layout=layout) * self.scheduler.init_noise_sigma # (1, 4, 64, 64)
        
        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2) # (2, 4, 64, 64)
            latent_model_input = self.scheduler.scale_model_input(torch.cat([latents] * 2), t) # (2, 4, 64, 64)
            
            noise_pred = self(latent_model_input, t, encoded_caption) # (2, 4, 64, 64)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) # (1, 4, 64, 64)
            
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0] # (1, 3, 512, 512)
        
        # image = self.image_processor.postprocess(image, output_type='pil', do_denormalize=[True] * image.shape[0])
        image = torch.stack([(image[0] / 2 + 0.5).clamp(0, 1)]) # (1, 3, 512, 512)
        image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy() # (1, 512, 512, 3)
        image = (image * 255).round().astype("uint8")
        image = Image.fromarray(image[0])
        return image

    def on_train_epoch_end(self):
        prompt = ["The person in the picture is attractive, young, smiling. This person is with arched eyebrows, brown hair, straight hair, pointy nose, mouth slightly open, high cheekbones. This woman wears lipstick, earrings, heavy makeup."]
        image = self.inference_step(prompt, max_timesteps=20)
        os.makedirs('./examples', exist_ok=True)
        id = len(os.listdir('./examples'))
        image.save(f'./examples/{id}.jpg')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.learning_rate)
        return optimizer
