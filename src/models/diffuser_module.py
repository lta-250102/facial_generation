from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL, StableDiffusionPipeline, LEditsPPPipelineStableDiffusion
from ldm.models.autoencoder import AutoencoderKL as LDMAutoencoderKL
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from src.models.components.attr_embeder import AttrEmbedding
from transformers import CLIPTextModel, CLIPTokenizer
from ldm.modules.encoders.modules import BERTEmbedder
from lightning.pytorch.loggers import WandbLogger
from lightning import LightningModule
from diffusers import DDIMScheduler
from PIL import Image
from tqdm import tqdm
import numpy as np
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
        first_dtype = noise_images.dtype
        first_device = noise_images.device
        pred = self.unet(noise_images.to(dtype=self.unet.dtype, device=self.unet.device), 
                         timesteps.to(dtype=self.unet.dtype, device=self.unet.device), 
                         encoded_caption.to(dtype=self.unet.dtype, device=self.unet.device)).sample
        return pred.to(dtype=first_dtype, device=first_device)

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

    def old_inference_step(self, prompt: list, max_timesteps=20, guidance_scale=7.5, img_size=512):
        tokens = self.tokenizer(prompt, padding='max_length', return_tensors="pt", max_length=self.max_seq_len, truncation=True).to(self.encoder.device)
        prompt_embeds = self.encoder(**tokens).last_hidden_state.to(dtype=self.unet.dtype) # (1, 77, 768)
        
        # free guidance: concat caption with negative prompt
        uncond_input = self.tokenizer([""] * len(prompt), padding='max_length', return_tensors="pt", max_length=self.max_seq_len, truncation=True).to(self.encoder.device)
        negative_prompt_embeds = self.encoder(**uncond_input).last_hidden_state.to(dtype=self.unet.dtype) # (1, 77, 768)
        encoded_caption = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0) # (2, 77, 768)

        self.scheduler.set_timesteps(max_timesteps)
        timesteps = self.scheduler.timesteps # (20,)

        shape = (1, 4, img_size//self.vae_scale_factor, img_size//self.vae_scale_factor)
        layout = torch.strided
        latents = torch.randn(shape, dtype=self.unet.dtype, layout=layout) * self.scheduler.init_noise_sigma # (1, 4, 64, 64)
        
        for t in timesteps:
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

    def inference_step(self, prompt: list, max_timesteps=20, guidance_scale=7.5, img_size=512):
        pipe = StableDiffusionPipeline(vae=self.vae, unet=self.unet, text_encoder=self.encoder, tokenizer=self.tokenizer, scheduler=self.scheduler, safety_checker=None, feature_extractor=None).to('cuda')
        return pipe(prompt=prompt, height=img_size, width=img_size, num_inference_steps=max_timesteps, guidance_scale=guidance_scale).images[0]

    def on_train_epoch_end(self):
        prompt = ["The person in the picture is attractive, young, smiling. This person is with arched eyebrows, brown hair, straight hair, pointy nose, mouth slightly open, high cheekbones. This woman wears lipstick, earrings, heavy makeup."]
        image = self.inference_step(prompt, max_timesteps=20)
        os.makedirs('./examples', exist_ok=True)
        id = len(os.listdir('./examples'))
        image.save(f'./examples/{id}.jpg')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.learning_rate)
        return optimizer

class RawDiffusionModule(LightningModule):
    def __init__(self, diffusion_pretrained: str, learning_rate=1e-4, scheduler_prediction_type="v_prediction", **kwargs) -> None:
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.loss = torch.nn.functional.mse_loss
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type=scheduler_prediction_type)
        # self.unet = UNet2DConditionModel(block_out_channels=(64, 128, 256, 256), cross_attention_dim=emb_dim)
        self.unet = UNet2DConditionModel.from_pretrained(diffusion_pretrained, subfolder='unet')
        self.vae = AutoencoderKL.from_pretrained(diffusion_pretrained, subfolder='vae')
        self.vae.requires_grad_(False)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # self.embeder = torch.nn.Embedding(40, emb_dim)
        self.embeder = AttrEmbedding()
        self.embeder.load_state_dict(torch.load('./pretrained/attr_embeder.pt'))
        self.linear = torch.nn.Linear(640, 768)
        
    def forward(self, noise_images, timesteps, cond):
        first_dtype = noise_images.dtype
        first_device = noise_images.device
        pred = self.unet(noise_images.to(dtype=self.unet.dtype, device=self.unet.device), 
                         timesteps.to(dtype=self.unet.dtype, device=self.unet.device), 
                         cond.to(dtype=self.unet.dtype, device=self.unet.device)).sample
        return pred.to(dtype=first_dtype, device=first_device)
    
    @torch.no_grad()
    def preprocess(self, batch):
        clean_images, attr = batch['image'], (batch['attr']+1)/2
        batch_size = clean_images.shape[0]

        clean_images = torch.cat([self.vae.encode(clean_images[i : i + 1]).latent_dist.sample() for i in range(batch_size)], dim=0) * self.vae.config.scaling_factor
        noise = torch.randn(clean_images.shape, device=clean_images.device)

        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=clean_images.device, dtype=torch.int64)
        noise_images = self.scheduler.add_noise(clean_images, noise, timesteps)

        cond = self.embeder(attr)

        return noise_images, timesteps, cond, noise
    
    def training_step(self, batch, batch_idx):
        self.unet.train()
        noise_images, timesteps, cond, noise = self.preprocess(batch)
        noise_preds = self(noise_images, timesteps, cond)
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(noise_images, noise, timesteps)    
        loss = self.loss(noise_preds.float(), target.float(), reduction="mean")
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.unet.eval()
        noise_images, timesteps, cond, noise = self.preprocess(batch)
        noise_preds = self(noise_images, timesteps, cond)
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(noise_images, noise, timesteps)    
        loss = self.loss(noise_preds.float(), target.float(), reduction="mean")
        self.log('val_loss', loss)

    def inference_step(self, attr, max_timesteps=20, guidance_scale=7.5, img_size=256):
        attr_embed = self.linear(self.embeder((attr+1)/2))
        cond = torch.cat([torch.zeros(attr_embed.shape, device=self.unet.device), attr_embed.to(self.unet.device)], dim=0)
        self.scheduler.set_timesteps(max_timesteps)
        timesteps = self.scheduler.timesteps # (20,)

        shape = (1, 4, img_size//self.vae_scale_factor, img_size//self.vae_scale_factor)
        latents = torch.randn(shape, dtype=self.unet.dtype, layout=torch.strided, device=self.unet.device) * self.scheduler.init_noise_sigma # (1, 4, 64, 64)
        
        for t in timesteps:
            latent_model_input = self.scheduler.scale_model_input(torch.cat([latents] * 2), t) # (2, 4, 64, 64)
            
            noise_pred = self(latent_model_input, t, cond) # (2, 4, 64, 64)
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_start(self):
        attr = torch.tensor([[-1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1]]).to(self.embeder.weight.device)
        image = self.inference_step(attr)
        os.makedirs('./logs/outputs', exist_ok=True)
        id = len(os.listdir('./logs/outputs'))
        image.save(f'./logs/outputs/{id}.jpg')

class CollaDiffusionModule(LightningModule):
    def __init__(self, learning_rate = 1e-4, state_dict_path = './pretrained/colla_module.pt', fine_tune = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert torch.cuda.is_available(), 'cuda unavailable'
        self.learning_rate = learning_rate
        self.unet = UNetModel(image_size=64, in_channels=3, out_channels=3, 
                 model_channels=192, attention_resolutions=[8, 4, 2], 
                 num_res_blocks=2, channel_mult=[1, 2, 3, 5], num_heads=32, 
                 use_spatial_transformer=True, transformer_depth=1, context_dim=640, 
                 use_checkpoint=True, legacy=False)
        self.vae = LDMAutoencoderKL(embed_dim=3, 
                                    ckpt_path=None,
                                    lossconfig={'target': 'torch.nn.Identity'},
                                    ddconfig={
                                        'double_z':True,
                                        'z_channels':3,
                                        'resolution':256,
                                        'in_channels':3,
                                        'out_ch':3,
                                        'ch':128,
                                        'ch_mult':[1, 2, 4],
                                        'num_res_blocks':2,
                                        'attn_resolutions':[],
                                        'dropout':0.0
                                    })
        self.embeder = AttrEmbedding()
        self.text_encoder = BERTEmbedder(n_embed=640, n_layer=32)
        self.scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=2e-2, beta_schedule='linear')
        self.scale_factor = 0.058
        if fine_tune:
            self.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
            self.text_encoder.requires_grad_(False)

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps.to(device=device)

    def forward(self, batch):
        clean_images = batch['image'].to(dtype=self.dtype, device=self.device)
        attr = ((batch['attr']+1)/2).to(dtype=torch.int32, device=self.device)

        clean_images_encoded = self.vae.encode(clean_images).sample() * self.scale_factor
        timestep = torch.randint(0, 1000, (clean_images.shape[0],), dtype=torch.int32, device=self.device)
        noise = torch.rand_like(clean_images_encoded, device=self.device, dtype=self.dtype)
        latents = self.scheduler.add_noise(clean_images_encoded, noise, timestep) * self.scheduler.init_noise_sigma
        condition = self.embeder(attr)
        
        noise_pred = self.unet(latents, timestep, context=condition)
        
        loss = torch.nn.functional.mse_loss(noise, noise_pred.to(dtype=noise.dtype))
        self.log(f"{'train' if self.training else 'val'}_loss", loss)
        return loss
    
    def training_step(self, batch):
        return self(batch)

    def validation_step(self, batch):
        return self(batch)

    def postprocess(self, x_0_batch):
        x_0 = x_0_batch[0, :, :, :].unsqueeze(0)  # [1, 3, 256, 256]
        x_0 = x_0.permute(0, 2, 3, 1).to('cpu').numpy()
        x_0 = (x_0 + 1.0) * 127.5
        np.clip(x_0, 0, 255, out=x_0)  # clip to range 0 to 255
        x_0 = x_0.astype(np.uint8)
        image = Image.fromarray(x_0[0])
        return image

    @torch.no_grad()
    def edit_image(self, image: Image.Image, strength, attr, guidance_scale = False, num_inference_steps=20, return_intermediate=False):
        attr = ((attr+1)/2).to(dtype=torch.int32, device=self.device)
        image = torch.from_numpy((np.array(image.convert('RGB')).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)).to(dtype=self.dtype, device=self.device).unsqueeze(0)

        attr_embed = self.embeder(attr)
        cond = torch.cat([torch.zeros(attr_embed.shape).to(device=attr_embed.device, dtype=attr_embed.dtype), attr_embed], dim=0) if guidance_scale else attr_embed # (2, 77, 640)
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.get_timesteps(num_inference_steps, strength, self.device)
        encoded_img = self.vae.encode(image).sample() * self.scale_factor
        noise = torch.randn(encoded_img.shape, device=self.device, dtype=self.dtype)
        latents = self.scheduler.add_noise(encoded_img, noise, timesteps[0]) * self.scheduler.init_noise_sigma

        intermediate = []
        for t in tqdm(timesteps):
            latent_model_input = self.scheduler.scale_model_input(torch.cat([latents] * 2) if guidance_scale else latents, t) # (2, 3, 64, 64)
            
            noise_pred = self.unet(latent_model_input, t.unsqueeze(-1), cond) # (2, 3, 64, 64)
            if guidance_scale:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond) # (1, 3, 64, 64)
            
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            intermediate.append(latents)
        if return_intermediate:
            return [self.postprocess(self.vae.decode(latent/self.scale_factor)) for latent in intermediate]
        else: return self.postprocess(self.vae.decode(intermediate[-1]/self.scale_factor))

    @torch.no_grad()
    def gen_image(self, prompt, guidance_scale = False, num_inference_steps=20, return_intermediate=False):
        prompt_encoded = self.text_encoder(prompt)
        cond = torch.cat([torch.zeros(prompt_encoded.shape).to(device=prompt_encoded.device, dtype=prompt_encoded.dtype), prompt_encoded], dim=0) if guidance_scale else prompt_encoded # (2, 77, 640)
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps.to(device=self.device) # (20,)
        latents = torch.randn((1, 3, 64, 64), device=self.device, dtype=self.dtype) * self.scheduler.init_noise_sigma
        
        intermediate = []
        for t in tqdm(timesteps):
            latent_model_input = self.scheduler.scale_model_input(torch.cat([latents] * 2) if guidance_scale else latents, t) # (2, 3, 64, 64)
            
            noise_pred = self.unet(latent_model_input, t.unsqueeze(-1), cond) # (2, 3, 64, 64)
            if guidance_scale:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond) # (1, 3, 64, 64)
            
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            intermediate.append(latents)
        if return_intermediate:
            return [self.postprocess(self.vae.decode(latent/self.scale_factor)) for latent in intermediate]
        else: return self.postprocess(self.vae.decode(intermediate[-1]/self.scale_factor))
    
    def on_train_epoch_start(self):
        attr = torch.tensor([[-1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1]])
        prompt = ''
        images = self.gen_image(prompt)
        edited_imgs = self.edit_image(image=images[-1], strength=0.5, attr=attr)
    
        if self.trainer and self.trainer.logger and isinstance(self.trainer.logger, WandbLogger):
            self.trainer.logger.log_image('generated_images', images)
            self.trainer.logger.log_image('edited_images', edited_imgs)
        else:
            os.makedirs('./logs/samples', exist_ok=True)
            id = len(os.listdir('./logs/samples'))
            images[0].save(f'./logs/samples/{id}.jpg')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

if __name__ == '__main__':
    model = CollaDiffusionModule()
    model.on_train_epoch_start()
