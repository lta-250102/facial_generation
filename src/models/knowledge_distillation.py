from ldm.modules.encoders.modules import BERTEmbedder
from src.models.components.attr_embeder import AttrEmbedding
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.models.autoencoder import AutoencoderKL as LDMAutoencoderKL
from diffusers import UNet2DConditionModel, AutoencoderKL
from lightning import LightningModule
import torch


class Text2Attr(LightningModule):
    def __init__(self, learning_rate = 1e-4, teacher_path = "./pretrained/text_encoder.pt", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.student = AttrEmbedding()
        self.teacher = BERTEmbedder(n_embed=640, n_layer=32)
        self.teacher.load_state_dict(torch.load(teacher_path))
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def forward(self, batch):
        caption = batch['caption']
        attr = ((batch['attr']+1)/2).to(dtype=torch.int32, device=self.device)

        student_out = self.student(attr)
        with torch.no_grad():
            teacher_out = self.teacher(caption)

        #Soften the student logits by applying softmax first and log() second
        soft_targets = torch.nn.functional.softmax(teacher_out / 2, dim=-1)
        soft_prob = torch.nn.functional.log_softmax(student_out / 2, dim=-1)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * 4

        self.log(f'{"train" if self.training else "val"}_loss', loss, prog_bar=True)
        return loss
    
    def training_step(self, batch):
        self.student.train()
        return self(batch)
    
    def validation_step(self, batch):
        self.student.eval()
        return self(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.learning_rate)
        return optimizer

class Unet2Unet(LightningModule):
    def __init__(self, diffusion_pretrained = './pretrained/portrait_10', learning_rate = 1e-4, teacher_path = "./pretrained/unet_cuda.pt", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.teacher = UNetModel(image_size=64, in_channels=3, out_channels=3, 
                                 model_channels=192, attention_resolutions=[8, 4, 2], 
                                 num_res_blocks=2, channel_mult=[1, 2, 3, 5], num_heads=32, 
                                 use_spatial_transformer=True, transformer_depth=1, context_dim=640, 
                                 use_checkpoint=True, legacy=False)
        self.teacher.load_state_dict(torch.load(teacher_path))
        self.student = UNet2DConditionModel.from_pretrained(diffusion_pretrained, subfolder='unet')
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def forward(self, batch):
        latent = batch['latent'].to(dtype=self.dtype, device=self.device)
        condition = batch['condition'].to(dtype=self.dtype, device=self.device)

        student_out = self.student(latent, condition)
        with torch.no_grad():
            teacher_out = self.teacher(latent, condition)

        #Soften the student logits by applying softmax first and log() second
        soft_targets = torch.nn.functional.softmax(teacher_out / 2, dim=-1)
        soft_prob = torch.nn.functional.log_softmax(student_out / 2, dim=-1)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * 4

        self.log(f'{"train" if self.training else "val"}_loss', loss, prog_bar=True)
        return loss
        pass

    def training_step(self, batch):
        self.student.train()
        return self(batch)
    
    def validation_step(self, batch):
        self.student.eval()
        return self(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.learning_rate)
        return optimizer
    
class VAE2VAE(LightningModule):
    def __init__(self, diffusion_pretrained = './pretrained/portrait_10', learning_rate = 1e-4, teacher_path = "./pretrained/256_vae.ckpt", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.teacher = LDMAutoencoderKL(embed_dim=3, 
                                    ckpt_path=teacher_path,
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
        self.student = AutoencoderKL.from_pretrained(diffusion_pretrained, subfolder='vae')
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def forward(self, batch):
        img = batch['image'].to(dtype=self.dtype, device=self.device)

        student_out = self.student.encode(img).latent_dist
        with torch.no_grad():
            teacher_out = self.teacher.encode(img).sample()

        #Soften the student logits by applying softmax first and log() second
        soft_targets = torch.nn.functional.softmax(teacher_out / 2, dim=-1)
        soft_prob = torch.nn.functional.log_softmax(student_out / 2, dim=-1)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * 4

        self.log(f'{"train" if self.training else "val"}_loss', loss, prog_bar=True)
        return loss

    def training_step(self, batch):
        self.student.train()
        return self(batch)
    
    def validation_step(self, batch):
        self.student.eval()
        return self(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.learning_rate)
        return optimizer