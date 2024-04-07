from ldm.modules.encoders.modules import BERTEmbedder
from src.models.components.attr_embeder import AttrEmbedding
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

        self.log(f'{"train" if self.training else "val"}_loss', loss)
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
