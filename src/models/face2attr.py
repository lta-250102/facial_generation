from lightning import LightningModule
from torchvision.models import vit_b_16
from PIL import Image
import torch


class Face2Attr(LightningModule):
    def __init__(self, num_classes: int = 40, lr: float = 1e-4, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = vit_b_16(pretrained=False, num_classes=num_classes)
        self.model.requires_grad_(True)

    def forward(self, x):
        logits = self.model(x)
        return logits.sign()
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['attr']
        logits = self(x)
        loss = torch.nn.functional.mse_loss(logits, y)
        acc = torch.sum(y == logits) / (y.shape[0] * self.hparams.num_classes)
        self.log('train_acc', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['attr']
        logits = self(x)
        loss = torch.nn.functional.mse_loss(logits, y)
        acc = torch.sum(y == logits) / (y.shape[0] * self.hparams.num_classes)
        self.log('val_acc', acc)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def predict(self, x):
        logits = self(x)
        return logits
    