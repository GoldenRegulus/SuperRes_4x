import pytorch_lightning as pl
import torch
from utils.utils import unnormalize, get_mean_std
from torch import optim
from torch.nn import functional as f

class SuperRes(pl.LightningModule):
    def __init__(self, model, learning_rate=5e-4, anneal_min_learning_rate=1e-5):
        super().__init__()
        self.learning_rate = learning_rate
        self.example_input_array = torch.ones((1, 3, 90, 80))
        self.model = model
        self.anneal_min = anneal_min_learning_rate
        self.mean, self.std = get_mean_std()
    
    def forward(self, inp):
        return self.model(inp)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.model(x)
        loss = f.l1_loss(x,y)
        self.log('train_loss_LR', loss, True, False, True, False)
        if self.global_step % 32 == 0:
            self.logger.experiment.add_image('input_img', unnormalize(batch[0], self.mean, self.std)[0], global_step=self.global_step)
            self.logger.experiment.add_image('output_img', unnormalize(x, self.mean, self.std)[0].clamp(0.0,1.0), global_step=self.global_step)
            self.logger.experiment.add_image('reference_img', unnormalize(y, self.mean, self.std)[0], global_step=self.global_step)
        self.logger.experiment.add_scalar('train_loss', loss, self.global_step)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 25000, eta_min=self.anneal_min, last_epoch=(self.global_step if self.global_step else -1)),
            'interval': 'step'
        }
        return [optimizer], [scheduler]
