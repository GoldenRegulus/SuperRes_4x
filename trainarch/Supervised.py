import pytorch_lightning as pl
import torch
from utils.utils import test_on_folder_luma, unnormalize, get_mean_std_luma
from torch import optim
from torch.nn import functional as f

class SuperRes(pl.LightningModule):
    def __init__(self, model, learning_rate=2e-4, anneal_min_learning_rate=5e-6):
        super().__init__()
        self.learning_rate = learning_rate
        self.example_input_array = torch.ones((1, 1, 90, 80))
        self.model = model
        self.anneal_min = anneal_min_learning_rate
        self.mean, self.std = get_mean_std_luma()
    
    def forward(self, inp):
        return self.model(inp)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.model(x)
        loss = f.l1_loss(x,y)
        self.log('train_loss_LR', loss, True, False, True, False)
        if self.global_step % 64 == 0:
            self.logger.experiment.add_image('input_img', unnormalize(batch[0], self.mean, self.std)[0], global_step=self.global_step)
            self.logger.experiment.add_image('output_img', unnormalize(x, self.mean, self.std)[0].clamp(0.0,1.0), global_step=self.global_step)
            self.logger.experiment.add_image('reference_img', unnormalize(y, self.mean, self.std)[0], global_step=self.global_step)
        self.logger.experiment.add_scalar('train_loss', loss, self.global_step)
        return loss
    
    def on_epoch_end(self) -> None:
        psnr, ssim = test_on_folder_luma('testimgs/Set5', self.eval().cpu())
        self.cuda().train()
        self.logger.experiment.add_scalar('PSNR', psnr, self.global_step)
        self.logger.experiment.add_scalar('SSIM', ssim, self.global_step)
        return super().on_epoch_end()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 29500, eta_min=self.anneal_min, last_epoch=(self.global_step if self.global_step else -1)),
            'interval': 'step'
        }
        return [optimizer], [scheduler]
