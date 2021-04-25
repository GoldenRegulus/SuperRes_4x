import torch
import pytorch_lightning as pl
from torchvision.transforms import Normalize
from torch import nn, optim
from torch.nn import functional as f
from ..utils.utils import unnormalize, get_mean_std

class GAN(pl.LightningModule):
    def __init__(self, gen, disc, feature_extractor, learning_rate=1e-4):
        super().__init__()
        self.gen = gen
        self.disc = disc
        self.example_input_array = torch.ones((1, 3, 90, 80))
        self.vgg = feature_extractor
        self.learning_rate=learning_rate
        self.mean, self.std = get_mean_std()
    
    def forward(self, inp):
        x = self.gen(inp)
        return x
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        self.vgg.eval()
        ganloss = nn.BCEWithLogitsLoss()
        if optimizer_idx == 0: #train gen
            x = self.gen(x)
            ox = ((x * torch.Tensor(self.std).type_as(x)[None,:,None,None]) + torch.Tensor(self.mean).type_as(x)[None,:,None,None])
            oy = ((y * torch.Tensor(self.std).type_as(y)[None,:,None,None]) + torch.Tensor(self.mean).type_as(y)[None,:,None,None])
            nx = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(ox)
            ny = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(oy)
            vx = self.vgg(nx)
            vy = self.vgg(ny)
            vgg22_loss = f.l1_loss(vx[0], vy[0].detach())
            vgg54_loss = f.l1_loss(vx[1], vy[1].detach())
            vgg_loss = vgg22_loss + 0.5*vgg54_loss
            pixel_loss = f.l1_loss(x, y)
            fake_preds = self.disc(x)
            preds = self.disc(y).detach()
            gan_loss = ganloss(fake_preds - preds.mean(0, keepdim=True), torch.ones((x.shape[0], 1, 23, 20), requires_grad=False).type_as(fake_preds))
            loss = vgg_loss + 5e-3*gan_loss + 1e-2*pixel_loss
            if (self.global_step % 16 == 0):
                self.logger.experiment.add_image('input_img', unnormalize(batch[0], self.mean, self.std)[0], global_step=self.global_step)
                self.logger.experiment.add_image('output_img', unnormalize(x, self.mean, self.std)[0].clamp(0.0,1.0), global_step=self.global_step)
                self.logger.experiment.add_image('reference_img', unnormalize(y, self.mean, self.std)[0], global_step=self.global_step)
            self.log('g_total_loss', loss, True, False, True, False)
            self.log('g_vgg_loss', vgg_loss, True, False, True, False)
            self.log('g_gan_loss', gan_loss, True, False, True, False)
            self.log('g_pixel_loss', pixel_loss, True, False, True, False)
            self.logger.experiment.add_scalar('g_total_loss', loss, global_step=self.global_step)
            self.logger.experiment.add_scalar('g_vgg_loss', vgg_loss, global_step=self.global_step)
            self.logger.experiment.add_scalar('g_gan_loss', gan_loss, global_step=self.global_step)
            self.logger.experiment.add_scalar('g_pixel_loss', pixel_loss, global_step=self.global_step)
            return loss
        if optimizer_idx == 1: #train disc
            preds = self.disc(y)
            fake_imgs = self.gen(x).detach()
            fake_preds = self.disc(fake_imgs)
            valid_loss = ganloss(preds - fake_preds.mean(0, keepdim=True), torch.ones((x.shape[0], 1, 23, 20), requires_grad=False).type_as(preds))
            fake_loss = ganloss(fake_preds - preds.mean(0, keepdim=True), torch.zeros((x.shape[0], 1, 23, 20), requires_grad=False).type_as(fake_preds))
            loss = (valid_loss + fake_loss) / 2
            self.log('d_total_loss', loss, True, False, True, False)
            self.log('d_fake_loss', fake_loss, True, False, True, False)
            self.log('d_real_loss', valid_loss, True, False, True, False)
            self.logger.experiment.add_scalar('d_total_loss', loss, global_step=self.global_step)
            self.logger.experiment.add_scalar('d_fake_loss', fake_loss, global_step=self.global_step)
            self.logger.experiment.add_scalar('d_real_loss', valid_loss, global_step=self.global_step)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.vgg.eval()
        ganloss = nn.BCEWithLogitsLoss()
        x = self.gen(x)
        ox = ((x * torch.Tensor(self.std).type_as(x)[None,:,None,None]) + torch.Tensor(self.mean).type_as(x)[None,:,None,None])
        oy = ((y * torch.Tensor(self.std).type_as(y)[None,:,None,None]) + torch.Tensor(self.mean).type_as(y)[None,:,None,None])
        nx = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(ox)
        ny = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(oy)
        vx = self.vgg(nx)
        vy = self.vgg(ny)
        vgg_loss = f.l1_loss(vx, vy.detach())
        pixel_loss = f.l1_loss(x, y)
        fake_preds = self.disc(x)
        preds = self.disc(y).detach()
        gan_loss = ganloss(fake_preds - preds.mean(0, keepdim=True), torch.ones((x.shape[0], 1, 23, 20), requires_grad=False).type_as(fake_preds))
        loss = vgg_loss + 5e-3*gan_loss + 1e-2*pixel_loss
        return loss
        
    
    def configure_optimizers(self):
        optimizer_G = optim.AdamW(self.gen.parameters(), lr=self.learning_rate)
        optimizer_D = optim.AdamW(self.disc.parameters(), lr=self.learning_rate)
        scheduler_G = {
            'scheduler': optim.lr_scheduler.MultiStepLR(optimizer_G, [50000,100000,200000,300000,400000], 0.5, self.global_step if self.global_step else -1),
            'interval': 'step'    
        }
        scheduler_D = {
            'scheduler': optim.lr_scheduler.MultiStepLR(optimizer_D, [50000,100000,200000,300000,400000], 0.5, self.global_step if self.global_step else -1),
            'interval': 'step'    
        }
        return [optimizer_G, optimizer_D], [scheduler_G, scheduler_D]
