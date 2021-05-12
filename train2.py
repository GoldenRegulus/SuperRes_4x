import argparse

from utils.utils import create_list, get_mean_std
import numpy as np
from pytorch_lightning import LightningModule
from pytorch_lightning import callbacks
from pytorch_lightning import Trainer
from datasetarch.ImageDataset import DivDataset
from torch import jit, nn, ones, no_grad, optim
from torch.utils import mobile_optimizer
from torchvision.transforms import ToTensor,ToPILImage
from PIL import Image

class Denoiser(LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.denoise = nn.Sequential(
            nn.Conv2d(3,48,9,1,4),
            nn.Conv2d(48,24,7,1,3),
            nn.Conv2d(24,12,5,1,2),
            nn.Conv2d(12,3,3,1,1)
        )
        self.loss = nn.L1Loss()
        self.learning_rate = lr
    
    def forward(self, x):
        return self.denoise(x)
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        x = self.denoise(x)
        loss = self.loss(x,y)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100, eta_min=5e-6, last_epoch=(self.global_step if self.global_step else -1)),
            'interval': 'step'
        }
        return [optimizer], [scheduler]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models.')
    parser.add_argument('-r', '--recreate_list', action='store_true', help='Rebuild numpy list of image names. Use if source images have been changed.')
    parser.add_argument('-c', '--checkpoint', nargs='?', default=None, help='Path to checkpoint, if any')
    parser.add_argument('-ld', '--logdir', nargs='?', default='./panlogs2/', help='Checkpoint logging directory for tensorboard')
    parser.add_argument('-nt', '--no_train', action='store_true', help='Skip main training loop')
    parser.add_argument('-j', '--jit', action='store_true', help='Compile and save model in TorchScript')
    parser.add_argument('-q', '--quantize', action='store_true', help='Quantize model (int8). Note: Enabling this will override --jit')
    parser.add_argument('-mo', '--mobile_optimize', action='store_true', help='Apply mobile optimizations to TorchScript model. Note: Enabling this will override --jit')
    parser.add_argument('-ms', '--manual_savepoint', nargs='?', default='models/pan/', help='Path to directory to which model(s) will be saved')
    ns = vars(parser.parse_args())
    if ns['recreate_list']:
        create_list()
    data = np.load('./divlist.npy')
    ds = DivDataset(data, batch_size=8)
    lr_callback = callbacks.LearningRateMonitor('step')
    loaded_model_ckpt = ns['checkpoint']
    logdir = ns['logdir']
    if loaded_model_ckpt:
        model = Denoiser.load_from_checkpoint(loaded_model_ckpt)
        if not ns['no_train']:
            trainer = Trainer(resume_from_checkpoint=loaded_model_ckpt, gpus=1, benchmark=True, precision=16, max_epochs=100, callbacks=[lr_callback, callbacks.StochasticWeightAveraging(swa_lrs=0.05)], default_root_dir=logdir)
            trainer.fit(model, datamodule=ds)
    else:
        model = Denoiser()
        if not ns['no_train']:
            trainer = Trainer(gpus=1, benchmark=True, precision=16, max_epochs=100, callbacks=[lr_callback, callbacks.StochasticWeightAveraging(swa_lrs=0.05)], default_root_dir=logdir)
            trainer.fit(model, datamodule=ds) 
    mean,std=get_mean_std()
    print('Testing:')
    with no_grad():
        ToPILImage()(model(ToTensor()(Image.open('testimgs/Set5/baby.png').convert('RGB')).unsqueeze(0))[0].clamp(0.0,1.0)).show()
    if ns['jit'] or ns['mobile_optimize'] or ns['quantize']:
        ex_inputs = ones(1,3,90,80)
        tr_model = model.to_torchscript(method='trace', example_inputs=ex_inputs)
        with no_grad():
            ToPILImage()(tr_model(ToTensor()(Image.open('testimgs/Set5/baby.png').convert('RGB')).unsqueeze(0))[0].clamp(0.0,1.0)).show()
        jit.save(tr_model, ns['manual_savepoint']+'pant5denoise.pt')
    if ns['mobile_optimize']:
        tr_mob_optimized = mobile_optimizer.optimize_for_mobile(tr_model)
        jit.save(tr_mob_optimized, ns['manual_savepoint']+'pant5optdenoise.pt')