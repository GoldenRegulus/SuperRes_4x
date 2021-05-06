import argparse

import torch
from utils.utils import create_list
import numpy as np
from pytorch_lightning import callbacks
from modelarch.PAN import PAN
from trainarch.Supervised import SuperRes
from pytorch_lightning import Trainer
from datasetarch.ImageDataset import UnsplashDataset
from torch import save

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models.')
    parser.add_argument('-r', '--recreate_list', action='store_true', help='Rebuild numpy list of image names. Use if source images have been changed.')
    parser.add_argument('-c', '--checkpoint', nargs='?', default=None, help='Path to checkpoint, if any')
    parser.add_argument('-ld', '--logdir', nargs='?', default='./panlogs2/', help='Checkpoint logging directory for tensorboard')
    ns = vars(parser.parse_args())
    if ns['recreate_list']:
        create_list()
    data = np.load('./unsplashlist.npy')
    ds = UnsplashDataset(data, batch_size=32)
    pan = PAN()
    lr_callback = callbacks.LearningRateMonitor('step')
    loaded_model_ckpt = ns['checkpoint']
    logdir = ns['logdir']
    if loaded_model_ckpt:
        model = SuperRes.load_from_checkpoint(loaded_model_ckpt, model=pan)
        # trainer = Trainer(resume_from_checkpoint=loaded_model_ckpt, gpus=1, precision=16, benchmark=True, max_epochs=340, callbacks=[lr_callback], default_root_dir=logdir)
        # trainer.fit(model, datamodule=ds)
    else:
        model = SuperRes(model=pan)
        trainer = Trainer(gpus=1, precision=16, benchmark=True, max_epochs=340, callbacks=[lr_callback], default_root_dir=logdir)
        trainer.fit(model, datamodule=ds) 
    save(pan, 'models/pan/pan3.pt')
    tr_model = model.to_torchscript('models/pan/pant3.pt', method='trace')
    model.to_onnx('./models/pan.onnx', opset_version=11, input_names = ['input'], output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size', 2 : 'height', 3 : 'width'},'output' : {0 : 'batch_size', 2 : 'height', 3 : 'width'}}, do_constant_folding=True)
    