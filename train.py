import argparse
from utils.utils import create_list
import numpy as np
from pytorch_lightning import callbacks
from modelarch.Discriminator import Discriminator
from modelarch.PAN import PAN
from trainarch.Supervised import SuperRes
from pytorch_lightning import Trainer
from datasetarch.ImageDataset import UnsplashDataset
from torch import save, load, jit, ones, quantization

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models.')
    parser.add_argument('-r', '--recreate_list', action='store_true', help='Rebuild numpy list of image names. Use if source images have been changed.')
    parser.add_argument('-c', '--checkpoint', nargs='?', default=None, help='Path to checkpoint, if any')
    parser.add_argument('-mc', '--manual_checkpoint', nargs='?', default=None, help='Path to torch checkpoint, if any')
    parser.add_argument('-ld', '--logdir', nargs='?', default='./panlogs2/', help='Checkpoint logging directory for tensorboard')
    ns = vars(parser.parse_args())
    if ns['recreate_list']:
        create_list()
    data = np.load('./unsplashlist.npy')
    ds = UnsplashDataset(data, batch_size=16)
    pan = PAN()
    disc = Discriminator(64)
    lr_callback = callbacks.LearningRateMonitor('step')
    loaded_model_ckpt = ns['checkpoint']
    logdir = ns['logdir']
    if loaded_model_ckpt:
        model = SuperRes.load_from_checkpoint(loaded_model_ckpt, model=pan)
        trainer = Trainer(resume_from_checkpoint=loaded_model_ckpt, gpus=1, accumulate_grad_batches=2, benchmark=True, max_epochs=340, callbacks=[lr_callback, callbacks.QuantizationAwareTraining()], default_root_dir=logdir)
        trainer.fit(model, datamodule=ds)
    else:
        if ns['manual_checkpoint']:
            SuperRes.load_from_checkpoint(ns['manual_checkpoint'], model=pan)
        model = SuperRes(model=pan)
        trainer = Trainer(gpus=1,benchmark=True, max_epochs=340, limit_train_batches=0.01, callbacks=[lr_callback, callbacks.QuantizationAwareTraining(observer_type='histogram', input_compatible=True)], default_root_dir=logdir)
        trainer.fit(model, datamodule=ds) 
    # save(model, 'models/pan/pan4.pt')
    tr_model = model.to_torchscript()
    jit.save(tr_model, 'models/pan/pant4.pt')
    model.to_onnx('./models/panq.onnx', opset_version=12, input_names = ['input'], output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size', 2 : 'height', 3 : 'width'},'output' : {0 : 'batch_size', 2 : 'height', 3 : 'width'}}, do_constant_folding=True)
    