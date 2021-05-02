import argparse
import functools
from utils.utils import create_list, get_mean_std, test_on_folder, unnormalize
import numpy as np
from pytorch_lightning import callbacks
from modelarch.PAN import PAN
from trainarch.Supervised import SuperRes
from pytorch_lightning import Trainer
from datasetarch.ImageDataset import UnsplashDataset
from torch import jit, nn, ones, onnx, no_grad, optim, quantization,load, backends
from torch.utils import mobile_optimizer
from torchvision.transforms import ToTensor,ToPILImage,Normalize
from PIL import Image

def wrap_func(model, func):
    @functools.wraps(func)
    def wrapper(x):
        x = model.quant(x)
        x = func(x)
        x = model.dequant(x)
        return x
    return wrapper

def prep_quantize(model):
    model.eval()
    model.quant = quantization.QuantStub()
    model.dequant = quantization.DeQuantStub()
    model.forward = wrap_func(model, model.forward)
    model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
    quantization.prepare_qat(model, inplace=True)

def qat_train_one_epoch(model, tr, criterion, opt, steps=20, device='cpu'):
    count = 0
    mmean = 0
    for x, y in tr:
        x, y = x.to(device), y.to(device)
        gen = model(x)
        loss = criterion(gen,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        mmean = ((mmean*4)+loss)/5
        print(f'Step {count}: Loss: {mmean:.4f}', end='\r')
        count += 1
        if count >= steps:
            break

def quantize_qat_finetune(model, ds, epochs=8, steps_per_epoch=20, criterion=nn.L1Loss(), opt=None, device='cuda:0'):
    ds.setup()
    tr = ds.train_dataloader()
    if opt is None:
        opt = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    if device != 'cpu':
            model = model.to(device)
    for epoch in range(epochs):
        print(f'Epoch {epoch}:')
        qat_train_one_epoch(model, tr, criterion, opt, steps_per_epoch, device=device)
        if epoch > ((epochs-1)//2):
            model.apply(quantization.disable_observer)
        if device != 'cpu':
            model = model.to('cpu')
        qmodel = quantization.convert(model.eval(), inplace=False)
        qmodel.eval()
        test_on_folder('testimgs/Set5', qmodel)
        if device != 'cpu':
            model = model.to(device)
        model.train()
    model.eval()
    if device != 'cpu':
        model = model.to('cpu')
    quantization.convert(model, inplace=True)


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
    lr_callback = callbacks.LearningRateMonitor('step')
    loaded_model_ckpt = ns['checkpoint']
    logdir = ns['logdir']
    if loaded_model_ckpt:
        model = SuperRes.load_from_checkpoint(loaded_model_ckpt, model=pan)
        trainer = Trainer(resume_from_checkpoint=loaded_model_ckpt, gpus=1, accumulate_grad_batches=2, benchmark=True, max_epochs=340, callbacks=[lr_callback, callbacks.QuantizationAwareTraining()], default_root_dir=logdir)
        trainer.fit(model, datamodule=ds)
    else:
        if ns['manual_checkpoint']:
            pan.load_state_dict({k:v for k, v in load('models/pan/pan3.pt').state_dict().items() if k in pan.state_dict()})
        # model = SuperRes(model=pan)
        # trainer = Trainer(gpus=0,benchmark=True, max_epochs=16, callbacks=[lr_callback, callbacks.QuantizationAwareTraining(qconfig=quantization.get_default_qat_qconfig('fbgemm'), observer_type='histogram', input_compatible=True)], default_root_dir=logdir)
        # trainer.fit(model, datamodule=ds) 
    # save(model, 'models/pan/pan4.pt')
    print('Quantization Aware Finetuning:')
    prep_quantize(pan)
    quantize_qat_finetune(pan, ds, device='cuda:0', steps_per_epoch=512, epochs=8)
    mean,std=get_mean_std()
    print('Testing:')
    with no_grad():
        ToPILImage()(unnormalize(pan(Normalize(mean=mean, std=std)(ToTensor()(Image.open('testimgs/Set5/baby.png').convert('RGB'))).unsqueeze(0)), mean, std)[0].clamp(0.0,1.0)).show()
    ex_inputs = ones(1,3,90,80)
    tr_model = pan.to_torchscript(method='trace', example_inputs=ex_inputs)
    with no_grad():
        ToPILImage()(unnormalize(tr_model(Normalize(mean=mean, std=std)(ToTensor()(Image.open('testimgs/Set5/baby.png').convert('RGB'))).unsqueeze(0)), mean, std)[0].clamp(0.0,1.0)).show()
    jit.save(tr_model, 'models/pan/pant4.pt')
    tr_mob_optimized = mobile_optimizer.optimize_for_mobile(tr_model)
    jit.save(tr_mob_optimized, 'models/pan/pant4opt.pt')