import argparse
import functools
from utils.utils import create_list, get_mean_std, get_mean_std_luma, test_on_folder, unnormalize
import numpy as np
from pytorch_lightning import callbacks
from modelarch.PAN import PAN
from trainarch.Supervised import SuperRes
from pytorch_lightning import Trainer
from datasetarch.ImageDataset import UnsplashDataset
from torch import jit, nn, ones, no_grad, optim, quantization,load, save
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

def qat_train_one_epoch(model, tr, criterion, opt, scheduler=None, steps=20, device='cpu'):
    count = 0
    mmean = 0
    for x, y in tr:
        x, y = x.to(device), y.to(device)
        gen = model(x)
        loss = criterion(gen,y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if scheduler:
            scheduler.step()
        mmean = ((mmean*4)+loss)/5
        print(f'Step {count}: Loss: {mmean:.4f}', end='\r')
        count += 1
        if count >= steps:
            break

def quantize_qat_finetune(model, ds, epochs=8, steps_per_epoch=20, criterion=nn.L1Loss(), opt=None, device='cuda', finetune_epoch=4, lr_scheduler=False, lr_cycles=1):
    ds.setup()
    tr = ds.train_dataloader()
    if opt is None:
        opt = optim.Adam(model.parameters(), lr=1e-4)
    if lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(steps_per_epoch*epochs)//lr_cycles, eta_min=1e-5)
    model.train()
    if device != 'cpu':
            model = model.to(device)
    try:
        for epoch in range(epochs):
            print(f'Epoch {epoch}:')
            qat_train_one_epoch(model, tr, criterion, opt, scheduler, steps_per_epoch, device=device)
            if epoch > finetune_epoch:
                model.apply(quantization.disable_observer)
            if device != 'cpu':
                model.to('cpu')
            qmodel = quantization.convert(model.eval(), inplace=False)
            qmodel.eval()
            test_on_folder('testimgs/Set5', qmodel)
            if device != 'cpu':
                model.to(device)
            model.train()
    except(KeyboardInterrupt):
        print('KeyboardInterrupt detected. Attempting to exit gracefully.')
    model.eval()
    if device != 'cpu':
        model.to('cpu')
    quantization.convert(model, inplace=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models.')
    parser.add_argument('-r', '--recreate_list', action='store_true', help='Rebuild numpy list of image names. Use if source images have been changed.')
    parser.add_argument('-c', '--checkpoint', nargs='?', default=None, help='Path to checkpoint, if any')
    parser.add_argument('-mc', '--manual_checkpoint', nargs='?', default=None, help='Path to torch checkpoint, if any')
    parser.add_argument('-ld', '--logdir', nargs='?', default='./panlogs2/', help='Checkpoint logging directory for tensorboard')
    parser.add_argument('-nt', '--no_train', action='store_true', help='Skip main training loop')
    parser.add_argument('-j', '--jit', action='store_true', help='Compile and save model in TorchScript')
    parser.add_argument('-ox', '--onnx', action='store_true', help='Compile and save model in ONNX')
    parser.add_argument('-q', '--quantize', action='store_true', help='Quantize model (int8). Note: Enabling this will override --jit')
    parser.add_argument('-mo', '--mobile_optimize', action='store_true', help='Apply mobile optimizations to TorchScript model. Note: Enabling this will override --jit')
    parser.add_argument('-ms', '--manual_savepoint', nargs='?', default='models/pan/', help='Path to directory to which model(s) will be saved')
    ns = vars(parser.parse_args())
    if ns['recreate_list']:
        create_list()
    data = np.load('./unsplashlist.npy')
    ds = UnsplashDataset(data, batch_size=8)
    pan = PAN()
    lr_callback = callbacks.LearningRateMonitor('step')
    loaded_model_ckpt = ns['checkpoint']
    logdir = ns['logdir']
    if loaded_model_ckpt:
        model = SuperRes.load_from_checkpoint(loaded_model_ckpt, model=pan)
        if not ns['no_train']:
            trainer = Trainer(resume_from_checkpoint=loaded_model_ckpt, gpus=1, benchmark=True, max_epochs=20, callbacks=[lr_callback], default_root_dir=logdir)
            trainer.fit(model, datamodule=ds)
    else:
        if ns['manual_checkpoint']:
            model_dict = pan.state_dict()
            pretrained_dict = {k:v for k, v in load(ns['manual_checkpoint']).state_dict().items() if (k in model_dict and (v.shape == model_dict[k].shape))}
            model_dict.update(pretrained_dict)
            pan.load_state_dict(model_dict, strict=False)
        model = SuperRes(model=pan)
        if not ns['no_train']:
            trainer = Trainer(gpus=1,benchmark=True, max_epochs=20, callbacks=[lr_callback], default_root_dir=logdir, accumulate_grad_batches=4)
            trainer.fit(model, datamodule=ds) 
    save(pan, ns['manual_savepoint']+'pan62.pt')
    if ns['quantize']:
        print('Quantization Aware Finetuning:')
        prep_quantize(pan)
        quantize_qat_finetune(pan, ds, device='cuda:0', steps_per_epoch=512, epochs=10, finetune_epoch=7, lr_scheduler=True, lr_cycles=1)
    mean,std=get_mean_std_luma()
    print('Testing:')
    with no_grad():
        ToPILImage()(unnormalize(pan(Normalize(mean=mean, std=std)(ToTensor()(Image.open('testimgs/Set5/baby.png').convert('YCbCr'))[0,:,:].unsqueeze(0)).unsqueeze(0)), mean, std)[0].clamp(0.0,1.0)).show()
    ex_inputs = ones(1,1,90,80)
    if ns['jit'] or ns['mobile_optimize'] or ns['quantize']:
        tr_model = pan.to_torchscript(method='trace', example_inputs=ex_inputs)
        with no_grad():
            ToPILImage()(unnormalize(tr_model(Normalize(mean=mean, std=std)(ToTensor()(Image.open('testimgs/Set5/baby.png').convert('YCbCr'))[0,:,:].unsqueeze(0)).unsqueeze(0)), mean, std)[0].clamp(0.0,1.0)).show()
        jit.save(tr_model, ns['manual_savepoint']+'pant62.pt')
    if ns['mobile_optimize']:
        tr_mob_optimized = mobile_optimizer.optimize_for_mobile(tr_model)
        jit.save(tr_mob_optimized, ns['manual_savepoint']+'pant62opt.pt')
    if ns['onnx']:
        on_model = pan.to_onnx(ns['manual_savepoint']+'pan62.onnx', ex_inputs, opset_version=12, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0:'batch', 2:'height', 3:'width'}, 'output': {0:'batch', 2:'height', 3:'width'}})