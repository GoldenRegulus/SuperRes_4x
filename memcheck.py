import onnxruntime
from PIL import Image
import numpy as np
from torch import no_grad
import torch
from torchvision.transforms import ToTensor, ToPILImage
from utils.utils import get_mean_std_luma
from torch.jit import load as jload
from memory_profiler import profile

_UNSPLASH_MEAN, _UNSPLASH_STD = get_mean_std_luma()

@profile(precision=4)
def torch_ver():
    img = Image.open('testimgs/Set5/baby.png').resize((256,256)).convert('YCbCr')
    imgn = np.asarray(img)/255.
    imgn = (imgn - _UNSPLASH_MEAN) / _UNSPLASH_STD
    imgn = np.moveaxis(imgn, -1, 0)[0][None,None,:,:].astype(np.float32)
    with no_grad():
        outs = model(torch.from_numpy(imgn)).detach().numpy()
    outs = np.moveaxis(outs, 1, -1)[0]
    outs = (outs * _UNSPLASH_STD) + _UNSPLASH_MEAN
    outs = np.clip(outs, 0.0, 1.0) * 255.
    outs_cbcr = np.asarray(img.resize((img.width*4, img.height*4)))[:,:,1:]
    outs = np.concatenate([outs, outs_cbcr], axis=-1)
    Image.fromarray(outs.astype(np.uint8), mode='YCbCr').convert('RGB')
    print('Torch ver done.')

@profile(precision=4)
def onnx_ver():
    img = Image.open('testimgs/Set5/baby.png').resize((256,256)).convert('YCbCr')
    imgn = np.asarray(img)/255.
    imgn = (imgn - _UNSPLASH_MEAN) / _UNSPLASH_STD
    input_name = sess.get_inputs()[0].name
    inputs = {input_name: np.moveaxis(imgn, -1, 0)[0][None,None,:,:].astype(np.float32)}
    outs = sess.run(None, inputs)[0]
    outs = np.moveaxis(outs, 1, -1)[0]
    outs = (outs * _UNSPLASH_STD) + _UNSPLASH_MEAN
    outs = np.clip(outs, 0.0, 1.0) * 255.
    outs_cbcr = np.asarray(img.resize((img.width*4, img.height*4)))[:,:,1:]
    outs = np.concatenate([outs, outs_cbcr], axis=-1)
    Image.fromarray(outs.astype(np.uint8), mode='YCbCr').convert('RGB')
    print('ONNX ver done')

@profile(precision=4)
def bicubic_ver():
    img = Image.open('testimgs/Set5/baby.png').resize((256,256)).convert('YCbCr')
    outs_cbcr = ToTensor()(img.resize((img.width*4, img.height*4)))
    ToPILImage(mode='YCbCr')(outs_cbcr).convert('RGB')
    print('Bicubic ver done.')

@profile(precision=4)
def bicubic_opt():
    img = Image.open('testimgs/Set5/baby.png').resize((256,256)).convert('YCbCr')
    img.resize((img.width*4, img.height*4)).convert('RGB')
    print('Bicubic opt ver done.')

if __name__ == '__main__':
    model = torch.load('models/pan/pan62.pt', 'cpu')
    sess = onnxruntime.InferenceSession('models/pan/pan62.onnx')
    for i in range(20):
        onnx_ver()

