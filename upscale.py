import onnxruntime
from PIL import Image
import numpy as np
import argparse
import os
from torch import load as pload, no_grad
import torch
from torchvision.transforms import Normalize, ToTensor, ToPILImage
from utils.utils import get_mean_std_luma, unnormalize
from torch.jit import load as jload

_UNSPLASH_MEAN, _UNSPLASH_STD = get_mean_std_luma()

parser = argparse.ArgumentParser(description='Upscale images by 4x.')
parser.add_argument('image_path', nargs='*', default=[i for i in os.listdir('./') if '.png' in i])
parser.add_argument('-o', '--output_path', nargs='?', default='./')
parser.add_argument('-m', '--model', nargs='?', default='./models/pan.onnx')
parser.add_argument('-d', '--directory', action='store_true')
parser.add_argument('-p', '--pytorch', action='store_true', help='Use pytorch .pt model instead of onnx.')
parser.add_argument('-ts', '--torchscript', action='store_true', help='Use torchscript compiled model instead of onnx.')
parser.add_argument('-pr', '--prefix', nargs='?', default='UP4x_', help='Prefix to add to end of filename.')
ns = vars(parser.parse_args())
if ns['directory']:
    directory = ns['image_path'][0]
    ns['image_path'] = [directory + '/' + i for i in os.listdir(directory)]
if not os.path.isdir(ns['output_path']):
    os.mkdir(ns['output_path'])
if ns['pytorch']:
    model = pload(ns['model'], 'cpu')
if ns['torchscript']:
    model = jload(ns['model'], 'cpu')
for i in ns['image_path']:
    img = Image.open(i).convert('YCbCr')
    if ns['pytorch'] or ns['torchscript']:
        imgn = Normalize(mean=_UNSPLASH_MEAN, std=_UNSPLASH_STD)(ToTensor()(img))[0]
        with no_grad():
            outs = model(imgn[None,None,:,:]).detach()
        outs_y = unnormalize(outs, _UNSPLASH_MEAN, _UNSPLASH_STD)[0].clamp(0.0, 1.0)
        outs_cbcr = ToTensor()(img.resize((img.width*4, img.height*4)))[1:,:,:]
        outs = ToPILImage(mode='YCbCr')(torch.cat([outs_y,outs_cbcr], dim=0)).convert('RGB')
        outs.save((ns['output_path'] + ns['prefix'] + i.split('/')[-1]))
    else:
        imgn = np.asarray(img)/255.
        imgn = (imgn - _UNSPLASH_MEAN) / _UNSPLASH_STD
        sess = onnxruntime.InferenceSession(ns['model'])
        input_name = sess.get_inputs()[0].name
        inputs = {input_name: np.moveaxis(imgn, -1, 0)[0][None,None,:,:].astype(np.float32)}
        outs = sess.run(None, inputs)[0]
        outs = np.moveaxis(outs, 1, -1)[0]
        outs = (outs * _UNSPLASH_STD) + _UNSPLASH_MEAN
        outs = np.clip(outs, 0.0, 1.0) * 255.
        outs_cbcr = np.asarray(img.resize((img.width*4, img.height*4)))[:,:,1:]
        outs = np.concatenate([outs, outs_cbcr], axis=-1)
        Image.fromarray(outs.astype(np.uint8), mode='YCbCr').convert('RGB').save((ns['output_path'] + ns['prefix'] + i.split('/')[-1]))
