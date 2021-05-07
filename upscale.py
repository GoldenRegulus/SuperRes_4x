import onnxruntime
from PIL import Image
import numpy as np
import argparse
import os
import torch
from torchvision.transforms import Normalize, ToTensor, ToPILImage
from utils.utils import unnormalize

_UNSPLASH_MEAN = [0.4257, 0.4211, 0.4025]
_UNSPLASH_STD = [0.3034, 0.2850, 0.2961]

parser = argparse.ArgumentParser(description='Upscale images by 4x.')
parser.add_argument('image_path', nargs='*', default=[i for i in os.listdir('./') if '.png' in i])
parser.add_argument('-o', '--output_path', nargs='?', default='./')
parser.add_argument('-m', '--model', nargs='?', default='./models/pan.onnx')
parser.add_argument('-d', '--directory', action='store_true')
parser.add_argument('-p', '--pytorch', action='store_true', help='Use TorchScript .pt model instead of onnx.')
ns = vars(parser.parse_args())
if ns['directory']:
    directory = ns['image_path'][0]
    ns['image_path'] = [directory + '/' + i for i in os.listdir(directory)]
if not os.path.isdir(ns['output_path']):
    os.mkdir(ns['output_path'])
if ns['pytorch']:
    model = torch.jit.load(ns['model'], 'cpu')
for i in ns['image_path']:
    img = Image.open(i).convert('RGB')
    if ns['pytorch']:
        img = Normalize(mean=_UNSPLASH_MEAN, std=_UNSPLASH_STD)(ToTensor()(img))
        with torch.no_grad():
            outs = model(img.unsqueeze(0)).detach()
        outs = unnormalize(outs, _UNSPLASH_MEAN, _UNSPLASH_STD)[0].clamp(0.0, 1.0)
        outs = ToPILImage()(outs)
        outs.save((ns['output_path'] + 'UP4x_' + i.split('/')[-1]))
    else:
        img = np.asarray(img)/255.
        img = (img - _UNSPLASH_MEAN) / _UNSPLASH_STD
        sess = onnxruntime.InferenceSession(ns['model'])
        input_name = sess.get_inputs()[0].name
        inputs = {input_name: np.moveaxis(img, -1, 0)[np.newaxis].astype(np.float32)}
        outs = sess.run(None, inputs)[0]
        outs = np.moveaxis(outs, 1, -1)[0]
        outs = (outs * _UNSPLASH_STD) + _UNSPLASH_MEAN
        outs = np.clip(outs, 0.0, 1.0) * 255.
        Image.fromarray(outs.astype(np.uint8)).save((ns['output_path'] + 'UP4x_' + i.split('/')[-1]))
