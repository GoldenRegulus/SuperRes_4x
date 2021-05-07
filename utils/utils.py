import numpy as np
import torch
import cv2
from PIL import Image
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms import ToPILImage, ToTensor, Normalize

_UNSPLASH_MEAN = [0.4257, 0.4211, 0.4025]
_UNSPLASH_STD = [0.3034, 0.2850, 0.2961]

def get_mean_std():
    return _UNSPLASH_MEAN, _UNSPLASH_STD

def unnormalize(x, mean, std):
    return ((x * torch.Tensor(std).type_as(x)[None,:,None,None]) + torch.Tensor(mean).type_as(x)[None,:,None,None])

def bgr2ycbcr(img: np.ndarray, only_y: bool = True):
    '''
    Converts a BGR `numpy.ndarray` to YCbCr

    only_y: only return Y channel

    Input:
        uint8, [0, 255]

        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def downscale(img: Image.Image, factor: int = 4):
    """
    Downscales `img` by `factor` using Bicubic resampling for testing purposes.
    
    Parameters:

    `img`: A `PIL.Image`
    `factor`: An `int` which defines the downscaling factor

    Returns:

    A `PIL.Image` downscaled by `factor`.
    """
    return img.resize((img.width//factor, img.height//factor), resample=Image.BICUBIC)

def get_test_images(dir: str, factor: int = 4):
    imgs = os.listdir(dir)
    imgs_in_array = []
    imgs_lab_array = []
    transform = ToTensor()
    for simg in imgs:
        with Image.open(dir+('' if dir[-1] == '/' else '/')+simg) as img:
            imgs_in_array.append(transform(downscale(img, factor)))
        oimg = cv2.imread(dir+('' if dir[-1] == '/' else '/')+simg)
        oimg = oimg[:((oimg.shape[0]//factor)*factor),:((oimg.shape[1]//factor)*factor),:]
        imgs_lab_array.append(oimg)
    return imgs_in_array, imgs_lab_array

def test_on_folder(dir: str, model):
    # model.to('cuda:0')
    inps, orig = get_test_images(dir)

    crop_border = 4
    test_Y = True  # True: test Y channel only; False: test RGB channels

    PSNR_all = []
    SSIM_all = []

    for i in range(len(orig)):
        im_GT = orig[i]/255.0
        pred = model(Normalize(mean=_UNSPLASH_MEAN, std=_UNSPLASH_STD)(inps[i]).unsqueeze(0))
        im_Gen = np.array(ToPILImage()(unnormalize(pred, _UNSPLASH_MEAN, _UNSPLASH_STD)[0].clamp(0.0,1.0).cpu()))[:, :, ::-1].copy()/255.0

        if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
            im_GT_in = bgr2ycbcr(im_GT)
            im_Gen_in = bgr2ycbcr(im_Gen)
        else:
            im_GT_in = im_GT
            im_Gen_in = im_Gen

        # crop borders
        if im_GT_in.ndim == 3:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
        elif im_GT_in.ndim == 2:
            cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
            cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
        else:
            raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))
        # calculate PSNR and SSIM

        # psnr = PSNR(cropped_GT * 255, cropped_Gen * 255)
        # ssim = SSIM(cropped_GT * 255, cropped_Gen * 255)
        psnr = peak_signal_noise_ratio(cropped_GT, cropped_Gen, data_range=1.0)
        ssim = structural_similarity(cropped_GT, cropped_Gen, gaussian_weights=True, sigma=1.5, multichannel=True, data_range=1.0)
        PSNR_all.append(psnr)
        SSIM_all.append(ssim)
    print('{}: PSNR: {:.4f} dB, SSIM: {:.4f}'.format(dir.split('/')[-1], sum(PSNR_all) / len(PSNR_all), sum(SSIM_all) / len(SSIM_all)))

def create_list():
    l720 = ['./Unsplash/720/'+i for i in os.listdir('./Unsplash/720/')]
    l180 = ['./Unsplash/180/'+i for i in os.listdir('./Unsplash/180/')]
    data = [i for i in zip(l180, l720)]
    del l180
    del l720
    data = np.array(data)
    np.save('./unsplashlist', data)