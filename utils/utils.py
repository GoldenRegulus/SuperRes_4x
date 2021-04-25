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

def total_variation(img):
    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

    reduce_axes = (-3, -2, -1)
    res1 = pixel_dif1.abs().sum(dim=reduce_axes)
    res2 = pixel_dif2.abs().sum(dim=reduce_axes)
    return res1 + res2

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

def _ssim(true: np.ndarray, pred: np.ndarray):
    """
    Computes SSIM value of predicted image with respect to ground truth.

    Parameters:
    
    `true`: A `numpy.ndarray` of shape `[H,W]` with values in range `[0, 255]`, the ground truth image.
    `pred`: A `numpy.ndarray` of shape `[H,W]` with values in range `[0, 255]`, the predicted image.

    Returns:

    The SSIM between `true` and `pred`.
    """
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    true = true.astype(np.float64)
    pred = pred.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(true, -1, window)[5:-5, 5:-5]  # valid padding
    mu2 = cv2.filter2D(pred, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(true**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(pred**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(true * pred, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def SSIM(true: np.ndarray, pred: np.ndarray):
    """
    Computes SSIM value of predicted image with respect to ground truth.

    Parameters:
    
    `true`: A `numpy.ndarray` of shape `[H,W[,C]]` with values in range `[0, 255]`, the ground truth image.
    `pred`: A `numpy.ndarray` of shape `[H,W[,C]]` with values in range `[0, 255]`, the predicted image.

    Returns:

    The SSIM between `true` and `pred`.
    """
    if not true.shape == pred.shape:
        raise ValueError('Input images must have the same dimensions.')
    if true.ndim == 2:
        return _ssim(true, pred)
    elif true.ndim == 3:
        if true.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(_ssim(true[:,:,i], pred[:,:,i]))
            return np.array(ssims).mean()
        elif true.shape[2] == 1:
            return _ssim(np.squeeze(true), np.squeeze(pred))
    else:
        raise ValueError('Wrong input image dimensions.')

def PSNR(true: np.ndarray, pred: np.ndarray):
    """
    Computes PSNR value of predicted image with respect to ground truth.

    Parameters:

    `true`: A `numpy.ndarray` of shape `[H,W]` with values in range `[0, 255]`, the ground truth image.
    `pred`: A `numpy.ndarray` of shape `[H,W]` with values in range `[0, 255]`, the predicted image.

    Returns:

    The PSNR between `true` and `pred`.
    """
    true = true.astype(np.float64)
    pred = pred.astype(np.float64)
    mse = np.mean((true - pred)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

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