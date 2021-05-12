from torch import load
from utils.utils import test_on_folder, test_on_folder_luma

if __name__ == '__main__':
    pan = load('models/pan/pant6.pt', 'cpu')
    print('Calculating PSNR and SSIM:')
    for imgset in ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']:
        psnr, ssim = test_on_folder_luma('testimgs/'+imgset, pan)
        print('{}: PSNR: {:.4f} dB, SSIM: {:.4f}'.format(imgset, psnr, ssim))