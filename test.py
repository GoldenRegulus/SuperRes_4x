from torch import load
from utils.utils import test_on_folder

if __name__ == '__main__':
    pan = load('models/pan/pan3.pt', 'cpu')
    print('Calculating PSNR and SSIM:')
    for imgset in ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']:
        test_on_folder('testimgs/'+imgset, pan)