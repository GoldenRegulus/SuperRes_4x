from torch.jit import load
from torch import load as tload, tensor
from torchvision.transforms import ToTensor,ToPILImage,Normalize
import cv2
import easygui
import numpy as np

_MEAN = [0.4257, 0.4211, 0.4025]
_STD = [0.3034, 0.2850, 0.2961]

def unnormalize(img):
    return (img * (tensor(_STD)[None,:,None,None])) + (tensor(_MEAN)[None,:,None,None])

if __name__ == '__main__':
    tensorconvert = ToTensor()
    norm = Normalize(_MEAN, _STD)
    modelname = easygui.fileopenbox(title='Select Model', default='*.pt', filetypes=[['*.pt', 'TorchScript Serialized Module']])
    try:
        model = load(modelname.replace('\\', '/'))
    except(Exception):
        try:
            model = tload(modelname.replace('\\', '/'))
        except(Exception):
            print('Invalid model. Exiting.')
            exit()
    repeat = True
    while repeat:
        try:
            filename = easygui.fileopenbox('Select image to upscale', 'Select Image', filetypes=[
                ['*.bmp', '*.dib', 'Windows bitmaps'],
                ['*.jpeg', '*.jpg', '*.jpe', 'JPEG files'],
                ['*.jp2', 'JPEG 2000 files'],
                ['*.png', 'Portable Network Graphics'],
                ['*.webp', 'WebP'],
                ['*.pbm', '*.pgm', '*.ppm', '*.pxm', '*.pnm', 'Portable image format'],
                ['*.pfm', 'PFM files'],
                ['*.sr', '*.ras', 'Sun rasters'],
                ['*.tiff', '*.tif', 'TIFF files'],
                ['*.exr', 'OpenEXR Image files'],
                ['*.hdr', '*.pic', 'Radiance HDR'],
            ])
            filename = filename.replace('\\', '/')
        except(AttributeError):
            print('No file selected. Exiting.')
            exit()
        img = cv2.imread(filename)
        imagename = filename.split('/')[-1]+' - '+str(img.shape[0])+'x'+str(img.shape[1]) # image name
        cv2.namedWindow(imagename, cv2.WINDOW_AUTOSIZE)
        repeatROI = True
        while repeatROI:
            rect = cv2.selectROI(imagename, img, True, False) # crop selection
            croppedimg = img[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])].copy() # crop
            timg = tensorconvert(croppedimg)[None,[2,1,0],:,:] # convert to tensor (scale between 0 and 1, convert [H,W,C] to [C,H,W]), BGR to RGB, expand dimension for extra batch dimension
            timg = unnormalize(model(norm(timg))).detach() # normalize image, upscale, un-normalize, detach from any grad tracking
            timg = timg.clamp(0.0,1.0) # clamp values between 0 and 1
            timg = timg[:,[2,1,0],:,:][0].permute(1,2,0).numpy() # convert RGB to BGR, remove batch dimension, convert from [C,H,W] to [H,W,C], convert to numpy array
            timg = timg * 255.0 # convert 0 to 1 mapping to 0,255 for 8bit image
            subwindowtitle = imagename+' - '+', '.join([str(i) for i in rect])+' Press any key to reselect, c to reselect image, q to quit'
            cv2.imshow(subwindowtitle, timg.astype(np.uint8))
            chchar = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(subwindowtitle)
            if chchar == 'c':
                repeatROI = False
            elif chchar == 'q':
                repeatROI = False
                repeat = False
        cv2.destroyAllWindows()