import onnxruntime
import cv2
import easygui
import numpy as np

_MEAN = [0.41846699]
_STD = [0.28337935]

if __name__ == '__main__':
    modelname = easygui.fileopenbox(title='Select Model', default='*.onnx', filetypes=[['*.onnx', '*.ort', 'ONNX Model']])
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    sess_options.inter_op_num_threads = 4
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = onnxruntime.InferenceSession(modelname, sess_options)
    showcase = easygui.boolbox('Showcase Mode? (Upsamples using model, bicubic and bilinear methods)', 'Showcase Mode', ('Yes', 'No'), None, 'Yes', 'No')
    repeat = True
    while repeat:
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
        if filename is None:
            print('No file selected. Exiting.')
            exit()
        filename = filename.replace('\\', '/')
        img = cv2.imread(filename)
        imagename = filename.split('/')[-1]+' - '+str(img.shape[0])+'x'+str(img.shape[1]) # image name
        cv2.namedWindow(imagename, cv2.WINDOW_AUTOSIZE)
        repeatROI = True
        while repeatROI:
            rect = cv2.selectROI(imagename, img, True, False) # crop selection
            croppedimg = img[int(rect[1]):int(rect[1]+rect[3]), int(rect[0]):int(rect[0]+rect[2])].copy() # crop
            croppedycrcb = croppedimg.astype(np.float32)/255.
            croppedycrcb = cv2.cvtColor(croppedycrcb, cv2.COLOR_BGR2YCR_CB)
            ins = (croppedycrcb - _MEAN) / _STD
            input_name = sess.get_inputs()[0].name
            inputs = {input_name: np.moveaxis(ins, -1, 0)[0][None,None,:,:].astype(np.float32)}
            outs = sess.run(None, inputs)[0]
            outs = np.moveaxis(outs, 1, -1)[0]
            outs = (outs * _STD) + _MEAN
            outs = np.clip(outs, 0.0, 1.0)
            outs_cbcr = cv2.resize(croppedycrcb, (0,0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)[:,:,1:]
            outs = np.concatenate([outs, outs_cbcr], axis=-1)
            outs = cv2.cvtColor(outs.astype(np.float32), cv2.COLOR_YCR_CB2BGR)*255.
            outs = np.clip(outs, 0.0, 255.0).astype(np.uint8)
            subwindowtitle = imagename+' - '+', '.join([str(i) for i in rect])
            cv2.imshow(subwindowtitle, outs)
            if showcase:
                bil = cv2.resize(croppedimg, (0,0), fx=4,fy=4, interpolation=cv2.INTER_LINEAR)
                subwindowtitlebil = 'bilinear-'+imagename+' - '+', '.join([str(i) for i in rect])
                cv2.imshow(subwindowtitlebil, bil)
                cub = cv2.resize(croppedimg, (0,0), fx=4,fy=4, interpolation=cv2.INTER_CUBIC)
                subwindowtitlecub = 'bicubic-'+imagename+' - '+', '.join([str(i) for i in rect])
                cv2.imshow(subwindowtitlecub, cub)
            tosave = easygui.boolbox('Save image?', 'Save', ('Yes', 'No'), None, 'Yes', 'No')
            if tosave:
                cv2.imwrite(filename.split('/')[-1]+'_'+'_'.join([str(i) for i in rect])+'_upscale'+'.png', outs)
            repeatROI = easygui.boolbox('Select another region?', 'Select region', ('Yes', 'No'), None, 'Yes', 'No')
            cv2.destroyWindow(subwindowtitle)
            if showcase:
                cv2.destroyWindow(subwindowtitlebil)
                cv2.destroyWindow(subwindowtitlecub)
        cv2.destroyAllWindows()
        repeat = easygui.boolbox('Select another image?', 'Select image', ('Yes', 'No'), None, 'Yes', 'No')
