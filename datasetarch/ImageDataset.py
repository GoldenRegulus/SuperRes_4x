from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import random as rand
from torchvision.transforms import functional as tvf, Compose, Normalize, ToTensor, CenterCrop
import pytorch_lightning as pl
from torchvision.transforms.transforms import Lambda
from utils.utils import get_mean_std, get_mean_std_luma

class SSImageDataset(Dataset):
    def __init__(self, data, mean, std, random_flips=False):
        self.data = data
        self.transform = ToTensor()
        self.norm = Normalize(mean, std)
        self.random_flips = random_flips
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        with Image.open(self.data[index][0]) as image:
            image = image.convert('YCbCr')
            if self.transform:
                image = self.transform(image)
                image = image[0,:,:].unsqueeze(0)
                image = self.norm(image)
        with Image.open(self.data[index][1]) as label:
            label = label.convert('YCbCr')
            if self.transform:
                label = self.transform(label)
                label = label[0,:,:].unsqueeze(0)
                label = self.norm(label)
        if self.random_flips:
            if rand.random() < 0.5:
                image = tvf.hflip(image)
                label = tvf.hflip(label)
            if rand.random() < 0.5:
                image = tvf.vflip(image)
                label = tvf.vflip(label)
        return (image, label)

class UnsplashDataset(pl.LightningDataModule):
    def __init__(self, data, batch_size=16):
        super().__init__()
        self.data = data
        self.dims = (1, 90, 80)
        self.batch_size = batch_size
        self.mean, self.std = get_mean_std_luma()

    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            self.utrainfull = SSImageDataset(self.data, self.mean, self.std, True)
            self.utrain = self.utrainfull
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.utrain, self.batch_size, num_workers=4, pin_memory=True)
    
    def open_image(self, img_string):
        with Image.open(img_string[0]) as image:
            image = image.convert('YCbCr')
        with Image.open(img_string[1]) as label:
            label = label.convert('YCbCr')
        return (image,label)
    
    def transform_image(self, img):
        return self.utrainfull.transform(img)

class SSDivDataset(Dataset):
    def __init__(self, data, random_flips=False):
        self.data = data
        self.random_flips = random_flips
        self.crop = CenterCrop((1920,1080))
        self.tensor = ToTensor()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        with Image.open(self.data[index][0]) as image:
            image = image.convert('RGB')
            image = self.tensor(image)
            image = self.crop(image)
        with Image.open(self.data[index][1]) as label:
            label = label.convert('RGB')
            label = self.tensor(label)
            label = self.crop(label)
        if self.random_flips:
            if rand.random() < 0.5:
                image = tvf.hflip(image)
                label = tvf.hflip(label)
            if rand.random() < 0.5:
                image = tvf.vflip(image)
                label = tvf.vflip(label)
        return (image, label)

class DivDataset(pl.LightningDataModule):
    def __init__(self, data, batch_size=16):
        super().__init__()
        self.data = data
        self.dims = (3, 1920, 1080)
        self.batch_size = batch_size

    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            self.utrainfull = SSDivDataset(self.data, True)
            self.utrain = self.utrainfull
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.utrain, self.batch_size, num_workers=4, pin_memory=True)
    
    def open_image(self, img_string):
        with Image.open(img_string[0]) as image:
            image = image.convert('RGB')
        with Image.open(img_string[1]) as label:
            label = label.convert('RGB')
        return (image,label)
    
    def transform_image(self, img):
        return self.transforms(img)