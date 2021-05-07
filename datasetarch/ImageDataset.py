from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import random as rand
from torchvision.transforms import functional as tvf, Compose, Normalize, ToTensor
import pytorch_lightning as pl
from utils.utils import get_mean_std

class SSImageDataset(Dataset):
    def __init__(self, data, transform=None, label_transform=None, random_flips=False):
        self.data = data
        self.transform = transform
        self.label_transform = label_transform
        self.random_flips = random_flips
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        with Image.open(self.data[index][0]) as image:
            image = image.convert('RGB')
            if self.transform:
                image = self.transform(image)
        with Image.open(self.data[index][1]) as label:
            label = label.convert('RGB')
            if self.label_transform:
                label = self.label_transform(label)
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
        self.dims = (3, 90, 80)
        self.batch_size = batch_size
        self.mean, self.std = get_mean_std()
        self.transforms = Compose([
            ToTensor(),
            Normalize(self.mean, self.std),
        ])

    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            self.utrainfull = SSImageDataset(self.data, self.transforms, self.transforms, True)
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
