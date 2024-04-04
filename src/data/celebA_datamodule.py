from lightning import LightningDataModule
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations
import numpy as np
import torch
import json
import os

class CelebAConditionalDataset(Dataset):
    def __init__(self, image_name_list, captions, attrs, image_folder, keys: list = ['caption', 'attr']):
        self.transform = albumentations.Compose([
            albumentations.Resize(256, 256),
            albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])
        self.image_folder = image_folder

        # list of valid image names & train test split
        self.keys = keys
        self.image_name_list = image_name_list
        if 'caption' in keys:
            self.captions = captions
        if 'attr' in keys:
            self.attrs = attrs

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, index):
        image_name = self.image_name_list[index]
        image = np.array(Image.open(os.path.join(self.image_folder, image_name)).convert('RGB'))
        data = {'image': self.transform(image=image)['image']}

        if 'caption' in self.keys:
            data['caption'] = self.captions.get(image_name, '').lower()
        if 'attr' in self.keys:
            data['attr'] = torch.tensor(self.attrs.get(image_name, [0]*40))

        return data

class CelebAConditionalDataModule(LightningDataModule):
    def __init__(
        self, batch_size, num_workers=None,
        caption_file='./data/celebA/captions.json',
        image_folder = './data/celebA/img_align_celeba',
        attr_file='./data/celebA/attrs.json',
        keys: list = ['caption', 'attr']
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else 0

        self.keys = keys
        self.captions, self.attrs = None, None
        if 'caption' in keys:
            with open(caption_file, 'r') as f:
                self.captions = json.load(f)
        if 'attr' in keys:
            with open(attr_file, 'r') as f:
                self.attrs = json.load(f)
                
        # list of valid image names & train test split
        self.image_name_list = os.listdir(image_folder)
        self.train_name_list, self.val_name_list = train_test_split(self.image_name_list, test_size=0.02, random_state=42)
        self.image_folder = image_folder

    def setup(self, stage=None):
        self.train_dataset = CelebAConditionalDataset(image_name_list=self.train_name_list, image_folder=self.image_folder, captions=self.captions, attrs=self.attrs, keys=self.keys)
        self.val_dataset = CelebAConditionalDataset(image_name_list=self.val_name_list, image_folder=self.image_folder, captions=self.captions, attrs=self.attrs, keys=self.keys)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True if self.num_workers != 0 else False,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True if self.num_workers != 0 else False)
