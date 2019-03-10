# -*- coding: utf-8 -*-

import os

import cv2
from PIL import Image
import numpy as np 

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import albumentations
from albumentations import ImageOnlyTransform
from albumentations import torch as AT

'''
Craete a dataloader for whale which can use Pandas DataFrame with specyfic structure

'''

class WhaleDataLoader(Dataset):
    def __init__(self, image_folder, process='train', df=None, transform=None):
        self.image_folder = image_folder
        self.imgs_list = [img for img in os.listdir(image_folder)]
        self.process = process
        self.transform = transform
        self.df = df
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        if self.process != 'test':
            img_name = os.path.join(self.image_folder, self.df.iloc[idx].Image)
            label = self.df.iloc[idx].label
        
        elif self.process == 'test':
            img_name = os.path.join(self.image_folder, self.df.iloc[idx].Image)
        
        #img = Image.open(img_name).convert('RGB')
        #img = self.transform(img)
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        
        if self.process != 'test':
            return img, label, idx
        elif self.process == 'test':
            return img, self.df.iloc[idx].Image, idx


class RGBToGray(ImageOnlyTransform):
    """
    Own to Gray funtion as albumentations revert image in case of mean pixel < 127
    
    Targets:
        image

    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        gray =  cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

class WeightedSubsetRandomSampler(Sampler):
    r"""Samples elements from a given list of indices with given probabilities (weights), with replacement.

    Arguments:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
    """

    def __init__(self, indices, weights, num_samples=0, replacement = True):
        #if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool):
            #raise ValueError("num_samples should be a non-negative integeral "
                             #"value, but got num_samples={}".format(num_samples))
        self.indices = indices
        weights = [ weights[i] for i in self.indices ]
        self.weights = torch.tensor(weights, dtype=torch.double)
        if num_samples == 0:
            self.num_samples = len(self.weights)
        else:
            self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, self.replacement))

    def __len__(self):
        return self.num_samples
    
class Empty(ImageOnlyTransform):
    def apply(self, img, **params):
        return img 
    
    
def getDataLoader(df, image_folder, process, batch_size = 64, image_size = 224, train_weights = None, replacement = True, option_da = []):
    if process == 'train':      
        trnsfms = albumentations.Compose([
                    albumentations.Resize(height=image_size, width=image_size),
                    RGBToGray(always_apply = True) if 'gray' in option_da else Empty(),
                    albumentations.HorizontalFlip(),
                    albumentations.OneOf([
                        albumentations.RandomContrast(),
                        albumentations.RandomBrightness(),
                    ]),
                    albumentations.ShiftScaleRotate(rotate_limit=20, scale_limit=0.2),
                    albumentations.JpegCompression(80),
                    albumentations.HueSaturationValue(),
                    albumentations.Normalize(),
                    AT.ToTensor()
                ])
    elif process == 'val' or process =='test':
        trnsfms = albumentations.Compose([
                    albumentations.Resize(height=image_size, width=image_size),
                    RGBToGray(always_apply = True) if 'gray' in option_da else Empty(),
                    albumentations.Normalize(),
                    AT.ToTensor()
                ])
            
    dataset = WhaleDataLoader(image_folder = image_folder, process=process, df=df, transform=trnsfms)
    if process == 'train':  
        tr_ind = np.arange(0, df.shape[0], 1)
        train_sampler = WeightedSubsetRandomSampler(tr_ind, train_weights, replacement = replacement)
        loader  = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=train_sampler)
    else:
        loader  = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False)
    
    return loader
