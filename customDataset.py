# -*- coding: utf-8 -*-
"""
Created on Wed May 19 00:06:36 2021

@author: s1972
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

#%%
class shopeeImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotation = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transfrom = transform
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotation['image'])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations['label_group']))
        
        if self.transform:
            image = self.tranform(image)
            
        return (image, y_label)