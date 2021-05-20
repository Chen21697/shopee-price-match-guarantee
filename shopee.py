# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:58:38 2021

@author: s1972
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import spacy
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

import re
import nltk
# uncomment the next line if you're running for the first time
# nltk.download('popular')

from customDataset import shopeeImageDataset

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
def preprocess_data(directory):
    df = pd.read_csv(directory)
    
    # clean the title
    df2 = df['title'].apply(lambda x: preprocess_text(x, flg_stemm=False, flg_lemm=True))
    df.insert(4, "clean_title", df2)
    
    df.to_csv('new_train.csv', index=False)

def preprocess_text(text, flg_stemm=False, flg_lemm=True):
    lst_stopwords = nltk.corpus.stopwords.words("english")
    
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()    
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text

def test():
    df = pd.read_csv('../shopee/new_train.csv')
    print(df.iloc[0,1])

#%%
if __name__ == "__main__":
    num_epochs = 1
    in_channel = 2
    batch_size = 10
    lr = 0.001
    
    directory = '../shopee/train.csv'
    preprocess_data(directory)

    
    my_transforms = transforms.Compose([
       transforms.ToTensor(), # range [0, 255] -> [0.0, 0.1]
       transforms.Resize((64, 64)),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # load images data
    images_dataset = shopeeImageDataset(csv_file = 'new_train.csv',
                                   root_dir = 'train_images',
                                   transform = my_transforms)
    
    images_train_set, images_test_set = torch.utils.data.random_split(images_dataset, [30000, 4250])
    train_loader = DataLoader(dataset = images_train_set, batch_size = batch_size, shuffle = True)
    test_loader =  DataLoader(dataset = images_test_set, batch_size = batch_size, shuffle = True)
    
    # load pre-trained efficientnet as feature extraction
    image_model = EfficientNet.from_pretrained('efficientnet-b3')
    image_model.to(dev)
    
    for epoch in range(num_epochs):
        train_loss, test_loss = [], []
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(dev)
            target = targets.to(dev)
            
            feature = image_model.extract_features(data)
            print('done')

    
    
   
    
    