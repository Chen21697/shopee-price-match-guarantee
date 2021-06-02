# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:58:38 2021

@author: s1972
"""
import os

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import spacy
import matplotlib.pyplot as plt
import sys
import tqdm as tq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import transformers
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from transformers import AutoModel, BertTokenizerFast
from transformers import AdamW

import re
import nltk
# uncomment the next line if you're running for the first time
# nltk.download('popular')

from customDataset import shopeeImageDataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
en = spacy.load('en_core_web_sm')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

parser = argparse.ArgumentParser()
parser.add_argument('--eval', dest='eval', action='store_true', default=False)
args = parser.parse_args()

#%%
def preprocess_data(directory):
    df = pd.read_csv(directory)
    
    # clean the title
    df2 = df['title'].apply(lambda x: preprocess_text(x, flg_stemm=False, flg_lemm=True))
    df.insert(4, "clean_title", df2)
    
    # transfomr the target columns and create a hashtable
    unique_targets = set(df['label_group'])
    dummy_list = list(range(0,len(unique_targets)))
    or_to_new = dict(zip(unique_targets, dummy_list))
    new_to_or = dict(zip(dummy_list, unique_targets))
    df['label_group'] = df['label_group'].map(or_to_new)
    
    df.to_csv('new_train.csv', index=False)
    
    return or_to_new, new_to_or

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

def accuracy(data_loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x1, x2, x3, y in data_loader:
            x1 = x1.to(dev)
            x2 = x2.to(dev)
            x3 = x3.to(dev)
            y = y.to(dev)
            
            scores = model(x1, x2, x3)
            _, preds = scores.max(1)
            print(preds == y)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        
        print('Training accuracy:', float(num_correct)/float(num_samples) *100)

def train(model, data_loader, optimizer, crit, epoch):
    model.train()
    epoch_loss = 0
    pbar = tq.tqdm(desc="Epoch {}".format(epoch), total=len(data_loader), unit="batch")
    
    for batch_idx, (image_data, text_seq, text_mask, targets) in enumerate(data_loader):
        image_data = image_data.to(dev)
        text_seq = text_seq.to(dev)
        text_mask = text_mask.to(dev)
        targets = targets.to(dev)
        
        model.zero_grad()
        preds = model(image_data, text_seq, text_mask)
        #print("targetssize", targets.size())
        #print("preds size", preds.size())
        
        loss = crit(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        pbar.update(1)
    
    pbar.close()
    avg_loss = epoch_loss / len(data_loader)
    return avg_loss
        
    
#%%
class bert_efficientNet(nn.Module):
    def __init__(self, bert, efficient_net):
        
        super().__init__()
        
        # two main models
        self.bert = bert
        self.efficient_net = efficient_net
        
        #TODO make sure it's the right one to use
        self.image_model_nums_ftrs = self.efficient_net._fc.in_features #1536, TODO
        
        self.drouput = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(768 + 1000, 2048)
        self.batch_normal = nn.BatchNorm1d(2048)
                
        self.fc2 = nn.Linear(2048, 11014)
        
    def forward(self, image, sent_id, mask):
        
        effi_output = self.efficient_net(image)
        bert_output = self.bert(sent_id, attention_mask = mask)
        
        # effi_output = torch.flatten(effi_output, 1)

        x = torch.cat((bert_output[1], effi_output), dim=1)
        
        x = self.fc1(x)
        x = self.batch_normal(x)
        x = F.relu(x)

        x = self.drouput(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        return x
    
#%%
if __name__ == "__main__":
    num_epochs = 10
    in_channel = 2
    batch_size = 16
    lr = 0.001
    
    directory = '../shopee/train.csv'
    or_to_new_hash, new_to_or_hash = preprocess_data(directory)

    
    my_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(), # range [0, 255] -> [0.0, 0.1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    
    
    # load images data
    images_dataset = shopeeImageDataset(csv_file = 'new_train.csv',
                                   root_dir = 'train_images',
                                   tokenizer = tokenizer,
                                   transform = my_transforms)
    
    images_train_set, images_test_set = torch.utils.data.random_split(images_dataset, [30000, 4250])
    train_loader = DataLoader(dataset = images_train_set, batch_size = batch_size, shuffle = True)
    test_loader =  DataLoader(dataset = images_test_set, batch_size = batch_size, shuffle = True)
        
    
    # import efficientnet b3 model
    image_model = EfficientNet.from_pretrained('efficientnet-b3')
    # import BERT-base pretrained model
    bert = AutoModel.from_pretrained('bert-base-uncased')
    
    image_model.to(dev)
    bert.to(dev)
    
    
    # freeze all the parameters
    for param in bert.parameters():
        param.requires_grad = False
    for param in image_model.parameters():
        param.requires_grad = False
        
    #sys.exit()
    model = bert_efficientNet(bert, image_model)
    model.to(dev)
    crit = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters())
    t_loss_history = []
    
    if not args.eval:
        print("Training the model")
    
        for epoch in range(num_epochs):
            t_loss = train(model, train_loader, optimizer, crit, epoch + 1)
            #accuracy(train_loader, model)
            t_loss_history.append(t_loss)
            print("Training loss:", t_loss)
            torch.save(model.state_dict(), "best-checkpoint.pt")
            
    model.load_state_dict(torch.load("best-checkpoint.pt"))
    print("\n")
    print("Running test evaluation:")
        

     