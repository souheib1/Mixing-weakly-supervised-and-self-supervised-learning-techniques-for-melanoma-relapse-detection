
import json
import os,glob
from omegaconf import OmegaConf

from solo.args.umap import parse_args_umap
from solo.data.classification_dataloader import prepare_data
from solo.methods import METHODS
from solo.utils.auto_umap import OfflineUMAP

import math
import os
import random
import string
import time
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import umap
import wandb
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from solo.utils.misc import gather, omegaconf_select
from tqdm import tqdm


import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle
from timm.models.vision_transformer import _create_vision_transformer

class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)
        sample = {'input': img}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        img=transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
        return {'input': img} 
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset(csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=512, shuffle=False, num_workers=10, drop_last=False)
    return dataloader, len(transformed_dataset)

def main():
    args = parse_args_umap()
    device = "cuda:0"

    # build paths
    args_path = '/tsi/data_education/IMA206/2022-2023/VisioMel/BenMH/solo-learn/trained_models/byol/offline-be607b0d/args.json'
    ckpt_path = '/tsi/data_education/IMA206/2022-2023/VisioMel/BenMH/solo-learn/trained_models/byol/offline-be607b0d/byol-res18-offline-be607b0d-ep=44.ckpt'


    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)

    # build the model

    cfg = OmegaConf.create(method_args)

    model = (
        METHODS[method_args["method"]]
        .load_from_checkpoint(ckpt_path, strict=False, cfg=cfg)
        .backbone
    )
    # move model to the gpu
    model.cuda()
    model = model.to(device)
    model.eval()


    # model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, num_classes=0, img_size=256)
    # model = _create_vision_transformer("vit_small_patch16_224", pretrained=False, **model_kwargs)
    # model.load_state_dict(torch.load('/home/ids/ipp-6635/vit_small_jz/vit_small.pth'))
    # model = model.to(device)
    # model.eval()

    bags_path = os.path.join('/tsi/data_education/IMA206/2022-2023/VisioMel/data/train','*','*')
    n_classes = glob.glob(os.path.join('/tsi/data_education/IMA206/2022-2023/VisioMel/data/train','*'+os.path.sep))
    save_path = '/tsi/data_education/IMA206/2022-2023/VisioMel/BenMH/extract_byol'
    num_bags=len(bags_list)
    
    # prepare data
    Tensor = torch.FloatTensor
    for i in range(0, num_bags):
        feats_list = []
        csv_file_path = glob.glob(os.path.join(bags_list[i], '*.png'))
        dataloader, bag_size = bag_dataset(csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda() 
                feats = model(patches)
                feats_list.extend(feats.cpu().numpy())
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list)
            os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-3], bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
            df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-3], bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')
    
if __name__ == "__main__":
    main()

 
