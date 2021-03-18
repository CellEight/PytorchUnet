import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from glob import glob
from PIL import Image
import time

class CamVidDataset(Dataset):
    def __init__(self, image_path, label_path, transform, colour_map_path="./CamVid/label_colors.txt"):
        self.image_paths = self.getPaths(image_path)
        self.label_paths = self.getPaths(label_path)
        self.transform = transform
        self.colour_map_path = colour_map_path
        self.colour_map = self.getColourMap()

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        raw_mask = Image.open(self.label_paths[index])
        x, y = self.transform(image), self.toOneHot(raw_mask)
        return x,y
    
    def __len__(self):
        return len(self.image_paths)
    
    def getPaths(self, root):
        return [img_path for img_path in glob(root+"/*.png")]

    def toOneHot(self, raw_mask):
        raw_mask = np.array(raw_mask.getdata()).reshape(raw_mask.size[1],raw_mask.size[0],3)
        mask = torch.zeros(raw_mask.shape[0], raw_mask.shape[1],dtype=torch.int64)
        for i in range(raw_mask.shape[0]):
            for j in range(raw_mask.shape[1]):
                c = self.colour_map[raw_mask[i,j,0]][raw_mask[i,j,1]][raw_mask[i,j,2]]
                mask[i,j] = c
        return mask
    
    def getColourMap(self):
        col_map_file = pd.read_csv(self.colour_map_path, sep=' ', header=None)[[0,1,2]].to_numpy()
        self.col_num = col_map_file.shape[0]
        colour_map = [[[0]*256]*256]*256
        for i, rgb in enumerate(col_map_file):
            colour_map[rgb[0]][rgb[1]][rgb[2]] = i
        return colour_map

        
