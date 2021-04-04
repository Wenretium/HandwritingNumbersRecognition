import pandas as pd
from torch.utils.data import Dataset,DataLoader
import numpy
import torch
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, transform, path):
        self.path = path
        data = pd.read_csv(self.path+'/dmos.csv')
        image_col = ['dist_img']
        data_image = data.loc[:,image_col]
        data_image = self.path+'/images/'+data_image
        score_col = ['dmos']
        data_score = data.loc[:, score_col]

        # print(data_name)
        data_image_numpy = data_image.values
        data_score_numpy = data_score.values
        # print(data_image_numpy)
        self.Image = data_image_numpy
        self.Score = torch.from_numpy(data_score_numpy).float()
        self.len = len(data)
        self.transform = transform


    def __getitem__(self, index):
        img_data = Image.open(self.Image[index][0]).convert('RGB')
        img_data = self.transform(img_data)
        return img_data, self.Score[index]

    def __len__(self):
        return self.len

