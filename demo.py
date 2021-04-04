# demo.py
# 包含本地识别函数Recognition()和实时识别类RecognitionRealtime()
# 调用segmentation.py进行分割和预处理
from torchvision import transforms
import torchvision
import torch
import os
import torch.nn as nn
import numpy as np
from LeNet_5 import LeNet_5
from tqdm import tqdm
from PIL import Image, ImageOps
import glob
from options import TestOptions
import cv2
from segmentation import segmentation, splitShow
from utils import num_img2tensor


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt, _ = TestOptions()

def Recognition():
    # load model
    MyLeNet = torch.load(opt.model_load_path+opt.model_load_name)
    MyLeNet = MyLeNet.to(device)

    # start
    files = glob.glob('../Dataset/MNIST/my_number/t*.png')
    files.sort()
    for fn in files:
        # load images
        img_ori = cv2.imread(fn)
        img = cv2.imread(fn, 0)
        borders, img = segmentation(img)
        img = num_img2tensor(img)
        # output of model
        out = MyLeNet(img).to(device)
        # process pred label
        out = out.detach().numpy().tolist()
        result = []
        for out_i in out:
            number = (out_i.index(max(out_i)))
            result.append(number)
            print('Img {} Number : {}'.format(fn, number))
        splitShow(img_ori, borders, result)

    print('Done!!!')

class RecognitionRealtime():
    def __init__(self):
        # load model
        self.MyModel = torch.load(opt.model_load_path + opt.model_load_name)
        self.MyModel = self.MyModel.to(device)

    def recognition(self, img_ori):
        img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
        borders, img = segmentation(img)
        if borders != []:
            img = num_img2tensor(img).to(device)
            # output of model
            out = self.MyModel(img).cpu()
            # process pred label
            out = out.detach().numpy().tolist()
            result = []
            for out_i in out:
                number = (out_i.index(max(out_i)))
                result.append(number)
            return splitShow(img_ori, borders, result)
        else:
            return []

