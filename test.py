# test.py
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
from options import TestOptions
import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import numpy as np
from utils import *
from LeNet_5 import LeNet_5
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt, _ = TestOptions()

# prepare
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([28, 28]),
    transforms.ToTensor()
])
test_data = ImageFolder(opt.test_data_path,transform = transform)
test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=True)

MyLeNet = torch.load(opt.model_load_path+opt.model_load_name)
MyLeNet = MyLeNet.to(device)

criterion = torch.nn.MSELoss()

# begin
print('Start:')
print('Find {} testing imgs:'.format(len(test_data)))
print('--Testing model:', opt.model_load_name)

correct = 0
for iteration, data in enumerate(tqdm(test_loader)):
    img_test, label_test = data
    img_test, label_test = img_test.to(device), label_test.to(device)
    # output of model
    out = MyLeNet(img_test).cpu()
    # process pred label
    out = out.detach().numpy().tolist()
    label_pred = []
    for out_i in out:
        label_pred.append(out_i.index(max(out_i)))
    # calculate accuracy
    for i in range(len(label_test)):
        if label_pred[i]==label_test[i]:
            correct += 1

print('Accuracy : {}'.format(correct/len(test_data)))
print('Done!!!')


