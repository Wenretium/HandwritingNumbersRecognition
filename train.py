#train.py
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
from options import TrainOptions
import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import numpy as np
from utils import *
from LeNet_5 import LeNet_5
from time import localtime,strftime


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt, _ = TrainOptions()

# prepare
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
train_data = ImageFolder(opt.train_data_path,transform = transform)
train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)

MyLeNet = LeNet_5()
MyLeNet = MyLeNet.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(MyLeNet.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

# settings
start_EPOCH = 1

# # continue training
# MyLeNet = torch.load(opt.model_load_path+'Epoch10.pth')
# MyLeNet = MyLeNet.to(device)
# start_EPOCH = 11

# begin
print('Start:')
print('Find {} training imgs:'.format(len(train_data)))
time = strftime("%m.%d %H %M", localtime())

for epoch in range(start_EPOCH, opt.EPOCH + 1):
    for iteration, data in enumerate(train_loader):
        # print(data)
        # print(scheduler.get_last_lr()[0])
        img_train, label_train = data
        img_train, label_train = img_train.to(device), label_train.to(device)
        # output of model
        label_pred = MyLeNet(img_train).to(device)
        # process train label
        label_train = torch.tensor(label_train)
        # calculate loss
        # print(label_pred.shape, label_train.shape)
        # print(label_pred, label_train)
        loss = criterion(label_pred, label_train)
        # optimize model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch {} Loss at iteration {} :".format(epoch, iteration + 1), loss.item())

    if epoch > 0:
        new_model_save_path = opt.model_save_path+time+'/'
        mkdir(new_model_save_path)
        torch.save(MyLeNet, new_model_save_path + 'Epoch' + str(epoch) + '.pth')
        print('Saving epoch model')

    # scheduler.step()

print('Done!!!')

