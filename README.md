# 摸鱼日志：搭建一个简易版实时手写数字识别项目

Author：Wenretium

杨老师的手写数字识别是深度学习领域的经典开山之作，也是入门者的必备练手项目之一。学期初任务很轻，我就复现了一下。

### 效果展示

+ 1993年

![](imgs\3e4a-khmynua4591710.gif)

+ 复现版本

![](imgs\3e4a-khmynua4591710.gif)



### 1. 准备MNIST数据集

从[官方途径](http://yann.lecun.com/exdb/mnist/)下载，得到后缀为**.gz**的文件。

为了方便预览图片，我将数据集转为了**png**格式，参考[这篇博客](https://blog.csdn.net/haoji007/article/details/76998140)。也可以直接从gz文件读取tensor进行训练。

![1](imgs\配图\1.jpg)



### 2. 还原LeNet-5

注意原作中**C3**卷积并不是6个channel一起卷积的，而是轮流采用部分channel。（论文中解释这样做的原因：1）减少参数；2）这种不对称的组合连接方式有利于提取多种组合特征。）我复现时没有按照这个做法，只是直接粗暴**全连接**。

##### 2.1 网络结构图

![](imgs\配图\LeNet-5.jpg)

原作中C3连接方式：



##### 2.2 代码实现

```python
# LeNet_5.py
import torch
import numpy as np
import torch.nn as nn

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(800, 500)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        return x
```



### 3. 训练

使用自己的代码进行训练。

```python
# train.py
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
```



### 4. 测试

使用自己的代码进行测试。

```python
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
```

输出结果：

<img src="imgs\image-20210404140904092.png" alt="image-20210404140904092" style="zoom:80%;" />

模型训练效果随迭代次数的关系：

| Epoch | Accuracy |
| :---: | :------: |
|   8   |  0.9903  |
|   9   |  0.9905  |
|  10   |  0.9916  |
|  11   |  0.9901  |
|  12   |  0.9906  |
|  13   |  0.9885  |

结论：从**Epoch8**开始，准确率就维持在**99%**左右，不再上升了。



### 5. 制作demo

字符分割与预处理操作不是本文章的重点，所以暂不展开细讲。我使用了[这篇博客](https://blog.csdn.net/qq8993174/article/details/89081859)的方法（整理成segmentation.py），基于行列像素扫描的思想来进行分割，也推荐阅读原博客。

```python
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
```

```python
# demo_realtime.py
# 调用demo中RecognitionRealtime类
import cv2
from demo import RecognitionRealtime


# 手机摄像头
# url = 'http://admin:admin@xxxxxx/'
# 笔记本摄像头
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
RecognitionRealtime = RecognitionRealtime()
while(True):
    ret, frame = cap.read()
    result_img = RecognitionRealtime.recognition(frame)
    if result_img == []:
        cv2.imshow('Real Time Recognition', frame)
    else:
        cv2.imshow('Real Time Recognition', result_img)
    k = cv2.waitKey(30) & 0xff
    if k==27: # 按Esc退出
        break

cap.release()
cv2.destroyAllWindows()
```



### 6. 项目记录

训练时遇到一个问题：**训练损失很快就不再下降了**。经过排查，发现是开始的**学习率设得太大了**，神经元很快就死掉了。将学习率调小后，模型很快就拟合了。

我在用MNIST本身测试集测完效果后，兴冲冲地将自己的手写图像放进去识别，结果输出惨不忍睹！开始还以为模型泛化能力很差，仔细一想，是自己的**输入图像没有经过预处理**。MNIST里都是28*28的黑底白字图像，而输入测试图像是像素数较大的白底黑字图像，自然没有办法正常识别了。

然后，我尝试自己用opencv函数进行处理，颜色的问题倒是解决了，但是**高像素的图像resize成低像素的图像**，总会出现**笔画断开**的情况，非常影响识别。第二天我直接用了参考博客的分割方案，和单个数字拍照相比，分割得到的图像本身就比较小，resize后整体笔画都保留了下来，识别效果就很好了。

另外，原博客分割出的borders可能不是正方形，直接拉伸成正方形会使数字变形。于是，在其基础上，我增加了操作函数fillImg和resize，将分割出的矩形图像**填充成正方形**，并resize为模型需要的大小。

根据我测试时的经验，识别时最好把灯开到最亮，让纸面背景尽量白、减少坑坑洼洼阴影的产生。不然画面会产生较多**噪声**，程序将这些噪声也识别为数字，影响效果。（改进方向：增加模型，判断矩形内图像是数字还是噪声）

