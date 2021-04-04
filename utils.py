import numpy as np
import torch


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print
        path + ' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print
        path + ' 目录已存在'
        return False

def num_img2tensor(num_img):
    for i, img in enumerate(num_img):  # N=len(num_img)
        img = np.transpose(img, (2, 0, 1))
        img = img[0]
        img = img / 255
        num_img[i] = img
    num_img = np.array(num_img)  # N, 28, 28
    num_img = num_img[:, np.newaxis, :, :]
    num_img_tensor = torch.from_numpy(num_img).float()
    return num_img_tensor