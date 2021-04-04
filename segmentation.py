import cv2
import numpy as np


# 反相灰度图，将黑白阈值颠倒
def accessPiexl(img):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
       for j in range(width):
           img[i][j] = 255 - img[i][j]
    return img


# 反相二值化图像
def accessBinary(img, threshold=128):
    img = accessPiexl(img)
    # 边缘膨胀，不加也可以
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
    return img


# 根据长向量找出顶点
def extractPeek(array_vals, min_vals=10, min_rect=20):
    extrackPoints = []
    startPoint = None
    endPoint = None
    for i, point in enumerate(array_vals):
        if point > min_vals and startPoint == None:
            startPoint = i
        elif point < min_vals and startPoint != None:
            endPoint = i

        if startPoint != None and endPoint != None:
            extrackPoints.append((startPoint, endPoint))
            startPoint = None
            endPoint = None

    # 剔除一些噪点
    for point in extrackPoints:
        if point[1] - point[0] < min_rect:
            extrackPoints.remove(point)
    return extrackPoints


# 寻找边缘，返回边框的左上角和右下角（利用直方图寻找边缘算法（需行对齐））(废弃方案)
def findBorderHistogram(img):
    borders = []
    # 行扫描
    hori_vals = np.sum(img, axis=1)
    hori_points = extractPeek(hori_vals)
    # 根据每一行来扫描列
    for hori_point in hori_points:
        extractImg = img[hori_point[0]:hori_point[1], :]
        vec_vals = np.sum(extractImg, axis=0)
        vec_points = extractPeek(vec_vals, min_rect=0)
        for vect_point in vec_points:
            border = [(vect_point[0], hori_point[0]), (vect_point[1], hori_point[1])]
            borders.append(border)
    return borders


# 寻找边缘，返回边框的左上角和右下角（利用cv2.findContours）
def findBorderContours(img, minArea=100, maxArea=1000):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    borders = []
    for contour in contours:
        # 将边缘拟合成一个边框
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > minArea and w * h < maxArea:
            border = [(x, y), (x + w, y + h)]
            borders.append(border)
    return borders


# 显示结果及边框
def splitShow(img, borders, results=None):
    # 绘制
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        if results:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        # cv2.circle(img, border[0], 1, (0, 255, 0), 0)
    return img


# 分割出数字
def splitNum(img, borders):
    num_img = []
    for i, border in enumerate(borders):
        num_img.append(img[border[0][1]:border[1][1], border[0][0]:border[1][0]])
    return num_img


# 填充成正方形
def fillImg(num_img):
    for i, img in enumerate(num_img):
        height = img.shape[0]
        width = img.shape[1]
        if height>width:
            fill_len = int((height-width)/2)
            num_img[i] = cv2.copyMakeBorder(img, 0, 0, fill_len, fill_len, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            fill_len = int((width - height) / 2)
            num_img[i] = cv2.copyMakeBorder(img, fill_len, fill_len, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return num_img


# resize成28*28
def resize(num_img):
    for i, img in enumerate(num_img):
        num_img[i] = cv2.resize(img, (28, 28))
        num_img[i] = num_img[i][:, :, np.newaxis]  # 28, 28 to 28, 28, 1
    return num_img


# 主要调用函数
def segmentation(img):
    img = accessBinary(img)
    # cv2.imshow('accessBinary', img)
    # cv2.waitKey(0)
    # borders = findBorderHistogram(img)
    borders = findBorderContours(img)
    num_img = splitNum(img, borders)
    num_img = fillImg(num_img)
    num_img = resize(num_img)
    return borders, num_img


