#-*-coding:utf-8-*-
import cv2
from datetime import datetime
import numpy as np


def test(img):
    moments = cv2.moments(img)
    humoments = cv2.HuMoments(moments)
    humoments = np.log(np.abs(humoments))  # 同样建议取对数
    return humoments


def dif_hu(f1, f2):
    f1m = []
    f2m = []
    c0 = 0
    c1 = 0
    for i in f1:
        c0 += i ** 2
    for i in f2:
        c1 += i ** 2
    for i in f1:
        f1m.append(i / np.sqrt(c0))
    for i in f2:
        f2m.append(i / np.sqrt(c1))
    dif = 0
    m = min(len(f1m), len(f2m))
    for i in range(m):
        res = (f1m[i] - f2m[i]) * (f1m[i] - f2m[i])
        dif = dif + res
    return 1 - np.sqrt(dif)


if __name__ == '__main__':
    t1 = datetime.now()
    fp = 'test/mn/mn1.png'
    img = cv2.imread(fp)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a = test(img_gray)
    fp1 = 'test/mn/mn4.png'
    img1 = cv2.imread(fp1)
    img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    b = test(img_gray1)
    print(a)
    print(b)
    print("Pixel comparison")
    print(dif_hu(a, b))
    contours1, hierarchy1 = cv2.findContours(img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv2.findContours(img_gray1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ret1 = cv2.matchShapes(contours1[0], contours2[0], 1, 0.0)
    print("Edge comparison")
    print(ret1)
    print(datetime.now() - t1)
