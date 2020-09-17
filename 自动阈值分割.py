import cv2
import numpy as np
# 回调函数，x表示滑块的位置，本例暂不使用
def nothing(x):
    pass
#cv2.createTrackbar()用来创建滑动条，
# 可以在回调函数中或使用cv2.getTrackbarPos()得到滑块的位置
img = cv2.imread('0.2mg.png', 0)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(img, 30, 70)  # canny边缘检测

cv2.imshow('image', np.hstack((img, edges)))



cv2.waitKey(0)