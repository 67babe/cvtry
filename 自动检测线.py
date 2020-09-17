import cv2
import numpy as np
# 回调函数，x表示滑块的位置，本例暂不使用
def nothing(x):
    pass
#cv2.createTrackbar()用来创建滑动条，
# 可以在回调函数中或使用cv2.getTrackbarPos()得到滑块的位置
img = cv2.imread('1.6mg.png', 0)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.createTrackbar('maxVal', 'image', 0, 255, nothing)
cv2.createTrackbar('minVal', 'image', 0, 255, nothing)

while(True):

    if cv2.waitKey(1) == 27:
        break
# 获取滑块的值
    maxVal = cv2.getTrackbarPos('maxVal', 'image')
    minVal = cv2.getTrackbarPos('minVal', 'image')
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(img, minVal, maxVal)  # canny边缘检测

    cv2.imshow('image', np.hstack((img, edges)))



