#encoding：utf-8
#黄色检测
import numpy as np
from PIL import Image,ImageDraw
from matplotlib import pyplot as plt
import argparse
import cv2
image = cv2.imread('02_1.6mg.png')
im = Image.open('02_1.6mg.png')
w,h=im.size #读取图片宽、高
# edges1 = cv2.Canny(image, 30, 70)  # canny边缘检测
# cv2.imshow('canny1', np.hstack((image, edges1)))
#
#
# _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# edges = cv2.Canny(image, 50, 60)  # canny边缘检测
# cv2.imshow('canny', np.hstack((image, edges)))
# cv2.waitKey(0)

#设置红色范围
# red=np.uint8([[[51,44,93]]])
# red2=np.uint8([[[120,117,132]]])
# hsv_red=cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
# hsv_red2=cv2.cvtColor(red2,cv2.COLOR_BGR2HSV)
# # print(hsv_red)
# # print(hsv_red2)
#
# #标准线的hsv范围
# lower_red=np.array([130,90,60])
# upper_red=np.array([210,210,190])
#
# #标准线和测试线的hsv范围
# # lower_red2=np.array([110,18,60])
# # upper_red2=np.array([210,210,190])
lower_red2=np.array([100,18,60])
upper_red2=np.array([210,210,190])
#将bgr转换成hsv
hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#介于范围之间的为白色，其余黑色
# mask=cv2.inRange(hsv,lower_red,upper_red)
mask2=cv2.inRange(hsv,lower_red2,upper_red2)

#
#
# res=cv2.bitwise_and(image,image,mask=mask)
res2=cv2.bitwise_and(image,image,mask=mask2)
# print(mask.shape)

# for x in range(w):#输出图片对象每个像素点的RBG值到array
#     for y in range(h):
#         r = mask[y,x]#获取当前像素点RGB值
#         if (r==255):
#             sum+=1
#
# print("标准线像素总数：",sum)
#
# sum2=0
# for x in range(w):#输出图片对象每个像素点的RBG值到array
#     for y in range(h):
#         r = mask2[y,x]#获取当前像素点RGB值
#         if (r==255):
#             sum2+=1
# print("标准线+测试线像素总数：",sum2)
#
# if(sum2-300>sum):
#     print("检测结果为阳性")
# else:
#     print("检测结果为阴性")


#显示
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',image)

# cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
# cv2.imshow('mask',mask)
# #
# cv2.namedWindow('mask2', cv2.WINDOW_NORMAL)
# cv2.imshow('mask2',mask2)

#
# cv2.namedWindow('res', cv2.WINDOW_NORMAL)
# cv2.imshow('res',res)
# cv2.imwrite('res.jpg', res)


cv2.namedWindow('res_02_1.6mg', cv2.WINDOW_NORMAL)
cv2.imshow('res_02_1.6mg',res2)
cv2.imwrite('res_02_1.6mg.jpg', res2)



# 灰度图读入
# img = cv2.imread('res2.jpg', 0)
# cv2.namedWindow('res3', cv2.WINDOW_NORMAL)
# cv2.imshow('res3',img)
# flag=1
# for y in range(h):#高
#     for x in range(w):#宽
#         r = img[y,x]#获取当前像素点RGB值
#         if (r>=10 and flag ==1):#找到测试线左上角
#             x1=x
#             y1=y
#             flag=0
#             print("x1=",x1)
#             print("y1=",y1)
#         if(flag==0):
#             for y2 in range(y1,h):
#                 for x2 in range(x1,w):  # 宽
#                     r1[]=img[y2,x2]



cv2.waitKey(0)