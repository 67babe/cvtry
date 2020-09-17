import cv2
import numpy as np
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
# ⚠️将试纸颜色外的部分变成黑色
image = cv2.imread('02_0.8mg.png')
im = Image.open('02_0.8mg.png')
w,h=im.size #读取图片宽、高
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

cv2.imwrite('res_02_0.8mg.jpg', res2)


# ⚠️开始分割
img = cv2.imread('res_02_0.8mg.jpg')
cv2.imshow('fengehou', img)
img_2 = cv2.imread('02_0.8mg.png')
img_3=cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
cv2.imshow('yuantu', img_2)
im = Image.open('res_02_0.8mg.jpg')
w,h=im.size #读取图片宽、高
# plt.imshow(img)
# plt.show()
#转换成灰度图
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.namedWindow('huidutu', cv2.WINDOW_NORMAL)
cv2.imshow('huidutu',grayImg)
#固定阈值二值化
#binimg输出图，[100：阈值] [ 255: 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值][ cv2.THRESH_BINARY:二值化操作的类型]
# ret,binImg = cv2.threshold(grayImg,35, 255, cv2.THRESH_BINARY)
ret,binImg = cv2.threshold(grayImg,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('erzhihua',binImg)
#轮廓检测:
# binImg：源图像，8位单通道图像，非0像素值均按像素值为1处理，该函数在提取轮廓的过程中会改变图像
# contours：轮廓信息，每个轮廓由点集组成，而所有的轮廓构成了contours列表
# hierarchy：可选参数，表示轮廓的层次信息（拓扑信息，有种树结构的感觉），
    # 每个轮廓元素contours[i]对应4个hierarchy元素hierarchy[i][0]~hierarchy[i][3]，
    # 分别表示后一个轮廓、前一个轮廓、父轮廓和内嵌轮廓的编号，若无对应项，则该参数为负值
# mode：轮廓检索模式
    # cv2.RETR_EXTERNAL：只检测最外层轮廓，并置hierarchy[i][2]=hierarchy[i][3]=-1
    # cv2.RETR_LIST：提取所有轮廓并记录在列表中，轮廓之间无等级关系
    # cv2.RETR_CCOMP：提取所有轮廓并建立双层结构（顶层为连通域的外围轮廓，底层为孔的内层边界）
    # cv2.RETR_TREE：提取所有轮廓，并重新建立轮廓层次结构
#method：轮廓逼近方法
    # cv2.CHAIN_APPROX_NONE：获取每个轮廓的每个元素，相邻像素的位置差不超过1，即连续的点，但通常我们并不需要所有的点
    # cv2.CHAIN_APPROX_SIMPLE：压缩水平方向、垂直方向和对角线方向的元素，保留该方向的终点坐标，
    # 如矩形的轮廓可用4个角点表示，这是一种常用的方法，比第一种方法能得出更少的点
    # cv2.CHAIN_APPROX_TC89_L1和cv2.CHAIN_APPROX_TC89_KCOS：对应Tch-Chain链逼近算法
contours, hierarchy = cv2.findContours(binImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# 轮廓绘制
# cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]])
# image：目标输出图像
# contours：所有的轮廓，可直接传入findContours函数中的contours参数
# contourIdx：轮廓的索引，即绘制哪个轮廓，若为负值则绘制所有轮廓
# color：绘制的颜色，元祖形式，如（255）或（255, 255, 255）
# thickness：绘制的轮廓的线条粗细程度，若为负数，表示要填充整个轮廓
cv2.drawContours(img, contours,-1, (255, 255, 255), 1)
plt.imshow(img)
plt.show()
li=[]
li2=[]
for i in range(len(contours)):
    if len(contours[i])>10:
        li.append(i)
        print("轮廓 %d 的点集合有: %s" %(i,contours[i]))
# print(contours[2][0])
# print("len(contours[2])=")
# print(len(contours[2]))
# print(((contours[2][0])[0])[1])
# print("分割线")
x1,y1=((contours[0][0])[0])[0],((contours[0][0])[0])[1]
for j in li:
    i=0
    min_x = ((contours[j][0])[0])[0]
    min_y=((contours[j][0])[0])[1]
    max_x = ((contours[j][0])[0])[0]
    max_y=((contours[j][0])[0])[1]
    while(i<len(contours[j])):
        n=contours[j][i]
        # print(n)
        x=((n)[0])[0]
        y=((n)[0])[1]
        if x<=min_x:
            min_x=x
        if y<=min_y:
            min_y=y
        if x>=max_x:
            max_x=x
        if y>=max_y:
            max_y=y
        i+=1
    # print(min_x,min_y)
    # print(max_x,max_y)
    print("-----------")
    lu=[min_x,min_y,max_x,max_y]
    li2.extend(lu)
print(li2)


grayImg_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cut1 = grayImg[li2[1]:li2[3], li2[0]:li2[2]]
cut2 = grayImg[li2[5]:li2[7], li2[4]:li2[6]]
# cv2.imshow('cut0', cut0)
cv2.imshow('cut1', cut1)
cv2.imshow('cut2', cut2)
print("截取后的")
print(cut1.shape,cut2.shape)
w_1,h_1=cut1.shape#读取图片宽、高
print(w_1,h_1)
w_2,h_2=cut2.shape #读取图片宽、高
cut1_1=cut1[int(w_1/10*5):int(w_1/10*9),int(h_1/10):int(h_1/10*9)]
cut2_1=cut2[int(w_2/10*5):int(w_2/10*9),int(h_2/10):int(h_2/10*9)]
cv2.imshow('cut1_1', cut1_1)
cv2.imshow('cut2_1', cut2_1)
sum=0
num=0
for x in range(cut1_1.shape[0]):#输出图片对象每个像素点的RBG值到array
    for y in range(cut1_1.shape[1]):
        r = cut1_1[x,y]#获取当前像素点RGB值
        if(r>10 and r<255):
            num+=1
            sum+=r
mean1=sum/num
print("num=",num)
print("mean1=",mean1)

sum=0
num=0
for x in range(cut2_1.shape[0]):#输出图片对象每个像素点的RBG值到array
    for y in range(cut2_1.shape[1]):
        r = cut2_1[x,y]#获取当前像素点RGB值
        if (r>10):
            num+=1
            sum+=r
mean2=sum/num
print("num=",num)
print("mean2=",mean2)

print("差值=",abs(mean2-mean1))
# cv2.imshow('yuantu2', img_2)
print(8//10)
ori_cut1 = img_2[li2[1]:li2[3], li2[0]:li2[2]]
ori_cut2 = img_2[li2[5]:li2[7], li2[4]:li2[6]]
cv2.imshow('ori1',ori_cut1)
cv2.imshow('ori2',ori_cut2)
# cv2.waitKey(0)

# cv2.namedWindow('lunkuo', cv2.WINDOW_NORMAL)
cv2.imshow('lunkuo',img)
# print("最后最后")
# print(img.shape)
# cv2.namedWindow('灰度图', cv2.WINDOW_NORMAL)
# cv2.imshow('灰度图',grayImg)
# cv2.waitKey(0)
cv2.waitKey(0)