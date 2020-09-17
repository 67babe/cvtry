import cv2
# 加载灰度图
img = cv2.imread('red.png', 1)
# cv2.imshow('baoliu', img)
# cv2.waitKey(0)
# cv2.imwrite('out.jpg',img)
px=img[1,1]
print(px)
print(img.size)
# b = img[:, :, 0]
cv2.imshow('red', img)
cv2.waitKey(0)
