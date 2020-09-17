# import cv2
# img=cv2.imread("/Users/liuqi/Desktop/试纸例子.png")
# # cv2.namedWindow("试纸")
# cv2.imshow("shizhi", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
from PIL import Image,ImageDraw
p = Image.open("/Users/liuqi/Desktop/试纸例子.png")
# 注意位置顺序为左、上、右、下
cuts = [(100,485,137,490),(100,571,137,577),(100,523,137,528)]
for i,n in enumerate(cuts,1):
 temp = p.crop(n) # 调用crop函数进行切割
 temp.save("cut%s.png" % i)
p1=Image.open("cut1.png")
p2=Image.open("cut2.png")
p3=Image.open("cut3.png")
w1,h1=p1.size
# print(w1//2,h1//2)
r1, g1, b1 ,n1= p1.getpixel((w1//2,h1//2))  # 获取图片中心像素点RGB值
r2, g2, b2, n2= p2.getpixel((w1//2,h1//2))  # 获取图片中心像素点RGB值
r3, g3, b3 ,n3= p3.getpixel((w1//2,h1//2))  # 获取图片中心像素点RGB值
print(r1,g1,b1)
print(r2,g2,b2)
print(r3,g3,b3)
if(abs(r3-r1)>10 or abs(g3-g1)>10 or abs(b3-b1)>10):
 result="阳性"
else:
 result="阴性"

print("检测结果为:"+result)