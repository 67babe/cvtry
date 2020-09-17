import numpy as np
from PIL import Image, ImageDraw
import argparse
import cv2


red2=np.uint8([[[130,128,143]]])
hsv_red2=cv2.cvtColor(red2,cv2.COLOR_BGR2HSV)
print(hsv_red2)