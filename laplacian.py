import cv2
import numpy as np
import os

original = "E:\\Ubuntu\CycleGAN-TensorFlow-master\X\B\one11.png"
input = 'E:\\Ubuntu\CycleGAN-TensorFlow-master\X\B\one11_350000.png'
out_path = os.path.join('E:\\Ubuntu\CycleGAN-TensorFlow-master\X\B\\' + 'output_one11' + '.png')

a = cv2.imread('E:\\Ubuntu\CycleGAN-TensorFlow-master\X\B\one11.png')
b = cv2.imread('E:\\Ubuntu\CycleGAN-TensorFlow-master\X\B\one11_350000.png')
h,w = a.shape[:2]
print(h,w)
a1 = cv2.pyrDown(a)
h1,w1 = a1.shape[:2]
a2 = cv2.pyrDown(a1)
h2,w2 = a2.shape[:2]
a3 = cv2.pyrDown(a2)
h3,w3 = a3.shape[:2]
a4 = cv2.pyrDown(a3)
h4,w4 = a4.shape[:2]
print(a1.shape,a2.shape,a3.shape,a4.shape)
size = a4.shape[:2]
print(a.shape[:2])

d1 = a - cv2.resize(a1,(w,h),cv2.INTER_NEAREST)#cv2.pyrUp(A1)
d2 = a1 - cv2.resize(a2,(w1,h1),cv2.INTER_NEAREST)#cv2.pyrUp(A2)
d3 = a2 - cv2.resize(a3,(w2,h2),cv2.INTER_NEAREST)#cv2.pyrUp(A3)
d4 = a3 - cv2.resize(a4,(w3,h3),cv2.INTER_NEAREST)#cv2.pyrUp(A4)
"""
d1 = a - cv2.pyrUp(a1)
d2 = a1 - cv2.pyrUp(a2)
d3 = a2 - cv2.pyrUp(a3)
d4 = a3 - cv2.pyrUp(a4)
"""
print(d1.shape,d2.shape,d3.shape,d4.shape)
c1 = cv2.resize(b,(w3,h3),cv2.INTER_NEAREST)+d4


c2 = cv2.resize(c1,(w2,h2),cv2.INTER_NEAREST)+d3
c3 = cv2.resize(c2,(w1,h1),cv2.INTER_NEAREST)+d2
c4 = cv2.resize(c3,(w,h),cv2.INTER_NEAREST)+d1
"""
c1 = cv2.pyrUp()
c2 = cv2.pyrUp(c1)+d3
c3 = cv2.pyrUp(c2)+d2
c4 = cv2.pyrUp(c3)+d1
"""
print(c1.shape,c2.shape,c3.shape,c4.shape)
cv2.imwrite(out_path,c4)

