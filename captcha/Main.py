import cv2
import numpy as np

raw_img = cv2.imread('./captcha/vpp.png')
img=cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(img,1,255,cv2.THRESH_OTSU)

kernel = np.ones([3,3])
img=cv2.dilate(img,kernel)

# img = cv2.erode(img,kernel)
contours, _ =cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
areas=[]
for c in contours:
    area = cv2.contourArea(c)
    areas.append(area)
areas = np.array(areas)
index = np.argsort(areas)[-5:-1]
print(index)
topcontours =[]
for i in range(4):
    topcontours.append(contours[index[i]])

for c in topcontours:
    x,y,w,h =cv2.boundingRect(c)
    cv2.rectangle(raw_img,[x,y],[x+w,y+h],[0,0,255],1)
cv2.imshow('vpp' ,raw_img)
cv2.waitKey(0)