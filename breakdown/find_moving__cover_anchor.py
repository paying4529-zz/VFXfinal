import cv2
import os
import numpy as np


":param :: find frame index"
base = cv2.imread(r'warp10.png')
# base[base[:, :].any() != 0] = 255

cv2.imshow("base", base)
cv2.waitKey(0)
for i in range(base.shape[0]):
    for j in range(base.shape[1]):
       if  base[i, j].any() != 0:
           base[i, j] = [255, 255, 255]


cv2.imshow("base", base)
kernel = np.ones((5,5), np.uint8)
dilation = cv2.dilate(base, kernel, iterations = 1)
dilation = cv2.erode(dilation, kernel, iterations = 1)
cv2.imshow("er", dilation)
cv2.waitKey(0)
cv2.imwrite("base.png", dilation)


vid_path = r'./moving 5_Trim.mp4'
cap = cv2.VideoCapture(vid_path)
count = 0
while (True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()
    if not ret:
        break
    count+=1
    # frame = cv2.resize(frame, (640, 360))
    frame = cv2.flip(frame, 0)
    cv2.imshow("frame", frame)
    print(count)
    cv2.waitKey(0)
