import cv2
import os
import numpy as np


vid_paths = [r'./final/out_fancy_breakdown_Trim.mp4', r'./final/invisible6_Trim_breakdown_Trim.mp4', r'./final/moving_breakdown_Trim.mp4']

cap = cv2.VideoCapture(vid_paths[0])
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
":param"
out = cv2.VideoWriter(r'./breakdown.avi', cv2.VideoWriter_fourcc(*'XVID'), int(fps),
                      (int(width), int(height)))
del cap

for vid_path in vid_paths:
    cap = cv2.VideoCapture(vid_path)
    while (True):
        # 從攝影機擷取一張影像
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

cap.release()
out.release()
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()