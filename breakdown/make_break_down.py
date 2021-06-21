import os

import cv2
import imutils
import numpy as np


def modify_contrast_and_brightness2(img, brightness=0, contrast=100):
    # 上面做法的問題：有做到對比增強，白的的確更白了。
    # 但沒有實現「黑的更黑」的效果
    import math

    brightness = 0
    contrast = 0  # - 減少對比度/+ 增加對比度

    B = brightness / 255.0
    c = contrast / 255.0
    k = math.tan((45 + 44 * c) / 180 * math.pi)

    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)

    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


":param"
# vid_path = r'./out_fancy.mp4'
vid_path = r'./moving 5_Trim.mp4'

cap = cv2.VideoCapture(vid_path)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

":param"
out = cv2.VideoWriter(r'./temp_breakdown.avi', cv2.VideoWriter_fourcc(*'XVID'), int(fps),
                      (int(width), int(height)))
# out = cv2.VideoWriter(r'./moving 5_Trim_breakdown.avi', cv2.VideoWriter_fourcc(*'XVID'), int(fps),
#                       (640, 360))
":param"
break_down_path = r'./makeof/moving'
cover_path = r'./makeof/moving/cover'

":param"
start_index = 0
end_index = 6
start2_index = 1
end2_index = 17

":param :: find frame index"
# base = cv2.imread(r'./makeof/magic/13.png')
# minerr = np.inf

count = 0
# linmit = 7 * fps
start = 0 * fps
end = 100 * fps
record = False
while (True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.resize(frame, (640, 360))
    # 顯示圖片

    count += 1
    if count > start and not record:
        print("Record")
        record = True
    if not record:
        continue
    # l1 = np.sum(np.abs(base - frame))
    # if l1 < minerr:
    #     print(count, minerr)
    #     minerr = l1
    #     cv2.imwrite("frame.jpg", frame)
    # frame = modify_contrast_and_brightness2(frame)
    # frame = np.transpose(frame, (1, 0, 2))
    # frame = cv2.flip(frame, 1)
    if count == 21:

        temp = cv2.imread(os.path.join(cover_path, str(1) + '.png'))
        base = cv2.imread('base.png')
        base = (base /255).astype(np.uint8)

        points = np.array([[0, 0], [temp.shape[1], 0], [temp.shape[1], temp.shape[0]], [0, temp.shape[0]]])
        del temp
        points_anchor = np.array([[553, 153],
                                  [1137, 48],
                                  [1336, 909],
                                  [745, 1040]])
        H, _ = cv2.findHomography(srcPoints=points, dstPoints=points_anchor)

        background_frame = frame.copy()
        one = np.ones(background_frame.shape)
        one *= 255
        background_frame = base * one + (1 - base) * background_frame
        sec = 0.8
        acclerate = int((width - 8) / (fps * sec))

        for i in range(start2_index, end2_index):
            breakdown_source = cv2.imread(os.path.join(cover_path, str(i) + '.png'))
            breakdown_frame = cv2.warpPerspective(breakdown_source, H, (frame.shape[1], frame.shape[0]))

            mask_ = breakdown_frame.copy()
            mask_[mask_ > 0] = 1

            kernel = np.ones((7, 7), np.uint8)
            mask_ = cv2.dilate(mask_, kernel, iterations=1)
            mask_ = cv2.erode(mask_, kernel, iterations=1)

            breakdown_frame = breakdown_frame * mask_ + background_frame * (1 - mask_)
            # cv2.imwrite("warp" + str(i) + ".png", breakdown_frame)

            for pointer in range(0, int(width - 8), acclerate):
                mask = np.zeros(background_frame.shape)
                mask[:, :pointer, :] = 1
                out_frame = breakdown_frame * mask + background_frame * (1 - mask)
                # out_frame[:, pointer:pointer + 1] =  15#[7, 56, 217]
                out_frame[:, pointer + 1:pointer + 3] = 240  # [54, 238,245]
                # out_frame[:, pointer + 3:pointer + 4] = 15#[7, 56, 217]
                out.write((out_frame).astype(np.uint8))
                cv2.imshow('frame', out_frame)
                # cv2.imshow('mask', breakdown_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # for i in range(3):
            #     out.write(breakdown_frame)
            #     cv2.imshow('frame', breakdown_frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            background_frame = breakdown_frame.copy()
    #
    #     # exit(0)
    #     # read cover
    #     # hography to background
    #
    #     # mask = processed.copy()
    #     # mask = mask[mask>0] = 1
    #     # same method with 126
    #126 229
    if count == 126:

        background_frame = frame.copy()
        sec = 0.9
        acclerate = int((width - 8) / (fps * sec))
        for i in range(start_index, end_index):
            breakdown_frame = cv2.imread(os.path.join(break_down_path, str(i) + '.png'))
            # if i == 0:
            #     base = cv2.imread('base.png')
            #     base = (base/255).astype(np.uint8)
            #     breakdown_frame = background_frame * (1 - base) + base * breakdown_frame
            print(i, breakdown_frame.shape)
            for pointer in range(0, int(width - 8), acclerate):
                mask = np.zeros(background_frame.shape)
                mask[:, :pointer, :] = 1
                out_frame = breakdown_frame * mask + background_frame * (1 - mask)
                # out_frame[:, pointer:pointer + 1] =  15#[7, 56, 217]
                out_frame[:, pointer + 1:pointer + 3] = 240  # [54, 238,245]
                # out_frame[:, pointer + 3:pointer + 4] = 15#[7, 56, 217]
                out.write((out_frame).astype(np.uint8))
                cv2.imshow('frame', out_frame)
                cv2.imshow('mask', breakdown_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            for i in range(3):
                out.write(breakdown_frame)
                cv2.imshow('frame', breakdown_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            background_frame = breakdown_frame.copy()

    out.write(frame)
    cv2.imshow('frame', frame)
    # cv2.imshow('base', base)
    # cv2.imwrite("test.jpg", frame)
    # print(count)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count > end:
        print("End")
        break
#
# exit(0)
cap.release()
out.release()
cv2.destroyAllWindows()
