import numpy as np
import random as rng
import time
import cv2
import os

cap = cv2.VideoCapture('SilhouetteJogger.mp4')
i = 0
frames = []
BWimg = []

if os.path.exists('./imgs') == False:
    os.mkdir('./imgs')


ret, frame = cap.read()
avg2 = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
frames = []
### READING VIDEO
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    frames.append(frame)
    BWimg.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cv2.accumulateWeighted(BWimg[i], avg2, 0.001)
    res2 = cv2.convertScaleAbs(avg2)

    i += 1
bg = cv2.convertScaleAbs(avg2)
### EDITING VIDEO
count = 0
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
for img in BWimg:
    k = cv2.waitKey(5)
    frameI = frames[count]

    b,g,r = cv2.split(frameI)
    rgb_img = cv2.merge([r,g,b])

    gray = cv2.cvtColor(frameI,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

    sure_bg = cv2.erode(closing,kernel,iterations=2)
    sure_bg = cv2.dilate(sure_bg,kernel,iterations=6)
    sure_bg = cv2.erode(sure_bg,kernel,iterations=5)
    sure_bg = cv2.dilate(sure_bg,kernel,iterations=16)

    dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)

    ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)

    markers = markers+1
    markers[unknown==255] = 0
    markers = cv2.watershed(frameI,markers)
    frameI[markers == -1] = [255,0,0]

    cv2.imshow('out', frameI)
    cv2.imwrite('imgs/output_'+str(count)+'.jpg', frameI)
    count += 1

cap.release()
cv2.destroyAllWindows()
