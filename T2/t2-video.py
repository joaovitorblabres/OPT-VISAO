import numpy as np
import time
import cv2
import os

cap = cv2.VideoCapture('WalkByShop1front.mpg')
i = 0
frames = []
BWimg = []

if os.path.exists('./imgs') == False:
    os.mkdir('./imgs')

### READING VIDEO
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    BWimg.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    i += 1

### EDITING VIDEO
avg2 = np.float32(BWimg[1])
i = 0
kernel = np.ones((3,3),np.uint8)
for img in BWimg:
    cv2.accumulateWeighted(img, avg2, 0.001)
    res2 = cv2.convertScaleAbs(avg2)

    k = cv2.waitKey(5)

    ### BACKGROUND

    cv2.imshow('img', img)
    bg = cv2.convertScaleAbs(avg2)
    imgDif = cv2.absdiff(img, bg)
    ret2, th2 = cv2.threshold(imgDif, 30, 255, cv2.THRESH_BINARY)

    ### MORPHOLOGY => thanks MORO, G.

    erosion = cv2.erode(th2, kernel,iterations = 2)
    dilation = cv2.dilate(erosion, kernel,iterations = 6)
    erosion1 = cv2.erode(dilation, kernel,iterations = 5)
    dilation1 = cv2.dilate(erosion1, kernel,iterations = 16)

    ### RECTANGLE

    x, y, w, h = cv2.boundingRect(dilation1)
    cv2.rectangle(img, (x, y), (x + w, y + h), 255, 1)
    cv2.imshow('out', img)
    cv2.imwrite('imgs/output_'+str(i)+'.jpg', img)
    i += 1

cap.release()
cv2.destroyAllWindows()
