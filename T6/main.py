import cv2
import numpy as np

cropping = False

video = cv2.VideoCapture("tracking.mp4")
_, first_frame = video.read()

clone = first_frame.copy()

r = cv2.selectROI(clone)

if len(r) == 4:
	roi = clone[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
	cv2.imshow("ROI", roi)

x = r[0]
y = r[1]
width = r[2]
height = r[3]

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [255], [10, 170])
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 170, cv2.NORM_MINMAX)
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 25)

while True:
    _, frame = video.read()
    if _ == False:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([hsv], [0], roi_hist, [10, 170], 1)
    _, track_window = cv2.meanShift(mask, (x, y, width, height), term_criteria)
    x, y, w, h = track_window
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(20)
    if key == 27:
        break
video.release()
