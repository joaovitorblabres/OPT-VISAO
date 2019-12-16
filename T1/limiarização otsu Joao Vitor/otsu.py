import numpy as np
import cv2
from matplotlib import pyplot as plt

def getWeight(ini, end, hist):
    return sum(hist[ini:end])

def getMu(ini, end, hist):
    soma = 0
    for i in range(ini, end):
        soma += (i+1)*hist[i]
    return soma

def getSigma(ini, end, hist, u, w):
    soma = 0
    for i in range(ini, end):
        soma += (((i+1)-u)**2)*hist[i]/w
    return soma

def getSigmaT(ini, end, hist, u):
    soma = 0
    for i in range(ini, end):
        soma += (((i+1)-u)**2)*hist[i]
    return soma

def limiarizacaoOtsu(hist):
    maxT = 0
    index = 0
    for k in range(1, 255):
        w0 = getWeight(0, k, hist)
        w1 = getWeight(k+1, 255, hist)
        if w1 == 0:
            continue
        u0 = getMu(0, k, hist)/w0
        u1 = getMu(k+1, 255, hist)/w1
        uL = getMu(0, 255, hist)

        sigma0 = getSigma(0, k, hist, u0, w0)
        sigma1 = getSigma(k+1, 255, hist, u1, w1)
        sigmaW = w0*sigma0 + w1*sigma1
        sigmaB = w0*w1*((u1-u0)**2)
        sigmaT = getSigmaT(0, 255, hist, uL)

        niK = sigmaB/sigmaT
        if maxT < niK:
            maxT = max(niK, maxT)
            index = k


    return index

# Load an color image in grayscale
img = cv2.imread('legolascage.jpg',0)
img2 = img.copy()
cv2.imshow('image',img)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
height, width = img.shape
N = height * width
p = hist/N
t = limiarizacaoOtsu(p)
for x in range(0, width):
    for y in range(0, height):
        if img2[y,x] < t:
            img2[y,x] = 0
        else:
            img2[y,x] = 255

#Mat newIgm(height, width, CV_8UC1, Scalar(0, 0, 0));
plt.hist(hist, 256, [0,256]);
cv2.imshow('image2',img2)
thresholdOtsu,opencvOtsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("OpenCV Otsu", opencvOtsu)
print(t, thresholdOtsu, t-thresholdOtsu)
plt.show()
#cv2.waitKey(0)
cv2.destroyAllWindows()
