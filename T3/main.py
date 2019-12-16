import cv2 as cv
import numpy as np
PI = 3.14159265359
def _gaussian(i, j, k, sigma):
	return (((i - (k+1))**2 + (j - (k+1))**2)/(2*sigma**2))*-1

def gaussian(k = 2, sigma = 1):
	g = np.empty((2*k+1,2*k+1))
	for i in range(0, 2*k+1):
		for j in range(0, 2*k+1):
			g[i][j] = (1/(2*PI*sigma**2))*np.exp(_gaussian(i, j, k, sigma))

	return g

###### ADAPTADAS DE https://github.com/FienSoP/canny_edge_detector/blob/master/canny_edge_detector.py ######
def non_max_suppression(img, theta):
    M, N = img.shape
    Z = np.zeros((M, N))
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255

               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z

def hysteresis(img, weak, strong = 255):
    M, N = img.shape

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

###### FIM DA ADAPTAÇÃO ######

img = cv.imread('./teste.jpg', 0)

cv.imshow('img', img)

## MEU GAUSSIAN
Gaus = cv.filter2D(img, -1, gaussian(1, 0.7))
#Gaus = cv.GaussianBlur(img,(3,3),0)

## FILTROS
abs_grad_x = cv.convertScaleAbs(cv.Sobel(Gaus, -1, 1, 0, ksize=3))
abs_grad_y = cv.convertScaleAbs(cv.Sobel(Gaus, -1, 0, 1, ksize=3))

theta = np.arctan2(abs_grad_y, abs_grad_x)
grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

## SUPRESSÃO DE NÃO MÁXIMOS
NMS = non_max_suppression(grad, theta)

## LIMIARIZAÇÃO
ret2, th = cv.threshold(NMS, 20, 255, cv.THRESH_BINARY)

## HISTERIA
img_final = hysteresis(th, ret2)

## CANNY DO OPENCV
cannyCV = cv.Canny(img, 100, 200)

cv.imshow('imgMY', img_final)
cv.imshow('imgCV', cannyCV)

cv.waitKey(0)
cv.destroyAllWindows()
