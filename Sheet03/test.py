import cv2  
import numpy as np 
import os 

print(os.getcwd())
img_path = os.path.join(os.getcwd(), 'Sheet03', 'data', 'coins.jpg') 
img = cv2.imread(img_path, cv2.IMREAD_COLOR_BGR)
img = cv2.bilateralFilter(img, d=3, sigmaColor=1, sigmaSpace=1)
cv2.imshow('', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



img_canny = cv2.Canny(img, 100, 150)
cv2.imshow('', img_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dp=2
mode=cv2.HOUGH_GRADIENT
circles = None 
param1 = 200
param2 = 80
min_dist = 10
min_radius = 30
max_radius = 50
output = cv2.HoughCircles(gray, mode, dp, min_dist, circles, param1, param2, min_radius, max_radius)

if output is not None:
    output = np.uint16(np.around(output))
    for i in output[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
cv2.imshow('Detected Circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()