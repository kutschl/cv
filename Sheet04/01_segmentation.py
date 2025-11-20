import numpy as np
import cv2 as cv
import matplotlib
import skimage

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# import uav image
img = cv.imread('data/img_mosaic.tif')
print('image_type:', img.dtype)
print('image_shape:', img.shape)

# filtering and resize 
img = cv.resize(img, (0,0), fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
img = cv.bilateralFilter(img, 9, 75, 75)

# mean shift for pre clustering original: 25, 35
shifted = cv.pyrMeanShiftFiltering(img, sp=2, sr=1)

# color splitting for detecting orange/yellow roofs
hsv = cv.cvtColor(shifted, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv)
# h 0–179, S/V: 0–255 
lower_yellow = np.array([10, 45, 60])    #<- tuned 
upper_yellow = np.array([30, 125, 250]) #<- tuned 
roof_mask = cv.inRange(hsv, lower_yellow, upper_yellow)
_, roof_mask = cv.threshold(roof_mask, 127, 255, cv.THRESH_BINARY)

# filtering 
kernel = np.ones((7,7), np.uint8)
roof_mask = cv.morphologyEx(roof_mask, cv.MORPH_OPEN, kernel, iterations=1)

# export segmentation result
cv.imwrite('data/img_mosaic_segment.png', roof_mask)

# export display result
img_canny = cv.Canny(roof_mask, 100,200)
img_canny = cv.morphologyEx(img_canny, cv.MORPH_CLOSE, (7,7)) 
cv.imwrite('data/img_mosaic_segment_display.png', img_canny)

