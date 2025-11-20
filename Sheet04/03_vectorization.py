import numpy as np
import cv2
import skimage
import scipy
import time
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
viz = True
uav_file = 'data/img_mosaic.tif'
snake_file = 'data/img_mosaic_snake.png'
min_area = 1           # ignore tiny blobs
approx_epsilon = 4.0     # simplification for noisy contours
line_threshold = 15
min_line_length = 15
max_line_gap = 20

def viz_image(img: np.ndarray):
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def contours(mask: np.ndarray):
    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    clean = np.zeros_like(mask) 

    valid_contours = []
    for c in cont:
        area = cv2.contourArea(c)
        if area >= min_area:
            valid_contours.append(c)
            cv2.drawContours(clean, [c], -1, 255, thickness=-1) 
    # reshaping
    cont_list = []
    for cont in valid_contours:
        cont = np.array(cont)
        cont = cont.reshape(-1, 2) 
        cont_list.append(cont)
    return cont_list

def mask_from_contours(cont: list, img_shape):
    mask = np.zeros(shape=img_shape)
    for c in cont: 
        for pt in c: 
            x, y = pt
            mask[y,x] = 255
    return mask


def polygons(cont: list, mask: np.ndarray):
    approx_polys = []
    for cont in cont:
        cont_for_cv = cont.reshape((-1, 1, 2)).astype(np.float32)
        perimeter = cv2.arcLength(cont_for_cv, True)
        eps = max(1.0, 0.005 * perimeter)  # adaptive threshold, clamp to avoid zero
        simplified = cv2.approxPolyDP(cont_for_cv, eps, True).astype(np.int32)
        approx_polys.append(simplified)
    poly_mask = np.zeros_like(mask)
    cv2.drawContours(poly_mask, approx_polys, -1, 255, thickness=-1)

    return poly_mask, approx_polys


# read uav file 
uav_img = cv2.imread(uav_file)

# read snake mask
snake_img = cv2.imread(snake_file)
snake_mask = cv2.cvtColor(snake_img, cv2.COLOR_BGR2GRAY)

# filtering
snake_mask = cv2.morphologyEx(snake_mask, cv2.MORPH_CLOSE, (7,7))
if False: 
    viz_image(snake_mask)
    
# find contours
snake_cont = contours(snake_mask)
snake_cont_mask = mask_from_contours(snake_cont, snake_mask.shape)
print(len(snake_cont))
if True: 
    viz_image(snake_cont_mask)

# polygon
poly_mask, poly_list = polygons(snake_cont, snake_cont_mask)

if False:
    viz_image(poly_mask)

def extract_long_lines(mask: np.ndarray,
                       threshold: int = line_threshold,
                       min_length: int = min_line_length,
                       max_gap: int = max_line_gap):
    mask_u8 = mask.astype(np.uint8)
    line_mask = np.zeros_like(mask_u8)
    lines = cv2.HoughLinesP(mask_u8, 1, np.pi / 180,
                            threshold=threshold,
                            minLineLength=min_length,
                            maxLineGap=max_gap)
    if lines is not None:
        for (x1, y1, x2, y2) in lines[:, 0]:
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)
    return line_mask, lines

line_mask, lines = extract_long_lines(snake_cont_mask)
if lines is not None:
    print(f'found {len(lines)} line segments >= {min_line_length}px')
else:
    print('no line segments detected')
if True:
    viz_image(line_mask)
