import numpy as np
import cv2
import skimage
import matplotlib
import math

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
viz = True
uav_file = 'data/img_mosaic.tif'
mask_file = 'data/img_mosaic_step1_mask.png'

window_radius = 2 #3
optim_iterations = 5 # 2
conture_resampling = 3 # 2
alpha = 0.05 # 0.05
gamma = 50.0 # 1.00
beta = 0.001


def compute_gradient(img: np.ndarray):
    # prefiltering 
    img = cv2.GaussianBlur(img, (5,5), 0)
    # apply sobel filter
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    # magnitude 
    grad_mag2 = np.abs(gx * gx) + np.abs(gy * gy)
    # filtering
    grad_mag2 = cv2.GaussianBlur(grad_mag2, (5,5), 0)
    # normalization
    g_min = grad_mag2.min()
    g_max = grad_mag2.max()
    grad_norm = (grad_mag2 - g_min) / (g_max - g_min + 1e-6) 
    
    grad_img = (grad_norm*255).astype(np.uint8)
    return grad_img, grad_norm


def viz_image(img: np.ndarray):
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def import_grayscale_image(file: str):
    img = cv2.imread(file)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def conture_from_mask(mask: np.ndarray, min_area: int = 500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    clean = np.zeros_like(mask) 

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            valid_contours.append(cnt)
            cv2.drawContours(clean, [cnt], -1, 255, thickness=-1) 
    # debugging:
    # if valid_contours:
    #     contour_vis = img.copy()
    #     cv2.drawContours(contour_vis, valid_contours, -1, (0, 0, 255), thickness=2)
    #     if viz: 
    #         cv2.imshow('roof_contours', contour_vis)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    valid_contour_mask = np.zeros_like(mask)
    if valid_contours:
        cv2.drawContours(valid_contour_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
    
    # reshaping
    cont_list = []
    for cont in valid_contours:
        cont = np.array(cont)
        cont = cont.reshape(-1, 2) 
        cont_list.append(cont)
    return valid_contour_mask, cont_list



def get_window_idx(center, r, img_shape):
    cy, cx = center
    H, W = img_shape
    
    window_points = []
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            y = cy + dy
            x = cx + dx
            if 0 <= y < H and 0 <= x < W:
                window_points.append((y, x))
            else:
                pass 
    return np.array(window_points)

def get_window_mask(center, r, img_shape):
    cy, cx = center
    H, W = img_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            y = cy + dy
            x = cx + dx
            if 0 <= y < H and 0 <= x < W:
                mask[y, x] = 1

    return mask


def compute_mean_dist(points: np.ndarray):
    diffs = np.roll(points, -1, axis=0) - points   # v_{i+1} - v_i   (geschlossen)
    seg_lengths = np.linalg.norm(diffs, axis=1)  # Länge pro Segment
    return float(np.mean(seg_lengths))           # Mittelwert = d


def run_optimization(cont_list: list, grad_img: np.ndarray, grad_norm: np.ndarray):
    run = 1
    # grad_norm = compute_gradient(grad_img)
    print('grad_norm_min:', np.min(grad_norm), 'grad_norm_max:', np.max(grad_norm))
    snake_cont = []
    for cont in cont_list:
        print('optim_cont = ', run)
        run += 1
        
        # resampling of conture 
        snake_pts = np.array(cont)
        snake_pts = np.vstack([snake_pts, snake_pts[0]])
        snake_pts = snake_pts[::conture_resampling]
        
        # snake_pts = snake_pts.reshape(-1, 2) # reshape
        for iter in range(optim_iterations): 
            mean_d = compute_mean_dist(snake_pts)
            win_size = window_radius*2 + 1
            prev_K = None
            prev_C = None 
            curr_C = None
            curr_K = None 
            curr_P = None 
            curr_U = None
            B = np.zeros(shape=(len(snake_pts), win_size**2))
            
            for s_idx, s in enumerate(snake_pts):
                curr_K = get_window_idx((s[1], s[0]), window_radius, grad_img.shape)
                curr_C = np.zeros(shape=len(curr_K))
                if s_idx == 0:    
                    
                    for k_idx, k in enumerate(curr_K):
                        curr_U = -gamma * grad_norm[k[0]][k[1]] # current unary term
                        
                        curr_C[k_idx] = curr_U  # current cost
                        B[s_idx][k_idx] = k_idx   # backtracking
                else: 
                    N = len(snake_pts)-1
                    p  = snake_pts[:-1][(s_idx - 1) % N]   # v_{i-1}
                    pp = snake_pts[:-1][(s_idx - 2) % N]   # v_{i-2}
                    for k_idx, k in enumerate(curr_K):
                        
                        curr_U = -gamma *  grad_norm[k[0]][k[1]] # current unary term
                        # current pairwise term
                        dists = np.linalg.norm(k - prev_K, axis=1)
                        alpha_term = alpha * (dists - mean_d)**2  
                        curv_vec = k - 2*p + pp
                        beta_term = beta * np.sum(curv_vec**2)
                        curr_P = alpha_term + beta_term
                        c = curr_U + curr_P + prev_C
                        
                        curr_C[k_idx] = np.min(c)
                        opt_k_prev_idx = np.argmin(c)
                        # print(k_idx, opt_k_prev)
                        B[s_idx][k_idx] = opt_k_prev_idx # backtracking
                # print(np.min(curr_U))
                prev_K = curr_K.copy()
                prev_C = curr_C.copy()
                
            # backtracking 
            opt_s_idx = np.argmin(curr_C).astype(int)
            opt_path = []
            for i  in range(1, len(snake_pts)):
                s_idx = len(snake_pts) - i - 1
                s = snake_pts[s_idx]
                
                K = get_window_idx((s[1], s[0]), window_radius, grad_img.shape)
                x, y = K[opt_s_idx]
                
                opt_path.append([y,x])
                
                # next new snake point
                opt_s_idx = B[s_idx][opt_s_idx].astype(int)
                
            # snake points reverse 
            opt_path.reverse()                
            opt_path = np.array(opt_path)
            
            # update snake points
            # print(snake_pts)
            snake_pts = opt_path.copy()
            
            # interpolation 
            
            # print(snake_pts)
        snake_cont.append(snake_pts)
    
    # create mask with polylines
    snake_mask = np.zeros(shape=grad_img.shape)
    for cont in snake_cont:
        cv2.polylines(snake_mask, [cont], isClosed=True, color=255, thickness=1)
        
    return snake_mask, snake_cont
        

def mask_from_conture(cont: list, img_shape: tuple):
    mask = np.zeros(shape=img_shape)
    for c in cont: 
        for pt in c: 
            x, y = pt
            mask[y,x] = 255
    return mask
        
            
# load uav image as bgr
uav_img_bgr = cv2.imread(uav_file)
uav_img_bgr = cv2.resize(uav_img_bgr, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# load uav image as grayscale
uav_img = import_grayscale_image(uav_file)
uav_img = cv2.resize(uav_img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# load mask image and compute contures
mask_img = import_grayscale_image(mask_file)
mask_cont, cont = conture_from_mask(mask_img, 500)

# gradient image
grad_img, grad_norm = compute_gradient(uav_img_bgr[:,:,1])
# hacky idea : 
blur_mask = cv2.dilate(mask_cont, (7,7), iterations=10)
blur_mask = cv2.GaussianBlur(blur_mask, (9,9), 20, 0)
blur_mask = cv2.GaussianBlur(blur_mask, (9,9), 12, 0)
blur_mask = cv2.GaussianBlur(blur_mask, (9,9), 12, 0)
grad_img = (blur_mask/255 * grad_img).astype(np.uint8)
grad_norm = (blur_mask/255 * grad_norm).astype(np.float32)
if False:
    viz_image((grad_img).astype(np.uint8))

if True:
    # optimization
    snake_mask, snake_cont = run_optimization(cont, grad_img, grad_norm)

    output = np.zeros(shape=(mask_img.shape[0], mask_img.shape[1], 3))
    uav_img_bgr[:,:,0] = mask_img

    uav_img_bgr[:,:,1] = snake_mask


    viz_image(uav_img_bgr)


