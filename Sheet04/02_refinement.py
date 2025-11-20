import numpy as np
import cv2
import skimage
import matplotlib
import math

np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
viz = True
uav_file = 'data/img_mosaic.tif'
mask_file = 'data/img_mosaic_segment.png' # 'data/img_mosaic_step1_mask.png'

window_radius = 2 #3
optim_iterations = 3 # 2
conture_resampling = 3 # 2
alpha = 2 # 0.05
gamma = 5.0 # 1.00
beta = 0.05


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
    # viz imaging
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
    diffs = np.roll(points, -1, axis=0) - points   
    seg_lengths = np.linalg.norm(diffs, axis=1)  
    return float(np.mean(seg_lengths))           


def optimize_snake_triple_dp(snake_pts, grad_img, grad_norm, window_radius, alpha, gamma, beta):
    # optim
    N = len(snake_pts)
    windows = []
    windows_f = []
    unary = []
    
    
    for s in snake_pts:
        cx, cy = int(round(s[0])), int(round(s[1]))
        K = get_window_idx((cy, cx), window_radius, grad_img.shape)
        windows.append(K)
        windows_f.append(K.astype(np.float32))
        U = -gamma * grad_norm[K[:,0], K[:,1]]
        unary.append(U)

    d = compute_mean_dist(snake_pts)

    K0 = windows_f[0]
    K1 = windows_f[1]
    
    sum_unary = unary[0][:, None] + unary[1][None, :]
    diff01 = K1[None, :, :] - K0[:, None, :]
    dist01 = np.linalg.norm(diff01, axis=2)
    
    E_el = alpha * (dist01 - d)**2
    C_prev = sum_unary + E_el

    backptr = []

    for i in range(2, N):
        K_im2 = windows_f[i-2]
        K_im1 = windows_f[i-1]
        K_i = windows_f[i]

        M_im2 = len(K_im2)
        M_im1 = len(K_im1)
        M_i = len(K_i)

        C_curr = np.full((M_im1, M_i), np.inf, dtype=np.float64)
        BP_i = np.zeros((M_im1, M_i), dtype=np.int32)

        for j in range(M_im1):
            v_im1 = K_im1[j]
            prev_cost = C_prev[:, j]

            diff_vi = K_i - v_im1
            dist = np.linalg.norm(diff_vi, axis=1)
            unary_term = unary[i] + alpha * (dist - d)**2

            base_curv = K_i - 2.0 * v_im1
            curv = base_curv[:, None, :] + K_im2[None, :, :]
            curv_cost = beta * np.sum(curv * curv, axis=2)

            total = curv_cost + prev_cost[None, :]
            total += unary_term[:, None]

            best_l = np.argmin(total, axis=1)
            best_cost = total[np.arange(M_i), best_l]

            C_curr[j, :] = best_cost
            BP_i[j, :] = best_l.astype(np.int32)

        backptr.append(BP_i)
        C_prev = C_curr

    last_j, last_k = np.unravel_index(np.argmin(C_prev), C_prev.shape)
    idx = [None] * N
    idx[N-1] = last_k
    idx[N-2] = last_j

    for t in range(len(backptr)-1, -1, -1):
        i = t + 2
        BP_i = backptr[t]
        l = BP_i[idx[i-1], idx[i]]
        idx[i-2] = l

    new_snake = []
    
    for i in range(N):
        K_i = windows[i]
        k_i = idx[i]
        y, x = K_i[k_i]
        new_snake.append([x, y])

    new_snake = np.array(new_snake, dtype=np.float32)
    
    return new_snake


def run_optimization(cont_list: list, grad_img: np.ndarray, grad_norm: np.ndarray):
    run = 1
    print('grad_norm_min:', np.min(grad_norm), 'grad_norm_max:', np.max(grad_norm))
    snake_cont = []
    for cont in cont_list:
        print('optim_cont = ', run)
        run += 1
        
        # setup optim
        snake_pts = np.array(cont)[::conture_resampling].astype(np.float32)
        for _ in range(optim_iterations): 
            snake_pts = optimize_snake_triple_dp(
                snake_pts,
                grad_img,
                grad_norm,
                window_radius=window_radius,
                alpha=alpha,
                beta=beta,
                gamma=gamma
            )
        snake_cont.append(snake_pts)
    
    snake_mask = np.zeros(shape=grad_img.shape)
    for cont in snake_cont:
        cv2.polylines(snake_mask, [cont.astype(np.int32)], isClosed=True, color=255, thickness=1)
        
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
if False: 
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
    # viz optim result
    uav_img_bgr[:,:,1] = snake_mask
    # viz_image(uav_img_bgr)

    # export 
    cv2.imwrite('data/img_mosaic_snake.png', snake_mask)
    # export display image 
    snake_mask = cv2.morphologyEx(snake_mask, cv2.MORPH_CLOSE, (7,7)) 
    cv2.imwrite('data/img_mosaic_snake_display.png', snake_mask)