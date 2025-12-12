import cv2
import numpy as np
import maxflow
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Configuration
file_name = 'scissors'
img_dir = 'images'
gt_dir = 'images-gt'
labels_dir = 'images-labels'

# Read image, groundtruth mask and scibble labels 
img_file = os.path.join('dataset', img_dir, f'{file_name}.jpg')
gt_file = os.path.join('dataset', gt_dir, f'{file_name}.png')
labels_file = os.path.join('dataset', labels_dir, f'{file_name}-anno.png')

img = cv2.imread(img_file, cv2.IMREAD_COLOR)
gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
labels = cv2.imread(labels_file, cv2.IMREAD_COLOR)

def viz(img):
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
class GraphCut():
    
    def __init__(self, img: np.ndarray, mask: np.ndarray):
        ### Import ##################################################
        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)
        
        # Convert input image to LAB color space 
        # L - intensity 0 -> 255 
        # A - green 0 -> 255 red
        # B - blue 0 -> 255 yellow
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_lab = img_lab.astype(np.uint8)
        
        # Create boolean foreground (white) and background mask (red)
        mask_fg = (mask[:,:,0] >= 200) & (mask[:,:,1] >= 200) & (mask[:,:,2] >= 200)
        mask_bg = (mask[:,:,0] <= 50) & (mask[:,:,1] <= 50) & (mask[:,:,2] >= 200)

        # Read pixel values from image with masks
        pixels_fg = img_lab[mask_fg==1].astype(np.float64)
        pixels_bg = img_lab[mask_bg==1].astype(np.float64)
        
        img_lab = img_lab.astype(np.float64)
        
        ### Learning Colors #########################################
        def learn_colors(X, n_components=3, covariance_type='diag'):
            gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    reg_covar=1e-3,        
                    init_params="kmeans",
                    max_iter=300,
                    random_state=0
            )
            gmm.fit(X)          
            return gmm
        
    
        gmm_fg = learn_colors(pixels_fg)
        gmm_bg = learn_colors(pixels_bg)
        
        ### Min Cut Graph ###########################################
        # Convention: source = fg, sink = bg 
        H, W = img_lab.shape[:2]
        g = maxflow.Graph[float](H*W, H*W*2)
        nodeids = g.add_grid_nodes((H,W))
        
        def add_unary():
            # add unarys to graph
            X = img_lab.reshape(-1, 3) 
            # D_FG = -gmm_fg.score_samples(X).reshape(H, W) 
            # D_BG = -gmm_bg.score_samples(X).reshape(H, W) 
            
            
            D_FG = -gmm_fg.score_samples(X).reshape(H, W)
            D_BG = -gmm_bg.score_samples(X).reshape(H, W)

            # shift unaries that they are non-negativ
            m = np.minimum(D_FG, D_BG)
            D_FG = D_FG - m
            D_BG = D_BG - m

            # safety clipping (optional)
            D_FG = np.maximum(D_FG, 0)
            D_BG = np.maximum(D_BG, 0)
            
            g.add_grid_tedges(nodeids, D_BG, D_FG) 
            
            # # add constrain for scribble pixels
            # INF = 1e9  

            # # fg-scribble must be on source side (foregound)
            # # => D_BG = INF         D_FG = 0  
            # # => cap_source=INF     cap_sink=0 
            # g.add_grid_tedges(nodeids[mask_fg], INF, 0) 

            # # bg-scribble must be on sink side (background)
            # # => D_BG = 0           D_FG = INF
            # # => cap_source=0     cap_sink=INF 
            # g.add_grid_tedges(nodeids[mask_bg], 0, INF) 
            
            INF = 1e9
            ys, xs = np.where(mask_fg)
            for y, x in zip(ys, xs):
                g.add_tedge(nodeids[y, x], INF, 0)

            ys, xs = np.where(mask_bg)
            for y, x in zip(ys, xs):
                g.add_tedge(nodeids[y, x], 0, INF)
                
            print("Unary mean:", np.mean(D_FG + D_BG))

        
        def compute_pair_cost(x, y, lam=40.0, sigma=80.0):
            # contrast sensitive potts term
            d2 = float(np.sum((x - y) ** 2))            # ||x-y||^2
            return lam * np.exp(-d2 / (2.0 * sigma**2)) # >=0
        
        def add_pairwise():
            H, W = nodeids.shape 
            
            # right 
            for y in range(H):
                for x in range(W-1): 
                    w = float(compute_pair_cost(img_lab[y, x], img_lab[y, x+1])) 
                    g.add_edge(nodeids[y, x], nodeids[y, x+1], w, w) 


            # down 
            for y in range(H-1): 
                for x in range(W): 
                    w = float(compute_pair_cost(img_lab[y, x], img_lab[y+1, x])) 
                    g.add_edge(nodeids[y, x], nodeids[y+1, x], w, w) 

        
            # down-right 
            for y in range(H-1): 
                for x in range(W-1): 
                    w = float(compute_pair_cost(img_lab[y, x], img_lab[y+1, x+1])) 
                    g.add_edge(nodeids[y, x], nodeids[y+1, x+1], w, w) 


            # down-left 
            for y in range(H-1): 
                for x in range(1, W): 
                    w = float(compute_pair_cost(img_lab[y, x], img_lab[y+1, x-1])) 
                    g.add_edge(nodeids[y, x], nodeids[y+1, x-1], w, w) 

        
        # Adding edges for unary and pairwise cost
        add_unary()
        add_pairwise()
        
        # Max flow optimization
        flow = g.maxflow() 
        segments = g.get_grid_segments(nodeids)  # bool array (H,W)
        mask = segments.astype(np.uint8) # 1=FG, 0=BG 
        mask_vis = (segments * 255).astype(np.uint8) 
        viz(mask_vis)
        
        w_test = []
        for _ in range(1000):
            y = np.random.randint(0, H-1)
            x = np.random.randint(0, W-1)
            w_test.append(compute_pair_cost(img_lab[y,x], img_lab[y,x+1]))
        print("w min/mean/max:", np.min(w_test), np.mean(w_test), np.max(w_test))
        
        



graph_cut = GraphCut(img, labels)
