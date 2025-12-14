import cv2
import numpy as np
import maxflow
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

iters = 10 #5 
sigma = 10
lam = 10
visualize = False

def run_viz(img):
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def overlay_labels(img, labels_img, alpha=0.7):
    """
    Overlay red (BG) and white (FG) scribbles onto the original image.

    img_bgr: original image (H,W,3), uint8
    scribble_bgr: scribble image (H,W,3), uint8
    alpha: blending factor for scribbles
    """
    overlay = img.copy()

    # --- define scribble masks ---
    # white FG scribble
    mask_fg = (
        (labels_img[:, :, 0] > 200) &
        (labels_img[:, :, 1] > 200) &
        (labels_img[:, :, 2] > 200)
    )

    # red BG scribble (BGR!)
    mask_bg = (
        (labels_img[:, :, 0] < 80) &
        (labels_img[:, :, 1] < 80) &
        (labels_img[:, :, 2] > 150)
    )

    # --- draw colors ---
    fg_color = np.array([255, 255, 255], dtype=np.uint8)  # white
    bg_color = np.array([0, 0, 255], dtype=np.uint8)      # red (BGR)

    overlay[mask_fg] = (
        alpha * fg_color + (1 - alpha) * overlay[mask_fg]
    ).astype(np.uint8)

    overlay[mask_bg] = (
        alpha * bg_color + (1 - alpha) * overlay[mask_bg]
    ).astype(np.uint8)

    return overlay

    
    
def compute_iou_score(pred_mask, gt_mask): 
    # pred_mask, gt_mask: bool oder 0/1 oder 0/255, Shape (H,W) 
    pred = pred_mask.astype(bool) 
    gt   = gt_mask.astype(bool) 
    inter = np.logical_and(pred, gt).sum() 
    union = np.logical_or(pred, gt).sum() 
    return inter / union if union > 0 else 1.0  


def make_iou_overlay(pred_mask, gt_mask, gt_threshold=128, pred_threshold=128,
                     footer_px=40, font_scale=0.8, thickness=2):
    """
    pred_mask: (H,W) uint8 (0/255) oder bool
    gt_mask:   (H,W) uint8 (0/255) oder bool
    Ausgabe:   (H+footer_px, W, 3) BGR overlay image
    """

    # --- binarisieren ---
    if pred_mask.dtype == np.bool_:
        pred = pred_mask
    else:
        pred = pred_mask > pred_threshold

    if gt_mask.dtype == np.bool_:
        gt = gt_mask
    else:
        gt = gt_mask > gt_threshold

    # --- IoU ---
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = (inter / union) if union > 0 else 1.0

    H, W = gt.shape

    # --- Overlay (BGR) ---
    overlay = np.zeros((H, W, 3), dtype=np.uint8)

    # GT only -> green
    gt_only = gt & (~pred)
    overlay[gt_only] = (0, 255, 0)

    # Pred only -> red
    pred_only = pred & (~gt)
    overlay[pred_only] = (0, 0, 255)

    # Intersection -> white
    both = pred & gt
    overlay[both] = (255, 255, 255)

    # --- Footer drunter mit Text ---
    out = np.zeros((H + footer_px, W, 3), dtype=np.uint8)
    out[:H, :, :] = overlay

    text = f"IoU: {iou:.4f}"
    # leichter "Outline" für bessere Lesbarkeit
    org = (10, H + int(footer_px * 0.7))
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out, iou


def main(file_name: str):
    # Configuration
    # file_name = 'aero_2008_002358'
    img_dir = 'images'
    gt_dir = 'images-gt'
    labels_dir = 'images-labels'

    # Read image, groundtruth mask and scibble labels 
    img_file = os.path.join('dataset', img_dir, f'{file_name}.jpg')
    gt_file = os.path.join('dataset', gt_dir, f'{file_name}.png')
    labels_file = os.path.join('dataset', labels_dir, f'{file_name}-anno.png')

    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
    labels_img = cv2.imread(labels_file, cv2.IMREAD_COLOR)
    
    overlay_img = overlay_labels(img, labels_img)
    # viz(overlay_img)

    # Graph Cut Optimization
    graph_cut = GraphCut(img, labels_img)
    graph_cut.optimize(iters=iters)
    segmentation_img = graph_cut.mask_img.copy()

    # Compute IoU score + overlay image
    overlay_img, iou_score = make_iou_overlay(segmentation_img, gt_img)
    cv2.imwrite(os.path.join("dataset", "images-output", file_name + "_iou_overlay.png"), overlay_img)
    # iou_score = compute_iou_score(segmentation_img, gt_img)
    print(f'File: {file_name} \t IoU score: {iou_score}')

    # Write segmentation image 
    output_file = os.path.join('dataset', 'images-output', file_name + '.png')
    cv2.imwrite(output_file, segmentation_img)
    
    
    
class GraphCut():
    
    def __init__(self, img: np.ndarray, mask: np.ndarray):
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
        
        self.img_lab = img_lab
        self.pixels_fg = pixels_fg
        self.pixels_bg = pixels_bg
        self.mask_fg = mask_fg
        self.mask_fg_fixed = mask_fg
        self.mask_bg = mask_bg
        self.mask_bg_fixed = mask_bg
    
    def optimize(self, iters=5):
        
        print('Optimizing...')
        
        
        for i in range(iters):
            
            print(f'Iteration:  {i+1}/{iters}')

            ### Learning Colors #########################################
            def learn_colors(X, n_components=3, covariance_type='diag'):
                gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        reg_covar=1e-3,        
                        init_params="kmeans",
                        max_iter=1000,
                        random_state=0
                )
                gmm.fit(X)          
                return gmm
        
        
            gmm_fg = learn_colors(self.pixels_fg)
            gmm_bg = learn_colors(self.pixels_bg)
        
            ### Min Cut Graph ###########################################
            # Convention: source = fg, sink = bg 
            H, W = self.img_lab.shape[:2]
            g = maxflow.Graph[float](H*W, H*W*2)
            nodeids = g.add_grid_nodes((H,W))
            
            def add_unary():
                # add unarys to graph
                X = self.img_lab.reshape(-1, 3) 
                
                D_FG = -gmm_fg.score_samples(X).reshape(H, W)
                D_BG = -gmm_bg.score_samples(X).reshape(H, W)

                # shift unaries that they are non-negativ
                m = np.minimum(D_FG, D_BG)
                D_FG = D_FG - m
                D_BG = D_BG - m

                # safety clipping (optional)
                D_FG = np.maximum(D_FG, 0)
                D_BG = np.maximum(D_BG, 0)
                
                # add unaries
                g.add_grid_tedges(nodeids, D_BG, D_FG) 
                
                # add constraints for scribble pixels             
                INF = 1e9
                ys, xs = np.where(self.mask_fg)
                for y, x in zip(ys, xs):
                    g.add_tedge(nodeids[y, x], INF, 0)

                ys, xs = np.where(self.mask_bg)
                for y, x in zip(ys, xs):
                    g.add_tedge(nodeids[y, x], 0, INF)
              
            def compute_pair_cost(x, y, lam=lam, sigma=sigma):
                # contrast sensitive potts term
                #d2 = float(np.sum((x - y) ** 2))            
                #return lam * np.exp(-d2 / (2.0 * sigma**2)) 
                d = float(np.sum(np.abs(x[1:] - y[1:])))  # nur a,b
                return  lam * np.exp(-d / sigma)
            
                d = float(np.sum(np.abs(x - y)))
                return lam * np.exp(-d / sigma)

            def add_pairwise():
                img_lab = self.img_lab
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
        

            def keep_component_with_fg_scribble(pred_bool, mask_fg):
                pred_u8 = pred_bool.astype(np.uint8)
                num, cc = cv2.connectedComponents(pred_u8, connectivity=8)
                fg_labels = np.unique(cc[mask_fg])
                fg_labels = fg_labels[(fg_labels != 0)]
                if len(fg_labels) == 0:
                    return pred_u8
                best = max(fg_labels, key=lambda l: np.sum((cc == l) & mask_fg))
                return (cc == best).astype(np.uint8)

            segments = g.get_grid_segments(nodeids).astype(bool)

            # 1) Invertierung automatisch entscheiden
            if segments[self.mask_fg_fixed].mean() > 0.5 and segments[self.mask_bg_fixed].mean() < 0.5:
                pred = segments
            else:
                pred = ~segments

            # 2) Scribble-Constraints hart erzwingen
            pred[self.mask_fg_fixed] = True
            pred[self.mask_bg_fixed] = False

            # 3) Nur die Komponente behalten, die das FG-Scribble enthält
            pred = keep_component_with_fg_scribble(pred, self.mask_fg_fixed).astype(bool)

            # 4) Für Refit: nur den "Kern" des FG verwenden (gegen Drift)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            pred_core = cv2.erode(pred.astype(np.uint8), kernel, iterations=1).astype(bool)

            # 5) Trainingsdaten aktualisieren (Scribbles immer inkludieren)
            fg_train = np.vstack([
                self.img_lab[self.mask_fg_fixed].reshape(-1,3),
                self.img_lab[pred_core].reshape(-1,3)
            ]).astype(np.float64)

            bg_train = np.vstack([
                self.img_lab[self.mask_bg_fixed].reshape(-1,3),
                self.img_lab[~pred].reshape(-1,3)
            ]).astype(np.float64)

            self.pixels_fg = fg_train
            self.pixels_bg = bg_train

            mask_img = (pred.astype(np.uint8) * 255)


        
            # pred = (~segments).astype(bool)   # <-- HIER invertieren (falls nötig)
            
            # # --- harte Scribbles erzwingen ---
            # pred[self.mask_fg] = True
            # pred[self.mask_bg] = False

            # # --- neue Trainingsdaten sammeln ---
            # self.pixels_fg = self.img_lab[pred].reshape(-1, 3)
            # self.pixels_bg = self.img_lab[~pred].reshape(-1, 3)
            
            # # mask = keep_component_with_fg_scribble(pred, mask_fg)  # arbeitet jetzt auf dem richtigen FG
            # mask = pred.astype(np.uint8)
            # mask_img = (mask * 255).astype(np.uint8)  # NICHT mehr invertieren
            
        if visualize: 
            run_viz(mask_img)
        self.mask_img = mask_img
        
                
        
        # Debugging to find the right parametrization for pairwise cost function
        # w_test = []
        # for _ in range(1000):
        #     y = np.random.randint(0, H-1)
        #     x = np.random.randint(0, W-1)
        #     w_test.append(compute_pair_cost(img_lab[y,x], img_lab[y,x+1]))
        # print("w min/mean/max:", np.min(w_test), np.mean(w_test), np.max(w_test))
        
        







if __name__ == "__main__":
    file_names = [
        '106024', 
        '208001',
        'aero_2008_002358',
        'bike_2007_005878',
        'person7',
        'scissors'
    ]
    
    for file_name in file_names:
        main(file_name)


# Results
# ((.venv) ) lukas@lukas:~/cv/Sheet05$ /home/lukas/cv/.venv/bin/python /home/lukas/cv/Sheet05/graphcut_core.py
# File: 106024     IoU score: 0.3944009283434871
# ((.venv) ) lukas@lukas:~/cv/Sheet05$ /home/lukas/cv/.venv/bin/python /home/lukas/cv/Sheet05/graphcut_core.py
# File: 208001     IoU score: 0.7602256699576869
# ((.venv) ) lukas@lukas:~/cv/Sheet05$ /home/lukas/cv/.venv/bin/python /home/lukas/cv/Sheet05/graphcut_core.py
# File: aero_2008_002358   IoU score: 0.6934233724759655
# ((.venv) ) lukas@lukas:~/cv/Sheet05$ /home/lukas/cv/.venv/bin/python /home/lukas/cv/Sheet05/graphcut_core.py
# File: bike_2007_005878   IoU score: 0.8280284756133436
# ((.venv) ) lukas@lukas:~/cv/Sheet05$ /home/lukas/cv/.venv/bin/python /home/lukas/cv/Sheet05/graphcut_core.py
# File: person7    IoU score: 0.6026295436968291
# ((.venv) ) lukas@lukas:~/cv/Sheet05$ /home/lukas/cv/.venv/bin/python /home/lukas/cv/Sheet05/graphcut_core.py
# File: scissors   IoU score: 0.8277917972324672