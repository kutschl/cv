"""
Task 1: Distance Transform using Chamfer 5-7-11
Template for MA-INF 2201 Computer Vision WS25/26
Exercise 03
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def chamfer_distance_transform_5_7_11(binary_image):
    """
    Compute Chamfer distance transform using 5-7-11 mask.
    
    Based on Borgefors "Distance transformations in digital images" (1986).
    
    Chamfer 5-7-11:
    - Horizontal/vertical neighbors: weight = 5
    - Diagonal neighbors: weight = 7
    - Knight's move neighbors: weight = 11
    
    Args:
        binary_image: Binary image where features are 255, background is 0
    
    Returns:
        Distance transform image
    """
    H, W = binary_image.shape
    dt = np.full((H, W), np.inf, dtype=np.float32)
    
    # Initialize: 0 if feature pixel, infinity otherwise
    dt[binary_image > 0] = 0
    
    # Define forward and backward masks with (row_offset, col_offset, distance)
    # Forward mask (as shown in slide 37)
    forward_mask = [
        (-1,  0,  5),   # up
        ( 0, -1,  5),   # left
        (-1, -1,  7),   # up-left
        (-1, +1,  7),   # up-right
        (-2, -1, 11),   # 2 up, 1 left
        (-1, -2, 11),   # 1 up, 2 left
    ]
    
    
    # Backward mask (as shown in slide 37)
    backward_mask = [
        (+1,  0,  5),   # down
        ( 0, +1,  5),   # right
        (+1, +1,  7),   # down-right
        (+1, -1,  7),   # down-left
        (+1, +2, 11),   # 1 down, 2 right
        (+2, +1, 11),   # 2 down, 1 right
    ]
    
    # Forward pass
    for i in range(H):
        for j in range(W):
            for dr, dc, w in forward_mask:
                r, c = i + dr, j + dc
                if 0 <= r < H and 0 <= c < W:
                    cand = dt[r, c] + w
                    if cand < dt[i, j]:
                        dt[i, j] = cand

    # Backward pass
    for i in range(H - 1, -1, -1):
        for j in range(W - 1, -1, -1):
            for dr, dc, w in backward_mask:
                r, c = i + dr, j + dc
                if 0 <= r < H and 0 <= c < W:
                    cand = dt[r, c] + w
                    if cand < dt[i, j]:
                        dt[i, j] = cand
    
    
    
    return dt


def main():    
    
    print("=" * 70)
    print("Task 1: Distance Transform using Chamfer 5-7-11")
    print("=" * 70)
    
    name = 'coins'
    # img_path = 'data/bonn.jpg'
    # img_path = 'data/circle.png'      # play with different images
    # img_path = 'data/square.png'      
    # img_path = 'data/triangle.png'  
    img_path = f'data/{name}.jpg'  
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return
    
    # Load image and convert to grayscale
    img = cv2.imread(img_path)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(grayscale_img, 100, 200)
    
    # Compute distance transform with the function chamfer_distance_transform_5_7_11
    dt_chamfer = chamfer_distance_transform_5_7_11(edges)
    dt_chamfer_px = dt_chamfer / 5.0
    
    # Preprocess dt chamfer img for viz
    inv = cv2.bitwise_not(edges)  # Kanten=0, Rest=255
    dt_cv = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_5)

    def norm8(x):
        x = x.copy()
        x[np.isinf(x)] = 0
        if x.max() > 0:
            x = (255 * (x - x.min()) / (x.max() - x.min())).astype(np.uint8)
        else:
            x = np.zeros_like(x, dtype=np.uint8)
        return x

    # Compute distance transform using cv2.distanceTransform
    dt_img_cv = cv2.distanceTransform(grayscale_img, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_5)
    
    # Visualize results
    vis_chamfer = norm8(dt_chamfer_px)
    vis_cv = norm8(dt_cv)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(grayscale_img, cmap='gray')
    axes[0, 0].set_title("Original (Gray)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title("Canny Edges")
    axes[0, 1].axis('off')

    im3 = axes[1, 0].imshow(dt_chamfer_px, cmap='viridis')
    axes[1, 0].set_title("Chamfer 5-7-11 (in Pixel)")
    axes[1, 0].axis('off')
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(dt_cv, cmap='viridis')
    axes[1, 1].set_title("OpenCV DistanceTransform (L2, mask=5)")
    axes[1, 1].axis('off')
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # --------------------------------------------------------
    # Annotation: I couldnt use the show function in my environment. 
    # This is why I decided to store all plot results in directory 
    # Sheet03/results_q1. I hope this is okay. 
    # --------------------------------------------------------
    plt.tight_layout()
    plt.show()
    out_path = f"results_q1/{name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Figure saved to: {os.path.abspath(out_path)}")

    print("\n" + "=" * 70)
    print("Task 1 complete!")
    print("REMINDER: all plot results are stored in Sheet03/results_q1")
    print("=" * 70)


if __name__ == "__main__":
    main()