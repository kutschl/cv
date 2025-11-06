"""
Task 2: Hough Transform for Circle Detection
Task 3: Mean Shift for Peak Detection in Hough Accumulator
Template for MA-INF 2201 Computer Vision WS25/26
Exercise 03
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os


def myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz):
    """
    Your implementation of HoughCircles
    
    Args:
        edges: single-channel binary source image (e.g: edges)
        min_radius: minimum circle radius
        max_radius: maximum circle radius
        threshold: minimum number of votes to consider a detection
        min_dist: minimum distance between two centers of the detected circles. 
        r_ssz: stepsize of r (integer > 0)
        theta_ssz: stepsize of theta in degrees (integer > 0)
    return: list of detected circles as (a, b, r, v), accumulator as [n_radii, h, w]
    """
    # --- Setup ---
    h, w = edges.shape
    # clamp radius to image diagonal for safety
    max_radius = min(int(max_radius), int(np.hypot(h, w)))
    min_radius = max(1, int(min_radius))
    r_vals = np.arange(min_radius, max_radius + 1, int(max(r_ssz, 1)))
    n_r = len(r_vals)
    if n_r == 0:
        return [], np.zeros((0, h, w), dtype=np.int32)

    # Accumulator: [radius_index, y_center, x_center]
    accumulator = np.zeros((n_r, h, w), dtype=np.int32)

    # Edge points (y, x)
    ys, xs = np.nonzero(edges)  # consider any nonzero as edge

    if ys.size == 0:
        return [], accumulator

    # Precompute angles
    theta_step = max(int(theta_ssz), 1)
    thetas_deg = np.arange(0, 360, theta_step, dtype=np.float32)
    thetas_rad = np.deg2rad(thetas_deg)
    cos_t = np.cos(thetas_rad)
    sin_t = np.sin(thetas_rad)

    # --- Vote in accumulator ---
    # For each radius, cast votes for centers (a, b) along the circle of each edge point.
    for ridx, r in enumerate(r_vals):
        # centers: a = x - r*cos, b = y - r*sin
        # We vectorisieren über Winkel; pro Kante clippen wir auf Bildgrenzen:
        rc = r * cos_t
        rs = r * sin_t
        for y, x in zip(ys, xs):
            a = (x - rc).round().astype(np.int32)
            b = (y - rs).round().astype(np.int32)
            # keep only valid centers
            valid = (a >= 0) & (a < w) & (b >= 0) & (b < h)
            aa = a[valid]
            bb = b[valid]
            # vote
            accumulator[ridx, bb, aa] += 1

    # --- Peak picking (simple NMS on centers per (a,b,r)) ---
    # Kandidaten = alle Zellen mit votes >= threshold
    candidates = np.argwhere(accumulator >= int(threshold))
    if candidates.size == 0:
        return [], accumulator

    # Sort by vote desc
    votes = accumulator[candidates[:, 0], candidates[:, 1], candidates[:, 2]]
    order = np.argsort(-votes)
    candidates = candidates[order]
    votes = votes[order]

    detected = []
    centers_taken = []  # keep accepted centers to enforce min_dist

    # Greedy NMS über alle Radien gemeinsam (kannst du auch radiusweise machen, falls gewünscht)
    for (ridx, cy, cx), v in zip(candidates, votes):
        # Mindestabstand prüfen gegen bereits akzeptierte
        ok = True
        for (ax, ay, _r, _v) in centers_taken:
            if (cx - ax) * (cx - ax) + (cy - ay) * (cy - ay) < (min_dist * min_dist):
                ok = False
                break
        if not ok:
            continue
        # akzeptieren
        detected.append((int(cx), int(cy), int(r_vals[ridx]), int(v)))
        centers_taken.append((int(cx), int(cy), int(r_vals[ridx]), int(v)))

    return detected, accumulator

def myMeanShift(accumulator, bandwidth, threshold=None):
    """
    Find peaks in Hough accumulator using mean shift.
    
    Args:
        accumulator: 3D Hough accumulator (n_radii, h, w)
        bandwidth: Bandwidth for mean shift
        threshold: Minimum value to consider (if None, use fraction of max)
        
    Returns:
        peaks: List of (x, y, r_idx, value) tuples
    """
    n_r, h, w = accumulator.shape
    
    # TODO
    
    # return peaks

def main():
    
    print("=" * 70)
    print("Task 2: Hough Transform for Circle Detection")
    print("=" * 70)
        
    img_path = 'data/coins.jpg'
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return
        
    # Load image and convert to grayscale
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    cv2.imwrite('results_q2/edges.png', edges)
    # Detect circles - parameters tuned for coins image
    print("\nDetecting circles...")
    min_radius = 30 # <- min radius nicht weiter reduzieren
    max_radius = 80 # <- max radius auch nicht weiter erhöhen
    threshold = 20 # <- threhsold reduziren hat geholfen damit alle detcted wurden 
    min_dist = 55
    r_ssz = 2
    theta_ssz = 5 
    
    # TODO
    detected_circles, accumulator = myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz)

    # Visualize detected circles
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for (x, y, r, v) in detected_circles:
        circ = plt.Circle((x, y), r, color='lime', fill=False, linewidth=2)
        ax.add_patch(circ)
        ax.text(x - 10, y - 10, f"{r}px", color='yellow', fontsize=8, weight='bold')
    ax.set_title(f"Detected {len(detected_circles)} circles (Hough 5–7–11)")
    ax.axis('off')
    
    plt.show()
    out_path = f"results_q2/det_circles.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Figure saved to: {os.path.abspath(out_path)}")

    
    # Visualize accumulator slices
    n_r, H, W = accumulator.shape

    # Wähle 3–4 Radius-Ebenen, um zu zeigen, wie sich Votes verteilen
    r_indices_to_show = np.linspace(0, n_r - 1, 4, dtype=int)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for idx, ridx in enumerate(r_indices_to_show):
        axes[idx].imshow(accumulator[ridx, :, :], cmap='inferno')
        axes[idx].set_title(f"r = {min_radius + ridx * r_ssz}px")
        axes[idx].axis('off')
    plt.suptitle("Accumulator slices for selected radii")
    plt.tight_layout()
    plt.show()
    out_path = f"results_q2/acc_slice.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Figure saved to: {os.path.abspath(out_path)}")
    
    # Visualize peak radius
    radius_votes = accumulator.sum(axis=(1, 2))
    peak_r_idx = np.argmax(radius_votes)
    peak_r = min_radius + peak_r_idx * r_ssz

    plt.figure(figsize=(6, 4))
    plt.plot(min_radius + np.arange(n_r) * r_ssz, radius_votes, color='teal')
    plt.axvline(peak_r, color='red', linestyle='--', label=f"Peak radius ≈ {peak_r}px")
    plt.title("Vote strength per radius")
    plt.xlabel("Radius [px]")
    plt.ylabel("Sum of votes")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    out_path = f"results_q2/peak_radius.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Figure saved to: {os.path.abspath(out_path)}")

    print(f"[INFO] Most voted radius ≈ {peak_r}px (index {peak_r_idx})")
    
    print("\n" + "=" * 70)
    print("Parameter Analysis:")
    print("  - Canny thresholds affect edge quality and thus detection")
    # ...more analysis can be added here
    print("=" * 70)
    print("Task 2 complete!")
    print("=" * 70)


    # =============================================================
    print("=" * 70)
    print("Task 3: Mean Shift for Peak Detection in Hough Accumulator")
    print("=" * 70)

    print("Applying mean shift to find peaks...")
    # peaks = myMeanShift # TODO
    
    # Visualize corresponding circles on original image    
    # TODO
    
    print("\n" + "=" * 70)
    print("Bandwidth Parameter Analysis:")
    # ...more analysis can be added here
    print("=" * 70)
    print("Task 3 complete!")
    

if __name__ == "__main__":
    main()