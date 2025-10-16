"""
Exercise 0 for MA-INF 2201 Computer Vision WS25/26
Introduction to OpenCV - Template
Python 3.12, OpenCV 4.11, NumPy 2.3.3
Image: bonn.jpeg
"""

import cv2
import numpy as np
import random
import string
import time

# ============================================================================
# Exercise 1: Read and Display Image (0.5 Points)
# ============================================================================
def exercise1():
    """
    Read and display the image bonn.jpeg.
    Print the image dimensions and data type.
    """
    print("Exercise 1: Read and Display Image")
    
    img = cv2.imread("bonn.jpeg")
    
    if img is None:
        raise FileNotFoundError("bonn.jpeg could not be loaded")
    
    cv2.imshow("bonn.jpeg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if img.ndim == 2:
        height, width = img.shape
        channels = 1
    else:
        height, width, channels = img.shape
    print(f"Image dimensions: {height} x {width} x {channels}")
    # TODO: Print image data type
    print(f"Image data type: {img.dtype}")
    
    
    print("Exercise 1 completed!\n")
    return img


# ============================================================================
# Exercise 2: HSV Color Space (0.5 Points)
# ============================================================================
def exercise2(img):
    """
    Convert image to HSV color space and display all three channels separately.
    """
    print("Exercise 2: HSV Color Space")
    
    # TODO: Convert to HSV using cv2.cvtColor() with cv2.COLOR_BGR2HSV
    hsv = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2HSV)
    
    # TODO: Split HSV into H, S, V channels using cv2.()
    h, s, v = cv2.split(hsv)
    
    # TODO: Display all three channels
    cv2.imshow("channels h s v", cv2.hconcat([h,s,v]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Hint: You can concatenate them horizontally using cv2.hconcat()
    
    print("Exercise 2 completed!\n")
    return hsv


# ============================================================================
# Exercise 3: Brightness Adjustment with Loops (1 Point)
# ============================================================================
def exercise3(img):
    """
    Add 50 to all pixel values and clip to [0, 255] using nested for-loops.
    Display original and brightened images side by side.
    """
    print("Exercise 3: Brightness Adjustment with Loops")
    
    # TODO: Create a copy of the image
    result = img.copy()
    
    # TODO: Get image dimensions
    dim = result.shape
    print(f'Image dimensions: {dim}')
    
    # TODO: Use nested for-loops to iterate through each pixel, add 50 to pixel value, and clip pixel value to [0, 255]
    for i in np.arange(dim[0]):
        for j in np.arange(dim[1]):
            result[i][j] = np.clip(a_min=0, a=img[i][j]+50, a_max=255)
            
    # TODO: Display original and result side by side
    cv2.imshow('Left: Original, Right: Brightness Adjustment', cv2.hconcat([img, result]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Exercise 3 completed!\n")
    return result


# ============================================================================
# Exercise 4: Vectorized Brightness Adjustment (1 Points)
# ============================================================================
def exercise4(img):
    """
    Perform the same brightness adjustment using NumPy in one line.
    Compare execution time with loop-based approach.
    """
    print("Exercise 4: Vectorized Brightness Adjustment")
    
    # TODO: Time the loop-based approach (from exercise 3)
    start_time_loop = time.time()
    result1 = img.copy()
    dim = result1.shape
    for i in np.arange(dim[0]):
        for j in np.arange(dim[1]):
            result1[i][j] = np.clip(a_min=0, a=img[i][j]+50, a_max=255)
    end_time_loop = time.time()
    
    # TODO: Time the vectorized approach
    start_time_vec = time.time()
    result2 = np.clip(img + 50, 0, 255)
    end_time_vec = time.time()
    
    # TODO: Print execution times
    print(f"Loop-based approach: {end_time_loop - start_time_loop:.4f} seconds")
    print(f"Vectorized approach: {end_time_vec - start_time_vec:.4f} seconds")
    
    cv2.imshow('Left: Loop, Right: Vector', cv2.hconcat([result1, result2]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Exercise 4 completed!\n")
    return result2


# ============================================================================
# Exercise 5: Extract and Paste Patch (0.5 Points)
# ============================================================================
def exercise5(img):
    """
    Extract a 32×32 patch from top-left corner and paste at 3 random locations.
    """
    print("Exercise 5: Extract and Paste Patch")
    
    # TODO: Extract 32x32 patch from top-left corner (starting at 0,0)
    patch_size = 32
    patch = img[0:patch_size, 0:patch_size].copy()
    
    # TODO: Create a copy of the image
    img_copy = img.copy()
    
    # TODO: Get image dimensions
    img_dim = img.shape
    print(img_dim)
    
    # TODO: Generate 3 random locations and paste the patch
    # Use random.randint() and ensure patch fits within boundaries
    for i in range(3):
        height = random.randint(0, img_dim[0] - patch_size)
        width = random.randint(0, img_dim[1] - patch_size)
        img_copy[height:height + patch_size, width:width + patch_size] = patch
    
    cv2.imshow("Patch Pasted", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Exercise 5 completed!\n")


# ============================================================================
# Exercise 6: Binary Masking (0.5 Points)
# ============================================================================
def exercise6(img):
    """
    Create masked version showing only bright regions.
    Convert to grayscale, threshold at 128, use as mask.
    """
    print("Exercise 6: Binary Masking")
    
    # TODO: Convert to grayscale using cv2.cvtColor() with cv2.COLOR_BGR2GRAY
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    
    # TODO: Apply binary threshold at value 128
    # Use cv2.threshold() with cv2.THRESH_BINARY
    _, mask = cv2.threshold(src=gray, thresh=128, maxval=255, type=cv2.THRESH_BINARY)
    
    # TODO: Apply mask to original color image
    # Hint: Use cv2.bitwise_and() with the mask
    masked = cv2.bitwise_and(src1=img, src2=img, mask=mask)
    
    cv2.imshow("Original, Mask, Masked", cv2.hconcat([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), masked]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Exercise 6 completed!\n")


# ============================================================================
# Exercise 7: Border and Annotations (1 Points)
# ============================================================================
def exercise7(img):
    """
    Add 20-pixel border and draw 5 circles and 5 text labels at random positions.
    """
    print("Exercise 7: Border and Annotations")
    
    bordered = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    height, width = bordered.shape[:2]
    
    max_radius = max(1, min(height, width) // 4)
    for _ in range(5):
        radius = random.randint(1, max_radius)
        center = (random.randint(0, width - 1), random.randint(0, height - 1))
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        thickness = random.randint(1, 4)
        cv2.circle(bordered, center, radius, color, thickness, lineType=cv2.LINE_AA)
    
    for _ in range(5):
        text = "".join(random.choices(string.ascii_uppercase, k=5))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = random.uniform(0.5, 1.5)
        thickness = random.randint(1, 3)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        max_x = max(0, width - text_size[0])
        org = (
            random.randint(0, max_x),
            random.randint(baseline, height - 1),
        )
        cv2.putText(bordered, text, org, font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    
    cv2.imshow("Exercise 7: Border and Annotations", bordered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Exercise 7 completed!\n")


# ============================================================================
# Main function
# ============================================================================
def main():
    """
    Run all exercises.
    """
    print("=" * 60)
    print("Exercise 0: Introduction to OpenCV")
    print("=" * 60 + "\n")
    
    # Uncomment the exercises you want to run:
    img = exercise1()
    if img is None:
        return
    exercise2(img)
    exercise3(img)
    exercise4(img)
    exercise5(img)
    exercise6(img)
    exercise7(img)
    
    print("=" * 60)
    print("All exercises completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
