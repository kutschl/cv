import cv2
import numpy as np
import time

# ==============================================================================
# 0. Setup: Loading Image and Converting to Grayscale
# ==============================================================================
print("--- 0. Setup: Loading Image and Converting to Grayscale ---")

'''
TODO: Load the image 'bonn.jpg' and convert it to grayscale
'''
# mat = np.array([[3,1,5,4,2],[2,5,1,3,4],[7,1,3,2,1],[2,3,5,1,4],[3,4,1,6,8]])

# Load image and convert to grayscale
original_img_color = cv2.imread('bonn.jpg')  # Load 'bonn.jpg'
# cv2.imshow("Original Image", original_img_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


gray_img = cv2.cvtColor(src=original_img_color, code=cv2.COLOR_BGR2GRAY)            # Convert to grayscale
# cv2.imshow("gray_img", gray_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(f"Image loaded successfully. Size: {gray_img.shape}")

# ==============================================================================
# 1. Calculate Integral Image (Part a)
# ==============================================================================
print("\n--- a) Calculating Integral Image ---")


def calculate_integral_image(img):
    """
    Calculate the integral image (summed area table).
    Each pixel contains the sum of all pixels above and to the left.
    
    Args:
        img: Input grayscale image
    
    Returns:
        Integral image with dimensions (height+1, width+1)
    
    TODO:
    1. Create an integral image array     
    2. Iterate through all pixels and compute integral values
    """
    int_img_arr = np.zeros(shape=(img.shape[0]+1, img.shape[1]+1), dtype=np.float64)
    
    # row, col
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            int_img_arr[r+1][c+1] = int_img_arr[r+1][c] + int_img_arr[r][c+1] - int_img_arr[r][c] + img[r][c]

    return int_img_arr


# Calculate integral image
integral_img = calculate_integral_image(gray_img)
print("Integral image calculated successfully.")
print(f"Integral image size: {integral_img.shape}")

# ==============================================================================
# 2. Compute Mean Using Integral Image (Part b)
# ==============================================================================
print("\n--- b) Computing Mean Using Integral Image ---")


# TODO: explain to me what it means 1-indexed
def mean_using_integral(integral, top_left, bottom_right):
    """
    Calculate mean gray value using integral image.
    Time Complexity: O(1)

    Args:
        integral: The integral image
        top_left: (row, col) - top left corner of the region
        bottom_right: (row, col) - bottom right corner of the region
    
    Returns:
        Mean gray value of the region
    
    TODO:
    1. Extract coordinates from top_left and bottom_right
    2. Adjust indices for integral image (remember it's 1-indexed)
    3. Return Sum / number_of_pixels
    """
    
    sum = integral[top_left[0]][top_left[1]] - integral[top_left[0]][bottom_right[1]+1] - integral[bottom_right[0]+1][top_left[1]] + integral[bottom_right[0]+1][bottom_right[1]+1] # Placeholder return statement
    num_pixels = (bottom_right[0] - top_left[0] + 1) * (bottom_right[1] - top_left[1] + 1)
    return sum / num_pixels
    
    
    
    pass


# Define region
top_left = (10, 10)
bottom_right = (60, 80)

# Calculate mean using integral image
mean_integral = mean_using_integral(integral_img, top_left, bottom_right) # None  # Call mean_using_integral()

print(f"Region: Top-left {top_left}, Bottom-right {bottom_right}")
print(f"Region size: {bottom_right[0] - top_left[0] + 1} x {bottom_right[1] - top_left[1] + 1} pixels")
print(f"Mean gray value (Integral Image Method): {mean_integral:.2f}")

# ==============================================================================
# 3. Compute Mean by Direct Summation (Part c)
# ==============================================================================
print("\n--- c) Computing Mean by Direct Summation ---")


def mean_by_direct_sum(img: np.ndarray, top_left, bottom_right):
    """
    Calculate mean gray value by summing all pixels in region.
    Time Complexity: O(w * h) where w and h are region dimensions

    Args:
        img: The grayscale image
        top_left: (row, col) - top left corner of the region
        bottom_right: (row, col) - bottom right corner of the region
    
    Returns:
        Mean gray value of the region
    
    TODO:
    1. Extract the region from the image using array slicing
    2. Calculate and return the mean of all pixels in the region
      """
    return img[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1].mean()
    


# Calculate mean using direct summation
mean_direct = mean_by_direct_sum(gray_img, top_left, bottom_right)  # None  # Call mean_by_direct_sum(  )

print(f"Mean gray value (Direct Summation Method): {mean_direct:.2f}")

# ==============================================================================
# 4. Analyze Computational Complexity (Part d)
# ==============================================================================
print("\n--- d) Computational Complexity Analysis ---")

'''
TODO:
1. Benchmark both methods by running them multiple times (e.g., 100 iterations)
2. Measure execution time for both methods using time.perf_counter()
3. Compare the execution times
4. Verify that both methods produce the same result
5. Print the results:
   - Method name
   - Average execution time
   - Performance improvement factor


'''

# Benchmark parameters
iterations = 100

print(f"\nBenchmarking with {iterations} iterations...\n")

# TODO: Implement benchmarking code here

# Experiment integral method
time_integral_start = time.time()
integral_res_exp = None
for _ in range(iterations):
    integral_res_exp = mean_using_integral(integral_img, top_left, bottom_right)
time_integral = time.time() - time_integral_start

# Experiment direct computation
time_direct_start = time.time()
direct_res_exp = None
for _ in range(iterations):
    direct_res_exp = mean_by_direct_sum(gray_img, top_left, bottom_right)
time_direct = time.time() - time_direct_start

if integral_res_exp == direct_res_exp:
    print("Both methods yield the same result.")
else:
    print("Results differ between methods!")

# TODO: Display results 
print(f"Integral Image Method: {time_integral/iterations:.10f} seconds per iteration")
print(f"Direct Summation Method: {time_direct/iterations:.10f} seconds per iteration")

# TODO: Print theoretical complexity explanation
print("\nTheoretical Complexity Analysis:")
print("Integral Image Method: O(1) - Constant time complexity regardless of region size.")
print("Direct Summation Method: O(w * h) - Time complexity grows with the area of the region.")
print("As the region size increases, the performance advantage of the Integral Image Method becomes more pronounced.")