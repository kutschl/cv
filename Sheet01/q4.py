import numpy as np 
import cv2 

# Read bonn.jpeg image
img = cv2.imread("bonn.jpg")

# Convert it to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Task 4a: Decompose each kernel using SVD. Determine which kernel is separable.
# Kernel 1
K1 = np.array([
    [0.0113, 0.0838, 0.0113],
    [0.0838, 0.6193, 0.0838],
    [0.0113, 0.0838, 0.0113]
])

# Kernel 2
K2 = np.array([
    [-0.8984,  0.1472,  1.1410],
    [-1.9075,  0.1566,  2.1359],
    [-0.8659,  0.0573,  1.0337]
])


w1, u1, vt1 = cv2.SVDecomp(K1)
w2, u2, vt2 = cv2.SVDecomp(K2)
print("Singular values for Kernel 1:", w1.flatten())
print("Singular values for Kernel 2:", w2.flatten())

# Singular values for Kernel 1: [6.41975860e-01 7.58595503e-05 0.00000000e+00]
# Singular values for Kernel 2: [3.48798585 0.1004873  0.03576043]

# Results indicate that Kernel 1 is separable (only one significant singular value), Kernel 2 clearly not separable.
# Property for seperable Kernel: only the first singular value has to be non-zero, all others singular values have to zero.


# Task 4b: Approximation and filter image 
approx_kernel2 = w2[0]* np.outer(u2[:,0], vt2[0,:]) 
print('Approx Kernel 2: \n', approx_kernel2)

dst_approx = cv2.filter2D(gray, -1, approx_kernel2)
cv2.imshow("Filtered with Approx Kernel 2", dst_approx)
cv2.waitKey(0)
cv2.destroyAllWindows()

dst_k2 = cv2.filter2D(gray, -1, K2)
cv2.imshow("Filtered with Original Kernel 2", dst_k2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Task 4c: Compute absolute pixel-wise difference between results usind full knerl vs seperable approximation -> print maximum pixel error 
abs_diff = cv2.absdiff(dst_k2, dst_approx)
max_pixel_error = np.max(abs_diff)
print("Maximum pixel error between full Kernel 2 and its separable approximation:", max_pixel_error)
