# Template for Exercise 3 – Spatial and Frequency Domain Filtering
import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_box_kernel(k):
    """
    Create a normalized k×k box filter kernel.
    """
    if k % 2 == 1: 
        return np.ones(shape=(k,k))/(k**2)
    else:
        raise ValueError("kernel size is even!")


def make_gauss_kernel(k, sigma):
    """
    Create a normalized 2D Gaussian filter kernel of size k×k.
    """
    if k % 2 == 1: 
        r = k // 2  
        x, y = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1)) 
        g = 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(x**2 + y**2)/2*sigma**2)
        g = g/ np.sum(g)
        return g
    else:
        raise ValueError("Kernel size is even!")
    


def conv2_same_zero(img, h):
    """
    Perform 2D spatial convolution using zero padding.
    Output should have the same size as the input image.
    (Do NOT use cv2.filter2D)
    """
    img = img.astype(np.float64)
    h   = h.astype(np.float64)
    s = h.sum()
    if s != 0: h /= s

    kernel = np.flipud(np.fliplr(h))
    i_h, i_w = img.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant', constant_values=0)

    out = np.zeros((i_h, i_w), dtype=np.float64)

    for y in range(i_h):
        for x in range(i_w):
            region = padded[y:y+k_h, x:x+k_w]
            out[y, x] = np.sum(region * kernel)

    return out   



def freq_linear_conv(img, h):
    """
    Perform linear convolution in the frequency domain.
    (You can use numpy.fft)
    """
    img = img.astype(np.float64)
    h   = h.astype(np.float64)
    if h.sum() != 0:
        h /= h.sum()

    # Image and kernel shapes
    i_h, i_w = img.shape
    k_h, k_w = h.shape

    # Output size for linear convolution
    out_h = i_h + k_h - 1
    out_w = i_w + k_w - 1

    # Zero-pad image and kernel to output size
    F = np.fft.fft2(img, s=(out_h, out_w))
    H = np.fft.fft2(h,   s=(out_h, out_w))

    # Multiply in frequency domain
    G = F * H

    # Inverse FFT → real part
    g_full = np.fft.ifft2(G).real

    # Crop to same size as input image (centered)
    start_y = (k_h - 1) // 2
    start_x = (k_w - 1) // 2
    g_same = g_full[start_y:start_y + i_h, start_x:start_x + i_w]

    return g_same.astype(np.float64)



def compute_mad(a, b):
    """
    Compute Mean Absolute Difference (MAD) between two images.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean(np.abs(a - b)))


def plot(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img_viz(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

# ==========================================================

# TODO: 1. Load the grayscale image (e.g., lena.png)
img = cv2.imread('data/lena.png', cv2.IMREAD_GRAYSCALE)
plot('Original lena image', img)

# TODO: 2. Construct 9×9 box and Gaussian kernels (same sigma)
box_kernel = make_box_kernel(9)
print('Box kernel')
print(box_kernel)
gaussian_kernel = make_gauss_kernel(k=9, sigma=1)
print('Gaussian kernel')
print(gaussian_kernel)

# TODO: 3. Apply both filters spatially (manual convolution)
img_manual_box = conv2_same_zero(img, box_kernel)
print(img_manual_box.shape, img.shape)
plot('Applied box filter with convolution', img_viz(img_manual_box))
img_manual_gaussian = conv2_same_zero(img, gaussian_kernel)
plot('Applied gaussian kernel with convolution', img_viz(img_manual_gaussian))

# TODO: 4. Apply both filters in the frequency domain
img_freq_box = freq_linear_conv(img, box_kernel)
plot('Applied box filter with multiplication in frequency domain', img_viz(img_freq_box))
img_freq_gaussian = freq_linear_conv(img, gaussian_kernel)
plot('Applied gaussian filter with multiplication in frequency domain', img_viz(img_freq_gaussian))


# TODO: 5. Compute and print MAD between spatial and frequency outputs
mad_box = compute_mad(img_manual_box, img_freq_box)
mad_gaussian = compute_mad(img_manual_gaussian, img_freq_gaussian)
print("MAD convolution box vs frequency box:", mad_box, mad_box < 10**-7)
print("MAD convolution gaussian vs frequency gaussian:", mad_gaussian, mad_gaussian < 10**-7)


# TODO: 6. Visualize all results (original, box/gaussian spatial, box/gaussian frequency, spectrum)
# TODO: 7. Verify that MAD < 1×10⁻⁷ for both filters
print('MAD box filter', mad_box, '<', 10**-7, mad_box < 10**-7)
print('MAD gaussian filter', mad_gaussian, '<', 10**-7, mad_gaussian < 10**-7)