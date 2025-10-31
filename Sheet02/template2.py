# Template for Exercise 2 –  Fourier Transform and Image Reconstruction
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_fft(img):
    """
    Compute the Fourier Transform of an image and return:
    - The shifted spectrum
    - The magnitude
    - The phase
    """
    img = img.astype(np.float32)
    F = np.fft.fft2(img)
    F_shift = np.fft.fftshift(F)
    mag = np.abs(F_shift)
    phase = np.arctan2(F_shift.imag, F_shift.real)
    return F_shift, mag, phase


def reconstruct_from_mag_phase(mag, phase):
    """
    Reconstruct an image from given magnitude and phase.
    """
    F_shift = mag*np.exp(phase*1j)
    F = np.fft.ifftshift(F_shift)
    img = np.fft.ifft2(F)
    img = img.real
    img = img.astype(np.float32)
    return img


def compute_mad(a, b):
    """
    Compute the Mean Absolute Difference (MAD) between two images.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean(np.abs(a - b)))


def plot(title:str, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def img_viz(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)
    
# ==========================================================

# TODO: 1. Load grayscale images
img1 = cv2.imread("data/1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("data/2.png", cv2.IMREAD_GRAYSCALE)
# cv2.imshow('min', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
compute_fft(img1)
# TODO: 2. Compute magnitude and phase of both images
img1_F_shift, img1_mag, img1_phase = compute_fft(img1)
img2_F_shift, img2_mag, img2_phase = compute_fft(img2)


# TODO: 3. Swap magnitude and phase between the two images
# TODO: 4. Reconstruct and save the swapped results
img1_mag_img2_phase = reconstruct_from_mag_phase(img1_mag, img2_phase)
img2_mag_img1_phase = reconstruct_from_mag_phase(img2_mag, img1_phase)

# TODO: 5. Compute and print the MAD values between originals and reconstructions
mad1 = compute_mad(img1.astype(np.float32), img1_mag_img2_phase)
mad2 = compute_mad(img2.astype(np.float32), img2_mag_img1_phase)
print("MAD img1 vs (mag1+phase2):", mad1)
print("MAD img2 vs (mag2+phase1):", mad2)

# TODO: 6. Visualize all images (originals, magnitude, phase, reconstructions)
plot('Image 1: Original, Magnitude, Magnitude (log), Phase', cv2.hconcat([img_viz(img1), img_viz(img1_mag), img_viz(np.log1p(img1_mag)), img_viz(img1_phase)]))
plot('Image 2: Original, Magnitude, Magnitude (log), Phase', cv2.hconcat([img_viz(img2), img_viz(img2_mag), img_viz(np.log1p(img2_mag)), img_viz(img2_phase)]))
plot('Magnitude Image 1, Magnitude Image 1 (log), Phase Image 2, Reconstructed swapped image', cv2.hconcat([img_viz(img1_mag), img_viz(np.log1p(img1_mag)), img_viz(img2_phase), img_viz(img1_mag_img2_phase)]))
plot('Magnitude Image 2, Magnitude Image 2 (log), Phase Image 1, Reconstructed swapped image', cv2.hconcat([img_viz(img2_mag), img_viz(np.log1p(img2_mag)), img_viz(img1_phase), img_viz(img2_mag_img1_phase)]))

save = True
if save: 
    cv2.imwrite("reconstructed_mag1_phase2.png", cv2.hconcat([img_viz(img1_mag), img_viz(np.log1p(img1_mag)), img_viz(img2_phase), img_viz(img1_mag_img2_phase)]))
    cv2.imwrite("reconstructed_mag2_phase1.png", cv2.hconcat([img_viz(img2_mag), img_viz(np.log1p(img2_mag)), img_viz(img1_phase), img_viz(img2_mag_img1_phase)]))
