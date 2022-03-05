import numpy as np
import cv2
import sys
import os
#from matplotlib import pyplot as plt

# Load an color image in grayscale
# img = cv2.imread('images_for_hybrid/san_francisco.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('images_for_hybrid/einstein.jpg', cv2.IMREAD_GRAYSCALE)

# apply FFT to image (spatial -> freq)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 15*np.log(np.abs(fshift))  # can apply different multipliers


# find center coordinate (crow, ccol)
rows, cols = img.shape
crow, ccol = rows//2, cols//2

# create a HPF with 64x64 square of 0s in center, 1s otherwise
mask = np.ones ((rows, cols), dtype=np.uint8)
mask[crow-30:crow+31, ccol-30:ccol+31] = 0

# create a LPF with circle of 1s in center, 0s otherwise
# mask = np.zeros ((rows, cols), dtype=np.uint8)
# mask = cv2.circle (mask, (int(cols/2), int(rows/2)), int(min(rows,cols)/5), 1, -1)

# create visible image version of FFT map/spectra
mag = cv2.convertScaleAbs (magnitude_spectrum)
masked_mag = mag * mask

# mask and apply inverse FFT (freq -> spatial)
mask_mag_f = mask * fshift
f_ishift = np.fft.ifftshift (mask_mag_f)
img_back = np.fft.ifft2 (f_ishift)
img_back = np.real (img_back)

#print (img_back)

# for HPF results, display image as offset from mid-range (in gray and JET)
#flt_img = cv2.convertScaleAbs (img_back, beta=100)         #beta=128)
#flt_jet = cv2.applyColorMap (flt_img, cv2.COLORMAP_JET)

# for LPF results, avg luminance values remain, so no beta (or zero beta) needed
flt_img = cv2.convertScaleAbs (img_back)        #, beta=0)
flt_jet = cv2.applyColorMap (flt_img, cv2.COLORMAP_JET)

#print (flt_img)


cv2.imshow ('Image', img)
cv2.imshow ('Mask', mask*255)
cv2.imshow ('Fourier Spectra', mag)
cv2.imshow ('Masked Fourier Spectra', masked_mag)
cv2.imshow ('Filtered image', flt_img)
cv2.imshow ('Filtered image - JET', flt_jet)

cv2.waitKey(0)
cv2.destroyAllWindows()

