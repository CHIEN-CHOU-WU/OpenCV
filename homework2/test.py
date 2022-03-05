import cv2
import numpy as np

def convertToFloat32(image):
    if image.dtype != np.float32:
        info = np.iinfo(image.dtype)
        image = np.float32(image) / info.max

    return image 

img = cv2.imread('images_for_hybrid/rhino.jpg')
img2 = cv2.imread('images_for_hybrid/car.jpg')


if img.shape[0] * img.shape[1] < img2.shape[0] * img2.shape[1]:
    resize_dim = (img.shape[1], img.shape[0])
    img2 = cv2.resize(img2, resize_dim, interpolation = cv2.INTER_AREA)

if img.shape[0] * img.shape[1] > img2.shape[0] * img2.shape[1]:
    resize_dim = (img2.shape[1], img2.shape[0])
    img = cv2.resize(img, resize_dim, interpolation = cv2.INTER_AREA)
print(img.shape)
print(img2.shape)
cv2.imshow('image 1', img)
cv2.imshow('image 2', img2)
img = convertToFloat32(img)
img2 = convertToFloat32(img2)

cv2.imshow('image', img / 2 + img2 / 2)

cv2.waitKey(0)
cv2.destroyAllWindows()

