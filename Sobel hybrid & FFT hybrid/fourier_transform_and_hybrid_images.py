import numpy as np
import cv2
import sys
import os
#from matplotlib import pyplot as plt

def main():
    
    # 1) modify the function for getting arguments to receive the filenames of two images from the command line and ensure that both files exist
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    img = cv2.imread(args[0])
    img2 = cv2.imread(args[1])

    try:
        # Resize if needed
        if img.shape[0] * img.shape[1] != img2.shape[0] * img2.shape[1]:
            resize_dim = (img2.shape[1], img2.shape[0])
            img = cv2.resize(img, resize_dim, interpolation = cv2.INTER_AREA)

        cv2.imshow('Image 1', img)
        cv2.imshow('Image 2', img2)
    except:
        print('\x1b[6;30;42m' + 'Make sure path is corect and picture exists.' + '\x1b[0m')
        return 0



    # 2) convert our images to their floating-point representation, which is cv2.CV_32F or cv2.CV_64F in OpenCV (vs. the default of cv2.CV_8U)

    # check the data type of images
    print('Image data type:', img.dtype)
    print('Image 2 data type:', img2.dtype) 

    # convert images to 32-bit float
    def convertToFloat32(image):
        if image.dtype != np.float32:
            info = np.iinfo(image.dtype)
            image = np.float32(image) / info.max

        return image

    # convert images to 8-bit integer
    def convertToInt8(image):
        if image.dtype != np.uint8:
            image = 255 * image
            image = image.astype(np.uint8)

        return image

    # 3) For implementing the Sobel version for Hybrid Images, we first need to compute the edge magnitude results from the Sobel filters.
    #    To do this, let's use the square-root of the sum of the squares method -- i.e. the tougher but more accurate version

    def getSobelEdgesFloat32(image):
         # convert it to a 32-bit float image
        image = convertToFloat32(image)

        #computes the Sobel gradient in the x-direction (i.e. vertical edges),
        sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        # computes the Sobel gradient in the y-direction (i.e. horizontal edges),
        sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        # cv2.imshow('Sobel Edges x', convertToInt8(sobel_x))
        # cv2.imshow('Sobel Edges y', convertToInt8(sobel_y))

        # squares each of these gradient images then adds them together, and
        # takes the square root (i.e. generate the square root of the sum of the squares)
        gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_x))
        # gradient_magnitude = cv2.addWeighted(sobel_x, 0.5, sobel_x, 0.5, 0)

        # the function should return the 32-bit float image
        return gradient_magnitude

    # 4) Create the Sobel version for making hybrid images
    def getSobelHybrid(image, image2, ksize1, ksize2):

        # For the first image
        # apply Gaussian blurring to create a significantly blurred image (preferably in color).
        blurred_image = cv2.GaussianBlur(image, ksize=(ksize1, ksize1), sigmaX=0, borderType=cv2.BORDER_CONSTANT)

        # For the second image
        # convert the image to grayscale
        gray_2d = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        gray = np.zeros_like(image2)
        gray[:, :, 0] = gray_2d
        gray[:, :, 1] = gray_2d
        gray[:, :, 2] = gray_2d

        # apply a Gaussian filter, to perform a little pre-smoothing before applying the Sobel filters
        gray_gaussian = cv2.GaussianBlur(gray, ksize=(ksize2, ksize2), sigmaX=0, borderType=cv2.BORDER_CONSTANT)

        # use the getSobelEdgesFloat32() to generate the edge magnitude image for the second image to extract its high-frequency information.
        sobel = getSobelEdgesFloat32(gray_gaussian)
        # cv2.imshow('sobel int8',convertToInt8(sobel))

        # check dtype of two images
        blurred_image = convertToFloat32(blurred_image)
        print('Sobel data type:', sobel.dtype)
        print('Blurred image data type:', blurred_image.dtype)

        # combine = cv2.bitwise_and(blurred_image, sobel)
        combine = blurred_image / 2 + sobel / 2

        return combine
    
    hybrid_img = getSobelHybrid(img, img2, 5, 3)
    cv2.imshow('Hybrid Image (Sobel)', hybrid_img)

    # 5) Create the Fourier Transform version for making hybrid images
    def getFourierHybrid(image, low_pass_first_cutoff, first_shape, image2, image2_filter_type, image2_first_cutoff, image2_second_cutoff, second_shape):

        rows, cols, channel = image.shape
        crow, ccol = rows // 2, cols // 2
        
        def createMask(filter_type, first_frequency_cutoff, second_frequency_cutoff, shape):
            # filter type (0 = low-pass, 1 = high-pass, 2 = band-pass)
            # first frequency cutoff (given as a value in range 0.0 to 1.0)
            # the frequency cutoffs are passed as values between 0.0 and 1.0, where 0 is 0% distance from center, and 1.0 is 100% distance from center;  values most commonly in range 0.1-0.5)
            # second frequency cutoff (only used for band-pass filter)
            # shape (0 = circle, 1 = rectangle)

            radius = int(min(rows, cols) / 2 * first_frequency_cutoff) 

            if filter_type == 0:    # low-pass
                mask = np.zeros((rows, cols), dtype=np.uint8)
                if shape == 0:  # circle
                    mask = cv2.circle(mask, (int(cols/2), int(rows/2)), radius, 1, -1)
                if shape == 1:  # square
                    mask[crow-int(crow*first_frequency_cutoff):crow+int(crow*first_frequency_cutoff), ccol-int(ccol*first_frequency_cutoff):ccol+int(ccol*first_frequency_cutoff)] = 1
            if filter_type == 1:    # high-pass
                mask = np.ones((rows, cols), dtype=np.uint8)
                if shape == 0:  # circle
                    mask = cv2.circle(mask, (int(cols/2), int(rows/2)), radius, 0, -1)
                if shape == 1:  # square
                    mask[crow-30:crow+31, ccol-30:ccol+31] = 0
            if filter_type == 2:    # band-pass
                mask = np.zeros((rows, cols), dtype=np.uint8)
                mask = cv2.circle(mask, (int(cols/2), int(rows/2)), radius, 1, int(radius * second_frequency_cutoff))

            return mask

        ### First Image ###
        # separate the first image into three channels
        img_1, img_2, img_3 = cv2.split(image)

        # convert each image channel to frequency domain
        f = np.fft.fft2(img_1)
        fshift_1 = np.fft.fftshift(f)

        f = np.fft.fft2(img_2)
        fshift_2 = np.fft.fftshift(f)

        f = np.fft.fft2(img_3)
        fshift_3 = np.fft.fftshift(f)

        # low-pass filter to create a blurred version of the first image
        low_pass_mask = createMask(filter_type=0, first_frequency_cutoff=low_pass_first_cutoff, second_frequency_cutoff=0, shape=first_shape)
        # cv2.imshow('Masked image (low-pass)', low_pass_mask * 255)

        # create visible image version of FFT map/spectra
        # magnitude_spectrum = 15*np.log(np.abs(fshift_1))  # can apply different multipliers
        # mag_1 = cv2.convertScaleAbs (magnitude_spectrum)
        # cv2.imshow('Forier Spectra 1', mag_1)
        # masked_mag_1 = mag_1 * low_pass_mask
        # cv2.imshow('Masked Fourier Spectra 1', masked_mag_1)

        # apply its mask and convert back to spatial domain
        mask_mag_f_1 = low_pass_mask * fshift_1
        f_shift_1 = np.fft.ifftshift(mask_mag_f_1)
        img_back_1 = np.fft.ifft2(f_shift_1)
        img_back_1 = np.real(img_back_1)
        flt_img_1 = cv2.convertScaleAbs (img_back_1)
        # cv2.imshow ('Filtered image 1', flt_img_1)

        mask_mag_f_2 = low_pass_mask * fshift_2
        f_ishft_2 = np.fft.ifftshift(mask_mag_f_2)
        img_back_2 = np.fft.ifft2(f_ishft_2)
        img_back_2 = np.real(img_back_2)
        flt_img_2 = cv2.convertScaleAbs (img_back_2)
        # cv2.imshow ('Filtered image 2', flt_img_2)

        mask_mag_f_3 = low_pass_mask * fshift_3
        f_shift_3 = np.fft.ifftshift(mask_mag_f_3)
        img_back_3 = np.fft.ifft2(f_shift_3)
        img_back_3 = np.real(img_back_3)
        flt_img_3 = cv2.convertScaleAbs (img_back_3)
        # cv2.imshow ('Filtered image 3', flt_img_3)

        # combine channels into two 3-channel images
        combined = cv2.merge([flt_img_1, flt_img_2, flt_img_3])
        # cv2.imshow('Combined 3 channels into one (low pass)', combined)

        ### Second Image ###
        # convert the second image to gray-scale
        img2_gray_2d = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        def fft_merged(combined, mask):
            # convert gray image to frequency domain
            f = np.fft.fft2(img2_gray_2d)
            f_shift = np.fft.fftshift(f)

            # create visible image version of FFT map/spectra
            magnitude_spectrum = 15*np.log(np.abs(f_shift))  # can apply different multipliers
            mag = cv2.convertScaleAbs (magnitude_spectrum)
            # cv2.imshow('Forier Spectra', mag)
            masked_mag = mag * mask
            # cv2.imshow('Masked Fourier Spectra', masked_mag)

            # apply its mask and convert back to spatial domain
            mask_mag_f = mask * f_shift
            f_shift = np.fft.ifftshift(mask_mag_f)
            img_back = np.fft.ifft2(f_shift)
            img_back = np.real(img_back)
            flt_img = cv2.convertScaleAbs(img_back, beta=70)
            flt_img = cv2.cvtColor(flt_img, cv2.COLOR_GRAY2BGR)


            # merge the two images into hybrid image
            flt_img = convertToFloat32(flt_img)
            combined = convertToFloat32(combined)
            fft_merged = flt_img / 2 + combined / 2

            return flt_img, fft_merged

        if image2_filter_type == 1:
            # high-pass or band-pass filter to create the details (high-frequency) image for the second image
            high_pass_mask = createMask(filter_type=1, first_frequency_cutoff=image2_first_cutoff, second_frequency_cutoff=image2_second_cutoff, shape=second_shape)
            high_pass_flt_img, high_pass_fft_merged = fft_merged(combined, high_pass_mask)
            # cv2.imshow ('High pass filtered second image (gray)', high_pass_flt_img)

            return high_pass_fft_merged
        if image2_filter_type == 2:
            band_pass_mask = createMask(filter_type=2, first_frequency_cutoff=image2_first_cutoff, second_frequency_cutoff=image2_second_cutoff, shape=second_shape)
            # cv2.imshow('Masked image (band-pass)', band_pass_mask * 255)
            band_pass_flt_img, band_pass_fft_merged = fft_merged(combined, band_pass_mask)
            # cv2.imshow ('Band pass filtered second image (gray)', band_pass_flt_img)

            return band_pass_fft_merged

    high_pass_fft_merged = getFourierHybrid(image = img, low_pass_first_cutoff=0.1, first_shape=0, image2=img2, image2_filter_type=1, image2_first_cutoff=0.5, image2_second_cutoff=0, second_shape=1)
    cv2.imshow('High pass Fourier merged', high_pass_fft_merged)
    
    band_pass_fft_merged = getFourierHybrid(image = img, low_pass_first_cutoff=0.1, first_shape=0, image2=img2, image2_filter_type=2, image2_first_cutoff=0.1, image2_second_cutoff=0.7, second_shape=0)
    cv2.imshow('Band pass Fourier merged', band_pass_fft_merged)

    # 6) Last, experiment with your code for the three different sets of images given above.  Try different parameterizations of your various filters and find the parameters that give good results for each of these image sets.
    img_dog = cv2.imread('images_for_hybrid/dog.jpg')
    img_cat = cv2.imread('images_for_hybrid/cat.jpg')

    # cv2.imshow('dog', img_dog)
    # cv2.imshow('cat', img_cat)

    ksizes = [1, 3, 5, 7]
    
    # for i in ksizes:
    #     for j in ksizes:
    #         hybrid_img = getSobelHybrid(img_dog, img_cat, i, j)
    #         cv2.imshow('Cat & Dog Sobel hybrid ksize1 = {}, ksize2 = {}'.format(i, j), hybrid_img)
    hybrid_img = getSobelHybrid(img_dog, img_cat, 3, 7)
    cv2.imshow('Best Sobel hybrid (Cat & Dog) ksize1 = {}, ksize2 = {}'.format(3, 7), hybrid_img)

    cutoff = [0.1, 0.2, 0.3, 0.4, 0.5]
    cutoff2 = [0.1, 0.2, 0.3, 0.4, 0.5]
    cutoff3 = [0.5, 0.6, 0.7, 0.8, 0.9]
    # for i in cutoff:
    #     for j in cutoff2:
    #         high_pass_fft_merged = getFourierHybrid(image = img_dog, low_pass_first_cutoff=i, first_shape=0, image2=img_cat, image2_filter_type=1, image2_first_cutoff=j, image2_second_cutoff=0, second_shape=0)
    #         cv2.imshow('Cat & Dog fft hybrid 1st_cutoff:{}, 2nd_cutoff:{}'.format(i, j), high_pass_fft_merged)
    high_pass_fft_merged = getFourierHybrid(image = img_dog, low_pass_first_cutoff=0.1, first_shape=0, image2=img_cat, image2_filter_type=1, image2_first_cutoff=0.1, image2_second_cutoff=0, second_shape=0)
    cv2.imshow('Best fft hybrid (Cat & Dog) 1st_cutoff = {}, 2nd_cutoff = {}'.format(0.1, 0.1), high_pass_fft_merged)

    # for i in cutoff:
    #     for j in cutoff2:
    #         for k in cutoff3:
    #             band_pass_fft_merged = getFourierHybrid(image = img_dog, low_pass_first_cutoff=i, first_shape=0, image2=img_cat, image2_filter_type=2, image2_first_cutoff=j, image2_second_cutoff=k, second_shape=0)
    #             cv2.imshow('Cat & Dog fft hybrid 1st_cutoff:{}, 2nd_cutoff:{}, 3rd_cutoff:{}'.format(i, j, k), band_pass_fft_merged)
    # band_pass_fft_merged = getFourierHybrid(image = img_dog, low_pass_first_cutoff=0.1, first_shape=0, image2=img_cat, image2_filter_type=2, image2_first_cutoff=0.1, image2_second_cutoff=1, second_shape=0)
    # cv2.imshow('Cat & Dog fft hybrid 1st_cutoff:{}, 2nd_cutoff:{}, 3rd_cutoff:{}'.format(0.1, 0.1, 1), band_pass_fft_merged)

    img_marilyn = cv2.imread('images_for_hybrid/marilyn.jpg')
    img_einstein = cv2.imread('images_for_hybrid/einstein.jpg')
    resize_dim = (img_einstein.shape[1], img_einstein.shape[0])
    img_marilyn = cv2.resize(img_marilyn, resize_dim, interpolation = cv2.INTER_AREA)

    # for i in ksizes:
    #     for j in ksizes:
    #         hybrid_img = getSobelHybrid(img_marilyn, img_einstein, i, j)
    #         cv2.imshow('Marilyn & Einstein Sobel hybrid ksize1 = {}, ksize2 = {}'.format(i, j), hybrid_img)
    hybrid_img = getSobelHybrid(img_marilyn, img_einstein, 5, 7)
    cv2.imshow('Best Sobel hybrid (Marilyn & Einstein) ksize1 = {}, ksize2 = {}'.format(5, 7), hybrid_img)

    # for i in cutoff:
    #     for j in cutoff2:
    #         high_pass_fft_merged = getFourierHybrid(image = img_marilyn, low_pass_first_cutoff=i, first_shape=1, image2=img_einstein, image2_filter_type=1, image2_first_cutoff=j, image2_second_cutoff=0, second_shape=1)
    #         cv2.imshow('Marilyn & Einstein fft hybrid 1st_cutoff:{}, 2nd_cutoff:{}'.format(i, j), high_pass_fft_merged)
    high_pass_fft_merged = getFourierHybrid(image = img_marilyn, low_pass_first_cutoff=0.1, first_shape=0, image2=img_einstein, image2_filter_type=1, image2_first_cutoff=0.1, image2_second_cutoff=0, second_shape=0)
    cv2.imshow('Best fft hybrid (Marilyn & Einstein) 1st_cutoff:{}, 2nd_cutoff:{}'.format(0.1, 0.1), high_pass_fft_merged)

    img_car = cv2.imread('images_for_hybrid/car.jpg')
    img_rhino = cv2.imread('images_for_hybrid/rhino.jpg')
    resize_dim = (img_rhino.shape[1], img_rhino.shape[0])
    img_car = cv2.resize(img_car, resize_dim, interpolation = cv2.INTER_AREA)

    # for i in ksizes:
    #     for j in ksizes:
    #         hybrid_img = getSobelHybrid(img_car, img_rhino, i, j)
    #         cv2.imshow('Car & Rhino Sobel hybrid ksize1 = {}, ksize2 = {}'.format(i, j), hybrid_img)    
    hybrid_img = getSobelHybrid(img_car, img_rhino, 5, 7)
    cv2.imshow('Best Sobel hybrid (Car & Rhino) ksize1 = {}, ksize2 = {}'.format(5, 7), hybrid_img)

    # for i in cutoff:
    #     for j in cutoff2:
    #         high_pass_fft_merged = getFourierHybrid(image = img_car, low_pass_first_cutoff=i, first_shape=0, image2=img_rhino, image2_filter_type=1, image2_first_cutoff=j, image2_second_cutoff=0, second_shape=0)
    #         cv2.imshow('Car & Rhino fft hybrid 1st_cutoff:{}, 2nd_cutoff:{}'.format(i, j), high_pass_fft_merged) 

    high_pass_fft_merged = getFourierHybrid(image = img_car, low_pass_first_cutoff=0.1, first_shape=0, image2=img_rhino, image2_filter_type=1, image2_first_cutoff=0.1, image2_second_cutoff=0, second_shape=0)
    cv2.imshow('Best fft hybrid (Car & Rhino) 1st_cutoff:{}, 2nd_cutoff:{}'.format(0.1, 0.1), high_pass_fft_merged) 

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()