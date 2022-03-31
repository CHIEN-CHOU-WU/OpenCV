
import cv2
from cv2 import CV_8UC1
import numpy as np
import os
import sys
import random as rnd
import matplotlib.pyplot as plt
import seaborn as sns

def getInputArgs ():
    if len(sys.argv) != 2:
        print (f'\nFormat:\n    {sys.argv[0]}  {"{image path/filename}"}\n')
        exit()

    if not os.path.isfile(sys.argv[1]):
        print (f'\nInvalid file:  {sys.argv[1]}\n')
        exit()

    return sys.argv[1]

def draw_contours (pix_labels, thresh):
    min_rect_len = 5
    max_rect_len = 1000
    num_labels = np.max(pix_labels) + 1

    boxed_comps_img = np.zeros([pix_labels.shape[0], pix_labels.shape[1], 3], dtype=np.uint8)
    boxed_comps_img[:,:,:] = 0

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    rnd.seed()

    # fill regions with random colors
    for i,cnt in enumerate(contours):
        one_pix_hsv = np.zeros([1,1,3],dtype=np.uint8)
        one_pix_hsv[0,0,:] = [ rnd.randint(0,255), rnd.randint(150,255), rnd.randint(200,255) ]
        bgr_color = cv2.cvtColor (one_pix_hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
        mask = np.zeros(thresh.shape,np.uint8)
        cv2.drawContours(boxed_comps_img,contours,i,bgr_color,-1)
        (x, y, w, h) = cv2.boundingRect(cnt)
        if h>min_rect_len and w>min_rect_len:
            # cv2.rectangle(boxed_comps_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(boxed_comps_img,[box],0,(0,0,255),5)
        
    return boxed_comps_img

def draw_contours_with_screen (pix_labels, thresh):
    min_rect_len = 100 # 40
    max_rect_len = 1000
    boxes = []
    extends = []
    num_labels = np.max(pix_labels) + 1

    boxed_comps_img = np.zeros([pix_labels.shape[0], pix_labels.shape[1], 3], dtype=np.uint8)
    boxed_comps_img[:,:,:] = 0

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    rnd.seed()

    # fill regions with random colors
    for i,cnt in enumerate(contours):
        one_pix_hsv = np.zeros([1,1,3],dtype=np.uint8)
        one_pix_hsv[0,0,:] = [ rnd.randint(0,255), rnd.randint(150,255), rnd.randint(200,255) ]
        bgr_color = cv2.cvtColor (one_pix_hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
        mask = np.zeros(thresh.shape,np.uint8)
        cv2.drawContours(boxed_comps_img,contours,i,bgr_color,-1)

        # display rotated rectangular bounding box around each region
        rect = cv2.minAreaRect(cnt)
        rect_w = rect[1][0]
        rect_h = rect[1][1]


        if max(rect[1])/min(rect[1]) < 3 and rect_w > min_rect_len and rect_h > min_rect_len and rect_w < max_rect_len and rect_h < max_rect_len:
            # for picture 2 only
            # if rect_h < rect_w:
            #     continue

            # print('rect_w', rect_w)
            # print('rect_h', rect_h)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxes.append(box)
            area = cv2.contourArea(cnt)
            rect_area = rect_w * rect_h
            extend = float(area) / rect_area
            extends.append(extend)
    
    max_index = extends.index(max(extends))

    cv2.drawContours(boxed_comps_img, [boxes[max_index]], 0, (0,0,255), 5)

    return boxed_comps_img
    

if __name__ == "__main__":
    filename = getInputArgs()
    image = cv2.imread (filename)
    cv2.imshow('img', image)
    gray = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)

    # 1) Thresholding
    ## Simple Thresholding
    thresh_100 = cv2.threshold (gray, 100, 255, cv2.THRESH_BINARY)[1]
    thresh_125 = cv2.threshold (gray, 125, 255, cv2.THRESH_BINARY)[1]
    thresh_150 = cv2.threshold (gray, 150, 255, cv2.THRESH_BINARY)[1]
    thresh_175 = cv2.threshold (gray, 175, 255, cv2.THRESH_BINARY)[1]
    thresh_200 = cv2.threshold (gray, 200, 255, cv2.THRESH_BINARY)[1]

    titles = ['Original Image', 'Thresh 100', 'Thresh 125', 'Thresh 150', 'Thresh 175', 'Thresh 200']
    images = [image, thresh_100, thresh_125, thresh_150, thresh_175, thresh_200]

    figure1 = plt.figure(1)
    for i in range(6):
        ax1 = plt.subplot(2, 3, i+1)
        if i == 0:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
        else:
            plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
        plt.suptitle('Simple thresholding')

    # figure1.savefig('results/image7_simple_thresh.png')

    ## Adaptive Thresholding
    img = cv2.imread (filename, 0)
    blur = cv2.medianBlur(img,5)
    ret,th1 = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)

    titles = ['Blurred Orig Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [blur, th1, th2, th3]

    figure2 = plt.figure(2)
    for i in range(4):
        ax2 = plt.subplot(2,2,i+1)
        plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
        plt.suptitle('Adaptive thresholding')
    
    # figure2.savefig('results/image7_adaptive_thresh.png')

    ## Otsu Thresholding
    blur = cv2.GaussianBlur(img, (5,5), 0)
    ret_blur, otsu_blur = cv2.threshold (blur,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow('otsu_blur', otsu_blur)

    figure3 = plt.figure(3)
    plt.imshow(otsu_blur, 'gray')
    plt.title('otsu blur')
    # figure3.savefig('results/otsu_blur.png')

    # 2) Region Elimination 1:  Filtering
    ## First do median filtering, and then use some morphology.
    median = cv2.medianBlur(otsu_blur,5)

    figure4 = plt.figure(4)
    plt.subplot(1, 2, 1)
    plt.imshow(otsu_blur, 'gray')
    plt.title('Adaptive Gaussian Thresholding')
    plt.subplot(1, 2, 2)
    plt.imshow(median, 'gray')
    plt.title('Median blur')

    plt.suptitle('Before / After median blur')

    # figure4.savefig('results/median_blur.png')
    
    def morphology(ksize=3):
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        kernel = np.ones((ksize, ksize),np.uint8)
        erosion = cv2.erode(median,kernel,iterations = 1)
        dilation = cv2.dilate(median,kernel,iterations = 1)
        opening = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel)
        combine = median.copy()
        combine = cv2.morphologyEx(combine, cv2.MORPH_OPEN, kernel)
        combine = cv2.morphologyEx(combine, cv2.MORPH_CLOSE, kernel)
        combine = cv2.erode(combine,kernel,iterations = 1)

        return erosion, dilation, opening, closing, combine

    # kernel size = 3
    erosion_3, dilation_3, opening_3, closing_3, combine_3 = morphology(ksize=3)

    morphology_images = [erosion_3, dilation_3, opening_3, closing_3, combine_3]    
    morphology_type = ['erosion', 'dilation', 'opening (erosion + dilation)', 'closing (dilation + erosion)', 'combine']

    figure5 = plt.figure(5, figsize=(12, 8))
    for i in range(5):
        plt.subplot(2, 3, i+1)
        plt.imshow(morphology_images[i], 'gray')
        plt.xticks([]),plt.yticks([])
        plt.title(morphology_type[i])

    plt.suptitle('3x3 ksize morphology')
    
    # figure5.savefig('results/morphology_3_ksize.png')

    # kernel size = 5
    erosion_5, dilation_5, opening_5, closing_5, combine_5 = morphology(ksize=5)

    morphology_images = [erosion_5, dilation_5, opening_5, closing_5, combine_5]    
    morphology_type = ['erosion', 'dilation', 'opening (erosion + dilation)', 'closing (dilation + erosion)', 'combine']

    figure6 = plt.figure(6, figsize=(12, 8))
    for i in range(5):
        plt.subplot(2, 3, i+1)
        plt.imshow(morphology_images[i], 'gray')
        plt.xticks([]),plt.yticks([])
        plt.title(morphology_type[i])

    plt.suptitle('5x5 ksize morphology')
    
    # figure6.savefig('results/morphology_5_ksize.png')

    # 3. Connected Components and Region Elimination 2
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilation_5, 8)
    labels_1d = labels.reshape(-1)
    labels_value_count = []
    for i in range(num_labels):
        # print('Label {} area: {} pixels'.format(i, (labels_1d == i).sum()))
        labels_value_count.append((labels_1d == i).sum())

    # area histagram of different regions
    figure_hist, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(labels_1d, ax=ax)
    plt.xticks(range(num_labels))
    plt.title("Area (pixels) of different regions")
    plt.xlabel('Labels (regions)')
    plt.ylabel('Pixels')

    # plt.savefig('results/hist_7.jpg')

    # area pie chart of different regions
    figure8 = plt.figure(8, figsize=(12, 8))
    colors = sns.color_palette('pastel')
    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{v:d}'.format(v=val)
        return my_format
    plt.pie(labels_value_count, colors=colors, labels=range(num_labels), autopct = autopct_format(labels_value_count), textprops={'fontsize': 20})

    # plt.savefig('results/pie_7.jpg')

    # 4. Contour Properties and Visualization
    boxed_conn_comps = draw_contours(labels, dilation_5)
    cv2.imshow('boxed_conn_comps', boxed_conn_comps)
    # cv2.imwrite('results/boxed_conn_2.jpg', boxed_conn_comps)


    # 5. 
    boxed_conn_comps_screen = draw_contours_with_screen(labels, dilation_5)
    cv2.imshow('boxed_conn_comps_screen', boxed_conn_comps_screen)
    # cv2.imwrite('results/boxed_conn_screen_2.jpg', boxed_conn_comps_screen)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # plt.show()


