<!-- <style>
* {
  box-sizing: border-box;
}

.column_33 {
  float: left;
  width: 33.33%;
  padding: 5px;
}

.column_50{
  float: left;
  width: 50%;
  padding: 5px 70px;
}

figcaption{
    text-align: center;
}
</style> -->

# 1. Thresholding
1. Image 1

<div>
  <p align="center">
    <img src="./results/image1_simple_thresh.png" width="350">
    <img src="./results/image1_adaptive_thresh.png" width="350">
    <img src="./results/otsu_blur_1.png" width="350">
  </p>
</div>

  - It seems like thresh v = 150 has the best result for simple thresholding.
  - Adaptive Gaussian has a better result with less noise.


2. Image 2

<div>
  <p align="center">
    <img src="./results/image2_simple_thresh.png" width="350">
    <img src="./results/image2_adaptive_thresh.png" width="350">
    <img src="./results/otsu_blur_2.png" width="350">
  </p>
</div>

  - It seems like thresh v = 200 has the best result for simple thresholding. It can seperate right monitor with other part of the image.
  - Adaptive Gaussian has a better result with less noise.

3. Image 3

<div>
  <p align="center">
    <img src="./results/image3_simple_thresh.png" width="350">
    <img src="./results/image3_adaptive_thresh.png" width="350">
    <img src="./results/otsu_blur_3.png" width="350">
  </p>
</div>

4. Image 4

<div>
  <p align="center">
    <img src="./results/image4_simple_thresh.png" width="350">
    <img src="./results/image4_adaptive_thresh.png" width="350">
    <img src="./results/otsu_blur_4.png" width="350">
  </p>
</div>

5. Image 5

<div>
  <p align="center">
    <img src="./results/image5_simple_thresh.png" width="350">
    <img src="./results/image5_adaptive_thresh.png" width="350">
    <img src="./results/otsu_blur_5.png" width="350">
  </p>
</div>

6. Image 6

<div>
  <p align="center">
    <img src="./results/image6_simple_thresh.png" width="350">
    <img src="./results/image6_adaptive_thresh.png" width="350">
    <img src="./results/otsu_blur_6.png" width="350">
  </p>
</div>

7. A-orig_img.jpg
<div>
  <p align="center">
    <img src="./results/image7_simple_thresh.png" width="350">
    <img src="./results/image7_adaptive_thresh.png" width="350">
    <img src="./results/otsu_blur_7.png" width="350">
  </p>
</div>

# 2. Region Elimination 1:  Filtering
Otsu thresholding is chosen.

First do median filtering (remove noise)
<div align="center">
  <img src="./results/median_blur.png" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;">
</div>

Second, use some morphology (based on otsu threshholding).
1. 1_ms_surface.jpg
<div>
  <p align="center">
    <img src="./results/morphology_3_ksize_1.png" width="550">
    <img src="./results/morphology_5_ksize_1.png" width="550">
  </p>
</div>

2. 2_dual_laptops.jpg
<div>
  <p align="center">
    <img src="./results/morphology_3_ksize_2.png" width="550">
    <img src="./results/morphology_5_ksize_2.png" width="550">
  </p>
</div>

3. 3_laptop_light_bkgnd.jpg
<div>
  <p align="center">
    <img src="./results/morphology_3_ksize_3.png" width="550">
    <img src="./results/morphology_5_ksize_3.png" width="550">
  </p>
</div>

4. 4_ms_surface_angled.png
<div>
  <p align="center">
    <img src="./results/morphology_3_ksize_4.png" width="550">
    <img src="./results/morphology_5_ksize_4.png" width="550">
  </p>
</div>

5. 5_screen_in_bkgnd.png
<div>
  <p align="center">
    <img src="./results/morphology_3_ksize_5.png" width="550">
    <img src="./results/morphology_5_ksize_5.png" width="550">
  </p>
</div>

6. 6_clipped_corner.jpg
<div>
  <p align="center">
    <img src="./results/morphology_3_ksize_6.png" width="550">
    <img src="./results/morphology_5_ksize_6.png" width="550">
  </p>
</div>

7. A-orig_img.jpg
<div>
  <p align="center">
    <img src="./results/morphology_3_ksize_7.png" width="550">
    <img src="./results/morphology_5_ksize_7.png" width="550">
  </p>
</div>

Kernel size 3 and 5 doesn't have that much of difference in these cases.

# 3. Connected Components and Region Elimination 2
Calculate the area of each regions in each images
| Image      | Area (Pixels) histogram | Area (Pixels) pie |
| :--- | :----: | :----: |
| 1_ms_surface.jpg      | <div> <img src="./results/hist_1.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;"></div> | <div> <img src="./results/pie_1.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;"></div> |
| 2_dual_laptops.jpg   | <div> <img src="./results/hist_2.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;"></div> | <div> <img src="./results/pie_2.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;"></div> |
| 3_laptop_light_bkgnd.jpg   | <div> <img src="./results/hist_3.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;"></div>         | <div> <img src="./results/pie_3.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;"></div> |
| 4_ms_surface_angled.png   | <div> <img src="./results/hist_4.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;"></div>         | <div> <img src="./results/pie_4.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;"></div> |
| 5_screen_in_bkgnd.png   | <div> <img src="./results/hist_5.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;"> </div>         | <div> <img src="./results/pie_5.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;"></div> |
| 6_clipped_corner.jpg   | <div> <img src="./results/hist_6.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;"></div>         | <div> <img src="./results/pie_6.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;"></div> |
| A-orig_img.jpg   | <div> <img src="./results/hist_7.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;"></div>         | <div> <img src="./results/pie_7.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;"></div> |

We can use area to find the unwanted regions by specific conditions. For example, if we want to find a circle like a ball in the image, we can draw a circle on those found region and calculate the ratio between the region and the circle. The highest ratio should be the area (ball) we are looking for.

# 4. Contour Properties and Visualization
For picture 1, 3, 4, 5, 6, A-ori_img, 5X5 kernel + median (otsu threshold) + dilation is selected.

For picture 2, 5X5 kernel + median (threshold v = 200) + opening is selected. 

1. 1_ms_surface.jpg

<div>
  <img src="./results/boxed_conn_1.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;">
</div>

2. 2_dual_laptops.jpg

<div>
  <img src="./results/boxed_conn_2.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;">
</div>

The minimal rectangle length for picture was set to 40 in order to filter out the screen.

3. 3_laptop_light_bkgnd.jpg

<div>
  <img src="./results/boxed_conn_3.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;">
</div>

4. 4_ms_surface_angled.png

<div>
  <img src="./results/boxed_conn_4.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;">
</div>

5. 5_screen_in_bkgnd.png

<div>
  <img src="./results/boxed_conn_5.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;">
</div>

6. 6_clipped_corner.jpg

<div>
  <img src="./results/boxed_conn_6.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;">
</div>

7. A-orig_img.jpg

<div>
  <img src="./results/boxed_conn_7.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;">
</div>

**Discussion:**

From all the properties show in https://docs.opencv.org/4.x/d1/d32/tutorial_py_contour_properties.html ,
$$\begin{equation}Extent=\frac{Object\ Area }{Bounding\ Rectangular\ Area} \end{equation}$$
 looks like a good way to filter out the screen from the image. Calculate all the extent from different contour and the screen should be the one with the maximum extend value. But before that, we should eliminate region whose length divided by width is greater than a certain value (3 for now) due to that fact that screens should fit a certain ratio.

# 5. Region Elimination 3 and Final Selection:
Before calculating for extent, we filter the region with rectangle height and width larger than 100 pixels and smaller than 1000 pixes to avoid small region and large region.

1. 1_ms_surface.jpg

<div>
  <img src="./results/boxed_conn_screen_1.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;">
</div>

2. 2_dual_laptops.jpg

<div>
  <img src="./results/boxed_conn_screen_2.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;">
</div>
Set minimun rectangular length to 40 pixels in this case.

To find this specific screen, only the ratio condition would not work well with this image. Hence, I added another condition which is find the area with rectangle width larger than rectangle height assuming that all the screen would place properly on the surface.

3. 3_laptop_light_bkgnd.jpg

<div>
  <img src="./results/boxed_conn_screen_3.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;">
</div>

4. 4_ms_surface_angled.png

<div>
  <img src="./results/boxed_conn_screen_4.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;">
</div>

5. 5_screen_in_bkgnd.png

<div>
  <img src="./results/boxed_conn_screen_5.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;">
</div>

6. 6_clipped_corner.jpg

<div>
  <img src="./results/boxed_conn_screen_6.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;">
</div>

7. A-orig_img.jpg

<div>
  <img src="./results/boxed_conn_screen_7.jpg" alt="Snow" style="display:block; margin-left:auto; margin-right:auto; width:50%;">
</div>
