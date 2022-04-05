**To run the script:**
`python3 threshold_morphology.py images/A-orig_img.jpg`

**Techniques**
- working with thresholding, and comparing thresholding approaches
- using the connected components algorithm to form regions
- using various region properties to eliminate undesired regions and hone in on region of interest
- identifying an object based on simple shape properties

**Problems  (50 points):**

1)  (10 points)  Thresholding

You can see that basic thresholding is already done in the initial program, but it doesn't perform very well done.  Try out a few different threshold algorithms and algorithm parameterizations and determine which give better/best results.
Report on what algorithms and parameters you tried, and explain how the results varied with the different attempts, and have your code implement the best overall option (for this task).
2)  (10 points)  Region Elimination 1:  Filtering

Thresholding typically produces noisy results with a LOT of pixels incorrectly classified as foreground or background.  It's desirable to clear out some of this noise right at the outset.  Doing some blurring/smoothing before thresholding can help with this, but post-thresholding options are available as well, including:
median filtering
morphology (opening, closing, etc.)
First do median filtering, and then use some morphology.  Figure out what combination of morphology transformations work well for filtering out noise while also improving (or not worsening) the primary region of interest (laptop screen).
Try different kernel sizes for morphology, including using different mask sizes for different operations if/when you're performing multiple morphology operations (e.g. doing both erosion and dilation, or both opening and closing, etc.).  For ease of experimentation, you may want to make your kernel sizes command line parameters
Try morphology for cleaning up the noise on the different images provided.   Does kernel size need to change with image size?  If so, explain how.
Report on the results of your morphology experiments, and have the code implement the best overall option.
3)  (10 points)  Connected Components and Region Elimination 2

The connected components algorithm provides our first step towards treating pixels as regions, as opposed to just individual pixels (and their neighbors)
After running connected components, we can determine a variety of properties about the various regions including area/size, center of mass, boundary size/perimeter, and a host of others
Area is fairly straightforward to implement, so compute the area of each of the regions in your (de-noised) thresholded image.
How might you use area to filter unwanted regions out of the image?  Implement this in your code.
4)  (10 points)  Contour Properties and Visualization:

The draw_contours() method in the given code provides an initial method for identifying different regions via different colors.  We can also show contours in a variety of other fashions:
as boundaries around the edge of the region,
as the convex hull,
as a bounding box (with horizontal and vertical edges parallel to image edges)
as a rotated bounding box (box edges not parallel to image edges)
Figure out how display the rotated rectangular bounding box around each region in the image.
Looking at the contour properties webpage (Links to an external site.), what properties do you think would be especially useful in detecting the screen of a laptop?  Implement the code to compute that property (or those properties) for each region.
Report on which property/properties you deem most useful for the contour-based filtering.
5)  (10 points)  Region Elimination 3 and Final Selection:

The last step is to use the contour properties you implemented in part 4 to filter out additional non-laptop-screen regions.  Once this is done, you should most likely have only one, or very few, region(s) remaining.  How might you go about selecting just one region (or no regions, if no screen in image) as the laptop screen?  Consider what other aspects of the laptop screen and the type of pictures it's (expected) to emit -- come up with a mechanism for selecting the best remaining option as the laptop screen.
Report on how your code selects the "most likely" laptop screen region when there multiple remaining regions.
NOTE:  When submitting your homework, be sure to include a document reporting the results of your experiments and discussing the choices you made during code development, as requested in the problems above.
