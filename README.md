# ImageMorphology_HoughTransform_CSE573
Remove Noise from binary images; Perform image segmentation; Detect shapes in images using Hough Transform. 
<b> Task 1: Using morphology operations, denoise the given images. Also, find the boundaries of the objects in the image.</b>
<br><br><i>Erosion, Dilation, Opening and Closing:</i>
<br><i>Erosion</i> of a binary image is the subtraction of pixels in an image by a structuring element. An erosion operation produces a new binary image of an image with 1s only at those locations (origins) in the new binary image where the structuring element fits the input image, otherwise replace those locations with 0s.
<br><i>Dilation</i> of a binary image is the addition of pixels in an image by a structuring element. A dilation operation produces a new binary image having 1s at all those locations (origins) where the structuring element hits the input image, else leave the main input image as it is.
<br><i>Opening</i> operation is a combination of erosion and dilation. While opening an image, we erode the input image with a structuring element and then dilate the eroded image with either the same structuring element or a new element.
<br><i>Closing</i> operation is just the opposite of opening, where we first dilate the image and then erode it.
<br><br><b>Denoising an image:</b>An image can be denoised by performing a combination of opening and closing. The order of operations does not matter, that is we may first perform opening and then closing, or vice-versa and still obtain the same results.

<br><br><b>Task 2: Perform point detection on an image.</b>
Also, use the concept of segmentation and thresholding to find an optimal threshold for segmenting the foreground from the background for a given grayscale image. </b>
<br>Image Segmentation.</b>
<br>Segmentation is a technique in image processing that helps us to isolate objects from each other in an image when they are one over another.
Segmentation subdivides an image into its constituent regions or objects. Segmentation can be done by using two basic properties of image intensities –
• Similarity – divide an image into similar parts based on a predefined criterion like thresholding.
• Discontinuity – divide an image based on sudden changes in intensities like gradients, etc. Methods in this include Point, Line and Edge Detection, etc.

<br><br><b> Task3: To implement Hough Transform on a given set of images and detect shapes like lines and circles in the image. </b>
