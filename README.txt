Requires Python PILLOW and PyPNG to run.

Takes as input an image "road.pgm" (must be pgm format, I converted on
this site: https://convertio.co/png-pgm/). Image must be present in the same
directory as the __init__.py file. 

Also will ask for 4 arguments as input: the RANSAC distance threshold, 
the RANSAC inlier threshold, the Hough transform theta accuracy, and the 
Hough transform rho accuracy. For my output images, I used values of 1.2, 60, 
1.5, and 1 respectively for each parameter.

The program will output 5 images: the hessian version of the road image, 
the ransac keypoints, the ransac lines plotted on the hessian image, the
hough keypoints, and the hough lines plotted on the hessian image.