# Init: This will both detect and crop the big image
import cv2
import numpy as np

# Function Definitions
def find_bright_area(image, radius, plot=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply a Gaussian blur to the image then find the brightest region
    gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    if plot:
        # display the results of our newly improved method
        img_plot = image.copy()
        cv2.circle(img_plot, maxLoc, radius, (255, 0, 0), 2)
        return maxLoc, img_plot
    else:
        return maxLoc

