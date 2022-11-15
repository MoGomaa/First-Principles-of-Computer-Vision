import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import pi, degrees, radians, sin, cos, tan, asin, acos, atan, atan2

Size = (300, 300)

walk = os.walk(".")
filenames = list(walk)[0][2]
images = [file for file in filenames if ".png" in file]

modified = False
for path in images:
    # read the image
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    # Make sure that image has 1-channel
    if len(image.shape) == 3:
        image = image[:, :, 0]
    
    # resize image
    if image.shape != Size:
        image = cv2.resize(image, Size)
    
    # Make sure image has only two values
    image[image <  128] = 0
    image[image >= 128] = 255
    
    # Save the preprocessed image 
    cv2.imwrite(path, image)