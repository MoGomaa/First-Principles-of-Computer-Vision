import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import pi, degrees, radians, sin, cos, tan, asin, acos, atan, atan2


Size = (300, 300)


walk = list(os.walk(os.path.join("images", "Binary")))
imagesDir, filenames = walk[0][0], walk[0][2]
images = [file for file in filenames if ".png" in file]

for imageName in images:    
    # read the image
    imagePath = os.path.join(imagesDir, imageName)
    image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    
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
    cv2.imwrite(imagePath, image)