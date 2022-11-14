import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("BinaryImageSingle.png", cv2.IMREAD_UNCHANGED)
# image = cv2.imread("BinaryImageMultiple.png", cv2.IMREAD_UNCHANGED)
print("Image shape:   ", image.shape)

unique, counts = np.unique(image, return_counts=True)
counts         = dict(zip(unique, counts))
print("Values: ", counts)

# image[image <  128] = 0
# image[image >= 128] = 255
# unique, counts = np.unique(image, return_counts=True)
# counts         = dict(zip(unique, counts))
# print("Values: ", counts)

Area = 0
x_   = 0
y_   = 0
# Area = counts[255]
# Area = np.sum(image) // 255
for x in range(image.shape[1]):
    for y in range(image.shape[0]):
        if image[y][x]:
            Area += 1
            x_   += x
            y_   += y
x_ //= Area
y_ //= Area 
print("Area:", Area)
print("x_:  ", x_)
print("y_:  ", y_)

# For sake of visualization
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 
image = cv2.circle(image, (x_, y_), radius=0, color=(0, 0, 255), thickness=5)

cv2.imshow("Binary", image)
cv2.waitKey(0) 
cv2.destroyAllWindows() 