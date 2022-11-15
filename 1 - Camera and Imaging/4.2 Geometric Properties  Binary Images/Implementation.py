import cv2
import numpy as np
from math import pi, degrees, radians, sin, cos, tan, asin, acos, atan, atan2


def getArea(image: np.ndarray):
    return np.sum(image) // 255


def getCenteroid(image: np.ndarray):
    x_ = y_ = 0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x]:
                x_   += x
                y_   += y
    area = getArea(image)
    return (x_ // area, y_ // area)


def getThetaEmin(image: np.ndarray, centeroid: tuple = None):
    if centeroid == None:
        centeroid = getCenteroid(image)
    x_, y_ = centeroid

    a_ = b_ = c_ = 0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x]:
                a_ += (x - x_) ** 2
                b_ += 2 * (x - x_) * (y - y_) 
                c_ += (y - y_) ** 2
    
    theta1 = atan2( b_, a_-c_) / 2     # Note: Radian angle
    theta2 = atan2(-b_, c_-a_) / 2
    check1 = (a_ - c_) * cos(2*theta1) + b_ * sin(2*theta1)
    if check1 > 0:
        return theta1
    else:
        return theta2


def getFullAxisPoints(image: np.ndarray, centeroid: tuple = None, theta_Emin: float = None):
    if centeroid == None:
        centeroid = getCenteroid(image)
    x_, y_ = centeroid

    if theta_Emin == None:
        theta_Emin = getThetaEmin(image, centeroid)
    rho = y_ * cos(theta_Emin) - x_ * sin(theta_Emin)
    m   = tan(theta_Emin)
    b   = rho / cos(theta_Emin)

    p1 = None
    p2 = None
    if degrees(theta_Emin) == 0.0:
        p1 = (0   , y_)
        p2 = (image.shape[1], y_)
    elif degrees(theta_Emin) == 90.0:
        p1 = (x_, 0)
        p2 = (x_, image.shape[0])
    else:
        y = b
        x = -b / m
        if y >= 0  and  y < image.shape[0]:
            p1 = (0, int(y))
        else:
            p1 = (int(x), 0)

        y = m * image.shape[1] + b
        x = (image.shape[0] - b) / m
        if y >= 0  and  y < image.shape[0]:
            p2 = (image.shape[1], int(y))
        else:
            p2 = (int(x), image.shape[0])
    
    return (p1, p2)