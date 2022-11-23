import cv2
import numpy as np

def isPosInShape(pos, shape):
    for i in range(len(pos)):
        if pos[i] < 0 or pos[i] >= shape[i]:
            return False
    return True

def morphologicalErode(image, kernel):
    def operateOnPixel(r, c):
        half = (kernel.shape[0] // 2, kernel.shape[1] // 2)
        if not image[r][c]:
            return 0
        indices = [(r+i-half[0], c+j-half[1]) for i in range(kernel.shape[0]) for j in range(kernel.shape[1]) if kernel[i][j] == 1]
        indices = [idx for idx in indices if isPosInShape(idx, image.shape)]
        for idx in indices:
            if not image[idx]:
                return 0
        return 1
    
    outputImage = np.zeros_like(image)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            outputImage[r][c] = operateOnPixel(r, c)
    return outputImage

def morphologicalDilate(image, kernel):
    def operateOnPixel(r, c):
        half = (kernel.shape[0] // 2, kernel.shape[1] // 2)
        if image[r][c]:
            return 1
        indices = [(r+i-half[0], c+j-half[1]) for i in range(kernel.shape[0]) for j in range(kernel.shape[1]) if kernel[i][j] == 1]
        indices = [idx for idx in indices if isPosInShape(idx, image.shape)]
        for idx in indices:
            if image[idx]:
                return 1
        return 0
    
    outputImage = np.zeros_like(image)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            outputImage[r][c] = operateOnPixel(r, c)
    return outputImage

def morphologicalOpen(image, kernel):
    return morphologicalDilate(morphologicalErode(image, kernel), kernel)

def morphologicalClose(image, kernel):
    return morphologicalErode(morphologicalDilate(image, kernel), kernel)

def morphologicalHitMiss(image, kernel):
    def operateOnPixel(r, c):
        half = (kernel.shape[0] // 2, kernel.shape[1] // 2)
        indices = [(i, j, r+i-half[0], c+j-half[1]) for i in range(kernel.shape[0]) for j in range(kernel.shape[1]) if kernel[i][j] == 1 or kernel[i][j] == 0]
        for idx in indices:
            if not isPosInShape((idx[2:]), image.shape) or kernel[idx[:2]] != image[idx[2:]] // 255:
                return 0
        return 1