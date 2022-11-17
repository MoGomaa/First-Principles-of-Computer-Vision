import cv2
import numpy as np

def getProirNeighbourLables(image, r, c):
    return [lable for lable in image[r-1:r+1, c-1:c+1].flatten()[:-1] if lable]

def getLable(table, lable):
    while len(table[lable]):
        lable = min(table[lable])
    return lable

def fixEquivalenceTable(table, key, lable):
    if lable not in table[key]:
        table[key].append(lable)
        for otherLable in table[key][:-1]:
            fixEquivalenceTable(table, otherLable, lable)

def equalizeNormalizeLabledImage(labledImage, equivalenceTable):
    newTable = {}
    for r in range(labledImage.shape[0]):
        for c in range(labledImage.shape[1]):
            if labledImage[r][c]:
                lable = getLable(equivalenceTable, labledImage[r][c])
                if lable not in newTable:
                    newTable[lable] = len(newTable.keys()) + 1
                labledImage[r][c] = newTable[lable]
    return labledImage



def getConnectedComponent(image: np.ndarray):   
    labledImage      = np.zeros_like(image)
    equivalenceTable = {}
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if image[r][c]:
                neighbourLables = getProirNeighbourLables(labledImage, r, c)
                if len(neighbourLables) == 0:
                    newLable = len(list(equivalenceTable.keys())) + 1
                    equivalenceTable[newLable] = []
                    labledImage[r][c] = newLable
                else:
                    lable = getLable(equivalenceTable, min(neighbourLables))
                    labledImage[r][c] = lable
                    for l in neighbourLables:
                        if l != lable:
                            fixEquivalenceTable(equivalenceTable, l, lable)
    labledImage = equalizeNormalizeLabledImage(labledImage, equivalenceTable)
    return labledImage


def getColors(numColors):
    exp = 0
    while True:
        if numColors <= exp**3:
            break
        exp += 1
    linspace = np.linspace(0, 255, exp, dtype=int)
    colors   = [[r, g, b] for r in linspace for g in linspace for b in linspace]
    print(len(colors))
    return [colors[idx] for idx in np.linspace(0, len(colors)-1, numColors, dtype=int)]

def colorize(image):
    unique = np.unique(image)
    colors = getColors(len(unique))
    print(colors)
    coloredImage = np.zeros((image.shape[0], image.shape[1], 3))
    print(coloredImage.shape)
    for i in range(1, len(colors)):
        coloredImage[image == i] = colors[i]
    return coloredImage

