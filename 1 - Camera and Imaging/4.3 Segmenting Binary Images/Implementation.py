import cv2
import numpy as np

def getProirNeighbourLabels(image, r, c):
    return [label for label in image[r-1:r+1, c-1:c+1].flatten()[:-1] if label]

def getLabel(table, label):
    while len(table[label]):
        label = min(table[label])
    return label

def fixEquivalenceTable(table, key, label):
    if label not in table[key]:
        table[key].append(label)
        for otherLabel in table[key][:-1]:
            fixEquivalenceTable(table, otherLabel, label)

def equalizeNormalizeLabeledImage(labeledImage, equivalenceTable):
    Areas      = []
    Centeroids = []
    newTable   = {}
    for r in range(labeledImage.shape[0]):
        for c in range(labeledImage.shape[1]):
            if labeledImage[r][c]:
                label = getLabel(equivalenceTable, labeledImage[r][c])
                if label not in newTable:
                    newTable[label]       = len(newTable.keys()) + 1
                    Areas.append(0)
                    Centeroids.append([0, 0])
                labeledImage[r][c]      = newTable[label]
                Areas[newTable[label]-1]         += 1
                Centeroids[newTable[label]-1][0] += c
                Centeroids[newTable[label]-1][1] += r
    for i in range(len(newTable)):
        Centeroids[i][0] /= Areas[i]
        Centeroids[i][1] /= Areas[i]
    return len(newTable), labeledImage, Areas, Centeroids


def getConnectedComponent(image: np.ndarray):   
    labeledImage      = np.zeros_like(image)
    equivalenceTable = {}
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if image[r][c]:
                neighbourLabels = getProirNeighbourLabels(labeledImage, r, c)
                if len(neighbourLabels) == 0:
                    newLabel = len(list(equivalenceTable.keys())) + 1
                    equivalenceTable[newLabel] = []
                    labeledImage[r][c] = newLabel
                else:
                    label = getLabel(equivalenceTable, min(neighbourLabels))
                    labeledImage[r][c] = label
                    for l in neighbourLabels:
                        if l != label:
                            fixEquivalenceTable(equivalenceTable, l, label)
    return equalizeNormalizeLabeledImage(labeledImage, equivalenceTable)


def getColors(numColors):
    exp = 0
    while True:
        if numColors <= exp**3:
            break
        exp += 1
    linspace = np.linspace(0, 255, exp, dtype=int)
    colors   = [[r, g, b] for r in linspace for g in linspace for b in linspace]
    return [colors[idx] for idx in np.linspace(0, len(colors)-1, numColors, dtype=int)]

def colorize(image):
    unique = np.unique(image)
    colors = getColors(len(unique))
    coloredImage = np.zeros((image.shape[0], image.shape[1], 3))
    for i in range(1, len(colors)):
        coloredImage[image == i] = colors[i]
    return coloredImage

