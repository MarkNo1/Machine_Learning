#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Nov 25 15:45:06 2016

@author: markno1
"""
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# Path for the Images
static_path = '/Users/marcotreglia/.bin/ML/Resource/coil-100/'
# Classes
classes = [5, 6, 7, 8]

##########################################################################
#                            LOAD FILE                                   #
##########################################################################
# Load 1 image


def load1_image(image_name):
    return Image.open(static_path + image_name)

    # Load 1 classes of images


def load1_class(classes):
    img_class = []
    name_classes = "obj" + str(classes) + "__"
    for i in range(72):
        img_class.append(load1_image(name_classes + str(i * 5) + ".png"))
    return img_class

    # Load  #Classes  images


def load_class():
    img_classes = []
    for i in range(len(classes)):
        img_classes.extend(load1_class(classes[i]))
    return img_classes

    # Convert 1 Classes of Images in Matrix


def convertImg_matrix(array_img):
    img_matrix = []
    for img in array_img:
        img_matrix.append(np.asarray(img))
    return np.asarray(img_matrix)


##########################################################################

# Ravel the data
# Reshape the data
# Mean = 0 | Variance = 1
def standardize(X):
    X.ravel()
    X = X.reshape(len(classes) * 72, -1)
    from sklearn import preprocessing
    return preprocessing.scale(X)

# Create Label for 1 classes


def y1(classes):
    y = []
    for i in range(72):
        y.append(classes)
    return y

# Create label for all classes


def y():
    y = []
    col = ['red', 'blue', 'green', 'yellow']
    for i in range(len(classes)):
        y.extend(y1(col[i]))
    return y

# Plot the data


def plot(X_t, value1, value2):
    plt.grid()
    plt.title("Principal Component Analisys")
    plt.scatter(X_t[:, value1], X_t[:, value2], c=y())
    plt.legend()
    plt.xlabel('Principal Componet ' + str(value1 + 1))
    plt.ylabel('Principal Componet ' + str(value2 + 1))
    plt.tight_layout()
    plt.show()
    plt.close()
