from __future__ import absolute_import, division, print_function, unicode_literals
from PIL import Image
import glob

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

inputData = np.load("./data/input.npy", allow_pickle=True)
outputData = np.load("./data/output.npy", allow_pickle=True)

# show an image
# firstInput = inputData[0]
# firstOutput = outputData[0]

# im1 = Image.fromarray(firstInput)
# im2 = Image.fromarray(firstOutput)

# im1.show()
# im2.show()
