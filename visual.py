"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Function to create synthetic dataset
def create_data(points, classes):
    X = np.zeros((points*classes, 2))          # data matrix (each row = single example)
    y = np.zeros(points*classes, dtype='uint8')  # class labels

    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number

    return X, y

# Create dataset
X, y = create_data(100, 3)

# Plot the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.show()
"""
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
nnfs.init()
X, y = vertical_data(samples=100, classes=3)


plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()