import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2
# doesnt matter which comes first inputs or weights:
output = np.dot(inputs, weights) + bias
print(output)

"""
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
bias = 2
# weights should come first:
output = np.dot(weights, inputs) + bias
"""