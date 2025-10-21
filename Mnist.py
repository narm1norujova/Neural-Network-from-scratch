import numpy as np
print(np.__version__)
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\afaqo\Downloads\mnist_train.csv.zip", compression='zip')
#print(data.head())

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
# print(n)
# print(m)
# print(data.shape)

data_dev = data[0:1000].T      # Take the first 1000 examples and transpose
Y_dev = data_dev[0]            # Labels
X_dev = data_dev[1:n]          # Features
X_dev = X_dev / 255.           # Normalize

data_train = data[1000:m].T    
Y_train = data_train[0]        
X_train = data_train[1:n]      
X_train = X_train / 255.       
_, m_train = X_train.shape     # m_train = number of training examples

# Initialize parameters
def init_params():
    w1 = np.random.rand(10, 784) - 0.5  # weights for hidden layer
    b1 = np.random.rand(10, 1) - 0.5    # biases for hidden layer
    w2 = np.random.rand(10, 10) - 0.5   # weights for output layer
    b2 = np.random.rand(10, 1) - 0.5    # biases for output layer
    return w1, b1, w2, b2

   
# Forward propagation
def forward_prop(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

# Define activation functions
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # stability trick
    A = expZ / np.sum(expZ, axis=0, keepdims=True)
    return A
# Ensure labels are integers for indexing.
Y_train = Y_train.astype(int)
Y_dev = Y_dev.astype(int)

# Backpropagation
def back_prop(z1, a1, z2, a2, w2, Y, X):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    dZ2 = a2 - one_hot_Y # error

    m = X.shape[1]

    dW2 = 1/m * dZ2.dot(a1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = w2.T.dot(dZ2) * (z1 > 0) # error
    
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# Parameter updates
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

# Calculate accuracy
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Gradient descent
def gradient_descent(X, Y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X)
        dW1, db1, dW2, db2 = back_prop(z1, a1, z2, a2, w2, Y, X)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha)
        if i % 100 == 0:
            print("Iterations: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(a2), Y))
    return w1, b1, w2, b2

# Actually train the model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations=500, alpha=0.1)

# Making predictions
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = np.argmax(A2, 0)
    return predictions

# Testing and evaluation
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show(block=True)



# Test on dev set
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print("Dev Set Accuracy:", get_accuracy(dev_predictions, Y_dev))
test_prediction(0, W1, b1, W2, b2)