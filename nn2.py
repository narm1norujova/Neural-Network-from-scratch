# ---------------------- Imports ----------------------
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
# ---------------------- Layer ----------------------
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # CHANGED: use randn for both positive & negative weights
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # store inputs for use in backward pass
        self.inputs = inputs   # NEW
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):  # NEW
        # Gradients of weights, biases, and inputs
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# ---------------------- ReLU ----------------------
class Activation_ReLu:
    def forward(self, inputs):
        self.inputs = inputs   # NEW
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):  # NEW
        # Pass gradient only where input > 0
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# ---------------------- Softmax ----------------------
class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs  # NEW
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # not used when combined with loss
    def backward(self, dvalues):  
        self.dinputs = dvalues.copy()

# ---------------------- Loss Base Class ----------------------
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        batch_loss = np.mean(sample_losses)
        return batch_loss

# ---------------------- Categorical Cross-Entropy ----------------------
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# ---------------------- Softmax + Cross-Entropy Combined (faster backprop) ----------------------
class Activation_Softmax_Loss_CategoricalCrossEntropy:  # NEW
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

# ---------------------- Optimizer ----------------------
class Optimizer_SGD:  # NEW
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

# ---------------------- Data ----------------------
X, y = spiral_data(samples=10000, classes=3)

# ---------------------- Model Setup ----------------------
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLu()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossEntropy()
optimizer = Optimizer_SGD(learning_rate=0.01)  # NEW
softmax_loss = Activation_Softmax_Loss_CategoricalCrossEntropy()  # NEW

# ---------------------- Training Loop ----------------------
for epoch in range(100001):  # NEW

    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Compute loss
    loss = loss_function.calculate(activation2.output, y)

    # Predictions and accuracy
    predictions = np.argmax(activation2.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    # Print every 1000 epochs
    if not epoch % 1000:
        print(f"Epoch {epoch} | Loss: {loss:.3f} | Accuracy: {accuracy:.3f}")

    # Backward pass
    softmax_loss.backward(activation2.output, y)
    dense2.backward(softmax_loss.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

# ---------------------- Final Output ----------------------
print("\nFinal Predictions:", predictions[:10])
print("Final Accuracy:", accuracy)