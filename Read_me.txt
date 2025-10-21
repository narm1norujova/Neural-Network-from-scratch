m = number of examples.
n = number of features + label (should be 785).
Shuffle so we train on random order (better learning).

## output = np.dot(inputs, weights) + biases:

inputs has shape (batch_size, n_inputs)

weights has shape (n_inputs, n_neurons)

output has shape (batch_size, n_neurons)

## In backpropagation:

dweights = ∂Loss/∂weights

dbiases = ∂Loss/∂biases

dinputs = ∂Loss/∂inputs
