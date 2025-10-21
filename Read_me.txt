
## output = np.dot(inputs, weights) + biases:

inputs has shape (batch_size, n_inputs)

weights has shape (n_inputs, n_neurons)

output has shape (batch_size, n_neurons)

## In backpropagation:

dweights = ∂Loss/∂weights

dbiases = ∂Loss/∂biases

dinputs = ∂Loss/∂inputs
