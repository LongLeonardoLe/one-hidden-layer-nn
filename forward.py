# This file performs the forward phase using sigmoid activation function
# It contains of functions that
#   - calculates sigmoid activation
#   - softens the outputs using softmax operation
#   - forwards to the next layer

from math import exp
import exceptions

# sigmoid activation function
def sigmoid(weight, x):
    z = 0
    # get the dot product between w and x
    for i in range(len(x)):
        z += weight[i]*x[i]
    # apply sigma(z)
    try:
        result = 1/(1+exp(0-z))
    except OverflowError:
        if (z > 0):
            result = 1.0
        else:
            result = 0.0
    return result

# softmax operation, return the output layer
def softmax(output):
    output_sum = 0.0
    # get the sum of all outputs
    for i in range(len(output)):
        output_sum += output[i]
    # soften the output by dividing by the sum
    for i in range(len(output)):
        output[i] /= output_sum
    return output

# forward to the next layer, size is the size of the next layer
# return the next layer
def forward(weight, x, size):
    nlayer = []
    for i in range(size):
        nlayer.append(sigmoid(weight[i*len(x):(i+1)*len(x)], x))
    return nlayer
