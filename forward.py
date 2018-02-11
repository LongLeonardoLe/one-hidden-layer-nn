# This file performs the forward phase using sigmoid activation function
# It contains of functions that
#   - calculates sigmoid activation
#   - softens the outputs using softmax operation
#   - forwards to the next layer
from math import exp


# sigmoid activation function
def sigmoid(weight, x):
    z = 0.0
    # get the dot product between w and x
    for i in range(len(x)):
        z += weight[i]*x[i]
    # apply sigma(z)
    return 1.0/(1.0+exp(-z))


# forward to the next layer
# size_nlayer is the size of the next layer
# return the next layer
def forward(weight, x, size_nlayer):
    nlayer = []
    for i in range(size_nlayer):
        # e.g. from weight[0] -> weight[784] is weights of
        # input[0] -> input[784] to hidden[0]
        nlayer.append(sigmoid(weight[i*len(x):(i+1)*len(x)], x))
    return nlayer
