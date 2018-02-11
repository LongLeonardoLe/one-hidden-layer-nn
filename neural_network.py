import forward
import backprop
import random

# initial weights from 1 layer to another in the network in the range (-0.05,0.05)
def initial_weight(y_size, x_size):
    weight = []
    for i in range(y_size*x_size):
        weight.append(random.uniform(-0.04999999,0.04999999))
    return weight 

# train the network, return the input-to-hidden and hidde-to-output
# weights lists. the function is designated for MNIST
# h_size is the number of hidden nodes
def train_network(data, target, cols, rows, h_size, o_size, weight_xh, weight_ho, eta, momentum):
    # initial weights if not yet exist 
    if not weight_xh: # input to hidden 
        weight_xh = initial_weight(h_size, cols*rows+1)
    if not weight_ho: # hidden to output
        weight_ho = initial_weight(o_size, h_size+1) 

    # previous update weight
    prev_xh = []
    prev_ho = []

    for i in range(len(data)):
        # take the input
        x = data[i][1:]

        # forward from input to hidden
        hidden = [1] # initial contain only bias
        hidden.extend(forward.forward(weight_xh, x, h_size))
        # forward from hidden to output
        output = forward.forward(weight_ho, hidden, o_size)

        # calculate errors in hidden and output layers
        error_output = backprop.error_output(output, target[i*o_size:(i+1)*o_size])
        error_hidden = backprop.error_hidden(error_output, hidden, weight_ho)

        # update weights
        weight_ho, prev_ho = backprop.update_weights(eta, momentum, weight_ho, error_output, hidden, prev_ho)
        weight_xh, prev_xh = backprop.update_weights(eta, momentum, weight_xh, error_hidden, x, prev_xh)
    
    return weight_xh, weight_ho

# output of 1 input
def single_output(x, h_size, o_size, w_xh, w_ho):
    # forward phase
    # from input to hidden layer
    hidden = [1]
    hidden.extend(forward.forward(w_xh, x, h_size))
    # from hidden to output layer
    output = forward.forward(w_ho, hidden, o_size)

    # find the max output node value and add it to prediction
    result = 0
    for i in range(1, len(output)):
        if output[i] > output[result]:
            result = i

    return result

# predict the output
def predict(data, h_size, o_size, cols, rows, w_xh, w_ho):
    prediction = []

    for i in range(len(data)):
        x = data[i][1:]
        # add output for each input
        prediction.append(single_output(x, h_size, o_size, w_xh, w_ho))

    return prediction
