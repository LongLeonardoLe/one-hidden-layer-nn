import forward
import backprop
import random

# initial weights from 1 layer to another in the network in the range [-0.05,0.05]
def initial_weight(y_size,x_size):
    weight = []
    for i in range(y_size*x_size):
        weight.append(random.uniform(-0.04999999,0.04999999))
    return weight

# train the network, return the input-to-hidden and hidde-to-output
# weights lists. the function is designated for MNIST
# h_size is the number of hidden nodes
def train_network(data, target, cols, rows, h_size, o_size, weight_xh, weight_ho, eta, momentum):
    # number of element, in this case should be 60000
    size = len(data)/(cols*rows)
    # initial weights for input-to-hidden and hidden-to-output
    if not weight_xh:
        weight_xh = initial_weight(h_size,cols*rows+1)
    if not weight_ho:
        weight_ho = initial_weight(10,h_size+1) # 10 for MNIST
    # previous update weight
    prev_xh = []
    prev_ho = []

    for i in range(size):
        print(i)
        x = data[i:(i+rows*cols)] # 784 (28*28) input nodes
        x.append(1) # bias
        # forward from input to hidden layer and from hidden to output layer
        hidden = forward.forward(weight_xh,x,h_size)
        hidden.append(1) # bias
        output = forward.forward(weight_ho,hidden,o_size)
        # calculate errors in hidden and output layers
        error_output = backprop.error_output(output,target[i:(i+o_size)])
        error_hidden = backprop.error_hidden(error_output,hidden,weight_ho)
        # update weights
        weight_ho,prev_ho = backprop.update_weights(eta,momentum,weight_ho,error_output,hidden,prev_ho)
        weight_xh,prev_xh = backprop.update_weights(eta,momentum,weight_xh,error_hidden,x,prev_xh)
    
    return weight_xh, weight_ho

# output of 1 input
def single_output(x, h_size, o_size, w_xh, w_ho):
    # get the output layer for 10 nodes
    hidden = forward.forward(w_xh,x,h_size-1)
    hidden.append(1)
    o = forward.forward(w_ho,hidden,o_size)
    # find the max output node value and add it to prediction
    result = 0

    for i in range(len(o)):
        if o[i] > o[result]:
            result = i

    return result

# predict the output
def predict(data, h_size, o_size, cols, rows, w_xh, w_ho):
    prediction = []
    size = len(data)/(cols*rows)

    for i in range(size):
        # take 784 elements in the data from i, 28x28 
        x = data[i:(i+rows*cols)]
        x.append(1) # add bias
        # add output for each input
        prediction.append(single_output(x,h_size+1,o_size,w_xh,w_ho))
        i += rows*cols

    return prediction
