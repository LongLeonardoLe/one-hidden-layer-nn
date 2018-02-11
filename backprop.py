# This file performs the back propagation and updates weights of the network

# error for output node
def error_output(output, target):
    errorout = []
    for i in range(len(output)):
        errorout.append(output[i]*(1-output[i])*(target[i]-output[i]))
    return errorout

# error for hidden unit
def error_hidden(err_output, hidden, weight_ho):
    errorhid = []
    # calculate error of every node in hidden layer except bias
    for i in range(1, len(hidden)):
        # sum of w_ij*error_ouput_j for j in outputs unit
        eo_sum = 0.0
        for j in range(len(err_output)):
            eo_sum += err_output[j]*weight_ho[j*len(hidden)+i]
        errorhid.append(hidden[i]*(1-hidden[i])*eo_sum)
    return errorhid

# calculate the delta_w_kj which is the needed update value
def value_update(eta, momentum, error, node_value, prev_update):
    return eta*error*node_value + momentum*prev_update

# update hidden-to-output weights, eta is the learning rate
def update_weights(eta, momentum, weight, error, layer, prev):
    # if the previous update weight list is empty, initial
    if not prev:
        prev = []
        for i in range(len(weight)):
            prev.append(0)
    # run through the previous layer (hidden)
    for i in range(len(layer)):
        # update weights to every output nodes from ith hidden node
        for j in range(len(error)):
            pos = i+j*len(layer)
            delta = value_update(eta, momentum, error[j], layer[i], prev[pos])
            weight[pos] += delta
            prev[pos] = delta

    return weight, prev
