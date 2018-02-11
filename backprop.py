# This file performs the back propagation and updates weights of the network


# error for output nodes, o_i(1-o_i)(t_i-o_i)
def error_output(output, target):
    errorout = []
    # loop 10 which is len(output)
    for i in range(len(output)):
        # target = 0.9 at i-th output
        if i == target:
            errorout.append(output[i]*(1.0-output[i])*(0.9-output[i]))
        # otherwise, target = 0.1
        else:
            errorout.append(output[i]*(1.0-output[i])*(0.1-output[i]))
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
        errorhid.append(hidden[i]*(1.0-hidden[i])*eo_sum)
    return errorhid


# calculate the delta_w_kj which is the needed update value
def value_update(eta, momentum, error, node_value, prev_update):
    return eta*error*node_value + momentum*prev_update


# update layer-to-layer weights, eta is the learning rate
# error is of the output layer when update hidden-to-output
# and of the hidden for update input-to-hidden
def update_weights(eta, momentum, weight, error, layer, prev):
    # run through the previous layer
    for i in range(len(layer)):
        # update weights to every output nodes from ith hidden node
        for j in range(len(error)):
            pos = i+j*len(layer)
            prev[pos] = value_update(eta, momentum, error[j], layer[i], prev[pos])
            weight[pos] += prev[pos]
