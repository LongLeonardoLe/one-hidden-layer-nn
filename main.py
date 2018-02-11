import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import neural_network


# preprocess data
def data_preprocessor(file_name):
    output = []
    with open(file_name, 'r') as in_file:
        output.extend(setup_vector(x) for x in in_file)
    return np.array(output, float)


# setup vector from input line
def setup_vector(input_line):
    # get a line, split comma-based, and get float representation
    vector = input_line.split(',')
    result = [float(x) for x in vector]
    # add bias
    result.insert(1, 1.0)
    # Normalize the vector
    for value in range(2, len(result)):
        result[value] /= 255
    return tuple(result)

"""
# from label to target array
def label_to_target(label):
    target = []
    for i in range(len(label)):
        for j in range(10):
            # k-th target node is 0.9 for label of k
            if j == label[i]:
                target.append(0.9)
            # otherwise 0.1
            else: 
                target.append(0.1)
    return target
"""

"""
# calculate accuracy
def accuracy(predictions, label):
    correct_num = 0.0
    # count the number of correct predictions
    for i in range(len(label)):
        if predictions[i] == label[i]:
            correct_num += 1.0
    return correct_num/len(label)
"""


if __name__ == "__main__":
    # constants: learning rate, momentum, size of hidden layer, # of rows and cols
    eta = 0.1
    momentum = 0.0
    h_size = 20
    o_size = 10
    rows = 28
    cols = 28

    # read and preprocess input files
    train_data = data_preprocessor("mnist_train.csv")
    test_data = data_preprocessor("mnist_test.csv")

    # retrieve labels from data
    train_label = []
    test_label = []
    for i in range(len(train_data)):
        train_label.append(train_data[i][0])
    for i in range(len(test_data)):
        test_label.append(test_data[i][0])
    print(len(train_data), len(test_data), len(train_label), len(test_label))

    # preprocess training label to target
    #target = label_to_target(train_label)
    
    # initial weights
    weight_xh = neural_network.initial_weight(h_size, cols*rows+1)
    weight_ho = neural_network.initial_weight(o_size, h_size+1)
    
    #prediction = neural_network.predict(test_data, h_size, o_size, weight_xh, weight_ho)
    #print("Before: ", accuracy_score(test_label, prediction)*100)

    # train the network
    for i in range(1):
        neural_network.train_network(train_data, h_size, o_size, weight_xh, weight_ho, eta, momentum)

    prediction = neural_network.predict(test_data, h_size, o_size, weight_xh, weight_ho)
    print("After: ", accuracy_score(test_label, prediction)*100)
