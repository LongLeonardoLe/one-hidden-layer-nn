import struct
from array import array
import neural_network

# read data
def read_data(path):
    fdata = open(path, 'rb')
    magic_num, size, rows, cols = struct.unpack(">IIII", fdata.read(16))
    data_array = array("B", fdata.read())
    fdata.close()
    data_list = data_array.tolist()
    for i in range(len(data_lis)):
        data_list[i] /= 255.0
    return size, rows, cols, data_list

# read labels
def read_label(path):
    flbl = open(path, 'rb')
    magic_num, size = struct.unpack(">II", flbl.read(8))
    label_array = array("b", flbl.read())
    flbl.close()
    label_list = label_array.tolist()
    return label_list

# from label to target array
def label_to_target(label):
    target = []

    for i in range(len(label)):
        for j in range(10):
            if j == label[i]:
                target.append(0.9)
            else: target.append(0.1)

    return target

# calculate accuracy
def accuracy(predictions, label, size):
    correct_num = 0.0
    for i in range(size):
        if predictions[i] == label[i]:
            correct_num += 1.0
    return correct_num/size


if __name__ == "__main__":
    # read and preprocess training data and label
    train_size,rows,cols,train_data = read_data("train_data")
    train_label = read_label("train_label")
    print(train_size, rows, cols)

    # read and preprocess testing data and label
    test_size,rows,cols,test_data = read_data("test_data")
    test_label = read_label("test_label")
    print(test_size, rows, cols)

    # preprocess training label to target
    target = label_to_target(train_label)
    
    # constants: learning rate, momemtum, size of hidden layer
    eta = 0.1
    momentum = 0.9
    h_size = 20
    o_size = 10

    # initial weights
    weight_xh = neural_network.initial_weight(h_size,cols*rows+1)
    weight_ho = neural_network.initial_weight(10,h_size+1)
    
    #print(accuracy(neural_network.predict(test_data,h_size,o_size,cols,rows,weight_xh,weight_ho),test_label,test_size))

    # train the network
    weight_xh,weight_ho = neural_network.train_network(train_data,target,cols,rows,h_size,o_size,weight_xh,weight_ho,eta,momentum)

    print(accuracy(neural_network.predict(test_data,h_size,o_size,cols,rows,weight_xh,weight_ho),test_label,test_size))
