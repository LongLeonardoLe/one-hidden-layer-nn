import struct
from array import array
import neural_network

"""# read data
def read_data(path):
    fdata = open(path, 'rb')
    magic_num, size, rows, cols = struct.unpack(">IIII", fdata.read(16))
    data_array = array("B", fdata.read())
    fdata.close()
    data_list = data_array.tolist()
    # normalize the input
    for i in range(len(data_list)):
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
    """
def data_preprocessor(file_name):
    output = []
    with open(file_name, 'r') as in_file:
        output.extend(setup_vector(x) for x in in_file)
    return np.array(output, float)

# Setup vector from input line
def setup_vector(input_line):
    # Get a line, split comma-based, and get float representation
    vec = input_line.split(',')
    ret = [float(x) for x in vec]
    # Normalize the vector
    for value in range(1, len(ret)):
        ret[value] /= 255
    ret.append(1.0)
    return tuple(ret)

# from label to target array
def label_to_target(label):
    target = []
    for i in range(len(label)):
        for j in range(10):
            # k-th target node is 0.9 for label of k
            if j == label[i]:
                target.append(0.9)
            # otherwise 0.1
            else: target.append(0.1)
    return target

# calculate accuracy
def accuracy(predictions, label, size):
    correct_num = 0.0
    print(predictions[0:10])
    print(label[0:10])
    # count the number of correct predictions
    for i in range(size):
        if predictions[i] == label[i]:
            correct_num += 1.0
    return correct_num/size


if __name__ == "__main__":
    train_data = data_preprocessor(“mnist_train.csv”)
    test_data = data_preprocessor(“mnist_test.csv”)
    rows = 28
    cols = 28
    train_size = len(train_data)/(cols*rows+2)
    test_size = len(test_data)/(cols*rows+2)
    train_label = []
    test_label = []
    for i in range(train_size):
        train_label.append(train_data[i][0])
    for i in range(test_size):
        test_label.append(test_data[i][0])
    # read and preprocess training data and label
    #train_size,rows,cols,train_data = read_data("train_data")
    #train_label = read_label("train_label")
    #train_size = 20000
    #train_data = train_data[0:train_size*cols*rows]
    #print(train_size, rows, cols)

    # read and preprocess testing data and label
    #test_size,rows,cols,test_data = read_data("test_data")
    #test_label = read_label("test_label")
    #test_size = 2000
    #print(test_size, rows, cols)

    # preprocess training label to target
    target = label_to_target(train_label)
    
    # constants: learning rate, momemtum, size of hidden layer
    eta = 0.1
    momentum = 0.9
    h_size = 20
    o_size = 10

    # initial weights
    weight_xh = neural_network.initial_weight(h_size,cols*rows+1)
    weight_ho = neural_network.initial_weight(o_size,h_size+1)
    
    print(h_size)
    prediction = neural_network.predict(test_data,h_size,o_size,cols,rows,weight_xh,weight_ho)
    before = accuracy(prediction,test_label,test_size)

    # train the network
    weight_xh,weight_ho = neural_network.train_network(train_data,target,cols,rows,h_size,o_size,weight_xh,weight_ho,eta,momentum)

    prediction = neural_network.predict(test_data,h_size,o_size,cols,rows,weight_xh,weight_ho)
    after = accuracy(prediction,test_label,test_size)
    print("Before: ", before*100)
    print("After: ", after*100)
