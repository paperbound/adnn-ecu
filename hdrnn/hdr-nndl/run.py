# Run HDRNN as described in http://neuralnetworksanddeeplearning.com/chap1.html
# @author Prasanth Shaji
#
# Please see README.md before running

# Load the data set
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

# Configure and train the network
import network
net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3, test_data=test_data)

# Dump weights
net.dump_weights("bias", "weights")
