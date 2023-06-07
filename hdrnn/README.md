# Handwritten Digit Recognition Neural Network

HDR-NN implementations using MNIST in `C`, `C++`, and `Python`.

## Network Description

HDR-NN weights and biases are stored in a simple binary file format. The file extension is `.nn`.

After a magic number, the next input is the _network shape_, followed by the network biases, and ends with the network weights.

Note that the HDR-NN network will always have a 784 dimension input layer and 10 neuron output layer.

The following shows the file contents numbered by byte number.

```
1.  MAGIC NUMBER
2.  NUMBER OF NETWORK SHAPE DESCRIPTION NUMBERS (2 in this example)
3.  SIZE OF HIDDEN LAYER 1 (1 in this example)
4.  SIZE OF HIDDEN LAYER 2 (1 in this example)
5.  BIAS VALUE 1 OF NEURON 1 LAYER 1
9.  BIAS VALUE 1 OF NEURON 1 LAYER 2
13. WEIGHTS VALUE 1 OF LAYER 1 (word-aligned float is 32 bits)
17. WEIGHTS VALUE 2 OF LAYER 1
...
3149. WEIGHTS VALUE 784 OF NEURON 1 LAYER 1
3153. WEIGHTS VALUE 1 OF NEURON 1 LAYER 2
...
6289. WEIGHTS VALUE 784 OF NEURON 1 LAYER 2
```
