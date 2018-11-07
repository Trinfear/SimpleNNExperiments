#!python3
"""
This is a simple Feed forward Neural network
It cannot be used for as wide a range as the other object based models, but should run much faster
Rather than generating interchangeable objects for nodes and layers, this uses a single object "Network"
This keeps all weights in a single 3d array and all biases in a 2d array
Also contains functions to read in values from a text file and generate and train a network object
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_p(x):
    return x * (1 - x)


class Network:

    def __init__(self, layer_set, input_size):
        # create layers
        layers = []     # 3d matrix of weights
        biases = []     # 2d array of biases
        for i in layer_set:
            layer = []
            bias_layer = []
            for j in range(i):
                node = []
                for k in range(input_size):
                    node.append(np.random.randn())
                layer.append(node)
                bias_layer.append(np.random.randn())
            input_size = i
            layers.append(layer)
            biases.append(bias_layer)
        layers = np.array(layers)
        biases = np.array(biases)
        self.layers = layers        # 3d matrix of weights
        self.biases = biases        # 2d array of biases
        self.values = []            # 2d array of outputs

    def calculate(self, inputs):
        self.values = []  # clear out previous value set
        self.values.append(inputs)
        # would appending inputs here help?
        for i in range(len(self.layers)):   # iterates through layers
            outputs = []
            for j in range(len(self.layers[i])):    # iterates through nodes
                value = np.dot(self.layers[i][j], inputs)
                value += self.biases[i][j]
                value = sigmoid(value)
                outputs.append(value)
            inputs = outputs
            self.values.append(inputs)
        return inputs

    def descend(self, expected):  # change so inputs are appended to values?
        costs = []
        # TODO make sure this can deal with vector targets
        expected = [expected]  # this should get fixed before it gets passed here
        for x in expected:
            costs.append(self.values[len(self.values) - 1][0] - x)  # fix this so its not just [0]
        for i in reversed(range(1, len(self.values))):  # breaks down to 2d array of weights
            layer = self.layers[i-1]
            new_costs = []  # calculate costs for next layer
            for t in range(len(layer[0])):
                new_costs.append(0.0)  # initialize new costs
            for j in range(len(layer)):  # fetches 1d array of weights
                dc_dz = costs[0] * sigmoid_p(self.values[i][j])
                for k in range(len(layer[j])):      # iterates through each weight
                    new_costs[k] += dc_dz * layer[j][k]  # find costs for previous nodes from current
                    dc_dw = dc_dz * self.values[i-1][k]  # value of node weight is connected to
                    layer[j][k] = layer[j][k] - 0.1 * dc_dw
                self.biases[i-1][j] = self.biases[i-1][j] - 0.1 * dc_dz
            costs = new_costs  # set costs to costs for next layer
            for n in range(len(costs)):  # normalize costs
                costs[n] = costs[n]/len(layer)


def train(rounds, network, inputs, targets):    # always moves towards predicting 0
    costs = []
    for i in range(rounds):
        x = np.random.randint(0, len(inputs))
        output = network.calculate(inputs[x])
        network.descend(targets[x])
        output = np.array([output])
        cost = (output - targets[x])**2
        cost = np.array(cost)
        cost = np.sum(cost)
        costs.append(cost)
    return costs


def get_values():
    inputs = []
    outputs = []
    # SFFNN_Data.txt
    # intakes training set
    text_data = open("SFFNN_Data.txt", "r")
    data = text_data.read()
    text_data.close()
    text_data = open("SFFNN_Targets.txt", "r")
    targets = text_data.read()
    text_data.close()
    data = data.split("==========")  # separate data and targets using 10 * =
    for i in data:  # split into each set of data
        inputs_set = []
        for j in i:
            if j.isdecimal():
                inputs_set.append(int(j))
        inputs.append(inputs_set)
    targets = targets.split('==========')
    for i in targets:
        # target_set = []
        for j in i:
            if j.isdecimal():
                # target_set.append(int(j)) re add this for higher level vectors?
                outputs.append(int(j))
        # outputs.append(target_set)
    return inputs, outputs


def generate_network():
    # get inputs for network size and stuff
    layers = int(input("How many layers for the NN? "))
    layer_set = []
    for i in range(layers):
        size = int(input("size of layer " + str(i + 1) + "? "))
        layer_set.append(size)
    input_size = int(input("How large will the input vector be? "))
    network = Network(layer_set, input_size)
    return network


# input_set, output_set = get_values()
input_set = [[0, 1, 1], [0, 0, 0], [1, 1, 1], [1, 1, 0]]
output_set = [0, 0, 1, 0]
NN = generate_network()
cost_totals = train(100000, NN, input_set, output_set)
print(NN.calculate([1, 0, 1]))
plt.plot(cost_totals)
plt.show()
