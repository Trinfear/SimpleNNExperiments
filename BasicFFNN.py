#!python3
# create new nn generation environment
# keep decentralized class organization to better add new node or layer types

import numpy
import matplotlib.pyplot as plt
import re


def sigmoid(x):
    return 1/(1 - numpy.exp(-x))


def sigmoid_p(x):
    return x * (1 - x)


class Node:

    def __init__(self, weight_size):
        self.output = 0
        self.bias = numpy.random.randn()
        self.weights = []
        for i in range(weight_size):
            self.weights.append(numpy.random.randn())

    def calculate(self, inputs):
        value = numpy.dot(inputs, self.weights)
        value += self.bias
        self.output = sigmoid(value)
        return self.output

    def descend(self, error, inputs, rate):
        dp_dc = error * sigmoid_p(self.output)
        errors = []
        for i in range(len(self.weights)):
            weight_p = inputs[i]
            dc_dw = dp_dc * weight_p
            dp_dw = dp_dc * self.weights[i]
            self.weights[i] = self.weights[i] - rate * dc_dw
            errors.append(dp_dw)
        self.bias = self.bias - rate * dp_dc
        return errors


class RecurrentNode:  # should this inherit from normal node? yes, at least for practice
    # TODO add in a a node which acts like a normal one but takes in its previous output as input
    # previous output can be multiplied by new intake
    # or previous output can be added to new intake before activation
    pass


class LSTMNode:  # does this inherit from normal node or a layer?
    # TODO add in functionality for LSTM Nodes
    # node intakes a value (or values?)
    # value(s?) is multiplied by a remember gate
    # current value(s?) is multiplied by a forget gate
    # the above two are added together
    # the network keeps the above as its new remember set, and passes on an activated set of values

    # use new information to prime forget gate
    # multiply forget gate by previous cell state
    # use new information to prime remember gate
    # add remember gate to cell state
    #
    pass


class BaseLayer:

    def __init__(self, nodes):
        self.nodes = nodes

    def calculate(self, inputs):
        output = []
        for i in self.nodes:
            output.append(i.calculate(inputs))
        return output

    def descend(self, targets):
        # targets is a list for each node and needs to be a list
        x = 0   # TODO this is a bad way to implement it, fix this
        errors = []
        for i in self.nodes:
            error = i.descend(targets[x])
            for j in range(len(error)):
                if errors[j]:
                    errors[j] += error[j]
                else:
                    errors[j] = 0
                    errors[j] += error[j]
            x = x + 1
        for i in range(len(errors)):
            errors[i] = errors[i]/len(self.nodes)
        return errors


class ConLayer:  # should this inherit from base layer?
    # TODO add in functionality for a con layer?
    # works like regular layer by only connects to some of the previous layer
    pass


class Network:

    def __init__(self, layers):
        self.layers = layers
        # add in new code to deal with different types of layers?

    def calculate(self, inputs):
        # do inputs need to be checked?
        for i in self.layers:
            inputs = i.calculate(inputs)
        return inputs

    def descend(self, targets):
        if type(targets) != list:
            targets = [targets]
        for i in reversed(self.layers):
            targets = i.descend(targets)


def generate_network(layers, input_size):
    # intake layer parameters, and generates a network based on it
    layer_set = []
    for i in layers:
        nodes = []
        for j in range(i):
            node = Node(input_size)
            nodes.append(node)
        layer = BaseLayer(nodes)
        layer_set.append(layer)
    network = Network(layer_set)
    return network


def get_parameters():
    network_size = int(input("How many layers? Not counting inputs"))
    layers = []
    for i in range(network_size):
        size = int(input("size of layer " + str(i) + "? "))
        layers.append(size)
    input_size = input("How large will the input vector be? ")
    return layers, input_size


def train(rounds):
    for i in range(rounds):
        pass
    pass


def get_data():
    # TODO read and intake a text file and convert it to readable data
    # TODO intake a separate text file for targets
    pass


def write_data_pass():
    pass
# TODO create a memory node type to inherent from node class
# TODO test to make sure the current functions work properly
