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
        self.weights = self.create_weights(weight_size)

    def create_weights(self, weight_size):
        weights = []
        for i in range(weight_size):
            weights.append(numpy.random.randn())
        return weights

    def calculate(self, inputs):
        value = numpy.dot(inputs, self.weights)
        value += self.bias
        self.output = sigmoid(value)
        return self.output

    def descend(self, target, inputs, rate):  # this seems inelegant...better strategy?
        # move all weights and the bias such that output is closer to target
        cost_p = 2 * (self.output - target)
        output_p = sigmoid_p(self.output)
        for i in range(len(self.weights)):
            weight_p = inputs[i]
            dc_dw = cost_p * output_p * weight_p
            self.weights[i] = self.weights[i] - rate * dc_dw
        self.bias = self.bias - rate * cost_p * output_p

    def targets(self):  # when each node descends have it pass on a vector of targets?
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
        x = 0
        for i in self.nodes:
            i.descend(targets[x])
            x = x + 1

    def get_targets(self):  # is there a better way to do this?
        # iterates through every node
        # takes the values the node wants to shift the previous targets to
        # averages all of them out and passes them down to be used as targets for the next descend
        x = len(self.nodes[0].weights)
        for i in self.nodes:
            for j in i.weights:
                pass

        pass


class Network:

    def __init__(self, layers):
        self.layers = layers
        # add in new code to deal with different types of layers?

    def calculate(self, inputs):
        for i in self.layers:
            inputs = i.calculate(inputs)
        return inputs

    def descend(self, targets):
        for i in self.layers:
            i.descend(targets)
            # find a new target


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
    pass


def write_data_pass():
    pass
# create a memory node type to inherent from node class
