# create a txt file and fill it with data following some simple rules
# add in a function in basicNNobjects to intake and read the txt file
import numpy
import pandas as pd


def get_parameters():
    input_size = int(input("Input size for data? "))
    # output_size = int(input("Output size for data? ")) always 1 for now
    data_size = int(input("How many examples? "))
    return input_size, data_size


def generate_data(input_size, data_size):
    inputs = []
    targets = []
    for i in range(data_size):
        input_vector = []
        for j in range(input_size):
            x = numpy.random.randint(0, 6)
            input_vector.append(x)
        inputs.append(input_vector)
    for i in inputs:
        targets.append(get_targets(i))
    return inputs, targets


def generate_txt_pd_free(inputs, outputs):  # occasionally the input vector is empty?
    new_text = open("SFFNN_Data.txt", "w")
    new_target = open("SFFNN_Targets.txt", "w")
    for i in inputs:
        new_text.write(str(i))
        new_text.write("==========")
    for i in outputs:
        new_target.write(str(i))
        new_target.write("==========")
    new_text.close()
    new_target.close()


def generate_txt(inputs, outputs):
    inputs = pd.DataFrame(inputs)
    outputs = pd.DataFrame(outputs)
    inputs.to_csv('SFFNN_Data.txt')
    outputs.to_csv("SFFNN_Targets.txt")


# write some target rules:
# if sum of odd numbers i bigger than even numbers?


def get_targets(inputs):
    x = 0
    y = 0
    for i in inputs:
        if i % 2 == 0:
            x += i
        else:
            y += i
    if x > y:
        return 1
    else:
        return 0


new_inputs, new_outputs = generate_data(*get_parameters())
generate_txt(new_inputs, new_outputs)
