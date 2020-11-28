import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Iris:
    def __init__(self, name, petal_lengths, petal_widths):
        self.name = name
        self.petal_lengths = petal_lengths
        self.petal_widths = petal_widths


def read_file(name):
    data = []
    with open(name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                data.append(row)
    return data


def classification(data, color_map):
    # to make it more general, color coding can be removed
    flower_types = set()
    for row in data:
        flower_types.add(row[4])

    iris_objects = {}
    for type in flower_types:
        iris_objects[color_map[type]] = (Iris(type, [], []))

    for row in data:
        iris_objects[color_map[row[4]]].petal_widths.append(float(row[3]))
        iris_objects[color_map[row[4]]].petal_lengths.append(float(row[2]))

    return iris_objects


# this is used for 1c
def line_calculation(length):
    return -1/4 * length + 2.9


# inputs are of the form [width, length], weights are of the form [w0, w1, w2], and w0 is the weight for the bias node.
def singleLayerOutput(inputs, weights):
    sum = 1 * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2]
    return logisticFct(sum)


def logisticFct(x):
    return 1/(math.pow(math.e, -x) + 1)


def main():
    data = read_file('irisdata.csv')
    color_map = {"virginica": "blue", "versicolor": "green", "setosa": "red"}
    iris_objects = classification(data, color_map)

    # General Legend
    red_patch = mpatches.Patch(color='red', label='Setosa')
    blue_patch = mpatches.Patch(color='blue', label='Virginica')
    green_patch = mpatches.Patch(color='green', label='Versicolor')

    # plots for Q1
    # Q1a
    f, (ax1, ax2) = plt.subplots(1, 2, sharex='none', sharey='none', figsize=(10, 5))
    ax1.scatter(iris_objects['green'].petal_lengths, iris_objects['green'].petal_widths, c='green', alpha=0.5)
    ax1.scatter(iris_objects['blue'].petal_lengths, iris_objects['blue'].petal_widths, c='blue', alpha=0.5)

    ax1.set_ylim([0.5, 3])
    ax1.set_xlim([2.5, 7.5])
    ax1.set_ylabel('Petal Width')
    ax1.set_xlabel('Petal Length')
    ax1.legend(handles=[green_patch, blue_patch])
    ax1.grid(True)

    # Q1c
    # Defining Boundary Line
    line_lengths = np.linspace(1, 7.5, 100)
    line_widths = []
    for i in range(len(line_lengths)):
        line_widths.append(line_calculation(line_lengths[i]))

    # Plot of line vs scatter
    ax2.scatter(iris_objects['green'].petal_lengths, iris_objects['green'].petal_widths, c='green', alpha=0.5)
    ax2.scatter(iris_objects['blue'].petal_lengths, iris_objects['blue'].petal_widths, c='blue', alpha=0.5)
    ax2.plot(line_lengths, line_widths, 'r')

    ax2.set_ylim([0.5, 3])
    ax2.set_xlim([2.5, 7.5])
    ax2.set_ylabel('Petal Width')
    ax2.set_xlabel('Petal Length')
    ax2.legend(handles=[green_patch, blue_patch])
    ax2.grid(True)

    # Actually graphing it
    plt.show()

    # now lets make a sicc 3d graph (work in progress)
    weights = [2.9, -1, -1 / 4]  # These separate the two important things nicely


    '''
    n = 100
    graphWidths = np.linspace(0, 2.6, n)
    graphLengths = np.linspace(1, 7, n)
    X, Y = np.meshgrid(graphWidths, graphLengths)
    graphOutputs = []
    print(len(inputMesh[0]), "is the input mesh length")
    for i in range(len(inputMesh[0])):
        input = [inputMesh]
        graphOutputs.append(singleLayerOutput(inputMesh, weights))
    '''




if __name__ == "__main__":
    main()