import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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
def line_calculation(inputs, weights):
    return -(inputs[0] * weights[0] + inputs[1] * weights[1])


# inputs are of the form [width, length], weights are of the form [w0, w1, w2], and w0 is the weight for the bias node.
def single_layer_output(inputs, weights):
    sum = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2]
    # TODO: I multiplied it by 3 to make it look steeper
    return logistic(sum * 3)


def logistic(x):
    return 1/(math.pow(math.e, -x) + 1)


def main():
    data = read_file('irisdata.csv')
    color_map = {"virginica": "blue", "versicolor": "green", "setosa": "red"}
    iris_objects = classification(data, color_map)

    # General Legend
    red_patch = mpatches.Patch(color='red', label='Setosa')
    blue_patch = mpatches.Patch(color='blue', label='Virginica')
    green_patch = mpatches.Patch(color='green', label='Versicolor')
    weights = [-2.9, 0.25, 1]  # These separate the two important things nicely

    # plots for Q1
    # Q1a
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(221)
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
        line_widths.append(line_calculation([1, line_lengths[i]], weights))

    # Plot of line vs scatter
    ax2 = fig.add_subplot(222, sharey=ax1, sharex=ax1)
    ax2.scatter(iris_objects['green'].petal_lengths, iris_objects['green'].petal_widths, c='green', alpha=0.5)
    ax2.scatter(iris_objects['blue'].petal_lengths, iris_objects['blue'].petal_widths, c='blue', alpha=0.5)
    ax2.plot(line_lengths, line_widths, 'r')

    ax2.legend(handles=[green_patch, blue_patch])
    ax2.grid(True)

    # TODO: Q3d graph
    ax3 = fig.add_subplot(223, projection='3d')

    n = 100
    graph_widths = np.linspace(0, 4, n)
    graph_lengths = np.linspace(0, 7.5, n)
    x_axis, y_axis = np.meshgrid(graph_lengths, graph_widths)
    graph_outputs = []
    for x in range(0, n):
        graph_outputs.append([])
        for y in range(0, n):
            graph_outputs[x].append(single_layer_output([1, x_axis[x][y], y_axis[x][y]], weights))
    graph_outputs = np.array(graph_outputs)

    ax3.set_zlim([0, 1])
    ax3.plot_surface(x_axis, y_axis, graph_outputs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax3.set_ylabel('Petal Width')
    ax3.set_xlabel('Petal Length')

    # Actually graphing it
    plt.show()




if __name__ == "__main__":
    main()