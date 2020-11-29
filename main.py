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


# Calculate MSE given array of iris_objects, weights and names of objects to compare
# error(iris_objects, weights, ['green', 'blue'])
def mean_squared_error(iris_objects, weights, names):
	# Computing the difference between the expected and observed data point
	n_of_points = 0
	error_sum = 0

	for name in names:
		for (x, y) in zip(iris_objects[name].petal_lengths, iris_objects[name].petal_widths):
			n_of_points += 1
			difference = line_calculation([x, y], weights) - float(y)
			error_sum = error_sum + math.pow(difference, 2)

	# Dividing by the number of data points
	error = error_sum / n_of_points
	return error


# Calculate Error given array of iris_objects, weights and names of objects to compare
# error(iris_objects, weights, ['green', 'blue'], {'green' : float(0), 'blue' : float(1)})
def error(iris_objects, weights, names, desired_output):
	# Computing the difference between the expected and observed data point
	error_sum = 0

	for name in names:
		for (x, y) in zip(iris_objects[name].petal_lengths, iris_objects[name].petal_widths):
			difference = single_layer_output([1, x, y], weights) - desired_output[name]
			error_sum = error_sum + math.pow(difference, 2)

	# Dividing by the number of data points
	error = error_sum / 2
	return error


# Uses original data to classify by the given names and single_layer_output
# decision(iris_objects, weights, ['green', 'blue'], {'0' : 'green', '1' : 'blue'})
def decision(iris_objects, weights, names, desired_class):
	decided_class = {names[0]: Iris(names[0], [], []), names[1]: Iris(names[1], [], [])}

	for name in names:
		for (x, y) in zip(iris_objects[name].petal_lengths, iris_objects[name].petal_widths):
			if single_layer_output([1, x, y], weights) > 0.5:
				decided_class[desired_class['1']].petal_lengths.append(x)
				decided_class[desired_class['1']].petal_widths.append(y)
			else:
				decided_class[desired_class['0']].petal_lengths.append(x)
				decided_class[desired_class['0']].petal_widths.append(y)

	return decided_class


def scatter_plot(plot, iris_objects, names):
	for name in names:
		plot.scatter(iris_objects[name].petal_lengths, iris_objects[name].petal_widths, c=name, alpha=0.5)


def main():
	data = read_file('irisdata.csv')
	color_map = {"virginica": "blue", "versicolor": "green", "setosa": "red"}
	iris_objects = classification(data, color_map)

	# General Legend
	red_patch = mpatches.Patch(color='red', label='Setosa')
	blue_patch = mpatches.Patch(color='blue', label='Virginica')
	green_patch = mpatches.Patch(color='green', label='Versicolor')
	weights = [-3.3, 0.35, 1]  # These separate the two important things nicely
	large_error_weights = [-1.3, 0.25, 1]

	# 1st Figure
	# TODO: set titles
	fig1 = plt.figure(1, figsize=(8, 6))
	fig1.subplots_adjust(hspace=0.3)
	ax12 = fig1.add_subplot(211)

	# plots for Q1
	# Q1a
	ax1 = fig1.add_subplot(221, sharey=ax12, sharex=ax12)
	ax1.scatter(iris_objects['green'].petal_lengths, iris_objects['green'].petal_widths, c='green', alpha=0.5)
	ax1.scatter(iris_objects['blue'].petal_lengths, iris_objects['blue'].petal_widths, c='blue', alpha=0.5)

	# Q1c
	# Defining Boundary Line
	line_lengths = np.linspace(1, 7.5, 100)
	line_widths = []
	for i in range(len(line_lengths)):
		line_widths.append(line_calculation([1, line_lengths[i]], weights))

	# Plot of line vs scatter
	ax2 = fig1.add_subplot(222, sharey=ax12, sharex=ax12)
	ax2.scatter(iris_objects['green'].petal_lengths, iris_objects['green'].petal_widths, c='green', alpha=0.5)
	ax2.scatter(iris_objects['blue'].petal_lengths, iris_objects['blue'].petal_widths, c='blue', alpha=0.5)
	ax2.plot(line_lengths, line_widths, 'r')

	# Common Labels
	# Turn off axis lines and ticks of the big subplot
	ax12.spines['top'].set_color('none')
	ax12.spines['bottom'].set_color('none')
	ax12.spines['left'].set_color('none')
	ax12.spines['right'].set_color('none')
	ax12.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
	ax12.set_ylabel('Petal Width')
	ax12.set_xlabel('Petal Length')
	ax12.set_ylim([0.5, 3])
	ax12.set_xlim([2.5, 7.5])

	# Legends
	ax1.legend(handles=[green_patch, blue_patch], loc='upper left')
	ax2.legend(handles=[green_patch, blue_patch], loc='upper left')
	ax1.set_title("Iris data")
	ax2.set_title("Decision Boundary")
	ax1.grid(True)
	ax2.grid(True)

	# TODO: Q1d graph
	ax3 = fig1.add_subplot(223, projection='3d')

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
	ax3.set_title('Logistic Curve')

	# Q1e graph
	decided_classes = decision(iris_objects, weights, ['green', 'blue'], {'0': 'green', '1': 'blue'})
	ax4 = fig1.add_subplot(224)
	ax4.scatter(decided_classes['green'].petal_lengths, decided_classes['green'].petal_widths, c='green', alpha=0.5)
	ax4.scatter(decided_classes['blue'].petal_lengths, decided_classes['blue'].petal_widths, c='blue', alpha=0.5)
	ax4.plot(line_lengths, line_widths, 'r')

	ax4.set_ylabel('Petal Width')
	ax4.set_xlabel('Petal Length')
	ax4.set_title('Classifier')
	ax4.set_ylim([0.5, 3])
	ax4.set_xlim([2.5, 7.5])
	ax4.legend(handles=[green_patch, blue_patch], loc='upper left')
	ax4.grid(True)

	# Q2b
	fig2 = plt.figure(2, figsize=(8, 6))

	# Actually graphing it
	plt.show()

	print("MSE: {:.4f}".format(mean_squared_error(iris_objects, weights, ['green', 'blue'])))
	print("Error: {:.4f}".format(error(iris_objects, weights, ['green', 'blue'], {'green' : float(0), 'blue' : float(1)})))




if __name__ == "__main__":
	main()