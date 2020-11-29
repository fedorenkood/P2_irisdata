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


# Names are color coded
def scatter_plot(plot, iris_objects, names, title, limits):
	red_patch = mpatches.Patch(color='red', label='Setosa')
	blue_patch = mpatches.Patch(color='blue', label='Virginica')
	green_patch = mpatches.Patch(color='green', label='Versicolor')

	for name in names:
		plot.scatter(iris_objects[name].petal_lengths, iris_objects[name].petal_widths, c=name, alpha=0.5)
	plot.legend(handles=[green_patch, blue_patch], loc='upper left')
	plot.set_ylim(limits['x'])
	plot.set_xlim(limits['y'])
	plot.set_title(title)
	plot.set_ylabel('Petal Width')
	plot.set_xlabel('Petal Length')
	plot.grid(True)


def line_plot(plot, weights, domain):
	line_lengths = np.linspace(domain, 100)
	line_widths = []
	for i in range(len(line_lengths)):
		line_widths.append(line_calculation([1, line_lengths[i]], weights))
	plot.plot(line_lengths, line_widths, 'r')


def converter(data, color_map):
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
	return -(inputs[0] * weights[0] + inputs[1] * weights[1]) / weights[2]


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
			difference = line_calculation([1, x], weights) - float(y)
			error_sum += math.pow(difference, 2)

	# Dividing by the number of data points
	error = error_sum / n_of_points
	return error


# Calculate Error given array of iris_objects, weights and names of objects to compare
# error(iris_objects, weights, ['green', 'blue'], {'green' : float(0), 'blue' : float(1)})
def logistic_error(iris_objects, weights, names, desired_output):
	# Computing the difference between the expected and observed data point
	error_sum = 0
	n_of_points = 0

	for name in names:
		for (x, y) in zip(iris_objects[name].petal_lengths, iris_objects[name].petal_widths):
			n_of_points += 1
			difference = single_layer_output([1, x, y], weights) - desired_output[name]
			error_sum = error_sum + math.pow(difference, 2)

	# TODO: Dividing by the number of data points?
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


def gradient(iris_objects, weights, names, desired_output):
	# TODO: it does not really work
	# TODO: should I calculate 2 or 3 gradients
	# TODO: used type of MSE
	n_of_points = 0
	gradient = [0, 0, 0]
	for name in names:
		for (x, y) in zip(iris_objects[name].petal_lengths, iris_objects[name].petal_widths):
			n_of_points += 1
			mse_error = line_calculation([1, x], weights) - float(y)
			logistic_e = single_layer_output([1, x, y], weights) - desired_output[name]
			gradient[0] += mse_error * 1
			gradient[1] += mse_error * float(x)
			# gradient[2] += mse_error * float(y)
	# TODO: adjustments for the number of data_points
	# calculating the actual step and multiplying it by the epsilon value to produce a new slope and intercept
	epsilon = 0.1/n_of_points
	print(f"Old Weights: {weights}")
	for i in range(0, len(gradient)):
		gradient[i] = (gradient[i] * 2 / n_of_points)
		change = gradient[i] * epsilon
		weights[i] += change
	print(f"New Weights: {weights}")
	return weights


def main():
	data = read_file('irisdata.csv')
	color_map = {"virginica": "blue", "versicolor": "green", "setosa": "red"}
	iris_objects = converter(data, color_map)

	# General Legend
	weights = [-3.3, 0.35, 1]  # These separate the two important things nicely

	# 1st Figure
	# TODO: set titles
	fig1 = plt.figure(1, figsize=(8, 6))
	fig1.subplots_adjust(hspace=0.35)

	# plots for Q1
	# Q1a
	scatter_plot(fig1.add_subplot(221), iris_objects, ['green', 'blue'], "Iris data", {'x': [0.5, 3], 'y': [2.5, 7.5]})

	# Q1c: Plot of line vs scatter
	f1ax2 = fig1.add_subplot(222)
	scatter_plot(f1ax2, iris_objects, ['green', 'blue'], "Decision Boundary", {'x': [0.5, 3], 'y': [2.5, 7.5]})
	line_plot(f1ax2, weights, [1, 7.5])

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
	f1ax4 = fig1.add_subplot(224)
	scatter_plot(f1ax4, decided_classes, ['green', 'blue'], "Classifier", {'x': [0.5, 3], 'y': [2.5, 7.5]})
	line_plot(f1ax4, weights, [1, 7.5])

	# Q2b
	fig2 = plt.figure(2, figsize=(8, 6))
	fig2.subplots_adjust(hspace=0.35)
	scatter_plot(fig2.add_subplot(221), iris_objects, ['green', 'blue'], "Iris data", {'x': [0.5, 3], 'y': [2.5, 7.5]})

	# classified with small error weights
	f2ax2 = fig2.add_subplot(222)
	scatter_plot(f2ax2, decided_classes, ['green', 'blue'], "Small Error Classification", {'x': [0.5, 3], 'y': [2.5, 7.5]})
	line_plot(f2ax2, weights, [1, 7.5])

	# classified with large error weights
	large_error_weights = [-2.3, 0.25, 1]
	le_decided_classes = decision(iris_objects, large_error_weights, ['green', 'blue'], {'0': 'green', '1': 'blue'})
	f2ax3 = fig2.add_subplot(223)
	scatter_plot(f2ax3, le_decided_classes, ['green', 'blue'], "Large Error Classification", {'x': [0.5, 3], 'y': [2.5, 7.5]})
	line_plot(f2ax3, large_error_weights, [1, 7.5])

	# TODO: not sure what error to use
	print("MSE Small Error: {:.4f}".format(mean_squared_error(iris_objects, weights, ['green', 'blue'])))
	print("Error Small Error: {:.4f}".format(logistic_error(iris_objects, weights, ['green', 'blue'], {'green': float(0), 'blue': float(1)})))
	print("MSE Large Error: {:.4f}".format(mean_squared_error(iris_objects, large_error_weights, ['green', 'blue'])))
	print("Error Large Error: {:.4f}".format(logistic_error(iris_objects, large_error_weights, ['green', 'blue'], {'green': float(0), 'blue': float(1)})))

	# Q2e gradient
	for i in range(0, 9):
		large_error_weights = gradient(iris_objects, large_error_weights, ['green', 'blue'], {'green': float(0), 'blue': float(1)})

	f2ax4 = fig2.add_subplot(224)
	scatter_plot(f2ax4, iris_objects, ['green', 'blue'], "10 Gradient Steps from LE", {'x': [0.5, 3], 'y': [2.5, 7.5]})
	line_plot(f2ax4, large_error_weights, [1, 7.5])
	# Actually graphing it
	plt.show()






if __name__ == "__main__":
	main()