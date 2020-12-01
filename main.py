import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import random


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
	plot.set_ylim(limits['y'])
	plot.set_xlim(limits['x'])
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


def sigmoid_plot(plot, weights, title, limits, n_intervals):
	graph_widths = np.linspace(limits['y'][0], limits['y'][1], n_intervals)
	graph_lengths = np.linspace(limits['x'][0], limits['x'][1], n_intervals)
	x_axis, y_axis = np.meshgrid(graph_lengths, graph_widths)
	graph_outputs = []
	for x in range(0, n_intervals):
		graph_outputs.append([])
		for y in range(0, n_intervals):
			graph_outputs[x].append(single_layer_output([1, x_axis[x][y], y_axis[x][y]], weights))
	graph_outputs = np.array(graph_outputs)

	plot.set_zlim(limits['z'])
	plot.plot_surface(x_axis, y_axis, graph_outputs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	plot.set_ylabel('Petal Width')
	plot.set_xlabel('Petal Length')
	plot.set_title(title)


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


def random_weights():
	return [-1 - random.random() * 10, random.random() * 3, random.random() * 6]


# this is used for 1c
def line_calculation(inputs, weights):
	# (-weights[2]) because when we say z=0, we divide by the negative of the y weight to get the equation
	return (inputs[0] * weights[0] + inputs[1] * weights[1]) / (-weights[2])


# inputs are of the form [1, width, length], weights are of the form [w0, w1, w2],
# and w0 is the weight for the bias node.
def weighting_input(inputs, weights):
	sum = 0
	for (i, w) in zip(inputs, weights):
		sum += i * w
	return sum


def single_layer_output(inputs, weights):
	return 1/(math.pow(math.e, -weighting_input(inputs, weights)) + 1)


# Calculate Error given array of iris_objects, weights and names of objects to compare
# error(iris_objects, weights, ['green', 'blue'], {'green' : float(0), 'blue' : float(1)})
def error(iris_objects, weights, names, desired_output):
	# Computing the difference between the expected desired_output and observed output of single_layer network
	error_sum = 0
	n_of_points = 0
	for name in names:
		for (x, y) in zip(iris_objects[name].petal_lengths, iris_objects[name].petal_widths):
			n_of_points += 1
			difference = single_layer_output([1, x, y], weights) - desired_output[name]
			error_sum = error_sum + math.pow(difference, 2)
	# Dividing by the number of data points
	return error_sum / n_of_points


# Uses original data to classify by the given names and single_layer_output
# decision(iris_objects, weights, ['green', 'blue'], {'0' : 'green', '1' : 'blue'})
def decision(iris_objects, weights, names, desired_class):
	decided_class = {names[0]: Iris(names[0], [], []), names[1]: Iris(names[1], [], [])}
	for name in names:
		for (x, y) in zip(iris_objects[name].petal_lengths, iris_objects[name].petal_widths):
			# if data point has single_layer_output > 0.5 it is classified as one class. Otherwise, it is another class
			if single_layer_output([1, x, y], weights) > 0.5:
				decided_class[desired_class['1']].petal_lengths.append(x)
				decided_class[desired_class['1']].petal_widths.append(y)
			else:
				decided_class[desired_class['0']].petal_lengths.append(x)
				decided_class[desired_class['0']].petal_widths.append(y)

	return decided_class


# computes the gradient of weights based on the dataset and updates weights
# the gradient represents the derivative of the objective function in terms of weights
def gradient(iris_objects, weights, names, desired_output):
	n_of_points = 0
	gradient = [0, 0, 0]
	for name in names:
		for (x, y) in zip(iris_objects[name].petal_lengths, iris_objects[name].petal_widths):
			n_of_points += 1
			sigmoid = single_layer_output([1, x, y], weights)
			objective_derivative = (sigmoid - desired_output[name]) * sigmoid * (1 - sigmoid)
			gradient[0] += objective_derivative * 1
			gradient[1] += objective_derivative * float(x)
			gradient[2] += objective_derivative * float(y)
	# calculating the actual step and multiplying it by the epsilon value to produce a new slope and intercept
	# 0.5 is a nice step size
	epsilon = 0.5
	for i in range(0, len(gradient)):
		gradient[i] = (gradient[i] * 2)/n_of_points
		change = gradient[i] * epsilon
		# we subtract the gradient because it is a descent function
		weights[i] -= change
	return weights


# gradient descent function updates weights until the change in error gets too small
def descent(iris_objects, weights, names, desired_output):
	errors = [0.0001 + error(iris_objects, weights, names, desired_output), error(iris_objects, weights, names, desired_output)]
	weights_array = [weights, weights]
	iteration = 1

	# for gradient descent to converge prev_error - current_error has to be > 0
	# gradient descent function updates weights until MSE doesn't decrease by more than 0.0000002
	while errors[iteration - 1] - errors[iteration] > 0.0000002:
		old_weights = weights_array[iteration].copy()
		new_weights = gradient(iris_objects, old_weights, names, desired_output)
		weights_array.append(new_weights)
		errors.append(error(iris_objects, new_weights, names, desired_output))
		iteration += 1

	# Creating Graphs to represent change
	fig3 = plt.figure(3, figsize=(8, 6))
	fig3.subplots_adjust(hspace=0.35)
	scatter_limits = {'x': [2.5, 7.5], 'y': [0.5, 3]}

	# Initial plot
	f3ax1 = fig3.add_subplot(221)
	scatter_plot(f3ax1, iris_objects, names, f"After {0} Iterations", scatter_limits)
	line_plot(f3ax1, weights_array[1], [1, 7.5])

	# Mid iteration plot
	f3ax2 = fig3.add_subplot(222)
	scatter_plot(f3ax2, iris_objects, names, f"After {int(iteration/2)} Iterations", scatter_limits)
	line_plot(f3ax2, weights_array[int(iteration/2)], [1, 7.5])

	# Final plot
	f3ax3 = fig3.add_subplot(223)
	scatter_plot(f3ax3, iris_objects, names, f"After {int(iteration)} Iterations", scatter_limits)
	line_plot(f3ax3, weights_array[int(iteration)], [1, 7.5])

	# Error plot
	f3ax4 = fig3.add_subplot(224)
	f3ax4.plot(range(0, iteration + 1), errors)
	f3ax4.set_ylabel('MSE')
	f3ax4.set_xlabel('N of Iterations')
	f3ax4.set_title('Decrease in Error')

	print(f"Number of iterations: {iteration}")
	print(f"Final Error: {errors[iteration]}")
	return weights_array[iteration]



def main():
	data = read_file('irisdata.csv')
	color_map = {"virginica": "blue", "versicolor": "green", "setosa": "red"}
	iris_objects = converter(data, color_map)

	# General Legend
	weights = [-10, 1.2, 3]  # These separate the two important things nicely
	names = ['green', 'blue']
	desired_output = {'green': float(0), 'blue': float(1)}
	classifying_input = {'0': 'green', '1': 'blue'}
	scatter_limits = {'x': [2.5, 7.5], 'y': [0.5, 3]}

	# 1st Figure
	fig1 = plt.figure(1, figsize=(8, 6))
	fig1.subplots_adjust(hspace=0.35)

	# plots for Q1
	# Q1a
	scatter_plot(fig1.add_subplot(221), iris_objects, names, "Iris data", scatter_limits)

	# Q1c: Plot of line vs scatter
	f1ax2 = fig1.add_subplot(222)
	scatter_plot(f1ax2, iris_objects, names, "Decision Boundary", scatter_limits)
	line_plot(f1ax2, weights, [1, 7.5])

	# Q1d graph
	sigmoid_plot(fig1.add_subplot(223, projection='3d'), weights, 'Logistic Curve', {'x': [0, 7.5], 'y': [0, 4], 'z': [0, 1]}, 100)

	# Q1e graph
	decided_classes = decision(iris_objects, weights, names, classifying_input)
	f1ax4 = fig1.add_subplot(224)
	scatter_plot(f1ax4, decided_classes, names, "Classifier", scatter_limits)
	line_plot(f1ax4, weights, [1, 7.5])

	# Q2b
	fig2 = plt.figure(2, figsize=(8, 6))
	fig2.subplots_adjust(hspace=0.35)
	scatter_plot(fig2.add_subplot(221), iris_objects, names, "Iris data", scatter_limits)

	# classified with small error weights
	f2ax2 = fig2.add_subplot(222)
	scatter_plot(f2ax2, decided_classes, names, "Small Error Classification", scatter_limits)
	line_plot(f2ax2, weights, [1, 7.5])

	# classified with large error weights
	large_error_weights = [-2.3, 0.25, 1]
	le_decided_classes = decision(iris_objects, large_error_weights, names, classifying_input)
	f2ax3 = fig2.add_subplot(223)
	scatter_plot(f2ax3, le_decided_classes, names, "Large Error Classification", scatter_limits)
	line_plot(f2ax3, large_error_weights, [1, 7.5])

	print("Small Error: {:.4f}".format(error(iris_objects, weights, names, desired_output)))
	print("Large Error: {:.4f}".format(error(iris_objects, large_error_weights, names, desired_output)))

	# Q2e gradient
	print(f"Old Weights: {large_error_weights}")
	for i in range(0, 10):
		large_error_weights = gradient(iris_objects, large_error_weights, names, desired_output)
		# print(error(iris_objects, large_error_weights, names, desired_output))

	print(f"New Weights: {large_error_weights}")
	print("Large Error after 10 iterations: {:.4f}".format(error(iris_objects, large_error_weights, names, desired_output)))

	f2ax4 = fig2.add_subplot(224)
	scatter_plot(f2ax4, iris_objects, names, "10 Gradient Steps from LE", scatter_limits)
	line_plot(f2ax4, large_error_weights, [1, 7.5])

	# Q3a
	descent(iris_objects, random_weights(), names, desired_output)
	# Actually graphing it
	plt.show()






if __name__ == "__main__":
	main()