import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from sklearn.neural_network import MLPClassifier


def main():
    input_data = []
    output_data = []
    test_input = []
    test_output = []

    #first, get the data from the csv file
    with open("irisdata.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line = 0
        for row in csv_reader:
            if line == 0:
                line += 1
            elif line == 3 or line == 4 or line == 5 or line == 82 or line == 83 or line == 84 or line == 146 or line == 147 or line == 148:
                test_output.append(row[4])
                row.pop(4)
                test_input.append(row)
                line += 1
            else:
                output_data.append(row[4])
                row.pop(4)
                input_data.append(row)
                line += 1


    # turn all those pesky strings into floating points
    for row in input_data:
        for i in range(len(row)):
            row[i] = float(row[i])

    for row in test_input:
        for i in range(len(row)):
            row[i] = float(row[i])


    relu_classifier = MLPClassifier(hidden_layer_sizes=(10), max_iter=100, activation='relu', solver='adam', random_state=1)
    linear_classifier = MLPClassifier(hidden_layer_sizes=(10), max_iter=1000, activation='identity', solver='adam', random_state=1)
    tanh_classifier = MLPClassifier(hidden_layer_sizes=(10), max_iter=200, activation='tanh', solver='adam', random_state=1)
    logistic_classifier = MLPClassifier(hidden_layer_sizes=(10), max_iter=450, activation='logistic', solver='adam', random_state=1)

    relu_classifier.fit(input_data, output_data)
    linear_classifier.fit(input_data, output_data)
    tanh_classifier.fit(input_data, output_data)
    logistic_classifier.fit(input_data, output_data)


    relu_predicted_output = relu_classifier.predict(test_input)
    linear_predicted_output = linear_classifier.predict(test_input)
    tanh_predicted_output = tanh_classifier.predict(test_input)
    logstic_predicted_output = logistic_classifier.predict(test_input)

    relu_number_correct = 0
    relu_number_total = 0
    for i in range(len(relu_predicted_output)):
        if test_output[i] == relu_predicted_output[i]:
            relu_number_correct += 1
        relu_number_total += 1

    linear_number_correct = 0
    linear_number_total = 0
    for i in range(len(linear_predicted_output)):
        if test_output[i] == linear_predicted_output[i]:
            linear_number_correct += 1
        linear_number_total += 1

    tanh_number_correct = 0
    tanh_number_total = 0
    for i in range(len(tanh_predicted_output)):
        if test_output[i] == tanh_predicted_output[i]:
            tanh_number_correct += 1
        tanh_number_total += 1

    logistic_number_correct = 0
    logistic_number_total = 0
    for i in range(len(logstic_predicted_output)):
        if test_output[i] == logstic_predicted_output[i]:
            logistic_number_correct += 1
        logistic_number_total += 1

    print("accuracy for relu =", relu_number_correct / relu_number_total * 100)
    print("accuracy for linear =", linear_number_correct / linear_number_total * 100)
    print("accuracy for tanh =", tanh_number_correct / tanh_number_total * 100)
    print("accuracy for logistic =", logistic_number_correct / logistic_number_total * 100)


if __name__ == "__main__":
    main()