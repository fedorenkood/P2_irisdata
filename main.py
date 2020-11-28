import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# each index will correspond to a row
petalLengths = []
petalWidths = []
flowerTypes = []
colorsToUse = []


def main():
    with open('irisdata.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                petalLengths.append(float(row[2]))
                petalWidths.append(float(row[3]))
                flowerTypes.append(row[4])

    colorMap = {"virginica": "#0000ff", "versicolor": "#00ff00", "setosa": "#ff0000"}

    for i in range(len(flowerTypes)):
        colorsToUse.append(colorMap.get(flowerTypes[i]))


    fig, ax = plt.subplots()
    ax.scatter(petalLengths, petalWidths, c=colorsToUse, alpha=0.5)

    red_patch = mpatches.Patch(color='blue', label='Virginica')
    green_patch = mpatches.Patch(color='green', label='Versicolor')
    blue_patch = mpatches.Patch(color='red', label='Setosa')

    plt.legend(handles=[red_patch, green_patch, blue_patch])

    ax.grid(True)

    plt.show()



if __name__ == "__main__":
    main()