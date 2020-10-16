import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
# df.tail()     <- I used Jupyter to get the output of this, which just confirms that we got the right data.

import matplotlib.pyplot as plt
import numpy as np

#grabs the first 100 class labels corresponding to setosa and versicolor.
y = df.iloc[0:100, 4].values

#coverts class labels to -1 (for setosa) and 1 (for the rest, i.e., versicolor).
y = np.where(y == 'Iris-setosa', -1, 1)

#grabs sepal length and petal length, two features in columns 0 and 2.
X = df.iloc[0:100, [0,2]].values

#plots the the two classes on a graph with sepal length as one axis and petal length as the other.
plt.scatter(X[:50, 0], X[:50, 1], color='green', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='purple', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

from Perceptron import *

#creates perceptron using the class we made
ppn = Perceptron(eta=0.1, n_iter=10)

#trains the perceptron on our Iris data
ppn.fit(X, y)

#plots the error for each epoch
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

"""
The perceptron converges after 6 epochs.
There is a final step to plot the decision boundary, which I have omitted.
"""