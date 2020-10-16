import numpy as np
class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ----------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes over training dataset
    random_state: int
        RNG seed for random weight initialization

    Attributes
    ----------
    w_: 1d-array
        Array of weights after fitting
    errors_: list
        Number of misclassifications in each epoch
        (where an epoch is one pass over the training data)
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fits training data.

        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Array with n_samples row vectors each with n_features feature values)
        y: array-like, shape = [n_samples]
            Target values
        
        Returns
        -------
        self: object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale = 0.01, size = 1+X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update!=0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input
        (dot product of inputs with weight vector, plus bias value)"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)