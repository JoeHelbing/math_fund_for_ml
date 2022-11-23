import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class LogisticRegression:
    """
    Impliments a logistic regression model
    """

    def __init__(self, learning_rate=0.01, n_iters=100, plot_loss=False):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss = []


    def train(self, X, y):
        """
        Train the model
        """
        num_samples, num_features = X.shape
        self.weights, self.bias = self.initialize_weights(num_features)
        self.gradient_descent(X, y, num_samples)


    def sigmoid_function(self, x):
        """
        Compute sigmoid function
        """
        return 1 / (1 + np.exp(-x))


    def initialize_weights(self, dimensions):
        """
        Initialize weights for logistic regression
        """
        weights = np.zeros(dimensions)
        bias = 0
        return weights, bias


    def linear_function(self, X):
        """
        Compute the linear function
        """
        return np.dot(X, self.weights) + self.bias


    def update_weights(self, gradient, dbias):
        """
        Update the weights and bias
        """
        self.weights -= self.lr * gradient
        self.bias -= self.lr * dbias


    def gradient_descent(self, X, y, num_samples):
        """
        Run gradient descent
        """
        for _ in tqdm(range(self.n_iters)):
            linear = self.linear_function(X)
            y_pred = self.sigmoid_function(linear)
            gradient = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            dbias = (1 / num_samples) * np.sum(y_pred - y)
            self.update_weights(gradient, dbias)
            loss = self.logistic_regression_loss_function(y, y_pred)
            self.loss.append(loss)
            if loss < 0.00001:
                print("Converged at iteration {}".format(_))
                break
        
        if self.plot_loss:
            self.plot_loss()
    

    def logistic_regression_loss_function(self, y, y_pred):
        """
        Compute the loss of the model
        """
        loss = np.sum(-(y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)))
        return loss
        

    def predict(self, X):
        """
        Predict the class of a sample
        """
        linear = self.linear_function(X)
        sigmoid = self.sigmoid_function(linear)
        y_pred = np.round(sigmoid)
        return y_pred
    

    def accuracy(self, y_true, y_pred):
        """
        Compute the accuracy of the model
        """
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    def plot_loss(self):
        """
        Plot the loss of the model
        """
        plt.plot(self.loss)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()


def bin_gps(df, number_vertical_bins, number_horizontal_bins):
    """
    creates a binning structure for GPS coordinates
    """
    # assigns a bin to each GPS coordinate
    df["LAT_BIN"] = pd.cut(df["LATITUDE"], number_vertical_bins)
    df["LON_BIN"] = pd.cut(df["LONGITUDE"], number_horizontal_bins)
    return df


def min_max_lat_lon(df):
    """
    Compute the minimum and maximum latitude and longitude
    from a pandas dataframe and removes data errors
    """
    min_lat = df.LONGITUDE.min()
    max_lat = df.LONGITUDE.max()
    min_lon = df.LATITUDE.min()
    max_lon = df.LATITUDE.max()
    return min_lat, max_lat, min_lon, max_lon


def dummy_data(df, number_vertical_bins, number_horizontal_bins):
    """
    creates dummy variables for the binning structure
    """
    df = bin_gps(df, number_vertical_bins, number_horizontal_bins)
    df = pd.get_dummies(df, columns=["LAT_BIN", "LON_BIN"])
    return df
