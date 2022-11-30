import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle


class LogisticRegression:
    """
    Impliments a logistic regression model
    """

    def __init__(
        self, learning_rate=0.01, n_iters=100, plot_loss=False, epsilon=(1 * 10 ^ -7)
    ):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.plot_loss = plot_loss
        self.cost = []
        self.epsilon = epsilon
        # self.intercept = intercept

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
        for iter in tqdm(range(self.n_iters)):
            linear = self.linear_function(X)
            y_pred = self.sigmoid_function(linear)
            gradient = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            dbias = (1 / num_samples) * np.sum(y_pred - y)
            self.update_weights(gradient, dbias)
            cost = self.logistic_regression_cost_function(y, y_pred, num_samples)
            self.cost.append(cost)
            if cost < self.epsilon:
                print("Converged at iteration {}".format(iter))
                break

        if self.plot_loss:
            self.plot_loss()

    def logistic_regression_cost_function(self, y, y_pred, num_samples):
        """
        Compute the loss of the model
        """
        cost = -1 * (
            (1 / num_samples)
            * np.sum((y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)))
        )
        return cost

    def predict(self, X):
        """
        Predict the class of a sample
        """
        linear = self.linear_function(X)
        sigmoid = self.sigmoid_function(linear)
        y_pred = np.sign(sigmoid)
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
        plt.plot(self.cost)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()


def bin_gps(df, number_vertical_bins, number_horizontal_bins, plot=False):
    """
    creates a binning structure for GPS coordinates
    """
    # assigns a bin to each GPS coordinate
    if plot:
        plot_bins(df, number_vertical_bins, number_horizontal_bins)

    df["LAT_BIN"] = pd.cut(df["LATITUDE"], number_vertical_bins)
    df["LON_BIN"] = pd.cut(df["LONGITUDE"], number_horizontal_bins)
    return df


def min_max_lat_lon(df):
    """
    Compute the minimum and maximum latitude and longitude
    from a pandas dataframe and removes data errors
    """
    min_lon = df.LONGITUDE.min()
    max_lon = df.LONGITUDE.max()
    min_lat = df.LATITUDE.min()
    max_lat = df.LATITUDE.max()
    return min_lat, max_lat, min_lon, max_lon


def dummy_data(df, number_vertical_bins, number_horizontal_bins):
    """
    creates dummy variables for the binning structure
    """
    df = bin_gps(df, number_vertical_bins, number_horizontal_bins)
    df = pd.get_dummies(df, columns=["LAT_BIN", "LON_BIN"])
    # remove original LATITUDE and LONGITUDE columns
    df.drop(["LATITUDE", "LONGITUDE"], axis=1, inplace=True)

    return df


def plot_bins(df, number_vertical_bins, number_horizontal_bins):
    """
    Plots the gps bins on a map
    """

    min_lat, max_lat, min_lon, max_lon = min_max_lat_lon(df)
    latlines = np.linspace(min_lat, max_lat, number_vertical_bins + 1)
    lonlines = np.linspace(min_lon, max_lon, number_horizontal_bins + 1)
    # plot the bins
    fig, ax = plt.subplots()
    # set size of plot
    fig.set_size_inches(5, 7.5)
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("GPS Bins")
    ax.grid()
    ax.scatter(df["LONGITUDE"], df["LATITUDE"], c="black", s=0.1)
    # plot the bin edges
    for i, val in enumerate(latlines[1:]):
        if i % 10 == 0:
            ax.axhline(val, color="red", alpha=0.5, lw=1)
    for i, val in enumerate(lonlines[1:]):
        if i % 10 == 0:
            ax.axvline(val, color="red", alpha=0.5, lw=1)
    plt.show()


def gps_crawl(crash_df, filename, n_iters=100, gps_only=True, plot_loss=False):
    """
    crawls across a bin structure and creates a new dataframe
    """
    y = crash_df.DAMAGE
    X = crash_df.drop("DAMAGE", axis=1)

    # remove all columns not LATITUDE or LONGITUDE
    if gps_only:
        X = X[["LATITUDE", "LONGITUDE"]]
    else:
        X = X[
            ["LATITUDE", "LONGITUDE", "CRASH_HOUR", "CRASH_DAY_OF_WEEK", "CRASH_MONTH"]
        ]

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.77, random_state=42, stratify=y
    )
    models = []

    for bins in range(1, 501, 50):
        print("=========================================")
        print(f"Number of vertical bins: {bins}")
        print(f"Number of horizontal bins: {501 - bins}")
        # plot the bin structure
        print(
            f"Plotting bins... (Only shows every 10th bin line for each vert and horz"
        )
        print(f"Gives bin shape but each bin 10x smaller")
        plot_bins(X_train, bins, 501 - bins)

        # bin the GPS data
        binned_X_train = dummy_data(X_train, bins, 501 - bins)
        binned_X_test = dummy_data(X_test, bins, 501 - bins)

        model = LogisticRegression(n_iters, plot_loss=False)
        model.train(binned_X_train, y_train)
        y_pred = model.predict(binned_X_test)
        save_object(model, f"{filename}_model_{bins}")
        models.append(model)
        # counts the unique values in the prediction
        print(f"First 10 values of prediction: {y_pred[:10]}")
        print(f"Unique values in prediction: {np.unique(y_pred, return_counts=True)}")
        print(f"First 10 values of actual: {y_test[:10]}")
        print(f"y_test total 1s divided by total: {np.sum(y_test) / len(y_test)}")
        new_accuracy = model.accuracy(y_test, y_pred)
        print("Accuracy: ", new_accuracy)
        if bins != 1:
            print(f"Accuracy Change: {accuracy - new_accuracy}")
        accuracy = new_accuracy

    return models


def save_object(obj, filename):
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

