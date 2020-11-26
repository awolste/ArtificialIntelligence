# learning.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this assignment, you will implement linear and logistic regression
using the gradient descent method, as well as the binary perceptron algorithm. 
To complete the assignment, please modify the linear_regression(), binary_perceptron(),
and logistic_regression() functions.

The package `matplotlib` is needed for the program to run.
You should also use the 'numpy' library to vectorize 
your code, enabling a much more efficient implementation of 
linear and logistic regression. You are also free to use the 
native 'math' library of Python. 

All provided datasets are extracted from the scikit-learn machine learning library. 
These are called `toy datasets`, because they are quite simple and small. 
For more details about the datasets, please see https://scikit-learn.org/stable/datasets/index.html

Each dataset is randomly split into a training set and a testing set using a ratio of 8 : 2. 
You will use the training set to learn a regression model. Once the training is done, the code
will automatically validate the fitted model on the testing set.  
"""

# use math and/or numpy if needed
import math
import numpy as np


def linear_regression(x, y, logger=None):
    """
    Linear regression using full batch gradient descent.
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by x^T w, where x is a column vector. 
    The intercept term can be ignored due to that x has been augmented by adding '1' as an extra feature. 
    You should use as learning rate alpha=0.0001. If you scale the cost function by 1/#samples, use alpha=0.001  

    Parameters
    ----------
    x: a 2D array of size [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array of size [N]
       It contains the target value for each sample in x
    logger: a logger instance for plotting the loss
       Usage: logger.log(i, loss) where i is the number of iterations
       Log updates can be performed every several iterations to improve performance.
    
    Returns
    -------
    w: a 1D array
       linear regression parameters
    """

    alpha = .0001
    iter = 1
    y_col_vector = np.transpose(np.array([y]))
    x_col_vector = np.array(x)
    w = np.zeros((len(np.transpose(x_col_vector)), 1))
    prev_w = 0

    while iter < 1000:
        # predicted value
        h = np.matmul(x_col_vector, w)
        # difference between predicted and actual
        d = h - y_col_vector
        # transpose x
        x_t = np.transpose(x_col_vector)
        # gradient
        g = np.matmul(x_t, d)
        # updates the weights
        w = w - (alpha * g)

        # calculate the least squares loss
        loss = .5 * np.sum(np.square(d))
        # log every 5 times to save time
        if iter % 5 == 0:
            logger.log(iter, loss)
        # when the absolute max of prev weights is close to current
        if (np.absolute(np.amax(np.absolute(w)) - np.amax(np.absolute(prev_w)))) < .000005:  # how small should this be
            return np.ravel(w)
        prev_w = w
        """if (abs(prev_loss - loss)) < .01:  # how small should this be
            converged = True
        prev_loss = loss"""
        iter += 1
    # be sure to return a 1d array
    return np.ravel(w)


def binary_perceptron(x, y, logger=None):
    """
    Binary classifaction using a perceptron. 
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by
        h = (x^T w) 
    with the decision boundary:
        h >= 0 => x in class 1
        h < 0  => x in class 0
    where x is a column vector. 
    The intercept/bias term can be ignored due to that x has been augmented by adding '1' as an extra feature. 
    
    
    Parameters
    ----------
    x: a 2D array with the shape [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array with the shape [N]
       It is the ground truth value for each sample in x
    logger: a logger instance through which plotting loss
       Usage: Please do not use the logger in this function.
    
    Returns
    -------
    w: a 1D array
       binary perceptron parameters
    """
    y_col_vector = np.transpose(np.array([y]))
    x_col_vector = np.array(x)
    w = np.zeros((len(np.transpose(x_col_vector)), 1))
    while True:
        # predicted value
        h = np.dot(x_col_vector, w)
        # cast to 0 or 1 depending on dot prod
        h = np.where(h >= 0, 1, 0)
        # difference between actual and predicted
        d = y_col_vector - h
        # transpose x
        x_t = np.transpose(x_col_vector)
        # gradient
        g = np.matmul(x_t, d)
        # update w
        w = w + g
        if np.array_equal(h, y_col_vector):
            return np.ravel(w)




def logistic_regression(x, y, logger=None):
    """
    Logistic regression using batch gradient descent.
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by p = sigmoid(x^T w)
    with the decision boundary:
        p >= 0.5 => x in class 1
        p < 0.5  => x in class 0
    where x is a column vector. 
    The intercept/bias term can be ignored due to that x has been augmented by adding '1' as an extra feature. 
    In gradient descent, you should use as learning rate alpha=0.001    

    Parameters
    ----------
    x: a 2D array of size [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array of size [N]
       It contains the ground truth label for each sample in x
    logger: a logger instance for plotting the loss
       Usage: logger.log(i, loss) where i is the number of iterations
       Log updates can be performed every several iterations to improve performance.
        
    Returns
    -------
    w: a 1D array
       logistic regression parameters
    """
    alpha = .001
    iter = 1
    y_col_vector = np.transpose(np.array([y]))
    x_col_vector = np.array(x)
    w = np.zeros((len(np.transpose(x_col_vector)), 1))
    prev_w = 0

    while iter < 1000:
        # do a sigmoid function for predicted value
        h = 1 / (1 + np.exp(-np.dot(x_col_vector, w)))
        d = h - y_col_vector
        x_t = np.transpose(x_col_vector)
        g = np.matmul(x_t, d)
        w = w - (alpha * g)

        # sum of logs
        loss = -np.sum(np.matmul(np.transpose(y_col_vector), np.log(h)) +
                       np.matmul(1 - np.transpose(y_col_vector), np.log(1 - h)))
        if iter % 5 == 0:
            logger.log(iter, loss)
        if (np.absolute(np.amax(np.absolute(w)) - np.amax(np.absolute(prev_w)))) < .0005:  # how small should this be
            return np.ravel(w)
        prev_w = w
        """if (abs(prev_loss - loss)) < .01:  # how small should this be
            converged = True
        prev_loss = loss"""
        iter += 1
    return np.ravel(w)



if __name__ == "__main__":
    import os
    import tkinter as tk
    from app.regression import App

    import data.load
    dbs = {
        "Boston Housing": (
            lambda : data.load("boston_house_prices.csv"),
            App.TaskType.REGRESSION
        ),
        "Diabetes": (
            lambda : data.load("diabetes.csv", header=0),
            App.TaskType.REGRESSION
        ),
        "Handwritten Digits": (
            lambda : (data.load("digits.csv", header=0)[0][np.where(np.equal(data.load("digits.csv", header=0)[1], 0) | np.equal(data.load("digits.csv", header=0)[1], 1))],
                      data.load("digits.csv", header=0)[1][np.where(np.equal(data.load("digits.csv", header=0)[1], 0) | np.equal(data.load("digits.csv", header=0)[1], 1))]),
            App.TaskType.BINARY_CLASSIFICATION
        ),
        "Breast Cancer": (
            lambda : data.load("breast_cancer.csv"),
            App.TaskType.BINARY_CLASSIFICATION
        )
     }

    algs = {
       "Linear Regression (Batch Gradient Descent)": (
            linear_regression,
            App.TaskType.REGRESSION
        ),
        "Logistic Regression (Batch Gradient Descent)": (
            logistic_regression,
            App.TaskType.BINARY_CLASSIFICATION
        ),
        "Binary Perceptron": (
            binary_perceptron,
            App.TaskType.BINARY_CLASSIFICATION
        )
    }

    root = tk.Tk()
    App(dbs, algs, root)
    tk.mainloop()
