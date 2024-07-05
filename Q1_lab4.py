import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

np.random.seed(42)

class Logistic_regression:
    def __init__(self, learnin_rate = 0.01, epochs = 1000, lamb_val = 0):
        self.epochs =  epochs
        self.learnin_rate = learnin_rate
        self.lamb_val = lamb_val
    
    def sigmoid(self, z):
        z=np.array(z)                                                        # this ensures that numpy is an array
        return 1/(1 + np.exp(-z))
    
    def grad_func(self, X, y, n, theta):
        h = self.sigmoid(np.dot(X, theta))                                   # gives the hypothesis
        L = (-1/n) * (np.dot(y.T, np.log(h)) + np.dot(1 - y, np.log(1 - h))) # calculatin the loss

        if self.lamb_val > 0:
            L += (self.lamb_val / (2 * n)) * np.sum(np.square(theta[1:]))    # adding the regularisation term 
        
        grad = (1/n) * np.dot(X.T, (h - y))                                  # calculates the grad
        if self.lamb_val > 0:
            grad[1 : ] += (self.lamb_val / n) * theta[1 : ]
        
        return L, grad                                                       # gives the loss and gradient back
    
    def grad_descent(self, X, y, n, theta):
        for epoch in range(self.epochs):
            L, grad = self.grad_func(X, y, n, theta)                         # gets the data for every epoch
            theta -= grad * self.learnin_rate                                # updates the theta value every time
            if epoch % 100 == 0:                                             # prints the epoch for 100th one  
                print(f"For the {epoch}, the loss is {L[0],[0]}")
            return theta

    def vectorized_map(X1, X2):
        degree = 6
        out = np.ones(X1.shape[0])[:, np.newaxis]
        for i in range(1, degree + 1):
            for j in range(i + 1):
                out = np.hstack((out, (X1 ** (i - j) * X2 ** j)[:, np.newaxis]))
        return out                                                            # Return the new feature matrix
