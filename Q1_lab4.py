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
    return out                                                                # return the new feature matrix

def load_data(file):
    df = pd.read_csv(file, sep= ",", header = None)
    df.columns = ["X1", "X2", "label"]
    return df                                                                 # returns the data frame
    
def visualize(df, task, theta):
    plt.figure(fig_size=(7,5))
    ax = sns.scatterplot(x = "X1", y = "X2", hue = "label", data = df, style = "label", s =80)
    plt.title("Scatter plot of the training data")

    if theta is not None:                                                     # Plot decision boundary if theta is provided
        x_min, x_max = df['X1'].min() - 1, df['X1'].max() + 1
        y_min, y_max = df['X2'].min() - 1, df['X2'].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        if task == 1 or task == 2:
            Z = model.sigmoid(np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()].dot(theta))
        else:
            out = vectorized_map(xx.ravel(), yy.ravel())
            Z = model.sigmoid(out.dot(theta))
        
        Z = Z.reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], colors='green')
    
    path = "plot_" + str(task) + ".png"
    plt.savefig(path)
    print("Data plot with decision boundary saved at: ", path)

def task_1():
    learnin_rate = 0.001
    epochs = 1000

    file = "task1_data.csv"
    df = load_data(file)

    n = len(df)                                                  # no of training examples
    X = np.hstack((np.ones((n, 1)), df[['X1', 'X2']].values))    # Add bias term to X
    y = np.array(df.label.values).reshape(-1, 1)                 # Convert y to column vector
    initial_theta = np.zeros(shape=(X.shape[1], 1))              # Initialize theta to zeros

    model = Logistic_regression(epochs, learnin_rate)            # Initialize model
    theta = model.grad_descent(X, y, initial_theta, m)           # Perform gradient descent

    print('Theta found by gradient descent:\n', theta)
    visualize(df, 1, theta)                                      # Plot the decision boundary on the data points

def task_2():
    learnin_rate = 0.001
    epochs = 1000

    file = "task2_data.csv"
    df = load_data(file)

    df["X1X2"] = df["X1"] * df["X2"]  
    n = len(df)
    X = np.hstack((np.ones((n, 1)), df[['X1', 'X2', 'X1X2']].values))        # Create X with new feature
    y = np.array(df.label.values).reshape(-1, 1)                             # Convert y to column vector
    initial_theta = np.zeros(shape=(X.shape[1], 1))                          # Initialize theta to zeros

    model = Logistic_regression(epochs, learnin_rate)                        # Initialize model
    theta = model.gradient_descent(X, y, initial_theta, m)                   # Perform gradient descent

    print('Theta found by gradient descent:\n', theta)
    visualize(df, 2, theta)                                                  # Plot the decision boundary on the data points

def task_3():
    learnin_rate = 0.01
    epochs = 1000
    lambda_value = 0

    file_name = "task3_data.csv"
    df = load_data(file_name)

    m = len(df)                                                              # Get the number of training examples
    X = vectorized_map(df['X1'].values, df['X2'].values)                     # Create polynomial features
    y = np.array(df.label.values).reshape(-1, 1)                             # Convert y to column vector
    initial_theta = np.zeros(shape=(X.shape[1], 1))                          # Initialize theta to zeros

    model = Logistic_regression(epochs, learnin_rate, lambda_value)          # Initialize model
    theta = model.grad_descent(X, y, initial_theta, m)                       # Perform gradient descent

    print('Theta found by gradient descent:\n', theta)
    visualize(df, 3, theta)                                                  # Plot the decision boundary on the data points

if __name__=='__main__':
    
    # Uncomment the line below to run task 1
    task_1()
    
    # Uncomment the line below to run task 2 
    task_2()
    
    # Uncomment the line below to run task 3 
    task_3()
