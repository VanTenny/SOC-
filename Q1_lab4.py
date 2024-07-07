import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import os

np.random.seed(42)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, lambda_value=0):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda_value = lambda_value
    
    def sigmoid(self, z):
        z = np.array(z)                                                        # This ensures that z is a numpy array
        return 1 / (1 + np.exp(-z))
    
    def grad_func(self, X, y, theta):
        n = len(y)
        h = self.sigmoid(np.dot(X, theta))                                   # Calculate the hypothesis
        L = (-1 / n) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h))) # Calculate the loss

        if self.lambda_value > 0:
            L += (self.lambda_value / (2 * n)) * np.sum(np.square(theta[1:]))    # Add the regularization term 
        
        grad = (1 / n) * np.dot(X.T, (h - y))                                  # Calculate the gradient
        if self.lambda_value > 0:
            grad[1:] += (self.lambda_value / n) * theta[1:]
        
        return L, grad                                                       # Return the loss and gradient
    
    def grad_descent(self, X, y, theta):
        for epoch in range(self.epochs):
            L, grad = self.grad_func(X, y, theta)                         # Calculate loss and gradient for each epoch
            theta -= grad * self.learning_rate                                # Update theta values
            if epoch % 100 == 0:                                              # Print the loss every 100 epochs
                print(f"Epoch {epoch}: Loss = {L[0][0]}")
        return theta

def vectorized_map(X1, X2):
    degree = 6
    out = np.ones((X1.shape[0], 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.hstack((out, (X1 ** (i - j) * X2 ** j)[:, np.newaxis]))
    return out                                                                # Return the new feature matrix

def load_data(file):
    df = pd.read_csv(file, sep=",", header=None)
    df.columns = ["X1", "X2", "label"]
    return df                                                                 # Return the data frame
    
def visualize(df, task, theta):
    plt.figure(figsize=(7, 5))
    ax = sns.scatterplot(x="X1", y="X2", hue="label", data=df, style="label", s=80)
    plt.title("Scatter plot of the training data")

    if theta is not None:                                                     # Plot decision boundary if theta is provided
        x_min, x_max = df['X1'].min() - 1, df['X1'].max() + 1
        y_min, y_max = df['X2'].min() - 1, df['X2'].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        if task == 1:
            Z = model.sigmoid(np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()].dot(theta))
        elif task == 2:
            Z = model.sigmoid(np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel(), (xx.ravel() * yy.ravel())].dot(theta))
        else:
            out = vectorized_map(xx.ravel(), yy.ravel())
            Z = model.sigmoid(out.dot(theta))
        
        Z = Z.reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], colors='green')
    
    path = "plot_" + str(task) + ".png"
    plt.savefig(path)
    print("Data plot with decision boundary saved at: ", path)

def task_1():
    learning_rate = 0.001
    epochs = 1000

    file = "task1_data.csv"
    df = load_data(file)

    m = len(df)                                                  # Number of training examples
    X = np.hstack((np.ones((m, 1)), df[['X1', 'X2']].values))    # Add bias term to X
    y = np.array(df.label.values).reshape(-1, 1)                 # Convert y to column vector
    initial_theta = np.zeros(shape=(X.shape[1], 1))              # Initialize theta to zeros

    global model
    model = LogisticRegression(learning_rate=learning_rate, epochs=epochs)  # Initialize model
    theta = model.grad_descent(X, y, initial_theta)                         # Perform gradient descent

    print('Theta found by gradient descent:\n', theta)
    visualize(df, 1, theta)                                                 # Plot the decision boundary on the data points

def task_2():
    learning_rate = 0.001
    epochs = 1000

    file = "task2_data.csv"
    df = load_data(file)

    df["X1X2"] = df["X1"] * df["X2"]  
    m = len(df)
    X = np.hstack((np.ones((m, 1)), df[['X1', 'X2', 'X1X2']].values))       # Create X with new feature
    y = np.array(df.label.values).reshape(-1, 1)                            # Convert y to column vector
    initial_theta = np.zeros(shape=(X.shape[1], 1))                         # Initialize theta to zeros

    global model
    model = LogisticRegression(learning_rate=learning_rate, epochs=epochs)  # Initialize model
    theta = model.grad_descent(X, y, initial_theta)                         # Perform gradient descent

    print('Theta found by gradient descent:\n', theta)
    visualize(df, 2, theta)                                                 # Plot the decision boundary on the data points

def task_3():
    learning_rate = 0.01
    epochs = 1000
    lambda_value = 0

    file_name = "task3_data.csv"
    df = load_data(file_name)

    m = len(df)                                                              # Get the number of training examples
    X = vectorized_map(df['X1'].values, df['X2'].values)                     # Create polynomial features
    y = np.array(df.label.values).reshape(-1, 1)                             # Convert y to column vector
    initial_theta = np.zeros(shape=(X.shape[1], 1))                          # Initialize theta to zeros

    global model
    model = LogisticRegression(learning_rate=learning_rate, epochs=epochs, lambda_value=lambda_value)  # Initialize model
    theta = model.grad_descent(X, y, initial_theta)                         # Perform gradient descent

    print('Theta found by gradient descent:\n', theta)
    visualize(df, 3, theta)                                                  # Plot the decision boundary on the data points

if __name__ == '__main__':
    # Change the working directory to the location of your script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Uncomment the line below to run task 1
    task_1()
    
    # Uncomment the line below to run task 2 
    task_2()
    
    # Uncomment the line below to run task 3 
    task_3()
