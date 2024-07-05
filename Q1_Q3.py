import numpy as np
import pandas as pd

def load_data():
    train_data = pd.read_csv("/Users/pdg/Desktop/lab2/splits/train_data.csv", header=None)
    dev_data = pd.read_csv("/Users/pdg/Desktop/lab2/splits/dev_data.csv", header=None)
    hidden_test_data = pd.read_csv("/Users/pdg/Desktop/lab2/splits/hidden_test_data.csv", header=None)
    test_data = pd.read_csv("/Users/pdg/Desktop/lab2/splits/test_data.csv", header=None)
    test_labels = pd.read_csv("/Users/pdg/Desktop/lab2/splits/test_labels.csv", header=None).values.ravel()

    return train_data, dev_data, test_data, hidden_test_data, test_labels

def prepare_data(data):
    if data.shape[1] > 90:  # If the given data has more than 90 columns, assume it to have the data I want
        y = data.iloc[:, 0].values  # Use the first column as target data
        X = data.iloc[:, 1:].values  # Use the other columns as feature values where I add a bias term
    else:
        X = data
        y = None
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add the bias column to the feature columns
    return X, y

def train_model(X_train, y_train):
    X_T = np.transpose(X_train)  # Take the transpose of the matrix X
    w = (np.linalg.pinv(X_T @ X_train)) @ X_T @ y_train  # Use the closed-form solution for least squares
    return w

def predict(X, w):
    Y1 = X @ w  # Predict the values of function y
    return Y1

def error_calculation(Y1, Y):
    MSE = np.mean((Y - Y1) ** 2)  # Calculate Mean Squared Error
    RMSE = np.sqrt(MSE)  # Calculate Root Mean Squared Error
    return MSE, RMSE

def TF(X, mu, s):
    # Calculate the radial basis function
    RBF = np.exp(-np.sum((X[:, 1:] - mu) ** 2, axis=1, keepdims=True) / (2 * (s ** 2)))
    X_TF = np.hstack((X, RBF))
    return X_TF

# Start the actual process of examining data and making predictions while also checking the error
if __name__ == "__main__":  # Check if the file is run as main or as a module

    # Load the CSV file data
    train_data, dev_data, test_data, hidden_test_data, test_labels = load_data()

    # Get the data required from the respective CSV files
    X_train, y_train = prepare_data(train_data)
    X_dev, y_dev = prepare_data(dev_data)
    X_test, _ = prepare_data(test_data)
    X_hidden_test, _ = prepare_data(hidden_test_data)

    # Get the weights for the model
    w = train_model(X_train, y_train)

    # Calculate for the development data CSV file
    Y1_dev = predict(X_dev, w)
    MSE_dev, RMSE_dev = error_calculation(Y1_dev, y_dev)
    print(f"The Mean Squared Error for dev: {MSE_dev}")
    print(f"The Root Mean Squared Error for dev: {RMSE_dev}")

    # Calculate for the test files
    Y1_test = predict(X_test, w)
    _, RMSE_test = error_calculation(Y1_test, test_labels)
    print(f"The Root Mean Squared Error for test: {RMSE_test}")

    # For Question 3
    mu = np.mean(X_train[:, 1:], axis=0)  # Assign mu as the mean of the feature columns
    s = 1  # Set the standard deviation to 1
    X_train_3 = TF(X_train, mu, s)
    X_dev_3 = TF(X_dev, mu, s)
    X_test_3 = TF(X_test, mu, s)
    X_hidden_test_3 = TF(X_hidden_test, mu, s)

    # Train the model again for the transformed data
    w_3 = train_model(X_train_3, y_train)

    Y1_dev_3 = predict(X_dev_3, w_3)
    MSE_dev_3, RMSE_dev_3 = error_calculation(Y1_dev_3, y_dev)
    print(f"The Mean Squared Error for dev for question 3 : {MSE_dev_3}")
    print(f"The Root Mean Squared Error for dev for question 3: {RMSE_dev_3}")

    # Predict for the hidden test set
    Y1_hid_3 = predict(X_hidden_test_3, w_3)

    # Save the predictions
    df = pd.DataFrame(Y1_hid_3, columns=['Year'])
    df.index.name = 'ID'
    df.to_csv('roll_number.csv', index=True)
