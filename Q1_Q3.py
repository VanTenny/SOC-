import numpy as np
import pandas as pd
 
def load_data():
    train_data = pd.read_csv("/Users/pdg/Desktop/SOC-/splits/train_data.csv",header=None)
    dev_data = pd.read_csv("/Users/pdg/Desktop/SOC-/splits/dev_data.csv", header=None)
    hidden_test_data = pd.read_csv("/Users/pdg/Desktop/SOC-/splits/hidden_test_data.csv", header=None)
    test_data = pd.read_csv("/Users/pdg/Desktop/SOC-/splits/test_data.csv", header=None)
    test_labels = pd.read_csv("/Users/pdg/Desktop/SOC-/splits/test_labels.csv", header=None).values.ravel()

    return train_data, dev_data, test_data, hidden_test_data, test_labels

def prepare_data(data):
    if data.shape[1] > 90:              # If the given data has more than 90 columns I assume it to have the data I want
        y = data.iloc[ : , 0 ].values   # I keep the first column as my target data
        X = data.iloc[ : , 1: ].values  # I keep the other columns as feature values where I add a bias term
    else:
        X = data
        y = None
    X = np.hstack((np.ones((X.shape[0], 1)), X)) # I then add the bias column to the featured columns
    return X,y

def train_model(X_train, y_train):
    X_T = np.transpose(X_train)                            # this takes the transpose of the matrix X
    w = (np.linalg.inv(X_T @ X_train)) @ X_T @ y_train     # Now we write the closed equation for w the least squares solution
    return w

def predict (X, w):
    Y1 = X @ w                                              # this just gives us the predicted values of function y  
    return Y1

def error_calculation(Y1, Y):                
    MSE = np.mean((Y - Y1)**2)                              # this is the function for Mean Squared Error 
    RMSE = np.sqrt(MSE)                                     # and this for the square root of MSE
    return MSE, RMSE

def TF(X, mu, s):                                           # a function to find the radial basis function
    RBF = np.exp(-np.sum((X[:, 1:] - mu) ** 2, axis=1, keepdims=True) / (2 * s ** 2))
    X_TF = np.hstack((X, RBF))
    return X_TF

# Now we start the actual process of examining data and making the predictions while also checking the error
if __name__ == "__main__":  # helps to see if the file is either run as main or as a module 

    train_data, dev_data, test_data, hidden_test_data, test_labels = load_data()          # loads the csv flile datas

    X_train, y_train = prepare_data(train_data)                                           # these just get the data required from the respective csv files
    X_dev, y_dev = prepare_data(dev_data)
    X_test, _ = prepare_data(test_data)
    X_hidden_test, _ = prepare_data(hidden_test_data)

    w = train_model(X_train, y_train)                                                      # this gives us the values for w
    
    # We now calculate for developing data csv file
    Y1_dev = predict(X_dev, w)                  
    MSE_dev, RMSE_dev = error_calculation(Y1_dev, y_dev)
    print(f"The Mean Squared Error for dev: {MSE_dev}")
    print(f"The Root Mean Squared Error for dev: {RMSE_dev}")

    # This one for test files
    Y1_test = predict(X_test, w)
    _, RMSE_test = error_calculation(Y1_test, test_labels)
    print(f"The Root Mean Squared Error for test: {RMSE_test}")

    # Now for Question 3 
    mu = np.mean(X_train[:, 1:], axis=0)             # assigning the mu as mean of the featured columns
    s = 1                                            # the standard deviation is set to 1
    X_train_3 = TF(X_train, mu, s)
    X_dev_3 = TF(X_dev, mu, s)
    X_test_3 = TF(X_test, mu, s)
    X_hidden_test_3 = TF(X_hidden_test, mu, s)

    Y1_hid = predict(X_hidden_test, w)

    # As for saving the predictions
    df = pd.DataFrame(Y1_hid, columns=['Year'])
    df.index.name = 'ID'
    df.to_csv('roll_number.csv', index=True)

