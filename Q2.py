import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

class Linear_Regression_Batch:
    def __init__(self, learnin_rate = 0.01, max_epochs = 200, batch_size = None):
        self.learnin_rate = learnin_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.weights = None
    
    def fit(self, X, y, X_dev, y_dev):

        if self.batch_size is None:
            self.batch_size = X.shape[0]        # Keeping the batch size as the no. of rows in X
        
        self.weights = np.zeros((X.shape[1],1)) # The weights being the 2-D array of no.of rows as X and one column
        prev_weights = self.weights

        self.error_list = []                    # The errors stay in this list later on
        for epoch in range(self.max_epochs):
            batches = create_bt(X, y, self.batch_size)
            for batch in batches:
                X_batch, y_batch = batch
                gradient = self.compute_grd(X_batch, y_batch, self.weights)
                self.weights -= self.learnin_rate * gradient
        
            loss = self.compute_rmse_loss(X_dev, y_dev, self.weights) # This computes the loss on development set
            self.error_list.append(loss)                              # stores the loss

            if np.linalg.norm(self.weights - prev_weights) < 1e-5:    # checkin for convergence
                print(f"We stoppin at epoch : {epoch}.")
                break

            prev_weights = self.weights.copy()                        # Ensure previous weights are updated

        print("The trainings complete.")
        print("Mean validation RMSE loss : ", np.mean(self.error_list))
        print("Batch size : ", self.batch_size)
        print("Learnin rate : ", self.learnin_rate)
        plot_loss(self.error_list, self.batch_size)

    def predict(self, X):
        return X @ self.weights  # gives the predicted new data
    
    def compute_rmse_loss(self, X, y, weights):
        predictions = X @ weights                    
        errors = predictions - y
        rmse_error = np.sqrt(np.mean(errors ** 2))
        return rmse_error
    
    def compute_grd(self, X, y, weights):
        predictions = X @ weights
        errors = predictions - y
        gradient = (2/X.shape[0]) * X.T @ errors
        return gradient

def plot_loss(error_list, batch_size):
    plt.plot(error_list, label = f"Batch size: {batch_size}")    # Plottin the loss values
    plt.xlabel("Epochs")
    plt.ylabel("RMSE loss")
    plt.legend()
    plt.title("Loss vs Epochs")
    plt.savefig("figures/plot.png")
    plt.show()

def standard_scaler(data):
    mean = np.mean(data , axis = 0)
    std = np.std(data, axis = 0)
    standarized_data = (data - mean)/std
    return standarized_data

def create_bt(X, y, batch_size):
    batches =[]
    data = np.hstack((X, y))      # Combinin them
    np.random.shuffle(data)       # and then shuffle em yo
    no_of_bt = data.shape[0] // batch_size
    
    for i in range(no_of_bt+1):
        if i < no_of_bt:
            batch = data[i * batch_size:(i + 1) * batch_size, :]         # Full batch.
        elif data.shape[0] % batch_size != 0 and i == no_of_bt:
            batch = data[i * batch_size:data.shape[0]]                   # Last batch with remaining data.
        
        X_batch = batch[:, :-1]
        y_batch = batch[:, -1].reshape((-1, 1))
        batches.append((X_batch, y_batch))
    
    return batches

def load_train_dev_dataset():
    train_set = pd.read_csv("/Users/pdg/Desktop/lab2/splits/train_data.csv", header=None)
    dev_set = pd.read_csv("/Users/pdg/Desktop/lab2/splits/dev_data.csv", header=None)

    X_train = train_set.iloc[ : , 1: ].to_numpy()                # Features
    y_train = train_set.iloc[ : , :1].to_numpy().reshape(-1,1)   # Targets
    y_t_mean, y_t_std = np.mean(y_train, axis = 0), np.std(y_train, axis =0)

    X_dev = dev_set.iloc[ : , 1:].to_numpy()
    y_dev = dev_set.iloc[ : , :1].to_numpy().reshape(-1,1)

    X_train, y_train, X_dev, y_dev = scalin(X_train, y_train, X_dev, y_dev)
    X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # Add bias term.
    X_dev = np.c_[np.ones((X_dev.shape[0], 1)), X_dev]

    return X_train, y_train, X_dev, y_dev, y_t_mean, y_t_std

def scalin(X_train, y_train, X_dev, y_dev):
    X_train = standard_scaler(X_train)  # Standardize training features.
    y_train = standard_scaler(y_train)  # Standardize training targets.
    X_dev = standard_scaler(X_dev)      # Standardize development features.
    y_dev = standard_scaler(y_dev)      # Standardize development targets.
    return X_train, y_train, X_dev, y_dev 

def load_test_dataset():
    X_test = pd.read_csv("/Users/pdg/Desktop/lab2/splits/test_data.csv", header = None).to_numpy()
    y_test = pd.read_csv("/Users/pdg/Desktop/lab2/splits/test_labels.csv", header=None).to_numpy().reshape(-1, 1)
    X_test = standard_scaler(X_test)                        # Standardize test features.
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]   # Add bias term.
    return X_test, y_test

def evaluate_model(weights, X, y, ymean, ystd):
    y_pred_scaled = X @ weights                             # Predict scaled target values.
    y_pred_actual = y_pred_scaled * ystd + ymean            # Convert to actual values.
    difference = y_pred_actual - y  
    rmse = np.sqrt(np.mean(difference**2))  
    return rmse

def save_pred(ymean, ystd, weights):
    X = pd.read_csv("/Users/pdg/Desktop/lab2/splits/hidden_test_data.csv", header=None).to_numpy()
    X = standard_scaler(X)  
    X = np.c_[np.ones((X.shape[0], 1)), X]  
    predictions = X @ weights  
    y_pred_hidden_test = predictions * ystd + ymean                                                                          # Convert scaled predictions to actual values
    pd.DataFrame(y_pred_hidden_test, columns=['Year']).to_csv('roll_number.csv', index=True, header=True, index_label="ID")  # Save predictions

if __name__ == '__main__':
    learnin_rate = 0.0001      # Set learning rate
    batch_size = None          # Set batch size (None means full-batch gradient descent)
    max_epochs = 150           # Set maximum number of epochs

    X_train, y_train, X_dev, y_dev, y_train_mean, y_train_std = load_train_dev_dataset()  
    X_test, y_test = load_test_dataset()  # Load test dataset

    model = Linear_Regression_Batch(learnin_rate= learnin_rate, max_epochs= max_epochs, batch_size=batch_size)  # Create model instance
    model.fit(X_train, y_train, X_dev, y_dev)                                                                   # Train the model
    print(evaluate_model(model.weights, X_test, y_test, y_train_mean, y_train_std))                             # Print RMSE on test dataset
    save_pred(y_train_mean, y_train_std, model.weights)                                                         # Save predictions for hidden test dataset
