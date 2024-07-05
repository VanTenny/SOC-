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
        
