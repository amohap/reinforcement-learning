import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import pandas as pd
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from generate_game import *
from Chess_env import *
import random

import argparse

parser = argparse.ArgumentParser(description='Hyperparameter of SARSA algorithm')
parser.add_argument('--epsilon', type=float, help='Starting value of Epsilon for the Epsilon-Greedy policy', default=0.2)
parser.add_argument('--beta', type=float, help='The parameter sets how quickly the value of Epsilon is decaying',default=0.00005)
parser.add_argument('--gamma', type=float, help='The Discount Factor', default=0.85)
parser.add_argument('--eta', type=float, help='The Learning Rate', default=0.0035)
args = parser.parse_args()

## INITIALISE THE ENVIRONMENT
size_board = 4
env=Chess_Env(size_board)

def EpsilonGreedy_Policy(Qvalues, epsilon, allowed_a):
    
    N_a=np.shape(Qvalues)[0]
    a = list(range(0, N_a))
    a = [i for idx, i in enumerate(a) if allowed_a[idx]]
    Qvalues = [i for idx, i in enumerate(Qvalues) if allowed_a[idx]]

    rand_value=np.random.uniform(0,1)
 ## epsilon is probability that we go random
    rand_a=rand_value<epsilon

    if rand_a==True:
        
        a = random.choice(a)

    else:
        idx=np.argmax(Qvalues)
     
        a = a[idx]
            
    return a

def ComputeQvalues(W1, W2, bias_W1, bias_W2, X, hiddenactivfunction , outeractivfunction):
    ## Qvalues=np.matmul(W2, np.matmul(W1,X)) ## this is direct computation of hidden layer and then output layer, without applying any non linear activation function
    ## below is a better solution:
    # Neural activation: input layer -> hidden layer
    H1 = np.matmul(W1,X)+bias_W1 ## make sure that bias_W1 does not need to be transposed
    # if hidden activ function is given:
    if (hiddenactivfunction == 1):
        H1 = np.round(1/(1+np.exp(-H1)), 5) ## sigmoid
    elif(hiddenactivfunction == 2):
         H1 = (H1>0).astype(int)*H1  ## RELU
        
    Qvalues = np.matmul(W2,H1) + bias_W2
    
    #if outer activ function is given
    if (outeractivfunction == 1):
        Qvalues = np.round(1/(1+np.exp(- Qvalues)), 5)
    elif (outeractivfunction == 2):
        Qvalues = (Qvalues>0).astype(int)*Qvalues

    return Qvalues
