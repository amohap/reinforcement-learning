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


# set environment
size_board = 4
env=Chess_Env(size_board)

# set seed
np.random.seed(2022)

# INITIALISE THE PARAMETERS OF YOUR NEURAL NETWORK
S,X,allowed_a=env.Initialise_game()

N_a=np.shape(allowed_a)[0]   # TOTAL NUMBER OF POSSIBLE ACTIONS
N_in=np.shape(X)[0]          ## INPUT SIZE
N_h=200                      ## NUMBER OF HIDDEN NODES

## INITALISE YOUR NEURAL NETWORK... Here weights from input to hidden layer and from the hidden layer to output layer are initialized
W1 = np.random.randn(N_h, N_in) * np.sqrt(1 / (N_in)) 
W2 = np.random.randn(N_a, N_h) * np.sqrt(1 / (N_h))
