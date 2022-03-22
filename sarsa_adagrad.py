# Import 
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


## INITIALISE THE ENVIRONMENT
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

bias_W1 = np.zeros((N_h,))
bias_W2 = np.zeros((N_a,))