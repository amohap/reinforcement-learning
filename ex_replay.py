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
