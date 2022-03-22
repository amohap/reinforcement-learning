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

bias_W1 = np.zeros((N_h,))
bias_W2 = np.zeros((N_a,))

# REWARD SCHEME DEFAULT
# Reward if episode is not ended: 0
# Reward if checkmate: 1
# Reward if draw: 0

N_episodes = 100000 # THE NUMBER OF GAMES TO BE PLAYED 100000

hiddenactivfunction = 0
outeractivfunction = 1

# SAVING VARIABLES
R_save = np.zeros([N_episodes, 1])
N_moves_save = np.zeros([N_episodes, 1])
Delta_save = np.zeros([N_episodes, 1])

for n in range(N_episodes):
    S,X,allowed_a=env.Initialise_game()
    epsilon_f = args.epsilon / (1 + args.beta * n)   ## DECAYING EPSILON
    Done=0                                   ## SET DONE TO ZERO (BEGINNING OF THE EPISODE)
    i = 1  ## COUNTER FOR NUMBER OF ACTIONS
    d = {}
    
    
    while (Done==0 and i<50):                           ## START THE EPISODE
       
        Qvalues = ComputeQvalues(W1, W2, bias_W1, bias_W2, X, hiddenactivfunction , outeractivfunction)
        a=EpsilonGreedy_Policy(Qvalues,epsilon_f, allowed_a)
        S_next, X_next,allowed_a_next,R,Done=env.OneStep(a)
        dataToStore = (X,a,R,X_next,Done, allowed_a_next)
        d[i] = dataToStore
     
        
        if Done==1:

            R_save[n,]=np.copy(R)
            N_moves_save[n,]=i
        
            while i>0: 
                x = d[i][0]
                a = d[i][1]
                r = d[i][2] 
                next_x = d[i][3]
                done = d[i][4]
                allowed = d[i][5]
                Qvalues=ComputeQvalues(W1, W2, bias_W1, bias_W2, x, hiddenactivfunction , outeractivfunction)
            
                if done:
                    dEdQ=r-Qvalues[a]
                else:
                    Qvalues1=ComputeQvalues(W1, W2, bias_W1, bias_W2, next_x, hiddenactivfunction , outeractivfunction)
                    a1=EpsilonGreedy_Policy(Qvalues1,0, allowed)
                    dEdQ=R+args.gamma*Qvalues1[a1]- Qvalues[a]

                Delta_save[n,]=0.5*(dEdQ)*(dEdQ)
                
                ## update W2 and B2
                dQdY = 1  
                if outeractivfunction == 1:
                        dYdQ = Qvalues[a]*(1-Qvalues[a])
                elif outeractivfunction == 2:
                        dYdQ = (Qvalues[a]>0).astype(int)

                H = np.matmul(W1,X) + bias_W1
                if hiddenactivfunction == 1:
                         H = np.round(1/(1+np.exp(-H)), 5)
                elif(hiddenactivfunction == 2):
                         H = (H>0).astype(int)*H
                dYdW = H

                W2[a,:]=W2[a,:]+args.eta*dEdQ*dQdY*dYdW
                bias_W2[a]=bias_W2[a]+args.eta*dEdQ*dQdY

                ## update W1 and B1 after W2 and B2 were updated
                if hiddenactivfunction == 1:
                    dYdZ =  (W2[a,:].reshape(1, 200) * H*(1-H).reshape(1, 200)).reshape(200,1)
                elif(hiddenactivfunction == 2):
                    dYdZ =  (W2[a,:].reshape(1, 200) * (H>0).astype(int)).reshape(200,1)
                else:
                    dYdZ =  W2[a,:].reshape(200, 1)

                dZDW = X.reshape(1, 58)        
                W1[:,:]=W1[:,:]+ args.eta*dEdQ*dQdY*dYdZ*dZDW 
                bias_W1=bias_W1+ args.eta*dEdQ*dQdY*dYdZ.reshape(200,) 
                i = i-1
                break
        
        else:
            
            S=np.copy(S_next)
            X=np.copy(X_next)
            allowed_a = np.copy(allowed_a_next)
            i = i+1

# Plot the performance
N_moves_save = pd.DataFrame(N_moves_save, columns = ['N_moves'])
N_moves_save.to_csv('N_moves_ER.csv')
N_moves_save['N_moves'] = N_moves_save['N_moves'].ewm(span=100, adjust=False).mean()


plt.plot(N_moves_save['N_moves'])
plt.xlabel('Episodes')
plt.ylabel('Number of Steps until "Done"')
plt.title('Average Number of Steps until "Done" per Episode')
# plt.show()

R_save = pd.DataFrame(R_save, columns = ['R_save'])
R_save.to_csv('R_save_ER.csv')
R_save['R_save'] = R_save['R_save'].ewm(span=100, adjust=False).mean()

plt.plot(R_save)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Average Rewards per Episode')
# plt.show()


Delta_save = pd.DataFrame(Delta_save, columns = ['Delta_save'])
Delta_save.to_csv('Delta_save_ER.csv')
Delta_save['Delta_save'] = Delta_save['Delta_save'].ewm(span=100, adjust=False).mean()

plt.plot(Delta_save)
plt.xlabel('Episodes')
plt.ylabel('Error')
plt.title('Average Loss')
# plt.show()
