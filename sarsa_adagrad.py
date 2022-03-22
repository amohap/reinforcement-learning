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
   
    epsilon_f = epsilon_0 / (1 + beta * n)   ## DECAYING EPSILON
    Done=0                                   ## SET DONE TO ZERO (BEGINNING OF THE EPISODE)
    i =  1                                        ## COUNTER FOR NUMBER OF ACTIONS
    if (n%10 ==0):

        diagonal2 = np.zeros((N_h+1,N_a))
        diagonal1 = np.zeros((N_in*N_h + N_h, ))
                               
    Qvalues = ComputeQvalues(W1, W2, bias_W1, bias_W2, X, hiddenactivfunction , outeractivfunction)

    a=EpsilonGreedy_Policy(Qvalues,epsilon_f, allowed_a)
    
    while (Done==0 and i<50):                           ## START THE EPISODE
       
        Qvalues = ComputeQvalues(W1, W2, bias_W1, bias_W2, X, hiddenactivfunction , outeractivfunction)
       
        S_next, X_next,allowed_a_next,R,Done=env.OneStep(a)
       
        ## THE EPISODE HAS ENDED, UPDATE...BE CAREFUL, THIS IS THE LAST STEP OF THE EPISODE
        if Done==1:
            
            R_save[n,]=np.copy(R)
            N_moves_save[n,]=i
            
            dEdQ=R-Qvalues[a]
           
            Delta_save[n,] = 0.5*(dEdQ)*(dEdQ)
            
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
    
            gradient2 =dEdQ*dQdY*dYdW
            grandientBias2 = dEdQ*dQdY
            G = np.round(np.append(gradient2, grandientBias2), 4)
            diagonal2[:,a] = diagonal2[:,a] + G**2
            diagonal_to_use2 = (eps + diagonal2[:,a])**(-1/2)
    
            W2[a,:]=W2[a,:]+eta*diagonal_to_use2[0:-1]*gradient2
            bias_W2[a]=bias_W2[a]+eta*diagonal_to_use2[-1]*grandientBias2
            
            ## update W1 and B1 after W2 and B2 were updated
            if hiddenactivfunction == 1:
                dYdZ =  (W2[a,:].reshape(1, 200) * H*(1-H).reshape(1, 200)).reshape(200,1)
            elif(hiddenactivfunction == 2):
                dYdZ =  (W2[a,:].reshape(1, 200) * (H>0).astype(int)).reshape(200,1)
            else:
                dYdZ =  W2[a,:].reshape(200, 1)
        
            dZDW = X.reshape(1, 58)        
            
            gradient1 =dEdQ*dQdY*dYdZ*dZDW
            grandientBias1 = dEdQ*dQdY*dYdZ.reshape(200,1) 
            G = np.ravel(np.concatenate((gradient1, grandientBias1), axis=1))

            diagonal1 = diagonal1 + G**2
            diagonal_to_use1 = ((eps+diagonal1).reshape(200, -1))**(-1/2)
            
            W1[:,:]=W1[:,:]+ eta*diagonal_to_use1[:, 0:-1]*gradient1
            bias_W1=bias_W1+ eta*diagonal_to_use1[:, -1]*grandientBias1.reshape(200,)
            
            break

        
        # IF THE EPISODE IS NOT OVER...
        else:
            
            Qvalues1=ComputeQvalues(W1, W2, bias_W1, bias_W2, X_next, hiddenactivfunction , outeractivfunction)

            a1=EpsilonGreedy_Policy(Qvalues1,epsilon_f, allowed_a_next)

            # Compute the delta
            dEdQ=R+gamma*Qvalues1[a1] - Qvalues[a]
         
                  
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
          
            gradient2 =dEdQ*dQdY*dYdW
            grandientBias2 = dEdQ*dQdY
            G = np.round(np.append(gradient2, grandientBias2), 4)
            
            diagonal2[:,a] = diagonal2[:,a] + G**2
            diagonal_to_use2 = (eps + diagonal2[:,a])**(-1/2)
    
            W2[a,:]=W2[a,:]+eta*diagonal_to_use2[0:-1]*gradient2
            bias_W2[a]=bias_W2[a]+eta*diagonal_to_use2[-1]*grandientBias2
            
            ## update W1 and B1 after W2 and B2 were updated
            if hiddenactivfunction == 1:
                dYdZ =  (W2[a,:].reshape(1, 200) * H*(1-H).reshape(1, 200)).reshape(200,1)
            elif(hiddenactivfunction == 2):
                dYdZ =  (W2[a,:].reshape(1, 200) * (H>0).astype(int)).reshape(200,1)
            else:
                dYdZ =  W2[a,:].reshape(200, 1)
        
            dZDW = X.reshape(1, 58)    
            
            gradient1 =dEdQ*dQdY*dYdZ*dZDW
            grandientBias1 = dEdQ*dQdY*dYdZ.reshape(200,1) 
            G = np.ravel(np.concatenate((gradient1, grandientBias1), axis=1))

            diagonal1 = diagonal1 + G**2
            diagonal_to_use1 = ((eps+diagonal1).reshape(200, -1))**(-1/2)
                    
            W1[:,:]=W1[:,:]+ eta*diagonal_to_use1[:, 0:-1]*gradient1
            bias_W1=bias_W1+ eta*diagonal_to_use1[:, -1]*grandientBias1.reshape(200,)
           
            
        # NEXT STATE AND CO. BECOME ACTUAL STATE...     
        S=np.copy(S_next)
        X=np.copy(X_next)
        a = np.copy(a1)

        i += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS   