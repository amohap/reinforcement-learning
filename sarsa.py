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
import os

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

#W1 = np.random.uniform(0, 1,(N_h,N_in))/(N_h*N_in) won't converge
#W2 = np.random.uniform(0, 1,(N_a,N_h))/(N_a*N_h) won't converge

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

#SARSA
for n in range(N_episodes):
    S,X,allowed_a=env.Initialise_game()
   
    epsilon_f = args.epsilon / (1 + args.beta * n)   ## DECAYING EPSILON
    Done=0                                   ## SET DONE TO ZERO (BEGINNING OF THE EPISODE)
    i = 1                                    ## COUNTER FOR NUMBER OF ACTIONS
                               
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
            
            break
        
        
        # IF THE EPISODE IS NOT OVER...
        else:
            
            Qvalues1=ComputeQvalues(W1, W2, bias_W1, bias_W2, X_next, hiddenactivfunction , outeractivfunction)

            a1=EpsilonGreedy_Policy(Qvalues1,epsilon_f, allowed_a_next)

            # Compute the delta
            dEdQ=R+args.gamma*Qvalues1[a1]- Qvalues[a]
         
                  
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
          
            W2[a,:]=   W2[a,:]+   args.eta*dEdQ*dQdY*dYdW
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
           
            
        # NEXT STATE AND CO. BECOME ACTUAL STATE...     
        S=np.copy(S_next)
        X=np.copy(X_next)
        a = np.copy(a1)

        i += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS

filename = os.path.splitext(__file__)[0]

save_path_dataframe_nmoves="experiments/{}/ep{}_be{}_ga{}_et{}/N_moves_SARSA.csv".format(filename, args.epsilon, args.gamma, args.eta)
save_path_plots_nmoves="experiments/{}/ep{}_be{}_ga{}_et{}/nmoves.png".format(filename, args.epsilon, args.gamma, args.eta)

save_path_dataframe_reward="experiments/{}/ep{}_be{}_ga{}_et{}/R_save_SARSA.csv".format(filename, args.epsilon, args.gamma, args.eta)
save_path_plots_rsave="experiments/{}/ep{}_be{}_ga{}_et{}/rsave.png".format(filename, args.epsilon, args.gamma, args.eta)

save_path_dataframe_delta="experiments/{}/ep{}_be{}_ga{}_et{}/Delta_save_SARSA.csv".format(filename, args.epsilon, args.gamma, args.eta)
save_path_plots_delta="experiments/{}/ep{}_be{}_ga{}_et{}/delta.png".format(filename, args.epsilon, args.gamma, args.eta)

# Plot the performance
N_moves_save = pd.DataFrame(N_moves_save, columns = ['N_moves'])
N_moves_save['N_moves'] = N_moves_save['N_moves'].ewm(span=100, adjust=False).mean()
N_moves_save.to_csv(save_path_dataframe_nmoves)

plt.plot(N_moves_save['N_moves'])
plt.xlabel('Episodes')
plt.ylabel('Number of Steps until "Done"')
plt.title('Average Number of Steps until "Done" per Episode')
# plt.show()
plt.savefig(save_path_plots_nmoves)


R_save = pd.DataFrame(R_save, columns = ['R_save'])
R_save['R_save'] = R_save['R_save'].ewm(span=100, adjust=False).mean()
R_save.to_csv(save_path_dataframe_reward)


plt.plot(R_save)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Average Rewards per Episode')
# plt.show()
plt.savefig(save_path_plots_rsave)


Delta_save = pd.DataFrame(Delta_save, columns = ['Delta_save'])
Delta_save['Delta_save'] = Delta_save['Delta_save'].ewm(span=100, adjust=False).mean()
Delta_save.to_csv(save_path_dataframe_delta)


plt.plot(Delta_save)
plt.xlabel('Episodes')
plt.ylabel('Error')
plt.title('Average Loss')
# plt.show()
plt.savefig(save_path_plots_delta)