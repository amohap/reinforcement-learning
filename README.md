# Reinforcement Learning
Implementation and report on a Reinforcement Learning problem solved by Deep RL. 

Link to out [Report](https://www.overleaf.com/project/620e551a6b5487b020e3aa29).

## Running Experiments for Hyperparameter Tunning

Running these below mentioned scripts creates a new directory inside `experiments` with algorithm and the
hyperparameters as the directory name, example `sarsa_adagrad/ep0.1_be0.2_ga0.3_et0.4/`

 - `measurements_sarsa.sh`
 - `measurements_sarsa_adagrad.sh`
 - `measurements_q_learning.sh`
 - `measurements_ex_replay.sh`

 In addition to this, the hyperparameters are saved in a text file `hyperparam.txt` and the plots are also saved inside this directory.