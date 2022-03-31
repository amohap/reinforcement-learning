# Reinforcement Learning
Implementation and report on a Reinforcement Learning problem solved by Deep RL. 

Link to out [Report](https://www.overleaf.com/project/620e551a6b5487b020e3aa29).

---

## Running Experiments for Hyperparameter Tunning

Running these below mentioned scripts creates a new directory inside `experiments` with algorithm and the
hyperparameters as the directory name, example `sarsa_adagrad/ep0.1_be0.2_ga0.3_et0.4/`

 - `measurements_sarsa.sh`
 - `measurements_sarsa_adagrad.sh`
 - `measurements_q_learning.sh`
 - `measurements_ex_replay.sh`

 In addition to this, the hyperparameters are saved in a text file `hyperparam.txt` and the plots are also saved inside this directory.

---
Experience replay: in ex_replay.py incremental version is stored. Once the episode is finished we take the batch fromthe database and unfold it backwards recalculating Q values. In Assignement - default.py mini batch version is realized.

## SARSA vs. Q-Learning

##### Number of Steps per Episode
![N_steps_sarsa_qlearning](https://github.com/amohap/reinforcement-learning/blob/main/plotting/sarsa_qlearning_default/svsq_better_report_N_moves.png)

##### Reward
![Reward_sarsa_qlearning](https://github.com/amohap/reinforcement-learning/blob/main/plotting/sarsa_qlearning_default/svsq_better_report_reward.png)

##### Loss
![Loss_sarsa_qlearning](https://github.com/amohap/reinforcement-learning/blob/main/plotting/sarsa_qlearning_default/svsq_better_report_delta.png)

---

## SARSA vs. Q-Learning vs. SARSA Adagrad vs. Q-Learning with Expererience Replay

##### Number of Steps per Episode
![N_steps_sarsa_qlearning](https://github.com/amohap/reinforcement-learning/blob/main/plotting/sarsa_qlearning_default/comp_new_report_N_moves.png)

##### Reward
![Reward_sarsa_qlearning](https://github.com/amohap/reinforcement-learning/blob/main/plotting/sarsa_qlearning_default/comp_new_report_reward.png)

##### Loss
![Loss_sarsa_qlearning](https://github.com/amohap/reinforcement-learning/blob/main/plotting/sarsa_qlearning_default/comp_new_report_delta.png)
