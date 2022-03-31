import matplotlib.pyplot as plt
import pandas as pd

delta_1 = pd.read_csv('Delta_save_SARSA.csv')
delta_2 = pd.read_csv('Delta_save_QLearning.csv')
# print(delta_1['Delta_save'])
delta_1['Delta_save'] = delta_1['Delta_save'].ewm(span=2000, adjust=False).mean()
delta_2['Delta_save'] = delta_2['Delta_save'].ewm(span=2000, adjust=False).mean()
plt.plot(delta_1['Delta_save'], label='SARSA')
plt.plot(delta_2['Delta_save'], label='Q Learning')
plt.xlabel('Episodes')
plt.ylabel('Error')
plt.title('Average Loss')
plt.legend()
# plt.show()
plt.savefig('svsq_report_delta.png')
plt.close()

N_moves_1 = pd.read_csv('N_moves_SARSA.csv')
N_moves_2 = pd.read_csv('N_moves_QLearning.csv')
N_moves_1['N_moves'] = N_moves_1['N_moves'].ewm(span=2000, adjust=False).mean()
N_moves_2['N_moves'] = N_moves_2['N_moves'].ewm(span=2000, adjust=False).mean()
plt.plot(N_moves_1['N_moves'], label='SARSA')
plt.plot(N_moves_2['N_moves'], label='Q Learning')
plt.xlabel('Episodes')
plt.ylabel('Number of Steps until "Done"')
plt.title('Average Number of Steps until "Done" per Episode')
# plt.show()
plt.savefig('svsq_report_N_moves.png')
plt.close()

reward_1 = pd.read_csv('R_save_SARSA.csv')
reward_2 = pd.read_csv('R_save_QLearning.csv')
reward_1['R_save'] = reward_1['R_save'].ewm(span=2000, adjust=False).mean()
reward_2['R_save'] = reward_2['R_save'].ewm(span=2000, adjust=False).mean()
plt.plot(reward_1['R_save'], label='SARSA')
plt.plot(reward_2['R_save'], label='Q Learning')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Average Rewards per Episode')
plt.legend()
# plt.show()
plt.savefig('svsq_report_reward.png')
plt.close()