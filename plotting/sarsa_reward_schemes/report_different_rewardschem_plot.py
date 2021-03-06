import matplotlib.pyplot as plt
import pandas as pd

delta_1 = pd.read_csv('./(-1)-0-1/Delta_save_SARSA.csv')
delta_2 = pd.read_csv('./0-0-1/Delta_save_SARSA.csv')
delta_3 = pd.read_csv('./0-0.1-1/Delta_save_SARSA.csv')
delta_4 = pd.read_csv('./1-0.1-0/Delta_save_SARSA.csv')
# print(delta_1['Delta_save'])
delta_1['Delta_save'] = delta_1['Delta_save'].ewm(span=2000, adjust=False).mean()
delta_2['Delta_save'] = delta_2['Delta_save'].ewm(span=2000, adjust=False).mean()
delta_3['Delta_save'] = delta_3['Delta_save'].ewm(span=2000, adjust=False).mean()
delta_4['Delta_save'] = delta_4['Delta_save'].ewm(span=2000, adjust=False).mean()
plt.plot(delta_1['Delta_save'], label='Scheme 1')
plt.plot(delta_2['Delta_save'], label='Scheme 2')
plt.plot(delta_3['Delta_save'], label='Scheme 3')
plt.plot(delta_4['Delta_save'], label='Scheme 4')
plt.xlabel('Episodes')
plt.ylabel('Error')
plt.title('Average Loss')
plt.legend()
# plt.show()
plt.savefig('report_delta.png')
plt.close()

N_moves_1 = pd.read_csv('./(-1)-0-1/N_moves_SARSA.csv')
N_moves_2 = pd.read_csv('./0-0-1/N_moves_SARSA.csv')
N_moves_3 = pd.read_csv('./0-0.1-1/N_moves_SARSA.csv')
N_moves_4 = pd.read_csv('./1-0.1-0/N_moves_SARSA.csv')
N_moves_1['N_moves'] = N_moves_1['N_moves'].ewm(span=2000, adjust=False).mean()
N_moves_2['N_moves'] = N_moves_2['N_moves'].ewm(span=2000, adjust=False).mean()
N_moves_3['N_moves'] = N_moves_3['N_moves'].ewm(span=2000, adjust=False).mean()
N_moves_4['N_moves'] = N_moves_4['N_moves'].ewm(span=2000, adjust=False).mean()
plt.plot(N_moves_1['N_moves'], label='Scheme 1')
plt.plot(N_moves_2['N_moves'], label='Scheme 2')
plt.plot(N_moves_3['N_moves'], label='Scheme 3')
plt.plot(N_moves_4['N_moves'], label='Scheme 4')
plt.xlabel('Episodes')
plt.ylabel('Number of Steps until "Done"')
plt.title('Average Number of Steps until "Done" per Episode')
plt.legend()
# plt.show()
plt.savefig('report_N_moves.png')
plt.close()

reward_1 = pd.read_csv('./(-1)-0-1/R_save_SARSA.csv')
reward_2 = pd.read_csv('./0-0-1/R_save_SARSA.csv')
reward_3 = pd.read_csv('./0-0.1-1/R_save_SARSA.csv')
reward_4 = pd.read_csv('./1-0.1-0/R_save_SARSA.csv')
reward_1['R_save'] = reward_1['R_save'].ewm(span=2000, adjust=False).mean()
reward_2['R_save'] = reward_2['R_save'].ewm(span=2000, adjust=False).mean()
reward_3['R_save'] = reward_3['R_save'].ewm(span=2000, adjust=False).mean()
reward_4['R_save'] = reward_4['R_save'].ewm(span=2000, adjust=False).mean()
plt.plot(reward_1['R_save'], label='Scheme 1')
plt.plot(reward_2['R_save'], label='Scheme 2')
plt.plot(reward_3['R_save'], label='Scheme 3')
plt.plot(reward_4['R_save'], label='Scheme 4')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Average Rewards per Episode')
plt.legend()
# plt.show()
plt.savefig('report_reward.png')
plt.close()