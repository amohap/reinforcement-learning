import argparse
import re

parser = argparse.ArgumentParser(description='Hyperparameter of SARSA algorithm')
parser.add_argument('--epsilon', type=float, help='Starting value of Epsilon for the Epsilon-Greedy policy', required=True)
parser.add_argument('--beta', type=float, help='The parameter sets how quickly the value of Epsilon is decaying', required=True)
parser.add_argument('--gamma', type=float, help='The Discount Factor', required=True)
parser.add_argument('--eta', type=float, help='The Learning Rate', required=True)
args = parser.parse_args()
print("\n")
print(args.epsilon)
print(args.beta)
