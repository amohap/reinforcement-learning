#!/bin/bash

epsilons=(0.1 0.2 0.9)
betas=(0.00005 0.0005 0.005)
gammas=(0.8 0.85 1)
etas=(0.0035 0.005 0.05)

for epsilon in ${epsilons[@]}; do
    for beta in ${betas[@]}; do
        for gamma in ${gammas[@]}; do
            for eta in ${etas[@]}; do
                path="ep${epsilon}_be${beta}_ga${gamma}_et${eta}"
                if [ ! -d experiments/sarsa/$path ]; then
                    mkdir -p experiments/sarsa/$path;
                fi
                echo "Epsilon: $epsilon" >> experiments/sarsa/$path/hyperparam.txt
                echo "Beta: $beta" >> experiments/sarsa/$path/hyperparam.txt
                echo "Gamma: $gamma" >> experiments/sarsa/$path/hyperparam.txt
                echo "Eta: $eta" >> experiments/sarsa/$path/hyperparam.txt
                python sarsa.py --epsilon $epsilon --beta $beta --gamma $gamma $eta eta
            done
        done
    done
done


