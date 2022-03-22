#!/bin/bash

epsilons=(1 2 4 8 16 32 64 128)
betas=(-1 -2 -4 -8 -16 -32 -64 -128)
gammas=(1 2 4 8 16 32 64 128)
etas=(1 2 4 8 16 32 64 128)

for epsilon in ${epsilons[@]}; do
    for beta in ${betas[@]}; do
        for gamma in ${gammas[@]}; do
            for eta in ${etas[@]}; do
                path="ep${epsilon}_be${beta}_ga${gamma}_et${eta}"
                if [ ! -d experiments/sarsa_adagrad/$path ]; then
                    mkdir -p experiments/sarsa_adagrad/$path;
                fi
                echo "Epsilon: $epsilon" >> experiments/sarsa_adagrad/$path/hyperparam.txt
                echo "Beta: $beta" >> experiments/sarsa_adagrad/$path/hyperparam.txt
                echo "Gamma: $gamma" >> experiments/sarsa_adagrad/$path/hyperparam.txt
                echo "Eta: $eta" >> experiments/sarsa_adagrad/$path/hyperparam.txt                
                python sarsa_adagrad.py --epsilon $epsilon --beta $beta --gamma $gamma $eta eta
            done
        done
    done
done