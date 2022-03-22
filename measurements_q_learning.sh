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
                if [ ! -d experiments/q_learning/$path ]; then
                    mkdir -p experiments/q_learning/$path;
                fi
                echo "Epsilon: $epsilon" >> experiments/q_learning/$path/hyperparam.txt
                echo "Beta: $beta" >> experiments/q_learning/$path/hyperparam.txt
                echo "Gamma: $gamma" >> experiments/q_learning/$path/hyperparam.txt
                echo "Eta: $eta" >> experiments/q_learning/$path/hyperparam.txt
                python q_learning.py --epsilon $epsilon --beta $beta --gamma $gamma $eta eta
            done
        done
    done
done


