#!/bin/bash

epsilons=(0.1 0.2 0.9)
betas=(0.00005 0.0005 0.005)
gammas=(0.8 0.85 1)
etas=(0.0035 0.005 0.05)

if [ ! -d experiments/ex_replay ]; then
  mkdir -p experiments/ex_replay;
fi

for epsilon in ${epsilons[@]}; do
    for beta in ${betas[@]}; do
        for gamma in ${gammas[@]}; do
            for eta in ${etas[@]}; do
                path="ep${epsilon}_be${beta}_ga${gamma}_et${eta}"
                if [ ! -d experiments/ex_replay/$path ]; then
                    mkdir -p experiments/ex_replay/$path;
                fi
                echo "Epsilon: $epsilon" >> experiments/ex_replay/$path/hyperparam.txt
                echo "Beta: $beta" >> experiments/ex_replay/$path/hyperparam.txt
                echo "Gamma: $gamma" >> experiments/ex_replay/$path/hyperparam.txt
                echo "Eta: $eta" >> experiments/ex_replay/$path/hyperparam.txt
                python ex_replay.py --epsilon $epsilon --beta $beta --gamma $gamma $eta eta
            done
        done
    done
done