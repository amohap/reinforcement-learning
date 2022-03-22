#!/bin/bash

epsilons=(1 2 4 8 16 32 64 128)
betas=(-1 -2 -4 -8 -16 -32 -64 -128)
gammas=(1 2 4 8 16 32 64 128)
etas=(1 2 4 8 16 32 64 128)

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