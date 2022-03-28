#!/bin/bash

epsilons=(0.10000 0.20000 0.90000)
betas=(0.00005 0.00050 0.00500)
gammas=(0.80000 0.85000 1.00000)
etas=(0.00350 0.00500 0.05000)

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
                echo $path
                echo "Epsilon: $epsilon" >> experiments/ex_replay/$path/hyperparam.txt
                echo "Beta: $beta" >> experiments/ex_replay/$path/hyperparam.txt
                echo "Gamma: $gamma" >> experiments/ex_replay/$path/hyperparam.txt
                echo "Eta: $eta" >> experiments/ex_replay/$path/hyperparam.txt
                
                python ex_replay.py --epsilon $(printf "%.5f" $epsilon) --beta $(printf "%.5f" $beta) --gamma $(printf "%.5f" $gamma) --eta $(printf "%.5f" $eta)
            done
        done
    done
done