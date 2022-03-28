#!/bin/bash

epsilons=(0.10000 0.20000 0.90000)
betas=(0.00005 0.00050 0.00500)
gammas=(0.80000 0.85000 1.00000)
etas=(0.00350 0.00500 0.05000)

for epsilon in ${epsilons[@]}; do
    for beta in ${betas[@]}; do
        for gamma in ${gammas[@]}; do
            for eta in ${etas[@]}; do
                path="ep${epsilon}_be${beta}_ga${gamma}_et${eta}"
                echo $path
                if [ ! -d experiments/q_learning/$path ]; then
                    mkdir -p experiments/q_learning/$path;
                fi
                echo "Epsilon: $epsilon" >> experiments/q_learning/$path/hyperparam.txt
                echo "Beta: $beta" >> experiments/q_learning/$path/hyperparam.txt
                echo "Gamma: $gamma" >> experiments/q_learning/$path/hyperparam.txt
                echo "Eta: $eta" >> experiments/q_learning/$path/hyperparam.txt
                
                python q_learning.py --epsilon $(printf "%.5f" $epsilon) --beta $(printf "%.5f" $beta) --gamma $(printf "%.5f" $gamma) --eta $(printf "%.5f" $eta)
            done
        done
    done
done


