#!/bin/bash

epsilons=(1 2 4 8 16 32 64 128)
betas=(-1 -2 -4 -8 -16 -32 -64 -128)
gammas=(1 2 4 8 16 32 64 128)
etas=(1 2 4 8 16 32 64 128)

for epsilon in ${epsilons[@]}; do
    # echo $epsilon
    for beta in ${betas[@]}; do
        # echo $beta
        for gamma in ${gammas[@]}; do
            # echo $gamma
            for eta in ${etas[@]}; do
                # echo $epsilon
                # echo $beta
                # echo $gamma
                # echo $eta
                python q_learning.py --epsilon $epsilon --beta $beta --gamma $gamma $eta eta
            done
        done
    done
done


