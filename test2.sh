#!/bin/bash

# path="ep$epsilon_be$beta`_ga$gamma_et$eta"

# if [ ! -d experiments/sarsa/$path ]; then
#     mkdir -p experiments/sarsa/$path
# fi

epsilon=0.1
beta=0.2
gamma=0.3
eta=0.4

path="ep${epsilon}_be${beta}_ga${gamma}_et${eta}"

if [ ! -d experiments/sarsa_adagrad/$path ]; then
    mkdir -p experiments/sarsa_adagrad/$path;
fi
