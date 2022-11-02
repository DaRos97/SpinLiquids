#!/bin/bash

# Declare a string array with type
declare -a THETA=("0" "0.16666666" "0.25" "0.33333333" "0.5" "0.666666" "0.75" "0.8333333333" )
declare -a PHI=("0" "0.16666666" "0.25" "0.33333333" "0.5" "0.666666" "0.75" "0.8333333333" )
declare -a ANS=("q0")
declare -a GAUGE=("1")

# Read the array values with space
for ans in "${ANS[@]}"; do
    for gauge in "${GAUGE[@]}"; do
        for theta in "${THETA[@]}"; do
            for phi in "${PHI[@]}"; do
                python new_ssf.py -o $ans -g $gauge --theta $theta --phi $phi --UC 3 --compute_new --save
            done
        done
    done
done

