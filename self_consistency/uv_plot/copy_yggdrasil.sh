#!/bin/bash

for S in 50 #36 34 #30 20
do
    for K in 13 #25 37 49
    do
        scp -r rossid@login1.yggdrasil.hpc.unige.ch:~/0_SELF-CONSISTENCY_UV/Data/S$S/$K/ ~/Desktop/git/SpinLiquids/Data/self_consistency/UV/S$S/
    done
done
