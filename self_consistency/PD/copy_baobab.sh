#!/bin/bash

for S in 50 
do
    for DM in 000
    do
        for K in $1
        do
            scp -r rossid@login2.baobab.hpc.unige.ch:~/0_SELF-CONSISTENCY_PD/Data/S${S}/phi${DM}/${K}/ ~/Desktop/git/SpinLiquids/Data/self_consistency/S${S}/phi${DM}/
        done
    done
done
