#!/bin/bash

for S in 50 #20 
do
    for DM in 000 #005 104 209
    do
        for K in 13 #25
        do
            scp -r rossid@login2.baobab.hpc.unige.ch:~/0_SELF-CONSISTENCY_PD/Data/S${S}/phi${DM}/${K}/ ~/Desktop/git/SpinLiquids/Data/self_consistency/S${S}/phi${DM}/
        done
    done
done
