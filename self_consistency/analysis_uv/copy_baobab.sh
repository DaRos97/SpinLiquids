#!/bin/bash

for S in 50 #36 30 20
do
    for K in 13
    do
        scp -r rossid@login2.baobab.hpc.unige.ch:~/0_SELF-CONSISTENCY_UV/Data/S${S}/${K}/ ~/Desktop/git/SpinLiquids/Data/self_consistency/UV/S${S}/
    done
done
