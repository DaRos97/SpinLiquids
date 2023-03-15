#!/bin/bash

for S in 50 #36 34 #30 20
do
    for DM in 000 #005 #104 209
    do
        for K in 13 #25 37 49
        do
            scp -r rossid@login1.yggdrasil.hpc.unige.ch:~/0_SELF-CONSISTENCY_PD/Data/440_small_S50/ ~/Desktop/git/SpinLiquids/Data/self_consistency/
#            scp -r rossid@login1.yggdrasil.hpc.unige.ch:~/0_SELF-CONSISTENCY_PD/Data/S${S}/phi${DM}/${K}/ ~/Desktop/git/SpinLiquids/Data/self_consistency/S${S}/phi${DM}/
        done
    done
done
