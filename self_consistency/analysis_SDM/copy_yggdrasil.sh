#!/bin/bash
DM_type=$2
for K in $1
do
    scp -r rossid@login1.yggdrasil.hpc.unige.ch:~/0_SELF-CONSISTENCY_SDM/Data/$DM_type/${K}/ ~/Desktop/git/SpinLiquids/Data/self_consistency/SDM/$DM_type/
done
