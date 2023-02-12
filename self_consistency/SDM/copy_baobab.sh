#!/bin/bash
for K in 13 25
do
    scp -r rossid@login2.baobab.hpc.unige.ch:~/0_SELF-CONSISTENCY_SDM/Data/${K}/ ~/Desktop/git/SpinLiquids/Data/self_consistency/SDM/
done
