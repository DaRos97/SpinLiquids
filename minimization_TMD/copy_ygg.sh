#!/bin/bash

for S in 50 36 34 30 20
do
    for K in 13 25 37 49
    do
        scp -r rossid@login1.yggdrasil.hpc.unige.ch:~/Data/S${S}/phi005/${K}/ ~/Desktop/git/SpinLiquids/Data/S${S}/phi005/
    done
done 
