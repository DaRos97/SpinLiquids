#!/bin/bash
for S in 50 36 34 30 20
do
    for DM in 000 104 209
    do
        for K in 13
        do 
            scp -r rossid@login1.yggdrasil.hpc.unige.ch:~/SC_data/S${S}/phi${DM}/${K}/ ~/Desktop/git/SpinLiquids/Data/SC_data/S${S}/phi${DM}/
        done
    done
done
