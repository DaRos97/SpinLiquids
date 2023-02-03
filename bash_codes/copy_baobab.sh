#!/bin/bash
for S in 50 36 34 30 20
do
    for DM in 000 104 209
    do
        for K in 37 49
        do 
            scp -r rossid@login2.baobab.hpc.unige.ch:~/SC_data/S${S}/phi${DM}/${K}/ ~/Desktop/git/SpinLiquids/Data/SC_data/S${S}/phi${DM}/
        done
    done
done
