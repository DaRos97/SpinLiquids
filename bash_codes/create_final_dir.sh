#!/bin/bash

for S in 50 36 34 30 20
do
    for DM in 000 104 209
    do
        cp -r ~/Desktop/git/SpinLiquids/Data/S${S}/phi${DM}/49/ ~/Desktop/git/SpinLiquids/Data/Final_${S}_${DM}/
    done
done
