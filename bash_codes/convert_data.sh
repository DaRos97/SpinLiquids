#!/bin/bash

for S in 50 36 34 30 20
do
    for DM in 000 104 209
    do
        python ~/Desktop/git/SpinLiquids/Analysis_of_Data/gap_scaling/convert_data.py -S ${S} --DM ${DM}
    done
done
