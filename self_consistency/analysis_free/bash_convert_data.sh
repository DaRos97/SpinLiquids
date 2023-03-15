#!/bin/bash

for S in 50 36 30 20
do
    for DM in 000 005
    do
        python finite_size.py -S $S --DM $DM
    done
done
