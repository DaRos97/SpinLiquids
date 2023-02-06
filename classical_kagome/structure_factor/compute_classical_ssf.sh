#!/bin/bash

for o in '3x3' 'q0' 'cb1' 'cb2' 'oct'
do
    for g in 0 1 2
    do
        for th in 0 #0.16666666 0.25 0.3333333333 0.5 0.6666666666 0.75 0.8333333333 
        do
            for ph in 0 #0.16666666 0.25 0.3333333333 0.5 0.6666666666 0.75 0.8333333333 
            do
                python new_ssf.py -o $o -g $g --theta $th --phi $ph --UC 12 --save
            done
        done
    done
done

