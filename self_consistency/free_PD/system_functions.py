import numpy as np
from pathlib import Path
import inputs as inp
import csv
import os
###############################################################
def find_p(ans,J2,J3):
    if ans in inp.ansatze_1:
        if J2:
            return zip([0,0,1,1],[0,1,0,1])
        else:
            return [(2,2)]
    elif ans in inp.ansatze_2:
        l4 = []
        l2 = []
        for p2 in [0,1]:
            for p3 in [0,1]:
                l2.append((p2,p3))
                for p4 in [0,1]:
                    for p5 in [0,1]:
                        l4.append((p2,p3,p4,p5))
        if (J2==0 and J3) or (J3==0 and J2):
            return l2
        elif J2 and J3:
            return l4
        else:
            return [(2,2)]
#
def find_head(ans,J2,J3):
    head_p = ['ans']
    if ans in inp.ansatze_1:
        if J2:
            head_p += ['p2','p3']
        pars = ['A1','B1','phiB1']
    elif ans in inp.ansatze_2:
        if J2:
            head_p += ['p2','p3']
        if J3:
            head_p += ['p4','p5']
        pars = ['A1','phiA1p','B1','phiB1']
    header = head_p + inp.header
    if J2:
        if ans in ['15','17']:
            pars += ['A2','phiA2','A2p','phiA2p','B2','B2p']
        elif ans in ['16','18']:
            pars += ['B2','B2p']
        if ans in ['19','20']:
            pars += ['A2','A2p','B2','phiB2','B2p','phiB2p']
    if J3:
        if ans == '15':
            pars += ['B3','phiB3']
        elif ans == '16':
            pars += ['A3','phiA3','B3','phiB3']
        elif ans == '17':
            pars += ['A3','phiA3']
        if ans in ['19','20']:
            pars += ['A3','B3']
    return header+pars, pars
#
def import_solutions(filename,ans,p,J2,J3):
    solutions = []
    ps = []
    if ans in inp.ansatze_1:
        if J2:
            bias = 2
        else:
            bias = 0
    elif ans in inp.ansatze_2:
        if (J2==0 and J3) or (J3==0 and J2):
            bias = 2
        elif J2 and J3:
            bias = 4
        else:
            bias = 0
    my_file = Path(filename)
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 +1        #2 lines per ansatz
        for i in range(N):
            data = lines[i*2+1].split(',')
            if data[0] == ans:
                temp_p = []
                for pps in range(1,bias+1):
                    temp_p.append(int(pps))
                if bias == 0:
                    temp_p = (2,2)
                right = True
                for bb in range(bias):
                    if temp_p[bb] != p[bb]:
                        right = False
                if right:
                    r = []
                    for par in range(5+bias,len(data)):
                        r.append(float(data[par]))
                    solutions.append(r)
    return solutions
#
def check_solutions(solutions,index,new_phase):
    for sol in solutions:
        phase = float(sol[index+1])
        if np.abs(phase-new_phase) < inp.cutoff_solution or np.abs(phase-new_phase-2*np.pi) < inp.cutoff_solution:
            return True
    return False
#
def amp_list(pars):
    res = []
    for i,p in enumerate(pars):
        if p[0] in ['A','B']:
            res.append(i)
    return res
#
def phase_list(pars):
    res = []
    for i,p in enumerate(pars):
        if p[0] == 'p':
            res.append(i)
    return res
#Save the dictionaries in the file given, rewriting the already existing data if precision is better
def SaveToCsv(Data,csvfile):
    header = Data.keys()
    with open(csvfile,'a') as f:
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writeheader()
        writer.writerow(Data)




