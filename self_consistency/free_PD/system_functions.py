import numpy as np
from pathlib import Path
import inputs as inp
import csv
import os
###############################################################
def find_p(ans,J2,J3,csvfile,K):
    if ans in inp.ansatze_1:
        if J2:
            result = zip([0,0,1,1],[0,1,0,1])
        else:
            result = [(2,2)]
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
            result = l2
        elif J2 and J3:
            result = l4
        else:
            result = [(2,2)]
    if K == 13:
        return result
    #Check what is the correct p
    my_file = Path(csvfile)
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 +1        #2 lines per ansatz
        for i in range(N):
            head = lines[i*2].split(',')
            data = lines[i*2+1].split(',')
            if data[0] == ans:
                correct = True
                for j in range(head.index('L'),len(data)):
                    if float(data[j]) < -1e-3:
                        correct = False
                if not correct:
                    continue
                if ans in inp.ansatze_1:
                    if J2:
                        result = [(int(data[head.index('p2')]),int(data[head.index('p3')])),]
                    else:
                        result = [(2,2),]
                elif ans in inp.ansatze_2:
                    if J2 and not J3:
                        result = [(int(data[head.index('p2')]),int(data[head.index('p3')])),]
                    if J2 and J3:
                        result = [(int(data[head.index('p2')]),int(data[head.index('p3')]),int(data[head.index('p4')]),int(data[head.index('p5')])),]
                    if J3 and not J2:
                        result = [(int(data[head.index('p4')]),int(data[head.index('p5')])),]
                return result
    #if is not found, compute the all
    return [] 
#
def find_list_phases(numb_it,csvfile,K,ans):
    if K == 13:
        result = []
        for iph in range(numb_it+1):
            result.append(np.pi - iph*np.pi/numb_it)
        return result
    else:
        return (1,)
#
def find_Pinitial(new_phase,S,ans,pars,csvfile,K):
    if K == 13:
        Ai = S
        Bi = S/2
        index_mixing_ph = 1 if ans in inp.ansatze_2 else 2
        Pinitial  = []
        for i in range(len(pars)):
            if i == index_mixing_ph:
                Pinitial.append(new_phase)
                continue
            if pars[i][0] == 'p':
                if ans == '16' and pars[i] == 'phiA3':
                    Pinitial.append(0)
                else:
                    Pinitial.append(np.pi)
            elif pars[i][0] == 'A':
                Pinitial.append(Ai)
            elif pars[i][0] == 'B':
                Pinitial.append(Bi)
        return Pinitial
    phase_name = 'phiA1p' if ans in inp.ansatze_2 else 'phiB1'
    my_file = Path(csvfile)
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 +1        #2 lines per ansatz
        for i in range(N):
            head = lines[i*2].split(',')
            data = lines[i*2+1].split(',')
            if data[0] == ans:
                Pinitial = []
                correct = True
                for j in range(head.index('A1'),len(data)):
                    if float(data[j]) < -1e-3:
                        correct = False
                    Pinitial.append(float(data[j]))
                if not correct:
                    continue
                return Pinitial
    return find_Pinitial(new_phase,S,ans,pars,csvfile,13)


    
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
                    temp_p.append(int(data[pps]))
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
#
def find_ansatze(csvfile):
    ansatze = []
    my_file = Path(csvfile)
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 +1        #2 lines per ansatz
        for i in range(N):
            data = lines[i*2+1].split(',')
            if data[0] not in ansatze:
                ansatze.append(data[0])
    return ansatze
#





