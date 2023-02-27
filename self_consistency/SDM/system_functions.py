import numpy as np
from pathlib import Path
import inputs as inp
import csv
import os
###############################################################
def find_Pinitial(S,ans,csvfile,K,new_phase):
    if K == 13:
        Ai = S
        Bi = S/2
        if ans in inp.ansatze_1:
            Pinitial = [Ai,Bi,np.pi]
        else:
            Pinitial = [Ai,0,Bi,np.pi]
        return Pinitial
    phase_name = 'phiA1p' if ans in inp.ansatze_2 else 'phiB1'
    my_file = Path(csvfile)
    Pinitial = []
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 +1        #2 lines per ansatz
        for i in range(N):
            head = lines[i*2].split(',')
            head[-1] = head[-1][:-1]
            data = lines[i*2+1].split(',')
            if data[0] == ans and np.abs(float(data[head.index(phase_name)])-new_phase)<inp.cutoff_F:
                for j in range(head.index('A1'),len(data)):
                    Pinitial.append(float(data[j]))
                return Pinitial
    print("error in fs.find_Pinitial")
    exit()
#
def find_list_phases(numb_it,csvfile,K,ans):
    if K == 13:
        result = []
        for iph in range(numb_it+1):
            result.append(np.pi - iph*np.pi/numb_it)
        return result
    result = []
    phase_name = 'phiA1p' if ans in inp.ansatze_2 else 'phiB1'
    my_file = Path(csvfile)
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 +1        #2 lines per ansatz
        for i in range(N):
            head = lines[i*2].split(',')
            head[-1] = head[-1][:-1]
            data = lines[i*2+1].split(',')
            if data[0] == ans:
               result.append(float(data[head.index(phase_name)]))
    return result
#
def import_solutions(filename,ans):
    solutions = []
    my_file = Path(filename)
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 +1        #2 lines per ansatz
        for i in range(N):
            data = lines[i*2+1].split(',')
            if data[0] == ans:
                r = []
                for p in range(5,len(data)):
                    r.append(float(data[p]))
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
