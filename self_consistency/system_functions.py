import numpy as np
from pathlib import Path
import inputs as inp
import csv
import os

#################################################################
#checks if the file exists and if it does, reads which ansatze have been computed and returns the remaining ones
#from the list of ansatze in inputs.py
def CheckCsv(csvf):
    my_file = Path(csvf)
    ans = []
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 +1        #2 lines per ansatz
        for i in range(N):
            data = lines[i*2+1].split(',')
            if data[3] == 'True':
                ans.append(lines[i*2+1].split(',')[0])
    res = []
    for a in inp.list_ans:
        if a not in ans:
            res.append(a)
    return res

#Extracts the initial point for the minimization from a file in a reference directory specified in inputs.py
#If the file matching the j2,j3 point is not found initialize the initial point with default parameters defined in inputs.py
def FindInitialPoint(J2,J3,ansatze,ReferenceDir,Pi_):
    P = {}  #parameters
    done = {}
    L = {}
    if Path(ReferenceDir).is_dir():
        for file_ in os.listdir(ReferenceDir):     #find file in dir
            j2 = float(file_[7:-5].split('_')[0])/10000  #specific for the name of the file
            j3 = float(file_[7:-5].split('_')[1])/10000
            if np.abs(j2-J2) < inp.cutoff_pts and np.abs(j3 - J3) < inp.cutoff_pts:         #once found read it
                with open(ReferenceDir+file_, 'r') as f:
                    lines = f.readlines()
                N = (len(lines)-1)//2 + 1
                for Ans in ansatze:
                    for i in range(N):
                        data = lines[i*2+1].split(',')
                        if data[0] == Ans:              #correct ansatz
                            L[data[0]] = float(data[6])
                            P[data[0]] = data[7:-1] if data[-1] == '\n' else data[7:]
                            for j in range(len(P[data[0]])):    #cast to float
                                P[data[0]][j] = float(P[data[0]][j])
    j2 = np.abs(J2) > inp.cutoff_pts    #bool for j2 not 0
    j3 = np.abs(J3) > inp.cutoff_pts
    #remove eventual 0 values
    nP = {}
    for ans in P.keys():
        nP[ans] = []
        for i in np.nonzero(P[ans])[0]:
            nP[ans].append(P[ans][i])
    P = nP
    #check eventual missing ansatze from the reference fileand initialize with default values
    for ans in ansatze:
        if ans in list(P.keys()):
            done[ans] = 1
            continue
        P[ans] = []
        L[ans] = 0
        for par in Pi_[ans].keys():
            if par[-1] == '1':
                P[ans].append(Pi_[ans][par])
            elif par[-1] == '2' and j2:
                P[ans].append(Pi_[ans][par])
            elif par[-1] == '3' and j3:
                P[ans].append(Pi_[ans][par])
        done[ans] = 0
    return P, done, L

#Save the dictionaries in the file given, rewriting the already existing data if precision is better
def SaveToCsv(Data,csvfile,PSG):
    N_ = 0
    if Path(csvfile).is_file():
        with open(csvfile,'r') as f:
            init = f.readlines()
    else:
        init = []
    ans = Data['ans']       #computed ansatz
    N = (len(init)-1)//2+1
    subscribe = False
    for i in range(N):
        D = init[i*2+1].split(',')
        if D[0] == ans:
            subscribe = True
            N_ = i+1
    ###
    header = inp.header[PSG][ans]
    if subscribe:
        with open(csvfile,'w') as f:
            for i in range(2*N_-2):
                f.write(init[i])
        with open(csvfile,'a') as f:
            writer = csv.DictWriter(f, fieldnames = header)
            writer.writeheader()
            writer.writerow(Data)
        with open(csvfile,'a') as f:
            for l in range(2*N_,len(init)):
                f.write(init[l])
    else:
        with open(csvfile,'a') as f:
            writer = csv.DictWriter(f, fieldnames = header)
            writer.writeheader()
            writer.writerow(Data)
