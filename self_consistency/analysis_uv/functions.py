import numpy as np
from pathlib import Path
ans_1 = ('15','16','17','18')
ans_2 = ('19','20')

CO_gap = 0.8                                #0.015
CO_phase = 1e-3
CO_mod = 1e-2
CO_amp = 1e-3

def find_ansatz(head,data):
    ans = data[0]
    #
    P = []
    for temp_p in ['p2','p3','p4','p5']:
        if temp_p in head:
            P.append(int(data[head.index(temp_p)]))
    #
    if ans in ans_1:
        relevant_phase = 'phiB1'
    elif ans in ans_2:
        relevant_phase = 'phiA1p'
    else:
        print('WTF is this?')
        exit()
    phase = float(data[head.index(relevant_phase)])
    if np.abs(phase) < CO_phase or np.abs(phase-2*np.pi) < CO_phase:
        phase_ = 'O'
    elif np.abs(phase-np.pi) < CO_phase:
        phase_ = 'P'
    else:
        phase_ = 'Z'
    #
    gap = float(data[head.index('Gap')])
    if gap < CO_gap:
        gap_ = 'L'
    else:
        gap_ = 'S'
    #

    result = ans + phase_ + gap_ + str(len(P))
    for p in P:
        result += str(p)
    return result
#
def min_energy(lines,considered_ans):
    N = (len(lines)-1)//2 + 1
    minE = 10
    index = 0
    ansatze = []
    ind_ = 0
    for i in range(N):
        bad = False
        head_data = lines[2*i].split(',')
        head_data[-1] = head_data[-1][:-1]
        data = lines[2*i+1].split(',')
        if data[0] not in considered_ans:
            continue
        for i in range(head_data.index('A1'),len(data)):
            if float(data[i]) < -1e-3:
                bad = True
        if bad:
            continue
        if data[0] in ['19','20'] and (np.abs(float(data[head_data.index('phiA1p')])) < 1e-3 or np.abs(float(data[head_data.index('phiA1p')])-np.pi) < 1e-3):
                #and (find_15(head_data,data) or find_16(head_data,data)):
            continue
        energy = float(data[head_data.index('Energy')])
        ansatze.append(find_ansatz(head_data,data))
        if energy < minE:
            minE = energy
            ind_ = index
        index += 1 
    if len (ansatze) == 0:
        return 0
    return ansatze[ind_]
#
def find_15(head,data):     #establish if 19 is same result as 15
    result = True
    J2 = float(data[head.index('J2')])
    J3 = float(data[head.index('J3')])
    phiA1p = float(data[head.index('phiA1p')])
    phiB1 = float(data[head.index('phiB1')])
    if np.abs(phiA1p) > CO_phase or np.abs(phiB1-np.pi) > CO_phase:
        result = False
    if J2:
        p2 = int(data[head.index('p2')])
        p3 = int(data[head.index('p3')])
        phiB2 = float(data[head.index('phiB2')])
        phiB2p = float(data[head.index('phiB2p')])
        if p2 != 1 or p3 != 1 or np.abs(phiB2-np.pi)>CO_phase or np.abs(phiB2p-np.pi)>CO_phase:
            result = False
    if J3:
        A3 = float(data[head.index('A3')])
        p5 = int(data[head.index('p5')])
        if np.abs(A3) > CO_amp or p5 != 0:
            result = False
    return result
#
def find_16(head,data):     #establish if 19 is same result as 16
    result = True
    J2 = float(data[head.index('J2')])
    J3 = float(data[head.index('J3')])
    phiA1p = float(data[head.index('phiA1p')])
    phiB1 = float(data[head.index('phiB1')])
    if np.abs(phiA1p-np.pi) > CO_phase or np.abs(phiB1-np.pi) > CO_phase:
        result = False
    if J2:
        A2 = float(data[head.index('A2')])
        A2p = float(data[head.index('A2p')])
        phiB2 = float(data[head.index('phiB2')])
        phiB2p = float(data[head.index('phiB2p')])
        if not (np.abs(phiB2)<CO_phase or np.abs(phiB2-2*np.pi)<CO_phase) or not (np.abs(phiB2p)<CO_phase or np.abs(phiB2p-2*np.pi)<CO_phase) or np.abs(A2) > CO_amp or np.abs(A2p) > CO_amp:
            result = False
    if J3:
        p4 = int(data[head.index('p4')])
        p5 = int(data[head.index('p5')])
        if p4 != 1 or p5 != 1:
            result = False
    return True
#
def find_gap(args,K):
    ans,txt_DM,J2,J3,txt_S = args
    DirName = '../../Data/self_consistency/S'+txt_S+'/phi'+txt_DM+"/"
    DataDir = DirName + str(K) + '/'
    csvname = 'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
    csvfile = DataDir + csvname
    my_file = Path(csvfile)
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 +1        #2 lines per ansatz
        for i in range(N):
            head = lines[i*2].split(',')
            head[-1] = head[-1][:-1]
            data = lines[i*2+1].split(',')
            valid = True
            for j in range(head.index('A1'),len(data)):
                if float(data[j]) < -1e-3:
                    valid = False
            if not valid:
                continue
            if data[0] != ans:
                continue
            if ans in ans_1:
                return float(data[head.index('Gap')])
            phiA1p = float(data[head.index('phiA1p')])
            if np.abs(phiA1p) < CO_phase or np.abs(phiA1p-2*np.pi) < CO_phase or np.abs(phiA1p-np.pi) < CO_phase:
                continue
            return float(data[head.index('Gap')])
#
def quadratic(x,a,b):
    return a/(3*x**2) + b
def ql(x,a,b,c):
    return a/(3*x**2) + c/x + b
def linear(x,a,b):
    return a/x + b


















