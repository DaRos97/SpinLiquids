import numpy as np

orders = ('15','16','19','17','18','20')
CO_gap = 0.015
CO_phase = 1e-3
CO_mod = 1e-2

def find_ansatz_free(data):     #function for finding ansatz in free SC 
    A1 = float(data[6])
    A1p = float(data[7])
    phiA1p = float(data[8])
    B1 = float(data[9])
    B1p = float(data[11])
    phiB1 = float(data[10])
    phiB1p = float(data[12])
    gap = float(data[3])
    p1 = int(data[4])
    phase = 'L' if gap > CO_gap else 'O'

    if np.abs(phiA1p) < CO_phase or np.abs(phiA1p-2*np.pi) < CO_phase:
        ord_ = 0
    elif np.abs(phiA1p-np.pi) < CO_phase:
        ord_ = 1
    else:# not (np.abs(phiA1p) < CO_phase or np.abs(phiA1p-2*np.pi) < CO_phase or np.abs(phiA1p-np.pi) < CO_phase):
        ord_ = 2

    if np.abs(phiB1-np.pi) > CO_phase or np.abs(phiB1p-np.pi) > CO_phase:
        phase = 'C'
    #Check moduli
    if np.abs(A1-A1p) < CO_mod and np.abs(B1-B1p) < CO_mod:
        symm = 's'
    else:
        symm = 'a'

    result = orders[p1*3+ord_] + symm + phase
    
    return result
#
def find_ansatz(head,data):
    ans = data[0]
    gap = float(data[head.index('Gap')])
    phase = 'L' if gap > CO_gap else 'O'
    phiB1 = float(data[head.index('phiB1')])
    if np.abs(phiB1-np.pi) > CO_phase:
        phase = 'C'
    symm = 's'
    result = ans + symm + phase
    
    return result

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
        if data[0] in ['19','20'] and (find_15(head_data,data) or find_16(head_data,data) or np.abs(float(data[head_data.index('phiA1p')])-np.pi/2) < CO_phase):
            continue
        energy = float(data[head_data.index('Energy')])
        if energy == 0:
            continue
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
    phiA1p = float(data[head.index('phiA1p')])
    phiB1 = float(data[head.index('phiB1')])
    if np.abs(phiA1p) > CO_phase or np.abs(phiB1-np.pi) > CO_phase:
        result = False
    return result
def find_16(head,data):     #establish if 19 is same result as 15
    result = True
    phiA1p = float(data[head.index('phiA1p')])
    phiB1 = float(data[head.index('phiB1')])
    if np.abs(phiA1p-np.pi) > CO_phase or np.abs(phiB1-np.pi) > CO_phase:
        result = False
    return result
























