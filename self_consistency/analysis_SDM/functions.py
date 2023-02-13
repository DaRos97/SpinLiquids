import numpy as np

orders = ('15','16','19','17','18','20')
CO_gap = 0.015
CO_phase = 1e-3
CO_mod = 1e-2

def find_ansatz(data):
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
