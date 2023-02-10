import numpy as np

orders = ('15','16','19','17','18','20')
CO_gap = 1e-2
CO_phase = 1e-3

def find_ansatz(data):
    phiA1p = float(data[8])
    gap = float(data[3])
#    S,DM,E,gap,p1,L,A1,A1p,phiA1p,B1,phiB1,B1p,phiB1p = float(data)
    p1 = int(data[4])
    phase = 'L' if gap < CO_gap else 'O'

    if np.abs(phiA1p) < CO_phase or np.abs(phiA1p-2*np.pi) < CO_phase:
        ord_ = 0
    elif np.abs(phiA1p-np.pi) < CO_phase:
        ord_ = 1
    else:# not (np.abs(phiA1p) < CO_phase or np.abs(phiA1p-2*np.pi) < CO_phase or np.abs(phiA1p-np.pi) < CO_phase):
        ord_ = 2

    result = orders[p1*3+ord_] + phase

    return result
