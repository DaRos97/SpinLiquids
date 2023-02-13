import numpy as np
from pathlib import Path
import inputs as inp
import csv
import os
###############################################################
orders = [('15','16','19'),('17','18','20')]
def import_solutions(filename,p1):
    solutions = []
    my_file = Path(filename)
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 +1        #2 lines per ansatz
        for i in range(N):
            data = lines[i*2+1].split(',')
            if int(data[4]) == p1:
                r = []
                for p in range(5,13):
                    r.append(float(data[p]))
                solutions.append(r)
    #CHeck if they are all already computed
    completed = check_solutions(solutions,p1)

    return solutions, completed

def check_solutions(solutions,p1):
    res = []
    for sol in solutions:
        phiA1p = float(sol[3])
        if np.abs(phiA1p) < inp.cutoff_solution or np.abs(phiA1p-2*np.pi) < inp.cutoff_solution:
            ord_ = 0
        elif np.abs(phiA1p-np.pi) < inp.cutoff_solution:
            ord_ = 1
        else:
            ord_ = 2
        res.append(orders[p1][ord_])
    completed = True
    for r_ in orders[p1]:
        if r_ not in res:
            completed = False

    return completed



#Save the dictionaries in the file given, rewriting the already existing data if precision is better
def SaveToCsv(Data,csvfile):
    header = Data.keys()
    with open(csvfile,'a') as f:
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writeheader()
        writer.writerow(Data)
