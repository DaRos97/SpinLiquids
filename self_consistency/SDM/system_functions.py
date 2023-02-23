import numpy as np
from pathlib import Path
import inputs as inp
import csv
import os
###############################################################
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
