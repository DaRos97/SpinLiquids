import numpy as np
import os

n = 0
dirname = '../Data/SDM/13/'
#dirname = '/home/users/r/rossid/SDM_Data/13/'
for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    N_ = (len(lines)-1)//2 + 1
    delete = []
    for i in range(N_):
        data = lines[i*2+1].split(',')
        if data[0] == '1d' and float(data[9]) < 0.001:
            print(filename,data[0])
            n += 1
print(n)
exit()
for i in range(0):
    if 0:
        if 0:
            delete.append(i+1)
            break
    if len(delete) > 0:
        with open(dirname+filename,'w') as f:
            for i in range(2*delete[0]-2):
                f.write(lines[i])
        with open(dirname+filename,'a') as f:
            for l in range(2*delete[0],len(lines)):
                f.write(lines[l])
        
    
