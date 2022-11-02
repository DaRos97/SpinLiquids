import numpy as np
import matplotlib.pyplot as plt

phi_t = ['000','006','013','019','026','032','039']#,'052']
phi_r = [0,np.pi/48,np.pi/24,np.pi/16,np.pi/12,np.pi*5/48,np.pi/8]#,np.pi/6]
list_ans = ['3x3_1','q0_1','cb1']
en = {}
for ans in list_ans:
    en[ans] = []
for phi in phi_t:
    filename = '../Data/S05/phi'+phi+'/13/J2_J3=(00000_00000).csv'
    with open(filename, 'r') as f:
        lines = f.readlines()
    N = (len(lines)-1)//2 + 1
    done = {}
    for ans in list_ans:
        done[ans] = 0
    for n in range(N):
        line = lines[2*n+1].split(',')
        if line[0] in list_ans:
            if line[3] == 'True':
                en[line[0]].append(float(line[4]))
                done[line[0]] = 1
    for ans in list_ans:
        if done[ans] == 0:
            en[ans].append(np.nan)

plt.figure(figsize=(8,8))
cols = {'3x3_1':'b',
        'q0_1':'r',
        'cb1':'m'}
for ans in list_ans:
    plt.plot(phi_r,en[ans],color=cols[ans],marker='*')
plt.show()

