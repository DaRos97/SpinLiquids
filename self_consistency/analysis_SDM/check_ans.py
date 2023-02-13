import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import getopt
from matplotlib import cm
from matplotlib.lines import Line2D
import functions as fs

Color = {'15':  ['blue','k'],
         '16': ['red','orange'],
         '17':  ['pink','grey'],
         '18':  ['orange','purple'],
         '19':  ['aqua','aqua'],
         '20':  ['limegreen','forestgreen'],
         'labels':  ['k','k']
         }
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "K:a:")
    K = 13
    a = '15'
except:
    print("Error in inputs")
    exit()
for opt, arg in opts:
    if opt in ['-K']:
        K = int(arg)
    if opt in ['-a']:
        a = arg

dirname = '../../Data/self_consistency/SDM/'+str(K)+'/' 

#
S_max = 0.5
DM_max = 0.15
S_pts = 30
DM_pts = 30
S_list = np.linspace(0.01,S_max,S_pts,endpoint=True)
DM_list = np.linspace(0,DM_max,DM_pts,endpoint=True)
X,Y = np.meshgrid(DM_list,S_list)
D = np.ndarray((DM_pts,S_pts),dtype='object')
DD_none = D[0,0]
for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    data = lines[1].split(',')
    dm = list(DM_list).index(float(data[1]))
    s = list(S_list).index(float(data[0]))
    D[dm,s] = list()
    N = (len(lines)-1)//2 + 1
    for i in range(N):
        data = lines[i*2+1].split(',')
        D[dm,s].append(fs.find_ansatz(data))
    D[dm,s] = list(D[dm,s])
##########
pts = len(os.listdir(dirname))
fig = plt.figure(figsize=(16,8))
plt.title("DM diagram")
#plt.gca().set_aspect('equal')
nn = dd = 0
sizem = 30
for i in range(DM_pts):
    for j in range(S_pts):
        done = False
        if D[i,j] == DD_none:
            c = 'gray'
            m = 'P'
            plt.scatter(DM_list[i],S_list[j],color=c,marker=m,s=sizem)
            dd += 1
            continue
        for ans in D[i,j]:
            if a == ans[:2]:
                c = 'green'
                m = 'o'
                if ans[-1] == 'C':
                    c = 'magenta'
                    m = 'P'
                if ans[-2] == 'a':
                    c = 'k'
                    m = '^'
                plt.scatter(DM_list[i],S_list[j],color=c,marker=m,s=sizem)
                done = True
        if not done:
            c = 'red'
            m = '*'
            nn += 1
            plt.scatter(DM_list[i],S_list[j],color=c,marker=m,s=sizem)
#plt.ylim(0.00,0.501)
#plt.xlim(-0.001,0.1)
#Legenda
list_leg = []
list_leg.append(a)
list_leg.append('non-converged:'+str(nn))
list_leg.append('non-computed:'+str(dd))
legend_lines = []
legend_lines.append(Line2D([], [], color="w", marker='o', markerfacecolor='green', markersize=10))
legend_lines.append(Line2D([], [], color="w", marker='*', markerfacecolor='red', markersize=15))
legend_lines.append(Line2D([], [], color="w", marker='P', markerfacecolor='gray', markersize=10))

plt.legend(legend_lines,list_leg,loc='upper right',bbox_to_anchor=(1,1),fancybox=True)
#
plt.show()
