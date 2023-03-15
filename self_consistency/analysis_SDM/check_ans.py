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
    opts, args = getopt.getopt(argv, "K:a:",['staggered','par='])
    K = 13
    a = '15'
    DM_type = 'uniform'
    par_name = 'Energy'
except:
    print("Error in inputs")
    exit()
for opt, arg in opts:
    if opt in ['-K']:
        K = int(arg)
    if opt in ['-a']:
        a = arg
    if opt == '--staggered':
        DM_type = 'staggered'
    if opt == '--par':
        par_name = arg

dirname = '../../Data/self_consistency/SDM/'+DM_type+'/'+str(K)+'/' 

#
S_max = 0.5
DM_max = 0.15
S_pts = 30
DM_pts = 30
S_list = np.linspace(0.01,S_max,S_pts,endpoint=True)
DM_list = np.linspace(0,DM_max,DM_pts,endpoint=True)
X,Y = np.meshgrid(DM_list,S_list)
D = np.zeros((DM_pts,S_pts))
for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0:
        continue
    data = lines[1].split(',')
    dm = list(DM_list).index(float(data[2]))        #was 1
    s = list(S_list).index(float(data[1]))          #was 0
    N = (len(lines)-1)//2 + 1
    for i in range(N):
        head = lines[i*2].split(',')
        head[-1] = head[-1][:-1]
        data = lines[i*2+1].split(',')
        if data[0] == a:
           D[dm,s] = float(data[head.index(par_name)])
    if D[dm,s] == 0:
        D[dm,s] = np.nan


fig = plt.figure(figsize=(10,10))
#plt.subplot(2,2,1)
plt.gca().set_aspect('equal')
plt.axhline(y=0,color='k',zorder=-1)
plt.axvline(x=0,color='k',zorder=-1)
#
ax = fig.add_subplot(1,1,1,projection='3d')
X,Y = np.meshgrid(DM_list,S_list)
ax.plot_surface(X,Y,D.T,cmap=cm.coolwarm)
plt.show()













