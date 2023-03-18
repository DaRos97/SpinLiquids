import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import functions as fs
import getopt
from matplotlib import cm
from matplotlib.lines import Line2D

Color = {'15':  ['blue','aqua','dodgerblue'],          #q=0      -> dodgerblue
         '16': ['red','y','orangered'],             #3x3      -> orangered
         '17':  ['pink','pink','gray'],           #cb2      -> magenta
         '18':  ['k','k','gray'],              #oct     -> orange
         '19':  ['gray','gray','magenta'],                #-> silver
         '20':  ['forestgreen','lime','limegreen'],    #cb1  -> forestgreen
         'labels':  ['k','k','k']
         }
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:K:a:", ['DM=','par='])
    txt_S = '50'
    phi_t = '000'
    K = '13'
    ans = '15'
    par_name = 'Energy'
except:
    print("Error in inputs")
    exit()
for opt, arg in opts:
    if opt in ['-S']:
        txt_S = arg
    if opt in ['-K']:
        K = arg
    if opt == '--DM':
        phi_t = arg
    if opt in '-a':
        ans = arg
    if opt == '--par':
        par_name = arg
#
phi_label = {'000':0, '005':0.05, '104':np.pi/3, '209':np.pi/3*2}
phi = phi_label[phi_t]
#dirname = '../Data/SC_data/final_'+txt_S+'_'+phi_t+'/' 
#dirname = '../../Data/self_consistency/S'+txt_S+'/phi'+phi_t+'/'+K+'/' 
dirname = '../../Data/self_consistency/440_small_S50/phi000/13/' 
title = "Phi = "+phi_t+", S = 0."+txt_S
#
Grid = 21#9
ph_grid = np.zeros((Grid,Grid))
Ji = -0.03
Jf = -Ji
J2 = np.linspace(Ji,Jf,Grid)
J3 = np.linspace(Ji,Jf,Grid)
X,Y = np.meshgrid(J2,J3)
ph_ans = 'phiA1p' if ans in ['19','20'] else 'phiB1'
for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    if len(lines) > 0:
        head_data = lines[0].split(',')
        data = lines[1].split(',')
        try:
            j2 = float(data[head_data.index('J2')])
            j3 = float(data[head_data.index('J3')])
            i2 = list(J2).index(j2) 
            i3 = list(J3).index(j3) 
        except:
            continue
    else:
        continue
    N = (len(lines)-1)//2 + 1
    for i in range(N):
        head = lines[2*i].split(',')
        head[-1] = head[-1][:-1]
        data = lines[2*i+1].split(',')
        if data[0] == ans:
#            ph_grid[i2,i3] = float(data[head.index(ph_ans)])
            try:
                ph_grid[i2,i3] = float(data[head.index(par_name)])
            except:
                ph_grid[i2,i3] = np.nan
            break
for i in range(Grid):
    for j in range(Grid):
        if ph_grid[i,j] == 0:
            ph_grid[i,j] = np.nan
##########
pts = len(os.listdir(dirname))
fig = plt.figure(figsize=(10,10))
#plt.subplot(2,2,1)
plt.title(title)
plt.gca().set_aspect('equal')
plt.axhline(y=0,color='k',zorder=-1)
plt.axvline(x=0,color='k',zorder=-1)
#
ax = fig.add_subplot(1,1,1,projection='3d')
X,Y = np.meshgrid(J2,J3)
ax.plot_surface(X,Y,ph_grid.T,cmap=cm.coolwarm)
plt.show()



















