import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import functions as fs
import getopt
from matplotlib import cm
from matplotlib.lines import Line2D

Color = {'15':  ['blue','aqua','dodgerblue'],          #q=0      -> dodgerblue
         '16': ['red','orange','orangered'],             #3x3      -> orangered
         '17':  ['pink','pink','gray'],           #cb2      -> magenta
         '18':  ['k','k','gray'],              #oct     -> orange
         '19':  ['gray','silver','magenta'],                #-> silver
         '20':  ['forestgreen','lime','limegreen'],    #cb1  -> forestgreen
         'labels':  ['k','k','k']
         }
marker = {  '15': ['s','D'],
            '16': ['s','D'],
            '17': ['s','D'],
            '18': ['s','D'],
            '19': ['P','X'],
            '20': ['P','X']
            }
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:K:", ['DM='])
    txt_S = '50'
    phi_t = '000'
    K = '13'
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
#
phi_label = {'000':0, '005':0.05, '104':np.pi/3, '209':np.pi/3*2}
phi = phi_label[phi_t]
dirname = '../../Data/self_consistency/S'+txt_S+'/phi'+phi_t+'/final/' 
#dirname = '../../Data/self_consistency/440_small_S50/phi'+phi_t+'/13/' 
title = "Phi = "+phi_t+", S = 0."+txt_S
#
Grid = 31#9
D = np.ndarray((Grid,Grid),dtype='object')
DD_none = D[0,0]
Ji = -0.3
Jf = -Ji
J2 = np.linspace(Ji,Jf,Grid)
J3 = np.linspace(Ji,Jf,Grid)
for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0:
        continue
    head = lines[0].split(',')
    head[-1] = head[-1][:-1]
    data = lines[1].split(',')
    j2 = float(data[head.index('J2')])
    j3 = float(data[head.index('J3')])
    i2 = list(J2).index(j2) 
    i3 = list(J3).index(j3) 
    minE = 10
    N = (len(lines)-1)//2 + 1
    i_ = 0
    for i in range(N):
        head = lines[2*i].split(',')
        head[-1] = head[-1][:-1]
        data = lines[2*i+1].split(',')
        tempE = float(data[head.index('Energy')])
        if tempE < minE:
            minE = tempE
            i_ = i
    head = lines[2*i_].split(',')
    head[-1] = head[-1][:-1]
    data = lines[2*i_+1].split(',')
    sol = data[0] + data[1]
    D[i2,i3] = sol
##########
pts = len(os.listdir(dirname))
fig = plt.figure(figsize=(10,10))
#plt.subplot(2,2,1)
plt.title(title)
plt.gca().set_aspect('equal')
plt.axhline(y=0,color='k',zorder=-1)
plt.axvline(x=0,color='k',zorder=-1)
for i in range(Grid):
    for j in range(Grid):
        if D[i,j] == DD_none:
            #if i == 7 and j == 30:
            #    c = 'forestgreen';  m = 'P'
            #else:
            #    c = 'b';            m = 's'
            #plt.scatter(J2[i],J3[j],color=c,marker=m,s=100)
            continue
        ans = D[i,j][:2]
        order = 0 if D[i,j][2:] == 'LRO' else 1
        c = Color[ans][order]
        m = marker[ans][order]
        plt.scatter(J2[i],J3[j],color=c,marker=m,s=100)

#plt.legend

list_leg = [r'$\mathbf{Q}=0$',
            r'$15$',
            r'$\sqrt{3}\times\sqrt{3}$',
            r'$16$',
            r'$19$',
            r'cuboc-$1$',
            r'$20$',
            ]
legend_lines = [Line2D([], [], color="w", marker=marker['15'][0], markerfacecolor=Color['15'][0]),
                Line2D([], [], color="w", marker=marker['15'][1], markerfacecolor=Color['15'][1]),
                Line2D([], [], color="w", marker=marker['16'][0], markerfacecolor=Color['16'][0]),
                Line2D([], [], color="w", marker=marker['16'][1], markerfacecolor=Color['16'][1]),
                Line2D([], [], color="w", marker=marker['19'][1], markerfacecolor=Color['19'][1]),
                Line2D([], [], color="w", marker=marker['20'][0], markerfacecolor=Color['20'][0]),
                Line2D([], [], color="w", marker=marker['20'][1], markerfacecolor=Color['20'][1]),
                ]

plt.legend(legend_lines,list_leg,loc='upper left',fancybox=True)

plt.show()





