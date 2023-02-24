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
         '17':  ['k','k','gray'],           #cb2      -> magenta
         '18':  ['k','k','gray'],              #oct     -> orange
         '19':  ['pink','pink','magenta'],                #-> silver
         '20':  ['forestgreen','lime','limegreen'],    #cb1  -> forestgreen
         'labels':  ['k','k','k']
         }
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:K:", ['DM=','only='])
    txt_S = '50'
    phi_t = '000'
    K = '13'
    do_only = False
    only = '15'
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
    if opt == '--only':
        only = arg
        do_only = True
#
if do_only:
    considered_ans = (only,)
else:
    considered_ans = Color.keys()
#
phi_label = {'000':0, '005':0.05, '104':np.pi/3, '209':np.pi/3*2}
phi = phi_label[phi_t]
#dirname = '../Data/SC_data/final_'+txt_S+'_'+phi_t+'/' 
dirname = '../../Data/self_consistency/S'+txt_S+'/phi'+phi_t+'/'+K+'/' 
title = "Phi = "+phi_t+", S = 0."+txt_S
#
D = np.ndarray((9,9),dtype='object')
DD_none = D[0,0]
Ji = -0.3
Jf = 0.3
J2 = np.linspace(Ji,Jf,9)
J3 = np.linspace(Ji,Jf,9)
X,Y = np.meshgrid(J2,J3)
for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    if len(lines) > 0:
        head_data = lines[0].split(',')
        data = lines[1].split(',')
        j2 = float(data[head_data.index('J2')])
        j3 = float(data[head_data.index('J3')])
        i2 = list(J2).index(j2) 
        i3 = list(J3).index(j3) 
    else:
        continue
    ansatz = fs.min_energy(lines,considered_ans)
    if ansatz == 0:
        continue
    D[i2,i3] = ansatz
##########
pts = len(os.listdir(dirname))
fig = plt.figure(figsize=(10,10))
#plt.subplot(2,2,1)
plt.title(title)
plt.gca().set_aspect('equal')
plt.axhline(y=0,color='k',zorder=-1)
plt.axvline(x=0,color='k',zorder=-1)
for i in range(9):
    for j in range(9):
        if D[i,j] == DD_none:
            continue
        ans = D[i,j][:2]
        LS = 0 if D[i,j][3] == 'L' else 2
        if ans == '15' and D[i,j][4] == '2':
            if D[i,j][5] != '1' or D[i,j][6] != '1':
                LS = 1
#                print(J2[i],J3[j],D[i,j])
        if ans == '16' and D[i,j][4] == '2':
            if D[i,j][5] != '0' or D[i,j][6] != '0':
                LS = 1
#                print(J2[i],J3[j],D[i,j])
        if ans == '20':
            if D[i,j][4] == '2' and J2[i]:      #only p2 and p3
                if D[i,j][5] != '1' or D[i,j][6] != '1':
                    LS = 1
#                    print(J2[i],J3[j],D[i,j])
            if D[i,j][4] == '2' and J3[j]:      #only p4 and p5
                if D[i,j][5] != '0':
                    LS = 1
#                    print(J2[i],J3[j],D[i,j])
            if D[i,j][4] == '4':
                if D[i,j][5] != '1' or D[i,j][6] != '1' or D[i,j][7] != '0':
                    LS = 1
#                    print(J2[i],J3[j],D[i,j])
        c = Color[ans][LS]
        #
        if D[i,j][2] == 'O':
            m = '^'
        elif D[i,j][2] == 'P':
            m = 'v'
        elif D[i,j][2] == 'Z':
            m = 'P'
        plt.scatter(J2[i],J3[j],color=c,marker=m,s=100)
plt.show()




















