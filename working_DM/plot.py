import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
import os
import sys
import getopt
from matplotlib import cm
from matplotlib.lines import Line2D

Color = {'3x3': ['r','orange'],
         'q0':  ['blue','aqua'],
         'cb1':  ['forestgreen','lime'],
         'cb1_nc':  ['yellow','y'],
         'labels':  ['k','k']
         }
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:", ['DM='])
    txt_S = '50'
    phi_t = '000'
except:
    print("Error in inputs")
    exit()
for opt, arg in opts:
    if opt in ['-S']:
        txt_S = arg
    if opt == '--DM':
        phi_t = arg
phi_label = {'000':0, '104':np.pi/3, '209':np.pi/3*2}
phi = phi_label[phi_t]
dirname = '../Data/final_'+txt_S+'_'+phi_t+'/' 
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
    N = (len(lines)-1)//2 + 1
    minE = 10
    for i in range(N):
        data = lines[i*2+1].split(',')
        if data[0] not in Color.keys():     #not a considered ansatz
            continue
        j2 = float(data[1]) - Ji
        j3 = float(data[2]) - Ji
        i2 = int(j2*8/(0.6))
        i3 = int(j3*8/(0.6))
        if float(data[4]) < minE:
            if data[3][-1] in ['L','O']:      #spin Liquid or long range Order
                txt_SL = data[3][-1]
                txt_conv = 'g' if data[3][0] == 'T' else 'b'
            else:
                txt_SL = 'n'            #not determined
                txt_conv = 'g' if data[3][0] == 'T' else 'b'
            D[i2,i3] = data[0] + txt_conv + txt_SL
            minE = float(data[4])
##########
pts = len(os.listdir(dirname))
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title(title)
plt.gca().set_aspect('equal')
plt.axhline(y=0,color='k',zorder=-1)
plt.axvline(x=0,color='k',zorder=-1)
for i in range(9):
    for j in range(9):
        if D[i,j] == DD_none:
            continue
        OL = 1 if D[i,j][-1] == 'L' else 0       #Order or Liquid
        m = 'o' if D[i,j][-2] == 'g' else 'x'
        if D[i,j][-1] == 'n':
            m = '^'
            OL = 0
        c = Color[D[i,j][:-2]][OL]
        J2 = -0.3+i*0.6/8
        J3 = -0.3+j*0.6/8
        plt.scatter(J2,J3,color=c,marker=m)
#Legenda
list_leg = []
for col in Color.keys():
    if col == 'labels':
        continue
    list_leg.append(col+' LRO')
    list_leg.append(col+' SL')
list_leg.append('just energy')
list_leg.append('TD limit')
legend_lines = []
for col in Color.values():
    if col == ['k','k']:
        legend_lines.append(Line2D([], [], color="w", marker='^', markerfacecolor=col[0]))
        legend_lines.append(Line2D([], [], color="w", marker='o', markerfacecolor=col[1]))
        continue
    legend_lines.append(Line2D([], [], color="w", marker='o', markerfacecolor=col[0]))
    legend_lines.append(Line2D([], [], color="w", marker='o', markerfacecolor=col[1]))

plt.legend(legend_lines,list_leg,loc='upper left',bbox_to_anchor=(1,1),fancybox=True)
plt.show()


list_leg = ['3x3 LRO','3x3 SL','q=0 LRO','q=0 SL','CB1 LRO','CB1 SL','CB1_NC LRO','CB1_NC SL','just energy','TD limit']
legend_lines = [Line2D([], [], color="w", marker='o', markerfacecolor="r"),  
                Line2D([], [], color="w", marker='o', markerfacecolor="orange"),
                Line2D([], [], color="w", marker='o', markerfacecolor="blue"),
                Line2D([], [], color="w", marker='o', markerfacecolor="aqua"),
                Line2D([], [], color="w", marker='o', markerfacecolor="lime"),
                Line2D([], [], color="w", marker='o', markerfacecolor="olive"),
                Line2D([], [], color="w", marker='^', markerfacecolor="k"),
                Line2D([], [], color="w", marker='o', markerfacecolor="k")
                ]
