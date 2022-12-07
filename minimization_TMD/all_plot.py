import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
import os
import sys
import getopt
from matplotlib import cm
from matplotlib.lines import Line2D

Color = {'3x3': ['r','orange'],
         'q0':  ['blue','dodgerblue'],
         'cb1':  ['forestgreen','lime'],
         #'cb1_nc':  ['yellow','y'],
         #'labels':  ['k','k']
         }
S_l = ['50','36','34','30','20']
DM_l = ['005']
list_plots = []
for s in S_l:
    for dm in DM_l:
        list_plots.append([s,dm])
phi_label = {'005':0.05}
xxx_dm = {'005':1}
yyy_s  = {'50':1,'36':2,'34':3,'30':4,'20':5}
fig = plt.figure(figsize=(4,8))
for s,dm in list_plots:
    phi = phi_label[dm]
    dirname = '../Data/final_'+s+'_'+dm+'/' 
    #dirname = '../Data/S50/phi000/13/'
    title_DM = {'005':'0.05','104': '\pi/3', '209':'2*\pi/3'}
    title_dm = r'Phi = '+title_DM[dm]
    title_s = r'S = 0.'+s
    #
    D = np.ndarray((9,9),dtype='object')
    delta = np.zeros((9,9))
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
        e1 = 0
        e2 = 0
        for i in range(N):
            data = lines[i*2+1].split(',')
            if data[0] not in Color.keys():     #not a considered ansatz
                continue
            j2 = float(data[1]) - Ji
            j3 = float(data[2]) - Ji
            i2 = int(j2*8/(0.6))
            i3 = int(j3*8/(0.6))
            if data[0] == 'cb1':
                e1 = float(data[4])
            if data[0] == 'cb1_2':
                e2 = float(data[4])
            if float(data[4]) < minE:
                if data[3][-1] in ['L','O']:      #spin Liquid or long range Order
                    txt_SL = data[3][-1]
                    txt_conv = 'g' if data[3][0] == 'T' else 'b'
                else:
                    txt_SL = 'n'            #not determined
                    txt_conv = 'g' if data[3][0] == 'T' else 'b'
                D[i2,i3] = data[0] + txt_conv + txt_SL
                minE = float(data[4])
        if e1 and e2:
            delta[i2,i3] = e1-e2
        else:
            delta[i2,i3] = np.nan
    ##########
    pts = len(os.listdir(dirname))
    xxx = xxx_dm[dm]
    yyy = yyy_s[s]
    plt.subplot(len(yyy_s),len(xxx_dm),(yyy-1)*len(xxx_dm)+xxx)
    if yyy == 1:
        plt.title(title_dm)
    if xxx == 1:
        plt.text(-1,0,title_s)
    if yyy == len(yyy_s):
        plt.xticks([-0.3,0,0.3],['-0.3','0','0.3'])
    else:
        plt.xticks([])
    if xxx == 1:
        plt.yticks([-0.3,0,0.3],['-0.3','0','0.3'])
    else:
        plt.yticks([])
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
#plt.figure()
#plt.subplot(1,3,2)
#plt.axis('off')
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

#plt.legend(legend_lines,list_leg,loc='upper left',bbox_to_anchor=(1,1),fancybox=True)
#
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
