import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
import os
import sys
import getopt
from matplotlib import cm
from matplotlib.lines import Line2D

Color = {'1b': ['r','orange'],
         '1a':  ['blue','aqua'],
         '1c':  ['pink','aqua'],
         #'1c1':  ['pink','aqua'],
         #'1c2':  ['pink','aqua'],
         '1d':  ['orange','aqua'],
         '1e':  ['yellow','aqua'],
         '1f':  ['forestgreen','lime'],
#         '1f0':  ['yellow','lime'],
#         '1f1':  ['red','lime'],
#         '1f2':  ['aqua','lime'],
#         '1f3':  ['gold','lime'],
#         '1f4':  ['limegreen','lime'],
         'labels':  ['k','k']
         }
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "K:")
    K = 0
except:
    print("Error in inputs")
    exit()
for opt, arg in opts:
    if opt in ['-K']:
        K = int(arg)
if K:
    dirname = '../Data/SDM/'+str(K)+'/' 
else: 
    dirname = '../Data/SDM/final_SDM/' 
dirname = '../Data/sdm/13/'
#
S_max = 0.5
DM_max = 0.3
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
    N = (len(lines)-1)//2 + 1
    minE1 = 10
    minE2 = 10
    minE3 = 10
    for i in range(N):
        data = lines[i*2+1].split(',')
        if data[0] not in Color.keys():     #not a considered ansatz
            continue
        dm = list(DM_list).index(float(data[2]))
        s = list(S_list).index(float(data[1]))
        if float(data[4]) < minE1:
            if data[3][-1] in ['L','O']:        #spin Liquid or long range Order
                txt_SL = data[3][-1]
                txt_conv = 'g' if data[3][0] == 'T' else 'b'
            else:
                txt_SL = 'n'                    #not determined
                txt_conv = 'g' if data[3][0] == 'T' else 'b'
            D[dm,s] = data[0] + txt_conv + txt_SL
            minE1 = float(data[4])
        elif float(data[4]) < minE2:
            minE2 = float(data[4])
        elif float(data[4]) < minE3:
            minE3 = float(data[4])
    if abs(minE2 - minE1) < 1e-5 and abs(minE1 - minE3) < 1e-5:
        print(filename)
##########
pts = len(os.listdir(dirname))
fig = plt.figure(figsize=(8,4))
plt.title("DM diagram")
#plt.gca().set_aspect('equal')
for i in range(DM_pts):
    for j in range(S_pts):
        OL = 1 if D[i,j][-1] == 'L' else 0       #Order or Liquid
        m = 'o' if D[i,j][-2] == 'g' else 'x'
        if D[i,j][-1] == 'n':
            m = 'o'
            OL = 0
        c = Color[D[i,j][:-2]][OL]
        plt.scatter(DM_list[i],S_list[j],color=c,marker=m)
plt.ylim(0,0.501)
plt.xlim(-0.001,0.1)
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
#
plt.show()

