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
    opts, args = getopt.getopt(argv, "S:K:", ['only='])
    txt_S = '50'
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
    if opt == '--only':
        only = arg
        do_only = True
#
if do_only:
    considered_ans = (only,)
else:
    considered_ans = Color.keys()
#
dirname = '../../Data/self_consistency/UV/S'+txt_S+'/'+K+'/' 
#
t1 = 1
t2 = t1*0.145
t3 = t1*0.08
Ui = t1*50
Uf = t1*150
UV_pts = 20
U_list = np.linspace(Ui,Uf,UV_pts)
V_list = []
for u in U_list:
    V_list.append(np.linspace(0.08*u,0.15*u,UV_pts))
D = np.ndarray((UV_pts,UV_pts),dtype='object')
DD_none = D[0,0]
cnt = 0
for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    if len(lines) > 0:
        x = filename.replace('_',',',1)
        st = x.index('_')
        ts = x.index(')')
        u = filename[5:st]
        v = filename[st+1:ts]
        for a in range(UV_pts):
            if u == '{:5.4f}'.format(U_list[a]).replace('.',''):
                a_ = a
        for b in range(UV_pts):
            if v == '{:5.4f}'.format(V_list[a_][b]).replace('.',''):
                b_ = b
    N = (len(lines)-1)//2 + 1
    minE = 1000
    i_ = 0
    done = False 
    for i in range(N):
        head = lines[2*i].split(',')
        head[-1] = head[-1][:-1]
        data = lines[2*i+1].split(',')
        if data[0] not in considered_ans:
            continue
        if data[0] == '17':
            continue
        if data[0] == '19' and np.abs(float(data[head.index('phiA1p')])-np.pi) < 1e-3:
            continue
        if data[0] == '20' and (np.abs(float(data[head.index('phiA1p')])-np.pi) < 1e-3 or np.abs(float(data[head.index('phiA1p')])) < 1e-3):
            continue
        tempE = float(data[head.index('Energy')])
        if tempE < minE:
            minE = tempE
            i_ = i
            done = True
    if not done:
        continue
    head = lines[2*i_].split(',')
    head[-1] = head[-1][:-1]
    data = lines[2*i_+1].split(',')
    res = data[0]
#    if res == '17':
#        continue
    for s in range(1,head.index('J1')):
        res += head[s]+data[s]
    D[a_,b_] = res
##########
fig = plt.figure(figsize=(10,10))
for i in range(UV_pts):
    for j in range(UV_pts):
        if D[i,j] == DD_none:
            continue
        ans = D[i,j][:2]
        x = U_list[i]
        y = V_list[i][j]/U_list[i]
        c = Color[ans][0]
        m = 'o'
#        plt.text(x,y,D[i,j])
        plt.scatter(x,y,color=c,marker=m,s=100)

plt.show()




















