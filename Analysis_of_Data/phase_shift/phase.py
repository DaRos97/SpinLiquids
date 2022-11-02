import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from pathlib import Path
import sys

#import data of cb1, DM = 0, J2,J3 = 0,0
ans = sys.argv[3]
phi = 0
phi_t = 0
S = '05'
J2 = float(sys.argv[1])
J3 = float(sys.argv[2])

FileName = '../../Data/S'+S+'/phi'+"{:3.2f}".format(phi_t).replace('.','')+"/19/"+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'

with open(Path(FileName), 'r') as f:
    lines = f.readlines()
N = (len(lines)-1)//2 + 1
P = {}
for i in range(N):
    pars = lines[i*2].split(',')
    pars[-1] = pars[-1][:-1]
    data = lines[i*2+1].split(',')
    if data[0] != ans:
        continue
    if data[3] != 'True':
        print("Non-converged point, abort")
        exit()
    p = 0
    for d in data[4:]:
        if float(d) != 0.0:
            P[pars[4+p]] = float(d)
        p += 1
print("Found params: ",P)
print("#########################")
#
header = {'3x3_1':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A3','B1','B2','B3','phiB1','phiB2','phiA3'],  #3x3
          '3x3_2':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A3','B1','B2','B3','phiA1','phiB1','phiB2','phiB3'],  #3x3
          'q0_1':     ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','B1','B2','B3','phiB1','phiA2','phiB2'],  #q0
          'q0_2':     ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','B1','B2','B3','phiA1','phiB1','phiA2','phiB2','phiB3'],  #q0
          'cb1':      ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiA1','phiB1','phiA2','phiB2'],  #cuboc1
          'cb1_nc':   ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiA1','phiB1','phiA2','phiB2'],  #cuboc1
          'cb2':      ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiA1','phiB1','phiA2','phiB2'],  #cuboc2
          'oct':      ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','B1','B2','B3','phiB1','phiB2','phiA2']}  #octahedral
Par_t = header[ans][7:]
if np.abs(J2) < 1e-5:
    for pp in ['A2','B2','phiA2','phiB2']:
        if pp in Par_t:
            Par_t.remove(pp)
if np.abs(J3) < 1e-5:
    for pp in ['A3','B3','phiA3','phiB3']:
        if pp in Par_t:
            Par_t.remove(pp)
Bnd = [0.1,2*np.pi,0.001]
pts_1 = 35       #odd
pts_2 = 101       #odd
En = {}
L = []
list_P = {}
args = (1,J2,J3,ans,phi)
print("Energy: ",fs.totE(P,args)[0])
for par_t in Par_t:
    pts = pts_1
    if par_t == 'L':
        pts = pts_2
        med_L = P[par_t]
        list_P[par_t] = np.linspace(med_L-Bnd[2],med_L+Bnd[2],pts)
        tempP = []
        list_excluded = ['L','Energy','Sigma','gap']
        for p_t in P.keys():
            if p_t not in list_excluded:
                tempP.append(P[p_t])
        En[par_t] = np.zeros(pts)
        for i,Par in enumerate(list_P[par_t]):
            L_temp = Par
            En[par_t][i] = fs.totEl(tempP,L_temp,args)
            if En[par_t][i] < -10:
                En[par_t][i] = np.nan
        continue
    bnd = Bnd[0] if par_t[:3] != 'phi' else Bnd[1]
    med_P = P[par_t]
    list_P[par_t] = np.linspace(med_P-bnd,med_P+bnd,pts)
    if 'phiA1' in Par_t:
        list_P['phiA1'] = np.linspace(med_P-0.5,med_P+0.5,pts)
    En[par_t] = np.zeros(pts)
    tempP = dict(P)
    for i,Par in enumerate(list_P[par_t]):
        tempP[par_t] = Par
        En[par_t][i] = fs.totE(tempP,args)[0]


fig = plt.figure()
for n,par_t in enumerate(Par_t):
###################################à    FIGURE
    plt.subplot(3,4,n+1)
    plt.plot(list_P[par_t],En[par_t])
    #plt.scatter(P[par_t],P['Energy'],color = 'g', marker = 'o')
    plt.scatter(list_P[par_t][len(list_P[par_t])//2],En[par_t][len(En[par_t])//2],color = 'r', marker = '^')
    plt.title(par_t)
plt.show()













