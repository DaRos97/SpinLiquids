import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from pathlib import Path
import sys

#import data of cb1, DM = 0, J2,J3 = 0,0
ans = 'cb1'
phi = 0
phi_t = 0
J2 = float(sys.argv[1])
J3 = float(sys.argv[2])

FileName = '../Data/S05/phi'+"{:3.2f}".format(phi_t).replace('.','')+"/13/"+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
args = (1,J2,J3,ans,phi)

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
####################################
Par_t = ['Energy','Sigma','gap','L','A1','A2','B1','B2','phiA1','phiB1','phiA2','phiB2']

newP = dict(P)
newP['phiA1'] = 2*np.pi - P['phiA1']
#newP['phiA2'] = 2*np.pi - P['phiA2']
#newP['phiB2'] = 2*np.pi - P['phiB2']

reference_energy = fs.totE(newP,args)
dp = 1e-6
df = 1e-5
derivatives = {}
for par_t in Par_t[4:]:
    tempP = dict(newP)
    DP = df if par_t[:3] == 'phi' else dp
    tempP[par_t] += DP
    temp_energy = fs.totE(tempP,args)
    derivatives[par_t] = (reference_energy-temp_energy)/DP

print(derivatives)
print("#########################")
####################################
#exit()
#
header = {'3x3_1':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A3','B1','B2','B3','phiB1','phiB2','phiA3'],  #3x3
          '3x3_2':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A3','B1','B2','B3','phiA1','phiB1','phiB2','phiB3'],  #3x3
          'q0_1':     ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','B1','B2','B3','phiB1','phiA2','phiB2'],  #q0
          'q0_2':     ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','B1','B2','B3','phiA1','phiB1','phiA2','phiB2','phiB3'],  #q0
          'cb1':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiA1','phiB1','phiA2','phiB2'],  #cuboc1
          'cb2':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiA1','phiB1','phiA2','phiB2'],  #cuboc2
          'oct':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','B1','B2','B3','phiB1','phiB2','phiA2']}  #octahedral
#Par_t = header[ans][4:]
Par_t = ['Energy','Sigma','gap','L','A1','A2','B1','B2','phiA1','phiB1','phiA2','phiB2']
Bnd = [0.1,np.pi]
pts = 25       #odd
En = {}
list_P = {}
args = (1,J2,J3,ans,phi)
for par_t in Par_t:
    bnd = Bnd[0] if par_t[:3] != 'phi' else Bnd[1]
#   
    med_P = P[par_t]
#list_P = np.linspace(med_P-bnd,med_P+bnd,pts)
    list_P[par_t] = np.linspace(med_P-bnd,med_P+bnd,pts)
    En[par_t] = np.zeros(pts)
    tempP = dict(P)
    tempP['phiA1'] = 2*np.pi - P['phiA1']
    for i,Par in enumerate(list_P[par_t]):
        tempP[par_t] = Par
        En[par_t][i] = fs.totE(tempP,args)


fig = plt.figure()
P['phiA1'] = 2*np.pi - P['phiA1']
for n,par_t in enumerate(Par_t[4:]):
###################################à    FIGURE
    plt.subplot(3,3,n+1)
    plt.plot(list_P[par_t],En[par_t])
    plt.scatter(P[par_t],En[par_t][pts//2],color = 'r', marker = '*')
    plt.title('Paramter: '+par_t)
plt.show()















