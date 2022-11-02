import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import getopt
import functions as fs
from scipy.optimize import curve_fit

#parameters are: ansatz, j2,j3, DM angle, Spin
list_ans = ['3x3_1','q0_1','cb1','cb2','oct']
DM_list = {'000':0, '006':np.pi/48, '013':2*np.pi/48, '019':3*np.pi/48, '026':4*np.pi/48, '032':5*np.pi/48, '039':6*np.pi/48, '209':2*np.pi/3}
argv = sys.argv[1:]
#try:
if 1:
    opts, args = getopt.getopt(argv, "S:", ['j2=','j3=','DM=','ans=','Nmax='])
    S = 0.5
    txt_S = '05'
    J2 = 0
    J3 = 0
    DM = '000'
    ans = '3x3_1'
    N_max = 13
#except:
else:
    print("Error in input parameters")
    exit()
for opt, arg in opts:
    if opt in ['-S']:
        txt_S = arg
        if txt_S not in ['05','03']:
            print('Error in -S argument')
            exit()
        else:
            S = 0.5 if txt_S == '05' else 0.366         #####CHECK
    if opt == '--j2':
        J2 = float(arg)
    if opt == '--j3':
        J3 = float(arg)
    if opt == '--DM':
        DM = arg.replace('.','')
        if DM not in DM_list.keys():
            print('Not computed DM angle')
            exit()
    if opt == '--ans':
        ans = arg 
        if ans not in list_ans:
            print('Error in -ans choice')
            exit()
    if opt == '--Nmax':
        N_max = int(arg)
print("Using arguments: ans-> ",ans," j2,j3 = ",J2,",",J3," Dm angle = ",DM," spin S = ",S)
#import data
arguments = (ans,DM,J2,J3,txt_S)
arg2 = (ans,DM_list[DM],J2,J3,txt_S)
data = []
gaps1 = []
gaps2 = []
#### N list for different ansatze depending on gap closing points
m_ans = {'q0_1':1, '3x3_1':3, 'cb1':4}
m_ans_gauge = {'q0_1':3, '3x3_1':3, 'cb1':12}
if DM != '209':
    m_ = m_ans[ans]
else:
    m_ = m_ans_gauge[ans]
N_ = np.arange(13,N_max - (N_max-13)%m_ + m_,m_,dtype = int)
NN = []
for n in N_:
    if n>25 and n < 37:
        continue
    NN.append(n)
N_ = NN
print(NN)
###
for n in N_:
    data.append(fs.get_data(arguments,n))
    gap = fs.find_gap(data[-1],n,arg2)
    gaps1.append(data[-1][0])
    gaps2.append(gap)
#fit
def exp_decay(x,a,b,c):
    return a*np.exp(b*x) + c
def linear(x,a,b):
    return a/x + b


if 0:
    z = np.polyfit(N_,gaps2,2)
    conv = 1
if 0:
    pars,cov = curve_fit(exp_decay,N_,gaps2,p0=[1,-1,0],bounds=(-1e5,np.inf))
    print("Fitted")
    conv = 1
try:
    pars,cov = curve_fit(linear,N_,gaps2,p0=[1,0],bounds=(-1e5,np.inf))
    print("Fitted")
    conv = 1
except:
    print("Not fitted")
    conv = 0

plt.figure()
#plt.plot(N_,gaps1,'b*')
plt.plot(N_,gaps2,'ro')
if conv:
    N_steps = np.linspace(N_[0],N_[-1],100)
    plt.plot(N_steps,linear(N_steps,*pars),'g--')
    plt.hlines(pars[1],N_[0],N_[-1],color='green',linestyles='--')
if 0:
    func = np.poly1d(z)
    N_steps = np.linspace(N_[0],100,1000)
    fitted = []
    for n in N_steps:
        fitted.append(func(n))
    plt.plot(N_steps,fitted,'g-')
    limit = func(100)
    plt.hlines(limit,N_[0],N_[-1],color='green',linestyles='--')
plt.show()
