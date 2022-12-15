import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import getopt
import functions as fs
from scipy.optimize import curve_fit

#parameters are: ansatz, j2,j3, DM angle, Spin
list_ans = ['3x3','q0','cb1','cb1_nc','cb2','oct']
DM_list = {'000':0, '005':0.05, '104':np.pi/3*2, '209':2*np.pi/3}
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:", ['j2=','j3=','DM=','ans=','Nmax=',"fit="])
    S = 0.5
    txt_S = '50'
    J2 = 0
    J3 = 0
    DM = '000'
    ans = '3x3'
    N_max = 13
    fit = 'lin'
except:
    print("Error in input parameters")
    exit()
for opt, arg in opts:
    if opt in ['-S']:
        txt_S = arg
        S_dic = {'50':0.5,'36':(np.sqrt(3)-1)/2,'34':0.34,'30':0.3,'20':0.2}
        if txt_S not in S_dic.keys():
            print('Error in -S argument')
            exit()
        S = S_dic[txt_S]         #####CHECK
    if opt == '--j2':
        J2 = float(arg)
    if opt == '--j3':
        J3 = float(arg)
    if opt == '--DM':
        DM = arg.replace('.','')
        if DM not in DM_list.keys():
            print('Not computed DM angle')
            exit()
        type_of_ans = 'SU2' if DM in ['000','104','209'] else 'TMD'
    if opt == '--ans':
        ans = arg 
        if ans not in list_ans:
            print('Error in -ans choice')
            exit()
    if opt == '--Nmax':
        N_max = int(arg)
    if opt == '--fit':
        fit = arg

print("Using arguments: ans-> ",ans," j2,j3 = ",J2,",",J3," Dm angle = ",DM," spin S = ",S)
#import data
arguments = (ans,DM,J2,J3,txt_S)
arg2 = (ans,DM_list[DM],J2,J3,txt_S,type_of_ans)
data = []
gaps1 = []
gaps2 = []
#### N list for different ansatze depending on gap closing points
m_ans = {'q0':1, '3x3':3, 'cb1':4}
m_ans_gauge = {'q0':3, '3x3':3, 'cb1':12}
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
#Just 13,25,37
N_ = [13,25,37]
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
def quadratic(x,a,b):
    return a/x**2 + b
def sqrt(x,a,b):
    return a/np.sqrt(x) + b
fit_curve_def = {'exp':exp_decay,'lin':linear,'quad':quadratic,'sqrt':sqrt}
try:
    pars,cov = curve_fit(fit_curve_def[fit],N_,gaps2,p0=[1,0],bounds=(0,np.inf))
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
