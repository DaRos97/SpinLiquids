import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import getopt
import functions as fs
from scipy.optimize import curve_fit
import scipy.odr as odr

#parameters are: ansatz, j2,j3, DM angle, Spin
list_ans = ['3x3','q0','cb1','cb1_nc','cb2','oct']
DM_list = {'000':0, '005':0.05, '104':np.pi/3, '209':2*np.pi/3}
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
N_ = [13,25,37] if N_max == 13 else [13,25,37,49]
###
for n in N_:
    data.append(fs.get_data(arguments,n))
    gap = fs.find_gap(data[-1],n,arg2)
    gaps1.append(data[-1][0])           #gap in .csv file
    gaps2.append(gap)                   #gap evaluated at the moment

#fit
fit_curve_def = {'lin':fs.linear,'quad':fs.quadratic, 'ql':fs.ql}
try:
    pin = [1,1,0] if fit == 'ql' else [1,0]
    pars,cov = curve_fit(fit_curve_def[fit],N_,gaps2,p0=pin,bounds=(0,np.inf))
    #print("Fitted")
    #print(pars,'\n',cov)
    conv = 1
except:
    print("Not fitted")
    conv = 0

plt.figure()
plt.title(ans+': DM = '+DM+', S = '+txt_S+', (J2,J3) = '+str(J2)+','+str(J3)+'. Fit: '+fit)
plt.plot(N_,gaps2,'ro')
plt.plot(N_,gaps1,'b*')
if conv:
    N_steps = np.linspace(N_[0],N_[-1],100)
    plt.plot(N_steps,fit_curve_def[fit](N_steps,*pars),'g--')
    plt.hlines(pars[1],N_[0],N_[-1],color='green',linestyles='--')
if 0:
    func = np.poly1d(z)
    N_steps = np.linspace(N_[0],100,1000)
    fitted = []
    for n in N_steps:
        fitted.append(func(n))
    plt.plot(N_steps,fitted,'g-')
    limit = func(100)
    plt.hlines(pars[1],N_[0],N_[-1],color='green',linestyles='--')
plt.text(30,abs(gaps2[0]+gaps2[1])/2,'a:  '+str(pars[0])+'\nb:  '+str(pars[1]))
plt.show()
