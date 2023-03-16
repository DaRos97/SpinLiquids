import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import getopt
import functions as fs
from scipy.optimize import curve_fit
from pathlib import Path
import scipy.odr as odr

#parameters are: ansatz, j2,j3, DM angle, Spin
list_ans = ['15','16','17','18','19','20']
DM_list = {'000':0, '005':0.05, '104':np.pi/3, '209':2*np.pi/3}
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:", ['j2=','j3=','DM=','ans=','Nmax='])
    S = 0.5
    txt_S = '50'
    J2 = 0
    J3 = 0
    DM = '000'
    ans = '3x3'
    N_max = 13
    fit = 'quad'
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
        DM = arg
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
data = []
gaps1 = []
gaps2 = []
#### N list for different ansatze depending on gap closing points
N_list = [13,25,37,49,62]
N_ = N_list[:N_list.index(N_max)+1]
bad_N = []
###
for n in N_:
    DirName = '../../Data/self_consistency/S'+txt_S+'/phi'+DM+"/"
    DataDir = DirName + str(n) + '/'
    csvname = 'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
    csvfile = DataDir + csvname
    my_file = Path(csvfile)
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 +1        #2 lines per ansatz
        for i in range(N):
            head = lines[i*2].split(',')
            head[-1] = head[-1][:-1]
            data = lines[i*2+1].split(',')
            if data[0] == ans:
                temp = float(data[head.index('Gap')])
    gaps1.append(temp)
#fit
fit_curve_def = {'lin':fs.linear,'quad':fs.quadratic, 'ql':fs.ql}
try:
    pin = [1,1,0] if fit == 'ql' else [1,0]
    new_N = []
    for n in N_:
        if n in bad_N:
            continue
        new_N.append(n)
    N_ = new_N
    pars,cov = curve_fit(fit_curve_def[fit],N_,gaps1,p0=pin,bounds=(-100,100))
    #print("Fitted")
    #print(pars,'\n',cov)
    conv = 1
except:
    print("Not fitted")
    conv = 0

plt.figure()
N_steps = np.linspace(N_[0],N_[-1],100)
#plt.xscale('log')
#plt.yscale('log')
plt.title(ans+': DM = '+DM+', S = '+txt_S+', (J2,J3) = '+str(J2)+','+str(J3))#+'. Fit: '+fit)
#plt.plot(N_,gaps2,'ro')
plt.plot(N_,gaps1,'b*')
if conv:
    plt.plot(N_steps,fit_curve_def[fit](N_steps,*pars),'g--')
    plt.hlines(pars[1],N_[0],N_[-1],color='blue',linestyles='--')
    plt.text(30,abs(gaps1[0]+gaps1[1])/2,'fit: a/(3*L^2)+b\n\na:  '+str(pars[0])+'\nb:  '+str(pars[1]))
if 0:
    func = np.poly1d(z)
    N_steps = np.linspace(N_[0],100,1000)
    fitted = []
    for n in N_steps:
        fitted.append(func(n))
    plt.plot(N_steps,fitted,'g-')
    limit = func(100)
    plt.hlines(pars[1],N_[0],N_[-1],color='green',linestyles='--')
plt.xlabel('L',fontsize=16)
plt.ylabel('gap',fontsize=16)
plt.show()
