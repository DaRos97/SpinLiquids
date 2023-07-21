import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import getopt
import functions as fs
from scipy.optimize import curve_fit
from pathlib import Path
import scipy.odr as odr
#
from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
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
    ans = '20'
    N_max = 49
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

fig = plt.figure(figsize=(30,5))
ss = 20
#lt.margins(x=0,y=0)
for iii,txt_S in enumerate(['50','36','30','20']):
#    ax = fig.add_subplot(1,4,iii+1)
    plt.subplot(1,4,iii+1)
    #import data
    arguments = (ans,DM,J2,J3,txt_S)
    data = []
    gaps1 = []
    gaps2 = []
    #### N list for different ansatze depending on gap closing points
    N_list = [13,25,37,49]
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

    N_steps = np.linspace(N_[0],N_[-1],100)
    #plt.xscale('log')
    #plt.yscale('log')
    dic_S = {'50':r'$0.5$','36':r'$(\sqrt{3}-1)/2$','30':r'$0.3$','20':r'$0.2$'}
    #ax.set_title('S = '+dic_S[txt_S],fontsize=ss+5)
    plt.title('S = '+dic_S[txt_S],fontsize=ss+10)
    #ax.plot(N_,gaps1,'r*')
    plt.plot(N_,gaps1,'r*')
    if conv:
        #ax.plot(N_steps,fit_curve_def[fit](N_steps,*pars),'g--')
        #ax.hlines(pars[1],N_[0],N_[-1],color='blue',linestyles='--')
        plt.plot(N_steps,fit_curve_def[fit](N_steps,*pars),'g--')
        plt.hlines(pars[1],N_[0],N_[-1],color='blue',linestyles='--')
        #plt.text(30,abs(gaps1[0]+gaps1[1])/2,'a:  '+str(pars[0])+'\nb:  '+str(pars[1]))
    if 0:
        func = np.poly1d(z)
        N_steps = np.linspace(N_[0],100,1000)
        fitted = []
        for n in N_steps:
            fitted.append(func(n))
        #ax.plot(N_steps,fitted,'g-')
        plt.plot(N_steps,fitted,'g-')
        limit = func(100)
        #ax.hlines(pars[1],N_[0],N_[-1],color='green',linestyles='--')
        plt.hlines(pars[1],N_[0],N_[-1],color='green',linestyles='--')
    plt.xticks(fontsize = ss+5)
    plt.yticks(fontsize = ss)
    #ax.set_xlabel(r'$N_k$',fontsize=ss+5)
    plt.xlabel(r'$N_k$',fontsize=ss+5)
    if iii == 0:
        #ax.set_ylabel('gap',fontsize=ss+5)
        plt.ylabel('gap',fontsize=ss+5)
    #ax.yaxis.set_major_formatter(formatter) 
    ind_off = -1 if iii != 3 else 0
#    plt.ticklabel_format(axis='y',style='scientific',scilimits=(0,0),useMathText=True,useOffset=gaps1[ind_off])
#fig.set_size_inches(15,5)
plt.show()

























