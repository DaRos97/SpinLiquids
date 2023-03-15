import numpy as np
import matplotlib.pyplot as plt
import csv
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
    opts, args = getopt.getopt(argv, "S:", ['DM=','cutoff='])
    S = 0.5
    txt_S = '50'
    DM = '000'
    fit = 'quad'
    cutoff_gap = 0.001
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
    if opt == '--DM':
        DM = arg
        if DM not in DM_list.keys():
            print('Not computed DM angle')
            exit()
    if opt == '--cutoff':
        cutoff_gap = float(arg)
print("doing ",S," ",DM," with cutoff ",cutoff_gap)
dirname = '../../Data/self_consistency/S'+txt_S+'/phi'+DM+'/'
save_dir = dirname + 'final/'
N_ = [13,25,37,49]
Grid = 31#9
Ji = -0.3
Jf = -Ji
J2_list = np.linspace(Ji,Jf,Grid)
J3_list = np.linspace(Ji,Jf,Grid)
#
P13 = np.ndarray((Grid,Grid),dtype='object')
gap = {'13':np.zeros((Grid,Grid)),'25':np.zeros((Grid,Grid)),'37': np.zeros((Grid,Grid)),'49': np.zeros((Grid,Grid))}
for n in N_:
    data_dir = dirname+str(n)+'/'
    for filename in os.listdir(data_dir):
        with open(data_dir+filename, 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            continue
        head_data = lines[0].split(',')
        data = lines[1].split(',')
        try:
            j2 = float(data[head_data.index('J2')])
            j3 = float(data[head_data.index('J3')])
            i2 = list(J2_list).index(j2) 
            i3 = list(J3_list).index(j3) 
        except:
            continue
        if n == 13:
            N = (len(lines)-1)//2 + 1
            minE = 100
            i_ = 0
            for i in range(N):
                head = lines[2*i].split(',')
                head[-1] = head[-1][:-1]
                data = lines[2*i+1].split(',')
                temp_E = float(data[head.index('Energy')])
                if temp_E < minE:
                    minE = temp_E
                    i_ = i
            pars = {}
            head = lines[2*i_].split(',')
            head[-1] = head[-1][:-1]
            data = lines[2*i_+1].split(',')
            for p in range(len(data)):
                if head[p] == 'ans':
                    pars[head[p]] = data[p]
                elif len(head[p]) == 2 and head[p][0] == 'p':
                    pars[head[p]] = int(data[p])
                else:
                    pars[head[p]] = float(data[p])
            P13[i2,i3] = pars
            gap[str(n)][i2,i3] = float(data[head.index('Gap')])
        else:
            gap[str(n)][i2,i3] = float(data[head_data.index('Gap')])

# Fitting
fit_curve_def = {'lin':fs.linear,'quad':fs.quadratic, 'ql':fs.ql}
for i in range(Grid):
    for j in range(Grid):
        J2_ = J2_list[i]
        J3_ = J3_list[j]
#        print(J2_,J3_)
        csvname = 'J2_J3=('+'{:5.4f}'.format(J2_).replace('.','')+'_'+'{:5.4f}'.format(J3_).replace('.','')+').csv'
        csvfile = save_dir + csvname
        if Path(csvfile).is_file():
            with open(csvfile, 'r') as f:
                lines = f.readlines()
            if float(lines[1].split(',')[2]) == cutoff_gap:
                continue
        gaps = []
        not_comp = False
        for n in N_:
            if gap[str(n)][i,j] == 0:
                not_comp = True
            gaps.append(gap[str(n)][i,j])
        if not_comp:
            continue
        pin = [1,0]
        pars,cov = curve_fit(fit_curve_def[fit],N_,gaps,p0=pin,bounds=(-100,100))
        #
        order = 'SL'
        if pars[1] < cutoff_gap:                  ######################
            order = 'LRO'
        DataDic = P13[i,j]
        head = list(DataDic.keys())
        data = list(DataDic.values())
        head[head.index('Gap')] = 'Gap13'
        head.insert(head.index('Gap13')+1,'Gap25')
        head.insert(head.index('Gap25')+1,'Gap37')
        head.insert(head.index('Gap37')+1,'Gap49')
        data.insert(head.index('Gap25'),gap['25'][i,j])
        data.insert(head.index('Gap37'),gap['37'][i,j])
        data.insert(head.index('Gap49'),gap['49'][i,j])
        head.insert(1,'order')
        data.insert(1,order)
        head.insert(2,'cutoff')
        data.insert(2,cutoff_gap)
        Dic = {}
        for ss in range(len(head)):
            Dic[head[ss]] = data[ss]
        with open(csvfile,'w') as f:
            writer = csv.DictWriter(f, fieldnames = head)
            writer.writeheader()
            writer.writerow(Dic)
        continue
        #plot
        #if data[0] != '19':
        #    continue
        print(data[0],J2_,J3_,order)
        print(pars,cov)
        plt.figure()
        N_steps = np.linspace(N_[0],N_[-1],100)
        plt.title(str(J2_list[i])+'_'+str(J3_list[j]))
        plt.plot(N_,gaps,'b*')
        plt.plot(N_steps,fit_curve_def[fit](N_steps,*pars),'g--')
        plt.hlines(pars[1],N_[0],N_[-1],color='blue',linestyles='--')
        plt.show()

