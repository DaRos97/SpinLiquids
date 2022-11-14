import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import getopt
import functions as fs
from scipy.optimize import curve_fit
import csv

#Takes the data in Data/final_S_DM and converts the "Converge" value from True to TrueL (SL) or TrueO (LRO) depending on the final value of the gap
#In order to work, final_S_DM has to contain all the ansatze, so that then we can compare it in the phase diagram using effort_DM/plot.py
cutoff_gap = 1e-2
#parameters are: ansatz, j2,j3, DM angle, Spin
list_ans = ['3x3','q0','cb1','cb1_nc','cb2','oct']
DM_list = {'000':0, '006':np.pi/48, '013':2*np.pi/48, '019':3*np.pi/48, '026':4*np.pi/48, '032':5*np.pi/48, '039':6*np.pi/48, '209':2*np.pi/3}
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:", ['DM='])
    S = 0.5
    txt_S = '50'
    DM = '000'
except:
    print("Error in input parameters")
    exit()
for opt, arg in opts:
    if opt in ['-S']:
        txt_S = arg
        S_label = {'50':0.5,'36':(np.sqrt(3)-1)/2,'34':0.34,'30':0.3,'20':0.2}
        S = S_label[txt_S]         #####CHECK
    if opt == '--DM':
        DM = arg.replace('.','')
        if DM not in DM_list.keys():
            print('Not computed DM angle')
            exit()
N_max = 37
print("Using arguments: Dm angle = ",DM," spin S = ",S)
#import data
data_dirname = '../../Data/final_'+txt_S+'_'+DM+'/'
##### 
m_ans = {'q0':1, '3x3':3, 'cb1':4, 'cb1_nc':4}
m_ans_gauge = {'q0':3, '3x3':3, 'cb1':12, 'cb1_nc':12}
########
for filename in os.listdir(data_dirname):
    data_name = data_dirname + filename
    with open(data_name, 'r') as f:
        lines = f.readlines()
    #
    N__ = (len(lines)-1)//2 + 1
    print(filename)
    for i in range(N__):
        data = lines[i*2+1].split(',')
        data[-1] = data[-1][:-1]
        header = lines[i*2].split(',')
        header[-1] = header[-1][:-1]
        #
        if data[3][0] == 'F' or data[3][-1] in ['L','O']:
            continue
        # Go to data directory and compute gap scaling up to this point
        ans = data[0]
        J2 = float(data[1])
        J3 = float(data[2])
        if DM != '209':
            m_ = m_ans[ans]
        else:
            m_ = m_ans_gauge[ans]
        N_reference = np.arange(13,N_max - (N_max-13)%m_ + m_,m_,dtype = int)
        NN_n_ = []
        for n in N_reference:
            if n>25 and n<37:
                continue
            NN_n_.append(n)
        N_reference = NN_n_
        #list directories to find list of Ns
        Data_dir = '../../Data/S'+txt_S+'/phi'+DM+'/'
        Ns = []
        params = {}
        #get all parameters at all Ns
        for N_dir in os.listdir(Data_dir):
            if int(N_dir) not in N_reference:
                continue
            for files in os.listdir(Data_dir+N_dir+'/'):
                with open(Data_dir+N_dir+'/'+files, 'r') as f:
                    lines_1 = f.readlines()
                N_ = (len(lines_1)-1)//2 +1
                for i_ in range(N_):
                    lines_2 = lines_1[i_*2+1].split(',')
                    if lines_2[0] == ans and np.abs(float(lines_2[1])-J2) < 1e-6 and np.abs(float(lines_2[2])-J3) < 1e-6 and lines_2[3][0] == 'T':
                        Ns.append(int(N_dir))
                        params[N_dir] = []
                        for p in lines_2[6:]:
                            params[N_dir].append(float(p))
        #compute Gap from interpolation
        gaps = []
        N_N = np.sort(Ns)
        for nnn_ in N_N:
            gaps.append(fs.find_gap(params[str(nnn_)],nnn_,[ans,DM_list[DM],J2,J3,txt_S]))
        try:
            parameters, covariance = curve_fit(fs.linear, N_N, gaps, p0=(1,0))
        except:
            print("Not fitted ",ans,"at ",J2,J3)
            fitted_ = fs.linear(np.linspace(N_N[0],N_N[-1],100),parameters[0],parameters[1])
            plt.figure()
            plt.plot(N_N,gaps,'ro')
            plt.plot(np.linspace(N_N[0],N_N[-1],100),fitted_,'g-')
            plt.hlines(parameters[1],N_N[0],N_N[-1],color='green',linestyles='--')
            plt.title(ans+"__"+str(J2)+"_"+str(J3))
            plt.show()
        #Change True -> TrueL or TrueO
        order = 'L' if parameters[1] > cutoff_gap else 'O'
        data[3] = data[3] + order
        #fill lines[2*i+1]
        lines[2*i+1] = ''
        for ini,a_ in enumerate(data):
            lines[2*i+1] += a_ 
            if ini != len(data)-1:
                lines[2*i+1] += ','
            else:
                lines[2*i+1] += '\n'
        #Save transformed data
        Data = {}
        I = i+1
        for iii,h in enumerate(header):
            Data[h] = data[iii]
        with open(data_name,'w') as f:
            for i in range(2*I-2):
                f.write(lines[i])
        with open(data_name,'a') as f:
            writer = csv.DictWriter(f, fieldnames = header)
            writer.writeheader()
            writer.writerow(Data)
        with open(data_name,'a') as f:
            for l in range(2*I,len(lines)):
                f.write(lines[l])
        #end


