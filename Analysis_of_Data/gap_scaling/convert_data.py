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
cutoff_gap = 1e-3
#parameters are: ansatz, j2,j3, DM angle, Spin
list_ans = ['3x3','q0','cb1','cb1_nc','cb2','oct']
DM_list = {'000':0, '005':0.05, '104':np.pi/3, '209':2*np.pi/3}
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
        S = S_label[txt_S]
    if opt == '--DM':
        DM = arg.replace('.','')
        if DM not in DM_list.keys():
            print('Not computed DM angle')
            exit()
        type_of_ans = 'SU2' if DM in ['000','104','209'] else 'TMD'

print("Using arguments: Dm angle = ",DM," spin S = ",S," and cutoff: ",cutoff_gap)
#import data
data_dirname = '../../Data/final_'+txt_S+'_'+DM+'/'
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
        if data[3][0] == 'F':# or data[3][-1] in ['L','O']:       #if the point did not converge or has already been computed the phase, skip it
            continue
        # Go to data directory and compute gap scaling up to this point
        ans = data[0]
        J2 = float(data[1])
        J3 = float(data[2])
        #
        N_reference = [13,25,37,49]
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
            gaps.append(fs.find_gap(params[str(nnn_)],nnn_,[ans,DM_list[DM],J2,J3,txt_S,type_of_ans]))
        #
        try:
            parameters, covariance = curve_fit(fs.quadratic, N_N, gaps, p0=(1,0))
        except:
            print("Not fitted ",ans,"at ",J2,J3)
            fitted_ = fs.quadratic(np.linspace(N_N[0],N_N[-1],100),parameters[0],parameters[1])
            plt.figure()
            plt.plot(N_N,gaps,'ro')
            plt.plot(np.linspace(N_N[0],N_N[-1],100),fitted_,'g-')
            plt.hlines(parameters[1],N_N[0],N_N[-1],color='green',linestyles='--')
            plt.title(ans+"__"+str(J2)+"_"+str(J3))
            plt.show()
        #Change True -> TrueL or TrueO
        #order = 'L' if parameters[1] > cutoff_gap else 'O'     #old version
        #Sofisticate: check if either at N = 49 the gap is above the fitting line (->SL) with at the same time b > cutoff OR if the fit did not converge (case of flat line)
        order = 'L' if ((gaps[-1] - fs.quadratic(N_N[-1],parameters[0],parameters[1]) > 0 and parameters[1] > cutoff_gap)
                        or parameters[0] < 1e-3) else 'O'
        data[3] = data[3][:-1] + order if data[3][-1] in ['L','O'] else data[3] + order
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


