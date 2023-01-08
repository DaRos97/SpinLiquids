import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import getopt
import functions as fs
from scipy.optimize import curve_fit
import csv

#Takes the data in Data/final_SDM and converts the "Converge" value from True to TrueL (SL) or TrueO (LRO) depending on the final value of the gap
#In order to work, final_S_DM has to contain all the ansatze, so that then we can compare it in the phase diagram using effort_DM/plot.py
cutoff_gap = 1e-6
#parameters are: ansatz, j2,j3, DM angle, Spin
list_ans = ['1a','1b','1c','1d','1e','1f']
S_max = 0.5
DM_max = 0.15
S_pts = 30
DM_pts = 30
S_list = np.linspace(0.01,S_max,S_pts,endpoint=True)
DM_list = np.linspace(0,DM_max,DM_pts,endpoint=True)

data_dirname = '../../Data/SDM/final_SDM/'
##### 
type_of_ans = 'SDM'
for filename in os.listdir(data_dirname):
    data_name = data_dirname + filename
    with open(data_name, 'r') as f:
        lines = f.readlines()
    #
    N__ = (len(lines)-1)//2 + 1
    print(filename)
    for i in range(N__):
        data = lines[i*2+1].split(',')
        data[-1] = data[-1][:-1]                #remove final \n
        header = lines[i*2].split(',')
        header[-1] = header[-1][:-1]
        S = float(data[1])
        DM = float(data[2])
        #
        if data[3][0] == 'F':# or data[3][-1] in ['L','O']:
            continue
        # Go to data directory and compute gap scaling up to this point
        ans = data[0]
        N_reference = [13,25,37]
        #list directories to find list of Ns
        Data_dir = '../../Data/SDM/'
        Ns = []
        params = {}
        #get all parameters at all Ns
        for N_dir in os.listdir(Data_dir):
            if N_dir == 'final_SDM':
                continue
            if int(N_dir) not in N_reference:
                continue
            for files in os.listdir(Data_dir+N_dir+'/'):
                with open(Data_dir+N_dir+'/'+files, 'r') as f:
                    lines_1 = f.readlines()
                N_ = (len(lines_1)-1)//2 +1
                for i_ in range(N_):
                    lines_2 = lines_1[i_*2+1].split(',')
                    if lines_2[0] == ans and np.abs(float(lines_2[1])-S) < 1e-6 and np.abs(float(lines_2[2])-DM) < 1e-6 and lines_2[3][0] == 'T':
                        Ns.append(int(N_dir))
                        params[N_dir] = []
                        for p in lines_2[6:]:
                            params[N_dir].append(float(p))
        #compute Gap from interpolation
        gaps = []
        N_N = np.sort(Ns)
        for nnn_ in N_N:
            DM_val = DM_list[list(DM_list).index(DM)]
            gaps.append(fs.find_gap(params[str(nnn_)],nnn_,[ans,DM_val,0,0,'0',type_of_ans]))
        try:
            parameters, covariance = curve_fit(fs.quadratic, N_N, gaps, p0=(1,0))
        except:
            print("Not fitted ",ans," at ",S,DM)
            fitted_ = fs.linear(np.linspace(N_N[0],N_N[-1],100),parameters[0],parameters[1])
            plt.figure()
            plt.plot(N_N,gaps,'ro')
            plt.plot(np.linspace(N_N[0],N_N[-1],100),fitted_,'g-')
            plt.hlines(parameters[1],N_N[0],N_N[-1],color='green',linestyles='--')
            plt.title(ans+"__"+str(S)+"_"+str(DM))
            plt.show()
        #Change True -> TrueL or TrueO
        order = 'L' if parameters[1] > 1e-3 else 'O'
        #((gaps[-1] - fs.quadratic(N_N[-1],parameters[0],parameters[1]) > 0 and parameters[1] > cutoff_gap) or parameters[0] < 1e-3) else 'O'
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


