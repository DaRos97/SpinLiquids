import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import getopt
import functions as fs
from scipy.interpolate import RectBivariateSpline as RBS

#parameters are: ansatz, j2,j3, DM angle, Spin
list_ans = ['3x3','q0','cb1','cb1_nc','cb2','oct']
DM_list = {'000':0, '005':0.05, '104':np.pi/3, '209':2*np.pi/3}
S_dic = {'50':0.5,'36':(np.sqrt(3)-1)/2,'34':0.34,'30':0.3,'20':0.2}
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:", ['j2=','j3=','DM=','ans=','Nmax='])
    txt_S = '50'
    J2 = 0
    J3 = 0
    txt_DM = '000'
    ans = '3x3'
    N_max = 13
    type_of_ans = 'SU2'
except:
    print("Error in input parameters")
    exit()
for opt, arg in opts:
    if opt in ['-S']:
        txt_S = arg
        if txt_S not in S_dic.keys():
            print('Error in -S argument')
            exit()
    if opt == '--j2':
        J2 = float(arg)
    if opt == '--j3':
        J3 = float(arg)
    if opt == '--DM':
        txt_DM = arg.replace('.','')
        if DM not in DM_list.keys():
            print('Not computed DM angle')
            exit()
        type_of_ans = 'SU2' if txt_DM in ['000','104','209'] else 'TMD'
    if opt == '--ans':
        ans = arg 
        if ans not in list_ans:
            print('Error in -ans choice')
            exit()
    if opt == '--Nmax':
        N_max = int(arg)
S = S_dic[txt_S]
DM = DM_list[txt_DM]

print("Using arguments: ans-> ",ans," j2,j3 = ",J2,",",J3," Dm angle = ",txt_DM," spin S = ",S)
#import data
m = 6
arg_data = (ans,txt_DM,J2,J3,txt_S,N_max)
data = fs.get_data(arg_data)
#construct M matrices for each band -> 6

#compute berry curvature for each n with formula (24). there are derivatives

#compute chern number by integrating over the BZ. Use RBS

gridx = gridy = N_max
step_k = 1e-2
arg_1 = (ans,DM,J2,J3,txt_S,N_max)
arg_2 = (gridx,gridy,type_of_ans,step_k)
arg_berry = (arg_1,data,arg_2)
#K-grid
K_grid = np.zeros((2,gridx,gridy))
kxg = np.linspace(0,1,gridx)
kyg = np.linspace(0,1,gridy)
for i in range(gridx):
    for j in range(gridy):
        K_grid[0,i,j] = kxg[i]*2*np.pi
        K_grid[1,i,j] = (kxg[i]+kyg[j])*2*np.pi/np.sqrt(3)
#berry
berry = np.zeros((gridx,gridy,2*m))
for ikx in range(gridx):
    for iky in range(gridy):
        kx,ky = K_grid[:,ikx,iky]
        berry[ikx,iky] = fs.compute_berry(kx,ky,arg_berry)
#interpolate and integrate
C = np.ndarray(2*m)
for n in range(2*m):
    C[n] = RBS(np.linspace(0,1,gridx),np.linspace(0,1,gridy),berry[:,:,n]).integral(0,1,0,1)

print("Chern numbers: ",C)
















