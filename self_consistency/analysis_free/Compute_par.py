import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import functions as fs
import getopt
from matplotlib import cm
from matplotlib.lines import Line2D

Color = {'15':  ['blue','aqua','dodgerblue'],          #q=0      -> dodgerblue
         '16': ['red','y','orangered'],             #3x3      -> orangered
         '17':  ['pink','pink','gray'],           #cb2      -> magenta
         '18':  ['k','k','gray'],              #oct     -> orange
         '19':  ['gray','gray','magenta'],                #-> silver
         '20':  ['forestgreen','lime','limegreen'],    #cb1  -> forestgreen
         'labels':  ['k','k','k']
         }
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:K:a:", ['j2=','j3=','DM=','par='])
    txt_S = '50'
    phi_t = '000'
    K = '13'
    ans = '15'
    par_name = 'A1'
    J2 = 0
    J3 = 0
except:
    print("Error in inputs")
    exit()
for opt, arg in opts:
    if opt in ['-S']:
        txt_S = arg
    if opt in ['-K']:
        K = arg
    if opt == '--DM':
        phi_t = arg
    if opt in '-a':
        ans = arg
    if opt == '--par':
        par_name = arg
    if opt == '--j2':
        J2 = float(arg)
    if opt == '--j3':
        J3 = float(arg)
#
S_dic = {'50':0.5,'36':0.36,'30':0.3,'20':0.2}
S = S_dic[txt_S]
phi_label = {'000':0, '005':0.05, '104':np.pi/3, '209':np.pi/3*2}
phi = phi_label[phi_t]
#dirname = '../Data/SC_data/final_'+txt_S+'_'+phi_t+'/' 
dirname = '../../Data/self_consistency/S'+txt_S+'/phi'+phi_t+'/'+K+'/' 
#dirname = '../../Data/self_consistency/440_small_S50/phi000/13/' 
csvname = 'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
csvfile = dirname+csvname

with open(csvfile, 'r') as f:
    lines = f.readlines()
N = (len(lines)-1)//2 + 1
for i in range(N):
    head = lines[2*i].split(',')
    head[-1] = head[-1][:-1]
    data = lines[2*i+1].split(',')
    if data[0] != ans:
        continue
    head = lines[2*i].split(',')
    head[-1] = head[-1][:-1]
    data = lines[2*i+1].split(',')
    break
print(head)
print(data)
res = fs.compute_par(data,head,par_name,S,phi,int(K))
print("par ",par_name," value is ",res)






