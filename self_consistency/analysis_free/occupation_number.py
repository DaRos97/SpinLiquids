import functions_occupation_number as fs
import numpy as np
import sys
import getopt
from tqdm import tqdm
import matplotlib.pyplot as plt 
from matplotlib import cm

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:K:a:", ['j2=','j3=','DM=','ind='])
    txt_S = '50'
    phi_t = '000'
    K = '13'
    ans = '15'
    J2 = 0
    J3 = 0
    index_m = 0
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
    if opt == '--ind':
        index_m = int(arg)
    if opt == '--j2':
        J2 = float(arg)
    if opt == '--j3':
        J3 = float(arg)

S_dic = {'50':0.5,'36':0.36,'30':0.3,'20':0.2}
S = S_dic[txt_S]
phi_label = {'000':0, '005':0.05, '104':np.pi/3, '209':np.pi/3*2}
phi = phi_label[phi_t]
dirname = '../../Data/self_consistency/S'+txt_S+'/phi'+phi_t+'/'+K+'/' 
#
J2i = J3i = -0.3 
J2f = J3f = 0.3 
Jpts = 7                ######
J= []
for i in range(Jpts):
    for j in range(Jpts):
        J.append((J2i+(J2f-J2i)/(Jpts-1)*i,J3i+(J3f-J3i)/(Jpts-1)*j))
Res = {}
list_ans = ['15']#,'16','19','20'] 
for ans in list_ans: 
    mm = 3 if ans in fs.ans_p0 else 6
    Res[ans] = np.zeros((Jpts,Jpts,mm))
    print(ans)
    for ind_J in tqdm(range(Jpts**2)):
        J2,J3 = J[ind_J]
        csvname = 'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
        csvfile = dirname+csvname
        for index_m in range(mm):
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
                #
                n2_i, n_i = fs.compute_fluc(data,head,index_m,S,phi,int(K))
                print(n2_i,n_i,n2_i-n_i**2)
                exit()
                Res[ans][ind_J//Jpts,ind_J%Jpts,index_m] = n2_i-n_i**2
                break
            if Res[ans][ind_J//Jpts,ind_J%Jpts,index_m] == 0:
                Res[ans][ind_J//Jpts,ind_J%Jpts,index_m] = np.nan
#
Grid = Jpts
Ji = -0.3
Jf = -Ji
J2 = np.linspace(Ji,Jf,Grid)
J3 = np.linspace(Ji,Jf,Grid)
X,Y = np.meshgrid(J2,J3)
for ans in list_ans:
    mm = 3 if ans in fs.ans_p0 else 6
    yy = 5 if mm == 3 else 10
    fig = plt.figure(figsize=(17,yy))
    plt.title(ans)
    plt.axis('off')
    for index_m in range(mm):
        rows = 1 if mm == 3 else 2
        plt.subplot(rows,3,index_m+1)
        plt.gca().set_aspect('equal')
        plt.contourf(X,Y,Res[ans][:,:,index_m].T)
        plt.colorbar()
    plt.show()
















