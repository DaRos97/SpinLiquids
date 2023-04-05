import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from matplotlib import cm
import getopt
import sys
import os
from tqdm import tqdm

#input arguments
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:K:a:", ['U=','V='])
    S = '50'
    DM = '209'
    ans = '15'
    K = '13'
    U = 50
    V = 4
except:
    print("Error")
for opt, arg in opts:
    if opt in ['-S']:
        S = arg
    if opt in ['-K']:
        K = arg
    if opt in ['-a']:
        ans = arg
    if opt == '--U':
        U = float(arg)
    if opt == '--V':
        V = float(arg)

dirname = '../../Data/self_consistency/UV/S'+S+'/'+K+'/'
#dirname = '../../Data/self_consistency/test/'+K+'/'
filename = dirname+'U_V=('+'{:5.4f}'.format(U).replace('.','')+'_'+'{:5.4f}'.format(V).replace('.','')+').csv'
print(filename)
J2 = U
J3 = V
savenameSFzz = "data_SF/SL_SFzz_"+ans+'_'+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+'_'+S+'.npy'
savenameSFxy = "data_SF/SL_SFxy_"+ans+'_'+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+'_'+S+'.npy'
command_plot = 'python plot_SF.py -S '+S+' -K '+K+' -a '+ans+' --U '+str(J2)+' --V '+str(J3)+' --ph SL'

if not os.path.isfile(filename):
    print(J2,J3,ans," values are not valid or the point was not computed")
if os.path.isfile(savenameSFzz):
    os.system(command_plot)
    printed = True
else:
    printed = False
    print("Computing it ....")
#
if printed:
    exit()
#
print("Using arguments: ans-> ",ans," Dm angle = ",DM," spin S = ",S," U = ",J2," V = ",J3)
#import data from file
t1 = 1
t2 = t1*0.145
t3 = t1*0.08
Ui = t1*50
Uf = t1*150
UV_pts = 20
U_list = np.linspace(Ui,Uf,UV_pts)
V_list = []
for u in U_list:
    V_list.append(np.linspace(0.08*u,0.15*u,UV_pts))
params,J = fs.import_data(ans,filename)
#Arguments
S_dic = {'50':0.5,'36':(np.sqrt(3)-1)/2,'34':0.34,'30':0.3,'20':0.2}
S_val = S_dic[S]
DM_dic = {'000':0,'005':0.05,'104':np.pi/3,'209':np.pi*2/3}
DM_val = DM_dic[DM]
DM1 = DM_val;   DM2 = 0;    DM3 = 2*DM_val
t1 = np.exp(-1j*DM1);    t1_ = np.conjugate(t1)
t2 = np.exp(-1j*DM2);    t2_ = np.conjugate(t2)
t3 = np.exp(-1j*DM3);    t3_ = np.conjugate(t3)
Tau = (t1,t1_,t2,t2_,t3,t3_)

#
p1 = 0 if ans in fs.ans_p0 else 1
m = fs.Mm[p1]
#######################################################################
Kx = 13     #points for summation over BZ
Ky = 13
######
f = np.sqrt(3)/4
D = np.array([  [1/2,-1/4,1/4,0,-3/4,-1/4],
                [0,f,f,2*f,3*f,3*f]])
if m == 6:
    Kxg = np.linspace(-np.pi,np.pi,Kx)
    Kyg = np.linspace(-np.pi/np.sqrt(3),np.pi/np.sqrt(3),Ky)
else:
    Kxg = np.linspace(-4/3*np.pi,4/3*np.pi,Kx)
    Kyg = np.linspace(-2*np.pi/np.sqrt(3),2*np.pi/np.sqrt(3),Ky)

##
Nx = 17     #points to compute in BZ (Q)
Ny = 17
Qxg = np.linspace(-8*np.pi/3,8*np.pi/3,Nx)
Qyg = np.linspace(-4*np.pi/np.sqrt(3),4*np.pi/np.sqrt(3),Ny)

#Result store
SFzz = np.zeros((Nx,Ny))
SFxy = np.zeros((Nx,Ny))
#
if J[0] > 0:
    dic_P = {'14':(0,0),'15':(1,1),'16':(0,0),'17':(1,1),'18':(1,1),'19':(1,1,0,0),'20':(1,1,0,0)}
else:
    dic_P = {'14':(0,0),'15':(1,1),'16':(0,0),'17':(1,1),'18':(1,1),'19':(0,0,1,0),'20':(0,0,1,0)}

#
args = (Tau,S_val,J,ans,dic_P[ans])

#Compute Xi(Q) for Q in BZ
for xx in tqdm(range(Nx*Ny)):
    ii = xx//Nx
    ij = xx%Ny
    Q = np.array([Qxg[ii],Qyg[ij]])
    if not fs.EBZ(Q):
        SFzz[ii,ij] = np.nan
        SFxy[ii,ij] = np.nan
        continue
    #
    delta = np.zeros((m,m),dtype=complex)
    for u in range(m):
        for g in range(m):
            delta[u,g] = np.exp(1j*np.dot(Q,D[:,g]-D[:,u]))
    #
    resxy = 0
    #summation over BZ
    for x in range(Kx*Ky):
        i = x//Kx
        j = x%Ky
        #
        K__ = np.array([Kxg[i],Kyg[j]])
        if m == 3 and not fs.BZ(K__):
            continue
        U1,X1,V1,Y1 = fs.M(K__,params,args,'nor')
        U2,X2,V2,Y2 = fs.M(-K__,params,args,'nor')
        U3,X3,V3,Y3 = fs.M(Q+K__,params,args,'nor')
        U4,X4,V4,Y4 = fs.M(-K__-Q,params,args,'nor')
        ##############################################
        temp1 = np.einsum('ua,ga->ug',np.conjugate(X1),X1) * np.einsum('ua,ga->ug',np.conjugate(Y4),Y4)
        temp2 = np.einsum('ua,ga->ug',np.conjugate(X1),Y1) * np.einsum('ua,ga->ug',np.conjugate(Y4),X4)
        temp3 = np.einsum('ua,ga->ug',V2,np.conjugate(V2)) * np.einsum('ua,ga->ug',U3,np.conjugate(U3))
        temp4 = np.einsum('ua,ga->ug',V2,np.conjugate(U2)) * np.einsum('ua,ga->ug',U3,np.conjugate(V3))
        temp = (temp1 + temp2 + temp3 + temp4) * delta
        resxy += temp.ravel().sum()
    #
    SFxy[ii,ij] = np.real(resxy)/(Kx*Ky)
#
np.save(savenameSFzz,SFzz)
np.save(savenameSFxy,SFxy)
print("Finished")
##################################
print("\n\nPlotting...")

os.system(command_plot)




















