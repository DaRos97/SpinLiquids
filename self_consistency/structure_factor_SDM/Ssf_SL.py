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
    opts, args = getopt.getopt(argv, "S:K:a:", ['DM=','type=','Nq='])
    S = 0.5
    DM = 0.0
    ans = '15'
    K = '13'
    DM_type = 'uniform'
    Nq = 35
except:
    print("Error")
for opt, arg in opts:
    if opt in ['-S']:
        S = float(arg)
    if opt in ['-K']:
        K = arg
    if opt in ['-a']:
        ans = arg
    if opt == '--DM':
        DM = float(arg)
    if opt == '--type':
        DM_type = arg
    if opt == '--Nq':
        Nq = int(arg)

Nx = Nq     #points to compute in BZ (Q)
Ny = Nx
dirname = '../../Data/self_consistency/SDM/'+DM_type+'/'+K+'/'
#dirname = '../../Data/self_consistency/test/'+K+'/'
filename = dirname + 'S_DM=('+'{:5.4f}'.format(S).replace('.','')+'_'+'{:5.4f}'.format(DM).replace('.','')+').csv'
savenameSFzz = "data_SF/SL_SFzz_"+ans+'_'+'{:5.4f}'.format(S).replace('.','')+'_'+'{:5.4f}'.format(DM).replace('.','')+str(Nq)+'.npy'
savenameSFxy = "data_SF/SL_SFxy_"+ans+'_'+'{:5.4f}'.format(S).replace('.','')+'_'+'{:5.4f}'.format(DM).replace('.','')+str(Nq)+'.npy'
command_plot = 'python plot_SF.py -S '+str(S)+' -K '+K+' --DM '+str(DM)+' -a '+ans+' --ph SL --type '+DM_type+' --Nq '+str(Nq)

if not os.path.isfile(filename):
    print(S,DM,ans," values are not valid or the point was not computed")
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
print("Using arguments: ans-> ",ans," Dm angle = ",DM," spin S = ",S)
#import data from file
params = fs.import_data(ans,filename)
#Arguments
t1 = np.exp(-1j*DM);    t1_ = np.conjugate(t1)
Tau = (t1,t1_)
#
args = (Tau,S,ans)
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
Qxg = np.linspace(-8*np.pi/3,8*np.pi/3,Nx)
Qyg = np.linspace(-4*np.pi/np.sqrt(3),4*np.pi/np.sqrt(3),Ny)

#Result store
SFzz = np.zeros((Nx,Ny))
SFxy = np.zeros((Nx,Ny))
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




















