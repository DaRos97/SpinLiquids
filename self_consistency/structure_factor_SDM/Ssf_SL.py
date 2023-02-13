import numpy as np
import functions_SF as fs
from time import time as T
import getopt
import sys
#######################################################################     Inputs
list_ans = ['3x3','q0','cb1','cb2','oct']
DM_list = {'000':0, '005':0.05, '104':np.pi/3, '209':2*np.pi/3}
S_dic = {'50': 0.5, '36':(np.sqrt(3)+1)/2, '34':0.34, '30':0.3, '20':0.2}
#input arguments
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:", ['j2=','j3=','DM=','ans=','kpts='])
    txt_S = '50'
    J2 = 0
    J3 = 0
    DM = '000'
    ans = '3x3'
    pts = '49'
except:
    print("Error")
for opt, arg in opts:
    if opt in ['-S']:
        txt_S = arg
    if opt == '--j2':
        J2 = float(arg)
    if opt == '--j3':
        J3 = float(arg)
    if opt == '--DM':
        DM = arg.replace('.','')
        if DM not in DM_list.keys():
            print('Not computed DM angle')
            exit()
    if opt == '--ans':
        ans = arg
        if ans not in list_ans:
            print('Error in -ans choice')
            exit()
    if opt == '--kpts':
        pts = arg
#
S = S_dic[txt_S]
DM_angle = DM_list[DM]
PSG = 'SU2'if DM == '000' else 'TMD'
#Arguments
args = (1,J2,J3,ans,DM_angle,PSG)
########################################
########################################
print("Using arguments: ans-> ",ans," j2,j3 = ",J2,",",J3," Dm angle = ",DM," spin S = ",S)
filename = '../Data/SC_data/S'+txt_S+'/phi'+DM+'/'+pts+'/'+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
#
savenameZZ = "data_SF/SL_SFzz_"+ans+'_'+DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
savenameXY = "data_SF/SL_SFxy_"+ans+'_'+DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
Kx = 17     #points to compute in the SF BZ
Ky = 17
kxg = np.linspace(-8*np.pi/3,8*np.pi/3,Kx)
kyg = np.linspace(-4*np.pi/np.sqrt(3),4*np.pi/np.sqrt(3),Ky)
##
Nx = 37     #points for summation over BZ
Ny = 37
nxg = np.linspace(0,1,Nx)
nyg = np.linspace(0,1,Ny)

params = fs.import_data(ans,filename)

SFzz = np.zeros((Kx,Ky))
SFxy = np.zeros((Kx,Ky))
#
for i in range(Kx):
    for j in range(Ky):
        #K = np.array([kxg[i]*2*np.pi,(kxg[i]+2*kyg[j])*2*np.pi/np.sqrt(3)])
        K = np.array([kxg[i],kyg[j]])
        if not fs.EBZ(K):
            SFzz[i,j] = 'nan'
            SFxy[i,j] = 'nan'
            continue
        reszz = 0
        resxy = 0
        Ti = T()
        for ii in range(Nx):
            for ij in range(Ny):
                Q = np.array([nxg[ii]*2*np.pi,(nxg[ii]+nyg[ij])*2*np.pi/np.sqrt(3)])
                U1,X1,V1,Y1 = fs.M(Q,params,args)
                U2,X2,V2,Y2 = fs.M(-Q,params,args)
                U3,X3,V3,Y3 = fs.M(K-Q,params,args)
                U4,X4,V4,Y4 = fs.M(Q-K,params,args)
                #zz1
                A1 = np.einsum('ba,bc->ac',np.conjugate(X1),U4)
                B1 = np.einsum('ac,dc->ad',A1,np.conjugate(U4))
                C1 = np.einsum('ad,de->ae',B1,X1)
                reszz += np.einsum('aa',C1)
                D1 = np.einsum('ac,cd->ad',A1,Y1)
                E1 = np.einsum('ad,ed->ae',D1,np.conjugate(V4))
                reszz -= np.einsum('aa',E1)
                #zz2
                A1 = np.einsum('ab,cb->ac',V2,np.conjugate(Y3))
                B1 = np.einsum('ac,cd->ad',A1,Y3)
                C1 = np.einsum('ad,ed->ae',B1,np.conjugate(V2))
                reszz += np.einsum('aa',C1)
                D1 = np.einsum('ac,dc->ad',A1,np.conjugate(U2))
                E1 = np.einsum('ad,de->ae',D1,X3)
                reszz -= np.einsum('aa',E1)
                #xx+yy1
                A1 = np.einsum('ba,cb->ac',np.conjugate(X1),np.conjugate(Y3))
                B1 = np.einsum('ac,cd->ad',A1,Y3)
                C1 = np.einsum('ad,de->ae',B1,X1)
                resxy += 2*np.einsum('aa',C1)
                D1 = np.einsum('ac,cd->ad',A1,Y1)
                E1 = np.einsum('ad,de->ae',D1,X3)
                resxy += 2*np.einsum('aa',E1)
                #xx+yy2
                A1 = np.einsum('ab,bc->ac',V2,U4)
                B1 = np.einsum('ac,dc->ad',A1,np.conjugate(U4))
                C1 = np.einsum('ad,ed->ae',B1,np.conjugate(V2))
                resxy += 2*np.einsum('aa',C1)
                D1 = np.einsum('ac,dc->ad',A1,np.conjugate(U2))
                E1 = np.einsum('ad,ed->ae',D1,np.conjugate(V4))
                resxy += 2*np.einsum('aa',E1)
        SFzz[i,j] = 3/2*np.real(reszz)/(Nx*Ny)
        SFxy[i,j] = 3/2*np.real(resxy)/(Nx*Ny)
        print("Step ",i*Kx+j,"/",Kx*Ky)#," took ",T()-Ti)
np.save(savenameZZ,SFzz)
np.save(savenameXY,SFxy)
