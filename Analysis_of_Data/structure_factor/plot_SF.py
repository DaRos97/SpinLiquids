import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import getopt

list_ans = ['3x3_1','q0_1','cb1','cb2','oct']
DM_list = {'000':0, '006':np.pi/48, '013':2*np.pi/48, '019':3*np.pi/48, '026':4*np.pi/48, '032':5*np.pi/48, '039':6*np.pi/48, '209':2*np.pi/3}
#input arguments
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:", ['j2=','j3=','DM=','ans=','kpts='])
    S = 0.5
    txt_S = '05'
    J2 = 0
    J3 = 0
    DM = '000'
    ans = '3x3_1'
    pts = '13'
except:
    print("Error")
for opt, arg in opts:
    if opt in ['-S']:
        txt_S = arg
        if txt_S not in ['05','03']:
            print('Error in -S argument')
            exit()
        else:
            S = 0.5 if txt_S == '05' else 0.366         #####CHECK
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
    if opt == '-kpts':
        pts = arg
print("Using arguments: ans-> ",ans," j2,j3 = ",J2,",",J3," Dm angle = ",DM," spin S = ",S)

ph = 0 #0->LRO, 1->SL
if ph:
    savenameZZ = "SFs/SFzz_"+ans+'_'+DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
    savenameXY = "SFs/SFxy_"+ans+'_'+DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
else:
    savenameZZ = "LRO_SSF/SSFzz_"+ans+'_'+DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
    savenameXY = "LRO_SSF/SSFxy_"+ans+'_'+DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
SFzz = np.load(savenameZZ)
SFxy = np.load(savenameXY)
Kx,Ky = SFzz.shape

kxg = np.linspace(-8*np.pi/3,8*np.pi/3,Kx)
kyg = np.linspace(-4*np.pi/np.sqrt(3),4*np.pi/np.sqrt(3),Ky)
K = np.zeros((2,Kx,Ky))
for i in range(Kx):
    for j in range(Ky):
        K[:,i,j] = np.array([kxg[i],kyg[j]])
#
plt.figure()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
title = ans+'_'+DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+')'
plt.title(title)
plt.axis('off')
plt.subplot(2,2,1)
plt.title(title+'--Sxy')
#hexagons
plt.plot(fs.X1,fs.fu1(fs.X1),'k-')
plt.hlines(2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fu3(fs.X2),'k-')
plt.plot(fs.X1,fs.fd1(fs.X1),'k-')
plt.hlines(-2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fd3(fs.X2),'k-')

plt.plot(fs.X3,fs.Fu1(fs.X3),'k-')
plt.hlines(4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fu3(fs.X4),'k-')
plt.plot(fs.X3,fs.Fd1(fs.X3),'k-')
plt.hlines(-4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fd3(fs.X4),'k-')
#
plt.scatter(K[0],K[1],c=SFxy,cmap = cm.get_cmap('plasma_r'))
plt.colorbar()
#
plt.subplot(2,2,2)
plt.title("Szz")
#hexagons
plt.plot(fs.X1,fs.fu1(fs.X1),'k-')
plt.hlines(2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fu3(fs.X2),'k-')
plt.plot(fs.X1,fs.fd1(fs.X1),'k-')
plt.hlines(-2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fd3(fs.X2),'k-')

plt.plot(fs.X3,fs.Fu1(fs.X3),'k-')
plt.hlines(4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fu3(fs.X4),'k-')
plt.plot(fs.X3,fs.Fd1(fs.X3),'k-')
plt.hlines(-4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fd3(fs.X4),'k-')
#
plt.scatter(K[0],K[1],c=SFzz,cmap = cm.get_cmap('plasma_r'))
plt.colorbar()
#
plt.subplot(2,2,3)
plt.title(title+'--S_tot')
#hexagons
plt.plot(fs.X1,fs.fu1(fs.X1),'k-')
plt.hlines(2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fu3(fs.X2),'k-')
plt.plot(fs.X1,fs.fd1(fs.X1),'k-')
plt.hlines(-2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fd3(fs.X2),'k-')

plt.plot(fs.X3,fs.Fu1(fs.X3),'k-')
plt.hlines(4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fu3(fs.X4),'k-')
plt.plot(fs.X3,fs.Fd1(fs.X3),'k-')
plt.hlines(-4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fd3(fs.X4),'k-')
#
plt.scatter(K[0],K[1],c=SFxy+SFzz,cmap = cm.get_cmap('plasma_r'))
plt.colorbar()


plt.show()
