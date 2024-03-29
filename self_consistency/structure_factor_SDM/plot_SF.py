import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import getopt

#input arguments
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:K:a:", ['DM=','type=','ph=','Nq='])
    S = 0.5
    K = 13
    ans = '16'
    DM = 0.0
    ph = 'SL'
    DM_type = 'uniform'
    Nq = 35
except:
    print("Error")
for opt, arg in opts:
    if opt in ['-S']:
        S = float(arg)
    if opt in ['-K']:
        K = int(arg)
    if opt in ['-a']:
        ans = arg
    if opt == '--DM':
        DM = float(arg)
    if opt == '--ph':
        ph = arg
    if opt == '--type':
        DM_type = arg
    if opt == '--Nq':
        Nq = int(arg)
################################
################################
if ph == 'SL':
    savenameSFzz = "data_SF/SL_SFzz_"+ans+'_'+'{:5.4f}'.format(S).replace('.','')+'_'+'{:5.4f}'.format(DM).replace('.','')+str(Nq)+'.npy'
    savenameSFxy = "data_SF/SL_SFxy_"+ans+'_'+'{:5.4f}'.format(S).replace('.','')+'_'+'{:5.4f}'.format(DM).replace('.','')+str(Nq)+'.npy'
else:
    exit()
    savenameSFzz = "data_SF/LRO_SFzz_"+a+'_'+'{:5.4f}'.format(DM).replace('.','')+'_'+'{:5.4f}'.format(S).replace('.','')+'_'+DM_type+'.npy'
    savenameSFxy = "data_SF/LRO_SFxy_"+a+'_'+'{:5.4f}'.format(DM).replace('.','')+'_'+'{:5.4f}'.format(S).replace('.','')+'_'+DM_type+'.npy'
SFzz = np.load(savenameSFzz)
SFxy = np.load(savenameSFxy)
Kx,Ky = SFzz.shape

kxg = np.linspace(-8*np.pi/3,8*np.pi/3,Kx)
kyg = np.linspace(-4*np.pi/np.sqrt(3),4*np.pi/np.sqrt(3),Ky)
K = np.zeros((2,Kx,Ky))
for i in range(Kx):
    for j in range(Ky):
        K[:,i,j] = np.array([kxg[i],kyg[j]])
#
plt.figure(figsize=(12,12))
#plt.subplot(1,2,1)
plt.gca().set_aspect('equal')
#title = 'S='+S+', DM='+DM+', (J2,J3)=('+str(J2)+','+str(J3)+'), ansatz: '+ans
#plt.title(title)
#
plt.plot(fs.X1,fs.fu1(fs.X1),'k-')
plt.hlines(2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fu3(fs.X2),'k-')
plt.plot(fs.X1,fs.fd1(fs.X1),'k-')
plt.hlines(-2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fd3(fs.X2),'k-')
#
plt.plot(fs.X3,fs.Fu1(fs.X3),'k-')
plt.hlines(4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fu3(fs.X4),'k-')
plt.plot(fs.X3,fs.Fd1(fs.X3),'k-')
plt.hlines(-4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fd3(fs.X4),'k-')
#
plt.scatter(K[0],K[1],c=SFxy,cmap = cm.get_cmap('plasma_r'))
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20)
plt.rcParams.update({
    "text.usetex": True,
#    "font.family": "Helvetica"
})
plt.axis('off')
plt.xlabel(r'$K_x$',size=15)
plt.ylabel(r'$K_y$',size=15,rotation='horizontal')
plt.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
plt.show()
exit()
###
###
###
###
plt.figure()
#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()
title = 'coool'#ans+'_'+DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+')'
plt.suptitle(title)
#plt.axis('off')
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
