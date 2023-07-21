import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import getopt
from scipy.interpolate import RectBivariateSpline as RBS

#input arguments
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:K:a:", ['DM=','j2=','j3=','ph=','Nq='])
    S = '50'
    DM = '000'
    ans = '15'
    K = '13'
    J2 = J3 = 0
    ph = 'LRO'
    Nq = 17
except:
    print("Error")
for opt, arg in opts:
    if opt in ['-S']:
        S = arg
    if opt in ['-K']:
        K = arg
    if opt in ['-a']:
        ans = arg
    if opt == '--DM':
        DM = arg
    if opt == '--j2':
        J2 = float(arg)
    if opt == '--j3':
        J3 = float(arg)
    if opt == '--ph':
        ph = arg
    if opt == '--Nq':
        Nq = int(arg)
################################
################################
Nx = Nq
if ph == 'SL':
    savenameSFzz = "data_SF/SL_SFzz_"+ans+'_'+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+'_'+S+DM+str(Nx)+'.npy'
    savenameSFxy = "data_SF/SL_SFxy_"+ans+'_'+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+'_'+S+DM+str(Nx)+'.npy'
else:
    savenameSFzz = "data_SF/LRO_SFzz_"+ans+'_'+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+S+DM+str(Nq)+'.npy'
    savenameSFxy = "data_SF/LRO_SFxy_"+ans+'_'+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+S+DM+str(Nq)+'.npy'
SFzz = np.load(savenameSFzz)
SFxy = np.load(savenameSFxy)
Kx,Ky = SFxy.shape

kxg = np.linspace(-8*np.pi/3,8*np.pi/3,Kx)
kyg = np.linspace(-4*np.pi/np.sqrt(3),4*np.pi/np.sqrt(3),Ky)
K = np.zeros((2,Kx,Ky))
for i in range(Kx):
    for j in range(Ky):
        K[:,i,j] = np.array([kxg[i],kyg[j]])
if ph == 'SL':
    #
    plt.figure(figsize=(12,12))
    plt.rcParams.update({
        "text.usetex": True,
    #    "font.family": "Helvetica"
    })
    #plt.subplot(1,2,1)
    plt.gca().set_aspect('equal')
    #title = 'S='+S+', DM='+DM+', (J2,J3)=('+str(J2)+','+str(J3)+'), ansatz: '+ans
    #plt.title(title)
    #
    #plt.scatter(K[0],K[1],c=SFxy+SFzz,cmap = cm.get_cmap('plasma_r'),s=150)
    plt.scatter(K[0],K[1],c=SFxy,cmap = cm.get_cmap('plasma_r'),s=70)
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
    #Middle lines
    plt.plot(fs.X2,np.sqrt(3)*fs.X2,'k--')
    plt.plot(fs.X2,-np.sqrt(3)*fs.X2,'k--')
    plt.plot(fs.X1,np.sqrt(3)*fs.X1,'k--')
    plt.plot(fs.X1,-np.sqrt(3)*fs.X1,'k--')
    plt.hlines(0, -8*np.pi/3, -4*np.pi/3, color = 'k', linestyles = 'dashed')
    plt.hlines(0, 4*np.pi/3, 8*np.pi/3, color = 'k', linestyles = 'dashed')
    #High symmetry points
    ccc = 40
    dd = 0.1
    sss = 20
    plt.scatter(0,0,s=sss,color='k')
    plt.text(0+dd,0+dd,r'$\Gamma$',size=ccc)
    plt.scatter(2*np.pi/3,2*np.pi/np.sqrt(3),s=sss,color='k')
    plt.text(2*np.pi/3+2*dd,2*np.pi/np.sqrt(3)-dd,r'$K$',size=ccc)
    plt.scatter(np.pi,np.pi/np.sqrt(3),s=sss,color='k')
    plt.text(np.pi+dd,np.pi/np.sqrt(3)+dd,r'$M$',size=ccc)
    #
    #cbar = plt.colorbar()
    #cbar.ax.tick_params(labelsize=20)
    plt.axis('off')
    plt.xlabel(r'$K_x$',size=15)
    plt.ylabel(r'$K_y$',size=15,rotation='horizontal')
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    
    if 0:#input('Plot (\'s\') or save (any)?')=='s':
        plt.show()
    else:
        namefile = '../../../../Figs_SB_paper/SSF_SL_'+ans+'_'+S+'_'+DM+'_'+'{:3.2f}'.format(J2).replace('.','')+'_'+'{:3.2f}'.format(J3).replace('.','')+'.pdf'
        plt.savefig(namefile,bbox_inches='tight')
    exit()
else:
    #
    plt.figure(figsize=(12,12))
    plt.rcParams.update({
        "text.usetex": True,
    #    "font.family": "Helvetica"
    })
    #plt.subplot(1,2,1)
    plt.gca().set_aspect('equal')
    #title = 'S='+S+', DM='+DM+', (J2,J3)=('+str(J2)+','+str(J3)+'), ansatz: '+ans
    #plt.title(title)
    #
    #BZ
    plt.plot(fs.X1,fs.fu1(fs.X1),'k-')
    plt.hlines(2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')#, linestyles = 'dashed')
    plt.plot(fs.X2,fs.fu3(fs.X2),'k-')
    plt.plot(fs.X1,fs.fd1(fs.X1),'k-')
    plt.hlines(-2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')#, linestyles = 'dashed')
    plt.plot(fs.X2,fs.fd3(fs.X2),'k-')
    #EBZ
    plt.plot(fs.X3,fs.Fu1(fs.X3),'k-')
    plt.hlines(4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
    plt.plot(fs.X4,fs.Fu3(fs.X4),'k-')
    plt.plot(fs.X3,fs.Fd1(fs.X3),'k-')
    plt.hlines(-4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
    plt.plot(fs.X4,fs.Fd3(fs.X4),'k-')
    #Middle lines
    plt.plot(fs.X2,np.sqrt(3)*fs.X2,'k--')
    plt.plot(fs.X2,-np.sqrt(3)*fs.X2,'k--')
    plt.plot(fs.X1,np.sqrt(3)*fs.X1,'k--')
    plt.plot(fs.X1,-np.sqrt(3)*fs.X1,'k--')
    plt.hlines(0, -8*np.pi/3, -4*np.pi/3, color = 'k', linestyles = 'dashed')
    plt.hlines(0, 4*np.pi/3, 8*np.pi/3, color = 'k', linestyles = 'dashed')
    #
    plt.scatter(K[0],K[1],c=SFxy+SFzz,cmap = cm.get_cmap('plasma_r'),s=260)
    #High symmetry points
    ccc = 40
    dd = 0.1
    sss = 20
    plt.scatter(0,0,s=sss,color='k')
    plt.text(0+dd,0+dd,r'$\Gamma$',size=ccc)
    plt.scatter(2*np.pi/3,2*np.pi/np.sqrt(3),s=sss,color='k')
    plt.text(2*np.pi/3+2*dd,2*np.pi/np.sqrt(3)-dd,r'$K$',size=ccc)
    plt.scatter(np.pi,np.pi/np.sqrt(3),s=sss,color='k')
    plt.text(np.pi+dd,np.pi/np.sqrt(3)+dd,r'$M$',size=ccc)
    #cbar = plt.colorbar()
    #cbar.ax.tick_params(labelsize=20)
    plt.axis('off')
    plt.xlabel(r'$K_x$',size=15)
    plt.ylabel(r'$K_y$',size=15,rotation='horizontal')
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

    if 1:
        namefile = '../../../../Figs_SB_paper/SSF_LRO_'+ans+'_'+S+'_'+DM+'_'+'{:3.2f}'.format(J2).replace('.','')+'_'+'{:3.2f}'.format(J3).replace('.','')+'.pdf'
        plt.savefig(namefile,bbox_inches='tight')
    else: 
        plt.show()
    
    exit()









###
plt.subplot(2,2,1)
plt.gca().set_aspect('equal')
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
plt.colorbar()
###
plt.subplot(2,2,2)
plt.gca().set_aspect('equal')
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

plt.scatter(K[0],K[1],c=SFzz,cmap = cm.get_cmap('plasma_r'))
plt.colorbar()
###
plt.subplot(2,2,3)
plt.gca().set_aspect('equal')
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

plt.scatter(K[0],K[1],c=SFxy+SFzz,cmap = cm.get_cmap('plasma_r'))
plt.colorbar()

plt.show()


exit()


############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
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
