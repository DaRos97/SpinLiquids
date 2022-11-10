import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import getopt
import tqdm
#input arguments
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "g:o:", ["theta=","phi=","compute_new"])
    order = ''
    gauge = ''
    theta = 0
    phi = 0
    compute_new = False
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-g']:
        DM = arg
    if opt in ['-o']:
        order = arg
    if opt == "--theta":
        theta = np.pi*float(arg)
    if opt == "--phi":
        phi = np.pi*float(arg)
    if opt == "--compute_new":
        compute_new = True

list_orders = ['q0','3x3','ferro','cb1','cb2','oct']
if order not in list_orders:
    print('unknown regular order. Chose one from:')
    for ord_ in list_orders:
        print(ord_)
    exit()
DM_list = ['1','0','2']
if DM not in DM_list:
    print('DM argument not good, choose either 0,1 or 2')
gauge_trsf = int(DM)
#
print('Computing spin structure factor of ',order,' gauged ',DM,' time(s)')
#rotation angle of all the directions, given as [theta,phi]
angles = [theta,phi]
#number of unit cells (UC) and number of points in each direction of the BZ
UC = 6
Nx = 17
Ny = 17
#Initialization of lattice with spin values
#Order of spins in lattice is, using up-pointing triangles as (lattice) unit cells, 0->lowleft - 1->lowright - 2->upcenter
#Also, littece unit vectors are a1 = (1,0) - a2 = (1/2,sqrt(3)/2)
lattice = fs.lattice(order,gauge_trsf,angles)
#K-points in the extended BZ (with redountant ones)
K = np.ndarray((2,Nx,Ny))
#Structure factors xy and z
Sxy = np.zeros((Nx,Ny))
Sz = np.zeros((Nx,Ny))
filename = 'data/' + order + '_' + DM 
if compute_new:
    for i in tqdm.tqdm(range(Nx)):
        for j in range(Ny):
            #Consider all points in an extended BZ and also some redountant ones
            K[0,i,j] = -8*np.pi/3 + 16/3*np.pi/(Nx-1)*i
            K[1,i,j] = -4*np.pi/np.sqrt(3) + 8*np.pi/np.sqrt(3)/(Ny-1)*j
            if fs.is_k_inside_BZ(K[:,i,j]):    #exclude points outside the extended BZ
                Sxy[i,j], Sz[i,j] = fs.structure_factor(K[:,i,j],lattice,UC)
            else:
                Sxy[i,j] = np.nan
                Sz[i,j]  = np.nan
                K[0,i,j] = np.nan
                K[1,i,j] = np.nan

    np.save(filename+'_xy.npy',Sxy)
    np.save(filename+'_z.npy',Sz)
else:
    for i in tqdm.tqdm(range(Nx)):
        for j in range(Ny):
            #Consider all points in an extended BZ and also some redountant ones
            K[0,i,j] = -8*np.pi/3 + 16/3*np.pi/(Nx-1)*i
            K[1,i,j] = -4*np.pi/np.sqrt(3) + 8*np.pi/np.sqrt(3)/(Ny-1)*j
            if not fs.is_k_inside_BZ(K[:,i,j]):    #exclude points outside the extended BZ
                Sxy[i,j] = np.nan
                Sz[i,j]  = np.nan
                K[0,i,j] = np.nan
                K[1,i,j] = np.nan
    Sxy = np.load(filename+'_xy.npy')
    Sz = np.load(filename+'_z.npy')

#Plotting

fig = plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.title('S_xy')

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

plt.scatter(K[0],K[1],c=Sxy,cmap = cm.get_cmap('plasma_r'))#, vmin = 0, vmax = 1)
plt.gca().set_aspect('equal')
plt.colorbar()
###
plt.subplot(2,2,2)
plt.title('S_z')

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

plt.scatter(K[0],K[1],c=Sz,cmap = cm.get_cmap('plasma_r'))#, vmin = 0, vmax = 1)
plt.gca().set_aspect('equal')
plt.colorbar()
###
plt.subplot(2,2,3)
plt.title('S_total')

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

plt.scatter(K[0],K[1],c=Sz+Sxy,cmap = cm.get_cmap('plasma_r'))#, vmin = 0, vmax = 1)
plt.gca().set_aspect('equal')
plt.colorbar()
###

plt.subplot(2,2,4)
plt.title('Phases: theta = '+str(theta)+' , phi = '+str(phi))
plt.axis('off')
dirname = "Figures_cb2_g1/"
the = angles[0]
phi = angles[1]
title = dirname+"theta="+"{:.5f}".format(theta).replace('.',',')+"phi="+"{:.5f}".format(phi).replace('.',',')+".png"
plt.savefig(title)
plt.show()
