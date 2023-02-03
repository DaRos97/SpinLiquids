import numpy as np
import functions_new as fs
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os
import getopt
import tqdm
#input arguments
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "g:o:", ["UC=","theta=","phi=","save","show"])
    order = '3x3'
    gauge = '0'
    theta = 0
    phi = 0
    UC = 4
    save_fig = False
    show = False
    compute_new = False
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-g']:
        gauge = arg
    if opt in ['-o']:
        order = arg
    if opt == "--theta":
        theta = np.pi*float(arg)
    if opt == "--phi":
        phi = np.pi*float(arg)
    if opt == "--UC":
        UC = int(arg)
    if opt == "--save":
        save_fig = True
    if opt == "--show":
        show = True

list_orders = ['q0','3x3','ferro','cb1','cb2','oct']
if order not in list_orders:
    print('unknown regular order. Chose one from:')
    for ord_ in list_orders:
        print(ord_)
    exit()
gauge_list = ['1','0','2']
if gauge not in gauge_list:
    print('gauge argument not good, choose either 0,1 or 2')
gauge_trsf = int(gauge)
angles = (theta,phi)
#################
#################   Constant parameters (need to fix Nx,Ny)
Nx = 19
Ny = 17
kxi = np.linspace(-4*np.pi/np.sqrt(3)-8*np.pi/np.sqrt(3)/(Nx-1),4*np.pi/np.sqrt(3)+8*np.pi/np.sqrt(3)/(Nx-1),Nx)
kyi = np.linspace(-8*np.pi/3,8*np.pi/3,Ny)
Kxi = list(kxi)*Ny
Kyi = []
for i in kyi:
    for n in range(Nx):
        Kyi.append(i)
kxs = Kxi
kys = Kyi
np.save('data/kxs.npy',kxs)
np.save('data/kys.npy',kys)
if False:
    for n in range(Nx*Ny):
        x = Kxi[n]
        y = Kyi[n]
        Kxi[n] = np.sqrt(3)/2*x - y/2
        Kyi[n] = np.sqrt(3)/2*y + x/2
    kxs = []
    kys = []
    Rz = fs.R_z(-np.pi/6)
    for i in range(Nx*Ny):
        point = np.array([Kxi[i], Kyi[i], 0])
        kxs.append(np.tensordot(Rz,point,1)[0])
        kys.append(np.tensordot(Rz,point,1)[1])

#plt.figure()
#plt.scatter(kxs,kys,cmap = cm.get_cmap('plasma_r'))#, vmin = 0, vmax = 1)
#plt.show()
#exit()

#################
#################   Build lattice with GT
#Order of spins in lattice is, using up-pointing triangles as (lattice) unit cells, 0->lowleft - 1->lowright - 2->upcenter
lattice = fs.lattice(order,gauge_trsf,angles)


#################
#################   Structure factor
a1 = np.array([np.sqrt(3)/2,-1/2])
a2 = np.array([0,1])
Sxy = np.zeros(Nx*Ny)
Sz = np.zeros(Nx*Ny)
data_dirname = 'data/'
point_name = order + '_g' + str(gauge_trsf) + '_t' + "{:.4f}".format(theta).replace('.','-') + '_p' + "{:.4f}".format(phi).replace('.','-') + '_UC' + str(UC) 
filename = data_dirname + point_name
if not os.path.isfile(filename):
    for i in tqdm.tqdm(range(Nx*Ny)):
       K = np.array([kxs[i], kys[i]])
       Sxy[i], Sz[i] = fs.structure_factor_new(K,lattice,UC,a1,a2)
    np.save(filename+'_xy.npy',Sxy)
    np.save(filename+'_z.npy',Sz)
else:
    Sxy = np.load(filename+'_xy.npy')
    Sz = np.load(filename+'_z.npy')


#################
#################   Plotting
plt.figure(figsize=(8,6))
#plt.axis('off')
plt.suptitle(order+', g='+gauge+', theta='+"{:.4f}".format(theta)+', phi='+"{:.4f}".format(phi))

plt.subplot(1,2,1)
plt.title('S_xy')

plt.plot(np.linspace(0,4*np.pi/np.sqrt(3),1000),np.linspace(0,4*np.pi/np.sqrt(3),1000)*1/np.sqrt(3)-8*np.pi/3,'k-')
plt.plot(np.linspace(0,4*np.pi/np.sqrt(3),1000),-np.linspace(0,4*np.pi/np.sqrt(3),1000)*1/np.sqrt(3)+8*np.pi/3,'k-')
plt.plot(np.linspace(-4*np.pi/np.sqrt(3),0,1000),np.linspace(-4*np.pi/np.sqrt(3),0,1000)*1/np.sqrt(3)+8*np.pi/3,'k-')
plt.plot(np.linspace(-4*np.pi/np.sqrt(3),0,1000),-np.linspace(-4*np.pi/np.sqrt(3),0,1000)*1/np.sqrt(3)-8*np.pi/3,'k-')
plt.vlines(4*np.pi/np.sqrt(3),-4*np.pi/3, 4*np.pi/3, color = 'k')
plt.vlines(-4*np.pi/np.sqrt(3),-4*np.pi/3, 4*np.pi/3, color = 'k')

plt.plot(np.linspace(0,2*np.pi/np.sqrt(3),1000),np.linspace(0,2*np.pi/np.sqrt(3),1000)*1/np.sqrt(3)-4*np.pi/3,'k-')
plt.plot(np.linspace(0,2*np.pi/np.sqrt(3),1000),-np.linspace(0,2*np.pi/np.sqrt(3),1000)*1/np.sqrt(3)+4*np.pi/3,'k-')
plt.plot(np.linspace(-2*np.pi/np.sqrt(3),0,1000),np.linspace(-2*np.pi/np.sqrt(3),0,1000)*1/np.sqrt(3)+4*np.pi/3,'k-')
plt.plot(np.linspace(-2*np.pi/np.sqrt(3),0,1000),-np.linspace(-2*np.pi/np.sqrt(3),0,1000)*1/np.sqrt(3)-4*np.pi/3,'k-')
plt.vlines(2*np.pi/np.sqrt(3),-2*np.pi/3, 2*np.pi/3, color = 'k')
plt.vlines(-2*np.pi/np.sqrt(3),-2*np.pi/3, 2*np.pi/3, color = 'k')

plt.scatter(kxs,kys,c=Sxy,cmap = cm.get_cmap('plasma_r'))#, vmin = 0, vmax = 1)
plt.gca().set_aspect('equal')
plt.colorbar()
###
plt.subplot(1,2,2)
plt.title('S_z')

plt.plot(np.linspace(0,4*np.pi/np.sqrt(3),1000),np.linspace(0,4*np.pi/np.sqrt(3),1000)*1/np.sqrt(3)-8*np.pi/3,'k-')
plt.plot(np.linspace(0,4*np.pi/np.sqrt(3),1000),-np.linspace(0,4*np.pi/np.sqrt(3),1000)*1/np.sqrt(3)+8*np.pi/3,'k-')
plt.plot(np.linspace(-4*np.pi/np.sqrt(3),0,1000),np.linspace(-4*np.pi/np.sqrt(3),0,1000)*1/np.sqrt(3)+8*np.pi/3,'k-')
plt.plot(np.linspace(-4*np.pi/np.sqrt(3),0,1000),-np.linspace(-4*np.pi/np.sqrt(3),0,1000)*1/np.sqrt(3)-8*np.pi/3,'k-')
plt.vlines(4*np.pi/np.sqrt(3),-4*np.pi/3, 4*np.pi/3, color = 'k')
plt.vlines(-4*np.pi/np.sqrt(3),-4*np.pi/3, 4*np.pi/3, color = 'k')

plt.plot(np.linspace(0,2*np.pi/np.sqrt(3),1000),np.linspace(0,2*np.pi/np.sqrt(3),1000)*1/np.sqrt(3)-4*np.pi/3,'k-')
plt.plot(np.linspace(0,2*np.pi/np.sqrt(3),1000),-np.linspace(0,2*np.pi/np.sqrt(3),1000)*1/np.sqrt(3)+4*np.pi/3,'k-')
plt.plot(np.linspace(-2*np.pi/np.sqrt(3),0,1000),np.linspace(-2*np.pi/np.sqrt(3),0,1000)*1/np.sqrt(3)+4*np.pi/3,'k-')
plt.plot(np.linspace(-2*np.pi/np.sqrt(3),0,1000),-np.linspace(-2*np.pi/np.sqrt(3),0,1000)*1/np.sqrt(3)-4*np.pi/3,'k-')
plt.vlines(2*np.pi/np.sqrt(3),-2*np.pi/3, 2*np.pi/3, color = 'k')
plt.vlines(-2*np.pi/np.sqrt(3),-2*np.pi/3, 2*np.pi/3, color = 'k')

plt.scatter(kxs,kys,c=Sz,cmap = cm.get_cmap('plasma_r'))#, vmin = 0, vmax = 1)
plt.gca().set_aspect('equal')
plt.colorbar()
####

fig_dirname = "Figures/"
the = angles[0]
phi = angles[1]
fig_filename = fig_dirname + point_name
if save_fig:
    plt.savefig(fig_filename)
if show:
    plt.show()


