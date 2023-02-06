import numpy as np
import functions_new as fs
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os
import getopt
import tqdm
#input arguments

list_orders = ['3x3','q0','cb1','cb2','oct']
angles = (0,0)
theta,phi = angles
UC = 9
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

SF = []
for order in list_orders:
    SF.append([])
    for gauge in range(3):
        #################
        #################   Build lattice
        #Order of spins in lattice is, using up-pointing triangles as (lattice) unit cells, 0->lowleft - 1->lowright - 2->upcenter
        lattice = fs.lattice(order,gauge,angles)
        #################
        #################   Structure factor
        a1 = np.array([np.sqrt(3)/2,-1/2])
        a2 = np.array([0,1])
        Sxy = np.zeros(Nx*Ny)
        Sz = np.zeros(Nx*Ny)
        data_dirname = 'data/'
        point_name = order + '_g' + str(gauge) + '_t' + "{:.4f}".format(theta).replace('.','-') + '_p' + "{:.4f}".format(phi).replace('.','-') + '_UC' + str(UC) 
        filename = data_dirname + point_name
        if not os.path.isfile(filename+'_z.npy'):
            for i in tqdm.tqdm(range(Nx*Ny)):
               K = np.array([kxs[i], kys[i]])
               Sxy[i], Sz[i] = fs.structure_factor_new(K,lattice,UC,a1,a2)
            np.save(filename+'_xy.npy',Sxy)
            np.save(filename+'_z.npy',Sz)
        else:
            Sxy = np.load(filename+'_xy.npy')
            Sz = np.load(filename+'_z.npy')
        SF[-1].append(Sxy+Sz)

#################
#################   Plotting
dic_order = {   '3x3':r'$\sqrt{3}\times\sqrt{3}$',
                'q0':r'$\mathbf{Q}=0$',
                'cb1':r'$cuboc-1$',
                'cb2':r'$cuboc-2$',
                'oct':r'$octahedral$',
                }
dic_gauge = ['no gauge', r'$1$'+' gauge',r'$2$'+' gauge']
plt.figure(figsize=(18,18))
for n,order in enumerate(list_orders):
    for gauge in range(3):
        plt.subplot(len(list_orders),3,n*3+gauge+1)
        #Single SSF plot
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

        plt.scatter(kxs,kys,c=SF[n][gauge],cmap = cm.get_cmap('plasma_r'))#, vmin = 0, vmax = 1)
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.tick_params(labelleft = False , labelbottom = False, bottom = False, left = False)
        if gauge == 0:
            plt.text(-22,0,dic_order[order],size='xx-large')
        if n == 0:
            plt.title(dic_gauge[gauge],size='xx-large')
plt.show()


