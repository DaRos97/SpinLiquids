import numpy as np
import functions as fs
import sys
#import tqdm
import os

#inputs:    DM_angle -> in str, J_pts

J1 = 1
lim = float(sys.argv[3])
J2i = -lim
J2f = lim
J3i = -lim
J3f = lim
J2pts = J3pts = int(sys.argv[2])
DM_orientation = 0      #0 for TMD and 1 for Messio
J2 = np.linspace(J2i,J2f,J2pts)
J3 = np.linspace(J3i,J3f,J3pts)


min_energy = np.zeros((J2pts,J3pts,2,16))

dic_DM = {'000':0,'005':0.05,'104':np.pi/3,'209':2*np.pi/3}
dm_angle_1nn = dic_DM[sys.argv[1]]
DM_angles = np.array([dm_angle_1nn,0,2*dm_angle_1nn])
spin_angles = (0,0)
#
dirname = 'data/'
#dirname = '/home/users/r/rossid/code_classical_pd/data/'
filename = dirname+'J2_'+str(J2i)+'--'+str(J2f)+'__J3_'+str(J3i)+'--'+str(J3f)+'__DM_'+sys.argv[1]+'__Pts_'+sys.argv[2]+'.npy'

orders = ['ferro', '3x3', 'q0', 'octa', 'cb1', 'cb2']
m = [1,3,3,1,3,3,2,6,6,2,6,6,2,6,6]
func_L = {'ferro': fs.ferro_lattice, 'q0': fs.q0_lattice, 'octa': fs.oct_lattice, 'cb1': fs.cb1_lattice, 'cb2': fs.cb2_lattice}
lattices = []
for o in orders:
    if o == '3x3':
        continue
    L = func_L[o](spin_angles)
    lattices.append(L.copy())
    for g in range(2):
        fs.gauge_trsf(L)
        lattices.append(L.copy())

if os.path.isfile(filename):        ####################
    energies = np.load(filename)
    print("Already computed")
    exit()
print('using: ',*sys.argv[1:])
for n2 in range(J2pts):
    j2 = J2[n2]
    for n3,j3 in enumerate(J3):
        J = np.array([J1,j2,j3])
        step_en = []
        for i in range(15):
            step_en.append(fs.energy(lattices[i],6,J, DM_angles))
        #spiral = fs.spiral(J,DM_angles)
        spiral = (0,(0,0))
        step_en.append(spiral[0])
        min_energy[n2,n3,0] = step_en
        min_energy[n2,n3,1] = np.append(spiral[1],np.zeros(16-len(spiral[1])))
        with open(filename,'w') as f:
            np.save(filename,min_energy)






