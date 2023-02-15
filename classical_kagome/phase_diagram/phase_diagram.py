import numpy as np
import functions as fs
import sys
import getopt
#import tqdm
import os

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "r:",["DM=","pts="])
    lim = 3
    DM = '000'
    pts = 21
except:
    print("Error in input parameters",argv)
    exit()
for opt, arg in opts:
    if opt in ['-r']:
        lim = float(arg)
    if opt == '--DM':
        DM = arg
    if opt == '--pts':
        pts = int(arg)

J1 = 1
J2i = -lim
J2f = lim
J3i = -lim
J3f = lim
J2pts = J3pts = pts
DM_orientation = 0      #0 for TMD and 1 for Messio
J2 = np.linspace(J2i,J2f,J2pts)
J3 = np.linspace(J3i,J3f,J3pts)


min_energy = np.zeros((J2pts,J3pts,16,16))
J2_i = J3_i = 0


dic_DM = {'000':0,'005':0.05,'104':np.pi/3,'209':2*np.pi/3}
dm_angle_1nn = dic_DM[DM]
DM_angles = np.array([dm_angle_1nn,0,2*dm_angle_1nn])
spin_angles = (0,0)
#
dirname = 'data/'
#dirname = '/home/users/r/rossid/code_classical_pd/data/'
filename = dirname+'J2_'+str(J2i)+'--'+str(J2f)+'__J3_'+str(J3i)+'--'+str(J3f)+'__DM_'+DM+'__Pts_'+str(pts)+'.npy'

orders = ['ferro', '3x3', 'q0', 'octa', 'cb1', 'cb2']
m__ = [1,3,3,1,3,3,2,6,6,2,6,6,2,6,6]
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


###########################################################################
if os.path.isfile(filename):        ####################
    energies = np.load(filename)
    i_ = j_ = J2pts-1
    bb = False
    for i in range(J2pts):
        for j in range(J3pts):
            if (energies[i,j] == np.zeros((16,16))).all():
                i_ = i
                j_ = j
                bb = True
            if bb:
                break
        if bb:
            break
    if i_ == J2pts-1 and j_ == J3pts-1:
        print("Already computed completely")
        exit()
    else:
        print("Starting from ",i_,j_)
        J2_i = i_
        J3_i = j_
        min_energy = energies

###########################################################################
print('using: ',lim,DM,pts)
for n2 in range(J2_i,J2pts):
    j2 = J2[n2]
    for n3 in range(0,J3pts):
        if n2 == J2_i and n3 < J3_i:
            continue
        j3 = J3[n3]
        print(j2,j3)
        J = np.array([J1,j2,j3])
        step_en = []
        args = []
        for i in range(15):
            L = np.copy(lattices[i])
            res = fs.lat_energy(L,m__[i],J,DM_angles)
            min_energy[n2,n3,i,0] = res[0]
            min_energy[n2,n3,i,1:3] = res[1]
        spiral = fs.spiral(J,DM_angles)
        min_energy[n2,n3,15,0] = spiral[0]
        min_energy[n2,n3,15,1:] = spiral[1]
        with open(filename,'w') as f:
            np.save(filename,min_energy)






