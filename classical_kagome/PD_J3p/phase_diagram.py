import numpy as np
import functions as fs
import sys
import tqdm
import os

#inputs:    DM_angle -> in str, J_pts

J1 = 1
J2i = 0
J2f = 3
J3i = 0
J3f = 3
J3p = -0.2
J2pts = J3pts = int(sys.argv[2])
DM_orientation = 0      #0 for TMD and 1 for Messio
J2 = np.linspace(J2i,J2f,J2pts)
J3 = np.linspace(J3i,J3f,J3pts)


min_energy = np.zeros((J2pts,J3pts,3))

dic_DM = {'000':0,'005':0.05,'104':np.pi/3,'209':2*np.pi/3}
dm_angle_1nn = dic_DM[sys.argv[1]]
DM_angles = np.array([dm_angle_1nn,0,2*dm_angle_1nn])
spin_angles = (0,0)
#
dirname = 'ferro_j1/' if J1 == -1 else 'antiferro_j1/'
#dirname = 'af2/'
filename = dirname+'DM1nn_'+sys.argv[1]+'_'+sys.argv[2]+'_spiral.npy'

if os.path.isfile(filename):
    print("Already computed")
    exit()

for n2 in tqdm.tqdm(range(J2pts)):
    j2 = J2[n2]
    for n3,j3 in enumerate(J3):
        J = np.array([J1,j2,j3,J3p])
        ferro = fs.ferro(J,spin_angles,DM_angles,DM_orientation)
        s3x3 = fs.s3x3(J,spin_angles,DM_angles,DM_orientation)
        s3x3_g1 = fs.s3x3_g1(J,spin_angles,DM_angles,DM_orientation)
        q0 = fs.q0(J,spin_angles,DM_angles,DM_orientation)
        q0_g1 = fs.q0_g1(J,spin_angles,DM_angles,DM_orientation)
        q0_g2 = fs.q0_g2(J,spin_angles,DM_angles,DM_orientation)
        octa = fs.octa(J,spin_angles,DM_angles,DM_orientation)
        octa_g1 = fs.octa_g1(J,spin_angles,DM_angles,DM_orientation)
        octa_g2 = fs.octa_g2(J,spin_angles,DM_angles,DM_orientation)
        cb1 = fs.cb1(J,spin_angles,DM_angles,DM_orientation)
        cb1_g1 = fs.cb1_g1(J,spin_angles,DM_angles,DM_orientation)
        cb1_g2 = fs.cb1_g2(J,spin_angles,DM_angles,DM_orientation)
        cb2 = fs.cb2(J,spin_angles,DM_angles,DM_orientation)
        cb2_g1 = fs.cb2_g1(J,spin_angles,DM_angles,DM_orientation)
        cb2_g2 = fs.cb2_g2(J,spin_angles,DM_angles,DM_orientation)
        spiral = fs.spiral(J,DM_angles)
        step_en = [ferro, s3x3, s3x3_g1, q0, q0_g1, q0_g2, octa, octa_g1, octa_g2, cb1, cb1_g1, cb1_g2, cb2, cb2_g1, cb2_g2, spiral]
        min_energy[n2,n3,0] = 0
        min_energy[n2,n3,1] = np.argmin(step_en)
        min_energy[n2,n3,2] = 0#lower_bound

np.save(filename,min_energy)
