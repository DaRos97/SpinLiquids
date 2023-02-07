import numpy as np
import functions as fs
import sys
#import tqdm
import os

#inputs:    DM_angle -> in str, J_pts

J1 = 1
lim = sys.argv[3]
J2i = -lim
J2f = lim
J3i = -lim
J3f = lim
J2pts = J3pts = int(sys.argv[2])
DM_orientation = 0      #0 for TMD and 1 for Messio
J2 = np.linspace(J2i,J2f,J2pts)
J3 = np.linspace(J3i,J3f,J3pts)


min_energy = np.zeros((J2pts,J3pts))

dic_DM = {'000':0,'005':0.05,'104':np.pi/3,'209':2*np.pi/3}
dm_angle_1nn = dic_DM[sys.argv[1]]
DM_angles = np.array([dm_angle_1nn,0,2*dm_angle_1nn])
spin_angles = (0,0)
#
dirname = 'data/'
filename = dirname+'J2_'+str(J2i)+'--'+str(J2f)+'__J3_'+str(J3i)+'--'+str(J3f)+'__DM_'+sys.argv[1]+'__Pts_'+sys.argv[2]+'.npy'

if os.path.isfile(filename):
    print("Already computed")
    exit()
print('using: ',*sys.argv[1:])
for n2 in range(J2pts):
    j2 = J2[n2]
    for n3,j3 in enumerate(J3):
        J = np.array([J1,j2,j3])
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
        min_energy[n2,n3] = np.argmin(step_en)

np.save(filename,min_energy)
