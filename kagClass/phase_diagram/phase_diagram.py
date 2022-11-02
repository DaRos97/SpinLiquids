import numpy as np
import functions as fs
import sys

J1 = 1
J2i = -0.3
J2f = 0.3
J3i = -0.3
J3f = 0.3
J2pts = 11
J3pts = 11
DM_orientation = 0      #0 for TMD and 1 for Messio
J2 = np.linspace(J2i,J2f,J2pts)
J3 = np.linspace(J3i,J3f,J3pts)


min_energy = np.zeros((J2pts,J3pts,3))

dm_angle_1nn = int(sys.argv[1])*np.pi/48
DM_angles = np.array([dm_angle_1nn,0,2*dm_angle_1nn])
spin_angles = (0,0)
#
dirname = 'ferro_j1/' if J1 == -1 else 'antiferro_j1/'
#dirname = 'af2/'
filename = dirname+'DM1nn_'+"{:.3f}".format(dm_angle_1nn).replace('.','-')+'.npy'

for n2,j2 in enumerate(J2):
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
        spiral = fs.spiral(J)
        #
        temp = ferro
        if temp[0] > s3x3[0]:
            temp = s3x3
        if temp[0] > s3x3_g1[0]:
            temp = s3x3_g1
        if temp[0] > q0[0]:
            temp = q0
        if temp[0] > q0_g1[0]:
            temp = q0_g1
        if temp[0] > q0_g2[0]:
            temp = q0_g2
        if temp[0] > octa[0]:
            temp = octa
        if temp[0] > octa_g1[0]:
            temp = octa_g1
        if temp[0] > octa_g2[0]:
            temp = octa_g2
        if temp[0] > cb1[0]:
            temp = cb1
        if temp[0] > cb1_g1[0]:
            temp = cb1_g1
        if temp[0] > cb1_g2[0]:
            temp = cb1_g2
        if temp[0] > cb2[0]:
            temp = cb2
        if temp[0] > cb2_g1[0]:
            temp = cb2_g1
        if temp[0] > cb2_g2[0]:
            temp = cb2_g2
        if temp[0] > spiral[0]:
            temp = spiral
        #
        #lower_bound = fs.lower_bound_energy(J)
        min_energy[n2,n3,0] = temp[0]
        min_energy[n2,n3,1] = temp[1]
        min_energy[n2,n3,2] = 0#lower_bound
np.save(filename,min_energy)
