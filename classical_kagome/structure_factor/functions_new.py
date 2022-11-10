import numpy as np
import matplotlib.pyplot as plt


N_ = 10
#Define the  gauge transformation
Gauge_trsf = np.ndarray((N_*3,N_*3,3))
Gauge_trsf[0,0,0] = 2; Gauge_trsf[0,0,1] = 1; Gauge_trsf[0,0,2] = 0
Gauge_trsf[0,1,0] = 0; Gauge_trsf[0,1,1] = 2; Gauge_trsf[0,1,2] = 1
Gauge_trsf[0,2,0] = 1; Gauge_trsf[0,2,1] = 0; Gauge_trsf[0,2,2] = 2
Gauge_trsf[1,0,0] = 0; Gauge_trsf[1,0,1] = 2; Gauge_trsf[1,0,2] = 1
Gauge_trsf[1,1,0] = 1; Gauge_trsf[1,1,1] = 0; Gauge_trsf[1,1,2] = 2
Gauge_trsf[1,2,0] = 2; Gauge_trsf[1,2,1] = 1; Gauge_trsf[1,2,2] = 0
Gauge_trsf[2,0,0] = 1; Gauge_trsf[2,0,1] = 0; Gauge_trsf[2,0,2] = 2
Gauge_trsf[2,1,0] = 2; Gauge_trsf[2,1,1] = 1; Gauge_trsf[2,1,2] = 0
Gauge_trsf[2,2,0] = 0; Gauge_trsf[2,2,1] = 2; Gauge_trsf[2,2,2] = 1
#fill the others
for i in range(0,N_*3,3):
    for j in range(0,N_*3,3):
        Gauge_trsf[i,j]   = Gauge_trsf[0,0]; Gauge_trsf[i+1,j]   = Gauge_trsf[1,0]; Gauge_trsf[i+2,j]   = Gauge_trsf[2,0];
        Gauge_trsf[i,j+1] = Gauge_trsf[0,1]; Gauge_trsf[i+1,j+1] = Gauge_trsf[1,1]; Gauge_trsf[i+2,j+1] = Gauge_trsf[2,1];
        Gauge_trsf[i,j+2] = Gauge_trsf[0,2]; Gauge_trsf[i+1,j+2] = Gauge_trsf[1,2]; Gauge_trsf[i+2,j+2] = Gauge_trsf[2,2];
def gauge_trsf(L):
    #Make the gauge transformation on a lattice 18*18*3 sites
    for x in range(N_*3):
        for y in range(N_*3):
            for m in range(3):
                L[x,y,m] = np.tensordot(R_z(Gauge_trsf[x,y,m]*2/3*np.pi),L[x,y,m],1)
    return L
###############################
############################### LATTICES
###############################
def ferro_lattice(spin_angles):
    Sa = np.array([0,1/2,0])
    Ry = R_y(spin_angles[0])
    Rz = R_z(spin_angles[1])
    a = np.tensordot(Rz,np.tensordot(Ry,Sa,1),1)
    L = np.ndarray((N_*3,N_*3,3,3))
    L[0,0,0] = a; L[0,0,1] = a; L[0,0,2] = a
    #fill the lattice
    for i in range(0,N_*3):
        for j in range(0,N_*3):
            L[i,j]      = L[0,0]
    return L
def s3x3_lattice(spin_angles):
    Sa = np.array([0,1/2,0])
    Sb = np.array([-np.sqrt(3)/4,-1/4,0])
    Sc = np.array([np.sqrt(3)/4,-1/4,0])
    Ry = R_y(spin_angles[0])
    Rz = R_z(spin_angles[1])
    a = np.tensordot(Rz,np.tensordot(Ry,Sa,1),1)
    b = np.tensordot(Rz,np.tensordot(Ry,Sb,1),1)
    c = np.tensordot(Rz,np.tensordot(Ry,Sc,1),1)
    L = np.ndarray((N_*3,N_*3,3,3))
    L[0,0,0] = b; L[0,0,1] = c; L[0,0,2] = a
    L[0,1,0] = a; L[0,1,1] = b; L[0,1,2] = c
    L[0,2,0] = c; L[0,2,1] = a; L[0,2,2] = b
    L[1,0,0] = a; L[1,0,1] = b; L[1,0,2] = c
    L[1,1,0] = c; L[1,1,1] = a; L[1,1,2] = b
    L[1,2,0] = b; L[1,2,1] = c; L[1,2,2] = a
    L[2,0,0] = c; L[2,0,1] = a; L[2,0,2] = b
    L[2,1,0] = b; L[2,1,1] = c; L[2,1,2] = a
    L[2,2,0] = a; L[2,2,1] = b; L[2,2,2] = c
    #fill the lattice
    for i in range(0,N_*3,3):
        for j in range(0,N_*3,3):
            L[i,j]      = L[0,0];   L[i,j+1]    = L[0,1];   L[i,j+2]    = L[0,2]
            L[i+1,j]    = L[1,0];   L[i+1,j+1]  = L[1,1];   L[i+1,j+2]  = L[1,2]
            L[i+2,j]    = L[2,0];   L[i+2,j+1]  = L[2,1];   L[i+2,j+2]  = L[2,2]
    return L
def q0_lattice(spin_angles):
    Sa = np.array([0,1/2,0])
    Sb = np.array([-np.sqrt(3)/4,-1/4,0])
    Sc = np.array([np.sqrt(3)/4,-1/4,0])
    Rz = R_y(spin_angles[0])
    Ry = R_z(spin_angles[1])
    a = np.tensordot(Rz,np.tensordot(Ry,Sa,1),1)
    b = np.tensordot(Rz,np.tensordot(Ry,Sb,1),1)
    c = np.tensordot(Rz,np.tensordot(Ry,Sc,1),1)
    Uc = 3      #octahedral unit cells in each direction (3*3*12 sites in total)
    L = np.ndarray((N_*3,N_*3,3,3))
    L[0,0,0] = b; L[0,0,1] = c; L[0,0,2] = a
    #fill the lattice
    for i in range(0,N_*3):
        for j in range(0,N_*3):
            L[i,j]      = L[0,0]
    return L
def oct_lattice(spin_angles):
    Sa = np.array([1/2,0,0])
    Sb = np.array([0,1/2,0])
    Sc = np.array([0,0,1/2])
    Ry = R_y(spin_angles[0])
    Rz = R_z(spin_angles[1])
    #
    a = np.tensordot(Rz,np.tensordot(Ry,Sa,1),1)
    b = np.tensordot(Rz,np.tensordot(Ry,Sb,1),1)
    c = np.tensordot(Rz,np.tensordot(Ry,Sc,1),1)
    Uc = 3      #octahedral unit cells in each direction (3*3*12 sites in total)
    L = np.ndarray((N_*3,N_*3,3,3))
    L[0,0,0] = b
    L[0,0,1] = c
    L[0,0,2] = -a
    L[1,0,0] = -b
    L[1,0,1] = -c
    L[1,0,2] = -a
    L[0,1,0] = b
    L[0,1,1] = -c
    L[0,1,2] = a
    L[1,1,0] = -b
    L[1,1,1] = c
    L[1,1,2] = a
    #fill the lattice
    for i in range(0,N_*3,2):
        for j in range(0,N_*3,2):
            L[i,j]      = L[0,0]
            L[i+1,j]    = L[1,0]
            L[i,j+1]    = L[0,1]
            L[i+1,j+1]  = L[1,1]
    return L
def cb1_lattice(spin_angles):
    t0 = np.arctan(np.sqrt(2))
    Sa = np.array([1/2,0,0])
    Sb = np.array([1/4,np.sqrt(3)/4,0])
    Sc = np.array([-1/4,np.sqrt(3)/4,0])
    Sd = np.array([0,np.cos(t0)/2,np.sin(t0)/2])
    Se = np.array([-np.cos(t0)*np.sqrt(3)/4,-np.cos(t0)/4,np.sin(t0)/2])
    Sf = np.array([np.cos(t0)*np.sqrt(3)/4,-np.cos(t0)/4,np.sin(t0)/2])
    Rz = R_y(spin_angles[0])
    Ry = R_z(spin_angles[1])
    a = np.tensordot(Rz,np.tensordot(Ry,Sa,1),1)
    b = np.tensordot(Rz,np.tensordot(Ry,Sb,1),1)
    c = np.tensordot(Rz,np.tensordot(Ry,Sc,1),1)
    d = np.tensordot(Rz,np.tensordot(Ry,Sd,1),1)
    e = np.tensordot(Rz,np.tensordot(Ry,Se,1),1)
    f = np.tensordot(Rz,np.tensordot(Ry,Sf,1),1)
    Uc = 3      #octahedral unit cells in each direction (3*3*12 sites in total)
    L = np.ndarray((N_*3,N_*3,3,3))
    L[0,0,0] = e
    L[0,0,1] = -d
    L[0,0,2] = b
    L[1,0,0] = c
    L[1,0,1] = a
    L[1,0,2] = -b
    L[0,1,0] = -e
    L[0,1,1] = -a
    L[0,1,2] = f
    L[1,1,0] = -c
    L[1,1,1] = d
    L[1,1,2] = -f
    #fill the lattice
    for i in range(0,N_*3,2):
        for j in range(0,N_*3,2):
            L[i,j]      = L[0,0]
            L[i+1,j]    = L[1,0]
            L[i,j+1]    = L[0,1]
            L[i+1,j+1]  = L[1,1]
    return L
def cb2_lattice(spin_angles):
    t0 = np.arctan(np.sqrt(2))
    Sa = np.array([1/2,0,0])
    Sb = np.array([1/4,np.sqrt(3)/4,0])
    Sc = np.array([-1/4,np.sqrt(3)/4,0])
    Sd = np.array([0,np.cos(t0)/2,np.sin(t0)/2])
    Se = np.array([-np.cos(t0)*np.sqrt(3)/4,-np.cos(t0)/4,np.sin(t0)/2])
    Sf = np.array([np.cos(t0)*np.sqrt(3)/4,-np.cos(t0)/4,np.sin(t0)/2])
    Rz = R_y(spin_angles[0])
    Ry = R_z(spin_angles[1])
    a = np.tensordot(Rz,np.tensordot(Ry,Sa,1),1)
    b = np.tensordot(Rz,np.tensordot(Ry,Sb,1),1)
    c = np.tensordot(Rz,np.tensordot(Ry,Sc,1),1)
    d = np.tensordot(Rz,np.tensordot(Ry,Sd,1),1)
    e = np.tensordot(Rz,np.tensordot(Ry,Se,1),1)
    f = np.tensordot(Rz,np.tensordot(Ry,Sf,1),1)
    L = np.ndarray((N_*3,N_*3,3,3))
    L[0,0,0] = -c
    L[0,0,1] = -d
    L[0,0,2] = -b
    L[1,0,0] = -e
    L[1,0,1] = a
    L[1,0,2] = b
    L[0,1,0] = c
    L[0,1,1] = -a
    L[0,1,2] = -f
    L[1,1,0] = e
    L[1,1,1] = d
    L[1,1,2] = f
    #fill the lattice
    for i in range(0,N_*3,2):
        for j in range(0,N_*3,2):
            L[i,j]      = L[0,0]
            L[i+1,j]    = L[1,0]
            L[i,j+1]    = L[0,1]
            L[i+1,j+1]  = L[1,1]
    return L

#
def lattice(order,GT,angles):
    if order == 'ferro':
        L = ferro_lattice(angles)
    elif order == '3x3':
        L = s3x3_lattice(angles)
    elif order == 'q0':
        L = q0_lattice(angles)
    elif order == 'cb1':
        L = cb1_lattice(angles)
    elif order == 'cb2':
        L = cb2_lattice(angles)
    elif order == 'oct':
        L = oct_lattice(angles)
    if GT == 1:
        L = gauge_trsf(L)
    elif GT == 2:
        L = gauge_trsf(gauge_trsf(L))

    return L
##########################
##########################      New_ssf

def structure_factor_new(k,L,UC,a1,a2):
    resxy = 0
    resz = 0
    distx_UC = np.array([[0,np.sqrt(3)/4,np.sqrt(3)/4],[-np.sqrt(3)/4,0,0],[-np.sqrt(3)/4,0,0]])
    disty_UC = np.array([[0,-1/4,1/4],[1/4,0,1/2],[-1/4,-1/2,0]])
    for i1 in range(UC):
        for j1 in range(UC):
            for l1 in range(3):
                L1 = L[i1,j1,l1]
                r1 = i1*a1+j1*a2
                for i2 in range(UC):
                    for j2 in range(UC):
                        for l2 in range(3):
                            L2 = L[i2,j2,l2]
                            r2 = i2*a1+j2*a2
                            dist = np.zeros(2)
                            dist[0] = r2[0] - r1[0] + distx_UC[l1,l2]
                            dist[1] = r2[1] - r1[1] + disty_UC[l1,l2]
                            SiSjxy = L1[0]*L2[0] + L1[1]*L2[1]#np.dot(Li,L[i2,j2,l2])
                            SiSjz = L1[2]*L2[2]
                            resxy += np.exp(-1j*np.dot(k,dist))*SiSjxy
                            resz += np.exp(-1j*np.dot(k,dist))*SiSjz
    return np.real(resxy), np.real(resz)








#z rotations
def R_z(t):
    R = np.zeros((3,3))
    R[0,0] = np.cos(t)
    R[0,1] = -np.sin(t)
    R[1,0] = np.sin(t)
    R[1,1] = np.cos(t)
    R[2,2] = 1
    return R
#x rotations
def R_x(t):
    R = np.zeros((3,3))
    R[1,1] = np.cos(t)
    R[1,2] = -np.sin(t)
    R[2,1] = np.sin(t)
    R[2,2] = np.cos(t)
    R[0,0] = 1
    return R
#y rotations
def R_y(t):
    R = np.zeros((3,3))
    R[0,0] = np.cos(t)
    R[0,2] = np.sin(t)
    R[2,0] = np.sin(t)
    R[2,2] = np.cos(t)
    R[1,1] = 1
    return R
