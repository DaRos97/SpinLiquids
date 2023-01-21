import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
##############################################
#Define the  gauge transformation
Gauge_trsf = np.ndarray((18,18,3))
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
for i in range(0,18,3):
    for j in range(0,18,3):
        Gauge_trsf[i,j]   = Gauge_trsf[0,0]; Gauge_trsf[i+1,j]   = Gauge_trsf[1,0]; Gauge_trsf[i+2,j]   = Gauge_trsf[2,0];
        Gauge_trsf[i,j+1] = Gauge_trsf[0,1]; Gauge_trsf[i+1,j+1] = Gauge_trsf[1,1]; Gauge_trsf[i+2,j+1] = Gauge_trsf[2,1];
        Gauge_trsf[i,j+2] = Gauge_trsf[0,2]; Gauge_trsf[i+1,j+2] = Gauge_trsf[1,2]; Gauge_trsf[i+2,j+2] = Gauge_trsf[2,2];
def gauge_trsf(L):
    #Make the gauge transformation on a lattice 18*18*3 sites
    for x in range(18):
        for y in range(18):
            for m in range(3):
                L[x,y,m] = np.tensordot(R_z(Gauge_trsf[x,y,m]*2/3*np.pi),L[x,y,m],1)
    return L
########
########
########

Z_ = [4,4,2]
#define regular orders with DM angles as functions of J1,J2,J3, the DM angle and also a representative spin orientation
def energy(L,m,j,DM_angles,DM_direction):
    energy_1nn = 0
    energy_2nn = 0
    energy_3nn = 0
    DM_0 = 1 if DM_direction == 0 else -1
    for x in range(6,6+m):
        for y in range(6,6+m):
            s0 = L[x,y,0]; s0_ = L[x-1,y-1,2]
            s1 = L[x,y,1]; s1_ = L[x+1,y,0]
            s2 = L[x,y,2]; s2_ = L[x,y+1,1]
            energy_1nn += j[0]*(     s0[2]*s0_[2]+s1[2]*s1_[2]+s2[2]*s2_[2] + 
                                     s0[2]*s1[2] + s1[2]*s2[2] + s2[2]*s0[2]
                                     + np.cos(2*DM_angles[0])*(s0[0]*s0_[0]+s0[1]*s0_[1]+s1[0]*s1_[0]+s1[1]*s1_[1]+s2[0]*s2_[0]+s2[1]*s2_[1] +
                                                               s0[0]*s1[0]+s0[1]*s1[1] + s1[0]*s2[0]+s1[1]*s2[1] + s2[0]*s0[0]+s2[1]*s0[1])
                                     + np.sin(2*DM_angles[0])*((s0[1]*s0_[0]-s0[0]*s0_[1] + s1[1]*s1_[0]-s1[0]*s1_[1] + s2[1]*s2_[0]-s2[0]*s2_[1])*DM_0 +
                                                               s0[1]*s1[0]-s0[0]*s1[1] + s1[1]*s2[0]-s1[0]*s2[1] + s2[1]*s0[0]-s2[0]*s0[1])
                                     )
            s0a = s2_; s0b = L[x,y-1,2]
            s1a = s0_; s1b = L[x+1,y+1,0]
            s2a = s1_; s2b = L[x-1,y,1]
            energy_2nn += j[1]*(np.dot(s0,s0a) + np.dot(s0,s0b) +
                                     np.dot(s1,s1a) + np.dot(s1,s1b) +
                                     np.dot(s2,s2a) + np.dot(s2,s2b)
                                     ) 
            s0h = L[x,y+1,0]
            s1h = L[x-1,y-1,1]
            s2h = L[x+1,y,2]
            energy_3nn += j[2]*(s0[2]*s0h[2]+s1[2]*s1h[2]+s2[2]*s2h[2] 
                                     + np.cos(2*DM_angles[2])*(s0[0]*s0h[0]+s0[1]*s0h[1]+s1[0]*s1h[0]+s1[1]*s1h[1]+s2[0]*s2h[0]+s2[1]*s2h[1])
                                     + np.sin(2*DM_angles[2])*(s0[1]*s0h[0]-s0[0]*s0h[1] + s1[1]*s1h[0]-s1[0]*s1h[1] + s2[1]*s2h[0]-s2[0]*s2h[1])
                                     )
    return (energy_1nn + energy_2nn + energy_3nn)/(m*m*3)*2
#ORBIT: F->3x3_g1->3x3
def ferro(j,spin_angles,DM_angles,DM):
    m = 6
    L = ferro_lattice(spin_angles)
    ferro_energy = energy(L,m,j,DM_angles,DM)
    return ferro_energy
def s3x3(j,spin_angles,DM_angles,DM):
    m = 6
    L = s3x3_lattice(spin_angles)
    s3x3_energy = energy(L,m,j,DM_angles,DM)
    return s3x3_energy
def s3x3_g1(j,spin_angles,DM_angles,DM):
    m = 6
    L_base = s3x3_lattice(spin_angles)
    L = gauge_trsf(L_base)
    s3x3_energy = energy(L,m,j,DM_angles,DM)
    return s3x3_energy
#ORBIT: q0->q0_g1->q0_g2
def q0(j,spin_angles,DM_angles,DM):
    m = 6
    L = q0_lattice(spin_angles)
    q0_energy = energy(L,m,j,DM_angles,DM)
    return q0_energy
def q0_g1(j,spin_angles,DM_angles,DM):
    m = 6
    L_base = q0_lattice(spin_angles)
    L = gauge_trsf(L_base)
    q0_energy = energy(L,m,j,DM_angles,DM)
    return q0_energy
def q0_g2(j,spin_angles,DM_angles,DM):
    m = 6
    L_base = q0_lattice(spin_angles)
    L_g1 = gauge_trsf(L_base)
    L = gauge_trsf(L_g1)
    q0_energy = energy(L,m,j,DM_angles,DM)
    return q0_energy
#Chiral orders -> difficult to compute by hand the energy (especially after the gauge transformation) -> automate the process
def octa(j,spin_angles,DM_angles,DM):
    m = 6
    L = oct_lattice(spin_angles)
    octa_energy = energy(L,m,j,DM_angles,DM)
    return octa_energy
def octa_g1(j,spin_angles,DM_angles,DM):
    m = 6
    L_base = oct_lattice(spin_angles)
    L = gauge_trsf(L_base)
    octa_energy = energy(L,m,j,DM_angles,DM)
    return octa_energy
def octa_g2(j,spin_angles,DM_angles,DM):
    m = 6
    L_base = oct_lattice(spin_angles)
    L_g1 = gauge_trsf(L_base)
    L = gauge_trsf(L_g1)
    octa_energy = energy(L,m,j,DM_angles,DM)
    return octa_energy
#
def cb1(j,spin_angles,DM_angles,DM):
    m = 6
    L = cb1_lattice(spin_angles)
    cb1_energy = energy(L,m,j,DM_angles,DM)
    return cb1_energy
def cb1_g1(j,spin_angles,DM_angles,DM):
    m = 6
    L_base = cb1_lattice(spin_angles)
    L = gauge_trsf(L_base)
    cb1_energy = energy(L,m,j,DM_angles,DM)
    return cb1_energy
def cb1_g2(j,spin_angles,DM_angles,DM):
    m = 6
    L_base = cb1_lattice(spin_angles)
    L_g1 = gauge_trsf(L_base)
    L = gauge_trsf(L_g1)
    cb1_energy = energy(L,m,j,DM_angles,DM)
    return cb1_energy
#
def cb2(j,spin_angles,DM_angles,DM):
    m = 6
    L = cb2_lattice(spin_angles)
    cb2_energy = energy(L,m,j,DM_angles,DM)
    return cb2_energy
def cb2_g1(j,spin_angles,DM_angles,DM):
    m = 6
    L_base = cb2_lattice(spin_angles)
    L = gauge_trsf(L_base)
    cb2_energy = energy(L,m,j,DM_angles,DM)
    return cb2_energy
def cb2_g2(j,spin_angles,DM_angles,DM):
    m = 6
    L_base = cb2_lattice(spin_angles)
    L_g1 = gauge_trsf(L_base)
    L = gauge_trsf(L_g1)
    cb2_energy = energy(L,m,j,DM_angles,DM)
    return cb2_energy
########################
########################
########################
def ferro_lattice(spin_angles):
    Sa = np.array([0,1/2,0])
    Ry = R_y(spin_angles[0])
    Rz = R_z(spin_angles[1])
    a = np.tensordot(Rz,np.tensordot(Ry,Sa,1),1)
    L = np.ndarray((18,18,3,3))
    L[0,0,0] = a; L[0,0,1] = a; L[0,0,2] = a
    #fill the lattice
    for i in range(0,18):
        for j in range(0,18):
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
    Uc = 3      #octahedral unit cells in each direction (3*3*12 sites in total)
    L = np.ndarray((18,18,3,3))
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
    for i in range(0,17,3):
        for j in range(0,17,3):
            L[i,j]      = L[0,0];   L[i,j+1]    = L[0,1];   L[i,j+2]    = L[0,2]
            L[i+1,j]    = L[1,0];   L[i+1,j+1]  = L[1,1];   L[i+1,j+2]  = L[1,2]
            L[i+2,j]    = L[2,0];   L[i+2,j+1]  = L[2,1];   L[i+2,j+2]  = L[2,2]
    return L
def q0_lattice(spin_angles):
    Sa = np.array([0,1/2,0])
    Sb = np.array([-np.sqrt(3)/4,-1/4,0])
    Sc = np.array([np.sqrt(3)/4,-1/4,0])
    Ry = R_y(spin_angles[0])
    Rz = R_z(spin_angles[1])
    a = np.tensordot(Rz,np.tensordot(Ry,Sa,1),1)
    b = np.tensordot(Rz,np.tensordot(Ry,Sb,1),1)
    c = np.tensordot(Rz,np.tensordot(Ry,Sc,1),1)
    Uc = 3      #octahedral unit cells in each direction (3*3*12 sites in total)
    L = np.ndarray((18,18,3,3))
    L[0,0,0] = b; L[0,0,1] = c; L[0,0,2] = a
    #fill the lattice
    for i in range(0,18):
        for j in range(0,18):
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
    L = np.ndarray((18,18,3,3))
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
    for i in range(0,17,2):
        for j in range(0,17,2):
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
    Ry = R_y(spin_angles[0])
    Rz = R_z(spin_angles[1])
    a = np.tensordot(Rz,np.tensordot(Ry,Sa,1),1)
    b = np.tensordot(Rz,np.tensordot(Ry,Sb,1),1)
    c = np.tensordot(Rz,np.tensordot(Ry,Sc,1),1)
    d = np.tensordot(Rz,np.tensordot(Ry,Sd,1),1)
    e = np.tensordot(Rz,np.tensordot(Ry,Se,1),1)
    f = np.tensordot(Rz,np.tensordot(Ry,Sf,1),1)
    Uc = 3      #octahedral unit cells in each direction (3*3*12 sites in total)
    L = np.ndarray((18,18,3,3))
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
    for i in range(0,17,2):
        for j in range(0,17,2):
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
    Ry = R_y(spin_angles[0])
    Rz = R_z(spin_angles[1])
    a = np.tensordot(Rz,np.tensordot(Ry,Sa,1),1)
    b = np.tensordot(Rz,np.tensordot(Ry,Sb,1),1)
    c = np.tensordot(Rz,np.tensordot(Ry,Sc,1),1)
    d = np.tensordot(Rz,np.tensordot(Ry,Sd,1),1)
    e = np.tensordot(Rz,np.tensordot(Ry,Se,1),1)
    f = np.tensordot(Rz,np.tensordot(Ry,Sf,1),1)
    L = np.ndarray((18,18,3,3))
    L[0,0,0] = f#-c
    L[0,0,1] = a#-d
    L[0,0,2] = -c#-b
    L[1,0,0] = b#-e
    L[1,0,1] = d#a
    L[1,0,2] = c#b
    L[0,1,0] = -f#c
    L[0,1,1] = -d#-a
    L[0,1,2] = -e#-f
    L[1,1,0] = -b#e
    L[1,1,1] = -a#d
    L[1,1,2] = e#f
    #fill the lattice
    for i in range(0,17,2):
        for j in range(0,17,2):
            L[i,j]      = L[0,0]
            L[i+1,j]    = L[1,0]
            L[i,j+1]    = L[0,1]
            L[i+1,j+1]  = L[1,1]
    return L





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





#################################################################################################à
#################################################################################################à
#################################################################################################à
#################################################################################################à
#################################################################################################à
#################################################################################################à
#################################################################################################à
#################################################################################################à



#define the search for the energy of the spiral states at a given J1,J2,J3
def spiral(j):
    spiral_energy = 100
    return spiral_energy

#define function for evaluation of minimal energy
def hopping_matrix(q,*args):
    j = args[0]
    a1 = np.array([1,0])
    a2 = np.array([-1/2,np.sqrt(3)/2])
    J_q = np.zeros((3,3),dtype=complex)
    J_q[0,0] = j[2]*2*np.cos(np.dot(q,a1))
    J_q[0,1] = j[1]*(1+np.exp(1j*np.dot(q,a1-a2)))
    J_q[0,2] = j[0]*(1+np.exp(-1j*np.dot(q,a2)))
    J_q[1,0] = np.conjugate(J_q[0,1])
    J_q[1,1] = j[2]*2*np.cos(np.dot(q,a2))
    J_q[1,2] = j[0]*(1+np.exp(-1j*np.dot(q,a1)))
    J_q[2,0] = np.conjugate(J_q[0,2])
    J_q[2,1] = np.conjugate(J_q[1,2])
    J_q[2,2] = j[2]*2*np.cos(np.dot(q,a1+a2))
    min_eigenvalue = eigh(J_q,eigvals_only = True)[0]
    return min_eigenvalue
def lower_bound_energy(j):
    arguments = (j,)
    min_energy = minimize(hopping_matrix,x0=(0,0),method='Nelder-Mead',bounds=((-2*np.pi,2*np.pi),(-4*np.pi,4*np.pi)),args=arguments)
    return min_energy.fun

