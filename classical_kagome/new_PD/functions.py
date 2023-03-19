import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
for i in range(0,9,3):
    for j in range(0,9,3):
        Gauge_trsf[i,j]   = Gauge_trsf[0,0]; Gauge_trsf[i+1,j]   = Gauge_trsf[1,0]; Gauge_trsf[i+2,j]   = Gauge_trsf[2,0];
        Gauge_trsf[i,j+1] = Gauge_trsf[0,1]; Gauge_trsf[i+1,j+1] = Gauge_trsf[1,1]; Gauge_trsf[i+2,j+1] = Gauge_trsf[2,1];
        Gauge_trsf[i,j+2] = Gauge_trsf[0,2]; Gauge_trsf[i+1,j+2] = Gauge_trsf[1,2]; Gauge_trsf[i+2,j+2] = Gauge_trsf[2,2];
def gauge_trsf(L):
    #Make the gauge transformation on a lattice 18*18*3 sites
    for x in range(9):
        for y in range(9):
            for m in range(3):
                L[x,y,m] = np.tensordot(R_z(Gauge_trsf[x,y,m]*2/3*np.pi),L[x,y,m],1)
    return L
########
########
########

Z_ = [4,4,2]
#define regular orders with DM angles as functions of J1,J2,J3, the DM angle and also a representative spin orientation
def energy_2(L,M,j,DM_angles):
    energy_1nn = 0
    energy_2nn = 0
    energy_3nn = 0
    DM1,DM2,DM3 = DM_angles
    for x in range(1,1+M):
        for y in range(1,1+M):
            s0 = L[x,y,0]; s0_2 = L[x-1,y-1,2]; s0_1 = L[x-1,y,1]
            s1 = L[x,y,1]; s1_0 = L[x+1,y,0];   s1_2 = L[x,y-1,2]
            s2 = L[x,y,2]; s2_0 = L[x+1,y+1,0]; s2_1 = L[x,y+1,1]
            energy_1nn += ( dots(s0,s2,-DM1) + dots(s0,s0_2,DM1) + dots(s0,s1,DM1) + dots(s0,s0_1,-DM1) +
                            dots(s1,s0,-DM1) + dots(s1,s1_0,DM1) + dots(s1,s2,DM1) + dots(s1,s1_2,-DM1) +
                            dots(s2,s0,DM1) + dots(s2,s2_0,-DM1) + dots(s2,s1,-DM1) + dots(s2,s2_1,DM1)
                            )   * j[0]
            s0a = L[x,y+1,1];   s0b = L[x,y-1,2]
            s1a = L[x-1,y-1,2]; s1b = L[x+1,y+1,0]
            s2a = L[x+1,y,0];   s2b = L[x-1,y,1]
            energy_2nn += j[1]*(np.dot(s0,s0a) + np.dot(s0,s0b) +
                                     np.dot(s1,s1a) + np.dot(s1,s1b) +
                                     np.dot(s2,s2a) + np.dot(s2,s2b)
                                     ) 
            s0 = L[x,y,0]; s0_a = L[x,y+1,0]; s0_b = L[x,y-1,0]
            s1 = L[x,y,1]; s1_a = L[x-1,y-1,1]; s1_b = L[x+1,y+1,1]
            s2 = L[x,y,2]; s2_a = L[x+1,y,2]; s2_b = L[x-1,y,2]
            energy_3nn += ( dots(s0,s0_a,DM3) + dots(s0,s0_b,-DM3) + 
                            dots(s1,s1_a,DM3) + dots(s1,s1_b,-DM3) +
                            dots(s2,s2_a,DM3) + dots(s2,s2_b,-DM3) 
                            )   *j[2]
    return (energy_1nn + energy_2nn*2 + energy_3nn)/(M*M*3)
#
bounds_lat = (   (0,np.pi),                  #t_0
                 (0,2*np.pi),                  #t_1
            )
def lat_energy(L_,m,j,DM):
    j = tuple(j)
    L = np.copy(L_)
    minimization = differential_evolution(
        energy_new,
        bounds = bounds_lat,
        args = (L,m,j,DM),

        popsize = 50,
        tol = 1e-8,
        polish = True,
        maxiter = 100,
        workers = -1,
        updating = 'deferred',
#        disp = True

    )
    temp = energy_new(minimization.x,*(L,m,j,DM))
    res = minimization.x
    return temp,res
def energy_new(angles,*args):
    L_,M,j,DM = args
    L = rotate(np.copy(L_),angles)
    en = energy_2(L,M,j,DM)
    return en
def rotate(L,angles):
    Ry = R_y(angles[0])
    Rz = R_z(angles[1])
    for i in range(9):
        for j in range(9):
            for m in range(3):
                L[i,j,m] = np.tensordot(Rz,np.tensordot(Ry,L[i,j,m],1),1)
    return L
#
#
#

def lat_energy_old(ind,m,j,DM):
    j = tuple(j)
    minimization = differential_evolution(
        energy_old,
        bounds = bounds_lat,
        args = (ind,m,j,DM),

        popsize = 50,
        tol = 1e-8,
        polish = True,
        maxiter = 1000,
        workers = -1,
        updating = 'deferred',
        disp = True

    )
    temp = energy(minimization.x,*(ind,m,j,DM))
    res = minimization.x
    return temp,res
#
def energy_old(angles,*args):
    ind,M,j,DM = args
    lattice_func = list_lattices[ind//3]
    L = lattice_func(angles)
    for i in range(ind%3):
        L = gauge_trsf(L)
    en = energy_2(L,M,j,DM)
    return en
#
########################
########################
########################
def ferro_lattice(spin_angles):
    Sa = np.array([0,1/2,0])
    Ry = R_y(spin_angles[0])
    Rz = R_z(spin_angles[1])
    a = np.tensordot(Rz,np.tensordot(Ry,Sa,1),1)
    L = np.ndarray((9,9,3,3))
    L[0,0,0] = a; L[0,0,1] = a; L[0,0,2] = a
    #fill the lattice
    for i in range(9):
        for j in range(9):
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
    L = np.ndarray((9,9,3,3))
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
    for i in range(0,9,3):
        for j in range(0,9,3):
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
    L = np.ndarray((9,9,3,3))
    L[0,0,0] = b; L[0,0,1] = c; L[0,0,2] = a
    #fill the lattice
    for i in range(0,9):
        for j in range(0,9):
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
    L = np.ndarray((9,9,3,3))
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
    for i in range(0,8,2):
        for j in range(0,8,2):
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
    L = np.ndarray((9,9,3,3))
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
    for i in range(0,8,2):
        for j in range(0,8,2):
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
    L = np.ndarray((9,9,3,3))
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
    for i in range(0,8,2):
        for j in range(0,8,2):
            L[i,j]      = L[0,0]
            L[i+1,j]    = L[1,0]
            L[i,j+1]    = L[0,1]
            L[i+1,j+1]  = L[1,1]
    return L
list_lattices = [ferro_lattice,q0_lattice,oct_lattice,cb1_lattice,cb2_lattice]





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
def dots(S_a,S_b,DM):
    return S_a[2]*S_b[2] + np.cos(2*DM)*(S_a[0]*S_b[0]+S_a[1]*S_b[1]) + np.sin(2*DM)*(S_a[1]*S_b[0]-S_a[0]*S_b[1])

#Sites are numbered counter-clockwise, in two up-pointing small triangles starting from lower left. The two small triangles are along a2->(-1/2,sqrt(3)/2)
def spiral_energy(P,*args):         
    J,I,DM = args
    J1,J2,J3 = J
    inv_1, inv_2 = I
    DM1, DM2, DM3 = DM
    p_0 = 0
    t_0,t_1,p_1,t_2,p_2,t_3,p_3,t_4,p_4,t_5,p_5,P_1,P_2 = P
    S_0 = 0.5*np.array([np.sin(t_0)*np.cos(p_0),np.sin(t_0)*np.sin(p_0),np.cos(t_0)])
    S_1 = 0.5*np.array([np.sin(t_1)*np.cos(p_1),np.sin(t_1)*np.sin(p_1),np.cos(t_1)])
    S_2 = 0.5*np.array([np.sin(t_2)*np.cos(p_2),np.sin(t_2)*np.sin(p_2),np.cos(t_2)])
    S_3 = 0.5*np.array([np.sin(t_3)*np.cos(p_3),np.sin(t_3)*np.sin(p_3),np.cos(t_3)])
    S_4 = 0.5*np.array([np.sin(t_4)*np.cos(p_4),np.sin(t_4)*np.sin(p_4),np.cos(t_4)])
    S_5 = 0.5*np.array([np.sin(t_5)*np.cos(p_5),np.sin(t_5)*np.sin(p_5),np.cos(t_5)])
    E1 = (dots(S_0,S_2,-DM1)    +dots(S_0,inv_1*inv_2*np.tensordot(R_z(-P_1-P_2),S_5,1),DM1) +
          dots(S_0,S_1,DM1)     +dots(S_0,inv_1*np.tensordot(R_z(-P_1),S_1,1),-DM1)+
          dots(S_1,S_0,-DM1)    +dots(S_1,inv_1*np.tensordot(R_z(P_1),S_0,1),DM1) +
          dots(S_1,S_2,DM1)     +dots(S_1,inv_2*np.tensordot(R_z(-P_2),S_5,1),-DM1) +
          dots(S_2,S_0,DM1)     +dots(S_2,inv_1*np.tensordot(R_z(P_1),S_3,1),-DM1) +
          dots(S_2,S_1,-DM1)    +dots(S_2,S_4,DM1) + 
          dots(S_3,S_5,-DM1)    +dots(S_3,inv_1*np.tensordot(R_z(-P_1),S_2,1),DM1) +
          dots(S_3,S_4,DM1)     +dots(S_3,inv_1*np.tensordot(R_z(-P_1),S_4,1),-DM1) +
          dots(S_4,S_3,-DM1)    +dots(S_4,inv_1*np.tensordot(R_z(P_1),S_3,1),DM1) +
          dots(S_4,S_5,DM1)     +dots(S_4,S_2,-DM1)         +
          dots(S_5,S_3,DM1)     +dots(S_5,inv_1*inv_2*np.tensordot(R_z(P_1+P_2),S_0,1),-DM1) +
          dots(S_5,S_4,-DM1)    +dots(S_5,inv_2*np.tensordot(R_z(P_2),S_1,1),DM1)         )
    E2 = (  np.dot(S_0,inv_2*inv_1*np.tensordot(R_z(-P_1-P_2),S_4,1))   +np.dot(S_0,inv_2*np.tensordot(R_z(-P_2),S_5,1)) +
            np.dot(S_0,inv_1*np.tensordot(R_z(-P_1),S_2,1))             +np.dot(S_0,S_4) + 
            np.dot(S_1,inv_2*np.tensordot(R_z(-P_2),S_3,1))             +np.dot(S_1,inv_1*inv_2*np.tensordot(R_z(-P_2-P_1),S_5,1))  +
            np.dot(S_1,inv_1*np.tensordot(R_z(P_1),S_3,1))              +np.dot(S_1,inv_1*np.tensordot(R_z(P_1),S_2,1))      +
            np.dot(S_2,S_3)                                             +np.dot(S_2,inv_1*np.tensordot(R_z(P_1),S_4,1))      +
            np.dot(S_5,inv_1*np.tensordot(R_z(-P_1),S_4,1))             +np.dot(S_5,inv_1*np.tensordot(R_z(P_1),S_3,1))
            )
    E3 = (  dots(S_0,S_3,DM3)       + dots(S_0,inv_2*np.tensordot(R_z(-P_2),S_3,1),-DM3) +
            dots(S_1,inv_1*inv_2*np.tensordot(R_z(-P_1-P_2),S_4,1),DM3)     + dots(S_1,inv_1*np.tensordot(R_z(P_1),S_4,1),-DM3) + 
            dots(S_2,inv_1*np.tensordot(R_z(-P_1),S_2,1),-DM3)      + dots(S_2,inv_1*np.tensordot(R_z(P_1),S_2,1),DM3)  + 
            dots(S_3,S_0,-DM3)      + dots(S_3,inv_2*np.tensordot(R_z(P_2),S_0,1),DM3) +
            dots(S_4,inv_1*np.tensordot(R_z(-P_1),S_1,1),DM3)       + dots(S_4,inv_1*inv_2*np.tensordot(R_z(P_1+P_2),S_1,1),-DM3) + 
            dots(S_5,inv_1*np.tensordot(R_z(-P_1),S_5,1),-DM3)      + dots(S_5,inv_1*np.tensordot(R_z(P_1),S_5,1),DM3)   
            )
    return (J1*E1+J2*E2*2+J3*E3)/6
#define the search for the energy of the spiral states at a given J1,J2,J3
bounds_spiral = (   (0,np.pi),                  #t_0
                    (0,np.pi),                  #t_1
                    (0,2*np.pi),                #p_1
                    (0,np.pi),                  #t_2
                    (0,2*np.pi),                #p_2
                    (0,np.pi),                  #t_3
                    (0,2*np.pi),                #p_3
                    (0,np.pi),                  #t_4
                    (0,2*np.pi),                #p_4
                    (0,np.pi),                  #t_5
                    (0,2*np.pi),                #p_5
                    (0,2*np.pi),                #P_1
                    (0,2*np.pi),                #P_2
                    )
def spiral(j,DM_angles):
    j = tuple(j)
    SS_energy = 1e5
    for inv_1 in [1,-1]:
        for inv_2 in [1,-1]:
            minimization = differential_evolution(
                spiral_energy,
                bounds = bounds_spiral,
                popsize = 50,
                tol = 1e-8,
                polish = True,
                maxiter = 1000,
                args = (j,(inv_1,inv_2),DM_angles),
#                workers = -1,
#                updating = 'deferred',
#                disp = True
            )
            temp = spiral_energy(minimization.x,*(j,(inv_1,inv_2),DM_angles))
            if temp < SS_energy:
                SS_energy = temp
                inv_final = np.array([inv_1,inv_2])
                res = minimization.x
#            print(inv_1,inv_2,temp,*minimization.x)
    return SS_energy,np.append(inv_final,res)

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





#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
def find_order(P,args):
    disp,plot = args
    CO = 1e-3
    COc = 1e-2
    p_0 = 0
    inv_1,inv_2,t_0,t_1,p_1,t_2,p_2,t_3,p_3,t_4,p_4,t_5,p_5,P_1,P_2 = P
    T_ = [t_0,t_1,t_2,t_3,t_4,t_5]
    P_ = [p_0,p_1,p_2,p_3,p_4,p_5]
    #change possible 2*pi to 0
    for i in range(6):
        if np.abs(P_[i] - 2*np.pi) < CO:
            P_[i] -= 2*np.pi
    #Find if the solution is O(3) or SO(3)
    chir = True if inv_1 != inv_2 else False
    #Find if the UC is 3 or 6 sites
    p1 = 0
    for i in range(3):
        if not(np.abs(P_[i]-P_[i+3]) < CO or np.abs(P_[i]-P_[i+3]-2*np.pi) < CO):
            p1 =1
    #Find if unit cell looks like planar q0/ferro
    q0 = False
    FM = False
    if p1==0:
        q0 = True
        for p in P_[1:3]:       #p_1 and p_2
            if not (np.abs(p-np.pi/3*2) < CO or np.abs(p-np.pi/3*4) < CO):
                q0 = False
        FM = True
        for p in P_[1:3]:
            if not (np.abs(p) < CO):
                FM = False
    if not FM:
        FM = True
        for t in T_:
            if not(np.abs(t) < CO or np.abs(t-np.pi) < CO):
                FM = False
    #Study rotations: pi and 0
    p1p = 1 if abs(P_1-np.pi) < CO else 0
    p2p = 1 if abs(P_2-np.pi) < CO else 0
    p1z = 1 if (abs(P_1) < CO or abs(P_1-2*np.pi) < CO) else 0
    p2z = 1 if (abs(P_2) < CO or abs(P_2-2*np.pi) < CO) else 0
    #pi/3 and 2pi/3
    p1c = 1 if abs(P_1-np.pi/3) < CO else 0
    p2c = 1 if abs(P_1-np.pi/3) < CO else 0
    p1d = 1 if abs(P_1-2*np.pi/3) < CO else 0
    p2d = 1 if abs(P_1-2*np.pi/3) < CO else 0
    if (p1p or p1z) and (p2p or p2z):
        UC = p1p*(1-p2p)*12 + (1-p1p)*p2p*12 + p1p*p2p*24
        R = 0
    elif (p1c or p1d) and (p2c or p2d):
        UC = p1c*(1-p2c)*18 + (1-p1c)*p2c*18 + p1c*p2c*54
        R = 1
    else:
        UC = 1000000000
        R = 2
    #Compute chirality of single triangles
    S_0 = 0.5*np.array([np.sin(t_0)*np.cos(p_0),np.sin(t_0)*np.sin(p_0),np.cos(t_0)])
    S_1 = 0.5*np.array([np.sin(t_1)*np.cos(p_1),np.sin(t_1)*np.sin(p_1),np.cos(t_1)])
    S_2 = 0.5*np.array([np.sin(t_2)*np.cos(p_2),np.sin(t_2)*np.sin(p_2),np.cos(t_2)])
    S_3 = 0.5*np.array([np.sin(t_3)*np.cos(p_3),np.sin(t_3)*np.sin(p_3),np.cos(t_3)])
    S_4 = 0.5*np.array([np.sin(t_4)*np.cos(p_4),np.sin(t_4)*np.sin(p_4),np.cos(t_4)])
    S_5 = 0.5*np.array([np.sin(t_5)*np.cos(p_5),np.sin(t_5)*np.sin(p_5),np.cos(t_5)])
    S_3r = inv_1*np.tensordot(R_z(P_1),S_3,1)
    S_1u = inv_2*np.tensordot(R_z(P_2),S_1,1)
    S_0ur = inv_1*inv_2*np.tensordot(R_z(P_2+P_1),S_0,1)
    chirality_up1 = np.dot(S_0,np.cross(S_1,S_2))
    chirality_up2 = np.dot(S_3,np.cross(S_4,S_5))
    chirality_dw1 = np.dot(S_2,np.cross(S_1,S_3r))
    chirality_dw2 = np.dot(S_5,np.cross(S_0ur,S_1u))
    planar = True if np.abs(chirality_up1) < COc else False
    ####
    if disp:
        print('Inversions: ',*P[:2])
        print('Phases theta: ',*T_)
        print('Phases phi: ',*P_)
        print('Rotations: ',*P[-2:])
        print("deducted: ")
        print("chr = ",chir,"\tp1 = ",p1,"\tplanar = ",planar)
        print("q0: ",q0)
        print("FM: ",FM)
        print("Rotation UC = ",UC)
        print("Chirality of small triangles up: ",chirality_up1,chirality_up2)
        print("Chirality of small triangles dw: ",chirality_dw1,chirality_dw2)
        input()
    if plot:
        fig = plt.figure(figsize =(12,12))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim3d(-4, 4)
        ax.set_ylim3d(-4, 4)
        ax.set_zlim3d(-1, 1)
        X = [0,1/2,1/4,-1/2,0,-1/4]
        Y = [0,0,np.sqrt(3)/4,np.sqrt(3)/2,np.sqrt(3)/2,3/4*np.sqrt(3)]
        L = np.zeros((6,6,6,3))
        S_uc = [S_0,S_1,S_2,S_3,S_4,S_5]
        for x in range(6):
            for y in range(6):
                for m in range(6):
                    I1 = 1 if x//2 else inv_1
                    I2 = 1 if y//2 else inv_2
                    S = I1*I2*np.tensordot(R_z(x*P_1+y*P_2),S_uc[m],1)
                    L[x,y,m] = S
        for x in range(4):
            for y in range(2):
                for m in range(6):
                    X0 = x-y+X[m]
                    Y0 = y*np.sqrt(3)+Y[m]
                    Z0 = 0
                    Xf = L[x,y,m,0]
                    Yf = L[x,y,m,1]
                    Zf = L[x,y,m,2]
                    ax.quiver(X0, Y0, Z0, Xf, Yf, Zf,normalize = False)#, length = 0.05, normalize = False)
                    ax.scatter(X0,Y0,Z0,c='k',marker='o')
        plt.show()
    color1 = 'g' if planar else 'r'
    color2 = color1
    ark   = 'v' if chir else '^'
    fills = 'right'# if np.abs(chirality_up1) > COc else 'top'
    if q0:
        color2 = 'blue'
    elif FM:
        color1 = 'k'
        color2 = 'k'
        ark = 'o'
    if p1p or p2p or p1z or p2z:
        color2 = 'aqua'
#        fills = 'right'
    if p1c or p2c or p1d or p2d:
#        color2 = ''    
        fills = 'top'
        color2 = 'orange'
    Marker = dict(
            color= color1,
            marker=   ark,
            markeredgecolor='none',
            markerfacecoloralt= color2,
            fillstyle=  fills,
            markersize=20,
        )

    return Marker









































