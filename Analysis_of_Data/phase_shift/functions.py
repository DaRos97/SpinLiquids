import numpy as np
import scipy.linalg as LA
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.optimize import minimize_scalar


###################################################################################
def totE(P_dic,args):
    L = P_dic['L']
    L_bounds = (L-1e-1,L+1e-1)
    P = []
    list_excluded = ['L','Energy','Sigma','gap']
    for p_t in P_dic.keys():
        if p_t not in list_excluded:
            P.append(P_dic[p_t])
    res = minimize_scalar(lambda l: -totEl(P,l,args),  #maximize energy wrt L with fixed P
            method = 'Brent',
            bracket = L_bounds,#bounds = inp.L_bounds,
            options={'xtol':1e-10}
            )
    L = res.x   #optimized L
    minE = -res.fun #optimized energy(total)
    if minE < -10:
        minE = np.nan
    return minE, L
#####################################################################################
def totEl(P,L,args):
    J1,J2,J3,ans,phi = args
    J = (J1,J2,J3)
    j2 = np.sign(int(np.abs(J2)*1e8))   #check if it is 0 or 1 --> problem for VERY small J2,J3 points
    j3 = np.sign(int(np.abs(J3)*1e8))
    Res = 0
    n = 0
    z = [4,4,2]
    Pp = np.zeros(6)
    Pp[0] = P[n]
    list_A2 = ['q0_1','cb1','cb1_nc']
    list_A3 = ['3x3_1','cb1','cb1_nc']
    list_B3 = ['q0_1','3x3_1']
    if ans in list_A2 and j2:
        n += 1
        Pp[1] = P[n]
    if ans in list_A3 and j3:
        n += 1
        Pp[2] = P[n]
    n += 1
    Pp[3] = P[n] #B1
    if j2:
        n += 1
        Pp[4] = P[n] #B2
    if ans in list_B3 and j3:
        n += 1
        Pp[5] = P[n]
    for i in range(3):
        Res += z[i]*(Pp[i]**2-Pp[i+3]**2)*J[i]/2
    S = 0.5
    Res -= L*(2*S+1)
    #
    N = N_(P,L,args) #compute Hermitian matrix
    Nx = 13
    Ny = 13
    m = 6
    res = np.zeros((m,Nx,Ny))
    for i in range(Nx):
        for j in range(Ny):
            Nk = N[:,:,i,j]
            try:
                Ch = LA.cholesky(Nk)     #not always the case since for some parameters of Lambda the eigenmodes are negative
            except LA.LinAlgError:      #matrix not pos def for that specific kx,ky
                return -100      #if that's the case even for a single k in the grid, return a defined value
            temp = np.dot(np.dot(Ch,J_),np.conjugate(Ch.T))    #we need the eigenvalues of M=KJK^+ (also Hermitian)
            res[:,i,j] = LA.eigvalsh(temp)[m:]
    r2 = 0
    for i in range(m):
        func = RBS(kxg,kyg,res[i])
        r2 += func.integral(0,1,0,1)
    r2 /= m
    gap = np.amin(res[0].ravel())
    #
    Res += r2
    return Res

#################################################################################àù
Nx = 13
Ny = 13
kxg = np.linspace(0,1,Nx)
kyg = np.linspace(0,1,Ny)
kkg = np.ndarray((2,Nx,Ny),dtype=complex)
for i in range(Nx):
    for j in range(Ny):
        kkg[0,i,j] = kxg[i]*2*np.pi
        kkg[1,i,j] = (kxg[i]+kyg[j])*2*np.pi/np.sqrt(3)
#### vectors of 1nn, 2nn and 3nn
a1 = (1,0)
a2 = (-1,np.sqrt(3))
a12p = (a1[0]+a2[0],a1[1]+a2[1])
a12m = (a1[0]-a2[0],a1[1]-a2[1])
ka1 = np.exp(1j*np.tensordot(a1,kkg,axes=1));   ka1_ = np.conjugate(ka1);
ka2 = np.exp(1j*np.tensordot(a2,kkg,axes=1));   ka2_ = np.conjugate(ka2);
ka12p = np.exp(1j*np.tensordot(a12p,kkg,axes=1));   ka12p_ = np.conjugate(ka12p);
ka12m = np.exp(1j*np.tensordot(a12m,kkg,axes=1));   ka12m_ = np.conjugate(ka12m);
####
J_ = np.zeros((2*6,2*6))
for i in range(6):
    J_[i,i] = -1
    J_[i+6,i+6] = 1
#### all ansatze
def N_(P,L,args):
    #### DM
    phi = args[-1]
    DM1 = phi
    DM2 = 0
    DM3 = phi*2
    t1 = np.exp(-1j*DM1);    t1_ = np.conjugate(t1)
    t2 = np.exp(-1j*DM2);    t2_ = np.conjugate(t2)
    t3 = np.exp(-1j*DM3);    t3_ = np.conjugate(t3)
    m = 6
    J1,J2,J3,ans,phi = args
    J1 /= 2.
    J2 /= 2.
    J3 /= 2.
    j2 = np.sign(int(np.abs(J2)*1e8))   #check if it is 0 or not --> problem with VERY small J2,J3
    j3 = np.sign(int(np.abs(J3)*1e8))
    A1 = P[0]
    #params
    if ans == '3x3_1':
        A2 = 0;     phiA2 = 0;    phiA2p = 0;
        A3 = P[1*j3]*j3
        B1 = P[2*j3]*j3 + P[1]*(1-j3)
        B2 = P[3*j2*j3]*j2*j3+P[2*j2*(1-j3)]*(1-j3)*j2
        B3 = P[4*j3*j2]*j3*j2+P[3*j3*(1-j2)]*j3*(1-j2)
        phiB1 = P[-3*j3]*j3*j2 + P[-2]*j3*(1-j2) + P[-2]*j2*(1-j3) + P[-1]*(1-j2)*(1-j3)
        phiB2 = P[-2]*j2*j3 + P[-1]*j2*(1-j3)
        phiA3 = P[-1]*j3
        phiA1p = np.pi
        phiB1p, phiB2p, phiB3 = (phiB1, phiB2, np.pi)
        p1 = 0
    elif ans == '3x3_2':
        A2 = 0;     phiA2 = 0;    phiA2p = 0;
        A3 = P[1*j3]*j3
        B1 = P[2*j3]*j3 + P[1]*(1-j3)
        B2 = P[3*j2*j3]*j2*j3+P[2*j2*(1-j3)]*(1-j3)*j2
        B3 = P[4*j3*j2]*j3*j2+P[3*j3*(1-j2)]*j3*(1-j2)
        phiA1p = P[-4*j3*j2]*j3*j2 + P[-3*j3]*j3*(1-j2) + P[-3*j2]*j2*(1-j3) + P[-2]*(1-j2)*(1-j3)
        phiB1 = P[-3*j3*j2]*j3*j2 + P[-2]*j3*(1-j2) + P[-2]*j2*(1-j3) + P[-1]*(1-j2)*(1-j3)
        phiB2 = P[-2]*j2*j3 + P[-1]*j2*(1-j3)
        phiB3 = P[-1]*j3
        phiA3 = (phiA1p+np.pi)/2
        phiB1p, phiB2p = (-phiB1, -phiB2)
        p1 = 0
    elif ans == 'q0_1':
        A3 = 0; phiA3 = 0
        A2 = P[1]*j2
        B1 = P[2*j2]*j2+P[1]*(1-j2)
        B2 = P[3*j2]*j2
        B3 = P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2)
        phiB1 = P[-3*j2]*j2 + P[-1]*(1-j2)
        phiA2 = P[-2]*j2
        phiB2 = P[-1]*j2
        phiA1p, phiA2p = (0, phiA2)
        phiB1p, phiB2p, phiB3 = (phiB1, phiB2, 0)
        p1 = 0
    elif ans == 'q0_2':
        A3 = 0; phiA3 = 0
        A2 = P[1]*j2
        B1 = P[2*j2]*j2+P[1]*(1-j2)
        B2 = P[3*j2]*j2
        B3 = P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2)
        phiA1p = P[-5*j3*j2]*j3*j2 + P[-3*j3*(1-j2)]*j3*(1-j2) + P[-4*j2*(1-j3)]*j2*(1-j3) + P[-2]*(1-j2)*(1-j3)
        phiB1 = P[-4*j3*j2]*j3*j2 + P[-2]*j3*(1-j2) + P[-3*j2]*j2*(1-j3) + P[-1]*(1-j2)*(1-j3)
        phiA2 = P[-3*j2*j3]*j2*j3 + P[-2]*j2*(1-j3)
        phiB2 = P[-2]*j2*j3 + P[-1]*j2*(1-j3)
        phiB3 = P[-1]*j3
        phiA2p = phiA1p - phiA2
        phiB1p ,phiB2p = (-phiB1, -phiB2)
        p1 = 0
    elif ans == 'cb1':      #A1,A2,A3,B1,B2,phiA1p,phiB2
        B3 = 0; phiB3 = 0
        A2 = P[1*j2]*j2
        A3 = P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2)
        B1 = P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3)
        B2 = P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3)
        phiA1p = P[-4*j2]*j2 + P[-2]*(1-j2)
        phiB1 = P[-3*j2]*j2 + P[-1]*(1-j2)
        phiA2 = P[-2]*j2
        phiB2 = P[-1]*j2
        phiA2p, phiA3 = (phiA1p-phiA2,phiA1p/2)
        phiB1p, phiB2p= (-phiB1 ,-phiB2)
        p1 = 1
    elif ans == 'cb1_nc':      #A1,A2,A3,B1,B2,phiA1p,phiB2
        B3 = 0; phiB3 = 0
        A2 = P[1*j2]*j2
        A3 = P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2)
        B1 = P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3)
        B2 = P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3)
        phiA1p = P[-4*j2]*j2 + P[-2]*(1-j2)
        phiB1 = P[-3*j2]*j2 + P[-1]*(1-j2)
        phiA2 = P[-2]*j2
        phiB2 = P[-1]*j2
        phiA2p, phiA3 = (phiA1p-phiA2,phiA1p/2+np.pi)
        phiB1p, phiB2p= (-phiB1 ,-phiB2)
        p1 = 1
    elif ans == 'oct':
        A3 = 0; phiA3 = 0
        A2 = P[1]*j2
        B1 = P[2*j2]*j2+P[1]*(1-j2)
        B2 = P[3*j2]*j2
        B3 = P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2)
        phiB1 = P[-3*j2]*j2 + P[-1]*(1-j2)
        phiA2 = P[-2]*j2
        phiB2 = P[-1]*j2
        phiA1p, phiA2p = (np.pi, phiA2+np.pi)
        phiB1p, phiB2p, phiB3 = (phiB1, phiB2, 3/2*np.pi)
        p1 = 1
    ################
    N = np.zeros((2*m,2*m,Nx,Ny), dtype=complex)
    ##################################### B
    b1 = B1*np.exp(1j*phiB1);               b1_ = np.conjugate(b1)
    b1p = B1*np.exp(1j*phiB1p);             b1p_ = np.conjugate(b1p)
    b1pi = B1*np.exp(1j*(phiB1p+p1*np.pi)); b1pi_ = np.conjugate(b1pi)
    b2 = B2*np.exp(1j*phiB2);               b2_ = np.conjugate(b2)
    b2i = B2*np.exp(1j*(phiB2+p1*np.pi));   b2i_ = np.conjugate(b2i)
    b2p = B2*np.exp(1j*phiB2p);             b2p_ = np.conjugate(b2p)
    b2pi = B2*np.exp(1j*(phiB2p+p1*np.pi)); b2pi_ = np.conjugate(b2pi)
    b3 = B3*np.exp(1j*phiB3);               b3_ = np.conjugate(b3)
    b3i = B3*np.exp(1j*(phiB3+p1*np.pi));   b3i_ = np.conjugate(b3i)
    #
    N[0,1] = J1*b1p_ *ka1  *t1_              + J2*b2*t2
    N[0,2] = J1*b1p        *t1               + J2*b2p_ *ka1*t2
    N[0,4] = J1*b1_  *ka2_ *t1               + J2*b2pi *ka12m*t2_
    N[0,5] = J1*b1   *ka2_ *t1_              + J2*b2i_ *ka12p_*t2_
    N[1,2] = J1*(b1_       *t1  + b1p_*ka1_*t1_)
    N[1,3] = J1*b1         *t1_              + J2*b2p_ *ka1_*t2
    N[1,5] =                                   J2*(b2  *ka12p_*t2 + b2p*t2_)
    N[2,3] = J1*b1_        *t1               + J2*b2   *ka1*t2
    N[2,4] =                                   J2*(b2p_*ka2_*t2 + b2i_*ka1*t2_)
    N[3,4] = J1*b1pi_*ka1  *t1_              + J2*b2*t2
    N[3,5] = J1*b1p        *t1               + J2*b2pi_*ka1*t2
    N[4,5] = J1*(b1_       *t1  + b1pi_*ka1_*t1_)

    N[0,0] = J3*b3i_ *ka1_ *t3_
    N[3,3] = J3*b3_  *ka1_ *t3_
    N[1,4] = J3*(b3_ *ka2_ *t3_ + b3       *t3)
    N[2,5] = J3*(b3  *ka12p_  *t3  + b3i_*ka1*t3_)
    ####other half square
    N[m+0,m+1] = J1*b1p  *ka1  *t1_           + J2*b2_*t2
    N[m+0,m+2] = J1*b1p_       *t1            + J2*b2p  *ka1*t2
    N[m+0,m+4] = J1*b1   *ka2_ *t1            + J2*b2pi_*ka12m*t2_
    N[m+0,m+5] = J1*b1_  *ka2_ *t1_           + J2*b2i  *ka12p_*t2_
    N[m+1,m+2] = J1*(b1        *t1  + b1p*ka1_*t1_)
    N[m+1,m+3] = J1*b1_        *t1_           + J2*b2p  *ka1_*t2
    N[m+1,m+5] =                                J2*(b2_ *ka12p_*t2 + b2p_*t2_)
    N[m+2,m+3] = J1*b1         *t1            + J2*b2_  *ka1*t2
    N[m+2,m+4] =                                J2*(b2p *ka2_*t2 + b2i *ka1*t2_)
    N[m+3,m+4] = J1*b1pi *ka1  *t1_           + J2*b2_*t2
    N[m+3,m+5] = J1*b1p_       *t1            + J2*b2pi *ka1*t2
    N[m+4,m+5] = J1*(b1        *t1  + b1pi*ka1_*t1_)

    N[m+0,m+0] = J3*b3i *ka1_ *t3_
    N[m+3,m+3] = J3*b3  *ka1_ *t3_
    N[m+1,m+4] = J3*(b3 *ka2_ *t3_ + b3_  *t3)
    N[m+2,m+5] = J3*(b3_*ka12p_  *t3  + b3i *ka1*t3_)
    ######################################## A
    a1 =    A1
    a1p =   A1*np.exp(1j*phiA1p)
    a1pi =  A1*np.exp(1j*(phiA1p+p1*np.pi))
    a2 =    A2*np.exp(1j*phiA2)
    a2i =   A2*np.exp(1j*(phiA2+p1*np.pi))
    a2p =   A2*np.exp(1j*phiA2p)
    a2pi =  A2*np.exp(1j*(phiA2p+p1*np.pi))
    a3 =    A3*np.exp(1j*phiA3)
    a3i =   A3*np.exp(1j*(phiA3+p1*np.pi))
    N[0,m+1] = - J1*a1p *ka1 *t1_           +J2*a2*t2
    N[0,m+2] =   J1*a1p      *t1            -J2*a2p  *ka1*t2
    N[0,m+4] = - J1*a1  *ka2_*t1            +J2*a2pi *ka12m*t2_
    N[0,m+5] =   J1*a1  *ka2_*t1_           -J2*a2i  *ka12p_*t2_
    N[1,m+2] = - J1*(a1      *t1   +a1p*ka1_*t1_)
    N[1,m+3] =   J1*a1       *t1_           -J2*a2p  *ka1_*t2
    N[1,m+5] =                               J2*(a2  *ka12p_*t2  +a2p*t2_)
    N[2,m+3] = - J1*a1       *t1            +J2*a2   *ka1*t2
    N[2,m+4] =                              -J2*(a2p *ka2_*t2  +a2i*ka1*t2_)
    N[3,m+4] = - J1*a1pi*ka1 *t1_           +J2*a2*t2
    N[3,m+5] =   J1*a1p      *t1            -J2*a2pi *ka1*t2
    N[4,m+5] = - J1*(a1      *t1   +a1pi*ka1_*t1_)

    N[0,m+0] = - J3*a3i *ka1_*t3_
    N[3,m+3] = - J3*a3  *ka1_*t3_
    N[1,m+4] = - J3*(a3 *ka2_*t3_  -a3 *t3)
    N[2,m+5] = - J3*(a3i*ka1*t3_  -a3 *ka12p_ *t3)
    #not the diagonal
    N[1,m]   =   J1*a1p *ka1_*t1            -J2*a2*t2_
    N[2,m]   = - J1*a1p      *t1_           +J2*a2p  *ka1_*t2_
    N[4,m]   =   J1*a1  *ka2 *t1_           -J2*a2pi *ka12m_*t2
    N[5,m]   = - J1*a1  *ka2 *t1            +J2*a2i  *ka12p*t2
    N[2,m+1] =   J1*(a1      *t1_  +a1p*ka1 *t1)
    N[3,m+1] = - J1*a1       *t1            +J2*a2p  *ka1_*t2_
    N[5,m+1] =                              -J2*(a2  *ka12p*t2_   +a2p*t2)
    N[3,m+2] =   J1*a1       *t1_           -J2*a2   *ka1_*t2_
    N[4,m+2] =                               J2*(a2p *ka2*t2_   +a2i*ka1_*t2)
    N[4,m+3] =   J1*a1pi*ka1_*t1            -J2*a2*t2_
    N[5,m+3] = - J1*a1p      *t1_           +J2*a2pi *ka1_*t2_
    N[5,m+4] =   J1*(a1      *t1_  +a1pi*ka1 *t1)

    N[0,m+0] +=  J3*a3i *ka1  *t3
    N[3,m+3] +=  J3*a3  *ka1  *t3
    N[4,m+1] =   J3*(a3 *ka2 *t3   -a3 *t3_)
    N[5,m+2] =   J3*(a3i*ka1_ *t3   -a3 *ka12p *t3_)
    #################################### HERMITIAN MATRIX
    #N += np.conjugate(N.transpose((1,0,2,3)))
    for i in range(2*m):
        for j in range(i,2*m):
            N[j,i] += np.conjugate(N[i,j])
    #################################### L
    for i in range(2*m):
        N[i,i] += L
    return N

