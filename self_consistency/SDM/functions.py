import inputs as inp
import numpy as np
from scipy import linalg as LA
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline as RBS
from pathlib import Path
import csv
import os
#Libraries needed only for debug and plotting -> not used in the cluster
#from colorama import Fore
#import matplotlib.pyplot as plt
#from matplotlib import cm


def total_energy(P,L,args):
    KM,Tau,K_,S,ans,DM_type,L_bounds = args
    p1 = 0 if ans in inp.ansatze_p0 else 1
    m = inp.m[p1]
    J_ = np.zeros((2*m,2*m))
    for i in range(m):
        J_[i,i] = -1
        J_[i+m,i+m] = 1
    A1 = P[0]
    if ans in inp.ansatze_1:
        B1 = P[1]
    else:
        B1 = P[2]
    Res = inp.z*(A1**2-B1**2)/2
    Res -= L*(2*S+1)            #part of the energy coming from the Lagrange multiplier
    #Compute now the (painful) part of the energy coming from the Hamiltonian matrix by the use of a Bogoliubov transformation
    args2 = (KM,Tau,K_,S,ans,DM_type)
#    big_Nk = big_Nk_uniform if DM_type == 'uniform' else big_Nk_staggered
    N = big_Nk(P,L,args2)                #compute Hermitian matrix from the ansatze coded in the ansatze.py script
    res = np.zeros((m,K_,K_))
    for i in range(K_):                 #cicle over all the points in the Brilluin Zone grid
        for j in range(K_):
            Nk = N[:,:,i,j]                 #extract the corresponding matrix
            try:
                Ch = LA.cholesky(Nk)        #not always the case since for some parameters of Lambda the eigenmodes are negative
            except LA.LinAlgError:          #matrix not pos def for that specific kx,ky
                print("not pos def!!!!!!!!!!!!!!")
                return 0,0           #if that's the case even for a single k in the grid, return a defined value
            temp = np.dot(np.dot(Ch,J_),np.conjugate(Ch.T))    #we need the eigenvalues of M=KJK^+ (also Hermitian)
            a = LA.eigvalsh(temp)      #BOTTLE NECK -> compute the eigevalues
            res[:,i,j] = a[m:]
    gap = np.amin(res.ravel())           #the gap is the lowest value of the lowest gap (not in the fitting if not could be negative in principle)
    #Now fit the energy values found with a spline curve in order to have a better solution
    r2 = 0
    for i in range(m):
        func = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),res[i])
        r2 += func.integral(0,1,0,1)        #integrate the fitting curves to get the energy of each band
    r2 /= m                             #normalize
    #Summation over k-points
    #r3 = res.ravel().sum() / len(res.ravel())
    energy = Res + r2
    return energy, gap

#### Computes Energy from Parameters P, by maximizing it wrt the Lagrange multiplier L. Calls only totEl function
def compute_L(P,args):
    res = minimize_scalar(lambda l: optimize_L(P,l,args),  #maximize energy wrt L with fixed P
            method = inp.L_method,          #can be 'bounded' or 'Brent'
            bracket = args[-1],             
            options={'xtol':inp.prec_L}
            )
    L = res.x                       #optimized L
    return L

#### Computes the Energy given the paramters P and the Lagrange multiplier L. 
#### This is the function that does the actual work.
def optimize_L(P,L,args):
    KM,Tau,K_,S,ans,DM_type,L_bounds = args
    p1 = 0 if ans in inp.ansatze_p0 else 1
    m = inp.m[p1]
    J_ = np.zeros((2*m,2*m))
    VL = 100
    for i in range(m):
        J_[i,i] = -1
        J_[i+m,i+m] = 1
    if L < L_bounds[0]:
        Res = -VL-(L_bounds[0]-L)
        return -Res
    elif L > L_bounds[1]:
        Res = -VL-(L-L_bounds[1])
        return -Res
    Res = -L*(2*S+1)            #part of the energy coming from the Lagrange multiplier
    #Compute now the (painful) part of the energy coming from the Hamiltonian matrix by the use of a Bogoliubov transformation
    args2 = (KM,Tau,K_,S,ans,DM_type)
#    big_Nk = big_Nk_uniform if DM_type == 'uniform' else big_Nk_staggered
    N = big_Nk(P,L,args2)                #compute Hermitian matrix from the ansatze coded in the ansatze.py script
    res = np.zeros((m,K_,K_))
    for i in range(K_):                 #cicle over all the points in the Brilluin Zone grid
        for j in range(K_):
            Nk = N[:,:,i,j]                 #extract the corresponding matrix
            try:
                Ch = LA.cholesky(Nk)        #not always the case since for some parameters of Lambda the eigenmodes are negative
            except LA.LinAlgError:          #matrix not pos def for that specific kx,ky
                r4 = -VL+(L-L_bounds[0])
                result = -(Res+r4)
#                print('e:\t',L,result)
                return result           #if that's the case even for a single k in the grid, return a defined value
            temp = np.dot(np.dot(Ch,J_),np.conjugate(Ch.T))    #we need the eigenvalues of M=KJK^+ (also Hermitian)
            res[:,i,j] = LA.eigvalsh(temp)[m:]      #BOTTLE NECK -> compute the eigevalues
    gap = np.amin(res[0].ravel())           #the gap is the lowest value of the lowest gap (not in the fitting if not could be negative in principle)
    #Now fit the energy values found with a spline curve in order to have a better solution
    r2 = 0
    for i in range(m):
        func = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),res[i])
        r2 += func.integral(0,1,0,1)        #integrate the fitting curves to get the energy of each band
    r2 /= m                             #normalize
    #Summation over k-pts
    #r2 = res.ravel().sum() / len(res.ravel())
    result = -(Res+r2)
#    print('g:\t',L,result)
    return result

def compute_O_all(old_O,L,args):
    new_O = np.zeros(len(old_O))
    KM,Tau,K_,S,pars,ans,DM_type = args
    p1 = 0 if ans in inp.ansatze_p0 else 1
    m = inp.m[p1]
    J_ = np.zeros((2*m,2*m))
    for i in range(m):
        J_[i,i] = -1
        J_[i+m,i+m] = 1
    #Compute first the transformation matrix M at each needed K
    args_M = (KM,Tau,K_,S,ans,DM_type)
#    big_Nk = big_Nk_uniform if DM_type == 'uniform' else big_Nk_staggered
    N = big_Nk(old_O,L,args_M)
    M = np.zeros(N.shape,dtype=complex)
    for i in range(K_):
        for j in range(K_):
            N_k = N[:,:,i,j]
            Ch = LA.cholesky(N_k) #upper triangular-> N_k=Ch^{dag}*Ch
            w,U = LA.eigh(np.dot(np.dot(Ch,J_),np.conjugate(Ch.T)))
            w = np.diag(np.sqrt(np.einsum('ij,j->i',J_,w)))
            M[:,:,i,j] = np.dot(np.dot(LA.inv(Ch),U),w)
    #for each parameter need to know what it is
    dic_O = {'A':compute_A,'B':compute_B}
    kxg = np.linspace(0,1,K_)
    kyg = np.linspace(0,1,K_)
    for p in range(len(pars)):
        par = pars[p]
        par_ = par[-2:] if par[-1]=='p' else par[-1]
        par_1 = par[-2] if par[-1]=='p' else par[-1]
        par_2 = 'A' if 'A' in par else 'B'
        li_ = dic_indexes[str(m)][par_][0]
        lj_ = dic_indexes[str(m)][par_][1]
        Tau_ = (Tau[2*(int(par_1)-1)],Tau[2*(int(par_1)-1)+1])
        DM_ch = True if par_ in ['1p','2p','3'] else False
        func = dic_O[par_2]
        #res = 0
        rrr = np.zeros((K_,K_),dtype=complex)
        for i in range(K_):
            for j in range(K_):
                K__ = np.array([kxg[i]*2*np.pi,(kxg[i]+2*kyg[j])*2*np.pi/np.sqrt(3)])
                U,X,V,Y = split(M[:,:,i,j],m,m)
                U_,V_,X_,Y_ = split(np.conjugate(M[:,:,i,j].T),m,m)
#                res += func(U,X,V,Y,U_,X_,V_,Y_,Tau,li_,lj_)
                rrr[i,j] = func(U,X,V,Y,U_,X_,V_,Y_,Tau_,li_,lj_,K__,DM_ch)
        interI = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),np.imag(rrr))
        res2I = interI.integral(0,1,0,1)
        interR = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),np.real(rrr))
        res2R = interR.integral(0,1,0,1)
        res = (res2R+1j*res2I)/2
        #res /= 2*K_**2
        if par[0] == 'p':
            new_O[p] = np.angle(res)
            if new_O[p] < 0:
                new_O[p] += 2*np.pi
            if new_O[p] > np.pi and par == 'phiA1p':
                new_O[p] = 2*np.pi - new_O[p]
        else:
            new_O[p] = np.absolute(res)
#        print(par,new_O[p])
#    input()
    return new_O

dic_indexes =   {'3':{'1': (1,2), '1p': (2,0), 
                      '2': (1,0), '2p': (2,1), 
                      '3': (1,1)},
                 '6':{'1': (1,2), '1p': (2,0), 
                      '2': (1,0), '2p': (5,1), 
                      '3': (4,1)}
                 }
def compute_A(U,X,V,Y,U_,X_,V_,Y_,Tau__,li_,lj_,K__,DM_ch):
    if DM_ch:
        Tau__ = np.conjugate(np.array(Tau__))
    if (li_,lj_) in [(2,1),(1,1)]:
        dist = np.array([-1/2,np.sqrt(3)/2])
    else:
        dist = np.zeros(2)
    return (np.einsum('ln,nm->lm',U,V_)[li_,lj_]     *Tau__[1] *np.exp(1j*np.dot(K__,dist))
            - np.einsum('nl,mn->lm',Y_,X)[li_,lj_]   *Tau__[0] *np.exp(-1j*np.dot(K__,dist)))
########
def compute_B(U,X,V,Y,U_,X_,V_,Y_,Tau__,li_,lj_,K__,DM_ch):
    if DM_ch:
        Tau__ = np.conjugate(np.array(Tau__))
    if (li_,lj_) in [(2,1),(1,1)]:
        dist = np.array([-1/2,np.sqrt(3)/2])
    else:
        dist = np.zeros(2)
    return (np.einsum('nl,mn->lm',X_,X)[li_,lj_]  *Tau__[0] *np.exp(-1j*np.dot(K__,dist))
            + np.einsum('ln,nm->lm',V,V_)[li_,lj_]*Tau__[1] *np.exp(1j*np.dot(K__,dist)))

def split(array, nrows, ncols):
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

#### Ansatze encoded in the matrix
def big_Nk(P,L,args):
    KM,Tau,K_,S,ans,DM_type = args
    p1 = 0 if ans in inp.ansatze_p0 else 1
    m = inp.m[p1]
    J1 = 1
    ka1,ka1_,ka2,ka2_,ka12p,ka12p_,ka12m,ka12m_ = KM
    t1,t1_ = Tau
    J1 /= 2.
    t1_diff = t1 if DM_type == 'uniform' else t1_
    t1_diff_ = np.conjugate(t1)
    #
    A1 = P[0]
    A1p = A1
    phiB1 = P[-1]
    if ans in ['14','15','17']:
        phiA1p = 0
    elif ans in ['16','18']:
        phiA1p = np.pi
    else:
        phiA1p = P[1]
    if ans in inp.ansatze_1:
        B1 = P[1]
        phiB1p = -phiB1
    else:
        B1 = P[2]
        phiB1p = phiB1
    B1p = B1
    #
    N = np.zeros((2*m,2*m,K_,K_), dtype=complex)
    b1 = B1*np.exp(1j*phiB1);               b1_ = np.conjugate(b1)
    b1p = B1p*np.exp(1j*phiB1p);             b1p_ = np.conjugate(b1p)
    b1pi = B1p*np.exp(1j*(phiB1p+p1*np.pi)); b1pi_ = np.conjugate(b1pi)
    a1 =    A1
    a1p =   A1p*np.exp(1j*phiA1p)
    a1pi =  A1p*np.exp(1j*(phiA1p+p1*np.pi))
    #
    N[0,1] = J1*b1p_ *ka1  *t1_diff_             # + J2*b2*t2                         #t1
    N[0,2] = J1*b1p        *t1_diff              # + J2*b2p_ *ka1*t2                  #t1_
    N[1,2] = J1*(b1_       *t1  + b1p_*ka1_*t1_diff_)                                #t1     t1
    ####other half square                                                       #Same ts
    N[m+0,m+1] = J1*b1p  *ka1  *t1_diff_         #  + J2*b2_*t2
    N[m+0,m+2] = J1*b1p_       *t1_diff          #  + J2*b2p  *ka1*t2
    N[m+1,m+2] = J1*(b1        *t1  + b1p*ka1_*t1_diff_)
    #   A
    N[0,m+1] = - J1*a1p *ka1 *t1_diff_          # +J2*a2*t2
    N[0,m+2] =   J1*a1p      *t1_diff           # -J2*a2p  *ka1*t2
    N[1,m+2] = - J1*(a1      *t1   +a1p*ka1_*t1_diff_)
    ####other half square (not the diagonal)                                        
    N[1,m]   =   J1*a1p *ka1_*t1_diff           # -J2*a2*t2_
    N[2,m]   = - J1*a1p      *t1_diff_          # +J2*a2p  *ka1_*t2_
    N[2,m+1] =   J1*(a1      *t1_  +a1p*ka1 *t1_diff)
    if m == 6:
        #   B
        N[0,4] = J1*b1_  *ka2_ *t1              # + J2*b2pi *ka12m*t2_               #t1
        N[0,5] = J1*b1   *ka2_ *t1_             # + J2*b2i_ *ka12p_*t2_              #t1_
        N[1,3] = J1*b1         *t1_             # + J2*b2p_ *ka1_*t2                 #t1_
        N[2,3] = J1*b1_        *t1              # + J2*b2   *ka1*t2                  #t1
        N[3,4] = J1*b1pi_*ka1  *t1_diff_             # + J2*b2*t2                         #t1
        N[3,5] = J1*b1p        *t1_diff              # + J2*b2pi_*ka1*t2                  #t1_
        N[4,5] = J1*(b1_       *t1  + b1pi_*ka1_*t1_diff_)                               #t1
        #
        N[m+0,m+4] = J1*b1   *ka2_ *t1           # + J2*b2pi_*ka12m*t2_
        N[m+0,m+5] = J1*b1_  *ka2_ *t1_          # + J2*b2i  *ka12p_*t2_
        N[m+1,m+3] = J1*b1_        *t1_          # + J2*b2p  *ka1_*t2
        N[m+2,m+3] = J1*b1         *t1           # + J2*b2_  *ka1*t2
        N[m+3,m+4] = J1*b1pi *ka1  *t1_diff_          # + J2*b2_*t2
        N[m+3,m+5] = J1*b1p_       *t1_diff           # + J2*b2pi *ka1*t2
        N[m+4,m+5] = J1*(b1        *t1  + b1pi*ka1_*t1_diff_)
        #   A
        N[0,m+4] = - J1*a1  *ka2_*t1           # +J2*a2pi *ka12m*t2_
        N[0,m+5] =   J1*a1  *ka2_*t1_          # -J2*a2i  *ka12p_*t2_
        N[1,m+3] =   J1*a1       *t1_          # -J2*a2p  *ka1_*t2
        N[2,m+3] = - J1*a1       *t1           # +J2*a2   *ka1*t2
        N[3,m+4] = - J1*a1pi*ka1 *t1_diff_          # +J2*a2*t2
        N[3,m+5] =   J1*a1p      *t1_diff           # -J2*a2pi *ka1*t2
        N[4,m+5] = - J1*(a1      *t1   +a1pi*ka1_*t1_diff_)  
        #
        N[4,m]   =   J1*a1  *ka2 *t1_          # -J2*a2pi *ka12m_*t2
        N[5,m]   = - J1*a1  *ka2 *t1           # +J2*a2i  *ka12p*t2
        N[3,m+1] = - J1*a1       *t1           # +J2*a2p  *ka1*t2_               #Second term was ka1_
        N[3,m+2] =   J1*a1       *t1_          # -J2*a2   *ka1_*t2_
        N[4,m+3] =   J1*a1pi*ka1_*t1_diff           # -J2*a2*t2_
        N[5,m+3] = - J1*a1p      *t1_diff_          # +J2*a2pi *ka1_*t2_
        N[5,m+4] =   J1*(a1      *t1_  +a1pi*ka1 *t1_diff)
    else:
        #   B
        N[0,1] += J1*b1_  *ka2_ *t1            #   + J2*b2pi *ka12m*t2_               #t1
        N[0,2] += J1*b1   *ka2_ *t1_           #   + J2*b2i_ *ka12p_*t2_              #t1_
        #
        N[m+0,m+1] += J1*b1  *ka2_ *t1         #      + J2*b2pi_ *ka12m*t2_               #t1
        N[m+0,m+2] += J1*b1_   *ka2_ *t1_      #        + J2*b2i *ka12p_*t2_              #t1_
        #   A
        N[0,m+1] += - J1*a1  *ka2_ *t1         #      + J2*a2pi *ka12m*t2_               #t1
        N[0,m+2] +=   J1*a1  *ka2_ *t1_        #      - J2*a2i  *ka12p_*t2_              #t1_
        #
        N[1,m] +=   J1*a1  *ka2 *t1_           #    - J2*a2pi *ka12m_*t2               #t1
        N[2,m] += - J1*a1  *ka2 *t1            #    + J2*a2i  *ka12p *t2              #t1_
    #################################### HERMITIAN MATRIX
    for i in range(2*m):
        for j in range(i,2*m):
            N[j,i] += np.conjugate(N[i,j])
    #################################### L
    for i in range(2*m):
        N[i,i] += L
    return N


def compute_KM(kkg,a1,a2,a12p,a12m):
    ka1 = np.exp(1j*np.tensordot(a1,kkg,axes=1));   ka1_ = np.conjugate(ka1);
    ka2 = np.exp(1j*np.tensordot(a2,kkg,axes=1));   ka2_ = np.conjugate(ka2);
    ka12p = np.exp(1j*np.tensordot(a12p,kkg,axes=1));   ka12p_ = np.conjugate(ka12p);
    ka12m = np.exp(1j*np.tensordot(a12m,kkg,axes=1));   ka12m_ = np.conjugate(ka12m);
    KM = (ka1,ka1_,ka2,ka2_,ka12p,ka12p_,ka12m,ka12m_)
    return KM














