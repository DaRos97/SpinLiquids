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


def total_energy(P_,L_,args):
    KM,Tau,K_,S,p1,L_bounds = args
    P = np.copy(P_)
    P[0] *= S
    P[1] *= S
    P[3] *= S
    P[5] *= S
    L = L_ * S
    m = inp.m[p1]
    J_ = np.zeros((2*m,2*m))
    for i in range(m):
        J_[i,i] = -1
        J_[i+m,i+m] = 1
    Res = inp.z*(P[0]**2+P[1]**2-P[3]**2-P[5]**2)/4#*S**2
    Res -= L*(2*S+1)#*S            #part of the energy coming from the Lagrange multiplier
    #Compute now the (painful) part of the energy coming from the Hamiltonian matrix by the use of a Bogoliubov transformation
    args2 = (KM,Tau,K_,S,p1)
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
    energy = (Res + r2)#*S)        #back to real values?????
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
    KM,Tau,K_,S,p1,L_bounds = args
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
    args2 = (KM,Tau,K_,S,p1)
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
    KM,Tau,K_,S,pars,p1 = args
    m = inp.m[p1]
    J_ = np.zeros((2*m,2*m))
    for i in range(m):
        J_[i,i] = -1
        J_[i+m,i+m] = 1
    #Compute first the transformation matrix M at each needed K
    args_M = (KM,Tau,K_,S,p1)
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
    for p in range(len(pars)):
        par = pars[p]
        par_ = par[-2:] if par[-1]=='p' else par[-1]
        par_1 = par[-2] if par[-1]=='p' else par[-1]
        par_2 = 'A' if 'A' in par else 'B'
        li_ = dic_indexes[par_][0]
        lj_ = dic_indexes[par_][1]
        Tau_ = (Tau[2*(int(par_1)-1)],Tau[2*(int(par_1)-1)+1])
        func = dic_O[par_2]
        #res = 0
        rrr = np.zeros((K_,K_),dtype=complex)
        for i in range(K_):
            for j in range(K_):
                U,X,V,Y = split(M[:,:,i,j],m,m)
                U_,V_,X_,Y_ = split(np.conjugate(M[:,:,i,j].T),m,m)
#                res += func(U,X,V,Y,U_,X_,V_,Y_,Tau,li_,lj_)
                rrr[i,j] = func(U,X,V,Y,U_,X_,V_,Y_,Tau_,li_,lj_)
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
            new_O[p] = np.absolute(res)/S                   #renormalization of amplitudes 
#        print(par,new_O[p])
#    input()
    return new_O

dic_indexes = { '1': (1,2), '1p': (2,0), 
                '2': (1,0), '2p': (5,1), 
                '3': (4,1)
                }
def compute_A(U,X,V,Y,U_,X_,V_,Y_,Tau,li_,lj_):
    if li_== 2 or li_ == 4:
        Tau = np.conjugate(np.array(Tau))
    return (np.einsum('ln,nm->lm',U,V_)[li_,lj_]     *Tau[1] 
            - np.einsum('nl,mn->lm',Y_,X)[li_,lj_]   *Tau[0])
def compute_B(U,X,V,Y,U_,X_,V_,Y_,Tau,li_,lj_):
    if li_== 2 or li_ == 4:
        Tau = np.conjugate(np.array(Tau))
    return (np.einsum('nl,mn->lm',X_,X)[li_,lj_]  *Tau[0] 
            + np.einsum('ln,nm->lm',V,V_)[li_,lj_]*Tau[1])

def split(array, nrows, ncols):
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))


#### Ansatze encoded in the matrix
def big_Nk(P,L,args):
    KM,Tau,K_,S,p1 = args
    m = inp.m[p1]
    J1 = 1
    ka1,ka1_,ka2,ka2_,ka12p,ka12p_,ka12m,ka12m_ = KM
    t1,t1_ = Tau
    J1 /= 2.
    A1,A1p,phiA1p,B1,phiB1,B1p,phiB1p = P
    ################
    N = np.zeros((2*m,2*m,K_,K_), dtype=complex)
    ##################################### B
    b1 = B1*np.exp(1j*phiB1);               b1_ = np.conjugate(b1)
    b1p = B1p*np.exp(1j*phiB1p);             b1p_ = np.conjugate(b1p)
    b1pi = B1p*np.exp(1j*(phiB1p+p1*np.pi)); b1pi_ = np.conjugate(b1pi)
    #
    N[0,1] = J1*b1p_ *ka1  *t1_              
    N[0,2] = J1*b1p        *t1   
    N[1,2] = J1*(b1_       *t1  + b1p_*ka1_*t1_)                                #t1     t1
    N[0,4%m] = J1*b1_  *ka2_ *t1               
    N[0,5%m] = J1*b1   *ka2_ *t1_
    N[3%m,4%m] += J1*b1pi_*ka1  *t1_              
    N[3%m,5%m] += J1*b1p        *t1               
    N[4%m,5%m] = J1*(b1_       *t1  + b1pi_*ka1_*t1_)                               #t1     t1
    ####other half square                                                       #Same ts
    N[m+0,m+1] = J1*b1p  *ka1  *t1_           
    N[m+0,m+2] = J1*b1p_       *t1            
    N[m+1,m+2] = J1*(b1        *t1  + b1p*ka1_*t1_)
    N[m+0,m+4%m] = J1*b1   *ka2_ *t1            
    N[m+0,m+5%m] = J1*b1_  *ka2_ *t1_           
    N[m+3%m,m+4%m] += J1*b1pi *ka1  *t1_           
    N[m+3%m,m+5%m] += J1*b1p_       *t1            
    N[m+4%m,m+5%m] = J1*(b1        *t1  + b1pi*ka1_*t1_)
    ######################################## A
    a1 =    A1
    a1p =   A1p*np.exp(1j*phiA1p)
    a1pi =  A1p*np.exp(1j*(phiA1p+p1*np.pi))
    N[0,m+1] = - J1*a1p *ka1 *t1_           
    N[0,m+2] =   J1*a1p      *t1            
    N[1,m+2] = - J1*(a1      *t1   +a1p*ka1_*t1_)
    N[0,m+4%m] = - J1*a1  *ka2_*t1            
    N[0,m+5%m] =   J1*a1  *ka2_*t1_           
    N[3%m,m+4%m] += - J1*a1pi*ka1 *t1_           
    N[3%m,m+5%m] +=   J1*a1p      *t1            
    N[4%m,m+5%m] = - J1*(a1      *t1   +a1pi*ka1_*t1_)
    #not the diagonal
    N[1,m]   =   J1*a1p *ka1_*t1            
    N[2,m]   = - J1*a1p      *t1_           
    N[2,m+1] =   J1*(a1      *t1_  +a1p*ka1 *t1)
    N[4%m,m]   =   J1*a1  *ka2 *t1_           
    N[5%m,m]   = - J1*a1  *ka2 *t1            
    N[4%m,m+3%m] +=   J1*a1pi*ka1_*t1            
    N[5%m,m+3%m] += - J1*a1p      *t1_           
    N[5%m,m+4%m] =   J1*(a1      *t1_  +a1pi*ka1 *t1)
    ############################################### Terms which are different between m = 3,6
    if m == 6:
        N[1,3%m] = J1*b1         *t1_
        N[2,3%m] = J1*b1_        *t1               
        N[m+1,m+3%m] = J1*b1_        *t1_           
        N[m+2,m+3%m] = J1*b1         *t1            
        N[1,m+3%m] =   J1*a1       *t1_           
        N[2,m+3%m] = - J1*a1       *t1            
        N[3%m,m+1] = - J1*a1       *t1            
        N[3%m,m+2] =   J1*a1       *t1_           
    #################################### HERMITIAN MATRIX
    #N += np.conjugate(N.transpose((1,0,2,3)))
    for i in range(2*m):
        for j in range(i,2*m):
            N[j,i] += np.conjugate(N[i,j])
    #################################### L
    for i in range(2*m):
        N[i,i] += L
    return N


def KM(kkg,a1,a2,a12p,a12m):
    ka1 = np.exp(1j*np.tensordot(a1,kkg,axes=1));   ka1_ = np.conjugate(ka1);
    ka2 = np.exp(1j*np.tensordot(a2,kkg,axes=1));   ka2_ = np.conjugate(ka2);
    ka12p = np.exp(1j*np.tensordot(a12p,kkg,axes=1));   ka12p_ = np.conjugate(ka12p);
    ka12m = np.exp(1j*np.tensordot(a12m,kkg,axes=1));   ka12m_ = np.conjugate(ka12m);
    KM = (ka1,ka1_,ka2,ka2_,ka12p,ka12p_,ka12m,ka12m_)
    return KM














