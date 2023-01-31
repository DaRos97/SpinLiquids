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

#### Matrix diagonal of -1 and 1
J_ = np.zeros((2*inp.m,2*inp.m))
for i in range(inp.m):
    J_[i,i] = -1
    J_[i+inp.m,i+inp.m] = 1

def total_energy(P,L,args):
    J1,J2,J3,ans,KM,Tau,K_,S,L_bounds = args
    J = (J1,J2,J3)
    j2 = np.sign(int(np.abs(J2)*1e8))   #check if it is 0 or 1 --> problem for VERY small J2,J3 points
    j3 = np.sign(int(np.abs(J3)*1e8))
    Res = 0                         #resulting energy
    n = 0                           #?????
    Pp = np.zeros(6)
    Pp[0] = P[n]
    #Compute the part of the energy coming from the moduli of the parameters (out of the Hamiltonian matrix)
    if ans in inp.list_A2 and j2:
        n += 1
        Pp[1] = P[n]
    if ans in inp.list_A3 and j3:
        n += 1
        Pp[2] = P[n]
    n += 1
    Pp[3] = P[n] #B1
    if j2:
        n += 1
        Pp[4] = P[n] #B2
    if ans in inp.list_B3 and j3:
        n += 1
        Pp[5] = P[n]
    for i in range(3):
        Res += inp.z[i]*(Pp[i]**2-Pp[i+3]**2)*J[i]/2
    Res -= L*(2*S+1)            #part of the energy coming from the Lagrange multiplier
    #Compute now the (painful) part of the energy coming from the Hamiltonian matrix by the use of a Bogoliubov transformation
    args2 = (J1,J2,J3,ans,KM,Tau,K_)
    N = big_Nk(P,L,args2)                #compute Hermitian matrix from the ansatze coded in the ansatze.py script
    res = np.zeros((inp.m,K_,K_))
    for i in range(K_):                 #cicle over all the points in the Brilluin Zone grid
        for j in range(K_):
            Nk = N[:,:,i,j]                 #extract the corresponding matrix
            try:
                Ch = LA.cholesky(Nk)        #not always the case since for some parameters of Lambda the eigenmodes are negative
            except LA.LinAlgError:          #matrix not pos def for that specific kx,ky
                print("not pos def!!!!!!!!!!!!!!")
                r4 = -3+(L-L_bounds[0])
                return np.nan,np.nan           #if that's the case even for a single k in the grid, return a defined value
            temp = np.dot(np.dot(Ch,J_),np.conjugate(Ch.T))    #we need the eigenvalues of M=KJK^+ (also Hermitian)
            res[:,i,j] = LA.eigvalsh(temp)[inp.m:]      #BOTTLE NECK -> compute the eigevalues
    #Now fit the energy values found with a spline curve in order to have a better solution
    gap = np.amin(res[0].ravel())           #the gap is the lowest value of the lowest gap (not in the fitting if not could be negative in principle)
    r2 = 0
    for i in range(inp.m):
        func = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),res[i])
        r2 += func.integral(0,1,0,1)        #integrate the fitting curves to get the energy of each band
    r2 /= inp.m                             #normalize
    #r2 = res.ravel().sum()
    #r2 /= len(res.ravel())
    return Res + r2, gap

#### Computes Energy from Parameters P, by maximizing it wrt the Lagrange multiplier L. Calls only totEl function
def compute_L(P,args):
    res = minimize_scalar(lambda l: -optimize_L(P,l,args),  #maximize energy wrt L with fixed P
            method = inp.L_method,          #can be 'bounded' or 'Brent'
            bracket = args[-1],             
            options={'xtol':inp.prec_L}
            )
    L = res.x                       #optimized L
    return L

#### Computes the Energy given the paramters P and the Lagrange multiplier L. 
#### This is the function that does the actual work.
def optimize_L(P,L,args):
    J1,J2,J3,ans,KM,Tau,K_,S,L_bounds = args
    Res = -L*(2*S+1)            #part of the energy coming from the Lagrange multiplier
    #Compute now the (painful) part of the energy coming from the Hamiltonian matrix by the use of a Bogoliubov transformation
    args2 = (J1,J2,J3,ans,KM,Tau,K_)
    N = big_Nk(P,L,args2)                #compute Hermitian matrix from the ansatze coded in the ansatze.py script
    res = np.zeros((inp.m,K_,K_))
    for i in range(K_):                 #cicle over all the points in the Brilluin Zone grid
        for j in range(K_):
            Nk = N[:,:,i,j]                 #extract the corresponding matrix
            try:
                Ch = LA.cholesky(Nk)        #not always the case since for some parameters of Lambda the eigenmodes are negative
            except LA.LinAlgError:          #matrix not pos def for that specific kx,ky
                r4 = -1+(L-L_bounds[0])
                return Res+r4           #if that's the case even for a single k in the grid, return a defined value
            temp = np.dot(np.dot(Ch,J_),np.conjugate(Ch.T))    #we need the eigenvalues of M=KJK^+ (also Hermitian)
            res[:,i,j] = LA.eigvalsh(temp)[inp.m:]      #BOTTLE NECK -> compute the eigevalues
    #Now fit the energy values found with a spline curve in order to have a better solution
    gap = np.amin(res[0].ravel())           #the gap is the lowest value of the lowest gap (not in the fitting if not could be negative in principle)
    r2 = 0
    for i in range(inp.m):
        func = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),res[i])
        r2 += func.integral(0,1,0,1)        #integrate the fitting curves to get the energy of each band
    r2 /= inp.m                             #normalize
    #r2 = res.ravel().sum()
    #r2 /= len(res.ravel())
    return Res + r2

def compute_O_all(old_O,L,args,p_):
    new_O = np.zeros(len(old_O))
    J1,J2,J3,ans,KM,Tau,K_,pars = args
    if -np.angle(Tau[0]) < np.pi/3+1e-4 and -np.angle(Tau[0]) > np.pi/3-1e-4:   #for DM = pi/3(~1.04) A1 and B1 change sign
        p104 = -1
    else:
        p104 = 1
    #Compute first the transformation matrix M at each needed K
    args_M = (J1,J2,J3,ans,KM,Tau,K_)
    m = 6
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
        par_2 = 'A' if 'A' in par else 'B'
        li_ = dic_indexes[par_][0]
        lj_ = dic_indexes[par_][1]
        func = dic_O[par_2]
        #res = 0
        rrr = np.zeros((K_,K_),dtype=complex)
        for i in range(K_):
            for j in range(K_):
                U,X,V,Y = split(M[:,:,i,j],inp.m,inp.m)
                U_,V_,X_,Y_ = split(np.conjugate(M[:,:,i,j].T),inp.m,inp.m)
        #        res += func(U,X,V,Y,U_,X_,V_,Y_,Tau,li_,lj_)
                rrr[i,j] = func(U,X,V,Y,U_,X_,V_,Y_,Tau,li_,lj_)
        interI = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),np.imag(rrr))
        res2I = interI.integral(0,1,0,1)
        interR = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),np.real(rrr))
        res2R = interR.integral(0,1,0,1)
        res = (res2R+1j*res2I)/2
        #res /= 2*K_**2
        res *= p104
        if par[0] == 'p':
            new_O[p] = np.angle(res)
        else:
            new_O[p] = np.absolute(res)
    #    print(par,res)
    #input()
    return new_O

#Compute new sing par using old O and new L
def compute_O_sing(old_O,L,args,p_):
    new_O = 0
    J1,J2,J3,ans,KM,Tau,K_,pars = args
    if -np.angle(Tau[0]) < np.pi/3+1e-4 and -np.angle(Tau[0]) > np.pi/3-1e-4:   #for DM = pi/3(~1.04) A1 and B1 change sign
        p104 = -1
    else:
        p104 = 1
    #Compute first the transformation matrix M at each needed K
    args_M = (J1,J2,J3,ans,KM,Tau,K_)
    m = 6
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
    par = pars[p_]
    par_ = par[-2:] if par[-1]=='p' else par[-1]
    par_2 = 'A' if 'A' in par else 'B'
    li_ = dic_indexes[par_][0]
    lj_ = dic_indexes[par_][1]
    func = dic_O[par_2]
    #res = 0
    rrr = np.zeros((K_,K_),dtype=complex)
    for i in range(K_):
        for j in range(K_):
            U,X,V,Y = split(M[:,:,i,j],inp.m,inp.m)
            U_,V_,X_,Y_ = split(np.conjugate(M[:,:,i,j].T),inp.m,inp.m)
    #        res += func(U,X,V,Y,U_,X_,V_,Y_,Tau,li_,lj_)
            rrr[i,j] = func(U,X,V,Y,U_,X_,V_,Y_,Tau,li_,lj_)
    interI = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),np.imag(rrr))
    res2I = interI.integral(0,1,0,1)
    interR = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),np.real(rrr))
    res2R = interR.integral(0,1,0,1)
    res = (res2R+1j*res2I)/2
    #res /= 2*K_**2
    res *= p104
    if par[0] == 'p':
        new_O = np.angle(res)
    else:
        new_O = np.absolute(res)
#    print(par,res)
    #input()
    return new_O

dic_indexes = { '1': (1,2), '1p': (2,0), 
                '2': (1,0), '2p': (5,1), 
                '3': (4,1)
                }
def compute_A(U,X,V,Y,U_,X_,V_,Y_,Tau,li_,lj_):
    if li_==2 and lj_ == 0:
        Tau = np.conjugate(Tau)
    return (np.einsum('ln,nm->lm',U,V_)[li_,lj_]     *Tau[1] 
            - np.einsum('nl,mn->lm',Y_,X)[li_,lj_]   *Tau[0])
def compute_B(U,X,V,Y,U_,X_,V_,Y_,Tau,li_,lj_):
    if li_==2 and lj_ == 0:
        Tau = np.conjugate(Tau)
    return (np.einsum('nl,mn->lm',X_,X)[li_,lj_]  *Tau[0] 
            + np.einsum('ln,nm->lm',V,V_)[li_,lj_]*Tau[1])

def split(array, nrows, ncols):
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def ans_3x3(P,j2,j3):
    A2 = 0;     phiA2 = 0;    phiA2p = 0;
    A3 = P[1*j3]*j3
    B1 = P[2*j3]*j3 + P[1]*(1-j3)
    B2 = P[3*j2*j3]*j2*j3+P[2*j2*(1-j3)]*(1-j3)*j2
    B3 = P[4*j3*j2]*j3*j2+P[3*j3*(1-j2)]*j3*(1-j2)
    phiA3 = P[-1]*j3
    phiA1p = np.pi
    phiB1, phiB1p, phiB2, phiB2p, phiB3 = (np.pi, np.pi, 0, 0, np.pi)
    p1 = 0
    return p1,A2,A3,B1,B2,B3,phiA1p,phiA2,phiA2p,phiA3,phiB1,phiB1p,phiB2,phiB2p,phiB3
def ans_q0(P,j2,j3):
    A3 = 0; phiA3 = 0
    A2 = P[1]*j2
    B1 = P[2*j2]*j2+P[1]*(1-j2)
    B2 = P[3*j2]*j2
    B3 = P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2)
    phiA2 = P[-1]*j2
    phiA1p, phiA2p = (0, phiA2)
    phiB1, phiB1p, phiB2, phiB2p, phiB3 = (np.pi, np.pi, np.pi, np.pi, 0)
    p1 = 0
    return p1,A2,A3,B1,B2,B3,phiA1p,phiA2,phiA2p,phiA3,phiB1,phiB1p,phiB2,phiB2p,phiB3
def ans_cb1(P,j2,j3):
    B3 = 0; phiB3 = 0
    A2 = P[1*j2]*j2
    A3 = P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2)
    B1 = P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3)
    B2 = P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3)
    phiA1p = P[-2*j2]*j2 + P[-1]*(1-j2)
    phiB2 = P[-1]*j2
    phiA2, phiA2p, phiA3 = (phiA1p/2+np.pi, phiA1p/2+np.pi, phiA1p/2)
    phiB1, phiB1p, phiB2p= (np.pi, np.pi ,-phiB2)
    p1 = 1
    return p1,A2,A3,B1,B2,B3,phiA1p,phiA2,phiA2p,phiA3,phiB1,phiB1p,phiB2,phiB2p,phiB3
def ans_cb2(P,j2,j3):
    B3 = 0; phiB3 = 0
    A2 = P[1*j2]*j2
    A3 = P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2)
    B1 = P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3)
    B2 = P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3)
    phiB1 = P[-2*j2]*j2 + P[-1]*(1-j2)
    phiA2 = P[-1]*j2
    phiA1p, phiA2p, phiA3 = (0, -phiA2, 0)
    phiB1p, phiB2, phiB2p= (-phiB1, 0 , 0)
    p1 = 1
    return p1,A2,A3,B1,B2,B3,phiA1p,phiA2,phiA2p,phiA3,phiB1,phiB1p,phiB2,phiB2p,phiB3
def ans_oct(P,j2,j3):
    A3 = 0; phiA3 = 0
    A2 = P[1]*j2
    B1 = P[2*j2]*j2+P[1]*(1-j2)
    B2 = P[3*j2]*j2
    B3 = P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2)
    phiB1 = P[-2*j2]*j2 + P[-1]*(1-j2)
    phiB2 = P[-1]*j2
    phiA1p, phiA2, phiA2p = (np.pi, 3*np.pi/2, np.pi/2)
    phiB1p, phiB2p, phiB3 = (phiB1, phiB2 , 3*np.pi/2)
    p1 = 1
    return p1,A2,A3,B1,B2,B3,phiA1p,phiA2,phiA2p,phiA3,phiB1,phiB1p,phiB2,phiB2p,phiB3

#### Ansatze encoded in the matrix
def big_Nk(P,L,args):
    m = inp.m
    J1,J2,J3,ans,KM,Tau,K_ = args
    ka1,ka1_,ka2,ka2_,ka12p,ka12p_,ka12m,ka12m_ = KM
    t1,t1_,t2,t2_,t3,t3_ = Tau
    J1 /= 2.
    J2 /= 2.
    J3 /= 2.
    j2 = np.sign(int(np.abs(J2)*1e8))   #check if it is 0 or not --> problem with VERY small J2,J3
    j3 = np.sign(int(np.abs(J3)*1e8))
    ans_func = {'3x3': ans_3x3, 'q0': ans_q0, 'cb1': ans_cb1, 'cb2': ans_cb2, 'oct': ans_oct}
    if -np.angle(t1) < np.pi/3+1e-4 and -np.angle(t1) > np.pi/3-1e-4:   #for DM = pi/3(~1.04) A1 and B1 change sign
        p104 = -1
    else:
        p104 = 1
    A1 = p104*P[0]
    p1,A2,A3,B1,B2,B3,phiA1p,phiA2,phiA2p,phiA3,phiB1,phiB1p,phiB2,phiB2p,phiB3 = ans_func[ans](P,j2,j3)
    B1 *= p104
    ################
    N = np.zeros((2*m,2*m,K_,K_), dtype=complex)
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
    N[0,1] = J1*b1p_ *ka1  *t1_              + J2*b2*t2                         #t1
    N[0,2] = J1*b1p        *t1               + J2*b2p_ *ka1*t2                  #t1_
    N[0,4] = J1*b1_  *ka2_ *t1               + J2*b2pi *ka12m*t2_               #t1
    N[0,5] = J1*b1   *ka2_ *t1_              + J2*b2i_ *ka12p_*t2_              #t1_
    N[1,2] = J1*(b1_       *t1  + b1p_*ka1_*t1_)                                #t1     t1
    N[1,3] = J1*b1         *t1_              + J2*b2p_ *ka1_*t2                 #t1_
    N[1,5] =                                   J2*(b2  *ka12p_*t2 + b2p*t2_)
    N[2,3] = J1*b1_        *t1               + J2*b2   *ka1*t2                  #t1
    N[2,4] =                                   J2*(b2p_*ka2_*t2 + b2i_*ka1*t2_)
    N[3,4] = J1*b1pi_*ka1  *t1_              + J2*b2*t2                         #t1
    N[3,5] = J1*b1p        *t1               + J2*b2pi_*ka1*t2                  #t1_
    N[4,5] = J1*(b1_       *t1  + b1pi_*ka1_*t1_)                               #t1     t1

    N[0,0] = J3*b3i_ *ka1_ *t3_
    N[3,3] = J3*b3_  *ka1_ *t3_
    N[1,4] = J3*(b3_ *ka2_ *t3_ + b3       *t3)
    N[2,5] = J3*(b3  *ka12p_  *t3  + b3i_*ka1*t3_)
    ####other half square                                                       #Same ts
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

#        |*          |*          |
#        |  *    B(k)|  *   A(k) |
#        |    *      |    *      |
#        |      *    |      *    |
#        |  c.c.  *  |  -A*(k)*  |
#        |          *|          *|
#  N  =  |-----------|-----------|  +  diag(L)
#        |*          |*          |
#        |  *   c.c. |  *  B*(-k)|
#        |    *      |    *      |
#        |      *    | c.c. *    |
#        | c.c.   *  |        *  |
#        |          *|          *|




#From the list of parameters obtained after the minimization constructs an array containing them and eventually 
#some 0 parameters which may be omitted because j2 or j3 are equal to 0.
def FormatParams(P,ans,J2,J3):
    j2 = np.sign(int(np.abs(J2)*1e8))
    j3 = np.sign(int(np.abs(J3)*1e8))
    newP = [P[0]]
    if ans == '3x3':
        newP.append(P[1]*j3)
        newP.append(P[2*j3]*j3+P[1]*(1-j3))
        newP.append(P[3*j2*j3]*j2*j3+P[2*j2*(1-j3)]*(1-j3)*j2)
        newP.append(P[4*j3*j2]*j3*j2+P[3*j3*(1-j2)]*j3*(1-j2))
        newP.append(P[-1]*j3)
    elif ans == 'q0':
        newP.append(P[1]*j2)
        newP.append(P[2*j2]*j2+P[1]*(1-j2))
        newP.append(P[3*j2]*j2)
        newP.append(P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2))
        newP.append(P[-1]*j2)
    elif ans[:3] == 'cb1':
        newP.append(P[1*j2]*j2)
        newP.append(P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2))
        newP.append(P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3))
        newP.append(P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3))
        newP.append(P[-2*j2]*j2 + P[-1]*(1-j2))
        newP.append(P[-1]*j2)
    elif ans == 'cb2':
        newP.append(P[1*j2]*j2)
        newP.append(P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2))
        newP.append(P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3))
        newP.append(P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3))
        newP.append(P[-2*j2]*j2 + P[-1]*(1-j2))
        newP.append(P[-1]*j2)
    elif ans == 'oct':
        newP.append(P[1]*j2)
        newP.append(P[2*j2]*j2+P[1]*(1-j2))
        newP.append(P[3*j2]*j2)
        newP.append(P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2))
        newP.append(P[-2*j2]*j2 + P[-1]*(1-j2))
        newP.append(P[-1]*j2)
    return tuple(newP)

## Looks if the convergence is good or if it failed
def IsConverged(P,pars,bnds,Sigma):
#    for i in range(len(P)):
#        try:
#            low = np.abs((P[i]-bnds[i][0])/P[i])
#        except:
#           low = 1
#       #
#       try:
#           high = np.abs((P[i]-bnds[i][1])/P[i])
#      except:
#          high = 1
#      #
#      if low < 1e-3 or high < 1e-3:
#          return False
    if Sigma > inp.cutoff:
        return False
    return True


def KM(kkg,a1,a2,a12p,a12m):
    ka1 = np.exp(1j*np.tensordot(a1,kkg,axes=1));   ka1_ = np.conjugate(ka1);
    ka2 = np.exp(1j*np.tensordot(a2,kkg,axes=1));   ka2_ = np.conjugate(ka2);
    ka12p = np.exp(1j*np.tensordot(a12p,kkg,axes=1));   ka12p_ = np.conjugate(ka12p);
    ka12m = np.exp(1j*np.tensordot(a12m,kkg,axes=1));   ka12m_ = np.conjugate(ka12m);
    KM = (ka1,ka1_,ka2,ka2_,ka12p,ka12p_,ka12m,ka12m_)
    return KM














