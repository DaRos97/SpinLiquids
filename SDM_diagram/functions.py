import numpy as np
import inputs as inp
from scipy import linalg as LA
from scipy.optimize import minimize_scalar
from scipy.interpolate import RectBivariateSpline as RBS
from pathlib import Path
import csv
import os

#### Matrix diagonal of -1 and 1
J_ = np.zeros((2*inp.m,2*inp.m))
for i in range(inp.m):
    J_[i,i] = -1
    J_[i+inp.m,i+inp.m] = 1

# Check also Hessians on the way --> more time (1 general + 2 energy evaluations for each P).
# Calls only the totE func
def Sigma(P,*Args):
    ans,der_range,pars,hess_sign,is_min,KM,Tau,K_,S = Args         #extract arguments
    L_bounds = inp.L_bounds                                     #bounds on Lagrange multiplier set by default
    args = (ans,L_bounds,KM,Tau,K_,S)                              #arguments to pass to totE
    init = totE(P,args)                                         #compute the initial point        #1
    if init[2] > 9 or np.abs(init[1]-L_bounds[0]) < 1e-3:       #check whether is good (has to be)
        return inp.shame2
    temp = []
    L_bounds = (init[1]-inp.L_b_2, init[1]+inp.L_b_2)           #restrict the bound on the Lagrange multiplier since we are staying close to its value of the #1 evaluation
    args = (ans,L_bounds,KM,Tau,K_,S)                    #new arguments to pass to totE in computing derivatives and Hessian
    for i in range(len(P)):                 #for each parameter
        pp = np.array(P)                    #copy the list of parameters
        dP = der_range[i]                   
        pp[i] = P[i] + dP                   #change only one by dP
        init_plus = totE(pp,args)           #compute first derivative               #2
        der1 = (init_plus[0]-init[0])/dP
        if is_min:
            temp.append(der1**2)
            continue
        pp[i] = P[i] + 2*dP                 #change again the same parameter by dP
        init_2plus = totE(pp,args)          #compute the second derivative          #3
        der2 = (init_2plus[0]-init_plus[0])/dP
        Hess = (der2-der1)/dP               #evaluate the Hessian
        hess = int(np.sign(Hess))
        if hess == hess_sign[pars[i]]:      #check if the sign of the Hessian is correct
            temp.append(der1**2)        #add it to the sum
        else:
            print("Hessian of ",pars[i]," not correct")
            if pars[i][0] == 'p':
                temp.append(der1**2)        #add it to the sum anyway if it is a phase
            else:
                temp.append(1e5)
    res = np.array(temp).sum()          #sum all the contributioms
    if is_min:
        return res
    else:                               #last computation -> Sigma, Energy, L, gap
        return res, init[0], init[1], init[2]

#### Computes Energy from Parameters P, by maximizing it wrt the Lagrange multiplier L. Calls only totEl function
def totE(P,args):
    res = minimize_scalar(lambda l: -totEl(P,l,args)[0],  #maximize energy wrt L with fixed P
            method = inp.L_method,          #can be 'bounded' or 'Brent'
            bracket = args[-5],             #bounds = inp.L_bounds,
            options={'xtol':inp.prec_L}
            )
    L = res.x                       #optimized L
    minE = -res.fun                 #optimized energy(total)
    gap = totEl(P,L,args)[1]        #result of sumEigs -> sum of ws and gap
    return minE, L, gap

#### Computes the Energy given the paramters P and the Lagrange multiplier L. 
#### This is the function that does the actual work.
def totEl(P,L,args):
    ans,L_bounds,KM,Tau,K_,S = args
    #The minimization function sometimes goes out of the given bounds so let it go back inside
    if L < L_bounds[0] :
        Res = -5-(L_bounds[0]-L)
        return Res, 10
    elif L > L_bounds[1]:
        Res = -5-(L-L_bounds[1])
        return Res, 10
    Res = inp.z[0]*(P[0]**2-P[1]**2)/2
    Res -= L*(2*S+1)            #part of the energy coming from the Lagrange multiplier
    #Compute now the (painful) part of the energy coming from the Hamiltonian matrix by the use of a Bogoliubov transformation
    args2 = (ans,KM,Tau,K_)
    N = big_Nk(P,L,args2)                #compute Hermitian matrix from the ansatze coded in the ansatze.py script
    res = np.zeros((inp.m,K_,K_))
    for i in range(K_):                 #cicle over all the points in the Brilluin Zone grid
        for j in range(K_):
            Nk = N[:,:,i,j]                 #extract the corresponding matrix
            try:
                Ch = LA.cholesky(Nk)        #not always the case since for some parameters of Lambda the eigenmodes are negative
            except LA.LinAlgError:          #matrix not pos def for that specific kx,ky
                r4 = -3+(L-L_bounds[0])
                return Res+r4, 10           #if that's the case even for a single k in the grid, return a defined value
            temp = np.dot(np.dot(Ch,J_),np.conjugate(Ch.T))    #we need the eigenvalues of M=KJK^+ (also Hermitian)
            res[:,i,j] = LA.eigvalsh(temp)[inp.m:]      #BOTTLE NECK -> compute the eigevalues
    #Now fit the energy values found with a spline curve in order to have a better solution
    r2 = 0
    for i in range(inp.m):
        func = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),res[i])
        r2 += func.integral(0,1,0,1)        #integrate the fitting curves to get the energy of each band
    r2 /= inp.m                             #normalize
    gap = np.amin(res[0].ravel())           #the gap is the lowest value of the lowest gap (not in the fitting if not could be negative in principle)
    #
    Res += r2                               #sum to the other part of the energy
    return Res, gap

#### Ansatze encoded in the matrix
def big_Nk(P,L,args):
    m = inp.m
    ans,KM,Tau,K_ = args
    ka1 = KM[0]; ka1_=KM[1]; ka2=KM[2]; ka2_=KM[3]; ka12p=KM[4]; ka12p_=KM[5]; ka12m=KM[6]; ka12m_=KM[7]
    t1 = Tau[0]; t1_ = Tau[1]; t2 = Tau[2]; t2_ = Tau[3]; t3 = Tau[4]; t3_ = Tau[5];
    J1 = 1/2
    A1 = P[0]
    B1 = P[1]
    phiB1 = P[-1]
    if ans == '1a':      #3
        phiA1p = 0
        phiB1p = -phiB1
        p1 = 0
    if ans == '1b':      #3
        phiA1p = np.pi
        phiB1p = -phiB1
        p1 = 0
    if ans == '1c':      #3
        phiA1p = 0
        phiB1p = -phiB1
        p1 = 1
    if ans == '1d':      #3
        phiA1p = np.pi
        phiB1p = -phiB1
        p1 = 1
    if ans == '1e':      #3
        phiA1p = P[-2]
        phiB1p = phiB1
        p1 = 0
    if ans[:2] == '1f':      #3
        phiA1p = P[-2]
        phiB1p = phiB1
        p1 = 1
    ################
    N = np.zeros((2*m,2*m,K_,K_), dtype=complex)
    ##################################### B
    b1 = B1*np.exp(1j*phiB1);               b1_ = np.conjugate(b1)
    b1p = B1*np.exp(1j*phiB1p);             b1p_ = np.conjugate(b1p)
    b1pi = B1*np.exp(1j*(phiB1p+p1*np.pi)); b1pi_ = np.conjugate(b1pi)
    #
    N[0,1] = J1*b1p_ *ka1  *t1_              #t1
    N[0,2] = J1*b1p        *t1               #t1_
    N[0,4] = J1*b1_  *ka2_ *t1               #t1
    N[0,5] = J1*b1   *ka2_ *t1_              #t1_
    N[1,2] = J1*(b1_       *t1  + b1p_*ka1_*t1_)                                #t1     t1
    N[1,3] = J1*b1         *t1_ 
    N[2,3] = J1*b1_        *t1   
    N[3,4] = J1*b1pi_*ka1  *t1_  
    N[3,5] = J1*b1p        *t1   
    N[4,5] = J1*(b1_       *t1  + b1pi_*ka1_*t1_)                               #t1     t1
    ####other half square                                                       #Same ts
    N[m+0,m+1] = J1*b1p  *ka1  *t1_ 
    N[m+0,m+2] = J1*b1p_       *t1  
    N[m+0,m+4] = J1*b1   *ka2_ *t1  
    N[m+0,m+5] = J1*b1_  *ka2_ *t1_ 
    N[m+1,m+2] = J1*(b1        *t1  + b1p*ka1_*t1_)
    N[m+1,m+3] = J1*b1_        *t1_ 
    N[m+2,m+3] = J1*b1         *t1  
    N[m+3,m+4] = J1*b1pi *ka1  *t1_ 
    N[m+3,m+5] = J1*b1p_       *t1  
    N[m+4,m+5] = J1*(b1        *t1  + b1pi*ka1_*t1_)
    ######################################## A
    a1 =    A1
    a1p =   A1*np.exp(1j*phiA1p)
    a1pi =  A1*np.exp(1j*(phiA1p+p1*np.pi))
    N[0,m+1] = - J1*a1p *ka1 *t1_  
    N[0,m+2] =   J1*a1p      *t1   
    N[0,m+4] = - J1*a1  *ka2_*t1   
    N[0,m+5] =   J1*a1  *ka2_*t1_  
    N[1,m+2] = - J1*(a1      *t1   +a1p*ka1_*t1_)
    N[1,m+3] =   J1*a1       *t1_ 
    N[2,m+3] = - J1*a1       *t1  
    N[3,m+4] = - J1*a1pi*ka1 *t1_ 
    N[3,m+5] =   J1*a1p      *t1  
    N[4,m+5] = - J1*(a1      *t1   +a1pi*ka1_*t1_)
    #not the diagonal
    N[1,m]   =   J1*a1p *ka1_*t1  
    N[2,m]   = - J1*a1p      *t1_ 
    N[4,m]   =   J1*a1  *ka2 *t1_ 
    N[5,m]   = - J1*a1  *ka2 *t1  
    N[2,m+1] =   J1*(a1      *t1_  +a1p*ka1 *t1)
    N[3,m+1] = - J1*a1       *t1  
    N[3,m+2] =   J1*a1       *t1_ 
    N[4,m+3] =   J1*a1pi*ka1_*t1  
    N[5,m+3] = - J1*a1p      *t1_ 
    N[5,m+4] =   J1*(a1      *t1_  +a1pi*ka1 *t1)
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

















