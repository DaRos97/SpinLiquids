import inputs as inp
import ansatze as an
import numpy as np
from scipy import linalg as LA
from scipy.optimize import minimize_scalar
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

# Check also Hessians on the way --> more time (1 general + 2 energy evaluations for each P).
# Calls only the totE func
def Sigma(P,*Args):
    J1,J2,J3,ans,der_range,pars,hess_sign,is_min = Args         #extract arguments
    j2 = int(np.sign(J2)*np.sign(int(np.abs(J2)*1e8)) + 1)      #j < 0 --> 0, j == 0 --> 1, j > 0 --> 2
    j3 = int(np.sign(J3)*np.sign(int(np.abs(J3)*1e8)) + 1)
    L_bounds = inp.L_bounds                                     #bounds on Lagrange multiplier set by default
    args = (J1,J2,J3,ans,L_bounds)                              #arguments to pass to totE
    init = totE(P,args)                                         #compute the initial point        #1
    if init[2] > 9 or np.abs(init[1]-L_bounds[0]) < 1e-3:       #check whether is good (has to be)
        return inp.shame2
    temp = []
    L_bounds = (init[1]-inp.L_b_2, init[1]+inp.L_b_2)           #restrict the bound on the Lagrange multiplier since we are staying close to its value of the #1 evaluation
    args = (J1,J2,J3,ans,L_bounds)                              #new arguments to pass to totE in computing derivatives and Hessian
    for i in range(len(P)):                 #for each parameter
        pp = np.array(P)                    #copy the list of parameters
        dP = der_range[i]                   
        pp[i] = P[i] + dP                   #change only one by dP
        init_plus = totE(pp,args)           #compute first derivative               #2
        der1 = (init_plus[0]-init[0])/dP
        pp[i] = P[i] + 2*dP                 #change again the same parameter by dP
        init_2plus = totE(pp,args)          #compute the second derivative          #3
        der2 = (init_2plus[0]-init_plus[0])/dP
        Hess = (der2-der1)/dP               #evaluate the Hessian
        hess = int(np.sign(Hess))
        if hess == hess_sign[pars[i]]:      #check if the sign of the Hessian is correct
            temp.append(der1**2)        #add it to the sum
        else:
            if der1:                    #add to the sum a value which will decrease going in the correct direction
                r2 = np.abs(der1)**2 + np.sqrt(np.abs(1/der1)) + np.abs(1/der1) + 10
            else:
                r2 = 1e5
            temp.append(r2)
    res = np.array(temp).sum()          #sum all the contributioms
    if is_min:
        return res
    else:                               #last computation -> Sigma, Energy, L, gap
        return res, init[0], init[1], init[2]

#### Computes Energy from Parameters P, by maximizing it wrt the Lagrange multiplier L. Calls only totEl function
def totE(P,args):
    res = minimize_scalar(lambda l: -totEl(P,l,args)[0],  #maximize energy wrt L with fixed P
            method = inp.L_method,          #can be 'bounded' or 'Brent'
            bracket = args[-1],             #bounds = inp.L_bounds,
            options={'xtol':inp.prec_L}
            )
    L = res.x                       #optimized L
    minE = -res.fun                 #optimized energy(total)
    gap = totEl(P,L,args)[1]        #result of sumEigs -> sum of ws and gap
    return minE, L, gap

#### Computes the Energy given the paramters P and the Lagrange multiplier L. 
#### This is the function that does the actual work.
def totEl(P,L,args):
    J1,J2,J3,ans,L_bounds = args
    #The minimization function sometimes goes out of the given bounds so let it go back inside
    if L < L_bounds[0] :
        Res = -5-(L_bounds[0]-L)
        return Res, 10
    elif L > L_bounds[1]:
        Res = -5-(L-L_bounds[1])
        return Res, 10
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
    Res -= L*(2*inp.S+1)            #part of the energy coming from the Lagrange multiplier
    #Compute now the (painful) part of the energy coming from the Hamiltonian matrix by the use of a Bogoliubov transformation
    N = an.Nk(P,L,args[:-1])                #compute Hermitian matrix from the ansatze coded in the ansatze.py script
    res = np.zeros((inp.m,inp.Nx,inp.Ny))
    for i in range(inp.Nx):                 #cicle over all the points in the Brilluin Zone grid
        for j in range(inp.Ny):
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
        func = RBS(inp.kxg,inp.kyg,res[i])
        r2 += func.integral(0,1,0,1)        #integrate the fitting curves to get the energy of each band
    r2 /= inp.m                             #normalize
    gap = np.amin(res[0].ravel())           #the gap is the lowest value of the lowest gap (not in the fitting if not could be negative in principle)
    #
    Res += r2                               #sum to the other part of the energy
    return Res, gap

#From the list of parameters obtained after the minimization constructs an array containing them and eventually 
#some 0 parameters which may be omitted because j2 or j3 are equal to 0.
def FormatParams(P,ans,J2,J3):
    j2 = np.sign(int(np.abs(J2)*1e8))
    j3 = np.sign(int(np.abs(J3)*1e8))
    newP = [P[0]]
    if ans == '3x3_1':
        newP.append(P[1]*j3)
        newP.append(P[2*j3]*j3+P[1]*(1-j3))
        newP.append(P[3*j2*j3]*j2*j3+P[2*j2*(1-j3)]*(1-j3)*j2)
        newP.append(P[4*j3*j2]*j3*j2+P[3*j3*(1-j2)]*j3*(1-j2))
        newP.append(P[-3*j3]*j3 + P[-1]*(1-j3))
        newP.append(P[-2]*j3)
        newP.append(P[-1]*j3)
    elif ans == 'q0_1':
        newP.append(P[1]*j2)
        newP.append(P[2*j2]*j2+P[1]*(1-j2))
        newP.append(P[3*j2]*j2)
        newP.append(P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2))
        newP.append(P[-4*j2*j3]*j2*j3 + P[-3*(1-j3)*j2]*(1-j3)*j2 + P[-2*j3*(1-j2)]*j3*(1-j2) + P[-1]*(1-j2)*(1-j3))
        newP.append(P[-3*j2*j3]*j2*j3 + P[-2*j2*(1-j3)]*j2*(1-j3))
        newP.append(P[-2*j2*j3]*j2*j3 + P[-1*j2*(1-j3)]*j2*(1-j3))
        newP.append(P[-1]*j3)
    elif ans == 'cb1':
        newP.append(P[1*j2]*j2)
        newP.append(P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2))
        newP.append(P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3))
        newP.append(P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3))
        newP.append(P[-4*j2]*j2 + P[-2]*(1-j2))
        newP.append(P[-3*j2]*j2 + P[-1]*(1-j2))
        newP.append(P[-2]*j2)
        newP.append(P[-1]*j2)
    elif ans == 'cb2':
        newP.append(P[1*j2]*j2)
        newP.append(P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2))
        newP.append(P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3))
        newP.append(P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3))
        newP.append(P[-4*j2*j3]*j2*j3 + P[-3*(1-j3)*j2]*(1-j3)*j2 + P[-2*j3*(1-j2)]*j3*(1-j2) + P[-1]*(1-j2)*(1-j3))
        newP.append(P[-3*j2*j3]*j2*j3 + P[-2*j2*(1-j3)]*j2*(1-j3))
        newP.append(P[-2*j2*j3]*j2*j3 + P[-1*j2*(1-j3)]*j2*(1-j3))
        newP.append(P[-1]*j3)
    elif ans == 'oct':
        newP.append(P[1*j2]*j2)
        newP.append(P[2*j2]*j2+P[1]*(1-j2))
        newP.append(P[3*j2]*j2)
        newP.append(P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2))
        newP.append(P[-4*j2]*j2 + P[-2]*(1-j2))
        newP.append(P[-3*j2]*j2 + P[-1]*(1-j2))
        newP.append(P[-2]*j2)
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
 
 
 ########
 ########        Additional lines of code
########

#### Computes the part of the energy given by the Bogoliubov eigen-modes
def sumEigs(P,L,args):
    J1,J2,J3,ans = args
    Args = (J1,J2,J3,ans)
    N = an.Nk(P,L,Args) #compute Hermitian matrix
    res = np.zeros((inp.m,inp.Nx,inp.Ny))
    for i in range(inp.Nx):
        for j in range(inp.Ny):
            Nk = N[:,:,i,j]
            try:
                K = LA.cholesky(Nk)     #not always the case since for some parameters of Lambda the eigenmodes are negative
            except LA.LinAlgError:      #matrix not pos def for that specific kx,ky
                res = -5-(inp.L_bounds[0]-L)
                return res, 10      #if that's the case even for a single k in the grid, return a defined value
            temp = np.dot(np.dot(K,J),np.conjugate(K.T))    #we need the eigenvalues of M=KJK^+ (also Hermitian)
            res[:,i,j] = LA.eigvalsh(temp)[inp.m:]
    r2 = 0
    for i in range(inp.m):
        func = RBS(inp.kxg,inp.kyg,res[i])
        r2 += func.integral(0,1,0,1)
    r2 /= inp.m
    gap = np.amin(res[0].ravel())
    return r2, gap
    if 0:
        #plot
        print("P: ",P,"\nL:",L,"\ngap:",gap)
        R = np.zeros((3,inp.Nx,inp.Ny))
        for i in range(inp.Nx):
            for j in range(inp.Ny):
                R[0,i,j] = np.real(inp.kkg[0,i,j])
                R[1,i,j] = np.real(inp.kkg[1,i,j])
                R[2,i,j] = res[0,i,j]
        func = RBS(inp.kxg,inp.kyg,res[0])
        X,Y = np.meshgrid(inp.kxg,inp.kyg)
        Z = func(inp.kxg,inp.kyg)
        #fig,(ax1,ax2) = plt.subplots(1,2)#,projection='3d')
        fig = plt.figure(figsize=(10,5))
        plt.axis('off')
        plt.title(str(inp.Nx)+' * '+str(inp.Ny))
        ax1 = fig.add_subplot(131, projection='3d')
        #ax1 = fig.gca(projection='3d')
        ax1.plot_trisurf(R[0].ravel(),R[1].ravel(),R[2].ravel())
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_surface(inp.kkgp[0],inp.kkgp[1],res[0],cmap=cm.coolwarm)
        ax3 = fig.add_subplot(133, projection='3d')     #works only for square grid
        ax3.plot_surface(X,Y,Z,cmap=cm.coolwarm)
        plt.show()
    return r2, gap
