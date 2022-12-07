#In this code we compute the diagram comparing different ansatze for
#TMD at J2=J3=0 varying the DM angle and the spin.
#After this maybe we can do the same plot for small values of J2, J3 
#to check the stability of the diagram under further neighbor 
#interactions.

import numpy as np
import inputs as inp
import system_functions as sf
import functions as fs
from time import time as t
import sys
import getopt

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "J:K:")
    J = 40      #inp.J point in phase diagram
    K = 13      #number ok cuts in BZ
except:
    print("Error in input parameters",argv)
    exit()
for opt, arg in opts:
    if opt in ['-J']:
        J = int(arg)
    elif opt in ['-K']:
        K = int(arg)
J1 = 1
J2 = 0
J3 = 0
#S and DM
S_max = 0.5
DM_max = 0.5
S_pts = 10
DM_pts = 15
S_list = np.linspace(0.05,S_max,S_pts,endpoint=True)
DM_list = np.logspace(-5,np.log(DM_max),DM_pts,base = np.e)
S = S_list[J%S_pts]
DM = DM_list[J//S_pts]
DM1 = DM; DM2 = 0; DM3 = 2*DM
#Filenames
DirName = '/home/users/r/rossid/SDM_Data/'
DataDir = DirName + str(K) + '/'
ReferenceDir = DirName + str(K-12) + '/'
csvfile = DataDir+'S_DM=('+'{:5.4f}'.format(S).replace('.','')+'_'+'{:5.4f}'.format(DM).replace('.','')+').csv'
#BZ points
Nx = K;     Ny = K
kxg = np.linspace(0,1,Nx)
kyg = np.linspace(0,1,Ny)
kkg = np.ndarray((2,Nx,Ny),dtype=complex)
kkgp = np.ndarray((2,Nx,Ny))
for i in range(Nx):
    for j in range(Ny):
        kkg[0,i,j] = kxg[i]*2*np.pi
        kkg[1,i,j] = (kxg[i]+kyg[j])*2*np.pi/np.sqrt(3)
        kkgp[0,i,j] = kxg[i]*2*np.pi
        kkgp[1,i,j] = (kxg[i]+kyg[j])*2*np.pi/np.sqrt(3)
#### vectors of 1nn, 2nn and 3nn
a1 = (1,0)
a2 = (-1,np.sqrt(3))
a12p = (a1[0]+a2[0],a1[1]+a2[1])
a12m = (a1[0]-a2[0],a1[1]-a2[1])
#### product of lattice vectors with K-matrix
ka1 = np.exp(1j*np.tensordot(a1,kkg,axes=1));   ka1_ = np.conjugate(ka1);
ka2 = np.exp(1j*np.tensordot(a2,kkg,axes=1));   ka2_ = np.conjugate(ka2);
ka12p = np.exp(1j*np.tensordot(a12p,kkg,axes=1));   ka12p_ = np.conjugate(ka12p);
ka12m = np.exp(1j*np.tensordot(a12m,kkg,axes=1));   ka12m_ = np.conjugate(ka12m);
KM = (ka1,ka1_,ka2,ka2_,ka12p,ka12p_,ka12m,ka12m_)
#### DM
t1 = np.exp(-1j*DM1);    t1_ = np.conjugate(t1)
t2 = np.exp(-1j*DM2);    t2_ = np.conjugate(t2)
t3 = np.exp(-1j*DM3);    t3_ = np.conjugate(t3)
Tau = (t1,t1_,t2,t2_,t3,t3_)

#Find the initial point for the minimization for each ansatz
t_0 = np.arctan(np.sqrt(2))
Pi_ = { '1a':{'A1':0.4,'B1':0.1,'phiB1':np.pi},
        '1b':{'A1':0.4,'B1':0.1,'phiB1':np.pi},
        '1c':{'A1':0.4,'B1':0.1,'phiB1':np.pi},
        '1d':{'A1':0.4,'B1':0.1,'phiB1':np.pi},
        '1e':{'A1':0.4,'B1':0.1,'phiA1':0,'phiB1':np.pi},
        '1f':{'A1':0.4,'B1':0.1,'phiA1':0,'phiB1':np.pi},
       }
ansatze = sf.CheckCsv(csvfile)
Pinitial, done  = sf.FindInitialPoint(S,DM,ansatze,ReferenceDir,Pi_)
#Find the bounds to the free parameters for each ansatz
bounds_ = {}
for ans in inp.list_ans:
    bounds_[ans] = {}
    minP = 0
    maxA = (2*S+1)/2
    maxB = S
    phase_step = 0.2
    #bounds
    for param in inp.header[ans][8:]:
        if param[0] == 'A':
            bb = (minP,maxA)
        elif param[0] == 'B':
            bb = (minP,maxB)
        elif param[:3] == 'phi':
            bb = (Pi_[ans][param]-phase_step,Pi_[ans][param]+phase_step)
        bounds_[ans][param] = bb
Bnds = bounds_
#Find the derivative range for the free parameters (different between moduli and phases) for each ansatz
DerRange = sf.ComputeDerRanges(ansatze)

print("Computing minimization for parameters: \nS=",S,"\nDM phase = ",DM,
      "\nCuts in BZ: ",K)

######################
###################### Compute the parameters by minimizing the energy for each ansatz
######################
Ti = t()    #Total initial time
for ans in ansatze:
    print("Computing ansatz ",ans)
    Tti = t()   #Initial time of the ansatz
    header = inp.header[ans]
    hess_sign = {}
    pars = Pinitial[ans].keys()
    for par in pars:
        hess_sign[par] = 1 if ('A' in [*par]) else -1
    #
    is_min = True   #needed to tell the Sigma function that we are minimizing and not just computing the final energy
    Args = (J1,J2,J3,ans,DerRange[ans],pars,hess_sign,is_min,KM,Tau,K,S)
    DataDic = {}
    #Actual minimization
    result = d_e(fs.Sigma,
        args = Args,
        x0 = Pinitial[ans],
        bounds = Bnds[ans],
        popsize = 21,                               #mbah
        maxiter = inp.MaxIter*len(Pinitial[ans]),   #max # of iterations
        #        disp = True,                       #whether to display in-progress results
        tol = inp.cutoff,
        atol = inp.cutoff,
        updating='deferred' if inp.mp_cpu != 1 else 'immediate',    #updating of the population for parallel computation
        workers = inp.mp_cpu                        #parallelization 
        )
    print("\nNumber of iterations: ",result.nit," / ",inp.MaxIter*len(Pinitial[ans]),'\n')
    #
    Pf = tuple(result.x)
    is_min = False
    Args = (J1,J2,J3,ans,DerRange[ans],pars,hess_sign,is_min,KM,Tau,K,S)
    try:
        Sigma, E, L, gap = fs.Sigma(Pf,*Args)
    except TypeError:
        print("!!!!!!!!!!!!!!!Initial point not correct!!!!!!!!!!!!!!!!")
        print("Found values: Pf=",Pf,"\nSigma = ",result.fun)
        print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
        continue
    conv = True if Sigma < inp.cutoff else False
    #Store the files in a dictionary
    data = [ans,S,DM,conv,E,Sigma,gap,L]
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(Pf)):
        DataDic[header[len(data)+ind2]] = Pf[ind2]
    #Save values to an external file
    print(DataDic)
    if not conv:
        print("!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!\\Did not converge -> Hessian sign not Correct?")
        continue
    print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
    sf.SaveToCsv(DataDic,csvfile)

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################






