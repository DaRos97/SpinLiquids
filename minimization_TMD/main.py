import numpy as np
import inputs as inp
import common_functions as cf
import system_functions as sf
from time import time as t
from scipy.optimize import differential_evolution as d_e
import sys
import getopt
######################
###################### Set the initial parameters
######################
####### Outside inputs
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:S:K:",["DM="])
    N = 40      #inp.J point in phase diagram
    txt_S = '50'
    K = 13      #number ok cuts in BZ
    txt_DM = '000'  #DM angle from list
except:
    print("Error in input parameters",argv)
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)
    elif opt in ['-S']:
        txt_S = arg
    elif opt in ['-K']:
        K = int(arg)
    if opt == '--DM':
        txt_DM = arg
J1 = 1
J2, J3 = inp.J[N]
S_label = {'50':0.5,'36':(np.sqrt(3)-1)/2,'34':0.34,'30':0.3,'20':0.2}
S = S_label[txt_S]
DM_list = {'005':0.05}
phi = DM_list[txt_DM]
DM1 = phi;      DM2 = 0;    DM3 = 2*phi
#BZ points
Nx = K;     Ny = K
#Filenames
DirName = '/home/users/r/rossid/Data/S'+txt_S+'/phi'+txt_DM+"/"
DataDir = DirName + str(Nx) + '/'
ReferenceDir = DirName + str(Nx-12) + '/'
csvfile = DataDir+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
#BZ points
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
########################
########################    Initiate routine
########################

#Find the initial point for the minimization for each ansatz
t_0 = np.arctan(np.sqrt(2))
Pi_ = { '3x3':{'A1':0.4, 'A3':0.5, 'B1':0.1, 'B2': 0.1, 'B3': 0.1, 'phiB1':np.pi, 'phiA3': 0, 'phiB3': np.pi},
        'q0':{'A1':0.4, 'A2':0.3, 'B1':0.1, 'B2': 0.1, 'B3': 0.1, 'phiB1':0, 'phiA2': np.pi, 'phiB3': 0},
        'cb1':{'A1':0.4, 'A2':0.1, 'A3':0.43, 'B1':0.1, 'B2': 0.1, 'phiA1': 2*t_0, 'phiB1':np.pi, 'phiB2': 2*np.pi-t_0},
        'cb1_nc':{'A1':0.4, 'A2':0.1, 'A3':0.43, 'B1':0.1, 'B2': 0.1, 'phiA1': 0, 'phiB1':np.pi, 'phiB2': np.pi}
        }
#Checks the file (specified by J2 and J3) and tells you which ansatze need to be computed
ansatze = sf.CheckCsv(csvfile)
Pinitial, done  = sf.FindInitialPoint(J2,J3,ansatze,ReferenceDir,Pi_)
#Find the bounds to the free parameters for each ansatz
bounds_ = {}
for ans in inp.list_ans:
    bounds_[ans] = {}
    min_dic = {'50':0.01, '36':0.005,'34':0.002, '30':0.002, '20':0.001}
    mM_A1 = {'50':(0.3,0.6), '36':(0.2,0.5),'34':(0.2,0.45), '30':(0.2,0.45), '20':(0.05,0.41)}
    minP = min_dic[txt_S]
    maxA = (2*S+1)/2
    maxB = S
    bounds_[ans]['A1'] = mM_A1[txt_S]
    phase_step = 0.2
    #bounds
    for param in inp.header[ans][9:]:
        if param[0] == 'A':
            bb = (minP,maxA)
        elif param[0] == 'B':
            bb = (minP,maxB)
        elif param[:3] == 'phi':
            bb = (Pi_[ans][param]-phase_step,Pi_[ans][param]+phase_step)
        bounds_[ans][param] = bb
        if ans == 'cb1_nc' and param == 'A2':
            bounds_[ans][param] = (-0.1,maxA)
Bnds = sf.FindBounds(J2,J3,ansatze,done,Pinitial,Pi_,bounds_)
#Find the derivative range for the free parameters (different between moduli and phases) for each ansatz
DerRange = sf.ComputeDerRanges(J2,J3,ansatze)

print("Computing minimization for parameters: \nS=",S,"\nDM phase = ",phi,'\nPoint in phase diagram(J2,J3) = ('+'{:5.4f}'.format(J2)+',{:5.4f}'.format(J3)+')',
      "\nCuts in BZ: ",K)
######################
###################### Compute the parameters by minimizing the energy for each ansatz
######################
Ti = t()    #Total initial time
for ans in ansatze:
    print("Computing ansatz ",ans)
    Tti = t()   #Initial time of the ansatz
    header = inp.header[ans]
    #Find the parameters that we actually need to use and their labels (some parameters are zero if J2 or J3 are zero
    j2 = int(np.sign(J2)*np.sign(int(np.abs(J2)*1e8)) + 1)   #j < 0 --> 0, j == 0 --> 1, j > 0 --> 2
    j3 = int(np.sign(J3)*np.sign(int(np.abs(J3)*1e8)) + 1)
    pars2 = Pi_[ans].keys()
    pars = []
    for pPp in pars2:
        if (pPp[-1] == '1') or (pPp[-1] == '2' and j2-1) or (pPp[-1] == '3' and j3-1):
            pars.append(pPp)
    #Compute the signs of the Hessians for each parameter
    hess_sign = {}
    for par in pars:
        if par[-2] == 'A':
            if par[-1] == '1' or (par[-1] == '2' and J2 > 0) or (par[-1] == '3' and J3 > 0):
                hess_sign[par] = 1
            else:
                hess_sign[par] = -1
        else:
            if par[-1] == '1' or (par[-1] == '2' and J2 > 0) or (par[-1] == '3' and J3 > 0):
                hess_sign[par] = -1
            else:
                hess_sign[par] = 1
    is_min = True   #needed to tell the Sigma function that we are minimizing and not just computing the final energy
    Args = (J1,J2,J3,ans,DerRange[ans],pars,hess_sign,is_min,KM,Tau,K,S)
    DataDic = {}
    #Actual minimization
    result = d_e(cf.Sigma,
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

    #Compute the final values using the result of the minimization
    Pf = tuple(result.x)
    is_min = False
    Args = (J1,J2,J3,ans,DerRange[ans],pars,hess_sign,is_min,KM,Tau,K,S)
    try:
        Sigma, E, L, gap = cf.Sigma(Pf,*Args)
    except TypeError:
        print("!!!!!!!!!!!!!!!Initial point not correct!!!!!!!!!!!!!!!!")
        print("Found values: Pf=",Pf,"\nSigma = ",result.fun)
        print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
        continue
    conv = cf.IsConverged(Pf,pars,Bnds[ans],Sigma)      #check whether the convergence worked and it is not too close to the boudary of the bounds
    #Format the parameters in order to have 0 values in the non-considered ones
    newP = cf.FormatParams(Pf,ans,J2,J3)
    #Store the files in a dictionary
    data = [ans,J2,J3,conv,E,Sigma,gap,L]
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(newP)):
        DataDic[header[len(data)+ind2]] = newP[ind2]
    #Save values to an external file
    print(DataDic)
    if not conv:
        print("!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!\\Din not converge -> Hessian sign not Correct?")
        continue
    print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
    sf.SaveToCsv(DataDic,csvfile)

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################
