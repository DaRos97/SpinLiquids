import numpy as np
import inputs as inp
import common_functions as cf
import system_functions as sf
from time import time as t
from scipy.optimize import differential_evolution as d_e
import sys
######################
###################### Set the initial parameters
######################
####### Outside inputs
N = int(sys.argv[1])    #number which defines the point in the phase diagram through inp.J
J2, J3 = inp.J[N]
#File where the result is going to be saved in
csvfile = inp.DataDir+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
#Checks the file (specified by J2 and J3) and tells you which ansatze need to be computed
ansatze = sf.CheckCsv(csvfile)
#Find the initial point for the minimization for each ansatz
Pinitial, done  = sf.FindInitialPoint(J2,J3,ansatze)
#Find the bounds to the free parameters for each ansatz
Bnds = sf.FindBounds(J2,J3,ansatze,done,Pinitial)
#Find the derivative range for the free parameters (different between moduli and phases) for each ansatz
DerRange = sf.ComputeDerRanges(J2,J3,ansatze)

print('\n(J2,J3) = ('+'{:5.4f}'.format(J2)+',{:5.4f}'.format(J3)+')\n')
######################
###################### Compute the parameters by minimizing the energy for each ansatz
######################
Ti = t()    #Total initial time
for ans in ansatze:
    Tti = t()   #Initial time of the ansatz
    header = inp.header[ans]
    #Find the parameters that we actually need to use and their labels (some parameters are zero if J2 or J3 are zero
    j2 = int(np.sign(J2)*np.sign(int(np.abs(J2)*1e8)) + 1)   #j < 0 --> 0, j == 0 --> 1, j > 0 --> 2
    j3 = int(np.sign(J3)*np.sign(int(np.abs(J3)*1e8)) + 1)
    pars2 = inp.Pi[ans].keys()
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
    Args = (inp.J1,J2,J3,ans,DerRange[ans],pars,hess_sign,is_min)
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
    Args = (inp.J1,J2,J3,ans,DerRange[ans],pars,hess_sign,is_min)
    try:
        S, E, L, gap = cf.Sigma(Pf,*Args)
    except TypeError:
        print("Initial point not correct")
        print("Found values: Pf=",Pf,"\nSigma = ",result.fun)
        print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
        continue
    if S >= 10:
        print("Hessian sign not Correct")
    conv = cf.IsConverged(Pf,pars,Bnds[ans],S)      #check whether the convergence worked and it is not too close to the boudary of the bounds
    #Format the parameters in order to have 0 values in the non-considered ones
    newP = cf.FormatParams(Pf,ans,J2,J3)
    #Store the files in a dictionary
    data = [ans,J2,J3,conv,E,S,gap,L]
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(newP)):
        DataDic[header[len(data)+ind2]] = newP[ind2]
    #Save values to an external file
    print(DataDic)
    print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
    sf.SaveToCsv(DataDic,csvfile)

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################
