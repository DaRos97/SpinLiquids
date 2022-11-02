import numpy as np
import inputs as inp
import common_functions as cf
from time import time as t
from scipy.optimize import differential_evolution as d_e
import sys
####### inputs
#li = [0,8,72,80]
#J2, J3 = inp.J[li[int(sys.argv[1])]]
J2, J3 = inp.J[int(sys.argv[1])]
print('\n(J2,J3) = ('+'{:5.4f}'.format(J2)+',{:5.4f}'.format(J3)+')\n')
#######
csvfile = inp.DataDir+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
print("File name for saving: ",csvfile)
#ansatze = cf.CheckCsv(csvfile)
ansatze = inp.list_ans
Ti = t()
Pinitial = cf.FindInitialPoint(J2,J3,ansatze)
Bnds = cf.FindBounds(Pinitial,ansatze)
#Bnds = cf.FindBounds2(J2,J3,ansatze)
DerRange = cf.ComputeDerRanges(J2,J3,ansatze)
for ans in ansatze:
    Tti = t()
    print("Using ansatz: ",ans)
    header = inp.header[ans]
    Pi = Pinitial[ans]
    bnds = Bnds[ans]
    der_range = DerRange[ans]
    Args1 = (inp.J1,J2,J3,ans,der_range,True)
    DataDic = {}
    HessDic = {}
    print("Initial point and bounds: \n",Pi,'\n',bnds)
    result = d_e(cf.Sigma,
        args = Args1,
        x0 = Pi,
        bounds = bnds,
        popsize = 15,
        maxiter = inp.MaxIter*len(Pi),
        disp = False,
        tol = inp.cutoff,
        atol = inp.cutoff,
        updating='deferred' if inp.mp_cpu != 1 else 'immediate',
        workers = inp.mp_cpu     #parallelization
        )
    print("\nNumber of iterations: ",result.nit," / ",inp.MaxIter,'\n')
    Pf = tuple(result.x)
    Args2 = (inp.J1,J2,J3,ans,der_range,False)
    S, HessVals, E, L, gap = cf.Sigma(Pf,*Args2)
    #Add 0 values
    newP = cf.arangeP(Pf,ans,J2,J3)
    data = [ans,J2,J3,E,S,gap,L]
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(newP)):
        DataDic[header[7+ind2]] = newP[ind2]
    ########Compute Hessian values
    hessian = cf.arangeP(HessVals,ans,J2,J3)
    for i in range(len(hessian)):
        HessDic[header[7+i]] = hessian[i]
    #save values
    print(DataDic)
    print(HessDic)
    print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
    cf.SaveToCsv(DataDic,HessDic,csvfile)

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################
