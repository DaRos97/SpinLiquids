import numpy as np
import inputs as inp
import functions as fs
import system_functions as sf
from time import time as t
import sys
import getopt
import random
######################
###################### Set the initial parameters
######################
####### Outside inputs
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:K:",["uniform","staggered"])
    J = 40      #inp.J point in phase diagram
    K = 13      #number ok cuts in BZ
    numb_it = 3
    DM_type = 'uniform'
    disp = False
except:
    print("Error in input parameters",argv)
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        J = int(arg)
    elif opt in ['-K']:
        K = int(arg)
    if opt == '--numb_it':
        numb_it = int(arg)
    if opt == '--staggered':
        DM_type = 'staggered'
    if opt == '--uniform':
        DM_type = 'uniform'
J1 = 1
S = inp.S_list[J%inp.S_pts]
DM = inp.DM_list[J//inp.S_pts]
print("Computing S=%f and DM=%f"%(S,DM))
DM1 = DM
#Filenames
DirName = '/home/users/r/rossid/0_SELF-CONSISTENCY_SDM/Data/'+DM_type+'/'
#DirName = '../../Data/self_consistency/SDM/test/'
#DirName = '../../Data/self_consistency/SDM/'+DM_type+'/'
DataDir = DirName + str(K) + '/'
ReferenceDir = DirName + str(13) + '/'
csvname = 'S_DM=('+'{:5.4f}'.format(S).replace('.','')+'_'+'{:5.4f}'.format(DM).replace('.','')+').csv'
csvfile = DataDir + csvname
csvref = ReferenceDir + csvname
#BZ points
Nx = Ny = K
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
#### product of lattice vectors with K-matrix
KM = fs.KM(kkg,inp.a1,inp.a2,inp.a12p,inp.a12m)
#### DM
t1 = np.exp(-1j*DM1);    t1_ = np.conjugate(t1)
Tau = (t1,t1_)
########################
########################    Initiate routine
########################
#need to check existing file in order to see which pars to compute, and if it is complete

######################
###################### Compute the parameters by self concistency
######################
L_bounds = inp.L_bounds
Ti = t()    #Total initial time
list_ansatze, list_phases = sf.find_lists(csvref,K,numb_it)
for ans in list_ansatze:
    if ans in inp.ansatze_1:
        pars = ['A1','B1','phiB1']
        index_ch_phase = 2
        list_amp = [0,1]
        list_phase = [2]
    else:
        pars = ['A1','phiA1p','B1','phiB1']
        index_ch_phase = 1
        list_amp = [0,2]
        list_phase = [1,3]
    header = ['ans','S','DM','Energy','Gap','L'] + pars
    Args_O = (KM,Tau,K,S,pars,ans,DM_type)
    Args_L = (KM,Tau,K,S,ans,DM_type,L_bounds)
    solutions = sf.import_solutions(csvfile,ans)
    for new_phase in range(list_phases[ans]):
        Pinitial = sf.find_Pinitial(S,ans,csvref,K,new_phase,index_ch_phase,numb_it)
        completed = sf.check_solutions(solutions,index_ch_phase,Pinitial[index_ch_phase])
        if completed:
#            print("Already found solution for ans ",ans," and phase ",new_phase)
            continue
        ####################################################
        print("Computing ans ",ans,", par:",pars[index_ch_phase],"=",Pinitial[index_ch_phase])
        Tti = t()   #Initial time of the ansatz
        #
        new_O = Pinitial;      old_O_1 = new_O;      old_O_2 = new_O
        new_L = (L_bounds[1]-L_bounds[0])/2 + L_bounds[0];       old_L_1 = 0;    old_L_2 = 0
        step = 0
        continue_loop = True
        while continue_loop:    #all pars at once
            if disp:
                print("Step ",step,": ",new_L,*new_O,end='\n')
            conv = 1
            old_O_2 = np.array(old_O_1)
            old_O_1 = np.array(new_O)
            old_L_2 = old_L_1
            old_L_1 = new_L
            new_L = fs.compute_L(new_O,Args_L)
            temp_O = fs.compute_O_all(new_O,new_L,Args_O)
            #
            mix_factor = random.uniform(0,1) if K == 13 else 0
            #
            for i in range(len(old_O_1)):
                if pars[i][0] == 'p' and np.abs(temp_O[i]-old_O_1[i]) > np.pi:
                    temp_O[i] -= 2*np.pi
                new_O[i] = old_O_1[i]*mix_factor + temp_O[i]*(1-mix_factor)
                if pars[i][0] == 'p' and new_O[i] < 0:
                    new_O[i] += 2*np.pi
            step += 1
            #Check if all parameters are stable up to precision
            if np.abs(old_L_2-new_L)/S > inp.cutoff_L:
                conv *= 0
            for i in range(len(new_O)):
                if pars[i][0] == 'p':
                    if np.abs(old_O_1[i]-new_O[i]) > inp.cutoff_F or np.abs(old_O_2[i]-new_O[i]) > inp.cutoff_F:
                        conv *= 0
                else:
                    if np.abs(old_O_1[i]-new_O[i])/S > inp.cutoff_O or np.abs(old_O_2[i]-new_O[i])/S > inp.cutoff_O:
                        conv *= 0
            if conv:
                continue_loop = False
            if step > inp.MaxIter:#*len(pars):
                break
######################################################################################################
######################################################################################################
#        print("\nNumber of iterations: ",step,'\n')
        conv = 'True' if conv == 1 else 'False'
        if conv == 'False':
            print("\n\nFound final parameters NOT converged: ",new_L,new_O,"\n")
            continue
        if new_L < inp.L_bounds[0] + 0.001 or new_L > inp.L_bounds[1] - 0.001:
            print("Suspicious L value: ",new_L," NOT saving")
            continue
        ###################################################     Check if result was already obtained
        amp_found = ph_found = False
        for sol in solutions:
            diff = 0
            diff += np.abs(new_L-sol[0])/S
            amp_found = False
            for p in list_amp:     #amplitudes
                diff += np.abs(new_O[p]-sol[p+1])/S
            if diff < inp.cutoff_solution:
                amp_found = True
            ph_found = True
            for p in list_phase:
                diff = np.abs(new_O[p]-sol[p+1])
                if not (diff < inp.cutoff_solution or np.abs(diff-2*np.pi) < inp.cutoff_solution):
                    ph_found = False
            if amp_found and ph_found:
                print("Already found solution")
                break
        if amp_found and ph_found:
            continue
        r = [new_L] + list(new_O)
        solutions.append(r)
        ################################################### Save solution
        E,gap = fs.total_energy(new_O,new_L,Args_L)
        print("Ansatz: ",ans)
        print('L = ',new_L)
        print('parameters: ',*new_O)
        print('energy and gap: ',E,gap)
        print("Time of solution : ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
        data = [ans,S,DM,E,gap,new_L]
        DataDic = {}
        for ind in range(len(data)):
            DataDic[header[ind]] = data[ind]
        for ind2 in range(len(new_O)):
            DataDic[header[len(data)+ind2]] = new_O[ind2]
        sf.SaveToCsv(DataDic,csvfile)


print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################














































































