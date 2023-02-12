import numpy as np
import inputs as inp
import functions_minimization as fs
import system_functions as sf
from time import time as t
import sys
import getopt
######################
###################### Set the initial parameters
######################
####### Outside inputs
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:K:")
    J = 40      #inp.J point in phase diagram
    K = 13      #number ok cuts in BZ
except:
    print("Error in input parameters",argv)
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        J = int(arg)
    elif opt in ['-K']:
        K = int(arg)
J1 = 1
S = inp.S_list[J%inp.S_pts]
DM = inp.DM_list[J//inp.S_pts]
print("Computing S=%f and DM=%f"%(S,DM))
DM1 = DM
#Filenames
DirName = '/home/users/r/rossid/0_SELF-CONSISTENCY_SDM/'
#DirName = '../../Data/self_consistency/SDM/'
#DirName = '../Data/SC_data/S'+txt_S+'/phi'+txt_DM+"/"
DataDir = DirName + str(K) + '/'
ReferenceDir = DirName + str(K-12) + '/'
csvfile = DataDir+'S_DM=('+'{:5.4f}'.format(S).replace('.','')+'_'+'{:5.4f}'.format(DM).replace('.','')+').csv'
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
pars = ['A1','A1p','phiA1p','B1','phiB1','B1p','phiB1p']
header = ['S','DM','Energy','Gap','p1','L'] + pars
#need to check existing file in order to see which pars to compute, and if it is complete

######################
###################### Compute the parameters by self concistency
######################
Ai = S
Bi = S/2
L_bounds = inp.L_bounds
Ti = t()    #Total initial time
for p1 in range(2):
    Args_O = (KM,Tau,K,pars,p1)
    Args_L = (KM,Tau,K,S,p1,L_bounds)
    solutions = sf.import_solutions(csvfile,p1)
    for ph in [2]:#,4,6]:
        for iph in range(12):
            Pinitial = [Ai,Ai,0,Bi,np.pi,Bi,np.pi]
            Pinitial[ph] = iph*np.pi*2/12
            ###################################################     Check if result was already obtained
            already_found = False
            for sol in solutions:
                #diff = 0
                #for p_ in [2,4,6]:
                diff = np.abs(Pinitial[ph]-sol[ph+1])
                if diff < inp.cutoff_solution or np.abs(diff-2*np.pi) < inp.cutoff_solution:
                    already_found = True
            if already_found:
                print("Already computed solution at p1=",p1,", par:",pars[ph],"=",Pinitial[ph])
                continue
            ####################################################
            print("Computing p1=",p1,", par:",pars[ph],"=",Pinitial[ph])
            Tti = t()   #Initial time of the ansatz
            #
            new_O = Pinitial;      old_O_1 = new_O;      old_O_2 = new_O
            new_L = (L_bounds[1]-L_bounds[0])/2 + L_bounds[0];       old_L_1 = 0;    old_L_2 = 0
            mix_list = [0, 0.1, 0.2, 0.9, 0.4, 0.5, 0.9, 0.9]#, 0.4, 0.6]
            for mix_factor in mix_list:
                print("Using mixing ",mix_factor)
                step = 0
                continue_loop = True
                exit_mixing = False
                while continue_loop:    #all pars at once
#                    print("Step ",step,": ",new_L,*new_O,end='\n')
                    conv = 1
                    old_O_2 = np.array(old_O_1)
                    old_O_1 = np.array(new_O)
                    old_L_2 = old_L_1
                    old_L_1 = new_L
                    new_L = fs.compute_L(new_O,Args_L)
                    temp_O = fs.compute_O_all(new_O,new_L,Args_O)
                    for i in range(len(old_O_1)):
                        if pars[i][0] == 'p':
                            new_O[i] = old_O_1[i]*mix_factor + temp_O[i]*(1-mix_factor)
                        else:
                            new_O[i] = old_O_1[i]*mix_factor + temp_O[i]*(1-mix_factor)
                    step += 1
                    #Check if all parameters are stable up to precision
                    if np.abs(old_L_2-new_L) > inp.cutoff_L:
                        conv *= 0
                    for i in range(len(new_O)):
                        if pars[i][0] == 'p':
                            if np.abs(old_O_1[i]-new_O[i]) > inp.cutoff_F or np.abs(old_O_2[i]-new_O[i]) > inp.cutoff_F:
                                conv *= 0
                        else:
                            if np.abs(old_O_1[i]-new_O[i]) > inp.cutoff_O or np.abs(old_O_2[i]-new_O[i]) > inp.cutoff_O:
                                conv *= 0
                    if conv:
                        continue_loop = False
                        exit_mixing = True
                    if step > inp.MaxIter:#*len(pars):
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Not converged!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        break
                if exit_mixing:
                    break
    ######################################################################################################
    ######################################################################################################
            print("\nNumber of iterations: ",step,'\n')
            conv = 'True' if conv == 1 else 'False'
            if conv == 'False':
                print("\n\nFound final parameters NOT converged: ",new_L,new_O,"\n")
                continue
            if new_L < inp.L_bounds[0] + 0.001 or new_L > inp.L_bounds[1] - 0.001:
                print("Suspicious L value: ",new_L," NOT saving")
                continue
            ###################################################     Check if result was already obtained
            already_found = False
            for sol in solutions:
                diff = 0
                diff += np.abs(new_L-sol[0])
                for p in range(7):
                    diff += np.abs(new_O[i]-sol[i+1])
                if diff < inp.cutoff_solution or np.abs(diff-2*np.pi) < inp.cutoff_solution:
                    already_found = True
            if already_found:
                print("Already found solution")
                continue
            else:
                r = [new_L] + list(new_O)
                solutions.append(r)
            ################################################### Save solution
            E,gap = fs.total_energy(new_O,new_L,Args_L)
            print('L = ',new_L)
            print('parameters: ',*new_O)
            print('energy and gap: ',E,gap)
            print("Time of solution : ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
            data = [S,DM,E,gap,p1,new_L]
            DataDic = {}
            for ind in range(len(data)):
                DataDic[header[ind]] = data[ind]
            for ind2 in range(len(new_O)):
                DataDic[header[len(data)+ind2]] = new_O[ind2]

            sf.SaveToCsv(DataDic,csvfile)

            continue
            
            for i in range(len(new_O)):
                if pars[i][0] == 'p':
                    if new_O[i] > 2*np.pi:
                        new_O[i] = new_O[i]-2*np.pi
                    if new_O[i] < -0.2:
                        new_O[i] = new_O[i]+2*np.pi

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################














































































