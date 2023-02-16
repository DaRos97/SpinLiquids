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
    opts, args = getopt.getopt(argv, "N:S:K:",["DM="])
    N = 40      #inp.J point in phase diagram
    txt_S = '50'
    K = 13      #number ok cuts in BZ
    txt_DM = '000'  #DM angle from list
    numb_it = 2
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
J = (J1,J2,J3)
S_label = {'50':0.5,'36':(np.sqrt(3)-1)/2,'34':0.34,'30':0.3,'20':0.2}
S = S_label[txt_S]
DM_list = {'000':0,'005':0.05,'104':np.pi/3,'209':2*np.pi/3}
PSG = 'SU2' if txt_DM == '000' else 'TMD'
phi = DM_list[txt_DM]
DM1 = phi;      DM2 = 0;    DM3 = 2*phi
#BZ points
Nx = K;     Ny = K
#Filenames
#DirName = '/home/users/r/rossid/SC_data/S'+txt_S+'/phi'+txt_DM+"/"
DirName = '../../Data/self_consistency/test/'
#DirName = '../Data/SC_data/S'+txt_S+'/phi'+txt_DM+"/"
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
KM = fs.KM(kkg,a1,a2,a12p,a12m)
#### DM
t1 = np.exp(-1j*DM1);    t1_ = np.conjugate(t1)
t2 = np.exp(-1j*DM2);    t2_ = np.conjugate(t2)
t3 = np.exp(-1j*DM3);    t3_ = np.conjugate(t3)
Tau = (t1,t1_,t2,t2_,t3,t3_)
########################
########################    Initiate routine
########################
#Find the parameters that we actually need to use and their labels (some parameters are zero if J2 or J3 are zero
pars = ['A1','A1p','phiA1p','B1','phiB1','B1p','phiB1p']
pars2 = ['A2','phiA2','A2p','phiA2p','B2','phiB2','B2p','phiB2p']
pars3 = ['A3','phiA3','B3','phiB3']
if J2 != 0:
    pars += pars2
if J3 != 0:
    pars += pars3
header = ['J2','J3','Energy','Gap','p1','L'] + pars

t_0 = np.arctan(np.sqrt(2))
print("Computing minimization for parameters: \nS=",S,"\nDM phase = ",phi,'\nPoint in phase diagram(J2,J3) = ('+'{:5.4f}'.format(J2)+',{:5.4f}'.format(J3)+')',
      "\nCuts in BZ: ",K)
######################
###################### Compute the parameters by self concistency
######################
Ai = 1
Bi = 1/2
L_bounds = inp.L_bounds
Ti = t()    #Total initial time
#######################################################################################
for p1 in [0,1]:
    Args_O = (KM,Tau,K,S,J,pars,p1)
    Args_L = (KM,Tau,K,S,J,p1,L_bounds)
    solutions,completed = sf.import_solutions(csvfile,p1)
    if completed:
        print("Ansatze p1=",p1," precedently already computed")
        continue
    for ph in [2]:
        for iph in range(numb_it+1):
            #
            complete_set = sf.check_solutions(solutions,p1)
            if complete_set:
                print("Found everything before the end of the initial phase cycle")
                break
            Pinitial = [Ai,Ai,0,Bi,np.pi,Bi,np.pi]
            if J2:
                Pinitial += [Ai,np.pi,Ai,np.pi,Bi,np.pi,Bi,np.pi]
            if J3:
                Pinitial += [Ai,np.pi,Bi,np.pi]
            Pinitial[ph] = iph*np.pi/numb_it
            ###################################################     Check if result was already obtained
            already_found = False
            for sol in solutions:
                diff = np.abs(Pinitial[ph]-sol[ph+1])
                if diff < inp.cutoff_solution or np.abs(diff-2*np.pi) < inp.cutoff_solution:
                    already_found = True
            if already_found:
                print("Already computed solution at p1=",p1,", par:",pars[ph],"=",Pinitial[ph])
                continue
            print("Computing p1=",p1,", par:",pars[ph],"=",Pinitial[ph])
            Tti = t()
            #
            new_O = Pinitial;      old_O_1 = new_O;      old_O_2 = new_O
            new_L = (L_bounds[1]-L_bounds[0])/2 + L_bounds[0];       old_L_1 = 0;    old_L_2 = 0
            mix_list = [0, 0.1, 0.9, 0.9, 0.4, 0.5, 0.9, 0.9]#, 0.4, 0.6]
            for STEP in range(10):
                print("STEP ",STEP)
                step = 0
                continue_loop = True
                exit_mixing = False
                while continue_loop:    #all pars at once
#                    print("Step ",step,": ",new_L,*new_O,end='\n')
                    conv = 1
                    old_L_2 = float(old_L_1)
                    old_L_1 = float(new_L)
                    new_L = fs.compute_L(new_O,Args_L)
                    old_O_2 = np.array(old_O_1)
                    old_O_1 = np.array(new_O)
                    temp_O = fs.compute_O_all(new_O,new_L,Args_O)
                    mix_factor = random.uniform(0,1)

                    for i in range(len(old_O_1)):
                        if pars[i][0] == 'p' and np.abs(temp_O[i]-old_O_1[i]) > np.pi:
                            temp_O[i] -= 2*np.pi
                        new_O[i] = old_O_1[i]*mix_factor + temp_O[i]*(1-mix_factor)
                        if pars[i][0] == 'p' and new_O[i] < 0:
                            new_O[i] += 2*np.pi
                    step += 1
                    #Check if all parameters are stable up to precision
                    if np.abs(old_L_2-new_L) > inp.cutoff_L:
                        conv *= 0
                    #print(old_O,new_O)
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
                    #Margin in number of steps
                    if step > inp.MaxIter:#*len(pars):
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Going to next STEP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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
            if new_L < inp.L_bounds[0] + 0.01 or new_L > inp.L_bounds[1] - 0.01:
                print("Suspicious L value: ",new_L," NOT saving")
                continue
            ########################################################
            amp_found = ph_found = False
            for sol in solutions:
                diff = 0
                diff += np.abs(new_L-sol[0])
                amp_found = False
                for p in [0,1,3,5]:     #amplitudes
                    diff += np.abs(new_O[p]-sol[p+1])
                if diff < inp.cutoff_solution:
                    amp_found = True
                ph_found = True
                for p in [2,4,6]:
                    diff = np.abs(new_O[p]-sol[p+1])
                    if not (diff < inp.cutoff_solution or np.abs(diff-2*np.pi) < inp.cutoff_solution):
                        ph_found = False
            if amp_found and ph_found:
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
            data = [J2,J3,E,gap,p1,new_L]
            DataDic = {}
            for ind in range(len(data)):
                DataDic[header[ind]] = data[ind]
            for ind2 in range(len(new_O)):
                DataDic[header[len(data)+ind2]] = new_O[ind2]

            sf.SaveToCsv(DataDic,csvfile)

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################













































































