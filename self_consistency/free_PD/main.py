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
Ti = t()
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:S:K:",["DM=","disp"])
    N = 40      #inp.J point in phase diagram
    txt_S = '50'
    K = 13      #number ok cuts in BZ
    txt_DM = '000'  #DM angle from list
    numb_it = 3
    disp = False
except:
    print("Error in input parameters",argv)
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)
    if opt in ['-S']:
        txt_S = arg
    if opt in ['-K']:
        K = int(arg)
    if opt == '--DM':
        txt_DM = arg
    if opt == '--disp':
        disp = True 
J1 = 1
J2, J3 = inp.J[N]
J = (J1,J2,J3)
S_label = {'50':0.5,'36':(np.sqrt(3)-1)/2,'34':0.34,'30':0.3,'20':0.2}
S = S_label[txt_S]
DM_list = {'000':0,'005':0.05,'104':np.pi/3,'209':2*np.pi/3}
phi = DM_list[txt_DM]
DM1 = phi;      DM2 = 0;    DM3 = 2*phi
#BZ points
Nx = K;     Ny = K
#Filenames
#DirName = '/home/users/r/rossid/0_SELF-CONSISTENCY_PD/Data/S'+txt_S+'/phi'+txt_DM+"/"
#DirName = '../../Data/self_consistency/test/'
DirName = '../../Data/self_consistency/S'+txt_S+'/phi'+txt_DM+"/"
DataDir = DirName + str(Nx) + '/'
ReferenceDir = DirName + str(13) + '/'
csvname = 'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
csvfile = DataDir + csvname
csvref = ReferenceDir + csvname
#BZ points
kxg = np.linspace(0,1,Nx)
kyg = np.linspace(0,1,Ny)
kkg = np.ndarray((2,Nx,Ny),dtype=complex)
kkg_small = np.ndarray((2,Nx,Ny),dtype=complex)
for i in range(Nx):
    for j in range(Ny):
        kkg[0,i,j] = kxg[i]*2*np.pi
        kkg[1,i,j] = (kxg[i]+kyg[j])*2*np.pi/np.sqrt(3)
        kkg_small[0,i,j] = kxg[i]*2*np.pi
        kkg_small[1,i,j] = (kxg[i]+2*kyg[j])*2*np.pi/np.sqrt(3)
#### vectors of 1nn, 2nn and 3nn
a1 = (1,0)
a2 = (-1,np.sqrt(3))
a2_small = (-1/2,np.sqrt(3)/2)
a12p = (a1[0]+a2[0],a1[1]+a2[1])
a12m = (a1[0]-a2[0],a1[1]-a2[1])
a12p_small = (a1[0]+a2_small[0],a1[1]+a2_small[1])
a12m_small = (a1[0]-a2_small[0],a1[1]-a2_small[1])
#### product of lattice vectors with K-matrix
KM_big = fs.compute_KM(kkg,a1,a2,a12p,a12m)     #large unit cell
KM_small = fs.compute_KM(kkg_small,a1,a2_small,a12p_small,a12m_small)
#### DM
t1 = np.exp(-1j*DM1);    t1_ = np.conjugate(t1)
t2 = np.exp(-1j*DM2);    t2_ = np.conjugate(t2)
t3 = np.exp(-1j*DM3);    t3_ = np.conjugate(t3)
Tau = (t1,t1_,t2,t2_,t3,t3_)
########################
########################    Initiate routine
########################
#Find the parameters that we actually need to use and their labels (some parameters are zero if J2 or J3 are zero
list_ansatze,list_PpP,list_phases,L_ref = sf.find_lists(J2,J3,csvfile,K,numb_it)
#
for ans in list_ansatze:
    L_bounds = inp.L_bounds #if K == 13 else (L_ref[ans]-inp.L_b_2,L_ref[ans]+inp.L_b_2)
    KM = KM_small if ans in inp.ansatze_p0 else KM_big
    index_mixing_ph = 1 if ans in inp.ansatze_2 else 2
    head_ans,pars = sf.find_head(ans,J2,J3)
    for it_p,PpP in enumerate(list_PpP[ans]):
        solutions = sf.import_solutions(csvfile,ans,PpP,J2,J3)
        Args_O = (KM,Tau,K,S,J,pars,ans,PpP)
        Args_L = (KM,Tau,K,S,J,pars,ans,PpP,L_bounds)
        for new_phase in range(list_phases[ans][it_p]):
            Pinitial = sf.find_Pinitial(new_phase,numb_it,S,ans,pars,csvref,K,PpP)
            completed = sf.check_solutions(solutions,index_mixing_ph,Pinitial[index_mixing_ph])
            if completed:
#                print("Already found solution for ans ",ans," at p=",PpP," and phase ",Pinitial[index_mixing_ph])
                continue
            print("\nComputing ans ",ans," p=",PpP,", par:",pars[index_mixing_ph],"=",Pinitial[index_mixing_ph])
            Tti = t()
            #
            new_O = Pinitial;      old_O_1 = new_O;      old_O_2 = new_O
            new_L = (L_bounds[1]-L_bounds[0])/2 + L_bounds[0];       old_L_1 = 0;    old_L_2 = 0
            #
            step = 0
            continue_loop = True
            while continue_loop:
                if disp:
                    print("Step ",step,": ",new_L,*new_O[:],end='\n')
                conv = 1
                old_L_2 = float(old_L_1)
                old_L_1 = float(new_L)
                new_L = fs.compute_L(new_O,Args_L)
                old_O_2 = np.array(old_O_1)
                old_O_1 = np.array(new_O)
                temp_O = fs.compute_O_all(new_O,new_L,Args_O)
                #
                mix_factor = random.uniform(0,1) #if K == 13 else 0
                mix_phase = 0.5#random.uniform(0,1)
                mix_phase2 = 0#mix_phase #random.uniform(0,1)
                #
                ind_imp_phase = 1 if ans in ['19','20'] else 2
#                print(pars)
                for i in range(len(old_O_1)):
                    if pars[i][0] == 'p' and i == ind_imp_phase:
#                        if np.abs(temp_O[i]-old_O_1[i]) > np.pi:
#                            temp_O[i] -= 2*np.pi
                        new_O[i] = np.angle(np.exp(1j*(old_O_1[i]*mix_phase + temp_O[i]*(1-mix_phase))))
                        if new_O[i] < 0:
                            new_O[i] += 2*np.pi
                    elif pars[i][0] == 'p':
                        new_O[i] = np.angle(np.exp(1j*(old_O_1[i]*mix_phase2 + temp_O[i]*(1-mix_phase2))))
                        if new_O[i] < 0:
                            new_O[i] += 2*np.pi
                        if np.abs(new_O[i]-2*np.pi) < 1e-3:
                            new_O[i] -= 2*np.pi
                    else:
                        new_O[i] = old_O_1[i]*mix_factor + temp_O[i]*(1-mix_factor)
                step += 1
                #Check if all parameters are stable up to precision
                if np.abs(old_L_2-new_L)/S > inp.cutoff_L:
                    conv *= 0
                #print(old_O,new_O)
                for i in range(len(new_O)):
                    if pars[i][0] == 'p':
                        if np.abs(old_O_1[i]-new_O[i]) > inp.cutoff_F or np.abs(old_O_2[i]-new_O[i]) > inp.cutoff_F:
                            conv *= 0
                    else:
                        if np.abs(old_O_1[i]-new_O[i])/S > inp.cutoff_O or np.abs(old_O_2[i]-new_O[i])/S > inp.cutoff_O:
                            conv *= 0
                if conv:
                    continue_loop = False
                    new_L = fs.compute_L(new_O,Args_L)
                #Margin in number of steps
                if step > inp.MaxIter:#*len(pars):
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Exceeded number of steps!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    break
            ######################################################################################################
            ######################################################################################################
#            print("\nNumber of iterations: ",step,'\n')
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
                diff = np.abs(new_L-sol[0])
                amp_found = False
                for p_ in sf.amp_list(pars):     #amplitudes
                    diff += np.abs(new_O[p_]-sol[p_+1])
                if diff < inp.cutoff_solution:
                    amp_found = True
                ph_found = True
                for p_ in sf.phase_list(pars):
                    diff = np.abs(new_O[p_]-sol[p_+1])
                    if not (diff < inp.cutoff_solution or np.abs(diff-2*np.pi) < inp.cutoff_solution):
                        ph_found = False
                if amp_found and ph_found:
#                    print("Already found solution, phase ",pars[index_mixing_ph],"=",new_O[index_mixing_ph])
                    break
            if amp_found and ph_found:
                continue
            if ans in ['19','20'] and (np.abs(new_O[1])<inp.cutoff_solution or np.abs(new_O[1]-np.pi)<inp.cutoff_solution or np.abs(new_O[1]-2*np.pi)<inp.cutoff_solution):
                continue
            r = [new_L] + list(new_O)
            solutions.append(r)
            ################################################### Save solution
            E,gap = fs.total_energy(new_O,new_L,Args_L)
            if E == 0:
                continue
            data = [ans]
            for p_ in PpP:
                if p_ == 2:
                    continue
                data.append(p_)
            data += [J2,J3,E,gap,new_L]
            DataDic = {}
            for ind in range(len(data)):
                DataDic[head_ans[ind]] = data[ind]
            for ind2 in range(len(new_O)):
                DataDic[head_ans[len(data)+ind2]] = new_O[ind2]
            print(DataDic)
            print("Time of solution : ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
            sf.SaveToCsv(DataDic,csvfile)

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################














































































