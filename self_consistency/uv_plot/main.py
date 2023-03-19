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
    opts, args = getopt.getopt(argv, "N:S:K:",["disp"])
    N = 40      #inp.J point in phase diagram
    txt_S = '50'
    K = 13      #number ok cuts in BZ
    numb_it = inp.numb_it
    save_to_file = True
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
    if opt == '--disp':
        disp = True 
U = inp.U_list[N//inp.UV_pts]
V = inp.V_list[N//inp.UV_pts][N%inp.UV_pts]
parameters = sf.coefficients(inp.t1,inp.t2,inp.t3,U,V,2*np.pi/3,np.pi,np.pi/3)

J1 = parameters['J1_z']
J2 = parameters['J2_z']#/np.abs(parameters['J1_z'])
J3 = parameters['J3e_z']#/np.abs(parameters['J1_z'])
J = np.array([J1,J2,J3])
J_max = np.abs(np.amax(np.abs(J)))
for i in range(3):
    J[i] /= J_max
inp.L_bounds[1] = 2*np.abs(np.amax(np.abs(J)))
J = tuple(J)
print('computing ',J,' and U,V = ',U,V/U)
S_label = {'50':0.5,'36':(np.sqrt(3)-1)/2,'34':0.34,'30':0.3,'20':0.2}
S = S_label[txt_S]
###########################
phi = 2*np.pi/3#####################################
###########################
DM1 = phi;      DM2 = 0;    DM3 = 2*phi
#BZ points
Nx = K;     Ny = K
#Filenames
DirName = '/home/users/r/rossid/0_SELF-CONSISTENCY_UV/Data/S'+txt_S+'/'
#DirName = '../../Data/self_consistency/test/'
#DirName = '../../Data/self_consistency/S'+txt_S+'/'
DataDir = DirName + str(Nx) + '/'
ReferenceDir = DirName + str(13) + '/'
csvname = 'U_V=('+'{:5.4f}'.format(U).replace('.','')+'_'+'{:5.4f}'.format(V).replace('.','')+').csv'
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
list_ansatze,list_PpP,list_phases,L_ref = sf.find_lists(J2,J3,csvfile,csvref,K,numb_it)
#
for ans in list_ansatze:
    KM = KM_small if ans in inp.ansatze_p0 else KM_big
    index_mixing_ph = 1 if ans in inp.ansatze_2 else 2
    head_ans,pars = sf.find_head(ans,J2,J3)
    for it_p,PpP in enumerate(list_PpP[ans]):
        solutions = sf.import_solutions(csvfile,ans,PpP,J2,J3)
        Args_O = (KM,Tau,K,S,J,pars,ans,PpP)
        for new_phase in range(list_phases[ans][it_p]):
            Pinitial,Linitial = sf.find_Pinitial(new_phase,numb_it,S,ans,pars,csvref,K,PpP)
            if K == 13:
                L_bounds = inp.L_bounds 
            else:
                L_bounds = (Linitial-inp.L_b_2,Linitial+inp.L_b_2)
            Args_L = (KM,Tau,K,S,J,pars,ans,PpP,L_bounds)
            completed = sf.check_solutions(solutions,index_mixing_ph,Pinitial[index_mixing_ph])
            if completed:
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
                mix_factor = random.uniform(0,1) #if K == 13 else random.uniform(0.5,1)
                mix_phase = 0.5#random.uniform(0,1)
                mix_phase2 = 0#mix_phase #random.uniform(0,1)
                #
                ind_imp_phase = 1 if ans in ['19','20'] else 2
                for i in range(len(old_O_1)):
                    if pars[i][0] == 'p' and i == ind_imp_phase:
                        new_O[i] = np.angle(np.exp(1j*(old_O_1[i]*mix_phase + temp_O[i]*(1-mix_phase))))
                        if new_O[i] < 0:
                            new_O[i] += 2*np.pi
                    elif pars[i][0] == 'p':
                        new_O[i] = np.angle(np.exp(1j*(old_O_1[i]*mix_phase2 + temp_O[i]*(1-mix_phase2))))
                        if new_O[i] < 0:
                            new_O[i] += 2*np.pi
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
                    break
            if amp_found and ph_found:
                continue
#            if ans in ['19'] and (np.abs(new_O[1])<inp.cutoff_solution or np.abs(new_O[1]-np.pi)<inp.cutoff_solution):
#                continue
#            if ans == '20' and (np.abs(new_O[1]) < inp.cutoff_solution or (np.abs(new_O[1]-np.pi)<inp.cutoff_solution and (np.abs(new_O[3])<inp.cutoff_solution or np.abs(new_O[3]-np.pi)<inp.cutoff_solution))):
#                continue
            ind_B1 = 1 if ans in inp.ansatze_1 else 2
            if np.abs(new_O[0])<inp.cutoff_solution and np.abs(new_O[ind_B1])<inp.cutoff_solution:       #if A1 == 0 and B1 == 0
                continue
            pos_sol = True
            for par_o in new_O:
                if par_o < -1e-3:
                    pos_sol = False
            if not pos_sol:
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
            data += [J[0],J[1],J[2],E,gap,new_L]
            DataDic = {}
            for ind in range(len(data)):
                DataDic[head_ans[ind]] = data[ind]
            for ind2 in range(len(new_O)):
                DataDic[head_ans[len(data)+ind2]] = new_O[ind2]
            print(DataDic)
            print("Time of solution : ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
            if save_to_file:
                sf.SaveToCsv(DataDic,csvfile)

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################














































































