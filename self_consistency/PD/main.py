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
DM_list = {'000':0,'005':0.05,'104':np.pi/3,'209':2*np.pi/3}
PSG = 'SU2' if txt_DM == '000' else 'TMD'
phi = DM_list[txt_DM]
DM1 = phi;      DM2 = 0;    DM3 = 2*phi
#BZ points
Nx = K;     Ny = K
#Filenames
#DirName = '/home/users/r/rossid/SC_data/S'+txt_S+'/phi'+txt_DM+"/"
DirName = '../../Data/self_consistency/S'+txt_S+'/phi'+txt_DM+"/"
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

#Checks the file (specified by J2 and J3) and tells you which ansatze need to be computed
ansatze = sf.CheckCsv(csvfile)
#Find the initial point for the minimization for each ansatz
t_0 = np.arctan(np.sqrt(2))
#Put initial values by classical paramters !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Pi_ = {'SU2': { '3x3':{'A1':np.sqrt(3)/4, 'A3':np.sqrt(3)/4, 'B1':0.25, 'B2': 0.5, 'B3': 0.5, 'phiA3': 0},
                'q0': {'A1':np.sqrt(3)/4, 'A2':np.sqrt(3)/4, 'B1':0.25, 'B2': 0.25, 'B3': 0.5, 'phiA2': np.pi},
                'cb1':{'A1':np.sqrt(3)/4, 'A2':0.25, 'A3':0.5, 'B1':0.25, 'B2': np.sqrt(3)/4, 'phiA1': 2*t_0, 'phiB2': 2*np.pi-t_0},
                'cb2':{'A1':0.25, 'A2':np.sqrt(3)/4, 'A3':0.5, 'B1':np.sqrt(3)/4, 'B2': 0.25, 'phiB1': np.pi+t_0, 'phiA2': np.pi+t_0},
                'oct':{'A1':1/(2*np.sqrt(2)), 'A2':1/(2*np.sqrt(2)), 'B1':1/(2*np.sqrt(2)), 'B2': 1/(2*np.sqrt(2)), 'B3':0.5, 'phiB1': 5/4*np.pi, 'phiB2': np.pi/4},    },
       'TMD': { '3x3':{'A1':np.sqrt(3)/4, 'A3':np.sqrt(3)/4, 'B1':0.25, 'B2': 0.5, 'B3': 0.5, 'phiB1':np.pi, 'phiA3': 0, 'phiB3': np.pi},
                'q0': {'A1':np.sqrt(3)/4, 'A2':np.sqrt(3)/4, 'B1':0.25, 'B2': 0.25, 'B3': 0.5, 'phiB1':0, 'phiA2': np.pi, 'phiB3': 0},
                'cb1':{'A1':np.sqrt(3)/4, 'A2':0.25, 'A3':0.5, 'B1':0.25, 'B2': np.sqrt(3)/4, 'phiA1': 2*t_0, 'phiB1':np.pi, 'phiB2': 2*np.pi-t_0},
                'cb2':{'A1':0.25, 'A2':np.sqrt(3)/4, 'A3':0.5, 'B1':np.sqrt(3)/4, 'B2': 0.25, 'phiB1': np.pi+t_0, 'phiA2':np.pi-t_0, 'phiA2p': np.pi+t_0, 'phiA3': 0},
                'oct':{'A1':1/(2*np.sqrt(2)), 'A2':1/(2*np.sqrt(2)), 'B1':1/(2*np.sqrt(2)), 'B2': 1/(2*np.sqrt(2)), 'B3':0.5, 'phiA1': np.pi, 'phiB1':5/4*np.pi, 'phiB2': np.pi/4},  },
        }
Pinitial, done, L_dic  = sf.FindInitialPoint(J2,J3,ansatze,ReferenceDir,Pi_[PSG])            ####################
#Find the bounds to the free parameters for each ansatz
print("Computing minimization for parameters: \nS=",S,"\nDM phase = ",phi,'\nPoint in phase diagram(J2,J3) = ('+'{:5.4f}'.format(J2)+',{:5.4f}'.format(J3)+')',
      "\nCuts in BZ: ",K)
######################
###################### Compute the parameters by self concistency
######################
Ai = S/S
Bi = S/2/S
Ti = t()    #Total initial time
for ans in ansatze:
    print("Computing ansatz ",ans)
    Tti = t()   #Initial time of the ansatz
    header = inp.header[PSG][ans]
    #Find the parameters that we actually need to use and their labels (some parameters are zero if J2 or J3 are zero
    j2 = int(np.sign(J2)*np.sign(int(np.abs(J2)*1e8)) + 1)   #j < 0 --> 0, j == 0 --> 1, j > 0 --> 2
    j3 = int(np.sign(J3)*np.sign(int(np.abs(J3)*1e8)) + 1)
    pars2 = Pi_[PSG][ans].keys()
    pars = []
    for pPp in pars2:
        if (pPp[-1] == '1') or (pPp[-1] == '2' and j2-1) or (pPp[-1] == '3' and j3-1):
            pars.append(pPp)
    L_bounds = (L_dic[ans] - inp.L_bnd_ref, L_dic[ans] + inp.L_bnd_ref) if L_dic[ans] else inp.L_bounds
    Args_L = (J1,J2,J3,ans,KM,Tau,K,S,PSG,L_bounds)
    pars2 = Pi_[PSG][ans].keys()
    pars = []
    for pPp in pars2:
        if (pPp[-1] == '1') or (pPp[-1] == '2' and j2-1) or (pPp[-1] == '3' and j3-1):
            pars.append(pPp)
    if 'phiA1' in pars:
        pars[pars.index('phiA1')] = 'phiA1p'
    Args_O = (J1,J2,J3,ans,KM,Tau,K,PSG,pars)
    #
    new_O = Pinitial[ans];      old_O_1 = new_O;      old_O_2 = new_O
    new_L = (L_bounds[1]-L_bounds[0])/2 + L_bounds[0];       old_L_1 = 0;    old_L_2 = 0
    print(pars)
    mix_list = [0, 0.1, 0.9, 0.9, 0.4, 0.5, 0.9, 0.9]#, 0.4, 0.6]
    for mix_factor in mix_list:
        print("Using mixing ",mix_factor)
        step = 0
        continue_loop = True
        exit_mixing = False
        while continue_loop:    #all pars at once
            print("Step ",step,": ",new_L,*new_O,end='\n')
            conv = 1
            old_L_2 = float(old_L_1)
            old_L_1 = float(new_L)
            new_L = fs.compute_L(new_O,Args_L)
            old_O_2 = np.array(old_O_1)
            old_O_1 = np.array(new_O)
            temp_O = fs.compute_O_all(new_O,new_L,Args_O)
            E1 = 1
            for i in range(len(new_O)):
                if np.abs(old_O_1[i]-temp_O[i]) > inp.cutoff_O:
                    E1 *= 0
            E2 = 1
            for i in range(len(new_O)):
                if np.abs(old_O_2[i]-temp_O[i]) > inp.cutoff_O:
                    E2 *= 0
            if not E2 and not E1:
                mix_factor = random.uniform(0,1)
            else: 
                mix_factor = 0

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
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Not converged!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                break
#        print("Step ",step,": ",new_L,new_O)
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
    E,gap = fs.total_energy(new_O,new_L,Args_L)
    for i in range(len(new_O)):
        if pars[i][0] == 'p':
            if new_O[i] > 2*np.pi:
                new_O[i] = new_O[i]-2*np.pi
            if new_O[i] < -0.2:
                new_O[i] = new_O[i]+2*np.pi
    if E==0:
        print("WTF?? not saving\n\n")
        continue
    #Format the parameters in order to have 0 values in the non-considered ones
    Format_params = {'SU2':fs.FormatParams_SU2, 'TMD':fs.FormatParams_TMD}
    newP = Format_params[PSG](new_O,ans,J2,J3)
    #Store the files in a dictionary
    data = [ans,J2,J3,conv,E,gap,new_L]
    DataDic = {}
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(newP)):
        DataDic[header[len(data)+ind2]] = newP[ind2]
    #Save values to an external file
    print(DataDic)
    print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################
    input()
    sf.SaveToCsv(DataDic,csvfile,PSG)

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################














































































