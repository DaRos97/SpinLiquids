import numpy as np
from pathlib import Path
import inputs as inp
import csv
import os
import random
def find_lists_2(J2,J3,csvfile,csvref,K,numb_it):
    if K == 13:
        ansatze = ['16','19','20','15','17','18']#inp.ansatze_1+inp.ansatze_2
        my_file = Path(csvfile)
        if my_file.is_file():
            with open(my_file,'r') as f:
                lines = f.readlines()
            N = (len(lines)-1)//2 +1        #2 lines per ansatz
            for i in range(N):
                head = lines[i*2].split(',')
                head[-1] = head[-1][:-1]
                data = lines[i*2+1].split(',')
                if data[0] in ansatze:
                    ansatze.remove(data[0])
    else:
        ansatze = ['nan',]
        my_file = Path(csvref)
        if my_file.is_file():
            with open(my_file,'r') as f:
                lines = f.readlines()
            N = (len(lines)-1)//2 +1        #2 lines per ansatz
            minE = 10
            for i in range(N):
                head = lines[i*2].split(',')
                head[-1] = head[-1][:-1]
                data = lines[i*2+1].split(',')
                temp_E = float(data[head.index('Energy')])
                if temp_E < minE:
                    minE = temp_E
                    ansatze[0] = data[0]
    if J2:
        list_p = {'15':[(1,1),], '16':[(0,0),], '20':[(1,1),], '17':[(1,1),], '19':[(1,1),], '18':[(0,0),]}
        if J3:
            list_p = {'15':[(1,1),], '16':[(0,0),], '20':[(1,1,0,0),], '17':[(1,1),], '19':[(1,1,0,0),], '18':[(0,0),]}
    elif J3:
        list_p = {'15':[(2,2),], '16':[(2,2),], '20':[(0,0),], '17':[(2,2),], '19':[(0,0),], '18':[(2,2),]}
    else:
        list_p = {'15':[(2,2),], '16':[(2,2),], '20':[(2,2),], '17':[(2,2),], '19':[(2,2),], '18':[(2,2),]}
    list_phases = {'15':[1,],'16':[1,],'20':[1,],'17':[1,],'19':[1,],'18':[1,]}
    return ansatze, list_p, list_phases, 0

###############################################################
def find_lists(J2,J3,csvfile,csvref,K,numb_it):
    if K == 13:
        ansatze = ['15','16','19','17','20','18']#inp.ansatze_1+inp.ansatze_2
        list_p = {}
        list_phases = {}
        for ans in ansatze:
            if ans in inp.ansatze_1:
                if J2:
                    result = [(0,0),(0,1),(1,0),(1,1),]
                else:
                    result = [(2,2)]
            elif ans in inp.ansatze_2:
                l4 = []
                l2 = []
                for p2 in [0,1]:
                    for p3 in [0,1]:
                        l2.append((p2,p3))
                        for p4 in [0,1]:
                            for p5 in [0,1]:
                                l4.append((p2,p3,p4,p5))
                if (J2==0 and J3) or (J3==0 and J2):
                    result = l2
                elif J2 and J3:
                    result = l4
                else:
                    result = [(2,2)]
            list_p[ans] = result
            list_phases[ans] = []
            for i in range(len(list_p[ans])):
                list_phases[ans].append(numb_it)
        return ansatze, list_p, list_phases,0
    ansatze = []
    list_p = {}
    Ls = {}
    list_phases = {}
    my_file = Path(csvref)
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 +1        #2 lines per ansatz
        for i in range(N):
            head = lines[i*2].split(',')
            head[-1] = head[-1][:-1]
            data = lines[i*2+1].split(',')
            valid = True
            for j in range(head.index('A1'),len(data)):
                if float(data[j]) < -1e-3:
                    valid = False
            if not valid:
                continue
            #Compute set of P
            temp_p = []
            for j in range(1,head.index('J2')):
                temp_p.append(int(data[j]))
            if len(temp_p) == 0:
                temp_p = [2,2]
            temp_p = tuple(temp_p)
            if data[0] not in ansatze:  #If new ansatz, add it, with corresponding set of P, and an entry in phases
                ansatze.append(data[0])
                list_p[data[0]] = [temp_p,]
                list_phases[data[0]] = [1,]
                Ls[data[0]] = float(data[head.index('L')])
            else:           #If ans already in list, but solution is anyway valid
                k_ = 0
                for ttt,PpP in enumerate(list_p[data[0]]):
                    same_p = True
                    for k in range(len(temp_p)):
                        if temp_p[k] != PpP[k]:
                            same_p = False
                    if same_p:
                        k_ = ttt
                        break
                if same_p:
                    list_phases[data[0]][k_] += 1
                else:
                    list_p[data[0]].append(temp_p)
                    list_phases[data[0]].append(1)
    return ansatze, list_p, list_phases, Ls
#
def find_Pinitial(new_phase,numb_it,S,ans,pars,csvfile,K,PpP):
    if K == 13:
        Ai = S
        Bi = S/2
        index_mixing_ph = 1 if ans in inp.ansatze_2 else 2
        Pinitial  = []
        for i in range(len(pars)):
            if i == index_mixing_ph:
                Pinitial.append(new_phase/(numb_it-1)*2*np.pi) ######
                #phase = {'15':np.pi,'16':np.pi,'20':1.95,'17':np.pi,'19':2,'18':np.pi}           ######
                #Pinitial.append(phase[ans])                         ######
                continue
            if pars[i][0] == 'p':
#                if ans == '16' and pars[i] == 'phiA3':
#                    Pinitial.append(0)
#                elif ans == '15' and pars[i] == 'phiB3':
#                    Pinitial.append(0)
#                else:
                Pinitial.append(random.uniform(0,2*np.pi))
            elif pars[i][0] == 'A':
                Pinitial.append(random.uniform(0.1,S))
            elif pars[i][0] == 'B':
                Pinitial.append(random.uniform(0.1,S))
        return Pinitial,0
    phase_name = 'phiA1p' if ans in inp.ansatze_2 else 'phiB1'
    my_file = Path(csvfile)
    cont_phi = -1
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 +1        #2 lines per ansatz
        for i in range(N):
            head = lines[i*2].split(',')
            head[-1] = head[-1][:-1]
            data = lines[i*2+1].split(',')
            valid = True
            for j in range(head.index('A1'),len(data)):
                if float(data[j]) < -1e-3:
                    valid = False
            if not valid:       #skip if not valid solution
                continue
            if data[0] != ans:      #skip if not correct ansatz
                continue
            temp_p = []
            for j in range(1,head.index('J2')):
                temp_p.append(int(data[j]))
            if len(temp_p) == 0:
                temp_p = [2,2]
            temp_p = tuple(temp_p)
            same_p = True
            for it_p in range(len(PpP)):
                if PpP[it_p] != temp_p[it_p]:
                    same_p = False
            if not same_p:
                continue
            cont_phi += 1
            if cont_phi != new_phase:
                continue
            Pinitial = []
            for j in range(head.index('A1'),len(data)):
                Pinitial.append(float(data[j]))
            Linitial = float(data[head.index('L')])
            return Pinitial, Linitial
#
def find_head(ans,J2,J3):
    head_p = ['ans']
    if ans in inp.ansatze_1:
        if J2:
            head_p += ['p2','p3']
        pars = ['A1','B1','phiB1']
    elif ans in inp.ansatze_2:
        if J2:
            head_p += ['p2','p3']
        if J3:
            head_p += ['p4','p5']
        pars = ['A1','phiA1p','B1','phiB1']
    header = head_p + inp.header
    if J2:
        if ans in ['15','17']:
            pars += ['A2','phiA2','A2p','phiA2p','B2','B2p']
        elif ans in ['16','18']:
            pars += ['B2','B2p']
        if ans in ['19','20']:
            pars += ['A2','A2p','B2','phiB2','B2p','phiB2p']
    if J3:
        if ans == '15':
            pars += ['B3','phiB3']
        elif ans == '16':
            pars += ['A3','phiA3','B3','phiB3']
        elif ans == '17':
            pars += ['A3','phiA3']
        if ans in ['19','20']:
            pars += ['A3','B3']
    return header+pars, pars
#
def import_solutions(filename,ans,p,J2,J3):
    solutions = []
    ps = []
    if ans in inp.ansatze_1:
        if J2:
            bias = 2
        else:
            bias = 0
    elif ans in inp.ansatze_2:
        if (J2==0 and J3) or (J3==0 and J2):
            bias = 2
        elif J2 and J3:
            bias = 4
        else:
            bias = 0
    my_file = Path(filename)
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 +1        #2 lines per ansatz
        for i in range(N):
            data = lines[i*2+1].split(',')
            if data[0] == ans:
                temp_p = []
                for pps in range(1,bias+1):
                    temp_p.append(int(data[pps]))
                if bias == 0:
                    temp_p = (2,2)
                right = True
                for bb in range(bias):
                    if temp_p[bb] != p[bb]:
                        right = False
                if right:
                    r = []
                    for par in range(5+bias,len(data)):
                        r.append(float(data[par]))
                    solutions.append(r)
    return solutions
#
def check_solutions(solutions,index,new_phase):
    for sol in solutions:
        phase = float(sol[index+1])
        if np.abs(phase-new_phase) < inp.cutoff_solution or np.abs(phase-new_phase-2*np.pi) < inp.cutoff_solution:
            return True
    return False
#
def amp_list(pars):
    res = []
    for i,p in enumerate(pars):
        if p[0] in ['A','B']:
            res.append(i)
    return res
#
def phase_list(pars):
    res = []
    for i,p in enumerate(pars):
        if p[0] == 'p':
            res.append(i)
    return res
#Save the dictionaries in the file given, rewriting the already existing data if precision is better
def SaveToCsv(Data,csvfile):
    header = Data.keys()
    with open(csvfile,'a') as f:
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writeheader()
        writer.writerow(Data)


#Coefficients Js in terms of ts and U,V
def coefficients(t1,t2,t3,U,V,phi1=0,phi2=0,phi3=0,by_order=False):

    # second order

    J1_2_z = 4*t1**2/(U-V)
    # print ('2nd order:', J1_2_z)
    J2_2_z = 4*t2**2/U
    J3_2_z = 4*t3**2/U

    J1_2_pm = 0.5*J1_2_z*np.exp(-2*1j*phi1)
    J2_2_pm = 0.5*J2_2_z*np.exp(-2*1j*phi2)
    J3_2_pm = 0.5*J3_2_z*np.exp(-2*1j*phi3)

    # third order

    J1_3a = -2*t1**3*(U+V)/((U-V)*V**2)
    J1_3b = -2*t1**2*t2*(U+2*V)/((U-V)*V**2)

    J1_3_z  = 2*np.cos(3*phi1)*J1_3a + 2*np.cos(phi2)*J1_3b
    # print ('3rd order:', J1_3_z)
    J1_3_pm = J1_3a*np.exp(1j*phi1) + J1_3b*np.cos(phi2)*np.exp(-2*1j*phi1)

    J2_3a = -2*t1**2*t2*(U+2*V)/(U*V**2)
    J2_3b = -2*t2**3*(U+2*V)/(U*V**2)
    J3_3a = -2*t1**2*t3*(U+2*V)/(U*V**2)
    J3_3b = -4*t1*t2*t3*(U+2*V)/(U*V**2)

    J2_3_z  = 2*np.cos(phi2)*J2_3a + 2*np.cos(3*phi2)*J2_3b
    J2_3_pm = J2_3a*np.exp(-1j*phi2) + J2_3b*np.exp(1j*phi2)
    J3_3_z  = 2*np.cos(phi3-2*phi1)*J3_3a
    J3_3_pm = J3_3a*np.exp(-1j*(phi3+2*phi1))

    J3_3f_z1  =  2*np.cos(phi1-phi2+phi3)*J3_3b
    J3_3f_z2  =  2*np.cos(phi1+phi2+phi3)*J3_3b
    J3_3f_pm1 =  np.exp(1j*(phi1-phi2-phi3))*J3_3b
    J3_3f_pm2 =  np.exp(1j*(phi1+phi2-phi3))*J3_3b

    # fourth order

    J1_4a = t1**4*(-42./((U-V)**3)  - 7./((U-V)**2*V) - 10./((U-V)*V**2) - 3./(V**3) + 18./((U+V)*(U-V)**2) + 4./((U+V)*V**2) + 16./(U*(U-V)**2) + 8./((U+V)*(U-V)*V) + 32./((U-V)**2*(2*U-V))  + 32./((U-V)**2*(2*U-3*V)))
    J1_4b = t1**4*( 3./((U-V)**3) + 2./((U-V)*V**2) + 4./((U+V)*V**2) + 3./(2*(U-V)**2*V) - 1./(2*V**3) )

    J1_4_z  = 2*(J1_4a + J1_4b)
    J1_4_pm = J1_4a*np.exp(-2*1j*phi1) + J1_4b*np.exp(4*1j*phi1)

    J2_4_pm = t1*4 * ( 5./((U-V)**3) - 2./(U*(U-V)**2) + 6./(U*V**2)  + 1./((U-V)**2*V) - 1./(V**3) - 1./((U-V)*V**2) )
    J2_4_z  = 2*J2_4_pm

    J3_4f = 2*t1**4 * (2./((U-V)**3) - 1./(U*(U-V)**2))
    J3_4e = 6*t1**4 / (U*V**2)

    J3_4f_z  = 2*J3_4f
    J3_4f_pm = J3_4f*np.exp(-4*1j*phi1)
    J3_4e_z  = 2*J3_4e
    J3_4e_pm = J3_4e*np.exp(-4*1j*phi1)

    # print ('4th order:', J1_4_z, '\n')

    J1_z  = J1_2_z + J1_3_z + J1_4_z
    J1_pm = J1_2_pm + J1_3_pm + J1_4_pm
    J2_z  = J2_2_z + J2_3_z + J2_4_z
    J2_pm = J2_2_pm + J2_3_pm + J2_4_pm
    J3f_z  = J3_2_z + J3_4f_z # + J3_3f_z1 + J3_3f_z2
    J3f_pm = J3_2_pm + J3_4f_pm # + J3_3f_pm1 + J3_3f_pm2
    J3e_z  = J3_2_z + J3_3_z + J3_4e_z
    J3e_pm  = J3_2_pm + J3_3_pm + J3_4e_pm

    J4 = 8*t1**4*(1./((U-V)**3) - 1./((U-V)**2*(2*U-V))- 1./((U-V)**2*(2*U-3*V)))

    J12 = (-6*t1**3*t2 + 4*t1**2*t2**2 + 6*t1**3*t3 + 3*t1*t2**2*t3) / (3*V**2*V) \
        - 16*t1**4 / ((U-V)**2*(U+V)) \
        + (32*t1**4 - 8*t1**3*t2 + 8*t1**2*t2**2) / ((U-V)**2*(U+3*V)) \
        + (-16*t1**4 - 18*t1**3*t2 + 24*t1**2*t2**2 + 8*t1**3*t3 + 8*t1*t2**2*t3) / (V*(U+3*V)*(U-V)) \
        + (-6*t1**4 - 4*t1**3*t2 + 10*t1**2*t2**2 + 2*t1**3*t3 + 2*t1*t2**2*t3) / (V**2*(U+3*V))

    J23 = (3*t1**3*t2 + 9*t1**3*t3 + 6*t1**2*t2*t3) / (3*V**3) \
        + (-4*t1**4 + 8*t1**3*t2 + 4*t1**2*t2**2) / ((U-V)**2*(U+3*V)) \
        + (-5*t1**4 + 6*t1**3*t2 + 8*t1**2*t2**2 + 12*t1**3*t3 + 12*t1**2*t2*t3) / (V*(U+3*V)*(U-V)) \
        + (-1*t1**4 + 1*t1**3*t2 + 4*t1**2*t2**2 + 3*t1**3*t3 + 3*t1**2*t2*t3) / (V**2*(U+3*V))


    coeffs = {}

    if by_order:
        coeffs['J1_z']   = [J1_2_z, J1_3_z, J1_4_z]
        coeffs['J1_pm']  = [J1_2_pm, J1_3_pm, J1_4_pm]
        coeffs['J2_z']   = [J2_2_z, J2_3_z, J2_4_z]
        coeffs['J2_pm']  = [J2_2_pm, J2_3_pm, J2_4_pm]
        coeffs['J3f_z']  = [J3_2_z, J3_4f_z]
        coeffs['J3f_pm'] = [J3_2_pm, J3_4f_pm]
        coeffs['J3e_z']  = [J3_2_z, J3_3_z, J3_4e_z]
        coeffs['J3e_pm'] = [J3_2_pm, J3_3_pm, J3_4e_pm]
        coeffs['J1_add'] = 0.5*J12 + J23
        
    else:
        coeffs['J1_z']   = J1_z # + 0.5*J12 + J23
        coeffs['J1_pm']  = J1_pm
        coeffs['J2_z']   = J2_z
        coeffs['J2_pm']  = J2_pm
        coeffs['J3f_z']  = J3f_z
        coeffs['J3f_pm'] = J3f_pm
        coeffs['J3e_z']  = J3e_z
        coeffs['J3e_pm'] = J3e_pm
        coeffs['J3f_add'] = {'z1': J3_3f_z1, 'z2': J3_3f_z2, 'pm1': J3_3f_pm1, 'pm2': J3_3f_pm2}
        
    coeffs['J4']     = J4

    return coeffs
