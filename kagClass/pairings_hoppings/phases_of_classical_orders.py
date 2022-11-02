import numpy as np
import cmath
import sys
import getopt
#input arguments
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "g:",["ans="])
    ans = '3x3'
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt == "--ans":
        ans = arg
print("Computing LRO ",ans)
#alfa = A^dag*A
def alfa(x,Spins_):
    return complex(1/2 + Spins_[x][2],0)
#beta = B^dag*B
def beta(x,Spins_):
    return complex(1/2 - Spins_[x][2],0)
#gamma = A^dag*B
def gamma(x,Spins_):
    return complex(Spins_[x][0],Spins_[x][1])

#Pairing and hopping "parameters". Need to put the * at the end only on the intermediate ones
#A_12 * A23 * A_31 * A14 * A_45 * A51
A_12 = ['A1*B2*','-B1*A2*']
A23  = ['a2*b3*','-b2*a3*']
A_31 = ['A3*B1*','-B3*A1*']
A14  = ['a1*b4*','-b1*a4*']
A_45 = ['A4*B5*','-B4*A5*']
A51  = ['a5*b1','-b5*a1']
B31 = ['A3*a1','B3*b1']
#
B12 = ['A1*a2*','B1*b2*']
A_23  = ['A2*B3*','-B2*A3*']
A31 = ['a3*b1','-b3*a1']
#
A_34 = ['A3*B4*','-B3*A4*']
A41 = ['a4*b1','-b4*a1']
A34 = ['a3*b4*','-b3*a4*']
B41 = ['A4*a1','B4*b1']

#Spins planar orders
a_p = np.array([0,1/2,0])
b_p = np.array([-np.sqrt(3)/4,-1/4,0])
c_p = np.array([np.sqrt(3)/4,-1/4,0])
#Spins cuboc 1 and 2
t0 = np.arctan(np.sqrt(2))
a_c = np.array([1/2,0,0])
b_c = np.array([1/4,np.sqrt(3)/4,0])
c_c = np.array([-1/4,np.sqrt(3)/4,0])
d_c = np.array([0,np.cos(t0)/2,np.sin(t0)/2])
e_c = np.array([-np.sqrt(3)*np.cos(t0)/4,-np.cos(t0)/4,np.sin(t0)/2])
f_c = np.array([np.sqrt(3)*np.cos(t0)/4,-np.cos(t0)/4,np.sin(t0)/2])
#Spins octahedral
a_o = np.array([1/2,0,0])
b_o = np.array([0,1/2,0])
c_o = np.array([0,0,1/2])

#Inputs:
##Loops
A1p = [A_12,A23,A_31,A14,A_45,A51]            #phi_A1'
B1 = [A_12,A23,B31]                          #phi_B1/phi_B1'
B1p = B1
A2 = [B12,A_23,A31]                          #phi_A2/phi_A2'
A2p = A2
B2 = [A_12,A23,B31]                          #phi_B2/phi_B2'
B2p = B2
A3 = [A_12,A23,A_34,A41]                     #phi_A3
B3 = [B12,A_23,A34,B41]                      #phi_B3
##Spins -> without repetitions
S_q0 =  {'A1p': [b_p,c_p,a_p,a_p,c_p],      'B1': [b_p,c_p,a_p],    'B1p': [c_p,a_p,b_p],   'A2': [b_p,a_p,c_p],    'A2p': [a_p,b_p,c_p],   'B2': [b_p,a_p,c_p],    'B2p': [a_p,b_p,c_p],   'A3': [a_p,b_p,c_p,a_p],    'B3': [a_p,b_p,c_p,a_p]}
S_3x3 = {'A1p': [c_p,a_p,b_p,a_p,b_p],      'B1': [b_p,c_p,a_p],    'B1p': [a_p,c_p,b_p],   'A2': [c_p,a_p,c_p],    'A2p': [a_p,c_p,a_p],   'B2': [c_p,a_p,c_p],    'B2p': [a_p,c_p,a_p],   'A3': [a_p,c_p,a_p,c_p],    'B3': [a_p,c_p,a_p,c_p]}
S_cb1 = {'A1p': [-c_c,d_c,-f_c,b_c,-a_c],   'B1': [-c_c,d_c,-f_c],  'B1p': [d_c,-b_c,-e_c], 'A2': [-c_c,b_c,-d_c],  'A2p': [b_c,-c_c,d_c],  'B2': [-c_c,b_c,-d_c],  'B2p': [b_c,-c_c,d_c],  'A3': [b_c,-c_c,d_c,-b_c],  'B3': [b_c,-c_c,d_c,-b_c]}
S_cb2 = {'A1p': [e_c,d_c,f_c,-b_c,-a_c],    'B1': [e_c,d_c,f_c],    'B1p': [d_c,b_c,c_c],   'A2': [c_c,b_c,a_c],    'A2p': [-b_c,e_c,d_c],  'B2': [c_c,b_c,a_c],    'B2p': [-b_c,e_c,d_c],  'A3': [-b_c,e_c,d_c,b_c],   'B3': [-b_c,e_c,d_c,b_c]}
S_oct = {'A1p': [-b_o,c_o,a_o,-a_o,-c_o],   'B1': [b_o,-c_o,a_o],   'B1p': [-c_o,a_o,b_o],  'A2': [b_o,-a_o,-c_o],  'A2p': [-a_o,b_o,-c_o], 'B2': [b_o,-a_o,-c_o],  'B2p': [-a_o,b_o,-c_o], 'A3': [-a_o,b_o,-c_o,-a_o],  'B3': [-a_o,b_o,-c_o,-a_o]}
#
Spins_dic = {'q0':S_q0, '3x3':S_3x3, 'cb1':S_cb1, 'cb2':S_cb2, 'oct':S_oct}
A_dic = {'A1p':A1p,'B1':B1,'B1p':B1p,'A2':A2,'A2p':A2p,'B2':B2,'B2p':B2p,'A3':A3,'B3':B3}
Phis = ['A1p','B1','B1p','A2','A2p','B2','B2p','A3','B3']
Anss = ['q0', '3x3', 'cb1', 'cb2', 'oct']
p1_dic = {'q0':0, '3x3':0, 'cb1':1, 'cb2':1, 'oct':1}
p1 = p1_dic[ans]
if ans not in Anss:
    print("not a good ansatz")
    exit()
###################
################### Actual code
phases_result = {}
#Loop over all phaseas because we need all of them to compute the longer range ones
for phi in Phis:
    A = A_dic[phi]
    eta = 0 if phi == 'A1p' else 1
    Spins = Spins_dic[ans][phi]
    ################
    #First compute all the possible terms coming from the operators in the A-list
    list_all = ['a','b','A','B']
    r = A[0]
    for i1 in range(len(Spins)-1*eta):      #-1 removed for phi_A1'
        temp = A[i1+1]
        r2 = []
        for i2 in range(len(r)):
            for i3 in range(len(temp)):
                if (r[i2][0] == '-' and temp[i3][0] == '-'):
                    r2.append(r[i2][1:]+temp[i3][1:])
                elif (r[i2][0] == '-' and temp[i3][0] in list_all):
                    r2.append(r[i2]+temp[i3])
                elif (temp[i3][0] == '-' and r[i2][0] in list_all):
                    r2.append('-'+r[i2]+temp[i3][1:])
                else:
                    r2.append(r[i2]+temp[i3])
        r = r2
    #Now calculate their value one by one
    Calculus = complex(0,0)
    result = []
    for num,res in enumerate(r):
        temp_C = complex(1,0)
        if res[0] == '-':       #remove the minus in front if it is there
            res = res[1:]
            result.append(str(num+1)+': -')
            sign = -1
        else:
            result.append(str(num+1)+': ')
            sign = 1
        #Split the terms in the single multiplication
        terms = res.split('*')
        #For each term consider the spin sites one by one
        for i in range(1,len(Spins)+1):        #6 for phi_A1'
            temp = []
            for t in terms:             #extract terms with same lattice position
                if t[1] == str(i):
                    temp.append(t)
            #For each spin site compute the associated alfa, beta or gamma
            if len(temp) > 2:#loops through i have more than two terms in the single product -> passes through 1 two times -> 4 terms
                temp1 = [[temp[0],temp[3]],[temp[1],temp[2]]]
                for temp_ in temp1:
                    if (temp_[0][0] == 'A' and temp_[1][0] == 'a') or (temp_[0][0] == 'a' and temp_[1][0] == 'A'):          #alfa
                        result[num] += 'a' + str(i)
                        temp_C *= alfa(i-1,Spins)
                    elif (temp_[0][0] == 'A' and temp_[1][0] == 'b') or (temp_[0][0] == 'b' and temp_[1][0] == 'A'):        #gamma
                        result[num] += 'g' + str(i)
                        temp_C *= gamma(i-1,Spins)
                    elif (temp_[0][0] == 'B' and temp_[1][0] == 'a') or (temp_[0][0] == 'a' and temp_[1][0] == 'B'):        #gamma_
                        result[num] += 'g_' + str(i)
                        temp_C *= np.conjugate(gamma(i-1,Spins))
                    elif (temp_[0][0] == 'B' and temp_[1][0] == 'b') or (temp_[0][0] == 'b' and temp_[1][0] == 'B'):        #beta
                        result[num] += 'b' + str(i)
                        temp_C *= beta(i-1,Spins)
                    else:
                        print("not recognized 1")
                        exit()
            else:
                if (temp[0][0] == 'A' and temp[1][0] == 'a') or (temp[0][0] == 'a' and temp[1][0] == 'A'):
                    result[num] += 'a' + str(i)
                    temp_C *= alfa(i-1,Spins)
                elif (temp[0][0] == 'A' and temp[1][0] == 'b') or (temp[0][0] == 'b' and temp[1][0] == 'A'):
                    result[num] += 'g' + str(i)
                    temp_C *= gamma(i-1,Spins)
                elif (temp[0][0] == 'B' and temp[1][0] == 'a') or (temp[0][0] == 'a' and temp[1][0] == 'B'):
                    result[num] += 'g_' + str(i)
                    temp_C *= np.conjugate(gamma(i-1,Spins))
                elif (temp[0][0] == 'B' and temp[1][0] == 'b') or (temp[0][0] == 'b' and temp[1][0] == 'B'):
                    result[num] += 'b' + str(i)
                    temp_C *= beta(i-1,Spins)
                else:
                    print("not recognized 2")
                    exit()
        #Sum the term to the result
        Calculus += temp_C * sign

    if np.abs(Calculus) < 1e-10:
        phases_result[phi] = np.nan
        continue

    if phi == 'A1p':
        phases_result[phi] = cmath.phase(Calculus) + np.pi
    elif phi == 'A2':
        phases_result[phi] = cmath.phase(Calculus) + phases_result['B1p'] - np.pi
    elif phi == 'A2p':
        phases_result[phi] = cmath.phase(Calculus) - phases_result['B1p'] - np.pi
    elif phi == 'B2':
        phases_result[phi] = cmath.phase(Calculus) + phases_result['A1p']
    elif phi == 'B2p':
        phases_result[phi] = -cmath.phase(Calculus) - phases_result['A1p']
    elif phi == 'A3':
        phases_result[phi] = cmath.phase(Calculus) + 2*phases_result['A1p'] - p1*np.pi
    elif phi == 'B3':
        phases_result[phi] = cmath.phase(Calculus) - phases_result['A1p'] - phases_result['B1p'] - p1*np.pi
    elif phi in ['B1','B1p']:
        phases_result[phi] = cmath.phase(Calculus)

    if phases_result[phi] >= 4*np.pi:
        phases_result[phi] -= 2*np.pi
    if phases_result[phi] >= 2*np.pi:
        phases_result[phi] -= 2*np.pi
    if phases_result[phi] < -2*np.pi:
        phases_result[phi] += 2*np.pi
    if phases_result[phi] < 0:
        phases_result[phi] += 2*np.pi

print('Result = ',phases_result)
