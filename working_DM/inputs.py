import numpy as np
#
m = 6
####
mp_cpu = 1#6
list_ans = ['3x3','q0','cb1']
#derivative
s_b_modulus = 0.01 #bound on values given by smaller grids
s_b_phase   = 0.1 #bound on values given by smaller grids
der_par = 1e-6
der_phi = 1e-5
der_lim = 1  #limit under which compute the Hessian for that parameter
cutoff = 1e-8   ############      #accettable value of Sigma to accept result as converged
MaxIter = 100
prec_L = 1e-10       #precision required in L maximization
cutoff_pts = 1e-12      #min difference b/w phase diagram points to be considered the same
L_method = 'Brent'
L_bounds = (0.4,1.5)
L_b_2 = 0.01
#phase diagram
z = (4,4,2)
J2i = -0.3; J2f = 0.3; J3i = -0.3; J3f = 0.3; Jpts = 9
J= []
for i in range(Jpts):
    for j in range(Jpts):
        J.append((J2i+(J2f-J2i)/(Jpts-1)*i,J3i+(J3f-J3i)/(Jpts-1)*j))
#initial point
header = {'3x3':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A3','B1','B2','B3','phiA3'],  #3x3
          'q0':     ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','B1','B2','B3','phiA2'],  #q0
          'cb1':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiA1','phiB2']  #cuboc1
          }
t_0 = np.arctan(np.sqrt(2))
Pi = {  '3x3':{'A1':0.51, 'A3':0.5, 'B1':0.25, 'B2': 0.41, 'B3': 0.25, 'phiA3': 0},
        'q0':{'A1':0.51, 'A2':0.3, 'B1':0.18, 'B2': 0.18, 'B3': 0.15, 'phiA2': np.pi},
        'cb1':{'A1':0.51, 'A2':0.1, 'A3':0.43, 'B1':0.17, 'B2': 0.17, 'phiA1': 2*t_0, 'phiB2': 2*np.pi-t_0},
        }
lAns = header.keys()
bounds = {}
num_phi = {}
list_A2 = []
list_A3 = []
list_B3 = []
for ans in lAns:
    bounds[ans] = {}
    num_phi[ans] = 0
    lPar = Pi[ans].keys()
    if 'A2' in lPar:
        list_A2.append(ans)
    if 'A3' in lPar:
        list_A3.append(ans)
    if 'B3' in lPar:
        list_B3.append(ans)
    #bounds
    if ans == '3x3':
        bounds[ans]['A1'] = (0.4,0.6)
        bounds[ans]['A3'] = (0,1)
        bounds[ans]['B1'] = (0,0.5)
        bounds[ans]['B2'] = (0,0.5)
        bounds[ans]['B3'] = (0,0.5)
        bounds[ans]['phiA3'] = (-np.pi,np.pi)
        num_phi[ans] = 1
    elif ans == 'q0':
        bounds[ans]['A1'] = (0.4,0.6)
        bounds[ans]['A2'] = (0,1)
        bounds[ans]['B1'] = (0,1)
        bounds[ans]['B2'] = (0,0.5)
        bounds[ans]['B3'] = (0,0.5)
        bounds[ans]['phiA2'] = (0,2*np.pi)
        num_phi[ans] = 1
    elif ans == 'cb1':
        bounds[ans]['A1'] = (0.4,0.6)
        bounds[ans]['A2'] = (0,1)
        bounds[ans]['A3'] = (0,1)
        bounds[ans]['B1'] = (0,0.5)
        bounds[ans]['B2'] = (0,0.5)
        bounds[ans]['phiA1'] = (-np.pi,np.pi)
        bounds[ans]['phiB2'] = (0,2*np.pi)
        num_phi[ans] = 2
shame2 = 100







