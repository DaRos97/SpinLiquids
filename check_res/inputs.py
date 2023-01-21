import numpy as np
m = 6
####
mp_cpu = 1
list_ans = ['3x3','q0','cb1','cb1_nc','cb1_2','cb1_nc2']#,'cb2','oct']
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
L_bounds = (0.2,1.5)
L_bnd_ref = 0.1                     #range for bounds of L when given as initial condition a previous result
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
          'cb1':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiA1','phiB2'],
          'cb1_nc':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiA1','phiB2'],
          'cb1_2':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiA1','phiB2'],
          'cb1_nc2':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiA1','phiB2'],
          'cb2':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiB1','phiA2'],
          'oct':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','B1','B2','B3','phiB1','phiB2']
          }
num_phi = {'3x3':1,'q0':1,'cb1':2,'cb1_nc':2,'cb1_2':2,'cb1_nc2':2,'cb2':2,'oct':2}
list_A2 = []
list_A3 = []
list_B3 = []
for ans in list_ans:
    lPar = header[ans][8:]
    if 'A2' in lPar:
        list_A2.append(ans)
    if 'A3' in lPar:
        list_A3.append(ans)
    if 'B3' in lPar:
        list_B3.append(ans)
shame2 = 100







