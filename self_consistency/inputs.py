import numpy as np
m = 6
####
list_ans = ['3x3','q0','cb1','cb2','oct']
#derivative
cutoff_L = 1e-8
cutoff_O = 1e-8
MaxIter = 100
prec_L = 1e-10       #precision required in L maximization
cutoff_pts = 1e-10      #min difference b/w phase diagram points to be considered the same
L_method = 'Brent'
L_bounds = (0.3,1.5)
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
header = {'SU2':{ '3x3':    ['ans','J2','J3','Converge','Energy','gap','L','A1','A3','B1','B2','B3','phiA3'],  #3x3
                  'q0':     ['ans','J2','J3','Converge','Energy','gap','L','A1','A2','B1','B2','B3','phiA2'],  #q0
                  'cb1':    ['ans','J2','J3','Converge','Energy','gap','L','A1','A2','A3','B1','B2','phiA1','phiB2'],
                  'cb2':    ['ans','J2','J3','Converge','Energy','gap','L','A1','A2','A3','B1','B2','phiB1','phiA2'],
                  'oct':    ['ans','J2','J3','Converge','Energy','gap','L','A1','A2','B1','B2','B3','phiB1','phiB2']    },
          'TMD':{ '3x3':    ['ans','J2','J3','Converge','Energy','gap','L','A1','A3','B1','B2','B3','phiB1','phiA3','phiB3'],  #3x3
                  'q0':     ['ans','J2','J3','Converge','Energy','gap','L','A1','A2','B1','B2','B3','phiB1','phiA2','phiB3'],  #q0
                  'cb1':    ['ans','J2','J3','Converge','Energy','gap','L','A1','A2','A3','B1','B2','phiA1','phiB1','phiB2'],
                  'cb2':    ['ans','J2','J3','Converge','Energy','gap','L','A1','A2','A3','B1','B2','phiB1','phiA2','phiA2p','phiA3'],
                  'oct':    ['ans','J2','J3','Converge','Energy','gap','L','A1','A2','B1','B2','B3','phiA1','phiB1','phiB2'],   },
          }
num_phi = { 'SU2':{'3x3':1,'q0':1,'cb1':2,'cb2':2,'oct':2},
            'TMD':{'3x3':3,'q0':3,'cb1':3,'cb2':4,'oct':3},
            }
list_A2 = []
list_A3 = []
list_B3 = []
for ans in list_ans:
    lPar = header['SU2'][ans][8:]
    if 'A2' in lPar:
        list_A2.append(ans)
    if 'A3' in lPar:
        list_A3.append(ans)
    if 'B3' in lPar:
        list_B3.append(ans)





