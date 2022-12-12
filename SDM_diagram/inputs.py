import numpy as np
m = 6
####
mp_cpu = 16
list_ans = ['1a','1b','1c','1c1','1c2','1d','1e','1f','1f0','1f1','1f2','1f3','1f4']
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
L_b_2 = 0.01
header = {'1a':    ['ans','S','DM','Converge','Energy','Sigma','gap','L','A1','B1','phiB1'], 
          '1b':    ['ans','S','DM','Converge','Energy','Sigma','gap','L','A1','B1','phiB1'],
          '1c':    ['ans','S','DM','Converge','Energy','Sigma','gap','L','A1','B1','phiB1'],
          '1c1':    ['ans','S','DM','Converge','Energy','Sigma','gap','L','A1','B1','phiB1'],
          '1c2':    ['ans','S','DM','Converge','Energy','Sigma','gap','L','A1','B1','phiB1'],
          '1d':    ['ans','S','DM','Converge','Energy','Sigma','gap','L','A1','B1','phiB1'],
          '1e':    ['ans','S','DM','Converge','Energy','Sigma','gap','L','A1','B1','phiA1','phiB1'],
          '1f':    ['ans','S','DM','Converge','Energy','Sigma','gap','L','A1','B1','phiA1','phiB1'],
          '1f0':    ['ans','S','DM','Converge','Energy','Sigma','gap','L','A1','B1','phiA1','phiB1'],
          '1f1':    ['ans','S','DM','Converge','Energy','Sigma','gap','L','A1','B1','phiA1','phiB1'],
          '1f2':    ['ans','S','DM','Converge','Energy','Sigma','gap','L','A1','B1','phiA1','phiB1'],
          '1f3':    ['ans','S','DM','Converge','Energy','Sigma','gap','L','A1','B1','phiA1','phiB1'],
          '1f4':    ['ans','S','DM','Converge','Energy','Sigma','gap','L','A1','B1','phiA1','phiB1'],
          }
num_phi = {'1a':1,'1b':1,'1c':1,'1c1':1,'1c2':1,'1d':1,'1e':2,'1f':2,'1f0':2,'1f1':2,'1f2':2,'1f3':2,'1f4':2}
shame2 = 100
list_A2 = []
list_A3 = []
list_B2 = []
list_B3 = []
z = [4,4,2]






