import numpy as np
####
ansatze_1 = ['15','16','17','18']
ansatze_2 = ['19','20']
ansatze_p0 = ['15','16','19']
#derivative
cutoff_L = 1e-6
cutoff_O = 1e-6
cutoff_F = 1e-4
cutoff_solution = 1e-3
MaxIter = 2000
L_method = 'Brent'
L_bounds = (0.1,5)
L_bnd_ref = 0.1                     #range for bounds of L when given as initial condition a previous result
prec_L = 1e-10          #precision in L minimization
#phase diagram
z = 4
m = (6,6)


#S and DM
S_max = 0.5
DM_max = 0.15
S_pts = 30
DM_pts = 30
S_list = np.linspace(0.01,S_max,S_pts,endpoint=True)
DM_list = np.linspace(0,DM_max,DM_pts,endpoint=True)

#Lattice vectors
a1 = (1,0)
a2 = (-1,np.sqrt(3))
a12p = (a1[0]+a2[0],a1[1]+a2[1])
a12m = (a1[0]-a2[0],a1[1]-a2[1])
