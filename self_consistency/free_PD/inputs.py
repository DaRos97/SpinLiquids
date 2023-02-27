import numpy as np
####
ansatze_1 = ['15','16','17','18']
ansatze_2 = ['19','20']
ansatze_p0 = ['15','16','19']
header = ['J2','J3','Energy','Gap','L']
#derivative
cutoff_L = 1e-6
cutoff_O = 1e-6
cutoff_F = 1e-4
cutoff_solution = 1e-3 
MaxIter = 500
prec_L = 1e-10       #precision required in L maximization
cutoff_pts = 1e-10      #min difference b/w phase diagram points to be considered the same
L_method = 'Brent'
L_bounds = (0.2,5)
L_b_2 = 0.01
#phase diagram
z = (4,4,2)
m = (6,6)
J2i = -0.3; J2f = 0.3; J3i = -0.3; J3f = 0.3; Jpts = 9
J= []
for i in range(Jpts):
    for j in range(Jpts):
        J.append((J2i+(J2f-J2i)/(Jpts-1)*i,J3i+(J3f-J3i)/(Jpts-1)*j))
#initial point





