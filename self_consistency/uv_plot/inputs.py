import numpy as np
####
ansatze_1 = ['15','16','17','18']
ansatze_2 = ['19','20']
ansatze_p0 = ['15','16','19']
header = ['J1','J2','J3','Energy','Gap','L']
#derivative
cutoff_L = 1e-6
cutoff_O = 1e-6
cutoff_F = 1e-4
cutoff_solution = 1e-3 
MaxIter = 200
prec_L = 1e-10       #precision required in L maximization
cutoff_pts = 1e-10      #min difference b/w phase diagram points to be considered the same
L_method = 'Brent'
L_bounds = (0,5)
L_b_2 = 0.05
#phase diagram
z = (4,4,2)
m = (3,6)

t1 = 1
t2 = t1*0.145
t3 = t1*0.08
Ui = t1*50
Uf = t1*150
UV_pts = 10
U_list = np.linspace(Ui,Uf,UV_pts)
V_list = []
for u in U_list:
    V_list.append(np.linspace(0.08*u,0.15*u,UV_pts))


