import numpy as np
import sys
import getopt
import functions as fs


argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "s:",["spiral","j2=","j3=","DM=","la="])
    J2 = 0
    J3 = 0
    DM1 = 0
    lattice = 0
    spiral = False
except:
    print("Error in input parameters",argv)
    exit()
for opt, arg in opts:
    if opt == '--spiral':
        spiral = True
    if opt == '--DM':
        DM1 = float(arg)
    if opt == '--j2':
        J2 = float(arg)
    if opt == '--j3':
        J3 = float(arg)
    if opt == '--la':
        lattice = int(arg)



spin_angles = (0,0)
orders = ['ferro', '3x3', 'q0', 'octa', 'cb1', 'cb2']
m = [1,3,3,1,3,3,2,6,6,2,6,6,2,6,6]
func_L = {'ferro': fs.ferro_lattice, 'q0': fs.q0_lattice, 'octa': fs.oct_lattice, 'cb1': fs.cb1_lattice, 'cb2': fs.cb2_lattice}
lattices = []
for o in orders:
    if o == '3x3':
        continue
    L = func_L[o](spin_angles)
    lattices.append(L.copy())
    for g in range(2):
        fs.gauge_trsf(L)
        lattices.append(L.copy())
#order of lattices: FM,3x3,3x3_g1,q0,q0_g1,q0_g2,....

##################################
J = (1,J2,J3)
DM_ = (DM1,0,2*DM1)
if not spiral:
    #en = fs.energy(lattices[lattice],6,J,DM_)
    en = fs.lat_energy(lattices[lattice],6,J,DM_)
    en1 = fs.lat_energy_old(lattice,6,J,DM_)
    en2 = fs.energy_2(lattices[lattice],6,J,DM_)
    print(en,en1,en2)
    exit()
else:
    inv1 = 1
    inv2 = 1
    #
    t0 = np.pi/6#0.5237263441912974
    t1 = 5*np.pi/6#2.6175274428655895
    t2 = np.pi/2#1.5711993671965403
    t3 = np.pi/6#0.5236833209913809
    t4 = 5*np.pi/6#2.617365299632181
    t5 = np.pi/2#1.5710006431964836
    #
    p1 = 0#0.04666007502382152
    p2 = np.pi#3.164914845074376
    p3 = 2*np.pi#6.237147983633946
    p4 = 0#0.0006431548314513869
    p5 = np.pi#3.118895424153782
    R1 = 0.09221588341957299
    R2 = 6.191096158387747
    #
    P = (t0,t1,p1,t2,p2,t3,p3,t4,p4,t5,p5,R1,R2)
    args = (J,(inv1,inv2),DM_)
    en = fs.spiral_energy(P,*args)
#    en = fs.spiral(J,DM_)
print("energy: ",en)
