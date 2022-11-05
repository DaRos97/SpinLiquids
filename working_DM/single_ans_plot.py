import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
import os
import sys
import getopt
from matplotlib import cm

Color = {'3x3': ['red','firebrick'],
         'q0':  ['yellow','y'],
         'cb1':  ['lime','limegreen']
         }
#Arguments: -S -> spin(03/05), -a -> ansatz, -p -> phase (0/0.06...)
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:a:p:N:",["plot"])
    S = '05'
    N = '13'
    ans = '3x3'
    phi = '000'
    plot = False
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-S']:
        S = arg
    elif opt in ['-a']:
        ans = arg
    elif opt in ['-p']:
        phi = arg
    elif opt in ['-N']:
        N = arg
    if opt == '--plot':
        plot = True
#S = '05'
#S = '03'
#ans = sys.argv[1]
phi = "{:3.2f}".format(float(phi)).replace('.','')
dirname = '../Data/S'+S+'/phi'+phi+'/'+N+'/'; title = 'With DM interactions'
D = {}
Ji = -0.3
Jf = 0.3
J2 = np.linspace(Ji,Jf,9)
J3 = np.linspace(Ji,Jf,9)
X,Y = np.meshgrid(J2,J3)
if ans == 'cb1_nc':
    ans_ = 'cb1'
else:
    ans_ = ans
Head = inp.header[ans_][3:]
head = []
for h in Head:
    head.append(h)
for h in head:
    D[h] = np.zeros((9,9))
    D[h][:] = np.nan
for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    N_ = (len(lines)-1)//2 + 1
    tempE = []
    for i in range(N_):
        data = lines[i*2+1].split(',')
        if data[0] == ans:
            j2 = float(data[1]) - Ji
            j3 = float(data[2]) - Ji
            i2 = int(j2*8/(0.6))
            i3 = int(j3*8/(0.6))
            for n,h in enumerate(head):
                if n == 0:
                    D[h][i2,i3] = (1 if data[n+3]=='True' else np.nan)
                    if data[n+3] == 'False':
                        for h2 in head[1:]:
                            D[h2][i2,i3] = np.nan
                        break
                else:
                    try:
                        D[h][i2,i3] = float(data[n+3])
                        if D[h][i2,i3] == 0:
                            D[h][i2,i3] = np.nan
                    except:
                        print("not good: ",h,i2,i3)
            break
print("Non converged points: ",int(81-np.sum(~np.isnan(D['Converge'].ravel()))),"\n",D['Converge'])
nP = len(head)
for i in range(nP):
    temp = []
    for l in range(9):
        for j in range(9):
            if D[head[i]][l,j] == 0:
                D[head[i]][l,j] = np.nan
    for p in D[head[i]][~np.isnan(D[head[i]])].ravel():
        if p != 0 and p != np.nan and p != 'nan':
            temp.append(p)
    try:
        print("Range of ",head[i],":",np.amin(temp),"--",np.amax(temp))
    except:
        print("Range with only 0 or nan values")
    #print("Range of ",head[i],":",np.amin(D[head[i]][np.nonzero(~np.isnan(D[head[i]]))]),"--",np.amax(D[head[i]][~np.isnan(D[head[i]])]))
fig = plt.figure()#(figsize=(16,16))
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.axis('off')
for i in range(nP):
    ax = fig.add_subplot(4,4,i+1,projection='3d')
    ax.plot_surface(X,Y,D[head[i]].T,cmap=cm.coolwarm)
    ax.set_title(ans)
    ax.set_xlabel("J2")
    ax.set_ylabel("J3")
    ax.set_title(head[i])
if plot:
    plt.show()
