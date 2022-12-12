import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
import os
import sys
import getopt
from matplotlib import cm

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "K:a:",["plot"])
    K = '13'
    ans = '1a'
    plot = False
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-a']:
        ans = arg
    if opt in ['-K']:
        K = arg
    if opt == '--plot':
        plot = True
dirname = '../Data/SDM/'+K+'/'
D = {}
S_max = 0.5
DM_max = 0.5
S_pts = 10
DM_pts = 15
S_list = np.linspace(0.05,S_max,S_pts,endpoint=True)
DM_list = np.logspace(-5,np.log(DM_max),DM_pts,base = np.e)
DM_list_neg = np.flip(-DM_list)#.reverse()
DM_list_neg = np.append(DM_list_neg,np.array([0]),axis = 0)
DM_list = np.concatenate((DM_list_neg,DM_list))
X,Y = np.meshgrid(DM_list,S_list)
Head = inp.header[ans][3:]
head = []
for h in Head:
    head.append(h)
for h in head:
    D[h] = np.zeros((2*DM_pts+1,S_pts))
    D[h][:] = np.nan
for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    N_ = (len(lines)-1)//2 + 1
    tempE = []
    for i in range(N_):
        data = lines[i*2+1].split(',')
        if data[0] == ans:
            dm = list(DM_list).index(float(data[2]))
            s = list(S_list).index(float(data[1]))
            for n,h in enumerate(head):
                if n == 0:
                    D[h][dm,s] = (1 if data[n+3]=='True' else np.nan)
                    if data[n+3] == 'False':
                        for h2 in head[1:]:
                            D[h2][dm,s] = np.nan
                        break
                else:
                    try:
                        D[h][dm,s] = float(data[n+3])
                        if D[h][dm,s] == 0:
                            D[h][dm,s] = np.nan
                    except:
                        print("not good: ",h,dm,s)
            break
print("Non converged points: ",int((2*DM_pts+1)*S_pts-np.sum(~np.isnan(D['Converge'].ravel()))),"\n",D['Converge'])
nP = len(head)
for i in range(nP):
    temp = []
    for l in range(2*DM_pts+1):
        for j in range(S_pts):
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
if plot:
    fig = plt.figure()#(figsize=(16,16))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.axis('off')
    for i in range(nP):
        ax = fig.add_subplot(3,3,i+1,projection='3d')
        ax.plot_surface(X,Y,D[head[i]].T,cmap=cm.coolwarm)
        ax.set_title(ans)
        ax.set_xlabel("DM")
        ax.set_ylabel("S")
        ax.set_title(head[i])
    plt.show()




