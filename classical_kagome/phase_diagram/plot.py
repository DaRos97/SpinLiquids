import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import sys
import getopt
import functions as fs



argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "r:",["DM=","pts=","disp","plot"])
    lim = 0.3
    DM = '005'
    pts = 51
    disp = False
    plot = False
except:
    print("Error in input parameters",argv)
    exit()
for opt, arg in opts:
    if opt in ['-r']:
        lim = float(arg)
    if opt == '--DM':
        DM = arg
    if opt == '--pts':
        pts = int(arg)
    if opt == '--disp':
        disp = True
    if opt == '--plot':
        plot = True
#inputs:    DM_angle -> in str, J_pts

args_sp = (disp,plot)

J1 = 1
J2i = -lim
J2f = lim
J3i = -lim
J3f = lim
J2pts = J3pts = pts
dirname = 'data/'

dic_DM = {'000':0,'005':0.05,'104':np.pi/3,'209':2*np.pi/3}
DM_angle = dic_DM[DM]

filename = dirname+'J2_'+str(J2i)+'--'+str(J2f)+'__J3_'+str(J3i)+'--'+str(J3f)+'__DM_'+DM+'__Pts_'+str(pts)+'.npy'
energies = np.load(filename)
min_E = np.zeros((J2pts,J3pts),dtype = int)
#k -> ferro, r -> s3x3, b -> s3x3_g1, y -> q0, g -> q0_g1, orange -> q0_g2,
#gray -> octa, purple -> octa_g1, m -> octa_g2, c -> cb1, cb2, spiral
legend_names = {
        'ferro':    'k',
        '3x3':      'red',
                '3x3_g1':   'firebrick',
        'q0':          'blue',
                'q0_g1':      'royalblue',
                'q0_g2':      'dodgerblue',
        'octa':        'magenta' ,
                'octa_g1':     'hotpink',
                'octa_g2':     'pink',
        'cb1':      'lime',
                'cb1_g1':      'lawngreen',
                'cb1_g2':      'chartreuse',
        'cb2':      'orange',
                'cb2_g1':      'darkorange',
                'cb2_g2':      'coral',
        'spiral':       'aqua',
        }
orders = list(legend_names.keys())
#
J2 = np.linspace(J2i,J2f,J2pts)
J3 = np.linspace(J3i,J3f,J3pts)
#######
####### Plot spiral energies
if 0:
    X,Y = np.meshgrid(J2,J3)
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(1,1,1,projection='3d')
    #ax.plot_surface(X,Y,energies[:,:,0,15].T,cmap=cm.coolwarm)
    ax.contour(X,Y,energies[:,:,0,15].T,levels=200,cmap=cm.coolwarm)
    plt.show()

fig = plt.figure(figsize=(16,16))
plt.gca().set_aspect('equal')
txt_dm = {'000':r'$0$','005':r'$0.05$','104':r'$\pi/3$','209':r'$2\pi/3$'}
used_o = []
for i in range(J2pts):
    for j in range(J3pts):
        ens = np.array(energies[i,j,:,0])        #Also spirals
        #ens = np.array(energies[i,j,0])        #Also spirals
#        ens = np.array(energies[i,j,0,:-1])     #No spirals
        if (ens == np.zeros(len(ens))).all():
            min_E[i,j] = 1e5
            continue
        aminE = []
        aminE.append(np.argmin(ens))
        minE = ens[aminE[0]]
        ens[aminE[0]] += 5
        while True:
            aminE2 = np.argmin(ens)
            if aminE2 == len(energies[i,j,0])-1:
                break
            minE2 = ens[aminE2]
            if np.abs(minE-minE2) < 1e-5:
                aminE.append(aminE2)
                ens[aminE[-1]] += 5
                continue
            break
        ord_E = []
        for amin in aminE:
            ord_E.append(orders[amin])
        if ord_E[0] == 'spiral' and len(ord_E) > 1:
            ord_E = list(ord_E[1:])
        for e in ord_E:
            if e not in used_o:
                used_o.append(e)
        min_E[i,j] = orders.index(ord_E[0])
        if len(ord_E) == 1:
            if ord_E[0] == 'spiral':
                parameters = energies[i,j,15,1:]            #there is one filling 0 at the end
                #parameters = energies[i,j,1,:-1]            #there is one filling 0 at the end
                if disp:
                    print('\n\nFinding spiral order at ',J2[i],J3[j])
                marker_style = fs.find_order(parameters,args_sp)
            else:
                color1 = legend_names[ord_E[0]]
                mark = 'o'
                marker_style = dict(
                    color=color1, 
                    marker=mark,
                    markeredgecolor='none',
                    markersize = 20,
                    )
            plt.plot(J2[i],J3[j],**marker_style)
        elif len(ord_E) > 1:
            r = [0.25]
            for t in range(len(ord_E)):
                color = legend_names[ord_E[t]]
                r.append(r[-1] + 1/len(ord_E))
                x1 = np.cos(2 * np.pi * np.linspace(r[-2], r[-1]))
                y1 = np.sin(2 * np.pi * np.linspace(r[-2], r[-1]))
                xy1 = np.row_stack([[0, 0], np.column_stack([x1, y1])])
                plt.plot(J2[i],J3[j],marker=xy1,markersize=20,markerfacecolor=color, markeredgecolor='none',linestyle='none')
        else:
            color1 = 'brown'
            marker_style = dict(
                    color=color1, 
                    marker='*',
                    markeredgecolor='none',
                    markersize = 15,
                    )
            plt.plot(J2[i],J3[j],**marker_style)
plt.title('DM = '+txt_dm[DM])
#plt.yticks([-0.3,0,0.3],['-0.3','0','0.3'])
#plt.xticks([-0.3,0,0.3],['-0.3','0','0.3'])

plt.axhline(y=0,color='k',zorder=-1,linewidth=0.5)
plt.axvline(x=0,color='k',zorder=-1,linewidth=0.5)
lp = (J2f-J2i)/10
plt.xlim(J2i-lp,J2f+lp)
plt.ylim(J3i-lp,J3f+lp)

plt.xlabel(r'$J_2$')
plt.ylabel(r'$J_3$')

legend_lines = []
for ord_u in used_o:
    if ord_u == 'spiral':
        legend_lines.append(Line2D([], [], color='none', marker='P', markerfacecolor='k'))
    else:
       legend_lines.append(Line2D([], [], color='none', marker='o', markerfacecolor=legend_names[ord_u]))
plt.legend(legend_lines,used_o,loc='upper left',fancybox=True)#,bbox_to_anchor=(1,1))
plt.show()
exit()


















































#plt.contourf(J2,J3,energies[0][:,:,1].T,alpha=0.3,levels=np.arange(0,12))
levels = np.unique(energies[:,:])
cont = plt.contour(J2,J3,energies[:,:].T,levels=levels)#,alpha=0.3,levels=np.arange(0,12))

def linear(x,a,b):
    return a*x + b
#There are three lines to fit
res = np.zeros((3,4))   #three lines and two parameters per line + 2 bounds on x
#Line 1: between 3x3 and cb1
p = cont.collections[1].get_paths()[0]
v = p.vertices
for i in range(len(v[:,0])):
    if v[i,0] > 0:      #> 0.1
        ix_max = i 
        break
x = v[:ix_max,0]
y = v[:ix_max,1]
pin = [-1,0]
res[0,:2],cov = curve_fit(linear,x,y,p0=pin,bounds=(-100,100))
res[0,2] = x[0]
plt.plot(x,linear(x,res[0,0],res[0,1]),'r-')
#Line 2: between cb1 and q=0
x = v[ix_max:,0]
y = v[ix_max:,1]
pin = [1,0]
res[1,:2],cov = curve_fit(linear,x,y,p0=pin,bounds=(-100,100))
res[1,3] = x[-1]
plt.plot(x,linear(x,res[1,0],res[1,1]),'r-')
#Line 3: between 3x3 and q=0
p = cont.collections[0].get_paths()[0]
v = p.vertices
for i in range(len(v[:,1])):
    if v[i,1] < 0:          #<0.1
        ix_max = i 
        break
x = v[ix_max:,0]
y = v[ix_max:,1]
pin = [1,0]
res[2,:2],cov = curve_fit(linear,x,y,p0=pin,bounds=(-100,100))
res[2,2] = x[-1]
plt.plot(x,linear(x,res[2,0],res[2,1]),'r-')
#
plt.show()

np.save("fit_"+sys.argv[1]+"_"+sys.argv[2]+".npy",res)

