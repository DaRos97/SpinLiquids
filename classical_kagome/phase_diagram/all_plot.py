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
    opts, args = getopt.getopt(argv, "r:",["pts=","PSG="])
    lim = 3.0
    pts = 21
    PSG = "SU2"
except:
    print("Error in input parameters",argv)
    exit()
for opt, arg in opts:
    if opt in ['-r']:
        lim = float(arg)
    elif opt == '--PSG':
        PSG = arg
    elif opt == '--pts':
        pts = int(arg)

J1 = 1
J2i = -lim
J2f = lim
J3i = -lim
J3f = lim
J2pts = J3pts = pts
dirname = 'data/'

dic_DM = {'000':0,'005':0.05,'104':np.pi/3,'209':2*np.pi/3}
list_DM = ['000','104','209'] if PSG == 'SU2' else ['005']

#k -> ferro, r -> s3x3, b -> s3x3_g1, y -> q0, g -> q0_g1, orange -> q0_g2,
#gray -> octa, purple -> octa_g1, m -> octa_g2, c -> cb1, cb2, spiral
legend_names = {
        'ferro':    'k',
        '3x3':      'red',
                '3x3_g1':   'firebrick',
        'q0':          'blue',
                'q0_g1':      'dodgerblue',
                'q0_g2':      'lightskyblue',
        'octa':        'magenta' ,
                'octa_g1':     'hotpink',
                'octa_g2':     'pink',
        'cb1':      'limegreen',
                'cb1_g1':      'lime',
                'cb1_g2':      'chartreuse',
        'cb2':      'hotpink',
                'cb2_g1':      'violet',
                'cb2_g2':      'lightpink',
        'spiral':       'aqua',
        }
dic_text_ord = {'ferro':    r'$FM$',
                '3x3':      r'$\sqrt{3}\times\sqrt{3}_a$',
                '3x3_g1':   r'$\sqrt{3}\times\sqrt{3}_b$',
                'q0':       r'$\mathbf{Q}=0$',
                'q0_g1':       r'$\mathbf{Q}=0_a$',
                'q0_g2':       r'$\mathbf{Q}=0_b$',
                'octa':     r'$octahedral$',
                'octa_g1':     r'$octahedral_a$',
                'octa_g2':     r'$octahedral_b$',
                'cb1':      r'$cuboc-1$',
                'cb1_g1':      r'$cuboc-1_a$',
                'cb1_g2':      r'$cuboc-1_b$',
                'cb2':      r'$cuboc-2$',
                'cb2_g1':      r'$cuboc-2_a$',
                'cb2_g2':      r'$cuboc-2_b$',
                'spiral':      r'$spiral$'
                }
orders = list(legend_names.keys())
txt_dm = {'000':r'$0$','005':r'$0.05$','104':r'$\pi/3$','209':r'$2\pi/3$'}
J2 = np.linspace(J2i,J2f,J2pts)
J3 = np.linspace(J3i,J3f,J3pts)
lp = (J2f-J2i)/15
width = 25 # if len(list_DM) == 3 else 15
fig = plt.figure()#figsize=(width,12))
plt.rcParams.update({
    "text.usetex": True,
#    "font.family": "Helvetica"
})
#   Routine
sub_plot_col = len(list_DM)
for nnnn,DM in enumerate(list_DM):
    plt.subplot(1,len(list_DM),nnnn+1)
    filename = dirname+'J2_'+str(J2i)+'--'+str(J2f)+'__J3_'+str(J3i)+'--'+str(J3f)+'__DM_'+DM+'__Pts_'+str(pts)+'.npy'
    energies = np.load(filename)
    min_E = np.zeros((J2pts,J3pts),dtype = int)
#
    plt.gca().set_aspect('equal')
#plt.title(Title)
    used_o = []
    for i in range(J2pts):
        for j in range(J3pts):
            ens = np.array(energies[i,j,0,:-1])
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
    #        print('Point: ',J2[i],J3[j],' has orders',*ord_E)
            ###
            for e in ord_E:
                if e not in used_o:
                    used_o.append(e)
            min_E[i,j] = orders.index(ord_E[0])
            if len(ord_E) == 1:
                if ord_E[0] == 'spiral':
                    parameters = energies[i,j,1,:-1]
                    #print('\n\nFinding spiral order at ',J2[i],J3[j])
                    color1 = fs.find_order(parameters)
                    mark = "P"
                else:
                    color1 = legend_names[ord_E[0]]
                    mark = 'o'
                marker_style = dict(
                        color=color1, 
                        marker=mark,
                        markeredgecolor='none',
                        markersize = 5,
                        )
#                plt.plot(J2[i],J3[j],**marker_style)
            elif len(ord_E) > 1:
                r = [0.25]
                for t in range(len(ord_E)):
                    color = legend_names[ord_E[t]]
                    r.append(r[-1] + 1/len(ord_E))
                    x1 = np.cos(2 * np.pi * np.linspace(r[-2], r[-1]))
                    y1 = np.sin(2 * np.pi * np.linspace(r[-2], r[-1]))
                    xy1 = np.row_stack([[0, 0], np.column_stack([x1, y1])])
#                    plt.plot(J2[i],J3[j],marker=xy1,markersize=5,markerfacecolor=color, markeredgecolor='none',linestyle='none')
            else:
                color1 = 'brown'
                marker_style = dict(
                        color=color1, 
                        marker='*',
                        markeredgecolor='none',
                        markersize = 5,
                        )
#                plt.plot(J2[i],J3[j],**marker_style)
    ss = 20
    plt.title(r'$\phi = $'+txt_dm[DM],fontsize=ss+5)

    plt.axhline(y=0,color='k',zorder=-1,linewidth=0.5)
    plt.axvline(x=0,color='k',zorder=-1,linewidth=0.5)
    plt.xlim(J2i-lp,J2f+lp)
    plt.ylim(J3i-lp,J3f+lp)
    plt.xticks(fontsize=ss)
    plt.yticks(fontsize=ss)
    plt.xlabel(r'$J_2$',fontsize=ss)
    if DM == '000' or DM == '005':
        plt.ylabel(r'$J_3$',fontsize=ss)
    ###
    ###
    used_o.sort()
    if 'ferro' in used_o:
        used_o.remove('ferro')
        used_o.insert(0,'ferro')
    legend_lines = []
    used_O = []
    for oo in used_o:
        used_O.append(dic_text_ord[oo])
    for ord_u in used_o:
        if ord_u == 'spiral':
            legend_lines.append(Line2D([], [], color='none', marker='P', markerfacecolor='k'))
        else:
           legend_lines.append(Line2D([], [], color='none', markeredgecolor='none', marker='s', markerfacecolor=legend_names[ord_u]))
    plt.legend(legend_lines,used_O,loc='upper left',fancybox=True,fontsize=ss-3)#,bbox_to_anchor=(1,1))

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
fig.set_size_inches(15,5.5)
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

