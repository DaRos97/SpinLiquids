import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import getopt
from matplotlib import cm
from matplotlib.lines import Line2D
import functions as fs

Color = {'15':  ['blue','blue'],          #q=0      -> dodgerblue
         '16': ['red','red'],             #3x3      -> orangered
         '17':  ['magenta','orchid'],           #cb2      -> magenta
         '18':  ['k','k'],              #oct     -> orange
         '19':  ['silver','silver'],                #-> silver
         '20':  ['limegreen','limegreen'],    #cb1  -> forestgreen
         'labels':  ['k','k']
         }
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "K:",['staggered','uniform','final','only='])
    K = 13
    DM_type = 'uniform'
    final = False
    do_only = False
    only = '15'
except:
    print("Error in inputs")
    exit()
for opt, arg in opts:
    if opt in ['-K']:
        K = int(arg)
    if opt == '--staggered':
        DM_type = 'staggered'
    if opt == '--uniform':
        DM_type = 'uniform'
    if opt == '--final':
        final = True
    if opt == '--only':
        do_only = True
        only = arg
if final:
    dirname = '../../Data/self_consistency/SDM/final_SDM/' 
else: 
    dirname = '../../Data/self_consistency/SDM/'+DM_type+'/'+str(K)+'/' 
plt.rcParams.update({
    "text.usetex": True,
#    "font.family": "Helvetica"
})
#
S_max = 0.5
DM_max = 0.15
S_pts = 30
DM_pts = 30
S_list = np.linspace(0.01,S_max,S_pts,endpoint=True)
DM_list = np.linspace(0,DM_max,DM_pts,endpoint=True)
X,Y = np.meshgrid(DM_list,S_list)
D = np.ndarray((DM_pts,S_pts),dtype='object')
DD_none = D[0,0]
if do_only:
    considered_ans = [only]
else:
    considered_ans = Color.keys()
for filename in os.listdir(dirname):
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    if len(lines) > 0:
        head = lines[0].split(',')
        head[-1] = head[-1][:-1]
        data = lines[1].split(',')
        dm = list(DM_list).index(float(data[2]))        #was 1
        s = list(S_list).index(float(data[1]))          #was 0
    else:
        continue
    ansatz = fs.min_energy(lines,considered_ans)
    if ansatz == 0:
        continue
    D[dm,s] = ansatz

##########
s = 90
pts = len(os.listdir(dirname))
fig = plt.figure(figsize=(16,8))
#plt.gca().set_aspect('equal')
for i in range(DM_pts):
    for j in range(S_pts):
        if D[i,j] == DD_none:
            c = 'gray'
            m = 'P'
            plt.scatter(DM_list[i],S_list[j],color=c,marker=m)
            continue
        if D[i,j][-1] == 'L':
            c = Color[D[i,j][:2]][1]
            m = '*'
        elif D[i,j][-1] == 'O':
            c = Color[D[i,j][:2]][0]
            m = 'o'
        elif D[i,j][-1] == 'C':
            c = 'magenta'
            m = 'P'
        else:
            c = Color[D[i,j][:2]][0]
            m = '^'
#        if c == 'k' and m == '*':
#            print(D[i,j],i,j)
#            input()
#        plt.scatter(DM_list[i],S_list[j],color=c,marker=m,s=s)
plt.ylim(0.01,0.50)
plt.xlim(0.0,0.15)
ggg = 20
#Legenda
plt.xticks([0,0.03,0.06,0.09,0.12,0.15], [r'$0$',r'$0.03$',r'$0.06$',r'$0.09$',r'$0.12$',r'$0.15$'],size = ggg)
plt.yticks(size=ggg)
list_leg = []
for col in ['16','19','20']:
    if col == 'labels':
        continue
    list_leg.append(col)
    #list_leg.append(col+' SL')
#ist_leg.append('LRO')
#ist_leg.append('SL')
legend_lines = []
for col in ['red','gray','forestgreen','k']:
    if col == 'k':
        legend_lines.append(Line2D([], [], color="w", marker='o', markerfacecolor=col))
        legend_lines.append(Line2D([], [], color="w", marker='*', markerfacecolor=col,markersize=10))
        continue
    legend_lines.append(Line2D([], [], color="w", marker='o', markerfacecolor=col))
    #legend_lines.append(Line2D([], [], color="w", marker='o', markerfacecolor=col[1]))

#plt.legend(legend_lines,list_leg,loc='upper right',bbox_to_anchor=(1,1),fancybox=True)
#
plt.xlabel(r'$\phi$',size=ggg+5)
plt.ylabel(r'$S$',size = ggg+5,rotation='horizontal')
#plt.title(r'$S-\phi$  phase diagram',size = 20)
fig.set_size_inches(6,5)
plt.show()
