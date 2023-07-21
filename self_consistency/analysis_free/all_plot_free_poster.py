import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import functions as fs
import getopt
from matplotlib import cm
from matplotlib.lines import Line2D

Color = {'15':  ['blue','aqua','dodgerblue'],          #q=0      -> dodgerblue
         '16': ['red','orange','orangered'],             #3x3      -> orangered
         '17':  ['pink','pink','gray'],           #cb2      -> magenta
         '18':  ['k','k','gray'],              #oct     -> orange
         '19':  ['gray','silver','magenta'],                #-> silver
         '20':  ['forestgreen','lime','limegreen'],    #cb1  -> forestgreen
         'labels':  ['k','k','k']
         }
marker = {  '15': ['s','D'],
            '16': ['s','D'],
            '17': ['s','D'],
            '18': ['s','D'],
            '19': ['s','X'],
            '20': ['s','X']
            }
list_leg = [r'$\mathbf{Q}=0$',
            r'$15$',
            r'$\sqrt{3}\times\sqrt{3}$',
            r'$16$',
            r'$19$',
            r'$\textit{cuboc-1}$',
            r'$20$',
            ]
legend_lines = [Line2D([], [], color="none", marker=marker['15'][0], markerfacecolor=Color['15'][0], markeredgecolor='none'),
                Line2D([], [], color="none", marker=marker['15'][0], markerfacecolor=Color['15'][1], markeredgecolor='none'),
                Line2D([], [], color="none", marker=marker['16'][0], markerfacecolor=Color['16'][0], markeredgecolor='none'),
                Line2D([], [], color="none", marker=marker['16'][0], markerfacecolor=Color['16'][1], markeredgecolor='none'),
                Line2D([], [], color="none", marker=marker['19'][0], markerfacecolor=Color['19'][1], markeredgecolor='none'),
                Line2D([], [], color="none", marker=marker['20'][0], markerfacecolor=Color['20'][0], markeredgecolor='none'),
                Line2D([], [], color="none", marker=marker['20'][0], markerfacecolor=Color['20'][1], markeredgecolor='none'),
                ]

list_S = ['50','36','30','20']
list_DM = ['005']

#
phi_label = {'000':0, '005':0.05}
S_label = {'50':r'$S=0.5$', '36':r'$S=(\sqrt{3}-1)/2$', '30':r'$S=0.3$', '20':r'$S=0.2$'}
DM_label = {'000':r'$\phi=0$', '005':r'$\phi=0.05$'}
plt.rcParams.update({
    "text.usetex": True,
#    "font.family": "Helvetica"
})
fig = plt.figure(figsize=(9,30))
for iii,txt_S in enumerate(list_S):
    for phi_t in list_DM:
        if phi_t == '000':
            fit_classical = np.load("../../classical_kagome/phase_diagram/fit_"+phi_t+"_101.npy")
        else:
            fit_classical = np.load("../../classical_kagome/phase_diagram/fit_010_51.npy")
        plt.subplot(4,1,iii+1)
        phi = phi_label[phi_t]
        dirname = '../../Data/self_consistency/S'+txt_S+'/phi'+phi_t+'/final/' 
        #dirname = '../../Data/self_consistency/440_small_S50/phi'+phi_t+'/13/' 
        title = S_label[txt_S] #+ ', ' + DM_label[phi_t]
        #
        Grid = 31#9
        D = np.ndarray((Grid,Grid),dtype='object')
        DD_none = D[0,0]
        Ji = -0.3
        Jf = -Ji
        J2 = np.linspace(Ji,Jf,Grid)
        J3 = np.linspace(Ji,Jf,Grid)
        for filename in os.listdir(dirname):
            with open(dirname+filename, 'r') as f:
                lines = f.readlines()
            if len(lines) == 0:
                continue
            head = lines[0].split(',')
            head[-1] = head[-1][:-1]
            data = lines[1].split(',')
            j2 = float(data[head.index('J2')])
            j3 = float(data[head.index('J3')])
            i2 = list(J2).index(j2) 
            i3 = list(J3).index(j3) 
            minE = 10
            N = (len(lines)-1)//2 + 1
            i_ = 0
            for i in range(N):
                head = lines[2*i].split(',')
                head[-1] = head[-1][:-1]
                data = lines[2*i+1].split(',')
                tempE = float(data[head.index('Energy')])
                if tempE < minE:
                    minE = tempE
                    i_ = i
            head = lines[2*i_].split(',')
            head[-1] = head[-1][:-1]
            data = lines[2*i_+1].split(',')
            sol = data[0] + data[1]
            D[i2,i3] = sol
        dirname = '../../Data/self_consistency/S'+txt_S+'/phi'+phi_t+'/25/'     ###va bene?
        for i in range(Grid):
            for j in range(Grid):
                if D[i,j] != DD_none:
                    continue
                j2 =  J2[i]
                j3 =  J3[j]
                csvname = 'J2_J3=('+'{:5.4f}'.format(j2).replace('.','')+'_'+'{:5.4f}'.format(j3).replace('.','')+').csv'
                filename = dirname+csvname
                with open(filename,'r') as f:
                    lines = f.readlines()
                head = lines[0].split(',')
                head[-1] = head[-1][:-1]
                data = lines[1].split(',')
                sol = data[0] + 'LRO'
                D[i,j] = sol


        ##########
        #plt.subplot(2,2,1)
        plt.gca().set_aspect('equal')
        for i in range(Grid):
            for j in range(Grid):
                if D[i,j] == DD_none:
                    #if i == 7 and j == 30:
                    #    c = 'forestgreen';  m = 'P'
                    #else:
                    #    c = 'b';            m = 's'
                    #plt.scatter(J2[i],J3[j],color=c,marker=m,s=100)
                    continue
                ans = D[i,j][:2]
                order = 0 if D[i,j][2:] == 'LRO' else 1
                c = Color[ans][order]
                m = marker[ans][order]
                plt.scatter(J2[i],J3[j],color=c,marker=m,s=10)
        plt.axhline(y=0,color='k')#,zorder=-1)
        plt.axvline(x=0,color='k')#,zorder=-1)
        if 0:#phi_t == '000':
            for i in range(3):
                x = np.linspace(fit_classical[i][2],fit_classical[i][3],100)
                plt.plot(x,fit_classical[i][0]*x+fit_classical[i][1],'k-',alpha = 0.5)#,zorder=0)
        plt.xlim(-0.3,0.3)
        plt.ylim(-0.3,0.3)
        ss = 18
        if iii == 3:
            #plt.legend(legend_lines,list_leg,loc='upper left',fancybox=True, bbox_to_anchor=(1.05,1.035),fontsize=ss-8)
            plt.xlabel(r'$J_2$',size=ss+5)
            plt.xticks([-0.3,-0.15,0,0.15,0.3],[r'$-0.3$','$-0.15$',r'$0$',r'$0.15$',r'$0.3$'],fontsize=ss)
        else:
            plt.xticks([])

        plt.ylabel(r'$J_3$',size=ss+5,rotation = 'horizontal')
        plt.yticks([-0.3,-0.15,0,0.15,0.3],[r'$-0.3$','$-0.15$',r'$0$',r'$0.15$',r'$0.3$'],fontsize=ss)

        #plt.xticks([-0.3,-0.15,0,0.15,0.3],[r'$-0.3$','$-0.15$',r'$0$',r'$0.15$',r'$0.3$'],fontsize=ss)
        plt.title(title,size=ss)


fig.set_size_inches(5,15)
if 1:
    #filename = '../../../../Figs_SB_paper/SB_'+list_DM[0]+'_new'+'.svg'
    filename = 'SB_'+list_DM[0]+'_new'+'.svg'
    plt.savefig(filename,bbox_inches='tight')
else:
    plt.show()





