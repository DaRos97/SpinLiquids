import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

J1 = 1
J2i = -0.3
J2f = 0.3
J3i = -0.3
J3f = 0.3
dirname = 'ferro_j1/' if J1 == -1 else 'antiferro_j1/'
def name(dm_angle_1nn):
    return 'DM1nn_'+"{:.3f}".format(dm_angle_1nn).replace('.','-')+'.npy'
angles = []
#for n in range(3):
#    angles.append(n*np.pi/3)
angles = [0.05]
energies = []
for ang in angles:
    energies.append(np.load(dirname+name(ang)))

#k -> ferro, r -> s3x3, b -> s3x3_g1, y -> q0, g -> q0_g1, orange -> q0_g2,
#gray -> octa, purple -> octa_g1, m -> octa_g2, c -> cb1, cb2, spiral
Colors = ['k','red','firebrick',
          'y','yellow','khaki',
          'deeppink','fuchsia','violet',
          'lime','limegreen','forestgreen',
          'blue','cyan','cornflowerblue'
          ]
legend_lines = [Line2D([], [], color="w", marker='o', markerfacecolor="k"),     #ferro
                Line2D([], [], color="w", marker='o', markerfacecolor="red"),     #3x3
                Line2D([], [], color="w", marker='o', markerfacecolor="firebrick"),
                Line2D([], [], color="w", marker='o', markerfacecolor="y"),     #q0
                Line2D([], [], color="w", marker='o', markerfacecolor="yellow"),
                Line2D([], [], color="w", marker='o', markerfacecolor="khaki"),
                #Line2D([], [], color="w", marker='o', markerfacecolor="deeppink"),  #octa
                #Line2D([], [], color="w", marker='o', markerfacecolor="fuchsia"),
                #Line2D([], [], color="w", marker='o', markerfacecolor="violet"),
                Line2D([], [], color="w", marker='o', markerfacecolor="lime"),     #cb1
                Line2D([], [], color="w", marker='o', markerfacecolor="limegreen"),
                Line2D([], [], color="w", marker='o', markerfacecolor="forestgreen"),
                #Line2D([], [], color="w", marker='o', markerfacecolor="blue"),     #cb2
                #Line2D([], [], color="w", marker='o', markerfacecolor="cyan"),
                #Line2D([], [], color="w", marker='o', markerfacecolor="cornflowerblue"),
                ]
legend_names = ['ferro','3x3','3x3_g1',
                'q=0','q=0_g1','q=0_g2',
                #'octa','octa_g1','octa_g2',
                'cb1','cb1_g1','cb1_g2',
                #'cb2','cb2_g1','cb2_g2'
                ]
J2pts = len(energies[0][:,0,0])
J3pts = len(energies[0][0,:,0])
J2 = np.linspace(J2i,J2f,J2pts)
J3 = np.linspace(J3i,J3f,J3pts)

fig = plt.figure(figsize=(3,3))
j3_label = [0,3,6]
j2_label = [6,7,8]
Title = 'J1 = -1 (FM)' if J1 == -1 else 'J1 = 1 (AFM)'
#plt.title(Title)
txt_dm = ['0','\pi/3','2\pi/3']
for pd in range(len(angles)):
    plt.subplot(1,1,pd+1)
    plt.gca().set_aspect('equal')
#    ax = fig.add_subplot(3,3,pd+1)
    for i in range(J2pts):
        for j in range(J3pts):
            color = Colors[int(energies[pd][i,j,1])]
            plt.scatter(J2[i],J3[j],color = color)
            #ax.scatter(J2[i],J3[j],color = color)
    #if pd ==5:
    #    ax.legend(legend_lines,legend_names,loc='center left',bbox_to_anchor=(1,0.3))
    plt.title('DM = '+txt_dm[pd])
    #ax.set_title('DM_ang(1nn) = '+'{:.3f}'.format(angles[pd]))
    #ax.grid(True)
    #if pd in j3_label:
        #ax.set_ylabel('J3')
    #if pd in j2_label:
        #ax.set_xlabel('J2')
    if pd == 0:
        plt.yticks([-0.3,0,0.3],['-0.3','0','0.3'])
    else:
        plt.yticks([])
    plt.xticks([-0.3,0,0.3],['-0.3','0','0.3'])
    if pd == 2:
        plt.legend(legend_lines,legend_names,loc='center left',bbox_to_anchor=(1,0.3))

plt.gca().set_aspect('equal')
plt.axhline(y=0,color='k',zorder=-1)
plt.axvline(x=0,color='k',zorder=-1)
#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()
plt.show()

