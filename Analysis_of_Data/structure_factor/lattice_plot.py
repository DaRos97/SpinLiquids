import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import getopt

list_ans = ['3x3_1','q0_1','cb1','cb2','oct']
DM_list = {'000':0, '006':np.pi/48, '013':2*np.pi/48, '019':3*np.pi/48, '026':4*np.pi/48, '032':5*np.pi/48, '039':6*np.pi/48, '209':2*np.pi/3}
#input arguments
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:", ['j2=','j3=','DM=','ans=','kpts='])
    S = 0.5
    txt_S = '05'
    J2 = 0
    J3 = 0
    DM = '000'
    ans = '3x3_1'
    pts = '13'
except:
    print("Error")
for opt, arg in opts:
    if opt in ['-S']:
        txt_S = arg
        if txt_S not in ['05','03']:
            print('Error in -S argument')
            exit()
        else:
            S = 0.5 if txt_S == '05' else 0.366         #####CHECK
    if opt == '--j2':
        J2 = float(arg)
    if opt == '--j3':
        J3 = float(arg)
    if opt == '--DM':
        DM = arg.replace('.','')
        if DM not in DM_list.keys():
            print('Not computed DM angle')
            exit()
    if opt == '--ans':
        ans = arg 
        if ans not in list_ans:
            print('Error in -ans choice')
            exit()
    if opt == '-kpts':
        pts = arg
print("Using arguments: ans-> ",ans," j2,j3 = ",J2,",",J3," Dm angle = ",DM," spin S = ",S)
file_S = "SpinOrientations/S_"+ans+'_'+DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
S = np.load(file_S)

list_dir = {}
cnt = int(0)
new_S = np.zeros((len(S[0,:,0,0]),len(S[0,0,:,0]),len(S[0,0,0,:])))
plt.figure()
a1 = np.array([1,0])
a2 = np.array([-1,np.sqrt(3)])
f = np.sqrt(3)/4
dUC = np.array([  [1/2,-1/4,1/4,0  ,-3/4,-1/4],
                [0  ,f   ,f  ,2*f,3*f ,3*f ]])
list_col = ['r','b','g','y','k','m','c','yellow','green','lime',
            'limegreen','firebrick','purple','pink']
legend_lines = [Line2D([], [], color="w", marker='o', markerfacecolor="r"),     #ferro
                Line2D([], [], color="w", marker='o', markerfacecolor="b"),     #3x3
                Line2D([], [], color="w", marker='o', markerfacecolor="g"),
                Line2D([], [], color="w", marker='o', markerfacecolor="y"),     #q0
                Line2D([], [], color="w", marker='o', markerfacecolor="k"),
                Line2D([], [], color="w", marker='o', markerfacecolor="m"),
                Line2D([], [], color="w", marker='o', markerfacecolor="c"),  #octa
                Line2D([], [], color="w", marker='o', markerfacecolor="yellow"),
                Line2D([], [], color="w", marker='o', markerfacecolor="green"),
                Line2D([], [], color="w", marker='o', markerfacecolor="lime"),     #cb1
                Line2D([], [], color="w", marker='o', markerfacecolor="limegreen"),
                Line2D([], [], color="w", marker='o', markerfacecolor="firebrick"),
                Line2D([], [], color="w", marker='o', markerfacecolor="purple"),     #cb2
                Line2D([], [], color="w", marker='o', markerfacecolor="pink"),
                ]
for x in range(len(S[0,0,:,0])):
    for y in range(len(S[0,0,0,:])):
        for m in range(len(S[0,:,0,0])):
            a = S[:,m,x,y]
            is_new, sp = fs.check_list(a,list_dir)
            if is_new:
                list_dir[str(cnt)] = a
                sp = cnt
                cnt += 1
            new_S[m,x,y] = sp
            #plot
            X = a1[0] * x + a2[0] * y + dUC[0,m]
            Y = a2[1] * y + dUC[1,m]
            plt.scatter(X,Y,color=list_col[int(new_S[m,x,y])])
a = 1
for i,s in enumerate(list_dir.values()):
    print(a,': ',list_col[i],' --> ',fs.cart2sph(s)[1:])
    a += 1
plt.legend(legend_lines,list_dir.keys(),loc='center left',bbox_to_anchor=(1,0.3))

plt.show()
