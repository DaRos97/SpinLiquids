import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import sys

#inputs:    DM_angle -> in str, J_pts

J1 = 1
J2i = -0.3
J2f = 0.3
J3i = -0.3
J3f = 0.3
J2pts = J3pts = int(sys.argv[2])
dirname = 'ferro_j1/' if J1 == -1 else 'antiferro_j1/'
filename = dirname+'DM1nn_'+sys.argv[1]+'_'+sys.argv[2]+'.npy'

dic_DM = {'000':0,'005':0.05,'104':np.pi/3,'209':2*np.pi/3}
DM_angle = dic_DM[sys.argv[1]]
energies = np.load(filename)

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
J2 = np.linspace(J2i,J2f,J2pts)
J3 = np.linspace(J3i,J3f,J3pts)

fig = plt.figure(figsize=(6,6))
plt.gca().set_aspect('equal')
j3_label = [0,3,6]
j2_label = [6,7,8]
Title = 'J1 = -1 (FM)' if J1 == -1 else 'J1 = 1 (AFM)'
#plt.title(Title)
txt_dm = {'000':r'$0$','005':r'$0.05$','104':r'$\pi/3$','209':r'$2\pi/3$'}

for i in range(J2pts):
    for j in range(J3pts):
        color = Colors[int(energies[i,j,1])]
        #plt.scatter(J2[i],J3[j],color = color,marker = '.')
plt.title('DM = '+txt_dm[sys.argv[1]])
plt.yticks([-0.3,0,0.3],['-0.3','0','0.3'])
plt.xticks([-0.3,0,0.3],['-0.3','0','0.3'])

plt.axhline(y=0,color='k',zorder=-1,linewidth=0.5)
plt.axvline(x=0,color='k',zorder=-1,linewidth=0.5)
plt.xlim(-0.3,0.3)
plt.ylim(-0.3,0.3)

#plt.contourf(J2,J3,energies[0][:,:,1].T,alpha=0.3,levels=np.arange(0,12))
levels = np.unique(energies[:,:,1])
cont = plt.contour(J2,J3,energies[:,:,1].T,levels=levels)#,alpha=0.3,levels=np.arange(0,12))

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

