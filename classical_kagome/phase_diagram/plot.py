import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import sys

#inputs:    DM_angle -> in str, J_pts

lim = float(sys.argv[3])
J1 = 1
J2i = -lim
J2f = lim
J3i = -lim
J3f = lim
J2pts = J3pts = int(sys.argv[2])
dirname = 'data/'
filename = dirname+'J2_'+str(J2i)+'--'+str(J2f)+'__J3_'+str(J3i)+'--'+str(J3f)+'__DM_'+sys.argv[1]+'__Pts_'+sys.argv[2]+'.npy'

dic_DM = {'000':0,'005':0.05,'104':np.pi/3,'209':2*np.pi/3}
DM_angle = dic_DM[sys.argv[1]]
energies = np.load(filename)

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
legend_lines = []
for col in legend_names.values():
    legend_lines.append(Line2D([], [], color="w", marker='o', markerfacecolor=col))
#
J2 = np.linspace(J2i,J2f,J2pts)
J3 = np.linspace(J3i,J3f,J3pts)
orders = ['ferro', '3x3', '3x3_g1', 'q0', 'q0_g1', 'q0_g2', 'octa', 'octa_g1', 'octa_g2', 'cb1', 'cb1_g1', 'cb1_g2', 'cb2', 'cb2_g1', 'cb2_g2', 'spiral']
fig = plt.figure(figsize=(6,6))
plt.gca().set_aspect('equal')
j3_label = [0,3,6]
j2_label = [6,7,8]
Title = 'J1 = -1 (FM)' if J1 == -1 else 'J1 = 1 (AFM)'
#plt.title(Title)
txt_dm = {'000':r'$0$','005':r'$0.05$','104':r'$\pi/3$','209':r'$2\pi/3$'}

for i in range(J2pts):
    for j in range(J3pts):
        color = legend_names[orders[int(energies[i,j])]]
        plt.scatter(J2[i],J3[j],color = color,marker = 'o', s=100)
plt.title('DM = '+txt_dm[sys.argv[1]])
#plt.yticks([-0.3,0,0.3],['-0.3','0','0.3'])
#plt.xticks([-0.3,0,0.3],['-0.3','0','0.3'])

plt.axhline(y=0,color='k',zorder=-1,linewidth=0.5)
plt.axvline(x=0,color='k',zorder=-1,linewidth=0.5)
plt.xlim(J2i,J2f)
plt.ylim(J3i,J3f)

plt.xlabel(r'$J_2$')
plt.ylabel(r'$J_3$')
plt.legend(legend_lines,legend_names.keys(),loc='upper left',fancybox=True)#,bbox_to_anchor=(1,1))
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

