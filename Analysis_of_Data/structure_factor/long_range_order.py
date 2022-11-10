import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from matplotlib import cm
import getopt
import sys

list_ans = ['3x3','q0','cb1','cb2','oct']
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
#import data from file
filename = '../../Data/S'+txt_S+'/phi'+DM+'/'+pts+'/'+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
data = fs.import_data(ans,filename)
#Arguments
DM_angle = DM_list[DM]
args = (1,J2,J3,ans,DM_angle)
#compute the Ks of the minimum band
Nx = 13     #points for looking at minima in BZ
Ny = 13
K_,is_LRO = fs.find_minima(data,args,Nx,Ny)
if not is_LRO:
    print("Not LRO, there is a gap now")
    exit()
#Compute the M in those K and extract the relative columns
V,degenerate = fs.get_V(K_,data,args)
#construct the spin matrix for each sublattice and compute the coordinates
#of the spin at each lattice site
UC = 12          #even
m = 6
S = np.zeros((3,6,UC,UC//2)) #3 spin components, 6 sites in UC, ij coordinates of UC
#Pauli matrices
sigma = np.zeros((3,2,2),dtype = complex)
sigma[0] = np.array([[0,1],[1,0]])
sigma[1] = np.array([[0,-1j],[1j,0]])
sigma[2] = np.array([[1,0],[0,-1]])
a1 = np.array([1,0])
a2 = np.array([-1,np.sqrt(3)])
if degenerate:
    if len(K_) > 1:
        print("Not supported multi-gap closing points with degeneracy")
        exit()
    K_.append(K_[0])
k1 = K_[0]
k2 = K_[-1]
v1 = V[0]/np.linalg.norm(V[0])
v2 = V[-1]/np.linalg.norm(V[-1])
#constants of modulo 1 (?) whic give the orientation of the condesate
c1 = (1+1j)/np.sqrt(2)
c1_ = np.conjugate(c1)
c2 = (1+1j)/np.sqrt(2)
c2_ = np.conjugate(c2)
c = [1j,1j,1j,1j]
f = np.sqrt(3)/4
d = np.array([  [1/2,-1/4,1/4,0  ,-3/4,-1/4],
                [0  ,f   ,f  ,2*f,3*f ,3*f ]])
r = np.zeros((2,m,UC,UC))
for i in range(UC):
    for j in range(UC//2):
        R = i*a1 + j*a2
        for s in range(m):
            r_ = R# + d[:,s]
            r[:,s,i,j] = R + d[:,s]
            cond = np.zeros(2,dtype=complex)
            for xx in range(len(K_)):
                cond[0] += c[xx]*V[xx][s]/np.linalg.norm(V[xx])*np.exp(1j*np.dot(K_[xx],r_))
                cond[1] += np.conjugate(c[xx])*np.conjugate(V[xx][s+m])/np.linalg.norm(V[xx])*np.exp(-1j*np.dot(K_[xx],r_))
            #cond[0] = c1*v1[s]*np.exp(1j*np.dot(k1,r_)) + c2*v2[s]*np.exp(1j*np.dot(k2,r_))
            #cond[1] = c2_*np.conjugate(v2[m+s])*np.exp(-1j*np.dot(k2,r_)) + c1_*np.conjugate(v1[m+s])*np.exp(-1j*np.dot(k1,r_))
            for x in range(3):
                S[x,s,i,j] = np.real(1/2*np.dot(np.conjugate(cond.T),np.einsum('ij,j->i',sigma[x],cond)))
            S[:,s,i,j] /= np.linalg.norm(S[:,s,i,j])
#Plotting spin values
#plt.figure()
#plt.subplot(2,2,1)
#plt.title("S_x")
#plt.scatter(r[0].ravel(),r[1].ravel(),c = S[0].ravel(), cmap = cm.plasma)
#plt.colorbar()
#plt.subplot(2,2,2)
#plt.title("S_y")
#plt.scatter(r[0].ravel(),r[1].ravel(),c = S[1].ravel(), cmap = cm.plasma)
#plt.colorbar()
#plt.subplot(2,2,3)
#plt.title("S_z")
#plt.scatter(r[0].ravel(),r[1].ravel(),c = S[2].ravel(), cmap = cm.plasma)
#plt.colorbar()
#plt.show()
print("Chirality single triangles: ")
for i in range(2):
    for j in range(1):
        ch1 = np.dot(S[:,1,i,j],np.cross(S[:,2,i,j],S[:,3,i,j]))
        ch2 = np.dot(S[:,4,i,j],np.cross(S[:,5,i,j],S[:,0,i,j+1]))
        print(ch1,'\t',ch2)
savenameS = "SpinOrientations/S_"+ans+'_'+DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
np.save(savenameS,S)
print("Spins computed, now compute the spin structure factor")
#
#Now compute the SSF
UC = 6
Kx = 17     #point to compute the SSF in the EBZ
Ky = 17
kxg = np.linspace(-8*np.pi/3,8*np.pi/3,Kx)
kyg = np.linspace(-4*np.pi/np.sqrt(3),4*np.pi/np.sqrt(3),Ky)
K = np.zeros((2,Kx,Ky))
SFzz = np.zeros((Kx,Ky))
SFxy = np.zeros((Kx,Ky))
for i in range(Kx):
    for j in range(Ky):
        K[:,i,j] = np.array([kxg[i],kyg[j]])
        if not fs.EBZ(K[:,i,j]):
            SFxy[i,j], SFzz[i,j] = ('nan','nan')
            continue
        SFxy[i,j], SFzz[i,j] = fs.SpinStructureFactor(K[:,i,j],S,UC)

savenameSSFzz = "LRO_SSF/SSFzz_"+ans+'_'+DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
savenameSSFxy = "LRO_SSF/SSFxy_"+ans+'_'+DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
np.save(savenameSSFzz,SFzz)
np.save(savenameSSFxy,SFxy)
print("Finished")
