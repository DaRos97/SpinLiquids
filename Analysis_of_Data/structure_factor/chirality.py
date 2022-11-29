import numpy as np
import sys
import getopt

list_ans = ['3x3','q0','cb1','cb2','oct']
DM_list = {'000':0, '006':np.pi/48, '013':2*np.pi/48, '019':3*np.pi/48, '026':4*np.pi/48, '032':5*np.pi/48, '039':6*np.pi/48,'104':np.pi/3, '209':2*np.pi/3}
S_dic = {'50': 0.5, '36':(np.sqrt(3)+1)/2, '34':0.34, '30':0.3, '20':0.2}
#input arguments
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "S:", ['j2=','j3=','DM=','ans=','kpts='])
    txt_S = '50'
    J2 = 0
    J3 = 0
    DM = '000'
    ans = '3x3'
    pts = '37'
except:
    print("Error")
for opt, arg in opts:
    if opt in ['-S']:
        txt_S = arg
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
S = S_dic[txt_S]
DM_angle = DM_list[DM]

########################################
########################################
print("Using arguments: ans-> ",ans," j2,j3 = ",J2,",",J3," Dm angle = ",DM," spin S = ",S)
#Import Spin orientations
savenameS = "SpinOrientations/S_"+ans+'_'+DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
S = np.load(savenameS)

ch_u = 0
ch_d = 0
ch_r = 0
ch_l = 0

dic_coord = {'cb1': {'000':[2,1,2,1,2,1], '209':[6,3,6,3,6,3]},
             'q0': {'000':[1,1,1,1,1,1], '209':[3,2,3,2,1,1]}
             }
xu,yu,xd,yd,xr,yr = dic_coord[ans][DM]

for i in range(xu):
    for j in range(yu):
        ch_u += np.dot(S[:,1,i,j],np.cross(S[:,2,i,j],S[:,3,i,j]))      #up
        if j < yu-1:
            if ans == 'q0' and DM == '000':
                continue
            ch_u += np.dot(S[:,4,i,j],np.cross(S[:,5,i,j],S[:,0,i,j+1]))      #up
for i in range(xd):
    for j in range(yd):
        ch_d += np.dot(S[:,0,i,j],np.cross(S[:,1,i+1,j],S[:,2,i,j]))      #down
        if j < yd-1:
            ch_d += np.dot(S[:,3,i,j],np.cross(S[:,4,i+1,j],S[:,5,i,j]))      #down
for i in range(xr):
    for j in range(yr):
        ch_r += np.dot(S[:,2,i,j],np.cross(S[:,3,i+1,j],S[:,4,i+1,j]))      #down
for i in range(xr):
    for j in range(yr):
        ch_l += np.dot(S[:,1,i+1,j],np.cross(S[:,5,i+1,j],S[:,3,i,j]))      #down

print("Chirality Up: ",ch_u)
print("Chirality Down: ",ch_d)
print("Chirality Right: ",ch_r)
print("Chirality Left: ",ch_l)






















