import numpy as np
import sys
import getopt

list_ans = ['3x3','q0','cb1','cb2','oct']
DM_list = {'000':0, '005':0.05,'104':np.pi/3, '209':2*np.pi/3}
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

ch = 0

for i in range(1,12-2):
    for j in range(1,6-2):
        ch += np.dot(S[:,1,i,j],np.cross(S[:,2,i,j],S[:,3,i,j]))      #up
        ch += np.dot(S[:,4,i,j],np.cross(S[:,5,i,j],S[:,0,i,j+1]))      #up
        ch += np.dot(S[:,0,i,j],np.cross(S[:,1,i+1,j],S[:,2,i,j]))      #down
        ch += np.dot(S[:,3,i,j],np.cross(S[:,4,i+1,j],S[:,5,i,j]))      #down

print("Chirality Up+Down: ",ch)






















