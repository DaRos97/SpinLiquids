import numpy as np
import functions as fs
import matplotlib.pyplot as plt

pts = 100
DM_list = np.linspace(0,np.pi/6,pts)

L_3x3 = fs.s3x3_lattice((0,0))
L_q0 = fs.q0_lattice((0,0))
L_ferro = fs.ferro_lattice((0,0))

en_3x3_TMD = [fs.energy_2(L_3x3,6,(1,0,0),(DM,0,0)) for DM in DM_list]
en_q0_TMD = [fs.energy_2(L_q0,6,(1,0,0),(DM,0,0)) for DM in DM_list]
en_ferro_TMD = [fs.energy_2(L_ferro,6,(1,0,0),(DM,0,0)) for DM in DM_list]
en_3x3_Stag = [fs.energy_Staggered_DM(L_3x3,6,(1,0,0),(DM,0,0)) for DM in DM_list]
en_q0_Stag = [fs.energy_Staggered_DM(L_q0,6,(1,0,0),(DM,0,0)) for DM in DM_list]
en_ferro_Stag = [fs.energy_Staggered_DM(L_ferro,6,(1,0,0),(DM,0,0)) for DM in DM_list]

plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
plt.plot(DM_list,en_3x3_TMD,label='3x3_TMD')
plt.plot(DM_list,en_q0_TMD,label='q0_TMD')
plt.plot(DM_list,en_ferro_TMD,label='ferro_TMD')

for i in range(0):
    plt.vlines(np.pi/3*i,ymin=min(en_3x3_TMD+en_q0_TMD+en_ferro_TMD),ymax=max(en_3x3_TMD+en_q0_TMD+en_ferro_TMD),color='k')
plt.legend()
plt.subplot(1,2,2)
plt.plot(DM_list,en_3x3_Stag,label='3x3_Stag')
plt.plot(DM_list,en_q0_Stag,label='q0_Stag')
plt.plot(DM_list,en_ferro_Stag,label='ferro_Stag')

for i in range(0):
    plt.vlines(np.pi/3*i,ymin=min(en_3x3_TMD+en_q0_TMD+en_ferro_TMD),ymax=max(en_3x3_TMD+en_q0_TMD+en_ferro_TMD),color='k')
plt.legend()
plt.show()
