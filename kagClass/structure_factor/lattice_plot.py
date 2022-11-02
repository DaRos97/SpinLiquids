import numpy as np
import matplotlib.pyplot as plt
import functions as fs

L = fs.s3x3_lattice((0,0))

a = np.array([0,1/2,0])
b = np.array([-np.sqrt(3)/4,-1/4,0])
c = np.array([np.sqrt(3)/4,-1/4,0])
dx = [0,1/2,1/4]
dy = [0,0,np.sqrt(3)/4]

plt.figure()
m = 4
a1 = (1,0)
a2 = (-1/2,np.sqrt(3)/2)
for i in range(m):
    for j in range(m):
        for l in range(3):
            x = a1[0]*i+a2[0]*j + dx[l]
            y = a2[1]*j + dy[l]
            if all(L[i,j,l] == a):
                col = 'r'
            elif all(L[i,j,l] == b):
                col = 'b'
            elif all(L[i,j,l] == c):
                col = 'g'
            plt.scatter(x,y,color=col)


plt.show()
