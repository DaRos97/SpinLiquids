import numpy as np
import matplotlib.pyplot as plt

S = 0.5
text_ans = ['(0,0)','(pi,0)','(pi,pi)','(0,pi)']
colors = ['b*-','r*-','m*-','g*-']
fig = plt.figure(figsize=(12,8))
dirname = 'DataTest/'

#ax1 = plt.subplot(1,2,1)
for i in range(2):
    text = dirname + text_ans[i] + 'E_gs-' + str(S).replace('.',',') + '.npy'
    data = np.load(text)
    plt.plot(data[0],data[1],colors[i],label=text_ans[i])
plt.title('S = '+str(S))
plt.xlabel(r"$\phi$ of DM")
plt.ylabel(r"$\frac{E}{S(S+1)}$")
plt.legend()

plt.show()
