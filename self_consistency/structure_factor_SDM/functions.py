import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import cm

m_ = [3,6]
orders = [('15','16','19'),('17','18','20')]
cutoff_solution = 1e-3
def import_data(ans,filename):
    P = []
    done = False
    with open(filename, 'r') as f:
        lines = f.readlines()
    N = (len(lines)-1)//2 + 1
    for i in range(N):
        head = lines[i*2].split(',')
        head[-1] = head[-1][:-1]
        data = lines[i*2+1].split(',')
        if data[0] == ans:
            print('Gap value found: ',data[head.index('Gap')])
            for p in range(head.index('L'),len(data)):
                P.append(float(data[p]))
            done = True
            break
    return P
####
def find_minima(args,Nx,Ny):
    S,DM,data,ans = args
    p1 = 0 if ans in ['15','16','19'] else 1
    m = m_[p1]
    J = np.zeros((2*m,2*m))
    for i in range(m):
        J[i,i] = -1
        J[i+m,i+m] = 1
    nxg = np.linspace(-1/2,1/2,Nx)
    nyg = np.linspace(-1/2,1/2,Ny)
    K = np.zeros((2,Nx,Ny))
    en = np.zeros((Nx,Ny))
    factor = 2 if m == 3 else 1
    for i in range(Nx):
        for j in range(Ny):
            K[:,i,j] = np.array([nxg[i]*2*np.pi,(nxg[i]+factor*nyg[j])*2*np.pi/np.sqrt(3)])
            N = Nk(K[:,i,j],DM,data,ans)
            Ch = LA.cholesky(N)
            temp = np.dot(np.dot(Ch,J),np.conjugate(Ch.T))
            en[i,j] = LA.eigvalsh(temp)[m]
    ########################Now find the minimum (can be more than one)
    ######################
    ###################### RECURSIVE METHOD
    ######################
    ind = np.argmin(en)
    k1 = K[:,ind//Nx,ind%Ny]
    k_list = [k1]
    e0 = en[ind//Nx,ind%Ny]
    en[ind//Nx,ind%Ny] += 10
    restore_list = [ind]
    cont = True
    while cont:
        add = True
        ind = np.argmin(en)
        k1 = K[:,ind//Nx,ind%Ny]
        if np.abs(en[ind//Nx,ind%Ny]-e0) > e0/100:
            cont = False
        #check if a close one or reciprocal-lattice equivalent was already added
        for k in k_list:
            if (np.abs(np.abs(k1[1]-k[1])-2*np.pi/np.sqrt(3)) < 1e-3 or #opposite side of BZ in Y
                LA.norm(k1-k) < 4*np.pi/np.sqrt(3)/(Ny-2) ):
                en[ind//Nx,ind%Ny] += 10
                restore_list.append(ind)
                add = False
                break
        if add and cont:
            k_list.append(k1)
    for ind in restore_list:
        en[ind//Nx,ind%Ny] -= 10
    #plot points for additional checking
    plt.figure()
    plt.scatter(K[0],K[1],c=en,cmap = cm.plasma)
    plt.colorbar()
    for k in k_list:
#        plt.scatter(k[0],k[1],c='g',marker='*')
        print('new: ',k)
#    plt.xlabel(r'$K_x$',size=15)
#    plt.ylabel(r'$K_y$',size=15)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.show()
    ok = input("Is it ok?[Y/n] ([1] for keeping only first value found,[2] for just first 2 values (NOT good)\t")
    if ok == 'n':
        exit()
    #Condition for correct closure of the gap (should be the case)
    LRO = True if en[restore_list[0]//Nx,restore_list[0]%Ny] < 0.05 else False
    return k_list, LRO
####
def get_V(K_,args):
    S,DM,data,ans = args
    p1 = int(data[0])
    m = m_[p1]
    J = np.zeros((2*m,2*m))
    for i in range(m):
        J[i,i] = -1
        J[i+m,i+m] = 1
    V = []
    degenerate = False
    for K in K_:
        N = Nk(K,DM,data,ans)
        Ch = LA.cholesky(N) #upper triangular
        w,U = LA.eigh(np.dot(np.dot(Ch,J),np.conjugate(Ch.T)))
        w_ = np.diag(np.sqrt(np.einsum('ij,j->i',J,w)))
        Mk = np.dot(np.dot(LA.inv(Ch),U),w_)
        V.append(Mk[:,m-1])
        if np.abs(w[m]-w[m+1]) < 1e-3:          #degeneracy case -> same K
            print("degenerate K point")
            V.append(Mk[:,m+1])
            degenerate = True
    return V,degenerate
####

def M(K,P,args):
    m = 6
    N = Nk(K,P,args)
    Ch = LA.cholesky(N) #upper triangular
    w,U = LA.eigh(np.dot(np.dot(Ch,J),np.conjugate(Ch.T)))
    w = np.diag(np.sqrt(np.einsum('ij,j->i',J,w)))
    Mk = np.dot(np.dot(LA.inv(Ch),U),w)
    U,X,V,Y = split(Mk,m,m) 
    return U,X,V,Y
####
def split(array, nrows, ncols):
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))
#extended brillouin zone
def EBZ(K):
    x = K[0]
    y = K[1]
    a = np.sqrt(3)
    b = np.pi*8/np.sqrt(3)
    if x < -4*np.pi/3 and (y < -a*x-b or y > a*x+b):
        return False
    if x > 4*np.pi/3 and (y < a*x-b or y > -a*x+b):
        return False
    return True
####
f = np.sqrt(3)/4
def SpinStructureFactor(k,L,UC):
    a1 = np.array([1,0])
    a2 = np.array([-1,np.sqrt(3)])
    d = np.array([  [1/2,-1/4,1/4,0,-3/4,-1/4],
                    [0,f,f,2*f,3*f,3*f]])
    resxy = 0
    resz = 0
    dist = np.zeros(2)
    for i in range(UC):#UC//2,UC//2+1):
        for j in range(UC//2):#UC//2,UC//2+1):
            for l in range(6):
#                Li = L[:,l+j%2*3,i,j//2*(j+1)%2 + (j-1)*j%2]
                Li = L[:,l,i,j]
                ri = i*a1+j*a2+d[:,l]
                for i2 in range(UC):
                    for j2 in range(UC//2):
                        for l2 in range(6):
#                            Lj = L[:,l2+j2%2*3,i2,j2//2*(j2+1)%2 + (j2-1)*j2%2]
                            Lj = L[:,l2,i2,j2]
                            rj = i2*a1+j2*a2+d[:,l2]
                            dist = ri-rj
                            SiSjxy = Li[0]*Lj[0] + Li[1]*Lj[1]#np.dot(Li,L[i2,j2,l2])
                            SiSjz = Li[2]*Lj[2]
                            resxy += np.exp(-1j*np.dot(k,dist))*SiSjxy/2
                            resz += np.exp(-1j*np.dot(k,dist))*SiSjz/2
    return np.real(resxy), np.real(resz)



def Nk(K,DM,data,ans):
    #
    t1 = np.exp(-1j*DM);    t1_ = np.conjugate(t1)
    
    if ans in ['15','16','19']:
        p1 = 0
    else:
        p1 = 1
    if ans in ['15','16','17','18']:
        L,A1,B1,phiB1 = data
        phiB1p = -phiB1
    else:
        L,A1,phiA1p,B1,phiB1 = data
        phiB1p = phiB1
    if ans in ['15','17']:
        phiA1p = 0
    if ans in ['16','18']:
        phiA1p = np.pi
    B1p = B1
    A1p = A1
    J1 = 1/2
    m = m_[p1]
    ################
    a1 = (1,0)
    factor = 2 if m == 3 else 1
    a2 = (-1/factor,np.sqrt(3)/factor)
    a12p = (a1[0]+a2[0],a1[1]+a2[1])
    a12m = (a1[0]-a2[0],a1[1]-a2[1])
    ka1 = np.exp(1j*np.dot(a1,K));   ka1_ = np.conjugate(ka1);
    ka2 = np.exp(1j*np.dot(a2,K));   ka2_ = np.conjugate(ka2);
    ka12p = np.exp(1j*np.dot(a12p,K));   ka12p_ = np.conjugate(ka12p);
    ka12m = np.exp(1j*np.dot(a12m,K));   ka12m_ = np.conjugate(ka12m);
    ################
    N = np.zeros((2*m,2*m), dtype=complex)
    ##################################### B
    b1 = B1*np.exp(1j*phiB1);               b1_ = np.conjugate(b1)
    b1p = B1p*np.exp(1j*phiB1p);             b1p_ = np.conjugate(b1p)
    b1pi = B1p*np.exp(1j*(phiB1p+p1*np.pi)); b1pi_ = np.conjugate(b1pi)
    #
    N[0,1] = J1*b1p_ *ka1  *t1_              
    N[0,2] = J1*b1p        *t1   
    N[1,2] = J1*(b1_       *t1  + b1p_*ka1_*t1_)                                #t1     t1
    N[0,4%m] = J1*b1_  *ka2_ *t1               
    N[0,5%m] = J1*b1   *ka2_ *t1_
    N[3%m,4%m] += J1*b1pi_*ka1  *t1_              
    N[3%m,5%m] += J1*b1p        *t1               
    N[4%m,5%m] = J1*(b1_       *t1  + b1pi_*ka1_*t1_)                               #t1     t1
    ####other half square                                                       #Same ts
    N[m+0,m+1] = J1*b1p  *ka1  *t1_           
    N[m+0,m+2] = J1*b1p_       *t1            
    N[m+1,m+2] = J1*(b1        *t1  + b1p*ka1_*t1_)
    N[m+0,m+4%m] = J1*b1   *ka2_ *t1            
    N[m+0,m+5%m] = J1*b1_  *ka2_ *t1_           
    N[m+3%m,m+4%m] += J1*b1pi *ka1  *t1_           
    N[m+3%m,m+5%m] += J1*b1p_       *t1            
    N[m+4%m,m+5%m] = J1*(b1        *t1  + b1pi*ka1_*t1_)
    ######################################## A
    a1 =    A1
    a1p =   A1p*np.exp(1j*phiA1p)
    a1pi =  A1p*np.exp(1j*(phiA1p+p1*np.pi))
    N[0,m+1] = - J1*a1p *ka1 *t1_           
    N[0,m+2] =   J1*a1p      *t1            
    N[1,m+2] = - J1*(a1      *t1   +a1p*ka1_*t1_)
    N[0,m+4%m] = - J1*a1  *ka2_*t1            
    N[0,m+5%m] =   J1*a1  *ka2_*t1_           
    N[3%m,m+4%m] += - J1*a1pi*ka1 *t1_           
    N[3%m,m+5%m] +=   J1*a1p      *t1            
    N[4%m,m+5%m] = - J1*(a1      *t1   +a1pi*ka1_*t1_)
    #not the diagonal
    N[1,m]   =   J1*a1p *ka1_*t1            
    N[2,m]   = - J1*a1p      *t1_           
    N[2,m+1] =   J1*(a1      *t1_  +a1p*ka1 *t1)
    N[4%m,m]   =   J1*a1  *ka2 *t1_           
    N[5%m,m]   = - J1*a1  *ka2 *t1            
    N[4%m,m+3%m] +=   J1*a1pi*ka1_*t1            
    N[5%m,m+3%m] += - J1*a1p      *t1_           
    N[5%m,m+4%m] =   J1*(a1      *t1_  +a1pi*ka1 *t1)
    ############################################### Terms which are different between m = 3,6
    if m == 6:
        N[1,3%m] = J1*b1         *t1_
        N[2,3%m] = J1*b1_        *t1               
        N[m+1,m+3%m] = J1*b1_        *t1_           
        N[m+2,m+3%m] = J1*b1         *t1            
        N[1,m+3%m] =   J1*a1       *t1_           
        N[2,m+3%m] = - J1*a1       *t1            
        N[3%m,m+1] = - J1*a1       *t1            
        N[3%m,m+2] =   J1*a1       *t1_           
    #################################### HERMITIAN MATRIX
    #N += np.conjugate(N.transpose((1,0,2,3)))
    for i in range(2*m):
        for j in range(i,2*m):
            N[j,i] += np.conjugate(N[i,j])
    #################################### L
    for i in range(2*m):
        N[i,i] += L
    return N


def fd1(x):
    return -np.sqrt(3)*x-4*np.pi/np.sqrt(3)
def fd3(x):
    return np.sqrt(3)*x-4*np.pi/np.sqrt(3)
def fu1(x):
    return np.sqrt(3)*x+4*np.pi/np.sqrt(3)
def fu3(x):
    return -np.sqrt(3)*x+4*np.pi/np.sqrt(3)
def Fd1(x):
    return -np.sqrt(3)*x-8*np.pi/np.sqrt(3)
def Fd3(x):
    return np.sqrt(3)*x-8*np.pi/np.sqrt(3)
def Fu1(x):
    return np.sqrt(3)*x+8*np.pi/np.sqrt(3)
def Fu3(x):
    return -np.sqrt(3)*x+8*np.pi/np.sqrt(3)

X1 = np.linspace(-4*np.pi/3,-2*np.pi/3,1000)
X2 = np.linspace(2*np.pi/3,4*np.pi/3,1000)
X3 = np.linspace(-8*np.pi/3,-4*np.pi/3,1000)
X4 = np.linspace(4*np.pi/3,8*np.pi/3,1000)




#######################################################
#######################################################
#######################################################
#######################################################

def check_list(a,list_dir):
    S = cart2sph(a)
    for key in list_dir.keys():
        S2 = cart2sph(list_dir[key])
        diff = np.abs(S[1]-S2[1]) + np.abs(S[2]-S2[2])
        if diff < 1e-2:
            return False, int(key)
    return True, 0


def cart2sph(vec):
    x = vec[0]; y = vec[1]; z = vec[2]
    r = x**2 + y**2 + z**2
    theta = np.arccos(z)
    phi = np.arctan2(y,x)
    if theta < 1e-3 or np.pi-theta < 1e-3:
        phi = 0
    if np.abs(phi) < 1e-3:
        phi = 0 
    elif phi < 0:
        phi += 2*np.pi
    return r, theta, phi


def cl(va,lt,cmap='Paired',invert=False,margin=0.1,lowcut=0,upcut=1): 
    cmap = plt.cm.get_cmap(cmap)
    ind=list(lt).index(va) 
    if len(lt)>1:
        rt=ind/(len(lt)-1)
        rt*=1-min(1-margin,lowcut+1-upcut)
        rt+=lowcut
    else:
        rt=0
        rt+=lowcut
    rt=rt*(1-2*margin)+margin
    if invert:
        rt=1-rt    
    return cmap(rt)
