import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import cm

Mm = [3,6]
ans_1 = ['15','16','17','18']
ans_2 = ['19','20']
ans_p0 = ['15','16','19']
cutoff_solution = 1e-3
CO_phase = 1e-3
#
def find_ans(head,data):
    ans = data[0]
    head[-1] = head[-1][:-1]
    J2 = float(data[head.index('J2')])
    J3 = float(data[head.index('J3')])
    P = []
    for p in ['p2','p3','p4','p5']:
        if p in head:
            P.append(data[head.index(p)])
    if ans in ans_1:
        relevant_phase = 'phiB1'
    elif ans in ans_2:
        relevant_phase = 'phiA1p'
    else:
        print('WTF is this?')
        exit()
    phase = float(data[head.index(relevant_phase)])
    if np.abs(phase) < CO_phase or np.abs(phase-2*np.pi) < CO_phase:
        phase_ = 'O'
    elif np.abs(phase-np.pi) < CO_phase:
        phase_ = 'P'
    else:
        phase_ = 'Z'
    #
    result = ans + phase_ + str(len(P))
    for p in P:
        result += p
    return result

def import_data(ans,filename):
    P = []
    done = False
    with open(filename, 'r') as f:
        lines = f.readlines()
    N = (len(lines)-1)//2 + 1
    for i in range(N):
        data = lines[i*2+1].split(',')
        head = lines[i*2].split(',')
#        ans2 = find_ans(head,data)
        ans2 = data[0]
        head[-1] = head[-1][:-1]
        if ans2 == ans:
            print('Gap value found: ',data[head.index('Gap')])
            for p in range(head.index('L'),len(data)):
                P.append(float(data[p]))
            return P
####
def find_minima(data,args,Nx,Ny):
    p1 = 1 if args[3] in ['17','18','20'] else 0
    m = Mm[p1]
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
            K[:,i,j] = np.array([nxg[i]*2*np.pi,(nxg[i]+nyg[j]*factor)*2*np.pi/np.sqrt(3)])
            N = Nk(K[:,i,j],data,args)
            Ch = LA.cholesky(N)
            temp = np.dot(np.dot(Ch,J),np.conjugate(Ch.T))
            en[i,j] = LA.eigvalsh(temp)[m]
    ########################Now find the minimum (can be more than one)
    ######################
    ###################### RECURSIVE METHOD
    ######################
    print("new gap found ",np.amin(en.ravel()))
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
    plt.figure(figsize=(12,12))
    plt.gca().set_aspect('equal')
    plt.scatter(K[0],K[1],c=en,cmap = cm.plasma)
    plt.colorbar()
    for k in k_list:
        plt.scatter(k[0],k[1],c='g',marker='*')
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
def get_V(K_,data,args):
    p1 = 1 if args[3] in ['17','18','20'] else 0
    m = Mm[p1]
    J = np.zeros((2*m,2*m))
    for i in range(m):
        J[i,i] = -1
        J[i+m,i+m] = 1
    V = []
    degenerate = False
    for K in K_:
        N = Nk(K,data,args)
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

def M(K,P,args,form):
    p1 = 0 if args[3] in ans_p0 else 1
    m = Mm[p1]
    J = np.zeros((2*m,2*m))
    for i in range(m):
        J[i,i] = -1
        J[i+m,i+m] = 1
    N = Nk(K,P,args)
    Ch = LA.cholesky(N) #upper triangular
    w,U = LA.eigh(np.dot(np.dot(Ch,J),np.conjugate(Ch.T)))
    w = np.diag(np.sqrt(np.einsum('ij,j->i',J,w)))
    Mk = np.dot(np.dot(LA.inv(Ch),U),w)
    return split(Mk,m,m) 
#    if form == 'dag':
#        return split(np.conjugate(Mk.T),m,m) 
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
def BZ(K):
    x = K[0]
    y = K[1]
    a = np.sqrt(3)
    b = np.pi*4/np.sqrt(3)
    if x < -2*np.pi/3 and (y < -a*x-b or y > a*x+b):
        return False
    if x > 2*np.pi/3 and (y < a*x-b or y > -a*x+b):
        return False
    return True
####
f = np.sqrt(3)/4
def SpinStructureFactor(k,L,UC,m):
    factor = 2 if m == 3 else 1
    a1 = np.array([1,0])
    a2 = np.array([-1/factor,np.sqrt(3)/factor])
    d = np.array([  [1/2,-1/4,1/4,0,-3/4,-1/4],
                    [0,f,f,2*f,3*f,3*f]])
    resxy = 0
    resz = 0
    dist = np.zeros(2)
    f2 = 2 if m == 6 else 1
    for i in range(UC):#UC//2,UC//2+1):
        for j in range(UC//f2):#UC//2,UC//2+1):
            for l in range(m):
#                Li = L[:,l+j%2*3,i,j//2*(j+1)%2 + (j-1)*j%2]
                Li = L[:,l,i,j]
                ri = i*a1+j*a2+d[:,l]
                for i2 in range(UC):
                    for j2 in range(UC//f2):
                        for l2 in range(m):
#                            Lj = L[:,l2+j2%2*3,i2,j2//2*(j2+1)%2 + (j2-1)*j2%2]
                            Lj = L[:,l2,i2,j2]
                            rj = i2*a1+j2*a2+d[:,l2]
                            dist = ri-rj
                            SiSjxy = Li[0]*Lj[0] + Li[1]*Lj[1]#np.dot(Li,L[i2,j2,l2])
                            SiSjz = Li[2]*Lj[2]
                            resxy += np.exp(-1j*np.dot(k,dist))*SiSjxy/2
                            resz += np.exp(-1j*np.dot(k,dist))*SiSjz/2
    return np.real(resxy), np.real(resz)
#
def Nk(K,P_,args):
    Tau,S,J,ans,p = args
    p1 = 0 if ans in ans_p0 else 1
    m = Mm[p1]
    factor = 2 if m == 3 else 1
    a1 = (1,0)
    a2 = (-1/factor,np.sqrt(3)/factor)
#    a2 = (-1,np.sqrt(3))
    a12p = (a1[0]+a2[0],a1[1]+a2[1])
    a12m = (a1[0]-a2[0],a1[1]-a2[1])
    ka1 = np.exp(1j*np.dot(a1,K));   ka1_ = np.conjugate(ka1);
    ka2 = np.exp(1j*np.dot(a2,K));   ka2_ = np.conjugate(ka2);
    ka12p = np.exp(1j*np.dot(a12p,K));   ka12p_ = np.conjugate(ka12p);
    ka12m = np.exp(1j*np.dot(a12m,K));   ka12m_ = np.conjugate(ka12m);
    L = P_[0]
    P = P_[1:]
    J1,J2,J3 = J
    t1,t1_,t2,t2_,t3,t3_ = Tau
    J1 /= 2.
    J2 /= 2.
    J3 /= 2.
    func_ans = {'15':ans_15, '16':ans_16,'17':ans_17,'18':ans_18,'19':ans_19,'20':ans_20}
    A1,phiA1p,B1,phiB1,phiB1p,A2,phiA2,A2p,phiA2p,B2,phiB2,B2p,phiB2p,A3,phiA3,B3,phiB3 = func_ans[ans](P,J2,J3,p)
    A1p = A1
    B1p = B1
    ################
    N = np.zeros((2*m,2*m), dtype=complex)
    ##################################### B
    b1 = B1*np.exp(1j*phiB1);               b1_ = np.conjugate(b1)
    b1p = B1p*np.exp(1j*phiB1p);             b1p_ = np.conjugate(b1p)
    b1pi = B1p*np.exp(1j*(phiB1p+p1*np.pi)); b1pi_ = np.conjugate(b1pi)
    b2 = B2*np.exp(1j*phiB2);               b2_ = np.conjugate(b2)
    b2i = B2*np.exp(1j*(phiB2+p1*np.pi));   b2i_ = np.conjugate(b2i)
    b2p = B2p*np.exp(1j*phiB2p);             b2p_ = np.conjugate(b2p)
    b2pi = B2p*np.exp(1j*(phiB2p+p1*np.pi)); b2pi_ = np.conjugate(b2pi)
    b3 = B3*np.exp(1j*phiB3);               b3_ = np.conjugate(b3)
    b3i = B3*np.exp(1j*(phiB3+p1*np.pi));   b3i_ = np.conjugate(b3i)
    #
    N[0,1] = J1*b1p_ *ka1  *t1_              + J2*b2*t2                         #t1
    N[0,2] = J1*b1p        *t1               + J2*b2p_ *ka1*t2                  #t1_
    N[1,2] = J1*(b1_       *t1  + b1p_*ka1_*t1_)                                #t1     t1
    N[0,0] = J3*b3i_ *ka1_ *t3_
    if m == 6:
        N[0,4] = J1*b1_  *ka2_ *t1               + J2*b2pi *ka12m*t2_               #t1
        N[0,5] = J1*b1   *ka2_ *t1_              + J2*b2i_ *ka12p_*t2_              #t1_
        N[1,3] = J1*b1         *t1_              + J2*b2p_ *ka1_*t2                 #t1_
        N[1,5] =                                   J2*(b2  *ka12p_*t2 + b2p*t2_)
        N[2,3] = J1*b1_        *t1               + J2*b2   *ka1*t2                  #t1
        N[2,4] =                                   J2*(b2p_*ka2_*t2 + b2i_*ka1*t2_)
        N[3,4] = J1*b1pi_*ka1  *t1_              + J2*b2*t2                         #t1
        N[3,5] = J1*b1p        *t1               + J2*b2pi_*ka1*t2                  #t1_
        N[4,5] = J1*(b1_       *t1  + b1pi_*ka1_*t1_)                               #t1
        N[3,3] = J3*b3_  *ka1_ *t3_
        N[1,4] = J3*(b3_ *ka2_ *t3_ + b3       *t3)
        N[2,5] = J3*(b3  *ka12p_  *t3  + b3i_*ka1*t3_)
    else:
        N[0,1] += J1*b1_  *ka2_ *t1               + J2*b2pi *ka12m*t2_               #t1
        N[0,2] += J1*b1   *ka2_ *t1_              + J2*b2i_ *ka12p_*t2_              #t1_
        N[1,2] +=                                   J2*(b2  *ka12p_*t2 + b2p*ka2*t2_)
        N[1,1] += J3*b3_  *ka2_ *t3_
        N[2,2] += J3*b3   *ka12p_ *t3

    ####other half square                                                       #Same ts
    N[m+0,m+1] = J1*b1p  *ka1  *t1_           + J2*b2_*t2
    N[m+0,m+2] = J1*b1p_       *t1            + J2*b2p  *ka1*t2
    N[m+1,m+2] = J1*(b1        *t1  + b1p*ka1_*t1_)
    N[m+0,m+0] = J3*b3i *ka1_ *t3_
    if m == 6:
        N[m+0,m+4] = J1*b1   *ka2_ *t1            + J2*b2pi_*ka12m*t2_
        N[m+0,m+5] = J1*b1_  *ka2_ *t1_           + J2*b2i  *ka12p_*t2_
        N[m+1,m+3] = J1*b1_        *t1_           + J2*b2p  *ka1_*t2
        N[m+1,m+5] =                                J2*(b2_ *ka12p_*t2 + b2p_*t2_)
        N[m+2,m+3] = J1*b1         *t1            + J2*b2_  *ka1*t2
        N[m+2,m+4] =                                J2*(b2p *ka2_*t2 + b2i *ka1*t2_)
        N[m+3,m+4] = J1*b1pi *ka1  *t1_           + J2*b2_*t2
        N[m+3,m+5] = J1*b1p_       *t1            + J2*b2pi *ka1*t2
        N[m+4,m+5] = J1*(b1        *t1  + b1pi*ka1_*t1_)
        N[m+3,m+3] = J3*b3  *ka1_ *t3_
        N[m+1,m+4] = J3*(b3 *ka2_ *t3_ + b3_  *t3)
        N[m+2,m+5] = J3*(b3_*ka12p_  *t3  + b3i *ka1*t3_)
    else:
        N[m+0,m+1] += J1*b1  *ka2_ *t1               + J2*b2pi_ *ka12m*t2_               #t1
        N[m+0,m+2] += J1*b1_   *ka2_ *t1_              + J2*b2i *ka12p_*t2_              #t1_
        N[m+1,m+2] +=                                   J2*(b2_  *ka12p_*t2 + b2p_*ka2*t2_)
        N[m+1,m+1] += J3*b3  *ka2_ *t3_
        N[m+2,m+2] += J3*b3_   *ka12p_ *t3

    ######################################## A
    a1 =    A1
    a1p =   A1p*np.exp(1j*phiA1p)
    a1pi =  A1p*np.exp(1j*(phiA1p+p1*np.pi))
    a2 =    A2*np.exp(1j*phiA2)
    a2i =   A2*np.exp(1j*(phiA2+p1*np.pi))
    a2p =   A2p*np.exp(1j*phiA2p)
    a2pi =  A2p*np.exp(1j*(phiA2p+p1*np.pi))
    a3 =    A3*np.exp(1j*phiA3)
    a3i =   A3*np.exp(1j*(phiA3+p1*np.pi))
    N[0,m+1] = - J1*a1p *ka1 *t1_           +J2*a2*t2
    N[0,m+2] =   J1*a1p      *t1            -J2*a2p  *ka1*t2
    N[1,m+2] = - J1*(a1      *t1   +a1p*ka1_*t1_)
    N[0,m+0] = - J3*a3i *ka1_*t3_
    if m == 6:
        N[0,m+4] = - J1*a1  *ka2_*t1            +J2*a2pi *ka12m*t2_
        N[0,m+5] =   J1*a1  *ka2_*t1_           -J2*a2i  *ka12p_*t2_
        N[1,m+3] =   J1*a1       *t1_           -J2*a2p  *ka1_*t2
        N[1,m+5] =                               J2*(a2  *ka12p_*t2  +a2p*t2_)
        N[2,m+3] = - J1*a1       *t1            +J2*a2   *ka1*t2
        N[2,m+4] =                              -J2*(a2p *ka2_*t2  +a2i*ka1*t2_)
        N[3,m+4] = - J1*a1pi*ka1 *t1_           +J2*a2*t2
        N[3,m+5] =   J1*a1p      *t1            -J2*a2pi *ka1*t2
        N[4,m+5] = - J1*(a1      *t1   +a1pi*ka1_*t1_)  
        N[3,m+3] = - J3*a3  *ka1_*t3_
        N[1,m+4] = - J3*(a3 *ka2_*t3_  -a3 *t3)
        N[2,m+5] = - J3*(a3i*ka1*t3_  -a3 *ka12p_ *t3)
    else:
        N[0,m+1] += - J1*a1  *ka2_ *t1               + J2*a2pi *ka12m*t2_               #t1
        N[0,m+2] +=   J1*a1  *ka2_ *t1_              - J2*a2i  *ka12p_*t2_              #t1_
        N[1,m+2] +=                                   J2*(a2  *ka12p_*t2 + a2p*ka2*t2_)
        N[1,m+1] += - J3*a3  *ka2_ *t3_
        N[2,m+2] += J3*a3   *ka12p_ *t3
    #not the diagonal
    N[1,m]   =   J1*a1p *ka1_*t1            -J2*a2*t2_
    N[2,m]   = - J1*a1p      *t1_           +J2*a2p  *ka1_*t2_
    N[2,m+1] =   J1*(a1      *t1_  +a1p*ka1 *t1)
    N[0,m+0] +=  J3*a3i *ka1  *t3
    if m == 6:
        N[4,m]   =   J1*a1  *ka2 *t1_           -J2*a2pi *ka12m_*t2
        N[5,m]   = - J1*a1  *ka2 *t1            +J2*a2i  *ka12p*t2
        N[3,m+1] = - J1*a1       *t1            +J2*a2p  *ka1*t2_               #Second term was ka1_
        N[5,m+1] =                              -J2*(a2  *ka12p*t2_   +a2p*t2)
        N[3,m+2] =   J1*a1       *t1_           -J2*a2   *ka1_*t2_
        N[4,m+2] =                               J2*(a2p *ka2*t2_   +a2i*ka1_*t2)
        N[4,m+3] =   J1*a1pi*ka1_*t1            -J2*a2*t2_
        N[5,m+3] = - J1*a1p      *t1_           +J2*a2pi *ka1_*t2_
        N[5,m+4] =   J1*(a1      *t1_  +a1pi*ka1 *t1)
        N[3,m+3] +=  J3*a3  *ka1  *t3
        N[4,m+1] =   J3*(a3 *ka2 *t3   -a3 *t3_)
        N[5,m+2] =   J3*(a3i*ka1_ *t3   -a3 *ka12p *t3_)
    else:
        N[1,m] +=   J1*a1  *ka2 *t1_               - J2*a2pi *ka12m_*t2               #t1
        N[2,m] += - J1*a1  *ka2 *t1                + J2*a2i  *ka12p *t2              #t1_
        N[2,m+1] +=                                - J2*(a2  *ka12p*t2_ + a2p*ka2_*t2)
        N[1,m+1] +=   J3*a3  *ka2   *t3
        N[2,m+2] += - J3*a3  *ka12p *t3_
    #################################### HERMITIAN MATRIX
    for i in range(2*m):
        for j in range(i,2*m):
            N[j,i] += np.conjugate(N[i,j])
    #################################### L
    for i in range(2*m):
        N[i,i] += L
    return N
#
def ans_15(P,J2,J3,p):
    A1,B1,phiB1 = P[:3]
    if J2:
        p2,p3 = p
        phiB2 = p2*np.pi
        phiB2p = p3*np.pi
        A2,phiA2,A2p,phiA2p,B2,B2p = P[3:9]
        if J3:
            B3,phiB3 = P[9:]
        else:
            B3,phiB3 = np.zeros(2)
    else:
        phiB2 = phiB2p = 0
        A2,phiA2,A2p,phiA2p,B2,B2p = np.zeros(6)
        if J3:
            B3,phiB3 = P[3:]
        else:
            B3,phiB3 = np.zeros(2)
    phiA1p = 0
    phiB1p = -phiB1
    A3,phiA3 = np.zeros(2)
    return A1,phiA1p,B1,phiB1,phiB1p,A2,phiA2,A2p,phiA2p,B2,phiB2,B2p,phiB2p,A3,phiA3,B3,phiB3
#
def ans_16(P,J2,J3,p):
    A1,B1,phiB1 = P[:3]
    if J2:
        p2,p3 = p
        phiB2 = p2*np.pi
        phiB2p = p3*np.pi
        B2,B2p = P[3:5]
        if J3:
            A3,phiA3,B3,phiB3 = P[5:]
        else:
            A3,phiA3,B3,phiB3 = np.zeros(4)
    else:
        phiB2 = phiB2p = 0
        B2,B2p = np.zeros(2)
        if J3:
            A3,phiA3,B3,phiB3 = P[3:]
        else:
            A3,phiA3,B3,phiB3 = np.zeros(4)
    phiA1p = np.pi
    phiB1p = -phiB1
    A2,phiA2,A2p,phiA2p = np.zeros(4)
    return A1,phiA1p,B1,phiB1,phiB1p,A2,phiA2,A2p,phiA2p,B2,phiB2,B2p,phiB2p,A3,phiA3,B3,phiB3
#
def ans_17(P,J2,J3,p):
    A1,B1,phiB1 = P[:3]
    if J2:
        p2,p3 = p
        phiB2 = p2*np.pi
        phiB2p = p3*np.pi
        A2,phiA2,A2p,phiA2p,B2,B2p = P[3:9]
        if J3:
            A3,phiA3 = P[9:]
        else:
            A3,phiA3 = np.zeros(2)
    else:
        phiB2 = phiB2p = 0
        A2,phiA2,A2p,phiA2p,B2,B2p = np.zeros(6)
        if J3:
            A3,phiA3 = P[3:]
        else:
            A3,phiA3 = np.zeros(2)
    phiA1p = 0
    phiB1p = -phiB1
    B3,phiB3 = np.zeros(2)
    return A1,phiA1p,B1,phiB1,phiB1p,A2,phiA2,A2p,phiA2p,B2,phiB2,B2p,phiB2p,A3,phiA3,B3,phiB3
#
def ans_18(P,J2,J3,p):
    A1,B1,phiB1 = P[:3]
    if J2:
        p2,p3 = p
        phiB2 = p2*np.pi
        phiB2p = p3*np.pi
        B2,B2p = P[3:5]
    else:
        phiB2 = phiB2p = 0
        B2,B2p = np.zeros(2)
    phiA1p = np.pi
    phiB1p = -phiB1
    A2,phiA2,A2p,phiA2p,A3,phiA3,B3,phiB3 = np.zeros(8)
    return A1,phiA1p,B1,phiB1,phiB1p,A2,phiA2,A2p,phiA2p,B2,phiB2,B2p,phiB2p,A3,phiA3,B3,phiB3
#
def ans_19(P,J2,J3,p):
    A1,phiA1p,B1,phiB1 = P[:4]
    if J2:
        p2,p3 = p[:2]
        phiA2 = phiA1p/2 + p2*np.pi
        phiA2p = phiA1p/2 + p3*np.pi
        A2,A2p,B2,phiB2,B2p,phiB2p = P[4:10]
        if J3:
            p4,p5 = p[2:]
            phiA3 = (phiA1p+np.pi+2*np.pi*p4)/2
            phiB3 = p5*np.pi
            A3,B3 = P[10:]
        else:
            phiA3 = phiB3 = 0
            A3,B3 = np.zeros(2)
    else:
        phiA2 = phiA2p = 0
        A2,A2p,B2,phiB2,B2p,phiB2p = np.zeros(6)
        if J3:
            p4,p5 = p[:2]
            phiA3 = (phiA1p+np.pi+2*np.pi*p4)/2
            phiB3 = p5*np.pi
            A3,B3 = P[4:]
        else:
            phiA3 = phiB3 = 0
            A3,B3 = np.zeros(2)
    phiB1p = phiB1
    return A1,phiA1p,B1,phiB1,phiB1p,A2,phiA2,A2p,phiA2p,B2,phiB2,B2p,phiB2p,A3,phiA3,B3,phiB3
#
def ans_20(P,J2,J3,p):
    A1,phiA1p,B1,phiB1 = P[:4]
    if J2:
        p2,p3 = p[:2]
        phiA2 = phiA1p/2 + p2*np.pi
        phiA2p = phiA1p/2 + p3*np.pi
        A2,A2p,B2,phiB2,B2p,phiB2p = P[4:10]
        if J3:
            p4,p5 = p[2:]
            phiA3 = (phiA1p+2*np.pi*p4)/2
            phiB3 = p5*np.pi-np.pi/2
            A3,B3 = P[10:]
        else:
            phiA3 = phiB3 = 0
            A3,B3 = np.zeros(2)
    else:
        phiA2 = phiA2p = 0
        A2,A2p,B2,phiB2,B2p,phiB2p = np.zeros(6)
        if J3:
            p4,p5 = p[:2]
            phiA3 = (phiA1p+2*np.pi*p4)/2
            phiB3 = p5*np.pi-np.pi/2
            A3,B3 = P[4:]
        else:
            phiA3 = phiB3 = 0
            A3,B3 = np.zeros(2)
    phiB1p = phiB1
    return A1,phiA1p,B1,phiB1,phiB1p,A2,phiA2,A2p,phiA2p,B2,phiB2,B2p,phiB2p,A3,phiA3,B3,phiB3


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

