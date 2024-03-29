import numpy as np
from pathlib import Path
from scipy import linalg as LA
from scipy.interpolate import RectBivariateSpline as RBS
ans_1 = ('14','15','16','17','18')
ans_2 = ('19','20')
ans_p0 = ('15','16','19')
Mm = (3,6)

CO_gap = 0.8                                #0.015
CO_phase = 1e-3
CO_mod = 1e-2
CO_amp = 1e-3

def find_ansatz(head,data):
    ans = data[0]
    #
    P = []
    for temp_p in ['p2','p3','p4','p5']:
        if temp_p in head:
            P.append(int(data[head.index(temp_p)]))
    #
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
    gap = float(data[head.index('Gap')])
    if gap < CO_gap:
        gap_ = 'L'
    else:
        gap_ = 'S'
    #

    result = ans + phase_ + gap_ + str(len(P))
    for p in P:
        result += str(p)
    return result
#
def min_energy(lines,considered_ans):
    N = (len(lines)-1)//2 + 1
    minE = 10
    index = 0
    ansatze = []
    ind_ = 0
    for i in range(N):
        bad = False
        head_data = lines[2*i].split(',')
        head_data[-1] = head_data[-1][:-1]
        data = lines[2*i+1].split(',')
        if data[0] not in considered_ans:
            continue
        for i in range(head_data.index('A1'),len(data)):
            if float(data[i]) < -1e-3:
                bad = True
        if bad:
            continue
        if data[0] in ['19','20'] and (np.abs(float(data[head_data.index('phiA1p')])) < 1e-3 or np.abs(float(data[head_data.index('phiA1p')])-np.pi) < 1e-3):
                #and (find_15(head_data,data) or find_16(head_data,data)):
            continue
        energy = float(data[head_data.index('Energy')])
        ansatze.append(find_ansatz(head_data,data))
        if energy < minE:
            minE = energy
            ind_ = index
        index += 1 
    if len (ansatze) == 0:
        return 0
    return ansatze[ind_]
#
def find_15(head,data):     #establish if 19 is same result as 15
    result = True
    J2 = float(data[head.index('J2')])
    J3 = float(data[head.index('J3')])
    phiA1p = float(data[head.index('phiA1p')])
    phiB1 = float(data[head.index('phiB1')])
    if np.abs(phiA1p) > CO_phase or np.abs(phiB1-np.pi) > CO_phase:
        result = False
    if J2:
        p2 = int(data[head.index('p2')])
        p3 = int(data[head.index('p3')])
        phiB2 = float(data[head.index('phiB2')])
        phiB2p = float(data[head.index('phiB2p')])
        if p2 != 1 or p3 != 1 or np.abs(phiB2-np.pi)>CO_phase or np.abs(phiB2p-np.pi)>CO_phase:
            result = False
    if J3:
        A3 = float(data[head.index('A3')])
        p5 = int(data[head.index('p5')])
        if np.abs(A3) > CO_amp or p5 != 0:
            result = False
    return result
#
def find_16(head,data):     #establish if 19 is same result as 16
    result = True
    J2 = float(data[head.index('J2')])
    J3 = float(data[head.index('J3')])
    phiA1p = float(data[head.index('phiA1p')])
    phiB1 = float(data[head.index('phiB1')])
    if np.abs(phiA1p-np.pi) > CO_phase or np.abs(phiB1-np.pi) > CO_phase:
        result = False
    if J2:
        A2 = float(data[head.index('A2')])
        A2p = float(data[head.index('A2p')])
        phiB2 = float(data[head.index('phiB2')])
        phiB2p = float(data[head.index('phiB2p')])
        if not (np.abs(phiB2)<CO_phase or np.abs(phiB2-2*np.pi)<CO_phase) or not (np.abs(phiB2p)<CO_phase or np.abs(phiB2p-2*np.pi)<CO_phase) or np.abs(A2) > CO_amp or np.abs(A2p) > CO_amp:
            result = False
    if J3:
        p4 = int(data[head.index('p4')])
        p5 = int(data[head.index('p5')])
        if p4 != 1 or p5 != 1:
            result = False
    return True
#
def find_gap(args,K):
    ans,txt_DM,J2,J3,txt_S = args
    DirName = '../../Data/self_consistency/S'+txt_S+'/phi'+txt_DM+"/"
    DataDir = DirName + str(K) + '/'
    csvname = 'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
    csvfile = DataDir + csvname
    my_file = Path(csvfile)
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//2 +1        #2 lines per ansatz
        for i in range(N):
            head = lines[i*2].split(',')
            head[-1] = head[-1][:-1]
            data = lines[i*2+1].split(',')
            valid = True
            for j in range(head.index('A1'),len(data)):
                if float(data[j]) < -1e-3:
                    valid = False
            if not valid:
                continue
            if data[0] != ans:
                continue
            if ans in ans_1:
                return float(data[head.index('Gap')])
            phiA1p = float(data[head.index('phiA1p')])
            if np.abs(phiA1p) < CO_phase or np.abs(phiA1p-2*np.pi) < CO_phase or np.abs(phiA1p-np.pi) < CO_phase:
                continue
            return float(data[head.index('Gap')])
#
def quadratic(x,a,b):
    return a/(3*x**2) + b
def ql(x,a,b,c):
    return a/(3*x**2) + c/x + b
def linear(x,a,b):
    return a/x + b

def compute_KM(kkg,a1,a2,a12p,a12m):
    ka1 = np.exp(1j*np.tensordot(a1,kkg,axes=1));   ka1_ = np.conjugate(ka1);
    ka2 = np.exp(1j*np.tensordot(a2,kkg,axes=1));   ka2_ = np.conjugate(ka2);
    ka12p = np.exp(1j*np.tensordot(a12p,kkg,axes=1));   ka12p_ = np.conjugate(ka12p);
    ka12m = np.exp(1j*np.tensordot(a12m,kkg,axes=1));   ka12m_ = np.conjugate(ka12m);
    KM_ = (ka1,ka1_,ka2,ka2_,ka12p,ka12p_,ka12m,ka12m_)
    return KM_
def compute_par(data,head,par_name,S,phi,K_):
    ans = data[0]
    J1 = 1
    J2 = float(data[head.index('J2')])
    J3 = float(data[head.index('J3')])
    J = (J1,J2,J3)
    st = head.index('J2')
    PpP = []
    for i in range(1,st):
        PpP.append(int(data[i]))
    PpP = tuple(PpP)
    DM1 = phi;      DM2 = 0;    DM3 = 2*phi
    t1 = np.exp(-1j*DM1);    t1_ = np.conjugate(t1)
    t2 = np.exp(-1j*DM2);    t2_ = np.conjugate(t2)
    t3 = np.exp(-1j*DM3);    t3_ = np.conjugate(t3)
    Tau = (t1,t1_,t2,t2_,t3,t3_)
    kxg = np.linspace(0,1,K_)
    kyg = np.linspace(0,1,K_)
    kkg = np.ndarray((2,K_,K_),dtype=complex)
    kkg_small = np.ndarray((2,K_,K_),dtype=complex)
    for i in range(K_):
        for j in range(K_):
            kkg[0,i,j] = kxg[i]*2*np.pi
            kkg[1,i,j] = (kxg[i]+kyg[j])*2*np.pi/np.sqrt(3)
            kkg_small[0,i,j] = kxg[i]*2*np.pi
            kkg_small[1,i,j] = (kxg[i]+2*kyg[j])*2*np.pi/np.sqrt(3)
    #### vectors of 1nn, 2nn and 3nn
    a1 = (1,0)
    a2 = (-1,np.sqrt(3))
    a2_small = (-1/2,np.sqrt(3)/2)
    a12p = (a1[0]+a2[0],a1[1]+a2[1])
    a12m = (a1[0]-a2[0],a1[1]-a2[1])
    a12p_small = (a1[0]+a2_small[0],a1[1]+a2_small[1])
    a12m_small = (a1[0]-a2_small[0],a1[1]-a2_small[1])
    #### product of lattice vectors with K-matrix
    KM_big = compute_KM(kkg,a1,a2,a12p,a12m)     #large unit cell
    KM_small = compute_KM(kkg_small,a1,a2_small,a12p_small,a12m_small)
    KM = KM_small if ans in ans_p0 else KM_big
    p1 = 0 if ans in ans_p0 else 1
    m = Mm[p1]
    J_ = np.zeros((2*m,2*m))
    for i in range(m):
        J_[i,i] = -1
        J_[i+m,i+m] = 1
    #Compute first the transformation matrix M at each needed K
    args_M = (KM,Tau,K_,S,J,ans,PpP)
    L = float(data[head.index('L')])
    old_O = []
    for i in range(head.index('A1'),len(data)):
        old_O.append(float(data[i]))
    N = big_Nk(old_O,L,args_M)
    M = np.zeros(N.shape,dtype=complex)
    for i in range(K_):
        for j in range(K_):
            N_k = N[:,:,i,j]
            Ch = LA.cholesky(N_k) #upper triangular-> N_k=Ch^{dag}*Ch
            w,U = LA.eigh(np.dot(np.dot(Ch,J_),np.conjugate(Ch.T)))
            w = np.diag(np.sqrt(np.einsum('ij,j->i',J_,w)))
            M[:,:,i,j] = np.dot(np.dot(LA.inv(Ch),U),w)
    #for each parameter need to know what it is
    dic_O = {'A':compute_A,'B':compute_B}
    phase_phiA1p = 0
    kxg = np.linspace(0,1,K_)
    kyg = np.linspace(0,1,K_)
    #
    #
    par = par_name
    par_ = par[-2:] if par[-1]=='p' else par[-1]
    par_1 = par[-2] if par[-1]=='p' else par[-1]
    par_2 = 'A' if 'A' in par else 'B'
    li_ = dic_indexes[str(m)][par_][0]
    lj_ = dic_indexes[str(m)][par_][1]
    Tau_ = (Tau[2*(int(par_1)-1)],Tau[2*(int(par_1)-1)+1])
    DM_ch = True if par_ in ['2p','3'] else False
    #func = dic_O[par_2]
    #res = 0
    rrr = np.zeros((K_,K_),dtype=complex)
    for i in range(K_):
        for j in range(K_):
            K__ = np.array([kxg[i]*2*np.pi,(kxg[i]+2*kyg[j])*2*np.pi/np.sqrt(3)])
            U,X,V,Y = split(M[:,:,i,j],m,m)
            U_,V_,X_,Y_ = split(np.conjugate(M[:,:,i,j].T),m,m)
#                res += func(U,X,V,Y,U_,X_,V_,Y_,Tau,li_,lj_)
            rrr[i,j] = dic_O[par_2](U,X,V,Y,U_,X_,V_,Y_,Tau_,li_,lj_,K__,DM_ch)
    interI = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),np.imag(rrr))
    res2I = interI.integral(0,1,0,1)
    interR = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),np.real(rrr))
    res2R = interR.integral(0,1,0,1)
    Res = (res2R+1j*res2I)/2
    #res /= 2*K_**2
    if par[0] == 'p':           #phases
        res = np.angle(Res)
    else:                   #Amplitudes
        res = np.absolute(Res)                   #renormalization of amplitudes 
    return res
# Computes the expected phase (given by the ansatz) of a given amplitude
def find_phase(ans,J,PpP,par,phase_phiA1p):
    if par == 'B2':
        return np.pi*PpP[0]
    if par == 'B2p':
        return np.pi*PpP[1]
    if par == 'A2':
        return np.pi*PpP[0] + phase_phiA1p/2
    if par == 'A2p':
        return np.pi*PpP[1] + phase_phiA1p/2
    J1,J2,J3 = J
    if par == 'A3':
        if J2:
            if ans == '19':
                return (phase_phiA1p + np.pi + 2*PpP[2]*np.pi)/2
            else:
                return (phase_phiA1p + 2*PpP[2]*np.pi)/2
        else:
            if ans == '19':
                return (phase_phiA1p + np.pi + 2*PpP[0]*np.pi)/2
            else:
                return (phase_phiA1p + 2*PpP[0]*np.pi)/2
    if par == 'B3':
        if J2:
            if ans == '19':
                return np.pi*PpP[3]
            else:
                return np.pi*PpP[3] - np.pi/2
        else:
            if ans == '19':
                return np.pi*PpP[1]
            else:
                return np.pi*PpP[1] - np.pi/2
    #The End
        

dic_indexes =   {'3':{'1': (1,2), '1p': (2,0), 
                      '2': (1,0), '2p': (2,1), 
                      '3': (1,1)},
                 '6':{'1': (1,2), '1p': (2,0), 
                      '2': (1,0), '2p': (5,1), 
                      '3': (4,1)}
                 }
def compute_A(U,X,V,Y,U_,X_,V_,Y_,Tau,li_,lj_,K__,DM_ch):
    if DM_ch:
        Tau = np.conjugate(np.array(Tau))
    if (li_,lj_) in [(2,1),(1,1)]:
        dist = np.array([-1/2,np.sqrt(3)/2])
    else:
        dist = np.zeros(2)
    return (np.einsum('ln,nm->lm',U,V_)[li_,lj_]     *Tau[1] *np.exp(1j*np.dot(K__,dist))
            - np.einsum('nl,mn->lm',Y_,X)[li_,lj_]   *Tau[0] *np.exp(-1j*np.dot(K__,dist)))
########
def compute_B(U,X,V,Y,U_,X_,V_,Y_,Tau,li_,lj_,K__,DM_ch):
    if DM_ch:
    #if li_== 2 or li_ == 4:
        Tau = np.conjugate(np.array(Tau))
    if (li_,lj_) in [(2,1),(1,1)]:
        dist = np.array([-1/2,np.sqrt(3)/2])
    else:
        dist = np.zeros(2)
    return (np.einsum('nl,mn->lm',X_,X)[li_,lj_]  *Tau[0] *np.exp(-1j*np.dot(K__,dist))
            + np.einsum('ln,nm->lm',V,V_)[li_,lj_]*Tau[1] *np.exp(1j*np.dot(K__,dist)))

def split(array, nrows, ncols):
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))


#### Ansatze encoded in the matrix
def big_Nk(P,L,args):
    KM,Tau,K_,S,J,ans,PpP = args
    J1,J2,J3 = J
    p1 = 0 if ans in ans_p0 else 1
    m = Mm[p1]
    ka1,ka1_,ka2,ka2_,ka12p,ka12p_,ka12m,ka12m_ = KM
    t1,t1_,t2,t2_,t3,t3_ = Tau
    J1 /= 2.
    J2 /= 2.
    J3 /= 2.
    func_ans = {'15':ans_15, '16':ans_16,'17':ans_17,'18':ans_18,'19':ans_19,'20':ans_20, '14':ans_17}
    A1,phiA1p,B1,phiB1,phiB1p,A2,phiA2,A2p,phiA2p,B2,phiB2,B2p,phiB2p,A3,phiA3,B3,phiB3 = func_ans[ans](P,J2,J3,PpP)
    A1p = A1
    B1p = B1
    ################
    N = np.zeros((2*m,2*m,K_,K_), dtype=complex)
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
    #   B
    N[0,1] = J1*b1p_ *ka1  *t1_              + J2*b2*t2                         #t1
    N[0,2] = J1*b1p        *t1               + J2*b2p_ *ka1*t2                  #t1_
    N[1,2] = J1*(b1_       *t1  + b1p_*ka1_*t1_)                                #t1     t1
    N[0,0] = J3*b3i_ *ka1_ *t3_
    ####other half square                                                       #Same ts
    N[m+0,m+1] = J1*b1p  *ka1  *t1_           + J2*b2_*t2
    N[m+0,m+2] = J1*b1p_       *t1            + J2*b2p  *ka1*t2
    N[m+1,m+2] = J1*(b1        *t1  + b1p*ka1_*t1_)
    N[m+0,m+0] = J3*b3i *ka1_ *t3_
    #   A
    N[0,m+1] = - J1*a1p *ka1 *t1_           +J2*a2*t2
    N[0,m+2] =   J1*a1p      *t1            -J2*a2p  *ka1*t2
    N[1,m+2] = - J1*(a1      *t1   +a1p*ka1_*t1_)
    N[0,m+0] = - J3*a3i *ka1_*t3_
    ####other half square (not the diagonal)                                                       #Same ts
    N[1,m]   =   J1*a1p *ka1_*t1            -J2*a2*t2_
    N[2,m]   = - J1*a1p      *t1_           +J2*a2p  *ka1_*t2_
    N[2,m+1] =   J1*(a1      *t1_  +a1p*ka1 *t1)
    N[0,m+0] +=  J3*a3i *ka1  *t3
    if m == 6:
        #   B
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
        #
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
        #   A
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
        #
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
        #   B
        N[0,1] += J1*b1_  *ka2_ *t1               + J2*b2pi *ka12m*t2_               #t1
        N[0,2] += J1*b1   *ka2_ *t1_              + J2*b2i_ *ka12p_*t2_              #t1_
        N[1,2] +=                                   J2*(b2  *ka12p_*t2 + b2p*ka2*t2_)
        N[1,1] += J3*b3_  *ka2_ *t3_
        N[2,2] += J3*b3   *ka12p_ *t3
        #
        N[m+0,m+1] += J1*b1  *ka2_ *t1               + J2*b2pi_ *ka12m*t2_               #t1
        N[m+0,m+2] += J1*b1_   *ka2_ *t1_              + J2*b2i *ka12p_*t2_              #t1_
        N[m+1,m+2] +=                                   J2*(b2_  *ka12p_*t2 + b2p_*ka2*t2_)
        N[m+1,m+1] += J3*b3  *ka2_ *t3_
        N[m+2,m+2] += J3*b3_   *ka12p_ *t3
        #   A
        N[0,m+1] += - J1*a1  *ka2_ *t1               + J2*a2pi *ka12m*t2_               #t1
        N[0,m+2] +=   J1*a1  *ka2_ *t1_              - J2*a2i  *ka12p_*t2_              #t1_
        N[1,m+2] +=                                   J2*(a2  *ka12p_*t2 + a2p*ka2*t2_)
        N[1,m+1] += - J3*a3  *ka2_ *t3_
        N[2,m+2] += J3*a3   *ka12p_ *t3
        #
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
def ans_15(P,J2,J3,PpP):
    A1,B1,phiB1 = P[:3]
    if J2:
        p2,p3 = PpP
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
def ans_16(P,J2,J3,PpP):
    A1,B1,phiB1 = P[:3]
    if J2:
        p2,p3 = PpP
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
def ans_17(P,J2,J3,PpP):
    A1,B1,phiB1 = P[:3]
    if J2:
        p2,p3 = PpP
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
def ans_18(P,J2,J3,PpP):
    A1,B1,phiB1 = P[:3]
    if J2:
        p2,p3 = PpP
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
def ans_19(P,J2,J3,PpP):
    A1,phiA1p,B1,phiB1 = P[:4]
    if J2:
        p2,p3 = PpP[:2]
        phiA2 = phiA1p/2 + p2*np.pi
        phiA2p = phiA1p/2 + p3*np.pi
        A2,A2p,B2,phiB2,B2p,phiB2p = P[4:10]
        if J3:
            p4,p5 = PpP[2:]
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
            p4,p5 = PpP[:2]
            phiA3 = (phiA1p+np.pi+2*np.pi*p4)/2
            phiB3 = p5*np.pi
            A3,B3 = P[4:]
        else:
            phiA3 = phiB3 = 0
            A3,B3 = np.zeros(2)
    phiB1p = phiB1
    return A1,phiA1p,B1,phiB1,phiB1p,A2,phiA2,A2p,phiA2p,B2,phiB2,B2p,phiB2p,A3,phiA3,B3,phiB3
#
def ans_20(P,J2,J3,PpP):
    A1,phiA1p,B1,phiB1 = P[:4]
    if J2:
        p2,p3 = PpP[:2]
        phiA2 = phiA1p/2 + p2*np.pi
        phiA2p = phiA1p/2 + p3*np.pi
        A2,A2p,B2,phiB2,B2p,phiB2p = P[4:10]
        if J3:
            p4,p5 = PpP[2:]
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
            p4,p5 = PpP[:2]
            phiA3 = (phiA1p+2*np.pi*p4)/2
            phiB3 = p5*np.pi-np.pi/2
            A3,B3 = P[4:]
        else:
            phiA3 = phiB3 = 0
            A3,B3 = np.zeros(2)
    phiB1p = phiB1
    return A1,phiA1p,B1,phiB1,phiB1p,A2,phiA2,A2p,phiA2p,B2,phiB2,B2p,phiB2p,A3,phiA3,B3,phiB3
















