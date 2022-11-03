import inputs as inp
import numpy as np

#### Ansatze encoded in the matrix
def Nk(P,L,args):
    m = inp.m
    J1,J2,J3,ans = args
    J1 /= 2.
    J2 /= 2.
    J3 /= 2.
    j2 = np.sign(int(np.abs(J2)*1e8))   #check if it is 0 or not --> problem with VERY small J2,J3
    j3 = np.sign(int(np.abs(J3)*1e8))
    A1 = P[0]
    #parameters of the various ansatze
    if ans == '3x3':      #3
        A2 = 0;     phiA2 = 0;    phiA2p = 0;
        A3 = P[1*j3]*j3
        B1 = P[2*j3]*j3 + P[1]*(1-j3)
        B2 = P[3*j2*j3]*j2*j3+P[2*j2*(1-j3)]*(1-j3)*j2
        B3 = P[4*j3*j2]*j3*j2+P[3*j3*(1-j2)]*j3*(1-j2)
        phiA3 = P[-1]*j3
        phiA1p = np.pi
        phiB1, phiB1p, phiB2, phiB2p, phiB3 = (np.pi, np.pi, 0, 0, np.pi)
        p1 = 0
    elif ans == 'q0':     #1
        A3 = 0; phiA3 = 0
        A2 = P[1]*j2
        B1 = P[2*j2]*j2+P[1]*(1-j2)
        B2 = P[3*j2]*j2
        B3 = P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2)
        phiA2 = P[-1]*j2
        phiA1p, phiA2p = (0, phiA2)
        phiB1, phiB1p, phiB2, phiB2p, phiB3 = (np.pi, np.pi, np.pi, np.pi, 0)
        p1 = 0
    elif ans == 'cb1':
        B3 = 0; phiB3 = 0
        A2 = P[1*j2]*j2
        A3 = P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2)
        B1 = P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3)
        B2 = P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3)
        phiA1p = P[-2*j2]*j2 + P[-1]*(1-j2)
        phiB2 = P[-1]*j2
        phiA2, phiA2p, phiA3 = (phiA1p/2+np.pi, phiA1p/2+np.pi, phiA1p/2)
        phiB1, phiB1p, phiB2p= (np.pi, np.pi ,-phiB2)
        p1 = 1
    ################
    N = np.zeros((2*m,2*m,inp.Nx,inp.Ny), dtype=complex)
    ##################################### B
    b1 = B1*np.exp(1j*phiB1);               b1_ = np.conjugate(b1)
    b1p = B1*np.exp(1j*phiB1p);             b1p_ = np.conjugate(b1p)
    b1pi = B1*np.exp(1j*(phiB1p+p1*np.pi)); b1pi_ = np.conjugate(b1pi)
    b2 = B2*np.exp(1j*phiB2);               b2_ = np.conjugate(b2)
    b2i = B2*np.exp(1j*(phiB2+p1*np.pi));   b2i_ = np.conjugate(b2i)
    b2p = B2*np.exp(1j*phiB2p);             b2p_ = np.conjugate(b2p)
    b2pi = B2*np.exp(1j*(phiB2p+p1*np.pi)); b2pi_ = np.conjugate(b2pi)
    b3 = B3*np.exp(1j*phiB3);               b3_ = np.conjugate(b3)
    b3i = B3*np.exp(1j*(phiB3+p1*np.pi));   b3i_ = np.conjugate(b3i)
    #
    N[0,1] = J1*b1p_ *ka1  *t1_              + J2*b2*t2                         #t1
    N[0,2] = J1*b1p        *t1               + J2*b2p_ *ka1*t2                  #t1_
    N[0,4] = J1*b1_  *ka2_ *t1               + J2*b2pi *ka12m*t2_               #t1
    N[0,5] = J1*b1   *ka2_ *t1_              + J2*b2i_ *ka12p_*t2_              #t1_
    N[1,2] = J1*(b1_       *t1  + b1p_*ka1_*t1_)                                #t1     t1
    N[1,3] = J1*b1         *t1_              + J2*b2p_ *ka1_*t2                 #t1_
    N[1,5] =                                   J2*(b2  *ka12p_*t2 + b2p*t2_)
    N[2,3] = J1*b1_        *t1               + J2*b2   *ka1*t2                  #t1
    N[2,4] =                                   J2*(b2p_*ka2_*t2 + b2i_*ka1*t2_)
    N[3,4] = J1*b1pi_*ka1  *t1_              + J2*b2*t2                         #t1
    N[3,5] = J1*b1p        *t1               + J2*b2pi_*ka1*t2                  #t1_
    N[4,5] = J1*(b1_       *t1  + b1pi_*ka1_*t1_)                               #t1     t1

    N[0,0] = J3*b3i_ *ka1_ *t3_
    N[3,3] = J3*b3_  *ka1_ *t3_
    N[1,4] = J3*(b3_ *ka2_ *t3_ + b3       *t3)
    N[2,5] = J3*(b3  *ka12p_  *t3  + b3i_*ka1*t3_)
    ####other half square                                                       #Same ts
    N[m+0,m+1] = J1*b1p  *ka1  *t1_           + J2*b2_*t2
    N[m+0,m+2] = J1*b1p_       *t1            + J2*b2p  *ka1*t2
    N[m+0,m+4] = J1*b1   *ka2_ *t1            + J2*b2pi_*ka12m*t2_
    N[m+0,m+5] = J1*b1_  *ka2_ *t1_           + J2*b2i  *ka12p_*t2_
    N[m+1,m+2] = J1*(b1        *t1  + b1p*ka1_*t1_)
    N[m+1,m+3] = J1*b1_        *t1_           + J2*b2p  *ka1_*t2
    N[m+1,m+5] =                                J2*(b2_ *ka12p_*t2 + b2p_*t2_)
    N[m+2,m+3] = J1*b1         *t1            + J2*b2_  *ka1*t2
    N[m+2,m+4] =                                J2*(b2p *ka2_*t2 + b2i *ka1*t2_)
    N[m+3,m+4] = J1*b1pi *ka1  *t1_           + J2*b2_*t2
    N[m+3,m+5] = J1*b1p_       *t1            + J2*b2pi *ka1*t2
    N[m+4,m+5] = J1*(b1        *t1  + b1pi*ka1_*t1_)

    N[m+0,m+0] = J3*b3i *ka1_ *t3_
    N[m+3,m+3] = J3*b3  *ka1_ *t3_
    N[m+1,m+4] = J3*(b3 *ka2_ *t3_ + b3_  *t3)
    N[m+2,m+5] = J3*(b3_*ka12p_  *t3  + b3i *ka1*t3_)
    ######################################## A
    a1 =    A1
    a1p =   A1*np.exp(1j*phiA1p)
    a1pi =  A1*np.exp(1j*(phiA1p+p1*np.pi))
    a2 =    A2*np.exp(1j*phiA2)
    a2i =   A2*np.exp(1j*(phiA2+p1*np.pi))
    a2p =   A2*np.exp(1j*phiA2p)
    a2pi =  A2*np.exp(1j*(phiA2p+p1*np.pi))
    a3 =    A3*np.exp(1j*phiA3)
    a3i =   A3*np.exp(1j*(phiA3+p1*np.pi))
    N[0,m+1] = - J1*a1p *ka1 *t1_           +J2*a2*t2
    N[0,m+2] =   J1*a1p      *t1            -J2*a2p  *ka1*t2
    N[0,m+4] = - J1*a1  *ka2_*t1            +J2*a2pi *ka12m*t2_
    N[0,m+5] =   J1*a1  *ka2_*t1_           -J2*a2i  *ka12p_*t2_
    N[1,m+2] = - J1*(a1      *t1   +a1p*ka1_*t1_)
    N[1,m+3] =   J1*a1       *t1_           -J2*a2p  *ka1_*t2
    N[1,m+5] =                               J2*(a2  *ka12p_*t2  +a2p*t2_)
    N[2,m+3] = - J1*a1       *t1            +J2*a2   *ka1*t2
    N[2,m+4] =                              -J2*(a2p *ka2_*t2  +a2i*ka1*t2_)
    N[3,m+4] = - J1*a1pi*ka1 *t1_           +J2*a2*t2
    N[3,m+5] =   J1*a1p      *t1            -J2*a2pi *ka1*t2
    N[4,m+5] = - J1*(a1      *t1   +a1pi*ka1_*t1_)

    N[0,m+0] = - J3*a3i *ka1_*t3_
    N[3,m+3] = - J3*a3  *ka1_*t3_
    N[1,m+4] = - J3*(a3 *ka2_*t3_  -a3 *t3)
    N[2,m+5] = - J3*(a3i*ka1*t3_  -a3 *ka12p_ *t3)
    #not the diagonal
    N[1,m]   =   J1*a1p *ka1_*t1            -J2*a2*t2_
    N[2,m]   = - J1*a1p      *t1_           +J2*a2p  *ka1_*t2_
    N[4,m]   =   J1*a1  *ka2 *t1_           -J2*a2pi *ka12m_*t2
    N[5,m]   = - J1*a1  *ka2 *t1            +J2*a2i  *ka12p*t2
    N[2,m+1] =   J1*(a1      *t1_  +a1p*ka1 *t1)
    N[3,m+1] = - J1*a1       *t1            +J2*a2p  *ka1_*t2_
    N[5,m+1] =                              -J2*(a2  *ka12p*t2_   +a2p*t2)
    N[3,m+2] =   J1*a1       *t1_           -J2*a2   *ka1_*t2_
    N[4,m+2] =                               J2*(a2p *ka2*t2_   +a2i*ka1_*t2)
    N[4,m+3] =   J1*a1pi*ka1_*t1            -J2*a2*t2_
    N[5,m+3] = - J1*a1p      *t1_           +J2*a2pi *ka1_*t2_
    N[5,m+4] =   J1*(a1      *t1_  +a1pi*ka1 *t1)

    N[0,m+0] +=  J3*a3i *ka1  *t3
    N[3,m+3] +=  J3*a3  *ka1  *t3
    N[4,m+1] =   J3*(a3 *ka2 *t3   -a3 *t3_)
    N[5,m+2] =   J3*(a3i*ka1_ *t3   -a3 *ka12p *t3_)
    #################################### HERMITIAN MATRIX
    #N += np.conjugate(N.transpose((1,0,2,3)))
    for i in range(2*m):
        for j in range(i,2*m):
            N[j,i] += np.conjugate(N[i,j])
    #################################### L
    for i in range(2*m):
        N[i,i] += L
    return N

#        |*          |*          |
#        |  *    B(k)|  *   A(k) |
#        |    *      |    *      |
#        |      *    |      *    |
#        |  c.c.  *  |  -A*(k)*  |
#        |          *|          *|
#  N  =  |-----------|-----------|  +  diag(L)
#        |*          |*          |
#        |  *   c.c. |  *  B*(-k)|
#        |    *      |    *      |
#        |      *    | c.c. *    |
#        | c.c.   *  |        *  |
#        |          *|          *|
