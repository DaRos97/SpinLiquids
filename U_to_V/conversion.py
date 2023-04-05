import numpy as np


def coefficients(t1,t2,t3,U,V,phi1=0,phi2=0,phi3=0,by_order=False):
    # second order
    J1_2_z = 4*t1**2/(U-V)
    # print ('2nd order:', J1_2_z)
    J2_2_z = 4*t2**2/U
    J3_2_z = 4*t3**2/U

    J1_2_pm = 0.5*J1_2_z*np.exp(-2*1j*phi1)
    J2_2_pm = 0.5*J2_2_z*np.exp(-2*1j*phi2)
    J3_2_pm = 0.5*J3_2_z*np.exp(-2*1j*phi3)

    # third order

    J1_3a = -2*t1**3*(U+V)/((U-V)*V**2)
    J1_3b = -2*t1**2*t2*(U+2*V)/((U-V)*V**2)

    J1_3_z  = 2*np.cos(3*phi1)*J1_3a + 2*np.cos(phi2)*J1_3b
    # print ('3rd order:', J1_3_z)
    J1_3_pm = J1_3a*np.exp(1j*phi1) + J1_3b*np.cos(phi2)*np.exp(-2*1j*phi1)

    J2_3a = -2*t1**2*t2*(U+2*V)/(U*V**2)
    J2_3b = -2*t2**3*(U+2*V)/(U*V**2)
    J3_3a = -2*t1**2*t3*(U+2*V)/(U*V**2)
    J3_3b = -4*t1*t2*t3*(U+2*V)/(U*V**2)

    J2_3_z  = 2*np.cos(phi2)*J2_3a + 2*np.cos(3*phi2)*J2_3b
    J2_3_pm = J2_3a*np.exp(-1j*phi2) + J2_3b*np.exp(1j*phi2)
    J3_3_z  = 2*np.cos(phi3-2*phi1)*J3_3a
    J3_3_pm = J3_3a*np.exp(-1j*(phi3+2*phi1))

    J3_3f_z1  =  2*np.cos(phi1-phi2+phi3)*J3_3b
    J3_3f_z2  =  2*np.cos(phi1+phi2+phi3)*J3_3b
    J3_3f_pm1 =  np.exp(1j*(phi1-phi2-phi3))*J3_3b
    J3_3f_pm2 =  np.exp(1j*(phi1+phi2-phi3))*J3_3b

    # fourth order

    J1_4a = t1**4*(-42./((U-V)**3)  - 7./((U-V)**2*V) - 10./((U-V)*V**2) - 3./(V**3) + 18./((U+V)*(U-V)**2) + 4./((U+V)*V**2) + 16./(U*(U-V)**2) + 8./((U+V)*(U-V)*V) + 32./((U-V)**2*(2*U-V))  + 32./((U-V)**2*(2*U-3*V)))
    J1_4b = t1**4*( 3./((U-V)**3) + 2./((U-V)*V**2) + 4./((U+V)*V**2) + 3./(2*(U-V)**2*V) - 1./(2*V**3) )

    J1_4_z  = 2*(J1_4a + J1_4b)
    J1_4_pm = J1_4a*np.exp(-2*1j*phi1) + J1_4b*np.exp(4*1j*phi1)

    J2_4_pm = t1*4 * ( 5./((U-V)**3) - 2./(U*(U-V)**2) + 6./(U*V**2)  + 1./((U-V)**2*V) - 1./(V**3) - 1./((U-V)*V**2) )
    J2_4_z  = 2*J2_4_pm

    J3_4f = 2*t1**4 * (2./((U-V)**3) - 1./(U*(U-V)**2))
    J3_4e = 6*t1**4 / (U*V**2)

    J3_4f_z  = 2*J3_4f
    J3_4f_pm = J3_4f*np.exp(-4*1j*phi1)
    J3_4e_z  = 2*J3_4e
    J3_4e_pm = J3_4e*np.exp(-4*1j*phi1)

    # print ('4th order:', J1_4_z, '\n')

    J1_z  = J1_2_z + J1_3_z + J1_4_z
    J1_pm = J1_2_pm + J1_3_pm + J1_4_pm
    J2_z  = J2_2_z + J2_3_z + J2_4_z
    J2_pm = J2_2_pm + J2_3_pm + J2_4_pm
    J3f_z  = J3_2_z + J3_4f_z # + J3_3f_z1 + J3_3f_z2
    J3f_pm = J3_2_pm + J3_4f_pm # + J3_3f_pm1 + J3_3f_pm2
    J3e_z  = J3_2_z + J3_3_z + J3_4e_z
    J3e_pm  = J3_2_pm + J3_3_pm + J3_4e_pm

    J4 = 8*t1**4*(1./((U-V)**3) - 1./((U-V)**2*(2*U-V))- 1./((U-V)**2*(2*U-3*V)))

    J12 = (-6*t1**3*t2 + 4*t1**2*t2**2 + 6*t1**3*t3 + 3*t1*t2**2*t3) / (3*V**2*V) \
        - 16*t1**4 / ((U-V)**2*(U+V)) \
        + (32*t1**4 - 8*t1**3*t2 + 8*t1**2*t2**2) / ((U-V)**2*(U+3*V)) \
        + (-16*t1**4 - 18*t1**3*t2 + 24*t1**2*t2**2 + 8*t1**3*t3 + 8*t1*t2**2*t3) / (V*(U+3*V)*(U-V)) \
        + (-6*t1**4 - 4*t1**3*t2 + 10*t1**2*t2**2 + 2*t1**3*t3 + 2*t1*t2**2*t3) / (V**2*(U+3*V))

    J23 = (3*t1**3*t2 + 9*t1**3*t3 + 6*t1**2*t2*t3) / (3*V**3) \
        + (-4*t1**4 + 8*t1**3*t2 + 4*t1**2*t2**2) / ((U-V)**2*(U+3*V)) \
        + (-5*t1**4 + 6*t1**3*t2 + 8*t1**2*t2**2 + 12*t1**3*t3 + 12*t1**2*t2*t3) / (V*(U+3*V)*(U-V)) \
        + (-1*t1**4 + 1*t1**3*t2 + 4*t1**2*t2**2 + 3*t1**3*t3 + 3*t1**2*t2*t3) / (V**2*(U+3*V))


    coeffs = {}

    if by_order:
        coeffs['J1_z']   = [J1_2_z, J1_3_z, J1_4_z]
        coeffs['J1_pm']  = [J1_2_pm, J1_3_pm, J1_4_pm]
        coeffs['J2_z']   = [J2_2_z, J2_3_z, J2_4_z]
        coeffs['J2_pm']  = [J2_2_pm, J2_3_pm, J2_4_pm]
        coeffs['J3f_z']  = [J3_2_z, J3_4f_z]
        coeffs['J3f_pm'] = [J3_2_pm, J3_4f_pm]
        coeffs['J3e_z']  = [J3_2_z, J3_3_z, J3_4e_z]
        coeffs['J3e_pm'] = [J3_2_pm, J3_3_pm, J3_4e_pm]
        coeffs['J1_add'] = 0.5*J12 + J23
        
    else:
        coeffs['J1_z']   = J1_z # + 0.5*J12 + J23
        coeffs['J1_pm']  = J1_pm
        coeffs['J2_z']   = J2_z
        coeffs['J2_pm']  = J2_pm
        coeffs['J3f_z']  = J3f_z
        coeffs['J3f_pm'] = J3f_pm
        coeffs['J3e_z']  = J3e_z
        coeffs['J3e_pm'] = J3e_pm
        coeffs['J3f_add'] = {'z1': J3_3f_z1, 'z2': J3_3f_z2, 'pm1': J3_3f_pm1, 'pm2': J3_3f_pm2}
        
    coeffs['J4']     = J4

    return coeffs

print(coefficients(1,0.1,0.1,75,75*0.14))


#######################################

# coffs = coefficients(1.,0.1448716923890239, 0.0799323919781765, 75, 10.5, phi1=2*np.pi/3,phi2=np.pi,phi3=np.pi/3,by_order=False)
# coffs = coefficients(1.,0.1448716923890239, 0.0799323919781765, 75, 10.5, phi1=0.,phi2=np.pi,phi3=np.pi,by_order=False)

# print ('J1_z   =', coffs['J1_z'])
# print ('J1_pm  =', np.abs(coffs['J1_pm'])*2, '/', coffs['J1_pm']*2)
# print ('2pi/3  =', 2*np.pi/3)
# print ('phi1   =', np.angle(coffs['J1_pm']), '\n')

# print ('J2_z   =', coffs['J2_z'])
# print ('J2_pm  =', np.abs(coffs['J2_pm'])*2, '/', coffs['J2_pm']*2)
# print ('2p     =', 2*np.pi)
# print ('phi2   =', np.angle(coffs['J2_pm']), '\n')

# print ('J3_z   =', coffs['J3e_z'])
# print ('J3_pm  =', np.abs(coffs['J3e_pm'])*2, '/', coffs['J3e_pm']*2)
# print ('-2pi/3 =', -2*np.pi/3)
# print ('phi3   =', np.angle(coffs['J3e_pm']), '\n')

# print ('J3f_z  =', coffs['J3f_z'])
# print ('J3_pm  =', np.abs(coffs['J3f_pm'])*2, '/', coffs['J3f_pm']*2)
# print ('-2pi/3 =', -2*np.pi/3)
# print ('phi3   =', np.angle(coffs['J3f_pm']), '\n')

# print ('J3a_z1 =', coffs['J3f_add']['z1'])
# print ('J3_pm1 =', np.abs(coffs['J3f_add']['pm1'])*2, '/', coffs['J3f_add']['pm1']*2)
# print ('J3a_z2 =', coffs['J3f_add']['z2'])
# print ('J3_pm2 =', np.abs(coffs['J3f_add']['pm2'])*2, '/', coffs['J3f_add']['pm2']*2)
# print ('-2pi/3 =', -2*np.pi/3)
# print ('phi3_1 =', np.angle(coffs['J3f_add']['pm1'])-np.pi)
# print ('phi3_1 =', np.angle(coffs['J3f_add']['pm2'])-np.pi, '\n')

# print ('J4     =', coffs['J4'])

# quit()


# eU = 1.2949284816862714*1000
# eV = 0.1834008087182555*1000

# t1 = 1.812922916256524
# t2 = 0.26264121104892724
# t3 = 0.14491126516843533

# ep = eU/(75*t1)

# print ('eps =', ep, '\n')

# U = eU/ep
# V = eV/ep

# print ('U  =', U)
# print ('V  =', V)
# print ('t1 =', t1)
# print ('t2 =', t2)
# print ('t3 =', t3, '\n')


# coffs = coefficients(t1,t2,t3,U,V,phi1=2*np.pi/3,phi2=np.pi,phi3=np.pi/3,by_order=False)


# print ('J1 =', coffs['J1_z'])

