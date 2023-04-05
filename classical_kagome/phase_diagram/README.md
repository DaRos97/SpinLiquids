Code for computing the CLASSICAL phase diagram of the Heisenberg model on the kagome lattice up to third nearest
neighbor interactions and with the inclusion of DM interactions. The possible configurations considered are regular
states (planar and chiral) as well as (hopefully) spiral states. It also computes the lower bound on the energy 
defined in Messio's paper on classical orders.

15/09/2022
The code runs and is exact for the phase diagram without DM interactions. Didn't try yet to change the orientations of the spins.
For the DM it looks wrong, because it seems that you have to implement also the DM-rotation on the spins (gauge) in order to get back
the original energies at *dm1 = 4pi/3*. But at what point should one implement this transformation?
Also, for small values of DM angle the 3x3 looks to get bigger while the q=0 should start to dominate, don't know why.

26/09/2022
Adjusted the code to do everything by itself (without problems regarding analytical derivations) by coding in directly the lattice directions and the Hamiltonian as a sum over all the bonds.
Added also all the gauge-transformed orders -> see the correct periodicity of energies by varying the DM angle (1nn).
Maybe problem with the ferromagnetic phase diagram since the cb2-gauged order dominated at DM1nn=pi/3 whereas in the AFM phase diagram we can see another periodicity in the energies. 
This problem may be related to a wrong energy in the *3x3_g1* order or in the cb2 lattice.

05/04/2022
All working. Use *all_plot.py* for SU(2) phase diagram and *all_plot_005.py* for the finite DM one. 
