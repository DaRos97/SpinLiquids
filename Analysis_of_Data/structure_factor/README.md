Compute the spin structure factor for the results of the minimization.
Need to distinguish BEFORE if it is LRO or SL.
For LRO compute the condensate direction using the K points where the gap closes
and the corresponding eigenvectors to compute the spin directions at each site. Then,
the computation of the ssf is as in the classical case.
For SL use the derivation (analytical) which multiplies the Bogoliubov transformation
matrices in complicated ways.

28/08/2022
LRO works perfectly for 3x3_1, q0_1 and cb1 at all phases and S values. Maybe should check the ssf function because goes a bit too fast.
SL still at the old version, should adjorn it to be able to compute the recent data structures.

