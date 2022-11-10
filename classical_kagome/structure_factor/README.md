Code for computing the spin structure factor of classical orders, also with the inclusion
of "DM-rotation" directly into the spin configuration.


14/09/2022
Finished the code for the Cuboc-2 order to verify the spin structure factor obtained with DMRG. The Figures folder is a bit useless and I didn't come out witha smart naming of the file. 
The code works only for the cuboc-2 , need to add the other orthers in the functions.py script. 
The function which computes the structure factor is slow since it goes brute force over all spins, surely can be made faster. 

27/09/2022
Carefull to the lattice vectors which are right and pos-right!!!! The lattice definition has to be done accordingly.
Works for all orders and also gauge_trsf (not tested)

28/09/2022
Fully working and checked for gauged transformations and rotation angles.
Changed the direction of the unit vectors to be consistent with the rest of the code.
