Code for Schwinger Boson paper on kagome lattice with 3rd nn and DM

Requirements: numpy, scipy, matplotlib, tqdm

Directories:
- Analysis_of_Data: Codes for analazing the data produced in the gradient descent method
- bash_codes: some reference bash codes for copy-pasting to and from the cluster
- classical_kagome: code for studying the classical model.
    - new_PD: old code for phase diagram
    - pairings_hoppings: code for computig classical values of phases of pairing and hopping parameters for the various classical regular orders.
    - PD_J3p: old code for phase diagram
    - phase_diagam: real code for evaluating the phase diagram of the classical model. It computes all regular orders as well as spiral states. Also good plots implemented.
    - structure_factor: code for determinign the spin structure factor of classical regular orders and gauged versions.
- Data: directory of saved data for phase diagrams obtained in the gradient descent method.
    - final_: all directories containing the final value of the minimization with the value True/False for the convergence and O/L for Ordered or (spin) Liquid phase (finite size scaling aready checked)
    - S: All raw data of minimization without finite siza scaling, classified first by spin (50,36,..), then DM phase (000,005,..) then system size (13,25,..)
    - SDM: all data as above but for the S-DM plot
    - self_consistency: data organized in the same way as above BUT for the self-consistent method -> results used in the paper.
        - S: as above.
        - SDM: as above.
        - test: raw data.
        - UV: a tentaive UV calculation (Hubbard model parameters, as the plot in DMRG paper).
        - free_: data obtained by keeping all parameters free
        - 440_small_S50: data for small phase diagram centered in the origin (+-0.03 in J2 and J3).
        - data_960.tar.gz: compressed dir of same as above but finer grid.
- Figures_temp: some random figures obtained during the process
- minimization_SU2: code for gradient descent minimization for SU(2) invariant points.
- minimization_TMD: code for gradient descent minimization for TMD not trivial, ansatze are different essentially.
- SDM_diagram: code for SDM diagram using gradient descent.
- self_consistency: actual code used for the paper results with self consistent method of minimization. 
    - analysis_: codes for analysis of J2-J3 (free), S-DM (SDM) and U-V (uv) phase diagrams. Takes data obtained before in the minimization and uses it to compute the phase diagram (compare energies) and plot it, compute the gap and perform the finite size scaling, and others.
    - free_PD: code for actual J2-J3 phase diagram
    - PD: empty -> did I lose something?
    - SDM: code for S-DM phase diagram.
    - structure_factor_: code for computing structure factor both in gapped and gapless solutions (two methods) for J2-J3 (free), SDM and uv data.
    - test: some test code.
    - uv_plot: code for U-V phase diagram.
- U_to_V: code to convert U to V for the U-V phase diagram.
- Various_Steps: Story of old codes used during the project, from the only 1st nn to older versions of minimization prcedure.
