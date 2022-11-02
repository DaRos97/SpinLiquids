Compute the scaling of the gap for increasing system sizes.
Parameters are self-explixatory in the gap.py script. 
Takes, for given ansatz point in the phase diagram, the parameters at different system sizes (strarting from Nx=Ny=13) 
and computes the gap by first interpolating the lowest band and then taking the minimum of the interpolation. For this the value
can also be in some cases negative. 
Additionally, interpolates the points with f(x) = a*x**2 + b. In fact, the gap should fall as 1/L**2 for increasing
system sizes in case of LRO and should saturate to a finite value (a<0) in case of finite gap.


3/10/2022
Works fine for all ansatze and sizes. Still need to find an offset to define what is a finite gap at infinity.
Taking a > or < 0 is not enough (maybe?). 
