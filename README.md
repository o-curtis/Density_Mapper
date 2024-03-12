# Density_Mapper
Takes in stellar particle data and creates 2D density maps using a cubic spline kernel.

Density_Mapper.py is the primary implementation of the Density_Mapper class. 

The class has inputs
    particle_coords - [N,2] array - particle coordinates projected along one axis
    N_cells_per_side- int - number of cells along one axis
    cell_length - float - physical length of each cell in the same units as particle_coords
    h_vals - [N,] array - softening length of each particle to be provided to the cubic spline fit
    star_mass - [N,] array - stellar masses of the particles in units h^-1 M_\odot

that instantiates the class, and the function assign_particles_to_grid() that creates the 2D density map.

Currently, a cubic spline kernel is the default option but one can change that to a CIC kernel
by editing line 72. In the future, I will generalize this so uses can select which
kernel they want to use when calling assign_particles_to_grid().

This code was used to create the density maps studied in
  McDonough, B., et al., "Resolved Star Formation in TNG100 Central and Satellite Galaxies." The Astrophysical Journal 958.1 (2023): 19.
  
If you use this code then please cite the above paper.

I have included the example script "make_D4000_parts_maps.py" that was used to take in the subhalo catalog TNG100 along with
all D4000 stellar particles identified in the simulation and make stellar mass density maps. Unfortunately, these files 
are too large to make a proper example, but one can use it as a reference for how to load in galaxy and particle data,
find particles associated with a galaxy, project the particles along an axis, instantiate the grid, create a 
Density_Mapper object, and run the cubic spline fitting.

Dependencies
-numba
-numpy
-h5py
-glob
