"""
Implemention of the the Density_Mapper class on TNG100 data.

This code was used to create the density maps studied in
  McDonough, B., et al., "Resolved Star Formation in TNG100 Central and Satellite Galaxies." The Astrophysical Journal 958.1 (2023): 19.
  
If you use this code then please cite the above paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
from numba import int32, float64, float32, double    # import the types
from numba.experimental import jitclass
from numba import jit

#instantiate variables
spec = [
    ('particle_coords', float64[:,:]),
    ('N_cells_per_side', int32),
    ('cell_length', float64),
    ('h_vals', float64[:]),
    ('star_mass', float32[:]),
    ('cell_length', float32),
    ('particle_mesh', float64[:,:]),
    ]

@jitclass(spec)
class Density_Mapper():
    
    def __init__(self, particle_coords, N_cells_per_side, cell_length, h_vals, star_mass):
        """
        Main density mapper class.
        Inputs
          particle_coords - [N,2] array - particle coordinates projected along one axis
          N_cells_per_side- int - number of cells along one axis
          cell_length - float - physical length of each cell in the same units as particle_coords
          h_vals - [N,] array - softening length of each particle to be provided to the cubic spline fit
          star_mass - [N,] array - stellar masses of the particles in units h^-1 M_\odot
        """
        self.particle_coords = particle_coords
        self.N_cells_per_side = N_cells_per_side
        self.cell_length = cell_length
        self.h_vals = h_vals
        self.star_mass = star_mass
        self.particle_mesh = np.zeros((self.N_cells_per_side,self.N_cells_per_side))
    
    def assign_particles_to_grid(self):
        """
        Assigns particles to a grid using either a CIC or cubic spline kernel (default).
        
        self.particle_mesh is a grid of size [N_cells_per_side, N_cells_per_side] containing the 
          local matter density within that cell.
        """
        for particle in range(self.particle_coords.shape[0]):
            nearest_cell_x_coord = (int(np.floor(self.particle_coords[particle][0]/self.cell_length))%self.N_cells_per_side) - 1
            nearest_cell_y_coord = (int(np.floor(self.particle_coords[particle][1]/self.cell_length))%self.N_cells_per_side) - 1
            for p_x in range(nearest_cell_x_coord - 5, nearest_cell_x_coord + 5):
                this_p_x = p_x%self.N_cells_per_side
                for p_y in range(nearest_cell_y_coord - 5, nearest_cell_y_coord + 5):
                    this_p_y = p_y%self.N_cells_per_side
                    input_x = self.particle_coords[particle][0] - this_p_x * self.cell_length
                    input_y = self.particle_coords[particle][1] - this_p_y * self.cell_length
                    input_r = np.sqrt(input_x**2 + input_y**2)
                    this_h = self.h_vals[particle]
                    this_mass = self.star_mass[particle]
                    #todo: generalize this so people can use either a CIC kernel (w below)
                    # or the cubic spline kernel (w2 below) 
                    # as an input to Density_Mapper
                    self.particle_mesh[this_p_x, this_p_y] += this_mass*w2(input_r, this_h)

#CIC kernel
@jit(nopython=True)
def w(x, delta):
    if np.abs(x) < delta:
        return 1 - (np.abs(x)/delta)
    else:
        return 0

#cubic spline kernel
@jit(nopython=True)
def w2(r, h):
    q=r/h
    constant = 4*10/(7*np.pi*h**2)
    if (q>=0) and (q<=1/2):
        return constant*(1 - 6*q**2 + 6*q**3)
    if (q>1/2) and (q<=1):
        return constant * 2 * (1-q)  **3
    if (q>1):
        return 0