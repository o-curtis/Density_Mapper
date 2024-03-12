"""
Example code implementing the Density_Mapper class on TNG100 data.

The code will create a stellar mass map of all galaxies in TNG100 using
D4000 particles.

This code was used to create the density maps studied in
  McDonough, B., et al., "Resolved Star Formation in TNG100 Central and Satellite Galaxies." The Astrophysical Journal 958.1 (2023): 19.
  
If you use this code then please cite the above paper.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from Density_Mapper import *

if __name__ == '__main__':
    
    PATH_TO_FILES = '/projectnb/gravlens/bnmcd/SFR/catalogs/D4000_parts/'
    #all_files = glob.glob(PATH_TO_FILES+"270967.pkl")
    all_files=['/projectnb/gravlens/bnmcd/SFR/catalogs/boundparts/D4000_parts/270967.pkl']
    
    #for each subhalo
    for this_pkl in range(len(all_files)):
    
        #load the subhalo data and get the necessary fields
        this_subid = all_files[this_pkl][62:-4]
        this_file = np.load(all_files[this_pkl], allow_pickle=True)
        star_pos = this_file['starpos'][0]
        flatten_on=this_file['flatten_on'] #primary axis to project particles along
        star_mass =this_file['starmass']
        
        #load all of the stellar particles
        #find stellar particles associated with this galaxy out to maxrad
        #define parameters that will be fed to Density_Mapper
        #  Pull the positions and softening values (h) associated with each particle.
        full_particle_info = np.load('density_maps/boundparts/particle_info_subID_'+str(this_subid)+'.npy')
        full_particle_pos = full_particle_info[:,0:3]
        full_particle_h = full_particle_info[:,3]
        maxrad = full_particle_info[0,5]
        full_particle_pos = np.around(full_particle_pos, 6)
        star_pos = np.around(star_pos, 6)
        h_vals = np.zeros_like(star_mass)
        #Only use stellar particles within maxrad of the galaxy's center
        pos_mask = (star_pos[:,0]>=-1*maxrad) & (star_pos[:,0]<=maxrad) & (star_pos[:,1]>=-1*maxrad) & (star_pos[:,1]<=maxrad) & (star_pos[:,2]>=-1*maxrad) & (star_pos[:,2]<=maxrad)
        star_pos = star_pos[pos_mask]
        star_mass = star_mass[pos_mask]
        for this_particle_ix in range(len(star_pos)):
            this_particle_ix_in_full = np.where (full_particle_pos==star_pos[this_particle_ix])
            this_particle_ix_in_full = this_particle_ix_in_full[0][0]
            h_vals[this_particle_ix] = full_particle_h[this_particle_ix_in_full]

        #if there are no stars in this galaxy, continue
        if len(star_pos)==0:
            np.save("density_maps/D4000_parts/stellar_mass_density_map_subID_"+str(this_subid), [])
            continue

        #creates 2D projections of the data along specified coordinates.
        if flatten_on==0:
            star_pos_flattened_axes = star_pos[:,1:3]
            full_first_axis_min = np.min(full_particle_pos[:,1])
            full_second_axis_min= np.min(full_particle_pos[:,2])
        if flatten_on==1:
            first_axis = star_pos[:,0]
            second_axis= star_pos[:,2]
            star_pos_flattened_axes = np.column_stack((first_axis, second_axis))
            full_first_axis_min = np.min(full_particle_pos[:,0])
            full_second_axis_min= np.min(full_particle_pos[:,2])
        if flatten_on==2:
            first_axis = star_pos[:,0]
            second_axis= star_pos[:,1]
            star_pos_flattened_axes = np.column_stack((first_axis, second_axis))
            full_first_axis_min = np.min(full_particle_pos[:,0])
            full_second_axis_min= np.min(full_particle_pos[:,1])

        #Finalizing preprocessing of particles and defining more parameters to be fed to Density_Mapper
        #namely getting softening lengths 
        h_vals = h_vals.astype(np.float64)
        first_axis_min = np.min(star_pos_flattened_axes[:,0])
        second_axis_min= np.min(star_pos_flattened_axes[:,1])
        star_pos_flattened_axes[:,0] += np.abs(full_first_axis_min) #shift entire galaxy to quadrant I
        star_pos_flattened_axes[:,1] += np.abs(full_second_axis_min)
        grid_length = 0.5 #h-1 Kpc
        grid_no = int(1+(2*maxrad/grid_length))
        gal_center_in_map_first_ax = (int(np.floor(np.abs(full_first_axis_min)/grid_length))%grid_no)
        gal_center_in_map_second_ax = (int(np.floor(np.abs(full_second_axis_min)/grid_length))%grid_no)
    
        #creates a Density_Mapper object as well as a density map constructed with a cubic spline kernel.
        this_Mapping = Density_Mapper(star_pos_flattened_axes, grid_no, grid_length, h_vals, star_mass)
        this_Mapping.assign_particles_to_grid()
    
        #converts the density map into a total stellar mass map
        density_mesh = this_Mapping.particle_mesh
        total_stellar_mass = density_mesh*grid_length*grid_length*10**10
        total_stellar_mass = np.log10(total_stellar_mass)
        
        #save the density map
        np.save("density_maps/D4000_parts/stellar_mass_density_map_subID_"+str(this_subid), density_mesh)
        
        #plots the stellar mass map.
        fig,ax = plt.subplots(1,1)
        plot = ax.imshow(total_stellar_mass, vmin=5, vmax=8.0, cmap='plasma')#, vmin=0.0001, vmax=0.05)#, vmax=35)
        cbar = plt.colorbar(plot)
        cbar.set_label(r"Total Stellar Mass [log($\rmM_*/h^{-1}M_\odot$)]")
        plt.show()
        exit()