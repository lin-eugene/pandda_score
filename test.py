import numpy as np
import gemmi
import matplotlib.pyplot as plt

"""
Removing fragment density pseudocode
1. Read in electron density map
    https://gemmi.readthedocs.io/en/latest/grid.html#map-from-grid — Map from grid,Example 1
2. Find coordinates of where ligand should be
3. Mask out volume (1å radius from ligand coordinate)
    Create new map with same dimensions as electron density map
    Compute Euclidean distance transform (with ligand coordinate as centre) 
    Boolean — if EDT < 1, Mask = 0
    Multiply Boolean with electron density map
    Write out electron density map
"""

filename_structure = 'BAZ2BA-x481/BAZ2BA-x481-pandda-input-coot-1.pdb'
filename_map = 'BAZ2BA-x481/BAZ2BA-x481-event_1_1-BDC_038_map,native.ccp4'

structure = gemmi.read_structure(filename_structure)
map = gemmi.read_ccp4_map(filename_map,setup=True)

print(map.grid.spacegroup)

def remove_density(x,y,z,map):
    """
    Masks out electron density at specified Cartesian coordinates
    x,y,z — coordinates
    map — gemmi map class
    """
    coord = gemmi.Position(x, y, z)
    map.grid.set_points_around(coord,radius=1,value=0)
    map.grid.symmetrize_min

remove_density(14.49,39.87,29.69,map)


ccp4_out = map
ccp4_out.write_ccp4_map('out.ccp4')

# mask = map.grid.clone()
# mask.fill(0)
# mask.set_points_around(coord,radius=10,value=1)


# # x,y,z = np.indices(map.grid.shape)
# # edf = np.sqrt((x-coord[0])**2 + (y-coord[1])**2 + (z-coord[1])**2) #compute Euclidean distance field

# # mask = edf < distance #mask excluding region 

# density_modified = map.grid.array * mask.array

# #writing out map
# ccp4_out = gemmi.Ccp4Map()
# ccp4_out.grid = gemmi.FloatGrid(density_modified)
# ccp4_out.grid.unit_cell = map.grid.unit_cell
# ccp4_out.grid.spacegroup = map.grid.spacegroup
# ccp4_out.update_ccp4_header()
# ccp4_out.write_ccp4_map('out.ccp4')

# #plotting

# x =  np.linspace(0,map.grid.unit_cell.a, num=density.shape[0], endpoint=False)
# y =  np.linspace(0,map.grid.unit_cell.b, num=density.shape[1], endpoint=False)
# xx,yy = np.meshgrid(x, y, indexing='ij')

# plt.subplot(121)
# plt.contour(xx,yy, density[:,:,36])

# plt.subplot(122)
# plt.contour(xx,yy, density_modified[:,:,36])
# plt.show()


