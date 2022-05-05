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
print(map.grid)
print(map.grid.spacing)

#TODO — need Angstrom to voxel conversion
distance = 30 #still need to convert this into Angstroms

density = np.array(map.grid, copy=False)

x,y,z = np.indices(density.shape)

coord = np.array([9.458363688, 34.30846021, 35.98092784])/map.grid.spacing
coord = np.array([9.639798479316351, 34.343529790592804, 35.665502553889404])/map.grid.spacing

edf = np.sqrt((x-coord[2])**2 + (y-coord[1])**2 + (z-coord[0])**2)

mask = edf < distance

density_modified = density * mask

#writing out map
ccp4_out = gemmi.Ccp4Map()
ccp4_out.grid = gemmi.FloatGrid(density_modified)
ccp4_out.grid.unit_cell = map.grid.unit_cell
ccp4_out.grid.spacegroup = map.grid.spacegroup
ccp4_out.update_ccp4_header()
ccp4_out.write_ccp4_map('out.ccp4')

#plotting

x =  np.linspace(0,map.grid.unit_cell.a, num=density.shape[0], endpoint=False)
y =  np.linspace(0,map.grid.unit_cell.b, num=density.shape[1], endpoint=False)
xx,yy = np.meshgrid(x, y, indexing='ij')

plt.subplot(121)
plt.contour(xx,yy, density[:,:,36])

plt.subplot(122)
plt.contour(xx,yy, density_modified[:,:,36])
plt.show()


