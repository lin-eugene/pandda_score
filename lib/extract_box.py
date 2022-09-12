import gemmi
import numpy as np
from lib import rmsdcalc
from typing import Optional, Tuple

def gen_two_fo_fc_from_mtz(mtz_path: str) -> type[gemmi.FloatGrid]:
    mtz = gemmi.read_mtz_file(mtz_path)
    two_fofc_grid = mtz.transform_f_phi_to_map('FWT', 'PHWT')
    
    return two_fofc_grid

def fetch_grid_from_pandda_map(map: type[gemmi.Ccp4Map]) -> type[gemmi.FloatGrid]:
    """
    reads in map
    modifies space group to P1 in header
    returns gemmi grid object

    """
    grid = map.grid
    grid.spacegroup = gemmi.find_spacegroup_by_name('P 1')
    map.setup()
    
    return grid

def gen_mask_from_atoms(grid_from_experimental_map: type[gemmi.FloatGrid], 
                residue: type[gemmi.Residue],
                mode='radius_1_Ang') -> type[gemmi.FloatGrid]:
    """
    grid — map.grid object from ccp4 map (e.g. pandda event map)
    """

    mask_grid = grid_from_experimental_map
    #initialising list of positions and vdw radii
    positions = []
    radii = []

    #appending ligand positions/coordinates and vdw radii into lists
    for atom in residue:
        positions.append(atom.pos)
        radii.append(atom.element.vdw_r)

    mask_grid.fill(0)

    #each branch should be separate function
    #control should be moved to main function — i.e. at highest level of control
    if mode == 'radius_1_Ang': 
        for position in positions:
            mask_grid.set_points_around(position,radius=1,value=1)
    else:
        for position, radius in zip(positions, radii):
            mask_grid.set_points_around(position,radius=radius,value=1)

    mask_grid.symmetrize_max() 

    return mask_grid

#move rand_rotations and rand_translations into separate file

def generate_gemmi_transform(residue: type[gemmi.Residue], 
            box_size=10,
            spacing=0.5,
            rot_mat: Optional[np.ndarray]=None,
            vec_rand: Optional[np.ndarray]=None
            ) -> Tuple[np.ndarray, type[gemmi.Transform]]:
    
    #calculating centre of mass of residue
    centre_of_mass, _ = rmsdcalc.calculate_com_residue(residue)
    if vec_rand is not None:
        centre_of_mass = centre_of_mass + vec_rand
    
    #computing translation vector and rotation matrix
    vec = np.empty(3)
    vec.fill(box_size/2)
    mat = np.identity(3)*spacing

    if rot_mat is not None:
        mat = mat @ rot_mat
        vec = rot_mat @ vec
        
    min = centre_of_mass - vec
    
    min_list = min.tolist()
    mat = mat.tolist()

    tr = gemmi.Transform()
    tr.vec.fromlist(min_list)
    tr.mat.fromlist(mat)

    #interpolating grid
    gridsize = box_size #in Ang
    npoints = int(gridsize / spacing)
    grid_shape = (npoints,npoints,npoints)
    arr = np.zeros(grid_shape, dtype=np.float32)

    return arr, tr

def create_numpy_array_with_gemmi_interpolate(
    residue: type[gemmi.Residue],
    grid: type[gemmi.FloatGrid],
    rot_mat: Optional[np.ndarray]=None,
    vec_rand: Optional[np.ndarray]=None,
    box_size: float = 10,
    spacing: float = 0.5) -> np.ndarray:
    
    array, transform = generate_gemmi_transform(residue, 
                                            rot_mat=rot_mat, 
                                            vec_rand=vec_rand,
                                            box_size=box_size,
                                            spacing=spacing)
    grid.interpolate_values(array, transform)

    return array

def make_gemmi_zeros_float_grid(grid: type[gemmi.FloatGrid]) -> type[gemmi.FloatGrid]:
    zeros_grid = gemmi.FloatGrid(np.zeros(grid.shape, dtype=np.float32),
                                grid.unit_cell, 
                                grid.spacegroup)
    
    return zeros_grid

def copy_gemmi_float_grid(grid: type[gemmi.FloatGrid]) -> type[gemmi.FloatGrid]:
    arr = grid.get_subarray(0,0,0,grid.shape[0],grid.shape[1],grid.shape[2])
    grid_copy = gemmi.FloatGrid(arr,
                                grid.unit_cell,
                                grid.spacegroup)
    
    return grid_copy

###


if __name__ == "__main__":
    pass