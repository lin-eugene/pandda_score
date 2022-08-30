import gemmi
import pathlib
import scipy
import numpy as np
from lib import rmsdcalc

def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

def get_grid_box(map: gemmi.Ccp4Map, 
            residue: gemmi.Residue,
            box_size=10,
            spacing=0.5,
            rand_rot=False):
    
    #calculating centre of mass of residue
    CoM = rmsdcalc.calculate_CoM_residue(residue)
    CoM = CoM
    
    #computing translation vector and matrix
    vec = np.empty(3)
    vec.fill(box_size/2)
    mat = np.identity(3)*spacing

    if rand_rot==True:
        M = rand_rotation_matrix() #random rotation matrix in Cartesian basis
        mat = mat @ M

        vec = np.empty(3)
        vec.fill(box_size/2)
        vec = M @ vec
        
    min = CoM - vec
    
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
    map.grid.interpolate_values(arr,tr)

    return arr

def gen_map_from_atoms(residue: gemmi.Residue):
    #initialising list of positions and vdw radii
    positions = []
    radii = []

    #appending ligand positions/coordinates and vdw radii into lists
    for atom in residue:
        positions.append(atom.pos)
        radii.append(atom.element.vdw_r)

    map.grid.fill(0)
    for i in range(len(positions)):
        map.grid.set_points_around(positions[i],radius=radii[i],value=1)

    map.grid.symmetrize_max()

    return map

###

    