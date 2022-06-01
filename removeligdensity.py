import numpy as np
import gemmi
import matplotlib.pyplot as plt

def remove_density(x,y,z,map):
    """
    Masks out electron density at specified Cartesian coordinates
    x,y,z — coordinates
    map — gemmi map class
    """
    coord = gemmi.Position(x, y, z)
    map.grid.set_points_around(coord,radius=1,value=0)
    map.grid.symmetrize_min

filename_structure = 'data/Namdinator test files/MURE/MUREECA-x0244good/MUREECA-x0244-ligandbuilt.pdb'
filename_map = 'data/Namdinator test files/MURE/MUREECA-x0244good/Remodelling results/MUREECA-x0244-aligned-event_1.ccp4'

structure = gemmi.read_structure(filename_structure)
map = gemmi.read_ccp4_map(filename_map,setup=True)
# print(structure[0])
# print(structure[0][0])
# print(structure[0][0][0])
# print(structure[0][0][0][0].pos)
# print(structure[0][0][0][0].element.vdw_r)
# print(dir(structure[0][0][0][0]))
# print(dir(structure))


positions = []
radii = []


for model in structure:
    for chain in model:
        for residue in chain:
            if residue.name == 'LIG':
                for atom in residue:
                    positions.append(atom.pos)
                    radii.append(atom.element.vdw_r)


for i in range(len(positions)):
    map.grid.set_points_around(positions[i],radius=radii[i],value=0)

map.grid.symmetrize_min

ccp4_out = map
ccp4_out.write_ccp4_map('out1.ccp4')