import numpy as np
import gemmi
import matplotlib.pyplot as plt
import pathlib
import shutil

def rmligdensity(filename_structure, filename_map, ccp4_out_name):
    structure = gemmi.read_structure(filename_structure)
    map = gemmi.read_ccp4_map(filename_map,setup=True)

    #initialising list of positions and vdw radii
    positions = []
    radii = []

    #appending ligand positions/coordinates and vdw radii into lists
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name == 'LIG':
                    for atom in residue:
                        positions.append(atom.pos)
                        radii.append(atom.element.vdw_r)


    #removing ligand electron density
    for i in range(len(positions)):
        map.grid.set_points_around(positions[i],radius=radii[i],value=0)

    #symmetrising map
    map.grid.symmetrize_min

    #writing output map
    ccp4_out = map
    ccp4_out.write_ccp4_map(ccp4_out_name)


filename_structure = 'data/Namdinator test files/MURE/MUREECA-x0244good/MUREECA-x0244-ligandbuilt.pdb'
filename_map = 'data/Namdinator test files/MURE/MUREECA-x0244good/Remodelling results/MUREECA-x0244-aligned-event_1.ccp4'
ccp4_out_name = 'out1.ccp4'

rmligdensity(filename_structure, filename_map, ccp4_out_name)


