import numpy as np
import gemmi
<<<<<<< HEAD
import os
import matplotlib.pyplot as plt
<<<<<<< HEAD
from pathlib import Path
import shutil

def rmligdensity(filename_structure: str, filename_map: str, ccp4_out_name: str):
    """
    removes ligand density of a built structure
    filename_structure = pdb file with ligand built in
    filename_map = ccp4 file to modify
    ccp4_out_name = name of output ccp4 file
    """
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
=======
=======
import matplotlib.pyplot as plt
>>>>>>> 42bbaaaaaba6b70d0f5205a579df08ece5e8205d
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
<<<<<<< HEAD
>>>>>>> modified removeligdensity.py and test.py

    #writing output map
    ccp4_out = map
    ccp4_out.write_ccp4_map(ccp4_out_name)

<<<<<<< HEAD
=======
    #removing ligand electron density
    for i in range(len(positions)):
        map.grid.set_points_around(positions[i],radius=radii[i],value=0)
>>>>>>> modified removeligdensity.py and test.py
=======


    #removing ligand electron density
    for i in range(len(positions)):
        map.grid.set_points_around(positions[i],radius=radii[i],value=0)
>>>>>>> 42bbaaaaaba6b70d0f5205a579df08ece5e8205d

    #symmetrising map
    map.grid.symmetrize_min

<<<<<<< HEAD
<<<<<<< HEAD
###########
=======
    #writing output map
    ccp4_out = map
    ccp4_out.write_ccp4_map(ccp4_out_name)
>>>>>>> modified removeligdensity.py and test.py

if __name__ == "__main__":
    filename_structure = 'data/Namdinator test files/MURE/MUREECA-x0244good/MUREECA-x0244-ligandbuilt.pdb'
    filename_map = 'data/Namdinator test files/MURE/MUREECA-x0244good/Remodelling results/MUREECA-x0244-aligned-event_1.ccp4'
    ccp4_out_name = 'data/Namdinator test files/MURE/MUREECA-x0244good/out1.ccp4'

<<<<<<< HEAD
    rmligdensity(filename_structure, filename_map, ccp4_out_name)
=======
=======
    #writing output map
    ccp4_out = map
    ccp4_out.write_ccp4_map(ccp4_out_name)


>>>>>>> 42bbaaaaaba6b70d0f5205a579df08ece5e8205d
filename_structure = 'data/Namdinator test files/MURE/MUREECA-x0244good/MUREECA-x0244-ligandbuilt.pdb'
filename_map = 'data/Namdinator test files/MURE/MUREECA-x0244good/Remodelling results/MUREECA-x0244-aligned-event_1.ccp4'
ccp4_out_name = 'out1.ccp4'

rmligdensity(filename_structure, filename_map, ccp4_out_name)
<<<<<<< HEAD
>>>>>>> modified removeligdensity.py and test.py
=======
>>>>>>> 42bbaaaaaba6b70d0f5205a579df08ece5e8205d


