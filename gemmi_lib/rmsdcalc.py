import pathlib
import gemmi
import numpy as np
from typing import Union, Dict

def map_chains(input_model: gemmi.Model, output_model: gemmi.Model):  
    """
    maps/pairs chains together according to chain centre of mass
    returns list with mapped chain pairs
    """
    chain = []
    chain_idx = []
    #renaming chains in two models to match each other if they are aligned
    print('mapping chains')
    for i, chain_input in enumerate(input_model):
        for j, chain_output in enumerate(output_model):
            input_chain_centre_of_mass = calculate_com_chain(chain_input) #np.array
            output_chain_centre_of_mass = calculate_com_chain(chain_output) #np.array

            dist = np.linalg.norm(input_chain_centre_of_mass-output_chain_centre_of_mass) #calculate euclidean dist between 2 CoMs

            if dist < 1:
                chain.append([chain_input,chain_output])
                chain_idx.append([i,j])

    return chain, chain_idx

def superpose(polymer1: gemmi.ResidueSpan, polymer2: gemmi.ResidueSpan):
    """
    superposes two gemmi polymer objects
    """
    print('superposing')
    ptype = polymer1.check_polymer_type()
    sup = gemmi.calculate_superposition(polymer1, polymer2, ptype, gemmi.SupSelect.All)
    polymer2.transform_pos_and_adp(sup.transform)

    return polymer2

def read_coords_and_mass_of_atom(atom: gemmi.Atom) -> np.ndarray:
    """
    reads the coordinates and mass of a single atom
    """
    coords = atom.pos.tolist()
    mass = atom.element.weight

    return coords, mass

def calculate_com_residue(residue: gemmi.Residue) -> np.ndarray:
    """
    calculates the centre of mass for a single residue
    only looks at one conformer
    """

    coords_mass_list = list(zip(*map(
                                read_coords_and_mass_of_atom, 
                                residue.first_conformer() # only looking at first conformer of residue
                                )))

    coords = np.array(coords_mass_list[0])
    mass = np.array(coords_mass_list[1])

    mass = mass.reshape(-1,1) #convert into column vector

    centre_of_mass = np.sum((mass * coords),axis=0) / np.sum(mass) # calculating centre of mass

    return centre_of_mass, np.sum(mass)

def calculate_com_chain(chain: gemmi.Chain) -> np.ndarray:
    """
    calculates the centre of mass for a single chain
    """
    coords = []
    mass = []
    coords_mass_list = list(zip(*map(
                                calculate_com_residue, 
                                chain)))     
    
    coords = np.array(coords_mass_list[0])
    mass = np.array(coords_mass_list[1])
    mass = mass.reshape(-1,1)
    
    centre_of_mass = np.sum((mass * coords),axis=0) / np.sum(mass) # calculating centre of mass

    return centre_of_mass



#######

def calc_dist_diff(polymer1: Union[type[gemmi.Chain], type[gemmi.ResidueSpan]], 
                    polymer2: Union[type[gemmi.Chain], type[gemmi.ResidueSpan]]) -> Dict[str,float]:
    """
    calculates centre of mass difference for each residue in two protein chains
    """
    print('calculating rmsds')
    dist_diff = {}

    for residue1 in polymer1:
        for residue2 in polymer2:
            if (str(residue1) == str(residue2)) and (residue1.het_flag == 'A' and residue2.het_flag == 'A'):
                com1 = calculate_com_residue(residue1)
                com2 = calculate_com_residue(residue2)
                calc_dist = np.linalg.norm(com1-com2) #calculate euclidean dist between 2 CoMs

                dist_diff[str(residue1)] = calc_dist
    
    return dist_diff


def calc_rmsds(input_pdb_name: pathlib.PosixPath, remodelled_pdb_name: pathlib.PosixPath):
    input_pdb = gemmi.read_structure(str(input_pdb_name))
    remodelled_pdb = gemmi.read_structure(str(remodelled_pdb_name))

    model_input = input_pdb[0]
    model_remodelled = remodelled_pdb[0]

    chain, chain_mapping = map_chains(model_input, model_remodelled)
    #print(chain)

    rmsd_dict = {}
    chain_mapping_filt = []
    for pair, chain_id in zip(chain, chain_mapping):
        dist_diff = calc_dist_diff(pair[0], pair[1])
        
        if bool(dist_diff)==True: #if dist_diff dictionary is not empty
            rmsd_dict[pair[0].name] = dist_diff
            chain_mapping_filt.append(chain_id)

    return rmsd_dict, chain_mapping_filt

def find_remodelled(rmsd_dict: dict, thresh: float):
    bool_dict = {}

    for chain in rmsd_dict.items():
        bool_res = {}
        chain_name = chain[0]
        for res in chain[1].items():
            res_name = res[0]
            bool_res[res_name]= (res[1] > thresh)
        
        bool_dict[chain_name] = bool_res
    
    return bool_dict

