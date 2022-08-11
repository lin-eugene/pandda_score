import pathlib
import gemmi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

def map_chains(input_model: gemmi.Model, output_model: gemmi.Model):  
    """
    maps/pairs chains together according to chain centre of mass
    returns list with mapped chain pairs
    """
    chain = []
    chain_id = []
    #renaming chains in two models to match each other if they are aligned
    print('mapping chains')
    for i, chain_gt in enumerate(input_model):
        for j, chain_rsr in enumerate(output_model):
            gt_CoM = chain_gt.calculate_center_of_mass() #gemmi.Position
            rsr_CoM = chain_rsr.calculate_center_of_mass() #gemmi.Position

            dist = gt_CoM.dist(rsr_CoM)
            # print(dist)
            # print('chain_gt=',chain_gt.name)
            # print('chain_rsr=',chain_rsr.name)
            if dist < 1:
                #chain_rsr.name = chain_gt.name
                chain.append([chain_gt,chain_rsr])
                chain_id.append([i,j])
    print(chain_id)
    return chain, chain_id

def superpose(polymer1: gemmi.ResidueSpan, polymer2: gemmi.ResidueSpan):
    """
    superposes two gemmi polymer objects
    """
    print('superposing')
    ptype = polymer1.check_polymer_type()
    sup = gemmi.calculate_superposition(polymer1, polymer2, ptype, gemmi.SupSelect.All)
    polymer2.transform_pos_and_adp(sup.transform)

    return polymer2

def calculate_CoM_residue(residue: gemmi.Residue):
    """
    calculates the centre of mass for a single residue
    """
    coords = []
    mass = []
    for atom in residue.first_conformer(): #just looking at one conformer
        coords.append(atom.pos.tolist())
        mass.append(atom.element.weight)
    
    coords = np.asarray(coords)
    mass = np.asarray(mass)
    mass = mass.reshape(-1,1) #convert into column vector

    CoM = np.sum((mass * coords),axis=0) / np.sum(mass) # calculating centre of mass
    #print(CoM) 

    return CoM

def calc_dist_diff(polymer1: gemmi.ResidueSpan, polymer2: gemmi.ResidueSpan):
    """
    calculates centre of mass difference for each residue in two protein chains
    """
    print('calculating rmsds')
    dist_diff = {}

    for residue1 in polymer1:
        for residue2 in polymer2:
            if (str(residue1) == str(residue2)) and (residue1.het_flag == 'A' and residue2.het_flag == 'A'):
                com1 = calculate_CoM_residue(residue1)
                com2 = calculate_CoM_residue(residue2)
                calc_dist = np.linalg.norm(com1-com2) #calculate euclidean dist between 2 CoMs

                dist_diff[str(residue1)] = calc_dist
    
    return dist_diff


def calc_rmsds(input_pdb_name: pathlib.PosixPath, remodelled_pdb_name: pathlib.PosixPath):
    input_pdb = gemmi.read_structure(str(input_pdb_name))
    remodelled_pdb = gemmi.read_structure(str(remodelled_pdb_name))

    model_input = input_pdb[0]
    model_remodelled = remodelled_pdb[0]

    chain, chain_id = map_chains(model_input, model_remodelled)
    #print(chain)

    rmsd_dict = {}
    for i, pair in enumerate(chain):
        # polymer1 = pair[0].get_polymer()
        # polymer2 = pair[1].get_polymer()
        # # print(polymer1)
        # # print(polymer2)
        # polymer2 = superpose(polymer1, polymer2)
        dist_diff = calc_dist_diff(pair[0], pair[1])
        rmsd_dict[pair[0].name] = dist_diff

    return rmsd_dict

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

