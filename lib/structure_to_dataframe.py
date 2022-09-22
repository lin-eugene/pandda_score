import pandas as pd
import gemmi
import itertools

__all__ = ['structure_to_dataframe']

def list_residues_in_chain(chain):
    residue_name = []
    residues_idx = []

    for i, residue in enumerate(chain):
        if residue.het_flag == 'A':
            residue_name.append(str(residue))
            residues_idx.append(i)
    
    return residue_name, residues_idx

def list_residues_in_structure(structure):
    chain_idx = []
    residues_idx = []
    residue_name = []
    for i, chain in enumerate(structure):
        res_name_per_chain, res_idx_per_chain = list_residues_in_chain(chain)

        chain_idx += [i] * len(res_name_per_chain)
        residues_idx += res_idx_per_chain
        residue_name += res_name_per_chain
    
    return chain_idx, residues_idx, residue_name

def record_per_residue_data(event_map_path, structure_path, chain_idx, residue_idx, residue_name):
   
    return {
        'event_map_path': event_map_path,
        'structure_path': structure_path,
        'chain_idx': chain_idx,
        'residue_idx': residue_idx,
        'residue_name': residue_name,
    }

def structure_to_dataframe(event_map_path, structure_path):
    structure = gemmi.read_structure(structure_path)[0]

    lists = list_residues_in_structure(structure)
    record = record_per_residue_data(itertools.repeat(event_map_path, len(lists[0])), 
                                    itertools.repeat(structure_path, len(lists[0])), 
                                    *lists)

    df = pd.DataFrame(record)

    return df

####

if __name__ == '__main__':
    import os
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Convert structure to dataframe')
    parser.add_argument('--event_map_path', type=str, required=True)
    parser.add_argument('--structure_path', type=str, required=True)
    # parser.add_argument('--output_path', type=str, required=True)

    args = parser.parse_args()

    df = structure_to_dataframe(args.event_map_path, args.structure_path)
    print(df)

    print('Done')