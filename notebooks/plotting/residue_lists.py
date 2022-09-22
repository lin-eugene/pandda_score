def residue_list_sorted_by_chem_similarity():

    residues = ['ALA', 'VAL', 'ILE', 'LEU', 
                'MET' , 'GLY', 'PRO', 'CYS',
                'HIS', 'PHE', 'TYR', 'TRP',
                'ASN', 'GLN', 'SER', 'THR', 
                'LYS', 'ARG', 'GLU', 'ASP']                

    return residues

def residue_list_sorted_by_molweight() -> list[str]:
    residues_sorted_molweight = ['GLY', 'ALA', 'SER', 'PRO',
                        'VAL', 'THR', 'CYS', 'ILE',
                        'LEU', 'ASN', 'ASP', 'GLN', 
                        'LYS', 'GLU', 'MET', 'HIS', 
                        'PHE', 'ARG', 'TYR', 'TRP']
    return residues_sorted_molweight