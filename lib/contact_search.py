import gemmi
from lib import rmsdcalc

def find_contacts_per_residue(structure: gemmi.Structure, residue: gemmi.Residue, radius=5):
    """function finding contacts close to specific residue"""

    CoM = rmsdcalc.calculate_CoM_residue(residue)
    point = gemmi.Position(CoM[0], CoM[1], CoM[2])

    contacts = []
    ns = gemmi.NeighborSearch(structure[0], structure.cell, radius).populate(include_h=False)
    marks = ns.find_atoms(point, '\0', radius=radius)

    for mark in marks:
        cra = mark.to_cra(structure[0])
        if (cra.residue.name != residue.name) and (cra.residue.het_flag == 'A') \
            and ([mark.chain_idx, mark.residue_idx, str(cra.residue)] not in contacts):
            contacts.append([mark.chain_idx, mark.residue_idx, str(cra.residue)])
    
    return contacts