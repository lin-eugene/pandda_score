from typing import Dict
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pathlib
import sys
import gemmi

codebase_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(codebase_path))
from lib import extract_box

class ResidueDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.residues_dframe = pd.read_csv(csv_file)
        self.transform=transform
        
    def __len__(self):
        return len(self.residues_dframe)
    
    def __getitem__(self,idx) -> Dict:
        """
        items - (1) event map, (2) 2fo-fc map, (3) input structure, (4) output structure at specified residue
        event map and 2fo-fc map at residue at input position
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        event_map_name = self.residues_dframe['event_map'][idx]
        mtz_name = self.residues_dframe['mtz'][idx]
        input_structure_name = self.residues_dframe['input_model'][idx]
        output_structure_name = self.residues_dframe['output_model'][idx]
        input_chain_idx = self.residues_dframe['input_chain_idx'][idx]
        input_residue_idx = self.residues_dframe['residue_input_idx'][idx]
        output_chain_idx = self.residues_dframe['output_chain_idx'][idx]
        output_residue_idx = self.residues_dframe['residue_output_idx'][idx]
        labels_remodelled_yes_no = self.residues_dframe['remodelled'][idx]

        event_map_grid = gemmi.read_ccp4_map(event_map_name).grid
        two_fo_fc_grid = extract_box.gen_two_fo_fc_from_mtz(mtz_name)
        input_residue = gemmi.read_structure(input_structure_name)[0][input_chain_idx][input_residue_idx]
        output_residue = gemmi.read_structure(output_structure_name)[0][output_chain_idx][output_residue_idx]

        sample = {
            'event_map': event_map_grid,
            'mtz': two_fo_fc_grid,
            'input_residue': input_residue,
            'output_residue': output_residue,
            'labels_remodelled_yes_no': labels_remodelled_yes_no
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandRot(object):
    """
    sampling random rotations
    """
    def __call__(self, sample):
        


class ToTensor(object):
    """
    convert ndarrays into torch tensors
    """
    def __call__(self, sample):




if __name__ == '__main__':
    csv_file = pathlib.Path(__file__).parent.parent.resolve() / 'training' / 'training_data.csv'
    residues_dataset = ResidueDataset(csv_file)
    
    print(len(residues_dataset))
    for i in range(len(residues_dataset)):
        sample = residues_dataset[i]
        print(sample['event_map'],
            sample['mtz'],
            sample['input_residue'],
            sample['output_residue'],
            sample['labels_remodelled_yes_no'])