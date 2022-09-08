from typing import Dict, Union, List
import numpy as np
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pathlib
import sys
import gemmi

codebase_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(codebase_path))
from lib import extract_box, random_sampling

"""
PyTorch Documentation on Custom Datasets: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
def fetch_grid_from_pandda_map(map: type[gemmi.Ccp4Map]) -> type[gemmi.FloatGrid]:
    """
    reads in map
    modifies space group to P1 in header
    returns gemmi grid object

    """
    grid = map.grid
    grid.spacegroup = gemmi.find_spacegroup_by_name('P 1')
    map.setup()
    
    return grid

def list_systems(training_csv_path: str) -> List[str]:
    """
    returns list of systems from training csv
    """
    training_dframe = pd.read_csv(training_csv_path)
    systems = training_dframe['system'].value_counts()
    print(systems)
    return systems

class ResidueDataset(Dataset):
    def __init__(self, residues_dframe: type[pd.DataFrame], transform=None):
        self.residues_dframe = residues_dframe
        self.transform=transform
        
    def __len__(self):
        return len(self.residues_dframe)
    
    def __getitem__(self, idx):
        """
        items - (1) event map, (2) 2fo-fc map, (3) input structure, (4) output structure at specified residue
        event map and 2fo-fc map at residue at input position
        """

        if torch.is_tensor(idx):
            idx = idx.tolist() # i don't understand why this works, but it works
        
        event_map_name = self.residues_dframe['event_map'][idx]
        input_structure_name = self.residues_dframe['input_model'][idx]
        input_chain_idx = self.residues_dframe['input_chain_idx'][idx]
        input_residue_idx = self.residues_dframe['residue_input_idx'][idx]
        labels_remodelled_yes_no = int(self.residues_dframe['remodelled'][idx]) # whether residue needs remodelling

        #fetch event_map and input residue
        event_map = gemmi.read_ccp4_map(event_map_name) #change space group of event map
        event_map_grid = fetch_grid_from_pandda_map(event_map)
        input_residue = gemmi.read_structure(input_structure_name)[0][input_chain_idx][input_residue_idx]
        
        sample = {
            'event_map': event_map_grid,
            'input_residue': input_residue,
            'labels_remodelled_yes_no': labels_remodelled_yes_no
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class RandRot(object):
    """
    sampling random rotations
    """
    def __call__(self, sample) -> Dict[str, Union[np.ndarray, List]]:
        event_map_grid = sample['event_map']
        input_residue = sample['input_residue']
        labels_remodelled_yes_no = sample['labels_remodelled_yes_no']
        
        # print(type(input_residue))
        # print(f'event_map_grid = {event_map_grid}')
        # print(f'axis_order = {event_map_grid.axis_order}')

        event_map_grid_copy = extract_box.make_gemmi_zeros_float_grid(event_map_grid)
        input_residue_masked_grid = extract_box.gen_mask_from_atoms(
            event_map_grid_copy,
            input_residue)

        vec_rand = random_sampling.rand_translations()
        rot_mat = random_sampling.rand_rotation_matrix()
        
        event_map_array = extract_box.create_numpy_array_with_gemmi_interpolate(input_residue, event_map_grid, rot_mat, vec_rand)
        input_residue_array = extract_box.create_numpy_array_with_gemmi_interpolate(input_residue, input_residue_masked_grid, rot_mat, vec_rand)
        

        return {
            'event_map': event_map_array,
            'input_residue': input_residue_array,
            'labels_remodelled_yes_no': labels_remodelled_yes_no
        }

class ToTensor(object):
    """
    converts ndarrays into torch tensors
    MIGHT NOT BE NECESSARY
    """
    def __call__(self, sample):
        event_map_array = sample['event_map']
        input_residue_array = sample['input_residue']
        labels_remodelled_yes_no = np.array(sample['labels_remodelled_yes_no'])

        return {
            'event_map': torch.from_numpy(event_map_array),
            'input_residue': torch.from_numpy(input_residue_array),
            'labels_remodelled_yes_no': torch.from_numpy(labels_remodelled_yes_no)
        }

if __name__ == '__main__':
    
    csv_file = pathlib.Path(__file__).parent.parent / 'training' / 'training_data.csv'
    residues_dframe=pd.read_csv(csv_file)
    transformed_dataset = ResidueDataset(residues_dframe=residues_dframe,
                                        transform=transforms.Compose([
                                            RandRot(),
                                            ToTensor()
                                        ]))
    list_systems(csv_file)
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['event_map'].shape, sample['input_residue'].shape, sample['labels_remodelled_yes_no'])

        if i == 3:
            break
    
    # dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

    # dataset = ResidueDataset(csv_file=csv_file)

    # for i in range(len(dataset)):
    #     sample = dataset[i]
    #     print(i, sample['event_map'], sample['input_residue'], sample['labels_remodelled_yes_no'])
        
    #     if i == 3:
    #         break

