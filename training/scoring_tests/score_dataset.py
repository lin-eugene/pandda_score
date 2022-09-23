from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import logging
import pandas as pd
import gemmi
from gemmi_lib import extract_box, random_sampling
from typing import Dict, List, Optional, Union
import numpy as np

__all__ = ['ResidueDataset',
           'SamplingRandomRotations',
            'ConcatEventResidueToTwoChannels',
            'ToTensor']
class ResidueDataset(Dataset):
    def __init__(self, 
                residues_dframe: type[pd.DataFrame], 
                transform: Optional[List] = None):

        self.residues_dframe = residues_dframe
        self.transform=transform

        self.residues_dframe.index += 1 # so that the index starts at 1, not 0
        
    def __len__(self):
        logging.debug(f'length of dataset: {len(self.residues_dframe)}')
        
        return len(self.residues_dframe)
    
    def __getitem__(self, idx):
        """
        items - (1) event map, (2) 2fo-fc map, (3) input structure, (4) output structure at specified residue
        event map and 2fo-fc map at residue at input position
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        event_map_name = self.residues_dframe.iloc[idx]['event_map']
        input_structure_name = self.residues_dframe.iloc[idx]['input_model']
        input_chain_idx = self.residues_dframe.iloc[idx]['input_chain_idx']
        input_residue_idx = self.residues_dframe.iloc[idx]['input_residue_idx']
        residue_name = self.residues_dframe.iloc[idx]['residue_name']

        #fetch event_map and input residue
        event_map = gemmi.read_ccp4_map(event_map_name) #change space group of event map
        event_map_grid = extract_box.fetch_grid_from_pandda_map(event_map)
        input_residue = gemmi.read_structure(input_structure_name)[0][input_chain_idx][input_residue_idx]
        
        sample = {

            'input_model': input_structure_name,
            'event_map_name': event_map_name,
            'input_chain_idx': input_chain_idx,
            'input_residue_idx': input_residue_idx,
            'event_map': event_map_grid,
            'input_residue': input_residue,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample



class SamplingRandomRotations(object):
    
    """
    Data Augmentation — sampling random rotations and translations
    """
    def __init__(self, translation_radius=0, random_rotation=False):
        self.translation_radius = translation_radius
        self.random_rotation = random_rotation

    def __call__(self, sample) -> Dict[str, np.ndarray]:
        event_map_grid = sample['event_map']
        input_residue = sample['input_residue']
        input_residue_name = str(input_residue)
        
        # logging.debug(type(input_residue))
        # logging.debug(f'event_map_grid = {event_map_grid}')
        # logging.debug(f'axis_order = {event_map_grid.axis_order}')

        normalized_event_map_grid = extract_box.copy_normalized_gemmi_float_grid(event_map_grid)
        event_map_grid_copy = extract_box.make_gemmi_zeros_float_grid(event_map_grid)
        input_residue_masked_grid = extract_box.gen_mask_from_atoms(
            event_map_grid_copy,
            input_residue)

        # setting random translation vector and random rotation matrix
        if self.translation_radius == 0:
            vec_rand = None
        else:
            vec_rand = random_sampling.rand_translations(R=self.translation_radius)
        
        if self.random_rotation == True:
            rot_mat = random_sampling.rand_rotation_matrix()
        else:
            rot_mat = None
        
        # extracting numpy array from event map and voxelised input residue
        event_map_array = extract_box.create_numpy_array_with_gemmi_interpolate(input_residue, 
                                                                                normalized_event_map_grid, 
                                                                                rot_mat, 
                                                                                vec_rand)
        input_residue_array = extract_box.create_numpy_array_with_gemmi_interpolate(input_residue, 
                                                                                    input_residue_masked_grid, 
                                                                                    rot_mat, 
                                                                                    vec_rand)
        
        # normalising array values
        # event_map_array_norm = (event_map_array - np.mean(event_map_array)) / np.std(event_map_array)
        input_residue_array = (input_residue_array - 0.5) #normalise to -0.5 to 0.5
        
        return {
            'event_map_name': sample['event_map_name'],
            'input_model': sample['input_model'],
            'input_chain_idx': sample['input_chain_idx'],
            'input_residue_idx': sample['input_residue_idx'],
            'residue_name': input_residue_name,
            'event_map': event_map_array,
            'input_residue': input_residue_array,
        }


class ConcatEventResidueToTwoChannels(object):
    """
    Combine event map and input residue into two channels
    """
    def __call__(self, sample) -> Dict[str, Union[np.ndarray, List]]:
        event_map_array = sample['event_map']
        input_residue_array = sample['input_residue']

        event_residue_array = np.stack((event_map_array, input_residue_array), axis=0) #order of channels along axis 0 = event map, input residue
        
        return {
            'event_map_name': sample['event_map_name'],
            'input_model': sample['input_model'],
            'input_chain_idx': sample['input_chain_idx'],
            'input_residue_idx': sample['input_residue_idx'],
            'residue_name': sample['residue_name'],
            'event_residue_array': event_residue_array,
        }


class ToTensor(object):
    """
    converts ndarrays into torch tensors
    MIGHT NOT BE NECESSARY
    """

    def __call__(self, sample) -> Dict[str, torch.Tensor]:
        event_residue_array = sample['event_residue_array']

        return {
            'event_map_name': sample['event_map_name'],
            'input_model': sample['input_model'],
            'input_chain_idx': sample['input_chain_idx'],
            'input_residue_idx': sample['input_residue_idx'],
            'residue_name': sample['residue_name'],
            'event_residue_array': torch.from_numpy(event_residue_array),
        }