import logging
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
from gemmi_lib import extract_box, random_sampling


"""
PyTorch Documentation on Custom Datasets: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""


class ResidueDataset(Dataset):
    def __init__(self, residues_dframe: type[pd.DataFrame], transform=None):
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
        
        row_idx = self.residues_dframe.index[idx]
        system = self.residues_dframe.iloc[idx]['system']
        dtag = self.residues_dframe.iloc[idx]['dtag']
        event_map_name = self.residues_dframe.iloc[idx]['event_map']
        mtz = self.residues_dframe.iloc[idx]['mtz']
        input_structure_name = self.residues_dframe.iloc[idx]['input_model']
        output_structure_name = self.residues_dframe.iloc[idx]['output_model']
        input_chain_idx = self.residues_dframe.iloc[idx]['input_chain_idx']
        input_residue_idx = self.residues_dframe.iloc[idx]['residue_input_idx']
        rmsd = self.residues_dframe.iloc[idx]['rmsd']

        labels_remodelled_yes_no = int(self.residues_dframe.iloc[idx]['remodelled']) # whether residue needs remodelling

        #fetch event_map and input residue
        event_map = gemmi.read_ccp4_map(event_map_name) #change space group of event map
        event_map_grid = extract_box.fetch_grid_from_pandda_map(event_map)
        input_residue = gemmi.read_structure(input_structure_name)[0][input_chain_idx][input_residue_idx]
        
        sample = {
            'row_idx': row_idx,
            'system': system,
            'dtag': dtag,
            'input_model': input_structure_name,
            'output_model': output_structure_name,
            'mtz': mtz,
            'event_map_name': event_map_name,
            'input_chain_idx': input_chain_idx,
            'input_residue_idx': input_residue_idx,
            'event_map': event_map_grid,
            'rmsd': rmsd,
            'input_residue': input_residue,
            'labels_remodelled_yes_no': labels_remodelled_yes_no
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class SamplingRandomRotations(object):
    
    """
    Data Augmentation — sampling random rotations and translations
    """
    def __init__(self, translation_radius=0, random_rotation=True, use_mtz=False):
        self.translation_radius = translation_radius
        self.random_rotation = random_rotation
        self.use_mtz = use_mtz

    def __call__(self, sample) -> Dict[str, np.ndarray]:
        event_map_grid = sample['event_map']
        input_residue = sample['input_residue']
        input_residue_name = str(input_residue)
        labels_remodelled_yes_no = np.array(sample['labels_remodelled_yes_no']).astype(np.float32) #needs remodelling = 1, doesn't need remodelling = 0
        
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
        
        input_residue_array = (input_residue_array - 0.5) #normalise to -0.5 to 0.5
        
        
        if self.use_mtz == True:
            mtz = sample['mtz']
            two_fo_fc_grid = extract_box.gen_two_fo_fc_from_mtz(str(mtz))
            normalized_two_fo_fc_grid = extract_box.copy_normalized_gemmi_float_grid(two_fo_fc_grid)
            two_fo_fc_array = extract_box.create_numpy_array_with_gemmi_interpolate(input_residue,
                                                                                    normalized_two_fo_fc_grid,
                                                                                    rot_mat,
                                                                                    vec_rand)
            return {
                'row_idx': sample['row_idx'],
                'system': sample['system'],
                'dtag': sample['dtag'],
                'input_model': sample['input_model'],
                'output_model': sample['output_model'],
                'mtz': sample['mtz'],
                'event_map_name': sample['event_map_name'],
                'input_chain_idx': sample['input_chain_idx'],
                'input_residue_idx': sample['input_residue_idx'],
                'input_residue_name': input_residue_name,
                'rmsd': sample['rmsd'],
                'event_map': event_map_array,
                'two_fo_fc_array': two_fo_fc_array,
                'input_residue': input_residue_array,
                'labels_remodelled_yes_no': labels_remodelled_yes_no
                }


        return {
            'row_idx': sample['row_idx'],
            'system': sample['system'],
            'dtag': sample['dtag'],
            'input_model': sample['input_model'],
            'output_model': sample['output_model'],
            'mtz': sample['mtz'],
            'event_map_name': sample['event_map_name'],
            'input_chain_idx': sample['input_chain_idx'],
            'input_residue_idx': sample['input_residue_idx'],
            'input_residue_name': input_residue_name,
            'rmsd': sample['rmsd'],
            'event_map': event_map_array,
            'input_residue': input_residue_array,
            'labels_remodelled_yes_no': labels_remodelled_yes_no
        }


class ConcatEventResidueToTwoChannels(object):
    """
    Combine event map and input residue into two channels
    """
    def __call__(self, sample) -> Dict[str, Union[np.ndarray, List]]:
        event_map_array = sample['event_map']
        input_residue_array = sample['input_residue']
        labels_remodelled_yes_no = sample['labels_remodelled_yes_no']

        if 'two_fo_fc_array' in sample:
            two_fo_fc_array = sample['two_fo_fc_array']
            event_residue_array = np.stack((event_map_array, two_fo_fc_array, input_residue_array), axis=0)
        else:
            event_residue_array = np.stack((event_map_array, input_residue_array), axis=0) #order of channels along axis 0 = event map, input residue
        
        return {
            'row_idx': sample['row_idx'],
            'system': sample['system'],
            'dtag': sample['dtag'],
            'input_model': sample['input_model'],
            'output_model': sample['output_model'],
            'mtz': sample['mtz'],
            'event_map_name': sample['event_map_name'],
            'input_chain_idx': sample['input_chain_idx'],
            'input_residue_idx': sample['input_residue_idx'],
            'input_residue_name': sample['input_residue_name'],
            'rmsd': sample['rmsd'],
            'event_residue_array': event_residue_array,
            'labels_remodelled_yes_no': labels_remodelled_yes_no
        }


class ToTensor(object):
    """
    converts ndarrays into torch tensors
    MIGHT NOT BE NECESSARY
    """

    def __call__(self, sample) -> Dict[str, torch.Tensor]:
        event_residue_array = sample['event_residue_array']
        labels_remodelled_yes_no = sample['labels_remodelled_yes_no']

        return {
            'row_idx': sample['row_idx'],
            'system': sample['system'],
            'dtag': sample['dtag'],
            'input_model': sample['input_model'],
            'output_model': sample['output_model'],
            'mtz': sample['mtz'],
            'event_map_name': sample['event_map_name'],
            'input_chain_idx': sample['input_chain_idx'],
            'input_residue_idx': sample['input_residue_idx'],
            'input_residue_name': sample['input_residue_name'],
            'rmsd': sample['rmsd'],
            'event_residue_array': torch.from_numpy(event_residue_array),
            'labels_remodelled_yes_no': torch.from_numpy(labels_remodelled_yes_no).long()
        }

class AddGaussianNoise(object):
    ##### NOT USED #####
    def __init__(self, mean=0., std=1.):
       
        self.std = std
        self.mean = mean
    
    def __call__(self, sample):
        event_residue_array = sample['event_residue_array']
        event_residue_array[1] = event_residue_array[1] + torch.randn(event_residue_array[1].size()) * self.std + self.mean
        
        return {
            'row_idx': sample['row_idx'],
            'systme': sample['system'],
            'dtag': sample['dtag'],
            'input_model': sample['input_model'],
            'output_model': sample['output_model'],
            'mtz': sample['mtz'],
            'event_map_name': sample['event_map_name'],
            'input_chain_idx': sample['input_chain_idx'],
            'input_residue_idx': sample['input_residue_idx'],
            'input_residue_name': sample['input_residue_name'],
            'rmsd': sample['rmsd'],
            'event_residue_array': event_residue_array,
            'labels_remodelled_yes_no': sample['labels_remodelled_yes_no']
        }
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def generate_dataset(residues_dframe, translation_radius=0, one_hot=False):
    training_tsfm = transforms.Compose([
                        SamplingRandomRotations(translation_radius),
                        ConcatEventResidueToTwoChannels(),
                        ToTensor(),
                    ])
    dataset = ResidueDataset(residues_dframe, transform=training_tsfm)
    return dataset

#####
# DEBUGGING
class DebugResidueDataset(Dataset):
    def __init__(self, residues_dframe: type[pd.DataFrame], transform=None):
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
        
        row_idx = self.residues_dframe.iloc[idx]['row_idx']
        system = self.residues_dframe.iloc[idx]['system']
        dtag = self.residues_dframe.iloc[idx]['dtag']
        mtz = self.residues_dframe.iloc[idx]['mtz']
        event_map_name = self.residues_dframe.iloc[idx]['event_map_name']
        input_structure_name = self.residues_dframe.iloc[idx]['input_model']
        output_structure_name = self.residues_dframe.iloc[idx]['output_model']
        input_chain_idx = self.residues_dframe.iloc[idx]['input_chain_idx']
        input_residue_idx = self.residues_dframe.iloc[idx]['input_residue_idx']
        rmsd = self.residues_dframe.iloc[idx]['rmsd']

        labels_remodelled_yes_no = int(self.residues_dframe.iloc[idx]['labels_remodelled_yes_no']) # whether residue needs remodelling

        #fetch event_map and input residue
        event_map = gemmi.read_ccp4_map(event_map_name) #change space group of event map
        event_map_grid = extract_box.fetch_grid_from_pandda_map(event_map)
        input_residue = gemmi.read_structure(input_structure_name)[0][input_chain_idx][input_residue_idx]
        
        sample = {
            'row_idx': row_idx,
            'system': system,
            'dtag': dtag,
            'input_model': input_structure_name,
            'output_model': output_structure_name,
            'mtz': mtz,
            'event_map_name': event_map_name,
            'input_chain_idx': input_chain_idx,
            'input_residue_idx': input_residue_idx,
            'event_map': event_map_grid,
            'rmsd': rmsd,
            'input_residue': input_residue,
            'labels_remodelled_yes_no': labels_remodelled_yes_no
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    """
    testing out if things are coded up properly
    """
    csv_file = pathlib.Path(__file__).parent.parent / 'training_data_paths' / 'training_data.csv'
    residues_dframe=pd.read_csv(csv_file)
    transformed_dataset = generate_dataset(residues_dframe,translation_radius=0)

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(i, sample['event_residue_array'].shape, sample['labels_remodelled_yes_no'])

        if i == 3:
            break
    
    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)
    
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['event_residue_array'].size(), sample_batched['labels_remodelled_yes_no'])

        if i_batch == 3:
            break
    
    sample1 = next(iter(dataloader))

    print(f'{sample1["row_idx"].tolist()=}')
    print(f'{sample1["dtag"]=}')
    print(f'event_map shape = {sample1["event_residue_array"].shape}')
    print(f'labels shape = {sample1["labels_remodelled_yes_no"].shape}')
