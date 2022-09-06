import argparse
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import pathlib
import os
from os import access, R_OK

import gemmi
import sys
import numpy as np

from lib import rmsdcalc, contact_search


def list_systems(path_year: pathlib.Path):

    paths_system = [
        x 
        for x 
        in path_year.iterdir() 
        if (x.is_dir() and access(x, R_OK))
        ] #black style of python

    return paths_system

def check_path_access(path: pathlib.Path) -> Tuple[Optional[pathlib.Path], bool]:
# could cause errors
    if not (path.is_dir()):
        return f'{path} does not exist!', False
    if not (access(path, R_OK)):
        return f'{path}: No Permission', False

    return path, True

def check_panddas_inspect_csv(path_events_csv: pathlib.Path) -> Union[str, pathlib.Path]:

    if not path_events_csv.is_file():
        return f'{path_events_csv} does not exist!'
    if not access(path_events_csv, R_OK):
        return f'{path_events_csv}: No Access'
    if os.stat(path_events_csv).st_size == 0:
        return 'pandda_inspect_events.csv not filled'
    
    return path_events_csv


def find_path_analysis(path_system: pathlib.Path) -> Optional[pathlib.Path]: 
    #note — this could cause problems down the line, because function returns different types according to different conditions

    if not path_system.is_dir():
        return None
    if not access(path_system, R_OK):
        return None
    
    path_processing = path_system / 'processing'
    path_analysis = path_system / 'processing' / 'analysis'
    if not path_processing.is_dir():
        return None
    if not access(path_processing, R_OK):
        return None

    if not path_analysis.is_dir():
        return None
    if not access(path_analysis, R_OK):
        return None
    
    print(path_analysis) #TODO — look at python logging

    return path_analysis


def find_panddas(path_analysis: pathlib.Path) -> list[pathlib.Path]:
    paths = [x for x in path_analysis.iterdir() if x.is_dir()]
    paths_panddas = []

    for path in paths:
        if not access(path, R_OK):
            continue
        
        path_analyses = [x for x in path.iterdir() if x.is_dir() and 'analyses' in x.stem]
        
        if len(path_analyses) == 0:
            continue
        
        paths_panddas.append(path)

    return paths_panddas

def find_inspect_csv(paths_panddas: list[pathlib.Path]) -> list[pathlib.Path]:

    csvs = []

    if len(paths_panddas) == 0:
        return csvs
    
    for path_panddas in paths_panddas:
        paths_analyses = [x for x in path_panddas.iterdir() if x.is_dir() and 'analyses' in x.stem]
        
        for path in paths_analyses:
            print(path)

            if not (path.is_dir() and access(path, R_OK)):
                continue

            path_events_csv = path / 'pandda_inspect_events.csv'

            if not (path_events_csv.is_file() and access(path_events_csv, R_OK)):
                continue

            if os.stat(path_events_csv).st_size == 0:
                continue
            
            csvs.append(path_events_csv)
    

    return csvs

        
def find_csv_from_system_path(path_system: pathlib.Path) -> list[pathlib.Path]:
    path_analysis = find_path_analysis(path_system)

    if path_analysis == None:
        return []
    
    paths_panddas = find_panddas(path_analysis)
    csvs = find_inspect_csv(paths_panddas)

    print(csvs)

    return csvs

def find_csvs_from_year(path_year: str) -> list[pathlib.Path]:
    path_year = pathlib.Path(path_year)
    csvs = []
    paths_system = [x for x in path_year.iterdir() if x.is_dir()]

    for path in paths_system:
        csv_paths = find_csv_from_system_path(path)
        csvs += csv_paths
    
    return csvs


######### 

def find_all_csvs(path_data: str) -> list[pathlib.Path]:
    path_data = pathlib.Path(path_data)
    
    csvs = []
    paths_year = [x for x in path_data.iterdir() if x.is_dir()]

    for path_year in paths_year:
        csv_paths = find_csvs_from_year(path_year)
        csvs += csv_paths
    
    print(csvs)
    print(len(csvs))
    
    return csvs

def find_pandda_inspect_csv_from_list_of_paths(list_of_paths: list[str]) -> list[pathlib.Path]:
    csvs = []

    for str_path in list_of_paths:
        path = pathlib.Path(str_path)
        csv_paths = find_csv_from_system_path(path)
        csvs += csv_paths
    
    return csvs


def if_high_confidence_ligand(csv_path: pathlib.Path) -> bool:
    df = pd.read_csv(csv_path)
    df_ligand_placed = df[(df['Ligand Placed']==True) & (df['Ligand Confidence']=='High')] # don't assign to
    sum = df_ligand_placed['Ligand Placed'].sum()
    if sum == 0:
        return False
    
    return True

def filter_csvs(csvs: list[pathlib.Path]) -> list[pathlib.Path]: #filtering function
    csvs_new = list(filter(if_high_confidence_ligand,csvs)) #easy to parallelise, and easy to read
    
    print(csvs_new)
    print(len(csvs_new))
    return csvs_new

def panddas_inspect_csv_to_row(path_csv: pathlib.Path):
    panddas_path = path_csv.parent.parent
    path_analysis = panddas_path.parent
    path_model_building = path_analysis / 'model_building'

    if not path_model_building.is_dir():
        path_model_building = path_analysis / 'initial_model'
    
    return [path_csv, panddas_path, path_model_building]

def list_pandda_model_paths(csvs): #mapping function
    models = []
    colnames = ['csv_path', 'panddas_path', 'model_building']

    models = list(map(panddas_inspect_csv_to_row, csvs))

    df_models = pd.DataFrame(models, columns=colnames)
    
    print(df_models)
    
    return df_models

def get_event_record(row: Any, panddas_path: pathlib.Path, model_building: pathlib.Path) -> Dict:
    system = str(row.dtag).split('-x')[0]
    dtag = str(row.dtag)
    event_idx = str(row.event_idx)
    x = row.x
    y = row.y
    z = row.z
    occupancy = row._6
    high_resolution = row.high_resolution
    ligand_placed = row._8
    ligand_confidence = row._9
    event_path = panddas_path / 'processed_datasets' / f'{row.dtag}' / f'{row.dtag}-event_{row.event_idx}_1-BDC_{row._6}_map.native.ccp4'
    mtz_path = panddas_path / 'processed_datasets'  / f'{row.dtag}' / f'{row.dtag}-pandda-input.mtz'
    input_model_path = panddas_path / 'processed_datasets' / f'{row.dtag}' / f'{row.dtag}-pandda-input.pdb'
    output_model_path = model_building / f'{row.dtag}' / f'{row.dtag}-pandda-model.pdb'

    return {
        'system' : system,
        'dtag': dtag,
        'event_idx': event_idx, 
        'x': x, 
        'y': y, 
        'z': z, 
        '1-BDC': occupancy, 
        'high_resolution': high_resolution,
        'Ligand Placed': ligand_placed, 
        'Ligand Confidence': ligand_confidence,
        'event_map': event_path,
        'mtz': mtz_path,
        'input_model': input_model_path,
        'output_model': output_model_path
    }

def find_events_per_dataset(csv_path, panddas_path, model_building) -> list[Dict]:
    csv_path = pathlib.Path(csv_path)
    panddas_path = pathlib.Path(panddas_path)
    model_building = pathlib.Path(model_building)

    pandda_inspect = pd.read_csv(csv_path)
    pandda_inspect = pandda_inspect.loc[(pandda_inspect['Ligand Placed']==True) & (pandda_inspect['Ligand Confidence']=='High')]
    pandda_inspect = pandda_inspect[['dtag','event_idx', 'x', 'y', 'z', '1-BDC', 'high_resolution','Ligand Placed', 'Ligand Confidence']]
    print(pandda_inspect)

    event_records = list(map(
            get_event_record, 
            pandda_inspect.itertuples(),
            itertools.repeat(panddas_path, pandda_inspect.shape[0]),
            itertools.repeat(model_building, pandda_inspect.shape[0])
            ))

    print(event_records)

    return event_records

def find_events_all_datasets(df_models):
    event_records = []
    for row in df_models.itertuples():
        event_records += find_events_per_dataset(row.csv_path, row.panddas_path, row.model_building)

    df_pandda_inspect = pd.DataFrame(event_records)
    print(df_pandda_inspect)

    return df_pandda_inspect

def filter_non_existent_paths(df_pandda_inspect, fname='model_paths.csv'):
    #filter operation
    drop = []
    for i, row in enumerate(df_pandda_inspect.itertuples()):
        event_map = pathlib.Path(row.event_map)
        mtz = pathlib.Path(row.mtz)
        input_model = pathlib.Path(row.input_model)
        output_model = pathlib.Path(row.output_model)

        if not (event_map.is_file() and mtz.is_file() and input_model.is_file() and output_model.is_file()):
            drop.append(i)

    df_pandda_inspect_new = df_pandda_inspect.drop(df_pandda_inspect.index[drop])
    outfname = pathlib.Path.cwd() / 'training' / fname
    df_pandda_inspect_new.reset_index(drop=True)
    df_pandda_inspect_new.to_csv(outfname, index=False)

    return df_pandda_inspect_new


########
def calc_rmsd_per_chain(chain_input, chain_output):
    rmsd = []
    residue_input_idx = []
    residue_output_idx = []
    residue_name = []

    # for residue_input, residue_output in itertools.product(chain_input, chain_output):
        
    for i, residue_input in enumerate(chain_input):
        for j, residue_output in enumerate(chain_output):
            if ((str(residue_input) == str(residue_output)) and 
                (residue_input.het_flag == 'A' and residue_output.het_flag == 'A')):
                CoM_input = rmsdcalc.calculate_CoM_residue(residue_input)
                CoM_output = rmsdcalc.calculate_CoM_residue(residue_output)
                dist = np.linalg.norm(CoM_input-CoM_output)

                rmsd.append(dist)
                residue_input_idx.append(i)
                residue_output_idx.append(j)
                residue_name.append(str(residue_input))
    
    return rmsd, residue_input_idx, residue_output_idx, residue_name
        
def calc_rmsd_per_model(model_input, model_output):
    # could be made cleaner by returning something as a list of dicts
    chain, chain_idx = rmsdcalc.map_chains(model_input, model_output)
    input_chain_idx = []
    output_chain_idx = []
    rmsd_1 = []
    residue_input_idx_1 = []
    residue_output_idx_1 = []
    residue_name_1 = []

    for pair, pair_idx in zip(chain, chain_idx):
        rmsd, residue_input_idx, residue_output_idx, residue_name = calc_rmsd_per_chain(pair[0], pair[1])

        if not len(rmsd) == 0: # if rmsd list is not empty
            input_chain_idx += [pair_idx[0]]*len(rmsd)
            output_chain_idx += [pair_idx[1]]*len(rmsd)
            rmsd_1 += rmsd
            residue_input_idx_1 += residue_input_idx
            residue_output_idx_1 += residue_output_idx
            residue_name_1 += residue_name
    
    return input_chain_idx, output_chain_idx, residue_input_idx_1, residue_output_idx_1, residue_name_1, rmsd_1


def record_per_residue_rmsd_data(input_chain_idx, 
        output_chain_idx, 
        residue_input_idx, 
        residue_output_idx, 
        residue_name, 
        rmsd, 
        row) -> Dict:

    return {'system': row.system,
            'dtag': row.dtag,
            'event_idx': row.event_idx, 
            'x': row.x, 
            'y': row.y, 
            'z': row.z,
            '1-BDC': row._7, 
            'high_resolution': row.high_resolution,
            'Ligand Placed': row._9, 
            'Ligand Confidence': row._10,
            'event_map': row.event_map,
            'mtz': row.mtz,
            'input_model': row.input_model,
            'output_model': row.output_model, 
            'input_chain_idx': input_chain_idx, 
            'output_chain_idx': output_chain_idx, 
            'residue_input_idx': residue_input_idx, 
            'residue_output_idx': residue_output_idx, 
            'residue_name': residue_name, 
            'rmsd': rmsd
            }

def calc_rmsds_from_csv(df_pandda_inspect):
    #loop creating a list of dictionaries
    #each loop creates a dictionary with just one element

    records = []
    for row in df_pandda_inspect.itertuples():
        input = gemmi.read_structure(str(row.input_model))[0]
        output = gemmi.read_structure(str(row.output_model))[0]

        rmsds = calc_rmsd_per_model(input, output)
        
        record = map(record_per_residue_rmsd_data, 
                    *rmsds,
                    itertools.repeat(row, len(rmsds[0])))
        records += record

    df_residues = pd.DataFrame(records)
    df_residues = df_residues.drop_duplicates()
    df_residues = df_residues.reset_index(drop=True)
    print(df_residues)
    return df_residues

def find_remodelled_residues(df_residues, threshold=0.8, fname='all_residues.csv'):
    df_residues['remodelled'] = df_residues['rmsd'] > threshold
    df_residues = df_residues.drop_duplicates(keep='first')
    # print(df_residues)
    outfname = pathlib.Path.cwd() / 'training' / fname
    df_residues.to_csv(outfname, index=False)

    return df_residues

def filter_remodelled_residues(df_residues, fname='remodelled.csv'):
    df_remodelled = df_residues[df_residues['remodelled']==True]
    df_remodelled = df_remodelled.reset_index(drop=True)
    print(df_remodelled)
    outfname = pathlib.Path.cwd() / 'training' / fname
    df_remodelled.to_csv(outfname, index=False)
    return df_remodelled

def find_contacts(df_residues, fname='neg_data.csv'):
    list = []

    df_remodelled = df_residues[df_residues['remodelled']==True]

    for row in df_remodelled.itertuples():
        structure = gemmi.read_structure(str(row.output_model))
        residue = structure[0][row.output_chain_idx][row.residue_output_idx]

        contact_list = contact_search.find_contacts_per_residue(structure, residue)

        for contact in contact_list:
            chain_idx = contact[0]
            residue_idx = contact[1]
            res_name = contact[2]
            contact_row = df_residues.loc[
                (df_residues['remodelled']==False) &           
                (df_residues['input_model']==row.input_model) &
                (df_residues['output_chain_idx']==chain_idx) &
                (df_residues['residue_output_idx']==residue_idx) &
                (df_residues['residue_name']==res_name)
                ].to_dict('records')
            
            # print(contact_row)
            list += contact_row
    
    df_negative_data = pd.DataFrame(list)
    print(df_negative_data)
    df_negative_data = df_negative_data.drop_duplicates()
    df_negative_data = df_negative_data.reset_index(drop=True)
    print(df_negative_data)
    outfname = pathlib.Path.cwd() / 'training' / fname
    df_negative_data.to_csv(outfname, index=False)
    
    return df_negative_data

def gen_training_data_csv(df_remodelled, df_negative_data, fname='training_data.csv'):
    df_training = pd.concat([df_remodelled, df_negative_data])
    df_training = df_training.sort_values(by=['dtag'])
    df_training = df_training.reset_index(drop=True)
    outfname = pathlib.Path.cwd() / 'training' / fname
    df_training.to_csv(outfname, index=False)

    return df_training

######

def look_for_training_data_to_csv(path_to_labxchem_data_dir: str, path_to_csv_with_dataset_paths: Optional[str], force=True):
        training_data_csv = pathlib.Path.cwd() / 'training' / 'training_data.csv'
        
        if force==True:
            pass
        elif training_data_csv.is_file():
            return None

        csvs = find_all_csvs(path_to_labxchem_data_dir)
        filtered_csvs = filter_csvs(csvs)

        if not path_to_csv_with_dataset_paths is None:
            df_paths = pd.read_csv(path_to_csv_with_dataset_paths)
            list_of_paths = df_paths['path'].tolist()
            csvs_from_csv_with_dataset_paths = find_pandda_inspect_csv_from_list_of_paths(list_of_paths)
            filtered_csvs_from_csv_with_dataset_paths = filter_csvs(csvs_from_csv_with_dataset_paths)
            filtered_csvs += filtered_csvs_from_csv_with_dataset_paths

        df = list_pandda_model_paths(filtered_csvs)
        events_csv = find_events_all_datasets(df)
        events_csv = filter_non_existent_paths(events_csv)
        df_residues = calc_rmsds_from_csv(events_csv)
        df_residues = find_remodelled_residues(df_residues)
        df_remodelled = filter_remodelled_residues(df_residues)
        df_negative_data = find_contacts(df_residues)
        df_training = gen_training_data_csv(df_remodelled, df_negative_data)
        
        return None

#######

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path',action='store',help='labxchem data path',required=True)
    parser.add_argument('-c','--csvfile',action='store',help='csv with dataset paths')

    args = parser.parse_args()
    path_to_labxchem_data_dir = args.path
    path_to_csv_with_dataset_paths = args.csvfile

    look_for_training_data_to_csv(path_to_labxchem_data_dir, 
                                path_to_csv_with_dataset_paths)




    
    


    
    



        
       
        
       

    

