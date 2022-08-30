import pandas as pd
import pathlib
import os
from os import access, R_OK
import pickle
import gemmi

from lib import rmsdcalc, contact_search
import sys
import numpy as np



def list_systems(path_year: pathlib.PosixPath):

    paths_system = [x for x in path_year.iterdir() if (x.is_dir() and access(x, R_OK))]

    return paths_system

def check_path_access(path):

    if not (path.is_dir()):
        return f'{path} does not exist!', False
    if not (access(path, R_OK)):
        return f'{path}: No Permission', False

    return path, True

def check_panddas_inspect_csv(path_events_csv):

    if not path_events_csv.is_file():
        return f'{path_events_csv} does not exist!', False
    if not access(path_events_csv, R_OK):
        return f'{path_events_csv}: No Access'
    if os.stat(path_events_csv).st_size == 0:
        return 'pandda_inspect_events.csv not filled'
    
    return path_events_csv


def find_path_analysis(path_system):

    if not path_system.is_dir():
        return
    if not access(path_system, R_OK):
        return
    
    path_analysis = path_system / 'processing' / 'analysis'

    if not path_analysis.is_dir():
        return
    if not access(path_analysis, R_OK):
        return
    
    print(path_analysis)

    return path_analysis


def find_panddas(path_analysis):
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

def find_inspect_csv(paths_panddas):

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

        
def find_csv_from_system_path(path_system):
    path_analysis = find_path_analysis(path_system)

    if path_analysis == None:
        return []
    
    paths_panddas = find_panddas(path_analysis)
    csvs = find_inspect_csv(paths_panddas)

    print(csvs)

    return csvs

def find_csvs_from_year(path_year):
    path_year = pathlib.Path(path_year)
    csvs = []
    paths_system = [x for x in path_year.iterdir() if x.is_dir()]

    for path in paths_system:
        csv_paths = find_csv_from_system_path(path)
        csvs += csv_paths
    
    return csvs

def find_all_csvs(path_data):
    path_data = pathlib.Path(path_data)
    
    csvs = []
    paths_year = [x for x in path_data.iterdir() if x.is_dir()]

    for path_year in paths_year:
        csv_paths = find_csvs_from_year(path_year)
        csvs += csv_paths
    
    print(csvs)
    print(len(csvs))
    
    return csvs

def filter_csvs(csvs):

    for path in csvs:
        df = pd.read_csv(path)
        df = df.loc[(df['Ligand Placed']==True) & (df['Ligand Confidence']=='High')]
        arr = df['Ligand Placed'].to_numpy()
        sum = np.sum(arr)
        if sum == 0:
            csvs.remove(path)
    
    print(csvs)
    print(len(csvs))
    return csvs

def list_pandda_model_paths(csvs):
    models = []
    colnames = ['csv_path', 'panddas_path', 'model_building']

    for path_csv in csvs:
        panddas_path = path_csv.parent.parent
        path_analysis = panddas_path.parent
        path_model_building = path_analysis / 'model_building'

        if not path_model_building.is_dir():
            path_model_building = path_analysis / 'initial_model'
        
        models.append([path_csv, panddas_path, path_model_building])

    df_models = pd.DataFrame(models, columns=colnames)
    
    print(df_models)
    
    return df_models
        
def find_events_per_dataset(csv_path, panddas_path, model_building):
    csv_path = pathlib.Path(csv_path)
    panddas_path = pathlib.Path(panddas_path)
    model_building = pathlib.Path(model_building)

    event_map = []
    mtz = []
    input_model = []
    output_model = []
    
    pandda_inspect = pd.read_csv(csv_path)
    pandda_inspect = pandda_inspect.loc[(pandda_inspect['Ligand Placed']==True) & (pandda_inspect['Ligand Confidence']=='High')]
    pandda_inspect = pandda_inspect[['dtag','event_idx', 'x', 'y', 'z', '1-BDC', 'high_resolution','Ligand Placed', 'Ligand Confidence']]
    print(pandda_inspect)

    for i, row in enumerate(pandda_inspect.itertuples()):
        print(row)
        event_path = panddas_path / 'processed_datasets' / f'{row.dtag}' / f'{row.dtag}-event_{row.event_idx}_1-BDC_{row._6}_map.native.ccp4'
        mtz_path = panddas_path / 'processed_datasets'  / f'{row.dtag}' / f'{row.dtag}-pandda-input.mtz'
        input_model_path = panddas_path / 'processed_datasets' / f'{row.dtag}' / f'{row.dtag}-pandda-input.pdb'
        output_model_path = model_building / f'{row.dtag}' / f'{row.dtag}-pandda-model.pdb'

        # if not (event_path.is_file() and mtz_path.is_file() and input_model_path.is_file() and output_model_path.is_file()):
        #     pandda_inspect = pandda_inspect.drop(pandda_inspect.index[i])
        #     continue
            
        event_map.append(event_path)
        mtz.append(mtz_path)
        input_model.append(input_model_path)
        output_model.append(output_model_path)

    pandda_inspect['event_map'] = event_map
    pandda_inspect['mtz'] = mtz
    pandda_inspect['input_model'] = input_model
    pandda_inspect['output_model'] = output_model

    return pandda_inspect

def find_events_all_datasets(df_models):
    df_pandda_inspect = pd.DataFrame(columns=['dtag','event_idx', 'x', 'y', 'z', '1-BDC', 'high_resolution','Ligand Placed', 'Ligand Confidence','event_map','mtz','input_model','output_model'])
    for row in df_models.itertuples():
        df = find_events_per_dataset(row.csv_path, row.panddas_path, row.model_building)
        df_pandda_inspect = pd.concat([df_pandda_inspect,df])

    print(df_pandda_inspect)

    outfname = pathlib.Path.cwd() / 'training' / 'model_paths.csv'
    df_pandda_inspect.reset_index(drop=True)
    df_pandda_inspect.to_csv(outfname)

    return df_pandda_inspect

def filter_non_existent_paths(df_pandda_inspect):
    drop = []
    for i, row in enumerate(df_pandda_inspect.itertuples()):
        event_map = pathlib.Path(row.event_map)
        mtz = pathlib.Path(row.mtz)
        input_model = pathlib.Path(row.input_model)
        output_model = pathlib.Path(row.output_model)

        if not (event_map.is_file() and mtz.is_file() and input_model.is_file() and output_model.is_file()):
            drop.append(i)

    df_pandda_inspect = df_pandda_inspect.drop(df_pandda_inspect.index[drop])

    return df_pandda_inspect




    


def calc_rmsd_per_chain(chain_input, chain_output):
    rmsd = []
    residue_input_idx = []
    residue_output_idx = []
    residue_name = []

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
    chain, chain_idx = rmsdcalc.map_chains(model_input, model_output)
    input_chain_idx = []
    output_chain_idx = []
    rmsd_1 = []
    residue_input_idx_1 = []
    residue_output_idx_1 = []
    residue_name_1 = []

    for pair, pair_idx in zip(chain, chain_idx):
        rmsd, residue_input_idx, residue_output_idx, residue_name = calc_rmsd_per_chain(pair[0], pair[1])

        if bool(rmsd)==True: # if rmsd list is not empty
            input_chain_idx += [pair_idx[0]]*len(rmsd)
            output_chain_idx += [pair_idx[1]]*len(rmsd)
            rmsd_1 += rmsd
            residue_input_idx_1 += residue_input_idx
            residue_output_idx_1 += residue_output_idx
            residue_name_1 += residue_name
    
    return input_chain_idx, output_chain_idx, residue_input_idx_1, residue_output_idx_1, residue_name_1, rmsd_1



def calc_rmsds_from_csv(df_pandda_inspect):
    dict = {
        'dtag': [],
        'event_idx': [], 
        'x': [], 
        'y': [], 
        'z': [],
        '1-BDC': [], 
        'high_resolution': [],
        'Ligand Placed': [], 
        'Ligand Confidence': [],
        'event_map': [],
        'mtz': [],
        'input_model': [],
        'output_model': [], 
        'input_chain_idx': [], 
        'output_chain_idx': [], 
        'residue_input_idx': [], 
        'residue_output_idx': [], 
        'residue_name': [], 
        'rmsd': []

    }

    for row in df_pandda_inspect.itertuples():
        input = gemmi.read_structure(str(row.input_model))[0]
        output = gemmi.read_structure(str(row.output_model))[0]

        input_chain_idx, output_chain_idx, residue_input_idx, residue_output_idx, residue_name, rmsd = calc_rmsd_per_model(input, output)
        dict['dtag'] += [row.dtag]*len(rmsd)
        dict['event_idx'] += [row.event_idx]*len(rmsd)
        dict['x'] += [row.x]*len(rmsd)
        dict['y'] += [row.y]*len(rmsd)
        dict['z'] += [row.z]*len(rmsd)
        dict['1-BDC'] += [row._6]*len(rmsd)
        dict['high_resolution'] += [row.high_resolution]*len(rmsd)
        dict['Ligand Placed'] += [row._8]*len(rmsd)
        dict['Ligand Confidence'] += [row._9]*len(rmsd)
        dict['event_map'] += [row.event_map]*len(rmsd)
        dict['mtz'] += [row.mtz]*len(rmsd)
        dict['input_model'] += [row.input_model]*len(rmsd)
        dict['output_model'] += [row.output_model]*len(rmsd)
        dict['input_chain_idx'] += input_chain_idx
        dict['output_chain_idx'] += output_chain_idx
        dict['residue_input_idx'] += residue_input_idx
        dict['residue_output_idx'] += residue_output_idx
        dict['residue_name'] += residue_name
        dict['rmsd'] += rmsd

    df_residues = pd.DataFrame.from_dict(dict)
    print(df_residues)
    return df_residues

def find_remodelled_residues(df_residues, threshold=0.8):
    rmsd = df_residues['rmsd'].to_numpy()
    remodelled = rmsd > threshold

    df_residues['remodelled'] = remodelled
    print(df_residues)
    df_residues.drop_duplicates()
    print(df_residues)
    outfname = pathlib.Path.cwd() / 'training' / 'residues.csv'
    df_residues.to_csv(outfname)

    return df_residues

def filter_remodelled_residues(df_residues):
    df_residues = df_residues.loc[df_residues['remodelled']==True]
    print(df_residues)
    outfname = pathlib.Path.cwd() / 'training' / 'training_data.csv'
    df_residues.to_csv(outfname)
    return df_residues

def find_contacts(df_residues):
    dict = {
            'dtag': [],
            'event_idx': [], 
            'x': [], 
            'y': [], 
            'z': [],
            '1-BDC': [], 
            'high_resolution': [],
            'Ligand Placed': [], 
            'Ligand Confidence': [],
            'event_map': [],
            'mtz': [],
            'input_model': [],
            'output_model': [], 
            'input_chain_idx': [], 
            'output_chain_idx': [], 
            'residue_input_idx': [], 
            'residue_output_idx': [], 
            'residue_name': [], 
            'rmsd': []

        }

    df_remodelled = df_residues.loc[df_residues['remodelled']==True]

    for row in df_remodelled.itertuples():
        structure = gemmi.read_structure(str(row.output_model))
        residue = structure[0][row.output_chain_idx][row.residue_output_idx]

        contact_list = contact_search.find_contacts_per_residue(structure, residue)

        print(contact_list)

        for contact in contact_list:
            chain_idx = contact[0]
            residue_idx = contact[1]
            res_name = contact[2]
            contact_row = df_residues.loc[
                (df_residues['remodelled']==False) &           
                (df_residues['input_model']==row.input_model) &
                (df_residues['x']==row.x) &
                (df_residues['y']==row.y) &
                (df_residues['z']==row.z) &
                (df_residues['output_chain_idx']==chain_idx) &
                (df_residues['residue_output_idx']==residue_idx) &
                (df_residues['residue_name']==res_name)
                ].to_dict('list')
            contact_row.pop('Unnamed: 0')
            print(contact_row)
            
            for dict_lists, contact_row_list in zip(dict.values(), contact_row.values()):
                dict_lists += contact_row_list

    df_negative_data = pd.DataFrame.from_dict(dict)
    print(df_negative_data)
    df_negative_data.drop_duplicates()
    print(df_negative_data)
    outfname = pathlib.Path.cwd() / 'training' / 'neg_data.csv'
    df_negative_data.to_csv(outfname)
    
    return df_negative_data

def find_contacts_from_contact_list(df_residues, contact_list):
    
    
    for contact in contact_list:
            chain_idx = contact[0]
            residue_idx = contact[1]
            res_name = contact[2]
            contact_row = df_residues.loc[
                (df_residues['remodelled']==False) &           (df_residues['input_model']==row.input_model) &
                (df_residues['output_chain_idx']==chain_idx) &
                (df_residues['residue_output_idx']==residue_idx) &
                (df_residues['residue_name']==res_name)
                ]
            df_negative_data = pd.concat([df_negative_data, contact_row])
            print(df_negative_data)

        



#######

if __name__ == "__main__":
    p = sys.argv[1]

    # csvs = find_all_csvs(p)
    # csvs = filter_csvs(csvs)
    # df = list_pandda_model_paths(csvs)
    # events_csv = find_events_all_datasets(df)
    # events_csv = filter_non_existent_paths(events_csv)
    # df2 = calc_rmsds_from_csv(events_csv)
    # df2 = find_remodelled_residues(df2)
    # df2 = filter_remodelled_residues(df2)

    path = pathlib.Path.cwd() / 'training' / 'residues.csv'
    res_csv = pd.read_csv(path)
    df = find_contacts(res_csv)
    

