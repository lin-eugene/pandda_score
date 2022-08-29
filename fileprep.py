import pandas as pd
import pathlib
import os
from os import access, R_OK
import pickle
import gemmi

from lib import rmsdcalc
import sys
import numpy as np



def directory_check_from_csv(path_csv: pathlib.PosixPath):
    
    df = pd.read_csv(path_csv)
    paths_system = df['path'].tolist()

    directories = {

                'system': [],
                'panddas_exist?': [],
                'initial_model_exist?': [],
                'panddas_analyses_path': [],
                'initial_model_path': [],
                
                }
    
    for path_system in paths_system:
        path_system = pathlib.Path(path_system)
        path_panddas = path_system / 'processing' / 'analysis' / 'panddas'
        path_initial_model = path_system / 'processing' / 'analysis' / 'initial_model'
        path_model_building = path_system / 'processing' / 'analysis' / 'model_building'
        
        try:
            if path_system.is_dir() and access(path_system, R_OK):
                
                if path_panddas.is_dir() and access(path_panddas, R_OK):
                    path_analyses = [x for x in path_panddas.iterdir() if x.is_dir() and 'analyses' in x.stem]
                    path_analyses = filter_path_analyses(path_analyses)

                    if len(path_analyses) > 0:
                        path_analyses = path_analyses[0]

                        if path_analyses.is_dir() and access(path_analyses, R_OK):
                            path_events_csv = path_analyses / 'pandda_inspect_events.csv'

                            if path_events_csv.is_file() and access(path_events_csv, R_OK):
                                
                                if os.stat(path_events_csv).st_size != 0:
                                    directories['panddas_exist?'].append('Y')
                                    directories['panddas_analyses_path'].append(path_analyses)
                                
                                else:
                                    directories['panddas_exist?'].append('pandda_inspect_events.csv is empty')
                                    directories['panddas_analyses_path'].append('')
                            
                            else:
                                directories['panddas_exist?'].append('pandda_inspect_events.csv does not exist')
                                directories['panddas_analyses_path'].append('')

                        else:
                            directories['panddas_exist?'].append('no access to panddas analyses')
                            directories['panddas_analyses_path'].append('')
                    
                    else:
                        directories['panddas_exist?'].append('panddas analyses dir does not exist')
                        directories['panddas_analyses_path'].append('')
                
                else:
                    directories['panddas_exist?'].append('No Permission panddas')
                    directories['panddas_analyses_path'].append('')


                if path_initial_model.is_dir():
                    if access(path_initial_model, R_OK) :
                        directories['initial_model_exist?'].append('Y')
                        directories['initial_model_path'].append(path_initial_model)
                    else:
                        directories['initial_model_exist?'].append('No Permission model_building')
                        directories['initial_model_path'].append('')
                elif path_model_building.is_dir():
                    if access(path_model_building, R_OK):
                        directories['initial_model_exist?'].append('Y')
                        directories['initial_model_path'].append(path_model_building)
                    else:
                        directories['initial_model_exist?'].append('No Permission model_building')
                        directories['initial_model_path'].append('')

                else:
                    directories['initial_model_exist?'].append('False')
                    directories['initial_model_path'].append('')

                directories['system'].append(path_system)

            else:
                directories['system'].append(path_system)
                directories['panddas_exist?'].append('No Permission')
                directories['initial_model_exist?'].append('No Permission')
                directories['panddas_analyses_path'].append('')
                directories['initial_model_path'].append('')
        
        except PermissionError:
            directories['system'].append(path_system)
            directories['panddas_exist?'].append('No Permission')
            directories['initial_model_exist?'].append('No Permission')
            directories['panddas_analyses_path'].append('')
            directories['initial_model_path'].append('')

    print(directories)
    
    for array in directories.values():
        print(len(array))

    
    pd_dircheck = pd.DataFrame.from_dict(directories)
    python_path = pathlib.Path(__file__).resolve(strict=True).parent #fetch path of python script
    outfname = python_path / 'training' / 'manual' / 'dircheck.csv'
    outfname.parent.mkdir(parents=True, exist_ok=True)
    pd_dircheck.to_csv(outfname)


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
    outfname = pathlib.Path.cwd() / 'training' / 'training_data.csv'
    df_residues.to_csv(outfname)

    return df_residues

def filter_remodelled_residues(df_residues):
    df_residues = df_residues.loc[df_residues['remodelled']==True]
    print(df_residues)
    return df_residues




##########

def filter_path_analyses(path_analyses):
    new_paths = []

    for path in path_analyses:

        if path.is_dir() and access(path, R_OK):
            path_events_csv = path / 'pandda_inspect_events.csv'

            if path_events_csv.is_file() and access(path_events_csv, R_OK):

                if os.stat(path_events_csv).st_size != 0:
                    new_paths.append(path)

    return new_paths





def directory_check(path_year: pathlib.PosixPath):

    """
    checks if directories exist within dataset path and writes into log.csv file
    """
   
    paths_system = [x for x in path_year.iterdir() if x.is_dir()]

    directories = {
                'system': [],
                'panddas_exist?': [],
                'initial_model_exist?': [],
                'panddas_analyses_path': [],
                'initial_model_path': [],
                
                }

    for path_system in paths_system:
        path_panddas = path_system / 'processing' / 'analysis' / 'panddas'
        path_initial_model = path_system / 'processing' / 'analysis' / 'initial_model'
        path_model_building = path_system / 'processing' / 'analysis' / 'model_building'
        
        try:
            if path_system.is_dir() and access(path_system, R_OK):
                
                if path_panddas.is_dir() and access(path_panddas, R_OK):
                    path_analyses = [x for x in path_panddas.iterdir() if x.is_dir() and 'analyses' in x.stem]
                    path_analyses = filter_path_analyses(path_analyses)

                    if len(path_analyses) > 0:
                        path_analyses = path_analyses[0]

                        if path_analyses.is_dir() and access(path_analyses, R_OK):
                            path_events_csv = path_analyses / 'pandda_inspect_events.csv'

                            if path_events_csv.is_file() and access(path_events_csv, R_OK):
                                
                                if os.stat(path_events_csv).st_size != 0:
                                    directories['panddas_exist?'].append('Y')
                                    directories['panddas_analyses_path'].append(path_analyses)
                                
                                else:
                                    directories['panddas_exist?'].append('pandda_inspect_events.csv is empty')
                                    directories['panddas_analyses_path'].append('')
                            
                            else:
                                directories['panddas_exist?'].append('pandda_inspect_events.csv does not exist')
                                directories['panddas_analyses_path'].append('')

                        else:
                            directories['panddas_exist?'].append('no access to panddas analyses')
                            directories['panddas_analyses_path'].append('')
                    
                    else:
                        directories['panddas_exist?'].append('panddas analyses dir does not exist')
                        directories['panddas_analyses_path'].append('')
                
                else:
                    directories['panddas_exist?'].append('No Permission panddas')
                    directories['panddas_analyses_path'].append('')


                if path_initial_model.is_dir():
                    if access(path_initial_model, R_OK) :
                        directories['initial_model_exist?'].append('Y')
                        directories['initial_model_path'].append(path_initial_model)
                    else:
                        directories['initial_model_exist?'].append('No Permission model_building')
                        directories['initial_model_path'].append('')
                elif path_model_building.is_dir():
                    if access(path_model_building, R_OK):
                        directories['initial_model_exist?'].append('Y')
                        directories['initial_model_path'].append(path_model_building)
                    else:
                        directories['initial_model_exist?'].append('No Permission model_building')
                        directories['initial_model_path'].append('')

                else:
                    directories['initial_model_exist?'].append('False')
                    directories['initial_model_path'].append('')
                
                directories['system'].append(path_system)
            
            else:
                directories['system'].append(path_system)
                directories['panddas_exist?'].append('No Permission')
                directories['initial_model_exist?'].append('No Permission')
                directories['panddas_analyses_path'].append('')
                directories['initial_model_path'].append('')
        
        except PermissionError:
            directories['system'].append(path_system)
            directories['panddas_exist?'].append('No Permission')
            directories['initial_model_exist?'].append('No Permission')
            directories['panddas_analyses_path'].append('')
            directories['initial_model_path'].append('')

    print(directories)
    
    pd_dircheck = pd.DataFrame.from_dict(directories)
    python_path = pathlib.Path(__file__).resolve(strict=True).parent #fetch path of python script
    outfname = python_path / 'training' / path_year.name / 'dircheck.csv'
    outfname.parent.mkdir(parents=True, exist_ok=True)
    pd_dircheck.to_csv(outfname)


def log_built_ligands(path_system: pathlib.PosixPath):
    """
    input: path of 1 pandda dataset/crystal system
    logs paths of pandda maps with ligands built into csv
    """
    path_panddas = path_system / 'processing' / 'analysis' / 'panddas'

    if path_panddas.is_dir() == False:
        print(f'{path_system} panddas path not found')
    
    #read csv logging all events
    path_analyses = [x for x in path_panddas.iterdir() if x.is_dir() and 'analyses' in x.stem]
    path_analyses = filter_path_analyses(path_analyses)
    path_events_csv = path_analyses[0] / 'pandda_inspect_events.csv'
    events_csv = pd.read_csv(path_events_csv)

    #extracting rows where 'Ligand Placed' is True
    ligand_built = events_csv.loc[(events_csv['Ligand Placed']==True) & (events_csv['Ligand Confidence']=='High')]
  
    #extracting and saving relevant columns
    ligand_built = ligand_built[['dtag','event_idx', 'x', 'y', 'z', '1-BDC', 'high_resolution','Ligand Placed', 'Ligand Confidence']]

    #saving pandas dataframe into csv
    python_path = pathlib.Path(__file__).resolve(strict=True).parent #fetch path of python script
    outfile = python_path / 'training' / f'{path_system.parent.name}' / f'{path_system.name}'/ 'pandda_builtligands.csv'
    outfile.parent.mkdir(parents=True, exist_ok=True)
    ligand_built.to_csv(outfile)

    return ligand_built

def log_training_data_paths2(path_system: pathlib.PosixPath, path_panddas: pathlib.PosixPath, path_initial_model: pathlib.PosixPath):
    path_proc_datasets = path_panddas / 'processed_datasets'
    python_path = pathlib.Path(__file__).resolve(strict=True).parent #fetch path of python script

    path_ligand_csv = python_path / 'training' / f'{path_system.parent.name}'

    if path_ligand_csv.is_file():
        ligand_csv = pd.read_csv(path_ligand_csv)
    else:
        print(f'{path_system.name} pandda_builtligands.csv does not exist!')

    data_paths = {'dataset': [], 
            'event_map': [],
            'mtz': [],
            'input_model': [], 
            'output_model': []}

    for row in ligand_csv.itertuples():
        data_paths['dataset'].append(row.dtag) #add dataset name

        #add event map path
        event_map_path = path_proc_datasets / f'{row.dtag}' / f'{row.dtag}-event_{row.event_idx}_1-BDC_{row._7}_map.native.ccp4' # only selecting relevant event map from which ligand was built
        event_map_path2 = path_initial_model / f'{row.dtag}' / f'{row.dtag}-event_{row.event_idx}_1-BDC_{row._7}_map.native.ccp4'

        if event_map_path.is_file():
            data_paths['event_map'].append(event_map_path)
        elif event_map_path2.is_file():
            data_paths['event_map'].append(event_map_path2)
        else:
            data_paths['event_map'].append('')

        #add intiial mtz path
        mtz_path = path_proc_datasets / f'{row.dtag}' / f'{row.dtag}-pandda-input.mtz'
        mtz_path2 = path_initial_model / f'{row.dtag}' / f'{row.dtag}-pandda-input.mtz'
        if mtz_path.is_file():
            data_paths['mtz'].append(mtz_path)
        elif mtz_path2.is_file():
            data_paths['mtz'].append(mtz_path2)
        else:
            data_paths['mtz'].append('')

        #add input model path
        input_path = path_proc_datasets / f'{row.dtag}' / f'{row.dtag}-pandda-input.pdb'
        input_path2 = path_initial_model / f'{row.dtag}' / f'{row.dtag}-pandda-input.pdb'
        if input_path.is_file():
            data_paths['input_model'].append(input_path)
        elif input_path2.is_file():
            data_paths['input_model'].append(input_path2)
        else:
            data_paths['input_model'].append('')

        #add output model path
        output_path = path_initial_model / f'{row.dtag}' / f'{row.dtag}-pandda-model.pdb'
        if output_path.is_file():
            data_paths['output_model'].append(output_path)
        else:
            data_paths['output_model'].append('')

    pd_datapaths = pd.DataFrame.from_dict(data_paths)
    python_path = pathlib.Path(__file__).resolve(strict=True).parent #fetch path of python script
    outfname = python_path / 'training' / f'{path_system.parent.name}' / f'{path_system.name}'/ 'training_data_paths.csv'
    outfname.parent.mkdir(parents=True, exist_ok=True)
    pd_datapaths.to_csv(outfname)

    return pd_datapaths

def log_training_data_paths(path_system: pathlib.PosixPath):

    """
    input: path of pandda dataset/crystal system
    logs paths of input and output models generated from pandda.inspect
    """

    path_panddas = path_system / 'processing' / 'analysis' / 'panddas'
    path_proc_datasets = path_panddas / 'processed_datasets'
    path_initial_model = path_panddas.parent / 'initial_model'

    python_path = pathlib.Path(__file__).resolve(strict=True).parent #fetch path of python script
    path_ligand_csv = python_path / 'training' / f'{path_system.parent.name}' / f'{path_system.name}' / 'pandda_builtligands.csv'
    

    if path_initial_model.is_dir():
        print('initial_model directory exists')

    elif path_initial_model.is_dir() == False:
        path_initial_model = path_panddas.parent / 'model_building'

        if path_initial_model.is_dir() == False:
            print(f'no built models for {path_system}')

    if path_ligand_csv.is_file():
        ligand_csv = pd.read_csv(path_ligand_csv)
    else:
        print(f'{path_system.name} pandda_builtligands.csv does not exist!')

    data_paths = {'dataset': [], 
            'event_map': [],
            'mtz': [],
            'input_model': [], 
            'output_model': []}

    for row in ligand_csv.itertuples():
        data_paths['dataset'].append(row.dtag) #add dataset name

        #add event map path
        event_map_path = path_proc_datasets / f'{row.dtag}' / f'{row.dtag}-event_{row.event_idx}_1-BDC_{row._7}_map.native.ccp4' # only selecting relevant event map from which ligand was built
        event_map_path2 = path_initial_model / f'{row.dtag}' / f'{row.dtag}-event_{row.event_idx}_1-BDC_{row._7}_map.native.ccp4'

        if event_map_path.is_file():
            data_paths['event_map'].append(event_map_path)
        elif event_map_path2.is_file():
            data_paths['event_map'].append(event_map_path2)
        else:
            data_paths['event_map'].append('')

        #add intiial mtz path
        mtz_path = path_proc_datasets / f'{row.dtag}' / f'{row.dtag}-pandda-input.mtz'
        mtz_path2 = path_initial_model / f'{row.dtag}' / f'{row.dtag}-pandda-input.mtz'
        if mtz_path.is_file():
            data_paths['mtz'].append(mtz_path)
        elif mtz_path2.is_file():
            data_paths['mtz'].append(mtz_path2)
        else:
            data_paths['mtz'].append('')

        #add input model path
        input_path = path_proc_datasets / f'{row.dtag}' / f'{row.dtag}-pandda-input.pdb'
        input_path2 = path_initial_model / f'{row.dtag}' / f'{row.dtag}-pandda-input.pdb'
        if input_path.is_file():
            data_paths['input_model'].append(input_path)
        elif input_path2.is_file():
            data_paths['input_model'].append(input_path2)
        else:
            data_paths['input_model'].append('')

        #add output model path
        output_path = path_initial_model / f'{row.dtag}' / f'{row.dtag}-pandda-model.pdb'
        if output_path.is_file():
            data_paths['output_model'].append(output_path)
        else:
            data_paths['output_model'].append('')

    pd_datapaths = pd.DataFrame.from_dict(data_paths)
    python_path = pathlib.Path(__file__).resolve(strict=True).parent #fetch path of python script
    outfname = python_path / 'training' / f'{path_system.parent.name}' / f'{path_system.name}'/ 'training_data_paths.csv'
    outfname.parent.mkdir(parents=True, exist_ok=True)
    pd_datapaths.to_csv(outfname)

    return pd_datapaths

def gen_rmsds(path_system: pathlib.PosixPath):
    """
    calculates rmsds between input and output models
    """
    python_path = pathlib.Path(__file__).resolve(strict=True).parent #fetch path of python script
    path_dataset_csv = python_path / 'training' / f'{path_system.parent.name}' / f'{path_system.name}' / 'training_data_paths.csv'


    if path_dataset_csv.is_file():
        dataset_csv = pd.read_csv(path_dataset_csv)
        dataset = dataset_csv[dataset_csv['output_model'].notnull()] #input into rmsd code
        
        print(dataset)

        thresh = 1
        data = [] #rmsd values for each dataset
        chain_mapping = []
        interest = [] #if dataset has been remodelled
        residues = [] #residues of interest for each dataset
        
        for row in dataset.itertuples():
            print(row.input_model)
            path_input_model = pathlib.Path(row.input_model)
            path_output_model = pathlib.Path(row.output_model)

            rmsd_dict, chain_map = rmsdcalc.calc_rmsds(path_input_model, path_output_model)   
            bool_dict = rmsdcalc.find_remodelled(rmsd_dict, thresh)

            rmsd_list = []
            for i, dict in enumerate(rmsd_dict.values()):
                for i, val in enumerate(dict.values()):
                    rmsd_list.append(val)


            rmsd_list = np.array(rmsd_list)
            bool = rmsd_list > thresh
            count = np.sum(bool)

            data.append(rmsd_dict)
            residues.append(bool_dict)
            chain_mapping.append(chain_map)
            
            if count > 0:
                interest.append(True)
            else:
                interest.append(False)
            
        dataset.insert(loc=6, column='rmsd', value=data)
        dataset.insert(loc=7, column='remodelled?', value=interest)
        dataset.insert(loc=8, column='remodelled_res', value=residues)
        dataset.insert(loc=9, column='chain_mapping', value=chain_mapping)
        
        outfname = python_path / 'training' / f'{path_system.parent.name}' / f'{path_system.name}' / 'rmsd.csv'
        outfname.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(outfname)

    else:
        print('training_data_paths.csv file not found!')


###############
def dirs(path_str: str):
    path = pathlib.Path(path_str)
    paths_year = [x for x in path.iterdir() if x.is_dir() and len(x.name)==4]
    for year in paths_year:
        directory_check(year)

def dir_check_csv():
    python_path = pathlib.Path(__file__).resolve(strict=True).parent #fetch path of python script
    path_csv = python_path / 'nonstandard_paths.csv'
    directory_check_from_csv(path_csv)

def make_training_files():
    """
    input: path to dir_check_csv
    """
    print('running make training csvs')
    python_path = pathlib.Path(__file__).resolve(strict=True).parent #fetch path of python script
    path_training = python_path / 'training'
    paths = [x for x in path_training.iterdir() if x.is_dir()]
    print(paths)
    
    for p in paths:
        csv_path = p / 'dircheck.csv'

        print(csv_path)

        if csv_path.is_file():
            df = pd.read_csv(csv_path)
            df_panddas = df[(df['panddas_exist?']=='Y') & (df['initial_model_exist?']=='Y')]

            print(df_panddas)
            panddas_paths = df_panddas['system'].tolist()

            print(panddas_paths)
        
        else:
            print('csv_path file does not exist')

        for path in panddas_paths:
            print(f'generating csvs for {path}...')
            path_system = pathlib.Path(path)
            log_built_ligands(path_system)
            log_training_data_paths(path_system)
            gen_rmsds(path_system)
    
    for p in paths:
        csv_path = python_path / 'training'

        if csv_path.is_file():
            df =  pd.read_csv(csv_path)
            df_panddas = df[(df['panddas_exist?']=='Y') & (df['initial_model_exist?']=='Y')]

            print(df_panddas)
            panddas_paths = df_panddas['system'].tolist()

        for path in panddas_paths:
            print(f'generating csvs for {path}...')
            path_system = pathlib.Path(path)
            log_built_ligands(path_system)
            log_training_data_paths(path_system)
            gen_rmsds(path_system)

        


#######

if __name__ == "__main__":
    p = sys.argv[1]
    # dirs(p)
    # dir_check_csv()
    # make_training_files()
    csvs = find_all_csvs(p)
    csvs = filter_csvs(csvs)
    df = list_pandda_model_paths(csvs)
    events_csv = find_events_all_datasets(df)
    events_csv = filter_non_existent_paths(events_csv)
    df2 = calc_rmsds_from_csv(events_csv)
    df2 = find_remodelled_residues(df2)
    df2 = filter_remodelled_residues(df2)
    




    print('data prep completed!')


    # directory_check(pathlib.Path.cwd() / 'data' / 'testdirs')
    # path_system = pathlib.Path.cwd() / 'data' / 'testdirs' / f'{system}' #path to dataset
    # ligand_built = log_built_ligands(path_system)
    # pd_datapaths = log_training_data_paths(path_system)
    # rmsd = gen_rmsds(path_system)

