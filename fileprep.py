from gc import is_finalized
from importlib.resources import path
from logging import NullHandler
from re import I
import pandas as pd
import pathlib
import os
from os import access, R_OK

from lib import rmsdcalc
import sys
import numpy as np

def directory_check(path_year: pathlib.PosixPath):
    """
    checks if directories exist within dataset path and writes into log.csv file
    """
   
    paths_system = [x for x in path_year.iterdir() if x.is_dir()]

    directories = {'system': [],
                'panddas_exist?': [],
                'initial_model_exist?': [],
                }

    for path_system in paths_system:
        path_panddas = path_system / 'processing' / 'analysis' / 'panddas'
        path_initial_model = path_system / 'processing' / 'analysis' / 'initial_model'
        path_model_building = path_system / 'processing' / 'analysis' / 'model_building'
        
        try:
            if path_system.is_dir() and access(path_system, R_OK):
                directories['system'].append(path_system)

                if path_panddas.is_dir() and access(path_panddas, R_OK):
                    path_analyses = [x for x in path_panddas.iterdir() if x.is_dir() and 'analyses' in x.stem]
                    
                    if len(path_analyses) > 0:
                        path_analyses = path_analyses[0]

                        if path_analyses.is_dir() and access(path_analyses, R_OK):
                            path_events_csv = path_analyses / 'pandda_inspect_events.csv'

                            if path_events_csv.is_file() and access(path_events_csv, R_OK):
                                
                                if os.stat(path_events_csv).st_size != 0:
                                    directories['panddas_exist?'].append(True)
                                
                                else:
                                    directories['panddas_exist?'].append('pandda_inspect_events.csv is empty')
                            
                            else:
                                directories['panddas_exist?'].append('pandda_inspect_events.csv does not exist')
                        
                        else:
                            directories['panddas_exist?'].append('no access to panddas analyses')
                    
                    else:
                        directories['panddas_exist?'].append('panddas analyses dir does not exist')
                
                else:
                    directories['panddas_exist?'].append('No Permission panddas')


                if (path_initial_model.is_dir() or path_model_building.is_dir()):
                    if access(path_initial_model, R_OK) or access(path_model_building, R_OK):
                        directories['initial_model_exist?'].append(True)
                    else:
                        directories['initial_model_exist?'].append('No Permission model_building')          
                else:
                    directories['initial_model_exist?'].append(False)
            
            else:
                directories['system'].append(path_system)
                directories['panddas_exist?'].append('No Permission')
                directories['initial_model_exist?'].append('No Permission')
        
        except PermissionError:
            directories['system'].append(path_system)
            directories['panddas_exist?'].append('No Permission')
            directories['initial_model_exist?'].append('No Permission')

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
        if event_map_path.is_file():
            data_paths['event_map'].append(event_map_path)
        else:
            data_paths['event_map'].append('')

        #add intiial mtz path
        mtz_path = path_proc_datasets / f'{row.dtag}' / f'{row.dtag}-pandda-input.mtz'
        if mtz_path.is_file():
            data_paths['mtz'].append(mtz_path)
        else:
            data_paths['mtz'].append('')

        #add input model path
        input_path = path_proc_datasets / f'{row.dtag}' / f'{row.dtag}-pandda-input.pdb'
        if input_path.is_file():
            data_paths['input_model'].append(input_path)
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

        thresh = 1
        data = [] #rmsd values for each dataset
        interest = [] #if dataset has been remodelled
        residues = [] #residues of interest for each dataset
        
        for row in dataset.itertuples():
            path_input_model = pathlib.Path(row.input_model)
            path_output_model = pathlib.Path(row.output_model)

            rmsd_dict = rmsdcalc.calc_rmsds(path_input_model, path_output_model)   
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
            
            if count > 0:
                interest.append(True)
            else:
                interest.append(False)
            
        dataset.insert(loc=6, column='rmsd', value=data)
        dataset.insert(loc=7, column='remodelled?', value=interest)
        dataset.insert(loc=8, column='remodelled_res', value=residues)
        
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

def make_training_files():
    """
    input: path to dir_check_csv
    """
    print('running make training csvs')
    python_path = pathlib.Path(__file__).resolve(strict=True).parent #fetch path of python script
    path_training = python_path / 'training'
    paths = [x for x in path_training.iterdir() if x.is_dir()]
    
    for p in paths:
        csv_path = p / 'dircheck.csv'

        if csv_path.is_file():
            df = pd.read_csv(csv_path)
            df_panddas = df[(df['panddas_exist?']==True) & (df['initial_model_exist?']==True)]
            panddas_paths = df_panddas['system'].tolist()
        
        else:
            print('csv_path file does not exist')

        for path in panddas_paths:
            print(f'generating csvs for {path}...')
            path_system = pathlib.Path(path)
            log_built_ligands(path_system)
            log_training_data_paths(path_system)
            gen_rmsds(path_system)

#######

if __name__ == "__main__":
    p = sys.argv[1]
    dirs(p)
    make_training_files()


    # directory_check(pathlib.Path.cwd() / 'data' / 'testdirs')
    # path_system = pathlib.Path.cwd() / 'data' / 'testdirs' / f'{system}' #path to dataset
    # ligand_built = log_built_ligands(path_system)
    # pd_datapaths = log_training_data_paths(path_system)
    # rmsd = gen_rmsds(path_system)

