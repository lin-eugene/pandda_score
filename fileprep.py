import pandas as pd
import pathlib


def log_built_ligands(path_analyses_dir):
    path_panddas = path_analyses_dir / 'panddas'
    
    path_analyses = [x for x in path_panddas.iterdir() if x.is_dir() and 'analyses' in x.stem]
    path_events_csv = path_analyses[0] / 'pandda_inspect_events.csv'

    events_csv = pd.read_csv(path_events_csv)

    #extracting rows where Ligand Placed is True
    ligand_built = events_csv.loc[(events_csv['Ligand Placed']==True) & (events_csv['Ligand Confidence']=='High')]

    #extracting and saving relevant columns
    ligand_built = ligand_built[['dtag','event_idx', 'x', 'y', 'z', '1-BDC', 'high_resolution','Ligand Placed', 'Ligand Confidence']]

    #saving pandas dataframe into csv
    outfile = path_analyses[0] / 'pandda_builtligands.csv'
    ligand_built.to_csv(outfile)

    return ligand_built


def log_training_data_paths(path_analyses_dir):

    path_panddas = path_analyses_dir / 'panddas'
    path_analyses = [x for x in path_panddas.iterdir() if x.is_dir() and 'analyses' in x.stem]
    path_proc_datasets = path_panddas / 'processed_datasets'
    path_ligand_csv = path_analyses[0] / 'pandda_builtligands.csv'
    path_initial_model = path_proc_datasets.parent.parent / 'initial_model'
    if path_initial_model.is_dir():
        print('initial_model directory exists')
    elif path_initial_model.is_dir() == False:
        path_initial_model = path_analyses / 'model_building'

        if path_initial_model.is_dir() == False:
            print('no built models')

    if path_ligand_csv.is_file():
        ligand_csv = pd.read_csv(path_ligand_csv)
    else:
        print('pandda_builtligands.csv does not exist!')

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
    outfname = path_analyses[0] / 'training_data_paths.csv'
    pd_datapaths.to_csv(outfname)

    return pd_datapaths

if __name__ == "__main__":
    path_analyses_dir = pathlib.Path.cwd() / 'data' / 'testdirs' / 'MID2A' / 'processing' / 'analysis'
    ligand_built = log_built_ligands(path_analyses_dir)
    #pd_datapaths = log_training_data_paths(path_analyses_dir)

