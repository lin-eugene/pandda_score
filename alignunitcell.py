import pathlib
import os, sys
import shutil

from numpy import size

# os.system('. /Applications/phenix-1.20.1-4487/phenix_env.sh') #doesn't work lol

dir_name='/Users/eugene/OneDrive - Nexus365/PhD/4. Rotation 2/code/data/Namdinator test files/MURE/' #dirname = name of directory with fragment dataset
path = pathlib.Path(f'{dir_name}')

subdirs = [x for x in path.iterdir() if x.is_dir()] # list out all the subdirectories in a folder
eventmaps = []
pdbs = []

for i, p in enumerate(subdirs):
    events_name = f'{p.parts[-1]}-event'
    # print(f'parsing event {events_name}')
    for j, file in enumerate(p.iterdir()):
        if file.suffix=='.ccp4' and events_name in file.stem:
            eventmaps.append(file)
            pdbs.append(p / f'{events_name}-pandda-input.pdb')


for i, event in enumerate(eventmaps):
    print(f'parsing event {str(event)}')
    output_dir = str(event.parent)
    print(output_dir)
    print(str(event))
  
    event_fname = str(event.stem)
    
    print('running map2sf...')
    os.system(f'phenix.map_to_structure_factors "{str(event)}" output_file_name="{output_dir}/{event_fname}.mtz"')

    event_mtz = event.parent / f'{event_fname}.mtz'
    print('running mtz2map...')
    os.system(f'phenix.mtz2map "{str(pdbs[i])}" "{event_mtz}" labels=F,PHIF prefix="{event_fname}_aligned" directory="{output_dir}"')
    
    event_mtz.unlink()