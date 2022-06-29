import pathlib
import os
import shutil
import csv
import numpy as np

# os.system('. /Applications/phenix-1.20.1-4487/phenix_env.sh') #doesn't work lol


class pandda_remodel():
    def __init__(self, dir_name):
        self.sandboxdir = pathlib.Path.cwd()
        self.dir_name = dir_name
        self.path = pathlib.Path(f'{dir_name}')

        self.subdirs = [x for x in self.path.iterdir() if x.is_dir()] # list out all the subdirectories in a folder

        #import event maps and associated pdbs
        self.eventmaps = []
        self.pdbs = {}
        self.resolution = {}

        for i, p in enumerate(self.subdirs):
            events_name = f'{p.parts[-1]}-event'

            #make directories
            remodel_results = [
                p / 'remodelling_results' / 'input', 
                p / 'remodelling_results' / 'phenix_rsr'
                ]
            
            for dir in remodel_results:
                #shutil.rmtree(str(dir))
                dir.mkdir(parents=True, exist_ok=True)
            
            self.pdbs[f"{p.name}"] = f'{p.parts[-1]}-pandda-input.pdb'
            infocsv_name = str(p / f'{p.parts[-1]}-info.csv')
            with open(infocsv_name, mode='r') as info:
                reader = list(csv.reader(info))
                self.resolution[f"{p.name}"]=float(reader[2][1])

            #import map and pdb file directories
            for j, file in enumerate(p.iterdir()):
                if (file.suffix=='.ccp4' and events_name in file.stem and not 'aligned' in file.stem):
                    self.eventmaps.append(file)
                                        
            print(self.pdbs)
            print(self.resolution)
    def align_input_ccp4(self):
        for i, event in enumerate(self.eventmaps):
            #make directory
            newdir = event.parent / 'remodelling_results' / 'input'
            newdir.mkdir(parents=True, exist_ok=True)

            print(f'parsing event {str(event)}')
            print(f'generating namdinator input map for dataset {str(event)}')
            output_dir = str(newdir)
        
            event_fname = str(event.stem)
            
            print('running map2sf...')
            os.system(f'phenix.map_to_structure_factors "{str(event)}" output_file_name="{output_dir}/{event_fname}.mtz"')

            event_mtz = newdir / f'{event_fname}.mtz'
            pdb = str(self.pdbs[f"{event.parent.name}"])

            print('running mtz2map...')
            os.system(f'phenix.mtz2map "{pdb}" "{event_mtz}" labels=F,PHIF prefix="{event_fname}_aligned" directory="{output_dir}"')
            
            event_mtz.unlink()
    
    def run_phenixrsr(self):
        self.aligned_maps = []
        for i, p in enumerate(self.subdirs):
            aligned_maps_dir = p / 'remodelling_results' / 'input'

            for j, file in enumerate(aligned_maps_dir.iterdir()):
                self.aligned_maps.append(file)

        if not self.aligned_maps:
            print('no aligned maps exist')
            pass
        else:
            for i, event in enumerate(self.aligned_maps):
                print(f'parsing event {str(event)}')
                print(f'running phenix_rsr for dataset {str(event)} ({i+1} of {len(self.aligned_maps)})')

                pdb = str(self.pdbs[f"{event.parent.parent.parent.name}"])
                res = self.resolution[f"{event.parent.parent.parent.name}"]

                print('running phenix_rsr')
                phenix_dir = event.parent.parent / "phenix_rsr"

                # cmd = f'phenix.real_space_refine "{pdb}" "{str(event)}" resolution={res} suffix={str(event.stem)}_rsr'
                # print(cmd)
                
                os.chdir(str(phenix_dir))
                os.system(f'phenix.real_space_refine "{pdb}" "{str(event)}" resolution={res} suffix={str(event.stem)}_rsr')
                os.chdir(str(self.sandboxdir))



##############
dir_name='/Users/eugene/OneDrive - Nexus365/PhD/4. Rotation 2/code/data/Namdinator test files/MURE/' #dirname = name of directory with fragment datasets
phenix_rsr = pandda_remodel(dir_name)
# phenix_rsr.align_input_ccp4()
phenix_rsr.run_phenixrsr()
