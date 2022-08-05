import pathlib
import os
import shutil
import csv
import subprocess
from Bio import SeqIO
import gemmi

def pdb2fasta(PDBfile_in, FASTA_out):
    with open(PDBfile_in, 'r') as pdb_file:
        sequence = next(SeqIO.parse(pdb_file, "pdb-atom"))
    with open(FASTA_out, 'w') as fasta_out:
        SeqIO.write(sequence, fasta_out, "fasta")

class pandda_remodel():
    def __init__(self, dir_name):
        self.sandboxdir = pathlib.Path.cwd()
        self.dir_name = dir_name
        self.path = pathlib.Path(f'{dir_name}')

        subprocess.call('chmod +x ./setup.sh', shell=True)

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
                p / 'remodelling_results' / 'phenix_rsr',
                p / 'remodelling_results' / 'buccaneer'
                ]
            
            for dir in remodel_results:
                #shutil.rmtree(str(dir))
                dir.mkdir(parents=True, exist_ok=True)
            
            #fetching input pdb paths
            self.pdbs[f"{p.name}"] = p / f'{p.parts[-1]}-pandda-input.pdb'

            #fetching resolutions
            mtz_name = p / f'{p.parts[-1]}-pandda-input.mtz'
            mtz = gemmi.read_mtz_file(str(mtz_name))
            self.resolution[f0{p.name}"] = mtz.resolution_high()

            #fetching event map paths
            for j, file in enumerate(p.iterdir()):
                if (file.suffix=='.ccp4' and events_name in file.stem and not 'aligned' in file.stem):
                    self.eventmaps.append(file)

    def generate_inputs(self):
        for i, event in enumerate(self.eventmaps):
            #make directory
            newdir = event.parent / 'remodelling_results' / 'input'
            newdir.mkdir(parents=True, exist_ok=True)

            #generating aligned map
            print(f'parsing event {str(event)}')
            print(f'generating namdinator input map for dataset {str(event)}')
            output_dir = str(newdir)
        
            event_fname = str(event.stem)
            
            print('running map2sf...')
            subprocess.call(f'phenix.map_to_structure_factors "{str(event)}" output_file_name="{output_dir}/{event_fname}.mtz"',shell=True)

            event_mtz = newdir / f'{event_fname}.mtz'
            pdb = self.pdbs[f"{event.parent.name}"]

            print('event_mtz=',event_mtz)
            print('pdb=',pdb)

            print('running mtz2map...')
            subprocess.call(f'phenix.mtz2map "{str(pdb)}" "{event_mtz}" labels=F,PHIF prefix="{event_fname}_aligned" directory="{output_dir}"',shell=True)
            
            #generating mtz for buccaneer
            print('running refmac')
            subprocess.call(f'refmac5 hklin "{event_mtz}" xyzin "{str(pdb)}" hklout "{newdir / event_fname}_refmac.mtz" xyzout "{newdir / str(event.parent.name)}_refmac.pdb" << eor', shell=True)

            #generating fasta for buccaneer
            fasta_name = newdir / f"{pdb.stem}.fasta"
            pdb2fasta(str(pdb),str(fasta_name))

    def run_phenixrsr(self):
        self.aligned_maps = []
        for i, p in enumerate(self.subdirs):
            inputs = p / 'remodelling_results' / 'input'

            for j, file in enumerate(inputs.iterdir()):
                if file.suffix == '.ccp4' and 'aligned' in file.name:
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
    
    def run_buccaneer(self):
        self.mtzs = []
        self.fasta = {}
        for i, p in enumerate(self.subdirs):
            inputs = p / 'remodelling_results' / 'input'

            for j, file in enumerate(inputs.iterdir()):
                if file.suffix == '.mtz' and 'refmac' in file.name:
                    self.mtzs.append(file)
                if file.suffix == '.fasta':
                    self.fasta[f"{p.name}"] = file

        if not self.mtzs:
            print('no input maps exist')
            pass
        else:
            for i, mtz in enumerate(self.mtzs):
                print(f'parsing event {str(mtz)}')
                print(f'running buccanneer for dataset {str(mtz)} ({i+1} of {len(self.mtzs)})')

                pdb = str(self.pdbs[f"{mtz.parent.parent.parent.name}"])
                fasta = str(self.fasta[f"{mtz.parent.parent.parent.name}"])
                pdbout = str(mtz.parent.parent / 'buccaneer' / f'{str(mtz.stem)[0:-6]}buccaneer.pdb')

                print('running buccaneer')
                buccaneer_dir = mtz.parent.parent / "buccaneer"
                print(f'cbuccaneer -mtzin "{mtz}" -pdbin "{pdb}" -pdbin-mr "{pdb}" -pdbin-sequence-prior "{pdb}" -seqin "{fasta}" -pdbout "{pdbout}" -colin-fo F,SIGF-obs -colin-phifom PHIC,FOM')
                os.chdir(str(buccaneer_dir))
                subprocess.call(f'cbuccaneer -mtzin "{mtz}" -pdbin "{pdb}" -pdbin-mr "{pdb}" -pdbin-sequence-prior "{pdb}" -seqin "{fasta}" -pdbout "{pdbout}" -colin-fo F,SIGF-obs -colin-phifom PHIC,FOM',shell=True)
                os.chdir(str(self.sandboxdir))




##############
dir_name='/Users/eugene/OneDrive - Nexus365/PhD/4. Rotation 2/code/data/Namdinator test files/MURE/' #dirname = name of directory with fragment datasets
phenix_rsr = pandda_remodel(dir_name)
#phenix_rsr.generate_inputs()
# phenix_rsr.run_phenixrsr()
phenix_rsr.run_buccaneer()
