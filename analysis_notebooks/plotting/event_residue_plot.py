import pathlib
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook'
import numpy as np
import gemmi
import subprocess 
from analysis_notebooks.plotting.confusion_matrix import read_residue_name

def plot(sample):
    x,y,z = np.mgrid[0:20,0:20,0:20]
    event_residue_array = sample['event_residue_array'].numpy()

    event_map_channel = 0
    input_residue_channel = 1
    event_map_values = event_residue_array[0][event_map_channel].flatten()
    input_residue_values = (event_residue_array[0][input_residue_channel]+0.5).flatten()

    fig_event_map = go.Figure(
        data=[
            go.Isosurface(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value = event_map_values,
                isomin=0.05,
                isomax=5,
                opacity=0.6,
                surface_count=1),
            go.Isosurface(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value = input_residue_values,
                isomin=0.05,
                isomax=5,
                opacity=0.4,
                surface_count=1,
                colorscale='viridis')
        ]
    )


    fig_event_map_only = go.Figure(
        data=go.Isosurface(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value = event_map_values,
            isomin=0.05,
            isomax=0.5,
            opacity=0.6,
            surface_count=1)
    )

    fig_residue_only = go.Figure(
        data=go.Isosurface(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value = input_residue_values,
            isomin=0.05,
            isomax=1,
            opacity=0.6,
            surface_count=1)
    )

    fig3 = go.Figure(
            data=[
            go.Isosurface(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value = event_map_values*input_residue_values,
                isomin=0.05,
                isomax=5,
                opacity=0.6,
                surface_count=1),
            go.Isosurface(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value = input_residue_values,
                isomin=0.05,
                isomax=5,
                opacity=0.4,
                surface_count=1,
                colorscale='viridis')
        ]
    )


    fig_event_map.show()
    fig_event_map_only.show()
    fig_residue_only.show()
    fig3.show()

    return None

class ShowMetadata():
    def __init__(self, sample, dataframe):
        self.sample = sample
        self.dataframe = dataframe
        
        self.load_data()

    def load_data(self):
        self.row_idx = self.sample['row_idx'].item()
        self.system = self.sample['system'][0]
        self.dtag = self.sample['dtag'][0]
        self.input_model = self.sample['input_model'][0]
        

        self.input_chain_idx = self.sample['input_chain_idx']
        self.input_residue_idx = self.sample['input_residue_idx']
        self.input_residue_name = self.sample['input_residue_name'][0]

        self.st = gemmi.read_structure(self.input_model)[0]
        self.chain = self.st[self.input_chain_idx]
        self.chain_name = self.chain.name

        row = self.dataframe.loc[self.dataframe['row_idx'] == self.row_idx]
        self.gt_label = row['labels_remodelled_yes_no']
        self.pred_label = row['pred_labels']
        self.res_type = read_residue_name(self.input_residue_name)

    def show_metadata(self):
        print(f'{self.row_idx=}')
        print(f'{self.system=}')
        print(f'{self.dtag=}')
        print(f'{self.chain_name=}')
        print(f'{self.input_residue_name=}')
        print(f'{self.gt_label=}')
        print(f'{self.pred_label=}')
        print(f'{self.res_type=}')

    def open_coot(self, show_output_model=False, show_mtz=False):
        "opens coot on Diamond Remote Desktop"
        event_map = self.sample['event_map_name'][0]
        if show_mtz:
            mtz = self.sample['mtz'][0]
        else:
            mtz = None
        
        if show_output_model:
            output_model = self.sample['output_model'][0]
        else:
            output_model = None

        # coordinates
        x = self.chain[self.input_residue_idx][0].pos.x
        print(x)
        y = self.chain[self.input_residue_idx][0].pos.y
        z = self.chain[self.input_residue_idx][0].pos.z
        script = f'set_rotation_center({x},{y},{z})'
        target_dir = pathlib.Path(__file__).parent
        script_filename = target_dir / 'coot_script.py'

        with open(script_filename, 'w') as f:
            f.write(script)
        
        # if show_output_model:
        cmd = f'module load ccp4/7.0.067 && coot --pdb {self.input_model} --pdb {output_model} --map {event_map} --data {mtz} --script {str(script_filename)}'
        # else:
        #     cmd = f'module load ccp4/7.0.067 && coot --pdb {self.input_model} --map {event_map} --script {str(script_filename)}'

        
        subprocess.Popen(cmd, shell=True)

