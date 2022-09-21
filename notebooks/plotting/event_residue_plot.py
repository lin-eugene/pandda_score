import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook'
import numpy as np
import gemmi

def plot(sample):
    x,y,z = np.mgrid[0:20,0:20,0:20]
    event_residue_array = sample['event_residue_array'].numpy()
    print(sample['event_residue_array'].shape)
    print(event_residue_array.shape)
    event_map_channel = 0
    input_residue_channel = 1
    event_map_values = event_residue_array[0][event_map_channel].flatten()
    input_residue_values = event_residue_array[0][input_residue_channel].flatten()
    print(sample['labels_remodelled_yes_no'])

    fig_event_map = go.Figure(
        data=[
            go.Isosurface(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value = event_map_values,
                isomin=0.15,
                isomax=5,
                opacity=0.6,
                surface_count=3),
            go.Isosurface(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value = input_residue_values,
                isomin=0.05,
                isomax=5,
                opacity=0.4,
                surface_count=3,
                colorscale='viridis')
        ]
    )

    fig_input_residue_values = go.Figure(
        data=go.Isosurface(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value = event_map_values,
            isomin=0.05,
            isomax=1,
            opacity=0.6,
            surface_count=3)
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
                surface_count=3),
            go.Isosurface(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value = input_residue_values,
                isomin=0.05,
                isomax=5,
                opacity=0.4,
                surface_count=3,
                colorscale='viridis')
        ]
    )


    fig_event_map.show()

    fig_input_residue_values.show()
    fig3.show()

    return None

def show_metadata(sample):
    row_idx = sample['row_idx'].item()
    system = sample['system'][0]
    dtag = sample['dtag'][0]
    input_model = sample['input_model'][0]
    

    input_chain_idx = sample['input_chain_idx']
    input_residue_name = sample['input_residue_name'][0]

    st = gemmi.read_structure(input_model)[0]
    chain = st[input_chain_idx]
    chain_name = chain.name

    print(f'{row_idx=}')
    print(f'{system=}')
    print(f'{dtag=}')
    print(f'{chain_name=}')
    print(f'{input_residue_name=}')


