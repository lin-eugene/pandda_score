
from typing import Dict, List
import pandas as pd
import pathlib

def save_output(output_labels: List[Dict],
                pandda_model_path: str):
    df = pd.DataFrame(output_labels)
    output_dir = pathlib.Path(pandda_model_path).resolve().parent
    output_file = output_dir / 'output_labels.csv'

    print(f'saving output labels to csv as: {output_file}..')
    df.to_csv(output_file, index=False)
    
    return None