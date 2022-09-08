from typing import List
import pandas as pd
import argparse

def list_systems_in_dataset(dataset_csv: str) -> List[str]:
    """
    returns list of systems from training csv
    """
    dataset_dframe = pd.read_csv(dataset_csv)
    systems = dataset_dframe['system'].value_counts()
    print(systems)
    return systems

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_csv', type=str, help='path to csv file')

    args = parser.parse_args()
    dataset_csv = args.dataset_csv
    list_systems_in_dataset(dataset_csv)