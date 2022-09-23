from typing import List
import pandas as pd
import argparse

def list_systems_in_dataset(dataset_dframe: pd.DataFrame) -> List[str]:
    """
    returns list of systems from training csv
    """
    systems = dataset_dframe['system'].value_counts()
    print(systems)
    return systems

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_csv_path', type=str, help='path to csv file')
    args = parser.parse_args()

    dataset_dframe = pd.read_csv(args.dataset_csv_path)
    list_systems_in_dataset(dataset_dframe)


if __name__=='__main__':
    main()