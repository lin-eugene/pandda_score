from typing import List
import pandas as pd
import sys, pathlib
import argparse
codebase_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(codebase_path))
from list_systems import list_systems_in_dataset

def check_systems_in_selection(systems_list: List[str], selection: List[str]) -> bool:
    """
    checks if systems in selection are in systems list
    """
    for system in selection:
        if system not in systems_list:
            print(system)
            return False

    return True

class DatasetSplitter():
    def __init__(self, 
                dataset_csv_path: str, 
                test_selection: List[str],
                save=False):

        self.dataset_csv_path = pathlib.Path(dataset_csv_path).resolve()
        self.test_selection = test_selection

        self.split_dataset()
        
        if save:
            self.save_training_and_test_sets_to_csv()

    def split_dataset(self) -> pd.DataFrame:
        """
        splits dataset into training and testing
        """
        
        dataset_dframe = pd.read_csv(self.dataset_csv_path)
        systems_list = list_systems_in_dataset(dataset_dframe).index.to_list()
        
        test_selection = self.test_selection
        training_selection = [system for system in systems_list if system not in test_selection]

        if not check_systems_in_selection(systems_list, training_selection):
            raise ValueError('systems in training selection not in dataset')
        
        if not check_systems_in_selection(systems_list, test_selection):
            raise ValueError('systems in test selection not in dataset')
        
        self.training_set_frame = dataset_dframe[dataset_dframe['system'].isin(training_selection)]
        self.test_set_frame = dataset_dframe[dataset_dframe['system'].isin(test_selection)]


        print('training set:')
        list_systems_in_dataset(self.training_set_frame)
        print('test set:')
        list_systems_in_dataset(self.test_set_frame)

        return self.training_set_frame, self.test_set_frame

    def save_training_and_test_sets_to_csv(self) -> None:
        self.training_set_frame.to_csv(self.dataset_csv_path.parent / 'training_set.csv', index=False)
        self.test_set_frame.to_csv(self.dataset_csv_path.parent / 'test_set.csv', index=False)

        return None

######
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_csv', type=str, help='path to csv file')
    parser.add_argument('-t','--test', type=str, help='comma delimited list input')
    parser.add_argument('-s', '--save', action='store_true', help='save training and test sets to csv')

    args = parser.parse_args()
    dataset_csv = args.dataset_csv
    test_selection = [item for item in args.test.split(',')]

    DatasetSplitter(dataset_csv, test_selection, save=args.save)


if __name__ == '__main__':
    main()
