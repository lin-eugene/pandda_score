import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from training.scoring_tests.structure_map_to_dataframe import structure_to_dataframe
from training.scoring_tests import score_dataset, score_utils, score_engine
from training.utils import load_model
import os

def main(pandda_input_model_path: str,
        pandda_event_map_path: str,
        nn_input_model_path: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    structure_map_frame = structure_to_dataframe(pandda_event_map_path, 
                                                pandda_input_model_path)

    # data setup
    print('reading data...')
    tsfm = transforms.Compose([score_dataset.SamplingRandomRotations(),
                                score_dataset.ConcatEventResidueToTwoChannels(),
                                score_dataset.ToTensor()])

    dataset = score_dataset.ResidueDataset(residues_dframe=structure_map_frame,
                                            transform=tsfm)
                
    dataloader = DataLoader(dataset=dataset,
                            batch_size=4,
                            shuffle=False,
                            num_workers=os.cpu_count())
    
    # load model
    print('loading model...')
    model = load_model(nn_input_model_path, device)

    # score
    print('scoring...')
    output_labels = score_engine.scoring_loop(model, 
                                            dataloader, 
                                            device)
    
    # save output
    print('saving output...')
    score_utils.save_output(output_labels, pandda_input_model_path)

    print('done, exiting...')

    return None

######

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--structure_path', type=str)
    parser.add_argument('-e','--event_map_path', type=str)
    parser.add_argument('-m', '--nn_model_path', type=str)
    args = parser.parse_args()

    main(args.structure_path,
        args.event_map_path,
        args.nn_model_path)