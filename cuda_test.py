import logging
import torch

from torch.utils.data import DataLoader
import pathlib
import pandas as pd
from learning.torch_data_setup import *
import os
from learning.models import SqueezeNet

device = "cuda" if torch.cuda.is_available() else "cpu"

training_csv = pathlib.Path.cwd() / "training" / "training_set.csv"
test_csv = pathlib.Path.cwd() / "training" / "test_set.csv"
training_dframe = pd.read_csv(training_csv)
test_dframe = pd.read_csv(test_csv)

training_dataset = generate_dataset(residues_dframe=training_dframe)
test_dataset = generate_dataset(residues_dframe=test_dframe)

training_dataloader = DataLoader(dataset=training_dataset, 
                                batch_size=1, 
                                shuffle=True, 
                                num_workers=os.cpu_count())
test_dataloader = DataLoader(dataset=test_dataset, 
                            batch_size=1, 
                            shuffle=True, 
                            num_workers=os.cpu_count())

model = SqueezeNet(
            kernel_size=3,
            stride=1
            )
model.to(device)

def check_cuda(training_dataloader,
                test_dataloader,
                model):
    logging.debug(f'{torch.cuda.is_available()=}')
    logging.debug(f'{next(model.parameters()).is_cuda=}')

    train_sample = next(iter(training_dataloader))
    logging.debug(f'{train_sample["event_residue_array"].is_cuda=}')
    logging.debug(f'{train_sample["labels_remodelled_yes_no"].is_cuda=}')

    test_sample = next(iter(test_dataloader))
    logging.debug(f'{test_sample["event_residue_array"].is_cuda=}')
    logging.debug(f'{test_sample["labels_remodelled_yes_no"].is_cuda=}')


check_cuda(training_dataloader, test_dataloader, model)