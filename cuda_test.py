import logging
import torch

from torch.utils.data import DataLoader
from torch import nn as nn
import pathlib
import pandas as pd
from learning.torch_data_setup import *
import os
from learning.models import SqueezeNet
from learning.train_engine import train_step, test_step

logging.basicConfig(level=logging.DEBUG)

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

model = SqueezeNet(kernel_size=3,
                    stride=1)

model.to(device)

def check_cuda(training_dataloader,
                test_dataloader,
                model,
                device):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    logging.info(f'{torch.cuda.is_available()=}')
    logging.info(f'{next(model.parameters()).is_cuda=}')

    train_test = train_step(model=model,
                            dataloader=training_dataloader,        
                            loss_fn=loss_fn,
                            optimiser=optimizer,
                            device=device)

    test_test = test_step(model=model,
                            dataloader=test_dataloader,
                            loss_fn=loss_fn,
                            device=device)


check_cuda(training_dataloader, test_dataloader, model, device)