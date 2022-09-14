import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from learning.torch_data_setup import *
from learning.models import SqueezeNet
from learning.train_engine import train
import os

import pathlib
import pandas as pd

from datetime import datetime
import pickle

import logging
logging.basicConfig(level=logging.INFO)

torch.manual_seed(42) 
torch.cuda.manual_seed(42)


def check_cuda(training_dataloader,
                test_dataloader,
                model):
    logging.info(f'{torch.cuda.is_available()=}')
    logging.info(f'{next(model.parameters()).is_cuda=}')

    train_sample = next(iter(training_dataloader))
    logging.info(f'{train_sample["event_residue_array"].is_cuda=}')
    logging.info(f'{train_sample["labels_remodelled_yes_no"].is_cuda=}')

    test_sample = next(iter(test_dataloader))
    logging.info(f'{test_sample["event_residue_array"].is_cuda=}')
    logging.info(f'{test_sample["labels_remodelled_yes_no"].is_cuda=}')

        

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set number of epochs
NUM_EPOCHS = 20
BATCH_SIZE = 4

# Create an instance of SqueezeNet
model = SqueezeNet(
            kernel_size=3,
            stride=1,
            num_classes=2
            )
model.to(device)


# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1, 2.158])) #NOTE — PyTorch combines softmax and cross entropy loss in one function
optimiser = torch.optim.Adam(params=model.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Load data
training_csv = pathlib.Path.cwd() / "training" / "training_set.csv"
test_csv = pathlib.Path.cwd() / "training" / "test_set.csv"
training_dframe = pd.read_csv(training_csv)
test_dframe = pd.read_csv(test_csv)

training_dataset = generate_dataset(residues_dframe=training_dframe)
test_dataset = generate_dataset(residues_dframe=test_dframe)

training_dataloader = DataLoader(dataset=training_dataset, 
                                batch_size=BATCH_SIZE, 
                                shuffle=True, 
                                num_workers=os.cpu_count())
test_dataloader = DataLoader(dataset=test_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True, 
                            num_workers=os.cpu_count())
print(f'test_dataloader length is {len(test_dataloader)}')

check_cuda(training_dataloader, test_dataloader, model)

# Train model_0 
model_0_results = train(model=model, \
                        train_dataloader=training_dataloader, \
                        test_dataloader=test_dataloader, \
                        optimiser=optimiser, \
                        loss_fn=loss_fn, \
                        epochs=NUM_EPOCHS, \
                        device=device)

training_results_path = pathlib.Path(__file__).resolve / "training_results"
training_results_path.mkdir(parents=True, exist_ok=True)
SUFFIX = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p}")
pickle_fname = training_results_path / f"model_0_results_{SUFFIX}.pkl"


with open(pickle_fname, 'wb') as handle:
    pickle.dump(model_0_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")