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
logging.basicConfig(level=logging.DEBUG)

torch.manual_seed(42) 
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set number of epochs
NUM_EPOCHS = 20
BATCH_SIZE = 4

# Create an instance of SqueezeNet
model = SqueezeNet(
            kernel_size=3,
            stride=1
            )

# Setup loss function and optimizer
loss_fn = nn.BCELoss()
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
# Train model_0 
model_0_results = train(model=model, 
                        train_dataloader=training_dataloader,
                        test_dataloader=test_dataloader,
                        optimiser=optimiser,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS)

pickle_fname = f'{str(pathlib.Path.cwd() / "training_results" / "model_0_results.pkl")}_{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p}")}'
with open(pickle_fname, 'wb') as handle:
    pickle.dump(model_0_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")