import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from learning.torch_data_setup import *
from learning.models import SqueezeNet
from learning.train_engine import train
import os

import pathlib
import pandas as pd

from datetime import datetime
from learning.utils import *
from timeit import default_timer as timer 

import logging
import argparse


####################
def train_main(training_csv_path: str,
               test_csv_path: str,
               model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               optimiser: torch.optim.Optimizer,
               BATCH_SIZE: int,
               NUM_EPOCHS: int,):
    
    training_csv_path = pathlib.Path(training_csv_path).resolve()
    test_csv_path = pathlib.Path(test_csv_path).resolve()
    training_dframe = pd.read_csv(training_csv_path)
    test_dframe = pd.read_csv(test_csv_path)

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

    # Train model
    model_results = train(model=model,
                        train_dataloader=training_dataloader,
                        test_dataloader=test_dataloader,
                        optimiser=optimiser,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS, 
                        device=device)
        
    return model_results, model

def print_hyperparams(model: torch.nn.Module,
                      loss_fn: torch.nn.Module,
                      optimiser: torch.optim.Optimizer,
                      NUM_EPOCHS: int,
                      BATCH_SIZE: int,
                      LEARNING_RATE: float,
                      OUTPUT_LOGITS: int,
                      LOSS_FN_WEIGHTS: torch.tensor):

    print(f"{model=}")
    print(f"{loss_fn=}")
    print(f"{optimiser=}")
    print(f"{NUM_EPOCHS=}")
    print(f"{BATCH_SIZE=}")
    print(f"{LEARNING_RATE=}")
    print(f"{OUTPUT_LOGITS=}")
    print(f"{LOSS_FN_WEIGHTS=}")

#####

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42) 
    torch.cuda.manual_seed(42)

    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=40)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-o', '--output_logits', type=int, default=2)
    parser.add_argument('-nl', '--nolog', action='store_true')

    args = parser.parse_args()

    if args.nolog:
        logging.basicConfig(leve=logging.ERROR)

    # Set hyperparameters
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    OUTPUT_LOGITS = args.output_logits
    LOSS_FN_WEIGHTS = torch.tensor([1, 2.158])

    # Define data
    training_csv_path = pathlib.Path.cwd() / "training" / "training_set.csv"
    test_csv_path = pathlib.Path.cwd() / "training" / "test_set.csv"

    # Create an instance of SqueezeNet
    model = SqueezeNet(
                kernel_size=3,
                stride=1,
                num_classes=OUTPUT_LOGITS
                )
    model.to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(weight=LOSS_FN_WEIGHTS.to(device)) #NOTE — PyTorch combines softmax and cross entropy loss in one function
    optimiser = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    print_hyperparams(model=model,
                      loss_fn=loss_fn,
                      optimiser=optimiser,
                      NUM_EPOCHS=NUM_EPOCHS,
                      BATCH_SIZE=BATCH_SIZE,
                      LEARNING_RATE=LEARNING_RATE,
                      OUTPUT_LOGITS=OUTPUT_LOGITS,
                      LOSS_FN_WEIGHTS=LOSS_FN_WEIGHTS)

    start_time = timer()

    model_results, model = train_main(training_csv_path=training_csv_path,
                                      test_csv_path=test_csv_path,
                                      model=model,
                                      loss_fn=loss_fn,
                                      optimiser=optimiser,
                                      BATCH_SIZE=BATCH_SIZE,
                                      NUM_EPOCHS=NUM_EPOCHS)
    
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")

    # Save model_results and model
    SUFFIX = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p}")
    target_dir = pathlib.Path(__file__).resolve().parent / "training_results" / SUFFIX
    model_name = f"model_{SUFFIX}"

    save_model(model=model,
               target_dir=target_dir,
               model_name=model_name)
    
    save_training_results(model_results=model_results,
                          target_dir=target_dir,
                          model_name=model_name)

    save_hyperparameters(model_name=model_name,
                            target_dir=target_dir,
                            model=model,
                            loss_fn=loss_fn,
                            optimizer=optimiser,
                            NUM_EPOCHS=NUM_EPOCHS,
                            BATCH_SIZE=BATCH_SIZE,
                            LEARNING_RATE=LEARNING_RATE,
                            OUTPUT_LOGITS=OUTPUT_LOGITS,
                            LOSS_FN_WEIGHTS=LOSS_FN_WEIGHTS)
                            