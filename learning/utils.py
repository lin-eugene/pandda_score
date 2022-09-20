"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
import logging
from datetime import datetime
import pickle
import pathlib
import torch
from typing import Union

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = pathlib.Path(target_dir).resolve()
    target_dir_path.mkdir(parents=True,
                            exist_ok=True)

    # Create model save path
    model_save_path = target_dir_path / f'{model_name}_saved.pth'

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
                f=model_save_path)
    
    return None


def save_training_results(model_results: dict,
                          target_dir: Union[str, pathlib.Path],
                          model_name: Union[str, pathlib.Path]):

    target_dir_path = pathlib.Path(target_dir).resolve()
    target_dir_path.mkdir(parents=True, exist_ok=True)
    pickle_fname = target_dir_path / f"{model_name}_results.pkl"
    print(f'Pickling results to {pickle_fname.name}')

    with open(pickle_fname, 'wb') as handle:
        pickle.dump(model_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return None

def save_hyperparameters(model_name: str,
                        target_dir: Union[str, pathlib.Path],
                        model: torch.nn.Module,
                        loss_fn: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        NUM_EPOCHS: int,
                        BATCH_SIZE: int,
                        LEARNING_RATE: float,
                        OUTPUT_LOGITS: int,
                        LOSS_FN_WEIGHTS: torch.tensor,
                        translation_radius: float):
    
    target_dir_path = pathlib.Path(target_dir).resolve()
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    txt = (f"Hypderparameters for {model_name=}: \n" 
            f"{model=} \n"
            f"{loss_fn=} \n"
            f"{optimizer=} \n"
            f"{NUM_EPOCHS=} \n"
            f"{BATCH_SIZE=} \n"
            f"{LEARNING_RATE=} \n"
            f"{OUTPUT_LOGITS=} \n"
            f"{LOSS_FN_WEIGHTS=} \n"
            f"{translation_radius=} \n")
    
    txt_save_path = target_dir_path / f'{model_name}_hyperparameters.txt'
    
    print('saving hyperparameters to: ', txt_save_path)
    with open(str(txt_save_path), 'w') as f:
        f.write(txt)
    

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

        