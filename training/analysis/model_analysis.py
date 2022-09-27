import argparse
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import torch
from training import torch_data_setup
import pandas as pd
import pathlib
from torch.utils.data import DataLoader
import os
import logging
from training.utils import load_model

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show()

#####
def get_training_results_per_residue(row_idx,
                                    system,
                                    dtag,
                                    input_model,
                                    output_model,
                                    mtz,
                                    event_map_name,
                                    one_minus_BDC,
                                    high_resolution,
                                    input_chain_idx,
                                    input_residue_idx,
                                    input_residue_name,
                                    rmsd,
                                    labels_remodelled_yes_no,
                                    test_pred_label,
                                    pred_probability):

    return {
        'row_idx': row_idx,
        'system': system,
        'dtag': dtag,
        'input_model': input_model,
        'output_model': output_model,
        'mtz': mtz,
        'event_map_name': event_map_name,
        '1-BDC': one_minus_BDC,
        'high_resolution': high_resolution,
        'input_chain_idx': input_chain_idx,
        'input_residue_idx': input_residue_idx,
        'input_residue_name': input_residue_name,
        'rmsd': rmsd,
        'labels_remodelled_yes_no': labels_remodelled_yes_no,
        'pred_labels': test_pred_label,
        'pred_probabilities': pred_probability
    }


def log_training_results_per_batch(batch: Dict[str, Any],
                                    test_pred_labels: torch.Tensor,
                                    pred_probabilities: torch.Tensor,):
    
    row_idx = batch['row_idx'].tolist()
    system = batch['system']
    dtag = batch['dtag']
    input_model = batch['input_model']
    output_model = batch['output_model']
    mtz = batch['mtz']
    event_map_name = batch['event_map_name']
    one_minus_BDC = batch['1-BDC'].tolist()
    high_resolution = batch['high_resolution'].tolist()
    input_chain_idx = batch['input_chain_idx'].tolist()
    input_residue_idx = batch['input_residue_idx'].tolist()
    input_residue_name = batch['input_residue_name']
    rmsd = batch['rmsd'].tolist()
    labels_remodelled_yes_no = batch['labels_remodelled_yes_no'].tolist()
    pred_labels = test_pred_labels.tolist()
    pred_probabilities = pred_probabilities.tolist()

    return list(map(get_training_results_per_residue, 
                    row_idx,
                    system,
                    dtag,
                    input_model,
                    output_model,
                    mtz,
                    event_map_name,
                    one_minus_BDC,
                    high_resolution,
                    input_chain_idx,
                    input_residue_idx,
                    input_residue_name,
                    rmsd,
                    labels_remodelled_yes_no,
                    pred_labels,
                    pred_probabilities))
                    

def debug_loop(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                device: torch.device):
    """
    Test function:
    same as train_step function
    but doesn't have optimiser to update parameters
    
    """

    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    debug_loss, debug_acc = 0, 0
    output_labels = []
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for sample in dataloader:
            # Send data to target device
            event_residue_array_batch = sample['event_residue_array'].to(device)
            labels_batch = sample['labels_remodelled_yes_no'].to(device)

            # 1. Forward pass
            debug_pred_logits = model(event_residue_array_batch)

            # 2. Calculate and accumulate loss
            loss = loss_fn(debug_pred_logits, labels_batch)
            debug_loss += loss.item()
            
            # Calculate and accumulate accuracy
            # print(f'test_pred_logits: {test_pred_logits}')
            if debug_pred_logits.size(dim=1) > 1:
                debug_pred_probability = torch.softmax(debug_pred_logits, dim=1)[:,1]
                debug_pred_labels = torch.argmax(torch.softmax(debug_pred_logits, dim=1), dim=1) #for multiclass classification

            else:
                debug_pred_probability = torch.sigmoid(debug_pred_logits)
                debug_pred_labels = torch.round(torch.sigmoid(debug_pred_logits)) #for binary classification

            logging.debug(sample)
            debug_acc += (debug_pred_labels == labels_batch).sum().item()/len(debug_pred_logits)
            output_labels += log_training_results_per_batch(sample, 
                                                            debug_pred_labels,
                                                            pred_probabilities=debug_pred_probability)

            
    # Adjust metrics to get average loss and accuracy per batch 
    debug_loss = debug_loss / len(dataloader)
    debug_acc = debug_acc / len(dataloader)
    return debug_loss, debug_acc, output_labels



def save_output_labels(output_labels, model_path):
    df = pd.DataFrame(output_labels)
    df['wrong_prediction'] = df['labels_remodelled_yes_no'] != df['pred_labels']
    df_wrong_predictions = df[df['wrong_prediction'] == True]
    output_dir = pathlib.Path(model_path).resolve().parent
    output_file = output_dir / 'output_labels.csv'
    output_file_wrong_predictions = output_dir / 'output_labels_wrong_predictions.csv'

    print('saving output labels to csv...')
    df.to_csv(output_file, index=False)
    df_wrong_predictions.to_csv(output_file_wrong_predictions, index=False)
    
    return None

####

def load_data(data_path: str) -> torch.utils.data.DataLoader:
    data_path = pathlib.Path(data_path).resolve()
    dataframe = pd.read_csv(data_path)
    dataset = torch_data_setup.generate_dataset(dataframe)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=4, 
                            shuffle=False,
                            num_workers=os.cpu_count())
    
    return dataloader


def main(model_path: str,
         data_path: str,):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
   # Load model
    model = load_model(model_path, device)
   
    # Load data
    dataloader = load_data(data_path)

   # Setup loss function
    loss_fn = torch.nn.CrossEntropyLoss(torch.tensor([1, 2.158]).to(device))

    _, _, output_labels = debug_loop(model, 
                                    dataloader, 
                                    loss_fn, 
                                    device)

    
    save_output_labels(output_labels, model_path)
    
    return None

#####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-d', '--data_path', type=str, required=True)
    args = parser.parse_args()

    main(args.model_path, args.data_path)