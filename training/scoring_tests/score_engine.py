import torch 
import logging
from typing import Dict, Any

__all__ = ['scoring_loop']

def get_labels_per_residue(event_map: str,
                            input_model: str,
                            input_chain_idx: int,
                            input_residue_idx: int,
                            residue_name: str,
                            pred_label: int):

    return {
        'event_map_name': event_map,
        'input_model': input_model,
        'input_chain_idx': input_chain_idx,
        'input_residue_idx': input_residue_idx,
        'residue_name': residue_name,
        'pred_label': pred_label,

    }


def record_labels_per_batch(batch: Dict[str, Any],
                                    test_pred_labels: torch.Tensor,):
    
    event_map_name = batch['event_map_name']
    input_model = batch['input_model']
    input_chain_idx = batch['input_chain_idx'].tolist()
    input_residue_idx = batch['input_residue_idx'].tolist()
    residue_name = batch['residue_name']
    pred_label = test_pred_labels.tolist()

    return list(map(get_labels_per_residue, 
                    event_map_name,
                    input_model,
                    input_chain_idx,
                    input_residue_idx,
                    residue_name,
                    pred_label))

def scoring_loop(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                device: torch.device):
    """
    Test function:
    same as train_step function
    but doesn't have optimiser to update parameters
    
    """

    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    output_labels = []
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for sample in dataloader:
            # Send data to target device
            event_residue_array_batch = sample['event_residue_array'].to(device)

            # 1. Forward pass
            debug_pred_logits = model(event_residue_array_batch)


            if debug_pred_logits.size(dim=1) > 1:
                debug_pred_labels = torch.argmax(torch.softmax(debug_pred_logits, dim=1), dim=1) #for multiclass classification

            else:
                debug_pred_labels = torch.round(torch.sigmoid(debug_pred_logits)) #for binary classification

            logging.debug(sample)
            output_labels += record_labels_per_batch(sample, debug_pred_labels)

    return output_labels