import torch
import torch.nn as nn
from tqdm.auto import tqdm
import logging

def train_step(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimiser: torch.optim.Optimizer,
                device: torch.device):

    # put model in train mode
    model.train()

    # initialise training loss and accuracy
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for sample in dataloader:
        # Send data to target device
        event_residue_array_batch = sample['event_residue_array'].to(device)
        labels_batch = sample['labels_remodelled_yes_no'].to(device)
        logging.debug(f'{event_residue_array_batch.is_cuda=}')
        logging.debug(f'{labels_batch.is_cuda=}')

        # 1. Forward pass
        label_pred = model(event_residue_array_batch)
        
        # 2. Calculate  and accumulate loss
        loss = loss_fn(label_pred, labels_batch)
        train_loss += loss.item() # .item() - returns scalar value of 1-element tensor

        # 3. Optimizer zero grad
        optimiser.zero_grad() # zero out gradients from previous step, so gradients don't accumulate

        # 4. Loss backward
        loss.backward() # computes dloss/dx gradients for every parameter x

        # 5. Optimizer step
        optimiser.step() # updates parameters based on current gradients x.grad (computed by loss.backward())
    
        # Calculate and accumulate accuracy metric across all batches
        if label_pred.size(dim=1) > 1:
            y_pred_class = torch.argmax(torch.softmax(label_pred, dim=1), dim=1) #torch.argmax — return max value of elements in tensor
        else:
            y_pred_class = torch.round(torch.sigmoid(label_pred))
        
        logging.info(f'{torch.softmax(label_pred, dim=1)=}')
        logging.info(f'{y_pred_class=}')
        logging.info(f'{labels_batch=}')

        train_acc += (y_pred_class == labels_batch).sum().item()/len(label_pred)
    
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
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
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for sample in dataloader:
            # Send data to target device
            event_residue_array_batch = sample['event_residue_array'].to(device)
            labels_batch = sample['labels_remodelled_yes_no'].to(device)

            logging.debug(f'{event_residue_array_batch.is_cuda=}')
            logging.debug(f'{labels_batch.is_cuda=}')    

            # 1. Forward pass
            test_pred_logits = model(event_residue_array_batch)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, labels_batch)
            test_loss += loss.item()
            

            # Calculate and accumulate accuracy
            # print(f'test_pred_logits: {test_pred_logits}')
            if test_pred_logits.size(dim=1) > 1:
                test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1) #for multiclass classification

            else:
                test_pred_labels = torch.round(torch.sigmoid(test_pred_logits)) #for binary classification
                
            logging.info(f'{torch.softmax(test_pred_logits, dim=1)=}')
            logging.info(f'{test_pred_labels=}')
            logging.info(f'{labels_batch=}')

            test_acc += (test_pred_labels == labels_batch).sum().item()/len(test_pred_logits)
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimiser: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device):
        
    
     # 2. Create empty results dictionary
    results = {"train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []
                }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimiser=optimiser,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                    dataloader=test_dataloader,
                                    loss_fn=loss_fn,
                                    device=device)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results