#engine.py

import torch
import torch.nn as nn

def loss_fn(outputs, targets):
    """
    Computes the binary cross-entropy loss between the model outputs and the target labels.

    Parameters:
        outputs (torch.Tensor): The output tensor from the model.
        targets (torch.Tensor): The target labels.

    Returns:
        torch.Tensor: The binary cross-entropy loss.
    """
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train_fn(data_loader, model, optimizer, device, scheduler):
    """
    Trains a model using the provided data loader, model, optimizer, device, and scheduler.

    Parameters:
    data_loader (DataLoader): The data loader to use for training.
    model (nn.Module): The model to train.
    optimizer (Optimizer): The optimizer to use for training.
    device (torch.device): The device to use for training.
    scheduler (Scheduler): The scheduler to use for training.

    Returns:
    None
    """

    # set model to training mode
    model.train()

    for d in data_loader:
        # extract ids, token_type_ids, masks and targets from current batch,
        # and move them to the device

        ids = d['ids'].to(device, dtype=torch.long)

        token_type_ids = d['token_type_ids'].to(device, dtype=torch.long)

        masks = d['masks'].to(device, dtype=torch.long)

        targets = d['target'].to(device, dtype=torch.float)

        # clear previous gradients
        model.zero_grad()

        # get predictions
        outputs = model(ids=ids, 
                        mask=masks,
                        token_type_ids=token_type_ids, 
                        )

        # compute the loss
        loss = loss_fn(outputs, targets)

        # compute gradients
        loss.backward()

        # update parameters
        optimizer.step()

        # update scheduler
        scheduler.step()


def eval_fn(data_loader, model, device):
    """
	Evaluates a model using the provided data loader and device.

	Parameters:
		data_loader (DataLoader): The data loader to use for evaluation.
		model (nn.Module): The model to evaluate.
		device (torch.device): The device to use for evaluation.

	Returns:
		None
    """
    # set model to evaluation mode
    model.eval()
    
    # initialize lists to store predictions and targets
    fin_targets=[]
    fin_outputs=[]

    # use no_grad context manager to disable gradient calculation
    with torch.no_grad():
        for d in data_loader:
            # extract ids, token_type_ids, masks and targets from current batch,
            # and move them to the device
            ids = d['ids'].to(device, dtype=torch.long)

            token_type_ids = d['token_type_ids'].to(device, dtype=torch.long)

            masks = d['masks'].to(device, dtype=torch.long)

            targets = d['targets'].to(device, dtype=torch.float)

            # get predictions
            outputs = model(ids=ids, 
                            masks=masks,
                            token_type_ids=token_type_ids, 
                            )
            
            # convert targets to cpu and extend the final list
            targets = targets.cpu().detach()
            fin_targets.extend(targets.numpy().tolist())

            # convert outputs to cpu and extend the final list
            outputs = torch.sigmoid(outputs).cpu().detach()
            fin_outputs.extend(outputs.numpy().tolist())


    return fin_outputs, fin_targets



    