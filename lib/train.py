import torch
from lib.exception import StopTrainingError
import numpy as np
from torch import nn
from lib.metrics import c_index
from lib.transform import mixup_dataset


def train_loop(dataloader, model, loss_fn, optimizer, device=None, required_grad=True):
    losses = []
    # for _, (feature, label) in enumerate(dataloader):
    #     current_feat = feature.to(torch.float32)
    #     current_label = label.to(torch.float32)

    #     if device:
    #         current_feat = current_feat.to(device)
    #         current_label = current_label.to(device)

    # embedded_feats = model.embed(current_feat)

    # # mixed_labels of shape (batch_size, num_stacked_tensors, num_features)
    # mixed_features, mixed_labels, lam = mixup_dataset(
    #     embedded_feats,
    #     current_label,
    #     alpha=0.02,
    #     device=device,
    # )
    # preds = model.net(mixed_features).squeeze()
    # loss = lam * loss_fn(preds, mixed_labels[:, 0, 0], mixed_labels[:, 0, 1]) + (
    #     1 - lam
    # ) * loss_fn(preds, mixed_labels[:, 1, 0], mixed_labels[:, 1, 1])

    # losses.append(loss.item() * len(current_feat))
    # if torch.isinf(loss):
    #     raise StopTrainingError("Loss is Inf!")
    # if torch.isnan(loss):
    #     raise StopTrainingError("Loss is NaN!")
    # if required_grad:
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    for patient, patches, clinical_outcomes, mask in dataloader:
        optimizer.zero_grad()
        patches = patches.to(device)
        patient = patient.to(device)
        clinical_outcomes = clinical_outcomes.to(device)
        mask = ~mask.to(device)  # invert mask for key_padding_mask
        preds = model(patches, patient, mask)
        failure_times = clinical_outcomes[:, 0]
        is_observed = clinical_outcomes[:, 1]
        loss = loss_fn(preds, failure_times, is_observed)
        losses.append(loss.item() * len(patches))
        if torch.isinf(loss):
            raise StopTrainingError("Loss is Inf!")
        if torch.isnan(loss):
            raise StopTrainingError("Loss is NaN!")
        if required_grad:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return np.mean(losses)


def train_model_with_config(
    model: nn.Module,
    optimizer,
    num_epoch,
    scheduler,
    loss_fn,
    train_loader,
    validation_loader,
    test_loader,
    device=None,
):
    avg_losses = []
    val_losses = []
    for _ in range(num_epoch):
        model.train()

        epoch_loss = train_loop(train_loader, model, loss_fn, optimizer, device)
        avg_losses.append(epoch_loss * len(train_loader) / len(train_loader.dataset))
        scheduler.step()

        # evaluate validation loss
        model.eval()
        with torch.no_grad():
            val_loss = train_loop(
                validation_loader,
                model,
                loss_fn,
                optimizer,
                device,
                required_grad=False,
            )
            val_losses.append(
                val_loss * len(validation_loader) / len(validation_loader.dataset)
            )

    # cindex on test set
    model.eval()
    collected_preds = []
    clinical_outcomes_list = []
    with torch.no_grad():
        for patient, patches, clinical_outcomes, mask in test_loader:
            patches = patches.to(device)
            patient = patient.to(device)
            clinical_outcomes = clinical_outcomes.to(device)
            mask = ~mask.to(device)  # invert mask for key_padding_mask
            preds = model(patches, patient, mask)
            collected_preds.append(preds.cpu())
            clinical_outcomes_list.append(clinical_outcomes.cpu())
            clinical_outcomes_list = torch.concat(clinical_outcomes_list, dim=0)
            clinical_outcomes_list = clinical_outcomes_list.squeeze(dim=1)
            collected_preds = torch.concat(collected_preds, dim=0)
            collected_preds = collected_preds.squeeze()
            clinical_outcomes_list = clinical_outcomes_list.squeeze()
            c_index_value = c_index(
                collected_preds,
                clinical_outcomes_list[:, 0],
                clinical_outcomes_list[:, 1],
            )

    # cindex on train set
    model.eval()
    with torch.no_grad():
        collected_preds = []
        clinical_outcomes_list = []
        for patient, patches, clinical_outcomes, mask in train_loader:
            patches = patches.to(device)
            patient = patient.to(device)
            clinical_outcomes = clinical_outcomes.to(device)
            mask = ~mask.to(device)  # invert mask for key_padding_mask
            preds = model(patches, patient, mask)
            collected_preds.append(preds.cpu())
            clinical_outcomes_list.append(clinical_outcomes.cpu())
        clinical_outcomes_list = torch.concat(clinical_outcomes_list, dim=0)
        clinical_outcomes_list = clinical_outcomes_list.squeeze()
        collected_preds = torch.concat(collected_preds, dim=0)
        collected_preds = collected_preds.squeeze()
        clinical_outcomes_list = clinical_outcomes_list.squeeze()
        train_c_index_value = c_index(
            collected_preds,
            clinical_outcomes_list[:, 0],
            clinical_outcomes_list[:, 1],
        )
    return avg_losses, val_losses, c_index_value, train_c_index_value
