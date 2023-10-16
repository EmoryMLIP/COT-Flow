import torch
import numpy as np
from datasets import tabulardata
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def tabloader(dataset, batch_size, test_ratio, valid_ratio, random_state):
    """
    :param dataset: dataset to load
    :param batch_size: batch size for Dataloader
    :param test_ratio: ratio for test set
    :param valid_ratio: ratio for validation set
    :param random_state: random seed for shuffling
    :return: Dataloaders for train, test, validation set
    """
    if dataset == 'wt_wine':
        data = tabulardata.get_wt_wine()
    elif dataset == 'rd_wine':
        data = tabulardata.get_rd_wine()
    elif dataset == 'parkinson':
        data = tabulardata.get_parkinson()
    elif dataset == 'concrete':
        data = tabulardata.get_concrete()
    elif dataset == 'energy':
        data = tabulardata.get_energy()
    elif dataset == 'yacht':
        data = tabulardata.get_yacht()
    else:
        raise Exception("Dataset is Incorrect")

    # removed correlated columns and standardize
    if dataset == 'rd_wine' or dataset == 'wt_wine' or dataset == 'parkinson':
        data = tabulardata.process_data(data)
        features = tabulardata.normalize_data(data)
        nTot = features.shape[0]
        # split data and convert to tensor
        train_valid, test = train_test_split(
            features, test_size=test_ratio,
            random_state=random_state
        )
        train, valid = train_test_split(
            train_valid, test_size=(nTot*valid_ratio)/train_valid.shape[0],
            random_state=random_state
        )
    else:
        nTot = data.shape[0]
        # split data and convert to tensor
        train_valid, test = train_test_split(
            data, test_size=test_ratio,
            random_state=random_state
        )
        train, valid = train_test_split(
            train_valid, test_size=(nTot * valid_ratio) / train_valid.shape[0],
            random_state=random_state
        )
        train_mean = np.mean(train, axis=0, keepdims=True)
        train_std = np.std(train, axis=0, keepdims=True)
        train = (train - train_mean) / train_std
        valid = (valid - train_mean) / train_std
        test = (test - train_mean) / train_std

    train_size = train.shape[0]

    # convert to tensor
    train_data = torch.tensor(train, dtype=torch.float32)
    valid_data = torch.tensor(valid, dtype=torch.float32)
    test_data = torch.tensor(test, dtype=torch.float32)

    # load train data
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size, shuffle=True
    )

    return train_loader, valid_loader, test_data, train_size

