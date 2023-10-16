import numpy as np
import pandas as pd


# TODO: change to correct data paths


def get_wt_wine():
    """
    returns white wine quality dataset from UCI
    with labels removed
    """
    dataset = pd.read_csv('.../COT-Flow/datasets/winequality-white.csv', sep=';')
    features = np.delete(dataset.values, -1, axis=1)
    return features


def get_rd_wine():
    """
    returns red wine quality dataset from UCI
    with labels removed
    """
    dataset = pd.read_csv('.../COT-Flow/datasets/winequality-red.csv', sep=';')
    features = np.delete(dataset.values, -1, axis=1)
    return features


def get_parkinson():
    """
    returns parkinson telemonitoring dataset from UCI
    with last column removed
    """
    dataset = pd.read_csv('.../COT-Flow/datasets/parkinsons_updrs.csv', sep=',')
    features = np.delete(dataset.values, [0, 1, 2], axis=1)
    return features


def get_concrete():
    """
    returns concrete dataset from UCI
    """
    dataset = pd.read_excel('.../COT-Flow/datasets/Concrete_Data.xls').to_numpy()
    return dataset


def get_energy():
    """
    returns energy dataset from UCI
    """
    dataset = pd.read_excel('.../COT-Flow/datasets/ENB2012_data.xlsx').to_numpy()
    return dataset


def get_yacht():
    """
    returns yacht dataset from UCI
    """
    dataset = pd.read_csv('.../COT-Flow/datasets/yacht_hydrodynamics.data', delim_whitespace=True,
                          names=['Long pos', 'Prismatic coeff',
                                 'Length-displacement ratio',
                                 'Beam-draught ratio',
                                 'Length-beam ratio',
                                 'Froude number',
                                 'Residuary resistance']).to_numpy()
    return dataset


def normalize_data(data):
    """
    returns normalized dataset and the associated mean, std
    """
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    data_norm = (data - mean) / std
    return data_norm


def get_correlation_numbers(data):
    """
    :param data: dataset
    :return: count of Pearson correlation greater than 0.98
    """
    data_trans = np.transpose(data)
    C = np.corrcoef(data_trans)
    Clarge = C > 0.98
    counts = np.sum(Clarge, axis=1)
    return counts


def process_data(data):
    """
    :param data: dataset
    :return: removes highly correlated columns
    """
    counts = get_correlation_numbers(data)
    while any(counts > 1):
        col_indx = np.where(counts > 1)[0][0]
        data = np.delete(data, col_indx, axis=1)
        counts = get_correlation_numbers(data)
    return data

