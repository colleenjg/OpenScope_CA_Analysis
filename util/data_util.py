"""
gen_nn_util.py

This module contains basic pytorch neural network tools.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 2.7.

"""
import os

import torch
import torch.utils.data
import numpy as np

import gen_util, math_util


#############################################
class Custom_ds(torch.utils.data.TensorDataset):
    """
    The Custom_ds object is a TensorDataset object. It takes data and optionally
    corresponding targets and initializes a custom TensorDataset.
    """

    def __init__(self, data, target=None):
        """
        self.__init__(data)

        Creates the new Custom_ds object using the specified data array, and
        optionally corresponding targets.

        Initializes data, target and n_samples attributes.

        Required arguments:
            - data (nd array): array of dataset datapoints, where the first
                               dimension is the samples.

        Optional arguments:
            - target (nd array): array of targets, where the first dimension
                                 is the samples. Must be of the same length 
                                 as data.
                                 default: None
        """
        self.data = data
        self.target = target
        self.n_samples = self.data.shape[0]

        if self.target is not None and (len(self.data) != len(self.target)):
            raise IOError('Data and Target must be of the same length.')
    
    def __len__(self):
        """
        self.__len__()

        Returns length of dataset, i.e. number of samples.

        Returns:
            - n_samples (int): length of dataset, i.e., nbr of samples.
        """
        return self.n_samples
    
    def __getitem__(self, index):
        """
        self.__getitem__()

        Returns data point and target, if not None, corresponding to index
        provided.

        Required arguments:
            - index (int): index

        Returns:
            - (torch Tensor): data at specified index
            
            if self.target is not None:
            - (torch Tensor): target at specified index
        """
        if self.target is not None:
            return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])
        else:
            return torch.Tensor(self.data[index])

#############################################
def data_indices(n, train_n, val_n, test_n=None):
    """
    data_indices(n, train_n, val_n)

    Assigns dataset indices randomly to training, validation and testing sets.
    Allows for a set to be empty, and also allows for only a subset of all 
    indices to be assigned if test_n is provided.

    Required arguments:
        - n (int)      : length of dataset
        - train_n (int): nbr of indices to assign to training set
        - val_n (int)  : nbr of indices to assign to validation set

    Optional arguments:
        - test_n (int): nbr of indices to assign to test set. If test_n is
                        None, test_n is inferred from n, train_n and val_n
                        so that all indices are assigned.

    Returns:
        - train_idx (list): unsorted list of indices assigned to training set.
        - val_idx (list)  : unsorted list of indices assigned to validation set.
        - test_idx (list) : unsorted list of indices assigned to test set. 
    """

    if test_n is None:
        test_n = n - train_n - val_n
   
    mixed_idx = range(n)
    np.random.shuffle(mixed_idx)

    train_idx = mixed_idx[0:train_n]
    val_idx = mixed_idx[train_n:train_n+val_n]
    test_idx = mixed_idx[train_n+val_n:train_n+val_n+test_n]

    return train_idx, val_idx, test_idx


#############################################
def check_prop(train_p, val_p=0, test_p=0):
    """
    check_prop(train_p)

    Checks that the proportions assigned to the sets are acceptable. Throws an
    error if proportions sum to greater than 1 or if a proportion is < 0. 
    Prints a warning (no error) if the sum to less than 1.
    
    Required arguments:
        - train_p (float): proportion of dataset assigned to training set

    Optional arguments:
        - val_p (float) : proportion of dataset assigned to validation set.
                          default: 0
        - test_p (float): proportion of dataset assigned to test set.
                          default: 0
    """

    set_p = [[x, y] for x, y in zip([train_p, val_p, test_p], 
             ['train_p', 'val_p', 'test_p'])]
    
    sum_p = sum(zip(*set_p)[0])
    min_p = min(zip(*set_p)[0])

    # raise error if proportions sum to > 1 or if a proportion is < 0.
    if sum_p != 1.0 or min_p < 0.0:
        props = ['\n{}: {}'.format(y, x) for x, y in set_p]
        prop_str = '{}\nsum_p: {}'.format(''.join(props), sum_p)
        
        if min_p < 0.0:
            raise ValueError('Proportions must not be < 0. {}'.format(prop_str))

        elif sum_p > 1.0:
            raise ValueError('Proportions must not sum to > 1. {}'.format(prop_str))
    
        elif len(set_p) == 3:
        # if all values are given and sum != 1.0
            print('WARNING: proportions given do not sum to 1. {}'.format(prop_str))


#############################################
def split_idx(n, train_p=0.75, val_p=None, test_p=None, thresh=10):
    """
    split_idx(n)

    Splits dataset indices into training, validation and test sets. If val_p 
    and test_p are None, the non training proportion is split between them.

    Required arguments:
        - n (int)      : length of dataset

    Optional arguments:
        - train_p (float): proportion of dataset assigned to training set
                           default: 0.75
        - val_p (float)  : proportion of dataset assigned to validation set. If 
                           None, proportion is calculated based on train_p and
                           test_p.
                           default: None
        - test_p (float) : proportion of dataset assigned to test set. If 
                           None, proportion is calculated based on train_p and
                           val_p.
                           default: None
        - thresh (int)   : size threshold for sets beneath which an error is
                           thrown if the set's proportion is not 0.
                           default: 10

    Returns:
        - train_idx (list): unsorted list of indices assigned to training set.
        - val_idx (list)  : unsorted list of indices assigned to validation set.
        - test_idx (list) : unsorted list of indices assigned to test set. 
    """
    
    if val_p is None and test_p is None:
        # split half half
        val_p = (1.0-train_p)/2
        test_p = val_p
    elif val_p is None:
        val_p = 1.0-train_p-test_p
    else:
        test_p = 1.0-train_p-val_p

    check_prop(train_p, val_p, test_p)
    
    val_n = int(np.round(val_p*n))
    test_n = int(np.round(test_p*n))
    train_n = n - val_n - test_n

    # raise error if val or test n is below threshold (unless prop is 0)
    for set_n, set_p, name in zip([val_n, test_n], [val_p, test_p], ['val n', 'test n']):
        if set_n < thresh:
            if set_p != 0:
                raise ValueError(('{} is {} (below threshold '
                                  'of {})').format(set_n, name, thresh))

    train_idx, val_idx, test_idx = data_indices(n, train_n, val_n, test_n)

    return train_idx, val_idx, test_idx


#############################################
def split_data(data, set_idxs):
    """
    split_data(data, set_idxs)

    Splits data (or targets) into training, validation and test sets.

    Required arguments:
        - data (nd array)       : array, where the first dimension is the 
                                  samples.
        - set_idxs (nested list): nested list of indices structured as:
                                  set (train, val, test) x indx

    Returns:
        - sets (list of torch Tensors): list of torch Tensors containing the 
                                        data for the train, val and test sets
                                        respectively.
                                        If a group is empty, None is used
                                        instead of an empty tensor.
    """

    sets = []
    for set_idx in set_idxs:
        if len(set_idx) > 0:
            sets.append(torch.Tensor(data[set_idx]))
        else:
            sets.append(None)
    
    return sets


#############################################
def init_dl(data, targ=None, batch_size=200, shuffle=False):
    """
    init_dl(data)

    Initializes a torch DataLoader.

    Required arguments:
        - data (nd array): array of dataset datapoints, where the first
                           dimension is the samples.

    Optional arguments:
        - target (nd array): array of targets, where the first dimension
                             is the samples. Must be of the same length 
                             as data.
                             default: None
        - batch_size (int) : nbr of samples dataloader will load per batch
                             default: 200
        - shuffle (bool)   : if True, data is reshuffled at each epoch
                             default: False

    Returns:
        - dl (torch DataLoader): torch DataLoader. If data is None, dl is None. 
    """

    if data is None:
        dl = None
    else:
        dl = torch.utils.data.DataLoader(Custom_ds(data, targ), 
                                         batch_size=batch_size, 
                                         shuffle=shuffle)
    return dl


#############################################
def create_dls(data, targets=None, train_p=0.75, val_p=None, test_p=None, 
               norm_dim=None, shuffle=False, batch_size=200, thresh=10,
               train_shuff=True):
    """
    create_dls(data)

    Creates torch DataLoaders for each set (training, validation, test).
    
    If a normalization dimension is passed, each set is normalized based on
    normalization factors calculated on the training set and the normalization
    factors are also returned.

    If shuffle is True, targets are shuffled for each dataset and the shuffled 
    indices are also returned.

    Required arguments:
        - data (nd array): array of dataset datapoints, where the first
                           dimension is the samples.

    Optional arguments:
        - target (nd array) : array of targets, where the first dimension
                              is the samples. Must be of the same length 
                              as data.
                              default: None
        - train_p (float)   : proportion of dataset assigned to training set
                              default: 0.75
        - val_p (float)     : proportion of dataset assigned to validation set. If 
                              None, proportion is calculated based on train_p and
                              test_p.
                              default: None
        - test_p (float)    : proportion of dataset assigned to test set. If 
                              None, proportion is calculated based on train_p and
                              val_p.
                              default: None
        - norm_dim (int)    : data array dimension along which to normalize data 
                              (int, None, 'last', 'all')
                              default: None
        - shuffle (bool)    : if True, targets are shuffled in all sets to 
                              create randomized datasets.
                              default: False
        - batch_size (int)  : nbr of samples dataloader will load per batch
                              default: 200
        - thresh (int)      : size threshold for sets beneath which an error is
                              thrown if the set's proportion is not 0.
                              default: 10
        - train_shuff (bool): if True, training data is set to be reshuffled at 
                              each epoch
                              default: True

    Returns:
        - returns (list): 
            - dls (list of torch DataLoaders): list of torch DataLoaders for 
                                               each set. If a set is empty, the 
                                               corresponding dls value is None.
            Optional:
            if norm_dim is not None:
                - norm_facts (nested list): list of normalization factors
                                            structured as 
                                            stat (mean, std) x vals
            if shuffle:
                - shuff_reidx (list): list of indices with which targets were
                                      shuffled 
                
    """


    # data: samples x []
    set_idxs = split_idx(n=len(data), train_p=train_p, val_p=val_p, 
                         test_p=test_p, thresh=thresh)

    set_data = split_data(data, set_idxs)
    
    returns = []
    if norm_dim not in [None, 'None', 'none']:
        train_means, train_stds = math_util.norm_facts(set_data[0], dim=norm_dim)
        for i in range(len(set_data)):
            set_data[i] = (set_data[i] - train_means)/train_stds
        norm_facts = [train_means.tolist(), train_stds.tolist()]
        returns.append(norm_facts)
    
    if targets is not None:
        if shuffle:
            shuff_reidx = range(len(targets))
            np.random.shuffle(shuff_reidx)
            returns.append(shuff_reidx)
            targets = targets[shuff_reidx]
    
        set_targets = split_data(targets, set_idxs)
    else:
        set_targets = [None] * 3
    
    dls = []
    for i, (data, targ) in enumerate(zip(set_data, set_targets)):
        if train_shuff and i == 0:
            shuff = True
        else:
            shuff = False
        dls.append(init_dl(data, targ, batch_size, shuff))

    returns = [dls] + returns

    return returns
