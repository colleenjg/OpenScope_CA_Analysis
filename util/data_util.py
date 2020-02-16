"""
data_util.py

This module contains basic pytorch dataset tools.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import os

import numpy as np
import torch
import torch.utils.data

from util import gen_util, math_util


#############################################
class CustomDs(torch.utils.data.TensorDataset):
    """
    The CustomDs object is a TensorDataset object. It takes data and 
    optionally corresponding targets and initializes a custom TensorDataset.
    """

    def __init__(self, data, targets=None):
        """
        self.__init__(data)

        Returns a CustomDs object using the specified data array, and
        optionally corresponding targets.

        Initializes data, targets and n_samples attributes.

        Required args:
            - data (nd array): array of dataset datapoints, where the first
                               dimension is the samples.

        Optional args:
            - targets (nd array): array of targets, where the first dimension
                                  is the samples. Must be of the same length 
                                  as data.
                                  default: None
        """

        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(targets)
        self.n_samples = self.data.shape[0]

        if self.targets is not None and (len(self.data) != len(self.targets)):
            raise ValueError('data and targets must be of the same length.')
    
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

        Returns data point and targets, if not None, corresponding to index
        provided.

        Required args:
            - index (int): index

        Returns:
            - (torch Tensor): data at specified index
            
            if self.targets is not None:
            - (torch Tensor): targets at specified index
        """

        if self.targets is not None:
            return [self.data[index], self.targets[index]]
        else:
            return torch.Tensor(self.data[index])


#############################################
def bal_classes(data, targets):
    """
    bal_classes(data, targets)

    Returns resampled data arrays where classes are balanced.

    Required args:
        - data (nd array)   : array of dataset datapoints, where the first
                              dimension is the samples.
        - targets (nd array): array of targets, where the first dimension
                              is the samples. Must be of the same length as 
                              data.

    Returns:
        - data (nd array)   : array of sampled dataset datapoints, where the
                              first dimension is the samples.
        - targets (nd array): array of sampled targets, where the first 
                              dimension is the samples.
    """
    

    if len(data) != len(targets):
        raise ValueError('data and targets must be of the same length.')

    cl_n   = np.unique(targets).tolist()
    counts = np.unique(targets, return_counts=True)[1]
    
    count_min = np.min(counts)
    
    sample_idx = []
    for cl in cl_n:
        idx = np.random.choice(np.where(targets==cl)[0], count_min, 
                               replace=False)
        sample_idx.extend(idx.tolist())
    
    sample_idx = sorted(sample_idx)

    data = data[sorted(sample_idx)]
    targets = targets[sample_idx]

    return data, targets


#############################################
def data_indices(n, train_n, val_n, test_n=None, targets=None, thresh_cl=2, 
                 strat_cl=True):
    """
    data_indices(n, train_n, val_n)

    Returns dataset indices assigned randomly to training, validation and 
    testing sets.
    Allows for a set to be empty, and also allows for only a subset of all 
    indices to be assigned if test_n is provided.

    Will keep shuffling until each non empty set contains the minimum number of 
    occurrences per class.

    Required args:
        - n (int)      : length of dataset
        - train_n (int): nbr of indices to assign to training set
        - val_n (int)  : nbr of indices to assign to validation set

    Optional args:
        - test_n (int)      : nbr of indices to assign to test set. If test_n 
                              is None, test_n is inferred from n, train_n and 
                              val_n so that all indices are assigned.
                              default: None
        - targets (nd array): array of targets, where the first dimension
                              is the samples. Must be of the same length 
                              as data.
                              default: None
        - thresh_cl (int)   : size threshold for classes in each non empty set 
                              beneath which the indices are reselected (only if
                              targets are passed). Raises an error if it is
                              impossible. 
                              default: 2
        - strat_cl (bool)   : if True, sets are stratified by class. 
                              default: True

    Returns:
        - train_idx (list): unsorted list of indices assigned to training set.
        - val_idx (list)  : unsorted list of indices assigned to validation set.
        - test_idx (list) : unsorted list of indices assigned to test set. 
    """

    if test_n is None:
        test_n = n - train_n - val_n
   
    mixed_idx = list(range(n))

    if targets is not None or strat_cl:
        cl_vals, cl_ns = np.unique(targets, return_counts=True)
        props = [float(cl_n)/n for cl_n in cl_ns.tolist()]
        train_idx, val_idx, test_idx = [], [], []
        for val, prop in zip(cl_vals, props):
            cl_mixed_idx = np.asarray(mixed_idx)[np.where(targets == val)[0]]
            np.random.shuffle(cl_mixed_idx)
            set_ns    = [int(np.ceil(set_n * prop)) 
                         for set_n in [0, val_n, test_n]]
            set_ns[0] = len(cl_mixed_idx) - sum(set_ns)
            for s, set_n in enumerate(set_ns):
                if [train_idx, val_idx, test_idx][s] != 0 and thresh_cl != 0:
                    if set_n < thresh_cl:
                        raise ValueError('Sets cannot meet the threshold '
                                         'requirement.')
            train_idx.extend(cl_mixed_idx[0 : set_ns[0]])
            val_idx.extend(cl_mixed_idx[set_ns[0] : set_ns[0] + set_ns[1]])
            test_idx.extend(cl_mixed_idx[set_ns[0] + set_ns[1] : 
                                         set_ns[0] + set_ns[1] + set_ns[2]])
        
    else:
        cont_shuff = True
        while cont_shuff:
            np.random.shuffle(mixed_idx)
            cont_shuff = False
            # count occurrences of each class in each non empty set and ensure 
            # above threshold, otherwise reshuff
            if targets is not None or thresh_cl != 0:
                # count number of classes
                n_cl = len(np.unique(targets).tolist())
                for s in [train_idx, val_idx, test_idx]:
                    if len(s) != 0: 
                        counts = np.unique(targets[s], return_counts=True)[1]
                        count_min = np.min(counts)
                        set_n_cl = len(counts)
                        # check all classes are in the set and above threshold
                        if count_min < thresh_cl or set_n_cl < n_cl:
                            cont_shuff = True

    return train_idx, val_idx, test_idx


#############################################
def checkprop(train_p, val_p=0, test_p=0):
    """
    checkprop(train_p)

    Checks that the proportions assigned to the sets are acceptable. Throws an
    error if proportions sum to greater than 1 or if a proportion is < 0. 
    Prints a warning (no error) if the sum to less than 1.
    
    Required args:
        - train_p (num): proportion of dataset assigned to training set

    Optional args:
        - val_p (num) : proportion of dataset assigned to validation set.
                        default: 0
        - test_p (num): proportion of dataset assigned to test set.
                        default: 0
    """

    set_p = [[x, y] for x, y in zip([train_p, val_p, test_p], 
             ['train_p', 'val_p', 'test_p'])]
    
    sum_p = sum(list(zip(*set_p))[0])
    min_p = min(list(zip(*set_p))[0])

    # raise error if proportions sum to > 1 or if a proportion is < 0.
    if sum_p != 1.0 or min_p < 0.0:
        props = [f'\n{y}: {x}' for x, y in set_p]
        prop_str = '{}\nsum_p: {}'.format(''.join(props), sum_p)
        
        if min_p < 0.0:
            raise ValueError(f'Proportions must not be < 0. {prop_str}')

        elif sum_p > 1.0:
            raise ValueError(f'Proportions must not sum to > 1. {prop_str}')
    
        elif len(set_p) == 3:
        # if all values are given and sum != 1.0
            print(f'WARNING: proportions given do not sum to 1. {prop_str}')


#############################################
def split_idx(n, train_p=0.75, val_p=None, test_p=None, thresh_set=10, 
              targets=None, thresh_cl=2, strat_cl=True):
    """
    split_idx(n)

    Returns dataset indices split into training, validation and test sets. If 
    val_p and test_p are None, the non training proportion is split between 
    them. If targets are passed, the number of targets from each class in the 
    sets are checked.

    Required args:
        - n (int): length of dataset

    Optional args:
        - train_p (num)     : proportion of dataset assigned to training set
                              default: 0.75
        - val_p (num)       : proportion of dataset assigned to validation set. 
                              If None, proportion is calculated based on 
                              train_p and test_p.
                              default: None
        - test_p (num)      : proportion of dataset assigned to test set. If 
                              None, proportion is calculated based on train_p 
                              and val_p.
                              default: None
        - thresh_set (int)  : size threshold for sets beneath which an error is
                              thrown if the set's proportion is not 0.
                              default: 10
        - targets (nd array): array of targets, where the first dimension
                              is the samples. Must be of the same length 
                              as data.
                              default: None
        - thresh_cl (int)   : size threshold for classes in each non empty set 
                              beneath which the indices are reselected (only if
                              targets are passed). Not checked if thresh_cl is 
                              0.
                              default: 2
        - strat_cl (bool)   : if True, sets are stratified by class. 
                              default: True

    Returns:
        - train_idx (list): unsorted list of indices assigned to training set.
        - val_idx (list)  : unsorted list of indices assigned to validation set.
        - test_idx (list) : unsorted list of indices assigned to test set. 
    """
    
    if val_p is None and test_p is None:
        # split half half
        val_p = (1.0 - train_p)/2
        test_p = val_p
    elif val_p is None:
        val_p = 1.0 - train_p - test_p
    elif test_p is None:
        test_p = 1.0 - train_p - val_p

    checkprop(train_p, val_p, test_p)
    
    val_n = int(np.ceil(val_p*n))
    test_n = int(np.ceil(test_p*n))
    train_n = n - val_n - test_n

    # raise error if val or test n is below threshold (unless prop is 0)
    for set_n, set_p, name in zip([val_n, test_n], [val_p, test_p], 
                                  ['val n', 'test n']):
        if set_n < thresh_set:
            if set_p != 0:
                raise ValueError(f'{name} is {set_n} (below threshold '
                                 f'of {thresh_set})')

    train_idx, val_idx, test_idx = data_indices(n, train_n, val_n, test_n, 
                                                targets, thresh_cl, strat_cl)

    return train_idx, val_idx, test_idx


#############################################
def split_data(data, set_idxs, make_torch=True):
    """
    split_data(data, set_idxs)

    Returns data (or targets), split into training, validation and test sets.

    Required args:
        - data (nd array)       : array, where the first dimension is the 
                                  samples.
        - set_idxs (nested list): nested list of indices structured as:
                                  set (train, val, test) x indx

    Optional args:
        - make_torch (bool): if True, date is returned in torch Tensors instead 
                             of input format, e.g. numpy array
                             default: True

    Returns:
        - sets (list of torch Tensors): list of torch Tensors or numpy arrays 
                                        containing the data for the train, val 
                                        and test sets respectively.
                                        If a group is empty, None is used
                                        instead of an empty tensor or array.
    """

    sets = []
    for set_idx in set_idxs:
        if len(set_idx) > 0:
            if make_torch:
                sets.append(torch.Tensor(data[set_idx]))
            else:
                sets.append(data[set_idx])
        else:
            sets.append(None)
    
    return sets
    

#############################################
def init_dl(data, targets=None, batchsize=200, shuffle=False):
    """
    init_dl(data)

    Returns a torch DataLoader.

    Required args:
        - data (nd array): array of dataset datapoints, where the first
                           dimension is the samples.

    Optional args:
        - targets (nd array): array of targets, where the first dimension
                              is the samples. Must be of the same length 
                              as data.
                              default: None
        - batchsize (int )  : nbr of samples dataloader will load per batch
                              default: 200
        - shuffle (bool)    : if True, data is reshuffled at each epoch
                              default: False

    Returns:
        - dl (torch DataLoader): torch DataLoader. If data is None, dl is None. 
    """

    if data is None:
        dl = None
    else:
        dl = torch.utils.data.DataLoader(CustomDs(data, targets), 
                                         batch_size=batchsize, 
                                         shuffle=shuffle)
    return dl


#############################################
def scale_datasets(set_data, sc_dim='all', sc_type='min_max', extrem='reg', 
                   mult=1.0, shift=0.0, sc_facts=None):
    """
    scale_datasets(set_data)

    Returns scaled set_data (sets scaled based on either the factors
    passed or the factors calculated on the first set.) to between 

    Required args:
        - set_data (list): list of datasets (torch Tensors) to scale
    
    Optional args:
        - sc_dim (int)    : data array dimension along which to scale 
                            data ('last', 'all')
                            default: 'all'
        - sc_type (str)   : type of scaling to use
                            'min_max'  : (data - min)/(max - min)
                            'scale'    : (data - 0.0)/std
                            'stand'    : (data - mean)/std
                            'stand_rob': (data - median)/IQR (75-25)
                            'center'   : (data - mean)/1.0
                            'unit'     : (data - 0.0)/abs(mean)
                            default: 'min_max'
        - extrem (str)    : only needed if min_max  or stand_rob scaling is 
                            used. 
                            'reg': the minimum and maximum (min_max) or 
                                   25-75 IQR of the data are used 
                            'perc': the 5th and 95th percentiles are used as 
                                    min and max respectively (robust to 
                                    outliers)
        - mult (num)      : value by which to multiply scaled data
                            default: 1.0
        - shift (num)     : value by which to shift scaled data (applied after
                            mult)
                            default: 0.0
        - sc_facts (list) : list of sub, div, mult and shift values to use on 
                            data (overrides all other optional arguments), 
                            where sub is the value subtracted and div is the 
                            value used as divisor (before applying mult and 
                            shift)
                            default: None


    Returns:
        - set_data (list)            : list of datasets (torch Tensors) to 
                                       scale
        if sc_facts is None, also:
        - sc_facts_list (nested list): list of scaling factors structured as 
                                       stat (mean, std or perc 0.05, perc 0.95) 
                                       (x vals)
                                       default: None
    """

    set_data = gen_util.list_if_not(set_data)

    new = False
    if sc_facts is None:
        new = True
        if sc_dim == 'all':
            data_flat = set_data[0].reshape([-1]).numpy()
        elif sc_dim == 'last':
            data_flat = set_data[0].reshape([-1, set_data[0].shape[-1]]).numpy()
        else:
            gen_util.accepted_values_error('sc_dim', sc_dim, ['all', 'last'])
        sc_facts = math_util.scale_facts(data_flat, 0, sc_type=sc_type, 
                                         extrem=extrem, mult=mult, shift=shift)

    for i in range(len(set_data)):
        sc_data = math_util.scale_data(set_data[i].numpy(), 0, facts=sc_facts)
        set_data[i] = torch.Tensor(sc_data)

    if new: 
        sc_facts_list = []
        for fact in sc_facts:
            if isinstance(fact, np.ndarray):
                fact = fact.tolist()
            sc_facts_list.append(fact)
        return set_data, sc_facts_list

    return set_data


#############################################
def create_dls(data, targets=None, train_p=0.75, val_p=None, test_p=None, 
               sc_dim='none', sc_type=None, extrem='reg', mult=1.0, shift=0.0, 
               shuffle=False, batchsize=200, thresh_set=5, thresh_cl=2, 
               strat_cl=True, train_shuff=True):
    """
    create_dls(data)

    Returns torch DataLoaders for each set (training, validation, test).
    
    If a scaling dimension is passed, each set is scaled based on scaling 
    factors calculated on the training set and the scaling factors are also 
    returned.

    If shuffle is True, targets are shuffled for each dataset and the shuffled 
    indices are also returned.

    Required args:
        - data (nd array): array of dataset datapoints, where the first
                           dimension is the samples.

    Optional args:
        - targets (nd array): array of targets, where the first dimension
                              is the samples. Must be of the same length 
                              as data.
                              default: None
        - train_p (num)     : proportion of dataset assigned to training set
                              default: 0.75
        - val_p (num)       : proportion of dataset assigned to validation set. 
                              If None, proportion is calculated based on  
                              train_p and test_p.
                              default: None
        - test_p (num)      : proportion of dataset assigned to test set. If 
                              None, proportion is calculated based on train_p 
                              and val_p.
                              default: None
        - sc_dim (int)      : data array dimension along which to scale 
                              data ('last', 'all')
                              default: 'all'
        - sc_type (str)     : type of scaling to use
                              'min_max'  : (data - min)/(max - min)
                              'scale'    : (data - 0.0)/std
                              'stand'    : (data - mean)/std
                              'stand_rob': (data - median)/IQR (75-25)
                              'center'   : (data - mean)/1.0
                              'unit'     : (data - 0.0)/abs(mean)
                              default: 'min_max'
        - extrem (str)      : only needed if min_max  or stand_rob scaling is 
                              used. 
                              'reg': the minimum and maximum (min_max) or 
                                     25-75 IQR of the data are used 
                              'perc': the 5th and 95th percentiles are used as 
                                      min and max respectively (robust to 
                                      outliers)
        - mult (num)        : value by which to multiply scaled data
                              default: 1.0
        - shift (num)       : value by which to shift scaled data (applied 
                              after mult)
                              default: 0.0
        - shuffle (bool)    : if True, targets are shuffled in all sets to 
                              create randomized datasets.
                              default: False
        - batchsize (int)   : nbr of samples dataloader will load per batch
                              default: 200
        - thresh_set (int)  : size threshold for sets beneath which an error is
                              thrown if the set's proportion is not 0.
                              default: 5
        - thresh_cl (int)   : size threshold for classes in each non empty set 
                              beneath which the indices are reselected (only if
                              targets are passed). Not checked if thresh_cl is 
                              0.
                              default: 2
        - strat_cl (bool)   : if True, sets are stratified by class. 
                              default: True
        - train_shuff (bool): if True, training data is set to be reshuffled at 
                              each epoch
                              default: True

    Returns:
        - returns (list): 
            - dls (list of torch DataLoaders): list of torch DataLoaders for 
                                               each set. If a set is empty, the 
                                               corresponding dls value is None.
            Optional:
            if shuffle:
            - shuff_reidx (list): list of indices with which targets were
                                  shuffled
            if sc_dim is not None:
            - sc_facts (nested list): list of scaling factors structured as 
                                        stat (mean, std or perc 0.05, perc 0.95) 
                                        (x vals)
    """

    returns = []

    # shuffle targets first
    if targets is not None:
        if shuffle:
            shuff_reidx = list(range(len(targets)))
            np.random.shuffle(shuff_reidx)
            returns.append(shuff_reidx)
            targets = targets[shuff_reidx]
    else:
        set_targets = [None] * 3

    # data: samples x []
    set_idxs = split_idx(n=len(data), train_p=train_p, val_p=val_p, 
                         test_p=test_p, thresh_set=thresh_set, targets=targets, 
                         thresh_cl=thresh_cl, strat_cl=strat_cl)

    set_data = split_data(data, set_idxs)
    if targets is not None:
        set_targets = split_data(targets, set_idxs)

    if sc_dim not in ['None', 'none']:
        set_data, sc_facts = scale_datasets(set_data, sc_dim, sc_type, extrem, 
                                            mult, shift)
        returns.append(sc_facts)
    
    dls = []
    # if training set, shuffle targets
    for i, (data, targ) in enumerate(zip(set_data, set_targets)):
        if train_shuff and i == 0:
            shuff = True
        else:
            shuff = False
        dls.append(init_dl(data, targ, batchsize, shuff))

    returns = [dls] + returns

    return returns


#############################################
def get_n_wins(leng, win_leng, step_size=1):
    """
    get_n_wins(leng, win_leng)

    Returns the number of windows is the data dimension, based on the 
    specified window length and step_size.

    Required args:
        - leng (int)    : length of the data along the dimension of interest
        - win_leng (int): length of the windows to use
    
    Optional args:
        - step_size (int): step size between each window


    Returns:
        - n_wins (int): number of windows along the dimension of interest
    """

    if leng < win_leng:
        n_wins = 0
    else:
        n_wins = int((leng - win_leng) // step_size) + 1

    return n_wins


##########################################
def get_win_xrans(xran, win_leng, idx, step_size=1):
    """
    get_win_xrans(xran, win_leng, idx)

    Returns x ranges for the specified windows, based on the full x range, 
    window length and specified indices.

    Required args:
        - xran (array-like): Full range of x values
        - win_leng (int)   : length of the windows used
        - idx (list)       : list of indices for which to return x ranges

    Optional args:
        - step_size (int): step size between each window

    Returns:
        - xrans (list): nested list of x values, structured as index x x_vals    
    """

    idx = gen_util.list_if_not(idx)
    n_wins = get_n_wins(len(xran), win_leng, step_size=1)
    xrans = []
    for i in idx:
        win_i = i%n_wins
        xrans.append(xran[win_i : win_i + win_leng])

    return xrans


#############################################
def window_1d(data, win_leng, step_size=1, writeable=False):
    """
    window_1d(data, win_leng)

    Returns original data array with updated stride view to be interpreted
    as a 2D array with the original data split into windows.

    Note: Uses 'numpy.lib.stride_tricks.as_strided' function to allow
    windowing without copying the data. May lead to unexpected behaviours
    when using functions on the new array. 
    See: https://docs.scipy.org/doc/numpy/reference/generated/
    numpy.lib.stride_tricks.as_strided.html 

    Required args:
        - data (1D array): array of samples
        - win_leng (int) : length of the windows to extract

    Optional args:
        - step_size (int) : number of samples between window starts 
                            default: 1
        - writeable (bool): if False, the array is unwriteable, to avoid bugs
                            that may occur with strided arrays
                            default: False

    Returns:
        - strided_data (2D array): original data array, with updated stride
                                   view, structured as:
                                       n_win x win_leng
    """

    # bytes to step in each dimension when traversing array
    strides = data.strides[0] 
    # resulting number of windows
    n_wins = get_n_wins(data.shape[0], win_leng, step_size)

    strided_data = np.lib.stride_tricks.as_strided(data, 
                                        shape=[n_wins, int(win_leng)], 
                                        strides=[strides * step_size, strides], 
                                        writeable=writeable)

    return strided_data


#############################################
def window_2d(data, win_leng, step_size=1, writeable=False):
    """
    window_2d(data, win_leng)

    Returns original data array with updated stride view to be interpreted
    as a 3D array with the original data split into windows along the first
    dimension.

    Note: Uses 'numpy.lib.stride_tricks.as_strided' function to allow
    windowing without copying the data. May lead to unexpected behaviours
    when using functions on the new array. 
    See: https://docs.scipy.org/doc/numpy/reference/generated/
    numpy.lib.stride_tricks.as_strided.html 

    Required args:
        - data (2D array): n_samples x n_items 
                           (windows extracted along sample dimension)
        - win_leng (int) : length of the windows to extract

    Optional args:
        - step_size (int) : number of samples between window starts 
                            default: 1
        - writeable (bool): if False, the array is unwriteable, to avoid bugs
                            that may occur with strided arrays
                            default: False

    Returns:
        - strided_data (3D array): original data array, with updated stride
                                   view, structured as:
                                       n_wins x n_items x win_leng
    """
    
    # bytes to step in each dimension when traversing array
    strides = data.strides
    n_wins  = get_n_wins(data.shape[0], win_leng, step_size)
    n_items = data.shape[1]

    strided_data = np.lib.stride_tricks.as_strided(data, 
                                   shape=[n_wins, n_items, int(win_leng)],
                                   strides=[strides[0] * step_size, strides[1],
                                   strides[0]], writeable=writeable)
    return strided_data

