'''
gen_util.py

This module contains general functions.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 2.7.

'''

import random

import numpy as np
import torch
import pandas as pd

#############################################
def accepted_values_error(var_name, wrong_value, accept_values):
    """
    accepted_values_error(var_name, wrong_value, accept_values)

    Raises a value error with a message indicating the variable name,
    accepted values for that variable and wrong value stored in the variable.

    Required arguments:
        - var_name (str)      : name of the variable
        - wrong_value (item)  : value stored in the variable
        - accept_values (list): list of accepted values for the variable
    """

    values_str = ', '.join(['\'{}\''.format(x) for x in accept_values])
    error_message = ('\'{}\' value \'{}\' unsupported. Must be in '
                     '{}.').format(var_name, wrong_value, values_str)
    raise ValueError(error_message)


#############################################
def remove_if(vals, rem):
    """
    remove_if(vals, rem)

    Removes items from a list if they are in the list.

    Required arguments:
        - vals (item or list): item or list from which to remove elements
        - rem (item or list) : item or list of items to remove from vals

    Return:
        vals (list): list with items removed.
    """

    if not isinstance(rem, list):
        rem = [rem]
    if not isinstance(vals, list):
        vals = [vals]
    for i in rem:
        if i in vals:
            vals.remove(i)
    return vals


#############################################
def remove_idx(vals, rem, axis=0):
    """
    remove_idx(vals, rem)

    Removes items with specific axis from a list or array.

    Required arguments:
        - vals (item or list): array or list from which to remove elements
        - rem (item or list) : list of idx to remove from vals

    Optional arguments:
        - axis (int): axis along which to remove indices if vals is an array

    Return:
        vals (list): list or array with idx removed.
    """

    if not isinstance(rem, list):
        rem = [rem]

    if isinstance(vals, list):
        make_list = True
        vals = np.asarray(vals)

    else:
        make_list = False

    all_idx = vals.shape[axis]
    keep = sorted(set(range(all_idx)) - set(rem))
    keep_slice = tuple([slice(None)] * axis + [keep])

    vals = vals[keep_slice]

    if make_list:
        vals = vals.tolist()
    
    return vals


#############################################
def list_if_not(vals):
    """
    list_if_not(vals)

    Converts input into a list if not a list.

    Required arguments:
        - vals (item or list): item or list

    Return:
        vals (list): list version of input.
    """
    
    if not isinstance(vals, list):
        vals = [vals]
    return vals


#############################################
def seed_all(seed, device='cpu', print_seed=True):
    """
    seed_all(seed)

    Seeds different random number generators using the provided seed or a
    randomly generated seed if no seed is given.

    Required arguments:
        - seed (int or None): seed value to use

    Optional arguments:
        - device (str):      if 'cuda', torch.cuda, else if 'cpu', cuda is not
                             seeded
                             default: 'cpu'
        - print_seed (bool): if True, seed value is printed to the console
                             default: True

    Return:
        seed (int): seed value
    """

    if seed in [None, 'None']:
        seed = random.randint(1, 10000)
        if print_seed:
            print('Random seed: {}'.format(seed))
    else:
        if print_seed:
            print('Preset seed: {}'.format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    return seed


#############################################
def conv_type(vals, dtype=int):
    """
    conv_type(vals)

    Converts values in a list to a specific type (int, float or str) 

    Required arguments:
        - vals (list): values to convert

    Optional arguments:
        - dtype (dtype): target datatype (int, float or str)

    Return:
        vals (list): converted values
    """

    vals = list_if_not(vals)

    for i in range(len(vals)):
        if dtype in [int, 'int']:
            vals[i] = int(vals[i])
        elif dtype in [float, 'float']:
            vals[i] = float(vals[i])
        elif dtype in [str, 'str']:
            vals[i] = str(vals[i])
        else:
            accepted_values_error('dtype', dtype, ['int', 'float', 'str'])

    return vals


#############################################
def get_df_vals(df, cols=[], criteria=[], label=None, unique=True, dtype=None):
    """
    get_df_vals(df, cols, criteria)

    Selects lines or values in a dataframe that correspond to specific criteria. 

    Required arguments:
        - df (pd Dataframe): dataframe

    Optional arguments:
        - cols (list)    : ordered list of columns for which criteria are provided
        - criteria (list): ordered list of criteria for each column

        - label (str)    : column for which to return values
                           if None, the dataframe lines are returned instead 
        - unique (bool)  : if True, only unique values are returned for the column
                           of interest
        - dtype (str)    : if not None, values are converted to the specified 
                           datatype (int, float or str)

    Return:
        lines or vals (pd Dataframe or list): dataframe containing lines that 
                                              corresponded to criteria or list 
                                              of values from a specific column
                                              from those lines. 
    """

    cols = list_if_not(cols)
    criteria = list_if_not(criteria)

    if len(cols) != len(criteria):
        raise ValueError('Must pass the same number of columns and criteria.')

    for att, cri in zip(cols, criteria):
        df = df.loc[(df[att] == cri)]
        
    if label is not None:
        vals = df[label].tolist()
        if unique:
            vals = sorted(list(set(vals)))
        if dtype is not None:
            vals = conv_type(vals, dtype)
        return vals
    else: 
        return df


#############################################
def set_df_vals(df, idx, cols=[], vals=[]):
    """
    set_df_vals(df, attributes, criteria)

    Sets columns in a dataframe line to specific values . 

    Required arguments:
        - df (pd Dataframe): dataframe
        - idx (int)        : dataframe line index (for use with .loc)

    Optional arguments:
        - cols (list): ordered list of columns for which vals are provided
        - vals (list): ordered list of values for each column

    Return:
        df (pd Dataframe): dataframe containing modified lines. 
    """

    cols = list_if_not(cols)
    vals = list_if_not(vals)

    if len(cols) != len(vals):
        raise ValueError('Must pass the same number of columns and values.')

    for col, val in zip(cols, vals):
        df.loc[idx, col] = val
    
    return df


#############################################
def idx_segs(idx, pre=0, leng=10):
    """
    idx_segs(idx)

    Calculates indices for segments surrounding given reference indices. 

    Required arguments:
        - idx (list): list of reference indices

    Optional arguments:
        - pre (float): indices to include before reference to include
        - len (float): length of segment
        
    Return:
        idx_segs (2D array): array of indices per segment (index x seg)

    """

    post = float(leng) - pre

    pre, post = [int(np.around(p)) for p in [pre, post]]

    idx_segs = np.asarray([range(x-pre, x+post) for x in idx])

    return idx_segs


#############################################
def get_device(cuda=False, device=None):
    """
    get_device(idx)

    Returns device to use based cuda availability and whether cuda is requested, 
    either via the 'cuda' or 'device' variable, with 'device' taking precedence.

    Optional arguments:
        - cuda (bool) : if True, cuda is used (if available)
                        default: False
        - device (str): indicates device to use, either 'cpu' or 'cuda', and 
                        will override cuda variable if not None
                        default: None 
        
    Return:
        device (str): device to use

    """

    if device is None:
        if cuda:
            device = 'cuda'
        else:
            device = 'cpu'
    if not(device == 'cuda' and torch.cuda.is_available()):
        device = 'cpu'

    return device

