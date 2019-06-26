"""
gen_util.py

This module contains general purpose functions.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import copy
import datetime
import logging
import os
import random
import re
import sys

import numpy as np
import pandas as pd
import torch


#############################################
def accepted_values_error(varname, wrong_val, accept_vals):
    """
    accepted_values_error(varname, wrong_value, accept_values)

    Raises a value error with a message indicating the variable name,
    accepted values for that variable and wrong value stored in the variable.

    Required args:
        - varname (str)     : name of the variable
        - wrong_val (item)  : value stored in the variable
        - accept_vals (list): list of accepted values for the variable
    """

    val_str = ', '.join(['`{}`'.format(x) for x in accept_vals])
    error_message = ('`{}` value `{}` unsupported. Must be in '
                     '{}.').format(varname, wrong_val, val_str)
    raise ValueError(error_message)


#############################################
def create_time_str():
    """
    create_time_str()

    Returns a string in a format appropriate for a directory or filename
    containing date and time information based on time at which the function is
    called.

    Return:
        dirname (str): string containing date and time formatted as 
                       YYMMDD_HHMMSS
    """

    now = datetime.datetime.now()
    dirname = ('{:02d}{:02d}{:02d}_'
               '{:02d}{:02d}{:02d}').format(now.year, now.month, now.day, 
                                            now.hour, now.minute, now.second)
    return dirname
    
    
#############################################
def remove_if(vals, rem):
    """
    remove_if(vals, rem)

    Returns input with items removed from it, if they were are the input.

    Required args:
        - vals (item or list): item or list from which to remove elements
        - rem (item or list) : item or list of items to remove from vals

    Returns:
        - vals (list): list with items removed.
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
def list_if_not(items):
    """
    list_if_not(items)

    Returns input in a list, if it is not a list.

    Required args:
        - items (obj or list): item or list

    Returns:
        - items (list): list version of input.
    """
    
    if not isinstance(items, list):
        items = [items]
    return items


#############################################
def remove_lett(lett_str, rem):
    """
    remove_lett(lett_str, rem)

    Returns input string with letters remove, as well as a list of the letters
    that were actually present, and removed.

    Required args:
        - lett_str (str): string of letters
        - rem (str)     : string of letters to remove

    Returns:
        - lett_str (str): string of letters where the letters to remove have 
                          been removed
        - removed (str) : string of letters that were actually present and 
                          removed
    """

    if not isinstance(lett_str, str):
        raise ValueError('lett_str must be a string.')
    
    if not isinstance(rem, str):
        raise ValueError('rem must be a string.')

    removed = ''
    for lett in rem:
        if lett in lett_str:
            lett_str = lett_str.replace(lett, '')
            removed += lett

    return lett_str, removed


#############################################
def remove_idx(items, rem, axis=0):
    """
    remove_idx(items, rem)

    Returns input with items at specific indices in a specified axis removed.

    Required args:
        - items (item or array-like): array or list from which to remove 
                                      elements
        - rem (item or array-like)  : list of idx to remove from items

    Optional args:
        - axis (int): axis along which to remove indices if items is an array
                      default: 0

    Returns:
        - items (array-like): list or array with specified items removed.
    """

    rem = list_if_not(rem)

    if isinstance(items, list):
        make_list = True
        items     = np.asarray(items)

    else:
        make_list = False

    all_idx = items.shape[axis]
    keep = sorted(set(range(all_idx)) - set(rem))
    keep_slice = tuple([slice(None)] * axis + [keep])

    items = items[keep_slice]

    if make_list:
        items = items.tolist()
    
    return items


#############################################
def pos_idx(idx, leng):
    """
    pos_idx(idx, leng)

    Returns a list of indices with any negative indices replaced with
    positive indices (e.g. -1 -> 4 for an axis of length 5).

    Required args:
        - idx (int or list): index or list of indices
        - leng (int)       : length of the axis

    Returns:
        - idx (int or list): modified index or list of indices (all positive)
    """

    if isinstance(idx, int):
        if idx < 0:
            idx = leng + idx
    
    else:
        for i in range(len(idx)):
            if idx[i] < 0:
                idx[i] = leng + idx[i]
        
    return idx


#############################################
def consec(idx, smallest=False):
    """
    consec(idx)

    Returns the first of each consecutive series in the input, as well as the
    corresponding number of consecutive values.
    
    Required args:
        - idx (list)  : list of values, e.g. indices
    
    Optional args:
        - smallest (bool): if True, the smallest interval present is considered 
                           consecutive
                           default: False
    
    Returns:
        - firsts (list)  : list of values with consecutive values removed
        - n_consec (list): list of number of consecutive values corresponding
                             to (and including) the values in firsts
    """


    if len(idx) == 0:
        return [], []

    interv = 1
    if smallest:
        interv = min(np.diff(idx))

    consec_bool = np.diff(idx) == interv
    firsts = []
    n_consec = []

    firsts.append(idx[0])
    count = 1
    for i in range(len(consec_bool)):
        if consec_bool[i] == 0:
            n_consec.append(count)
            firsts.append(idx[i+1])
            count = 1
        else:
            count += 1
    n_consec.append(count)
    
    return firsts, n_consec
    

#############################################
def deepcopy_items(item_list):
    """
    deepcopy_items(item_list)

    Returns a deep copy of each item in the input.

    Required args:
        - item_list (list): list of items to deep copy

    Returns:
        - new_item_list (list): list of deep copies of items
    """
    
    item_list = list_if_not(item_list)

    new_item_list = []
    for item in item_list:
        new_item_list.append(copy.deepcopy(item))

    return new_item_list


#############################################
def str_to_list(item_str, only_int=False):
    """
    str_to_list(item_str)

    Returns a list of items taken from the input string, in which different 
    items are separated by spaces. 

    Required args:
        - item_str (str): items separated by spaces

    Optional args:
        - only_int (bool): if True, items are converted to ints
                           default: False

    Returns:
        - item_list (list): list of values.
    """

    if len(item_str) == 0:
        item_list = []
    else:
        item_list = item_str.split()
        if only_int:
            item_list = [int(re.findall('\d+', it)[0]) for it in item_list]
        
    return item_list


#############################################
def seed_all(seed=None, device='cpu', print_seed=True, seed_now=True):
    """
    seed_all()

    Seeds different random number generators using the seed provided or a
    randomly generated seed if no seed is given.

    Required args:
        

    Optional args:
        - seed (int or None): seed value to use. (-1 treated as None)
                              default: None
        - device (str)      : if 'cuda', torch.cuda, else if 'cpu', cuda is not
                              seeded
                              default: 'cpu'
        - print_seed (bool) : if True, seed value is printed to the console
                              default: True
        - seed_now (bool)   : if True, random number generators are seeded now

    Returns:
        - seed (int): seed value
    """

    if seed in [None, -1]:
        seed = random.randint(1, 10000)
        if print_seed:
            print('Random seed: {}'.format(seed))
    else:
        if print_seed:
            print('Preset seed: {}'.format(seed))
    
    if seed_now:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)
    
    return seed


#############################################
def conv_type(items, dtype=int):
    """
    conv_type(items)

    Returns input list with items converted to a specific type (int, float or 
    str). 

    Required args:
        - items (list): values to convert

    Optional args:
        - dtype (dtype): target datatype (int, float or str)
                         default: int

    Returns:
        - vals (list): converted values
    """

    items = list_if_not(items)

    for i in range(len(items)):
        if dtype in [int, 'int']:
            items[i] = int(items[i])
        elif dtype in [float, 'float']:
            items[i] = float(items[i])
        elif dtype in [str, 'str']:
            items[i] = str(items[i])
        else:
            accepted_values_error('dtype', dtype, ['int', 'float', 'str'])

    return items


#############################################
def get_df_label_vals(df, label, vals=None):
    """
    get_df_label_vals(df, label)

    Returns values for a specific label in a dataframe. If the vals is 'any', 
    'all' or None, returns all different values for that label.
    Otherwise, vals are returned as a list.

    Required args:
        - df (pandas df): dataframe
        - label (str)   : label of the dataframe column of interest

    Optional args:
        - val (str or list): values to return. If val is None, 'any' or 'all', 
                             all values are returned.
                             default=None
    Return:
        - vals (list): values
    """

    if vals in [None, 'any', 'all']:
        vals = df[label].unique().tolist()
    else:
        vals = list_if_not(vals)
    return vals


#############################################
def get_df_vals(df, cols=[], criteria=[], label=None, unique=True, dtype=None):
    """
    get_df_vals(df, cols, criteria)

    Returns dataframe lines or values that correspond to the specified 
    criteria. 

    Required args:
        - df (pandas df): dataframe

    Optional args:
        - cols (list)    : ordered list of columns for which criteria are 
                           provided
                           default: []
        - criteria (list): ordered list of criteria for each column
                           default: []
        - label (str)    : column for which to return values
                           if None, the dataframe lines are returned instead
                           default: None
        - unique (bool)  : if True, only unique values are returned for the 
                           column of interest
                           default: True
        - dtype (str)    : if not None, values are converted to the specified 
                           datatype (int, float or str)
                           dtype: None

    Returns:
        if label is None:
            - lines (pd Dataframe): dataframe containing lines corresponding to 
                                    the specified criteria.
        else:
            - vals (list)         : list of values from a specific column 
                                    corresponding to the specified criteria. 
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
def set_df_vals(df, idx, cols, vals):
    """
    set_df_vals(df, attributes, criteria)

    Returns dataframe with certain values changed. These are specified by one
    index and a list of columns and corresponding new values.

    Required args:
        - df (pandas df): dataframe
        - idx (int)     : dataframe line index (for use with .loc)
        - cols (list)   : ordered list of columns for which vals are 
                          provided
        - vals (list)   : ordered list of values for each column

    Returns:
        - df (pd Dataframe): dataframe containing modified lines. 
    """

    cols = list_if_not(cols)
    vals = list_if_not(vals)

    if len(cols) != len(vals):
        raise ValueError('Must pass the same number of columns and values.')

    for col, val in zip(cols, vals):
        df.loc[idx, col] = val
    
    return df


#############################################
def num_ranges(ns, pre=0, leng=10):
    """
    num_ranges(ns)

    Returns all indices within the specified range of the provided reference 
    indices. 

    Required args:
        - ns (list): list of reference numbers

    Optional args:
        - pre (num) : indices to include before reference to include
                      default: 0
        - leng (num): length of range
                      default: 10
    Returns:
        - num_ran (2D array): array of indices where each row is the range
                              around one of the input numbers (ns x ranges)
    """

    post = float(leng) - pre

    pre, post = [int(np.around(p)) for p in [pre, post]]

    num_ran = np.asarray([list(range(n-pre, n+post)) for n in ns])

    return num_ran


#############################################
def get_device(cuda=False, device=None):
    """
    get_device()

    Returns name of device to use based on cuda availability and whether cuda  
    is requested, either via the 'cuda' or 'device' variable, with 'device' 
    taking precedence.

    Optional args:
        - cuda (bool) : if True, cuda is used (if available), but will be 
                        overridden by device.
                        default: False
        - device (str): indicates device to use, either 'cpu' or 'cuda', and 
                        will override cuda variable if not None
                        default: None 
        
    Returns:
        - device (str): device to use
    """

    if device is None:
        if cuda:
            device = 'cuda'
        else:
            device = 'cpu'
    if not(device == 'cuda' and torch.cuda.is_available()):
        device = 'cpu'

    return device


#############################################
def get_logger(logtype='both', name='all logs', filename='logs.txt', 
               fulldir='.', level='info'):
    """
    get_logger()

    Returns logger and handler(s).

    Optional args:
        - logtype (str) : type or types of handlers to add to logger 
                          ('stream', 'file', 'both', 'none')
                          default: 'both'
        - name (str)    : logger name
                          default: 'all logs'
        - filename (str): name under which to save file handler, if it is 
                          included
                          default: 'logs.txt'
        - fulldir (str) : path under which to save file handler, if it is
                          included
                          default: '.'
        - level (str)   : level of the handler ('info', 'error', 'warning', 
                          'debug')
                          default: 'info'
        
    Returns:
        - logger (Logger): logger object
    """


    # create one instance
    logger = logging.getLogger(name)
    logger.handlers = []
    
    # create handlers
    sh, fh = None, None
    if logtype in ['stream', 'both']:
        sh = logging.StreamHandler(sys.stdout)
        logger.addHandler(sh)
    if logtype in ['file', 'both']:
        fh = logging.FileHandler(os.path.join(fulldir, filename))
        logger.addHandler(fh)
    all_types = ['file', 'stream', 'both', 'none']
    if logtype not in all_types:
        accepted_values_error('logtype', logtype, all_types)
    
    if level.lower() == 'info':
        level = logging.INFO
    elif level.lower() == 'error':
        level = logging.ERROR
    elif level.lower() == 'warning':
        level = logging.WARNING
    elif level.lower() == 'debug':
        level = logging.DEBUG
    else:
        accepted_values_error('level', level, 
                              ['info', 'error', 'warning', 'debug'])
    logger.setLevel(level)

    return logger

