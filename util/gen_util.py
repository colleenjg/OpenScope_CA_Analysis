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
import multiprocessing
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

    val_str = ', '.join([f'`{x}`' for x in accept_vals])
    error_message = (f'`{varname}` value `{wrong_val}` unsupported. Must be in '
                     f'{val_str}.')
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
    dirname = (f'{now.year:02d}{now.month:02d}{now.day:02d}_'
               f'{now.hour:02d}{now.minute:02d}{now.second:02d}')
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
def delist_if_not(items):
    """
    delist_if_not(items)

    If a list contains only one element, returns the element. Otherwise,
    returns the original list.

    Required args:
        - items (list): list

    Returns:
        - items (item or list): only item in the list or original list.
    """
    
    if len(items) == 1:
        items = items[0]
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
def slice_idx(axis, pos):
    """
    slice_idx(axis, pos)

    Returns a tuple to index an array based on an axis and position on that
    axis.

    Required args:
        - axis (int): axis number (non negative)
        - post (int): position on axis

    Returns:
        - sl_idx (slice): slice corresponding to axis and position passed.
    """

    if axis is None and pos is None:
        sl_idx = tuple([slice(None)])

    elif axis < 0:
        raise ValueError('Do not pass -1 axis value as this will always '
                         'be equivalent to axis 0.')

    else:
        sl_idx = tuple([slice(None)] * axis + [pos])

    return sl_idx


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
    keep_slice = slice_idx(axis, keep)

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
def intlist_to_str(intlist):
    """
    intlist_to_str(intlist)

    Returns a string corresponding to the list of values, e.g. 1-4 or 1-3-6.

    Required args:
        - intlist (list): list of int values

    Returns:
        - intstr (str): corresponding string. If range, end is included
    """

    if isinstance(intlist, list):
        extr = [min(intlist), max(intlist) + 1]
        if set(intlist) == set(range(*extr)):
            intstr = f'{extr[0]}-{extr[1]-1}'
        else:
            intstr = '-'.join([str(i) for i in sorted(intlist)])
    else:
        raise ValueError('`intlist` must be a string.')

    return intstr


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
            print(f'Random seed: {seed}')
    else:
        if print_seed:
            print(f'Preset seed: {seed}')
    
    if seed_now:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)
    
    return seed


#############################################
def conv_type(item, dtype=int):
    """
    conv_type(item)

    Returns input item converted to a specific type (int, float or str). 

    Required args:
        - item (item): value to convert

    Optional args:
        - dtype (dtype): target datatype (int, float or str)
                         default: int

    Returns:
        - item (item): converted value
    """

    if dtype in [int, 'int']:
        item = int(item)
    elif dtype in [float, 'float']:
        item = float(item)
    elif dtype in [str, 'str']:
        item = str(item)
    else:
        accepted_values_error('dtype', dtype, ['int', 'float', 'str'])

    return item


#############################################
def conv_types(items, dtype=int):
    """
    conv_types(items)

    Returns input list with items converted to a specific type (int, float or 
    str). 

    Required args:
        - items (list): values to convert

    Optional args:
        - dtype (dtype): target datatype (int, float or str)
                         default: int

    Returns:
        - items (list): converted values
    """

    items = list_if_not(items)

    for i in range(len(items)):
        items[i] = conv_type(items[i], dtype)

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
def get_df_vals(df, cols=[], criteria=[], label=None, unique=True, dtype=None, 
                single=False):
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
        - criteria (list): ordered list of single criteria for each column
                           default: []
        - label (str)    : column for which to return values
                           if None, the dataframe lines are returned instead
                           default: None
        - unique (bool)  : if True, only unique values are returned for the 
                           column of interest
                           default: True
        - dtype (dtype)  : if not None, values are converted to the specified 
                           datatype (int, float or str)
                           dtype: None
        - single (bool)  : if True, checks whether only one value or row is 
                           found and if so, returns it
                           dtype: False 

    Returns:
        if label is None:
            - lines (pd Dataframe): dataframe containing lines corresponding to 
                                    the specified criteria.
        else:
            if single:
            - vals (item)         : value from a specific column corresponding 
                                    to the specified criteria. 
            else:
            - vals (list)         : list of values from a specific column 
                                    corresponding to the specified criteria. 
    """

    if not isinstance(cols, list):
        cols = [cols]
        criteria = [criteria]

    if len(cols) != len(criteria):
        raise ValueError('Must pass the same number of columns and criteria.')

    for att, cri in zip(cols, criteria):
        df = df.loc[(df[att] == cri)]
        
    if label is not None:
        vals = df[label].tolist()
        if unique:
            vals = sorted(list(set(vals)))
        if dtype is not None:
            vals = conv_types(vals, dtype)
        if single:
            if len(vals) != 1:
                raise ValueError('Expected to find 1 value, but '
                                 f'found {len(vals)}.')
            else:
                vals = vals[0]
        return vals
    else: 
        if single and len(df) != 1:
            raise ValueError('Expected to find 1 dataframe line, but '
                             f'found {len(df)}.')
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
def drop_unique(df):
    """
    drop_unique(df)

    Returns dataframe with columns containing only a unique value dropped.

    Required args:
        - df (pd DataFrame): dataframe

    Returns:
        - df (pd DataFrame): dataframe with columns containing only a unique 
                             value dropped
    """

    for col in df.columns:
        uniq_vals = df[col].unique().tolist()
        if len(uniq_vals) == 1:
            df = df.drop(columns=col)

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
               fulldir='', level='info'):
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
                          default: ''
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


#############################################
def hierarch_argsort(data, sorter='fwd', axis=0, dtypes=None):
    """
    hierarch_argsort(data)

    Returns the sorting argument and sorted data. Data is sorted hierarchically
    based on the sorter (top -> bottom hierarchy) along the specified axis.

    Required args:
        - data (nd array): array of data to use for sorting

    Optional args:
        - sorter (str or list): order to use for the sorting hierarchy, from
                                top to bottom (list of indices or 'fwd' or 
                                'rev')
                                default: 'fwd'
        - axis (int)          : axis number
                                default: 0
        - dtypes (list)       : datatypes to which to convert each data sorting
                                sub array (one per sorting position)
                                default: None
    
    Returns:
        - overall_sort (list): sorting index
        - data (nd array)    : sorted data array
    """

    if len(data.shape) != 2:
        raise ValueError('Only implemented for 2D arrays.')

    axis, rem_axis = pos_idx([axis, 1-axis], len(data.shape))
    axis_len = data.shape[axis]

    data = copy.deepcopy(data)

    if sorter in ['fwd', 'rev']:
        sorter = range(axis_len)
        if sorter == 'rev':
            sorter = reversed(sorter)
    else:
        sorter = list_if_not(sorter)
        sorter = pos_idx(sorter, data.shape[axis])

    if dtypes is None:
        dtypes = [None] * len(sorter)
    elif len(dtypes) != len(sorter):
        raise ValueError('If `dtypes` are provided, must pass one per '
                         'sorting position.')

    overall_sort = np.asarray(range(data.shape[rem_axis]))

    for i, dt in zip(reversed(sorter), dtypes):
        sc_idx = slice_idx(axis, i)
        sort_data = data[sc_idx]
        if dt is not None:
            sort_data = sort_data.astype(dt)
        sort_arr = np.argsort(sort_data)
        overall_sort = overall_sort[sort_arr]
        sort_slice   = slice_idx(rem_axis, sort_arr)
        data         = data[sort_slice]

    return overall_sort, data


#############################################
def compile_dict_list(dict_list):
    """
    compile_dict_list(dict_list)

    Returns a dictionary of lists created from a list of dictionaries with 
    shared keys.

    Required args:
        - dict_list (list): list of dictionaries with shared keys

    Returns:
        - full_dict (dict): dictionary with lists for each key
    """

    full_dict = dict()

    all_keys = []
    for sing_dict in dict_list:
        all_keys.extend(sing_dict.keys())
    all_keys = list(set(all_keys))

    for key in all_keys:
        vals = [sub_dict[key] for sub_dict in dict_list 
                              if key in sub_dict.keys()]
        full_dict[key] = vals

    return full_dict


#############################################
def num_to_str(num, n_dec=2, dec_sep='-'):
    """
    num_to_str(num)

    Returns number converted to a string with the specified number of decimals 
    and decimal separator

    Required args:
        - num (num): number
    
    Optional args:
        - n_dec (int)  : number of decimals to retain
                         default: 2
        - dec_sep (str): string to use as a separator
                         default: '-'
    
    Returns:
        - num_str (str): number as a string
    """

    num_str = str(int(num))

    num_res = np.round(num % 1, n_dec)
    if num_res != 0:
        num_str = f'{num_str}{dec_sep}{str(num_res)[2:]}'

    return num_str


#############################################
def get_n_jobs(n_tasks, parallel=True, max_cores='all'):
    """
    get_n_jobs(n_tasks)

    Returns number of jobs corresponding to the criteria passed.

    Required args:
        - n_tasks (int): number of tasks to run
    
    Optional args:
        - parallel (bool)       : if False, n_jobs of None is returned
                                  default: True
        - max_cores (str or num): max number or proportion of cores to use 
                                  ('all', proportion or int)
                                  default: 'all'

    Returns:
        - n_jobs (int): number of jobs to use (None if not parallel or fewer 
                        than 2 jobs calculated)
    """

    if not parallel:
        n_jobs = None

    else:
        n_cores = multiprocessing.cpu_count()
        if max_cores != 'all':
            max_cores = float(max_cores)
            if max_cores >= 0.0 and max_cores <= 1.0:
                n_cores = int(n_cores * max_cores)
            else:
                n_cores = np.min(n_cores, max_cores)
        n_cores = int(n_cores)
        n_jobs = min(int(n_tasks), n_cores)
        if n_jobs < 2:
            n_jobs = None

    return n_jobs

