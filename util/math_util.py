'''
math_util.py

This module contains basic math functions.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 2.7.

'''
import numpy as np
import scipy.stats as st
import torch

import gen_util

#############################################
def mean_med(data, stats='mean', axis=None, nanpol=None):
    """
    mean_med(data)

    Returns the mean or median of the data along a specified axis, depending on
    which statistic is needed.

    Required arguments:
        - data (np.array): data on which to calculate statistic

    Optional arguments:
        - stats (str) : 'mean' or 'median'
                        default: 'mean'
        - axis (int)  : axis along which to take the statistic
                        default: None
        - nanpol (str): policy for NaNs, 'omit' or None
                        default: None
    
    Returns:
        - (nd array): mean or median of data along specified axis
    """

    if stats == 'mean':
        if nanpol is None:
            return np.mean(data, axis=axis)
        elif nanpol == 'omit':
            return np.nanmean(data, axis=axis)
    elif stats == 'median':
        if nanpol is None:
            return np.median(data, axis=axis)
        elif nanpol == 'omit':
            return np.nanmedian(data, axis=axis)
    else:
        gen_util.accepted_values_error('stats', stats, ['mean', 'median'])
    if nanpol is not None and nanpol != 'omit':
        gen_util.accepted_values_error('nanpol', nanpol, ['None', 'omit'])


#############################################
def error_stat(data, stats='mean', error='std', axis=None, nanpol=None, qu=[25, 75]):
    """
    error_stat(data)

    Returns the std, SEM, quartiles or median absolute deviation (MAD) of data 
    along a specified axis, depending on which statistic is needed.

    Required arguments:
        - data (np.array): data on which to calculate statistic

    Optional arguments:
        - stats (str) : 'mean' or 'median'
                        default: 'mean'
        - error (str) : 'std' (for std or quintiles) or 'sem' (for SEM or MAD)
                        default: 'std'
        - axis (int)  : axis along which to take the statistic
                        default: None
        - nanpol (str): policy for NaNs, 'omit' or None
                        default: None
        - qu (list)   : quintiles to take, if median and std along which 
                        to take the statistic
                        default: [25, 75]
    
    Returns:
        - (nd array): std, SEM, quintiles or MAD of data along specified axis
    """

    if stats == 'mean' and error == 'std':
        if nanpol is None:
            return np.std(data, axis=axis)
        elif nanpol == 'omit':
            return np.nanstd(data, axis=axis)
    elif stats == 'mean' and error == 'sem':
        if nanpol is None:
            return st.sem(data, axis=axis)
        elif nanpol == 'omit':
            return st.sem(data, axis=axis, nan_policy='omit')
    elif stats == 'median' and error == 'std':
        if nanpol is None:
            return [np.percentile(data, qu[0], axis=axis), 
                    np.percentile(data, qu[1], axis=axis)]
        elif nanpol == 'omit':
            return [np.nanpercentile(data, qu[0], axis=axis), 
                    np.nanpercentile(data, qu[1], axis=axis)]
    elif stats == 'median' and error == 'sem':
        # MAD: median(abs(x - median(x)))
        if axis is not None:
            me_shape = list(data.shape)
            me_shape[axis] = 1
        else:
            me_shape = -1
        if nanpol is None:
            me = np.median(data, axis=axis).reshape(me_shape)
            return np.median(np.absolute(data - me), axis=axis)
        elif nanpol == 'omit':
            me = np.nanmedian(data, axis=axis).reshape(me_shape)
            return np.nanmedian(np.absolute(data - me), axis=axis)
    elif stats != 'median' and stats != 'mean':
        gen_util.accepted_values_error('stats', stats, ['mean', 'median'])
    else:
        gen_util.accepted_values_error('error', error, ['std', 'sem'])
    if nanpol is not None and nanpol != 'omit':
        gen_util.accepted_values_error('nanpol', nanpol, ['None', 'omit'])


#############################################
def integ(data, dx, axis=None):
    """
    integ(data, dx)

    Returns integral of data along specified axis.

    Required arguments:
        - data (np.array): data on which to calculate integral
        - dx (float)     : interval between data points

    Optional arguments:
        - axis (int) : axis along which to take the statistic
                       default: None
    
    Returns:
        - (nd array): integral of data along specified axis
    """

    # sum * freq
    return np.sum(data, axis)*1./dx


#############################################
def calc_norm(data, dimpos=None, out_range='pos'):
    """
    calc_norm(data)

    Normalizes data.

    Required arguments:
        - data (nd array): data to normalize

    Optional arguments:
        - dimpos (tuple) : dimension and position to use in calculating 
                           normalizing factor [dim, pos]. 
                           If None, the whole data is used.
                           default: None
        - out_range (pos): range to which to normalize data, i.e. 'pos': [0, 1], 
                           'neg': [-1, 1], 'one': around 1 or -1, 
                           'onepos': around 1
                           default: 'pos'

    Returns:
        - norm_data (nd array): normalized data
    """  
    if dimpos is None: 
        minim = np.min(data)
        maxim = np.max(data)
    else:
        norm_ind = [slice(None)] * dimpos[0] + [dimpos[1]]
        minim = np.min(data[norm_ind])
        maxim = np.max(data[norm_ind])
        mean = np.mean(data[norm_ind])

    if out_range in ['pos', 'neg']:
        norm_data = (data - minim)/(maxim - minim)
        if out_range == 'neg':
            norm_data = norm_data * 2.0 - 1.0
    elif out_range == 'one':
        norm_data = data/np.absolute(mean)
    elif out_range == 'onepos':
        norm_data = data/mean
    else:
        raise gen_util.accepted_values_error('out_range', out_range, 
                                             ['pos', 'neg', 'one', 'onepos'])
    
    return norm_data


#############################################
def norm_facts(data, dim='all'):
    """
    norm_facts(data)

    Calculates normalizing factors (mean and std) for a torch Tensor.

    Required arguments:
        - data (nd torch Tensor): data on which to calculate norm factors

    Optional arguments:
        - dim (str): if 'all', factors are calculated across entire Tensor.
                     if 'last', factors are calculated along the last dimension.
                     default: 'all'
    
    Returns:
        - means (1 or 2D torch Tensor): Tensor containing the mean or means.
        - stds (1 or 2D torch Tensor) : Tensor containing the std or stds.
    """   
    if dim == 'all':
        all_tr_flat = data.view(-1, 1)
        means = torch.mean(all_tr_flat)
        stds = torch.std(all_tr_flat)
    
    elif dim == 'last':
        # flatten along all but last dimension
        all_data_flatter = data.view((-1,) + data.size()[-1:])
        means = torch.mean(all_data_flatter, dim=0)
        stds = torch.std(all_data_flatter, dim=0)

    else:
        gen_util.accepted_values_error('dim', dim, ['all', 'last'])

    return means, stds


#############################################
def calc_op(data, vals='diff', op='diff', surp_dim=0):
    """
    calc_op(data)

    Applies to appropriate operation on a data array along the surprise 
    dimension.

    Required arguments:
        - data (nd array): data on which to calculate integral

    Optional arguments:
        - vals (str): 'surp', 'nosurp' or 'diff'
                           default: 'diff'
        - op (str)  : 'surp', 'nosurp' or 'diff'
                      default: 'diff'
        - dim (int) : surprise/no surprise dimension in data
    
    Returns:
        - data (nd array): data on which operation has been applied
    """
    nosurp_ind = tuple([slice(None)] * surp_dim + [0])
    surp_ind = tuple([slice(None)] * surp_dim + [1])

    if vals == 'diff':
        if op == 'diff':
            data = (data[surp_ind] - data[nosurp_ind])
        elif op == 'ratio':
            data = (data[surp_ind]/data[nosurp_ind])
        else:
            gen_util.accepted_values_error('op', op, ['diff', 'ratio'])
    elif vals == 'nosurp':
        data = data[nosurp_ind]
    elif vals == 'surp':
        data = data[surp_ind]
    else:
        gen_util.accepted_values_error('op', op, ['diff', 'surp', 'nosurp'])
    if len(data.shape) > surp_dim:
        if data.shape[surp_dim] == 1:
            data = data.squeeze(surp_dim)
    
    return data


#############################################
def run_permute(all_data, act_diff, div='half', stats='mean', op='diff', 
                tails='2', n_perms=10000, p_val=0.05):       
    """
    run_permute(all_data, act_data)

    Run a permutation analysis on data to identify elements (e.g., ROIs) showing 
    a significant difference between groups (e.g., surp vs no surp).

    Required arguments:
        - all_data (2D array)  : full data on which to run permutation
                                 (element x datapoints (all groups))
        - act_diff (1D array)  : actual differences between the groups
                                 by element
    Optional arguments:
        - div (str or int)  : nbr of datapoints in first group
                              default: 'half'
        - stats (str)       : statistic parameter, i.e. 'mean' or 'median'
                              default: 'mean'
        - op (str)          : operation to use to compare groups, 
                              i.e. 'diff': grp1-grp2, or 'ratio': grp1/grp2
                              default: 'diff'
        - tails (str or int): which tail(s) to test: 'up', 'lo', '2'
                              default: '2'
        - n_perms (int)     : nbr of permutations to run
                              default: 10000
        - p_val (float)     : p-value to use for significance thresholding 
                              (0 to 1)
                              default: 0.05
    Returns:
        - sign_elems (list or 1D array): array of elements showing significant
                                         differences, or list of arrays if 
                                         2-tailed analysis.
    """
    # create permutation indices
    if len(all_data.shape) > 2:
        raise NotImplementedError(('Permutation analysis not implemented '
                                    'for {}D data yet.').format(len(all_data.shape)))

    perms_inds = np.argsort(np.random.rand(all_data.shape[1], n_perms), axis=0)[np.newaxis, :, :]
    dim_data = np.arange(all_data.shape[0])[:, np.newaxis, np.newaxis]
    # generate permutation array
    permed_data = np.stack(all_data[dim_data, perms_inds])
    if div == 'half':
        div = int(all_data.shape[1]/2)
    # calculate grp1-grp2 or grp1/grp2: elem x datapoints x perms
    if op == 'diff':
        rand_vals = (mean_med(permed_data[:, 0:div], stats, axis=1) - 
                     mean_med(permed_data[:, div:], stats, axis=1))
    if op == 'ratio':
        rand_vals = (mean_med(permed_data[:, 0:div], stats, axis=1)/ 
                     mean_med(permed_data[:, div:], stats, axis=1))        
    
    sign_elem = id_elem(rand_vals, act_diff, tails, p_val, print_elems=True)

    return sign_elem


#############################################    
def id_elem(rand_vals, act_vals, tails='2', p_val=0.05, print_elems=False):
    """
    id_elem(rand_vals, act_vals)

    Identify elements whose actual values are beyond the threshold(s) obtained 
    with distributions of randomly generated values.

    Required arguments:
        - rand_vals (2D array): random values for each element: elem x val
        - act_vals (1D array) : actual values for each element

    Optional arguments:
        - tails (str or int): which tail(s) to test: 'up', 'lo', '2'
                              default: '2'
        - p_val (float)     : p-value to use for significance thresholding 
                              (0 to 1)
                              default: 0.05
        - print_elems (bool): if True, the numbers of significant elements and
                              their actual values are printed
    Returns:
        - elems (list or 1D array): array of elements showing significant
                                    differences, or list of arrays if 
                                    2-tailed analysis.
    """
    # calculate threshold difference for each element
    if tails == 'lo':
        threshs = np.percentile(rand_vals, p_val*100, axis=1)
        elems = np.where(act_vals < threshs)[0]
        if print_elems:
            print_elem_list(elems, 'lo', act_vals[elems])
    elif tails == 'up':
        threshs = np.percentile(rand_vals, 100-p_val*100, axis=1)
        elems = np.where(act_vals > threshs)[0]
        if print_elems:
            print_elem_list(elems, 'up', act_vals[elems])
    elif str(tails) == '2':
        lo_threshs = np.percentile(rand_vals, p_val*100/2.0, axis=1)
        lo_elems = np.where(act_vals < lo_threshs)[0]
        up_threshs = np.percentile(rand_vals, 100-p_val*100/2.0, axis=1)
        up_elems = np.where(act_vals > up_threshs)[0]
        if print_elems:
            print_elem_list(lo_elems, 'lo', act_vals[lo_elems])
            print_elem_list(up_elems, 'up', act_vals[up_elems])
        elems = [lo_elems, up_elems]
    else:
        gen_util.accepted_values_error('tails', tails, ['up', 'lo', '2'])
    return elems


#############################################
def print_elem_list(elems, tail='up', act_vals=None):
    """
    id_elem(rand_vals, act_vals)

    Print numbers of elements showing significant difference in a specific tail,
    and optionally their actual values.

    Required arguments:
        - elems (1D array): array of elements showing significant differences

    Optional arguments:
        - tails (str)        : which tail the elements are in: 'up', 'lo'
                               default: 'up'
        - act_vals (1D array): array of actual values corresponding to elems 
                               (same length). If None, actual values are not 
                               printed.
    """

    if len(elems) == 0:
        print('\tSignif {}: None'.format(tail))
    else:
        print('\tSignif {}: {}'.format(tail, ', '.join('{}'.format(x) 
            for x in elems)))
        if act_vals is not None:
            if len(act_vals) != len(elems):
                raise ValueError(('\'elems\' and \'act_vals\' should be the '
                                  'same length, but are of length {} and {} '
                                  'respectively.').format(len(elems), len(act_vals)))
            print('\tvals: {}'.format(', '.join(['{:.2f}'.format(x) 
                                        for x in act_vals])))    


#############################################
def get_percentiles(CI=95):
    """
    get_percentiles()

    Returns percentile and names corresponding to the confidence interval
    (centered on the median).

    Optional arguments:
        - CI (float): confidence interval
                      default: 95

    Returns:
        - ps (list)     : list of percentile values, e.g., [2.5, 97.5]
        - p_names (list): list of percentile names, e.g., ['p2p5', 'p97p5'] 

    """

    ps = [(100.-CI)*0.5, CI*0.5+50.] # high and lo quartiles
    p_names = []
    for p in ps:
        p_res = p%1
        if p_res == 0:
            p_names.append('p{}'.format(int(p)))
        else:
            p_names.append('p{}p{}'.format(int(p), str(p_res)[2]))

    return ps, p_names