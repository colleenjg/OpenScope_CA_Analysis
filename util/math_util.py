"""
math_util.py

This module contains basic math functions.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import re

import numpy as np
import scipy.ndimage as scn
import scipy.stats as st
from sklearn import svm
from sklearn.model_selection import cross_val_score
import torch

from util import gen_util


#############################################
def mean_med(data, stats='mean', axis=None, nanpol=None):
    """
    mean_med(data)

    Returns the mean or median of the data along a specified axis, depending on
    which statistic is requested.

    Required args:
        - data (nd array): data on which to calculate statistic

    Optional args:
        - stats (str) : 'mean' or 'median'
                        default: 'mean'
        - axis (int)  : axis along which to take the statistic
                        default: None
        - nanpol (str): policy for NaNs, 'omit' or None
                        default: None
    
    Returns:
        - me (nd array or num): mean or median of data along specified axis
    """

    if stats == 'mean':
        if nanpol is None:
            me = np.mean(data, axis=axis)
        elif nanpol == 'omit':
            me = np.nanmean(data, axis=axis)
    elif stats == 'median':
        if nanpol is None:
            me = np.median(data, axis=axis)
        elif nanpol == 'omit':
            me = np.nanmedian(data, axis=axis)
    else:
        gen_util.accepted_values_error('stats', stats, ['mean', 'median'])
    if nanpol is not None and nanpol != 'omit':
        gen_util.accepted_values_error('nanpol', nanpol, ['None', 'omit'])

    return me


#############################################
def error_stat(data, stats='mean', error='sem', axis=None, nanpol=None, 
               qu=[25, 75]):
    """
    error_stat(data)

    Returns the std, SEM, quartiles or median absolute deviation (MAD) of data 
    along a specified axis, depending on which statistic is requested.

    Required args:
        - data (nd array): data on which to calculate statistic

    Optional args:
        - stats (str) : 'mean' or 'median'
                        default: 'mean'
        - error (str) : 'std' (for std or quintiles) or 'sem' (for SEM or MAD)
                        default: 'sem'
        - axis (int)  : axis along which to take the statistic
                        default: None
        - nanpol (str): policy for NaNs, 'omit' or None
                        default: None
        - qu (list)   : quintiles to take, if median and std along which 
                        to take the statistic
                        default: [25, 75]
    
    Returns:
        - error (nd array or num): std, SEM, quintiles or MAD of data along 
                                   specified axis
    """

    if stats == 'mean' and error == 'std':
        if nanpol is None:
            error = np.std(data, axis=axis)
        elif nanpol == 'omit':
            error = np.nanstd(data, axis=axis)
    elif stats == 'mean' and error == 'sem':
        if nanpol is None:
            error = st.sem(data, axis=axis)
        elif nanpol == 'omit':
            error = st.sem(data, axis=axis, nan_policy='omit')
    elif stats == 'median' and error == 'std':
        if nanpol is None:
            error = [np.percentile(data, qu[0], axis=axis), 
                     np.percentile(data, qu[1], axis=axis)]
        elif nanpol == 'omit':
            error = [np.nanpercentile(data, qu[0], axis=axis), 
                     np.nanpercentile(data, qu[1], axis=axis)]
        
    elif stats == 'median' and error == 'sem':
        # MAD: median(abs(x - median(x)))
        if axis is not None:
            me_shape       = list(data.shape)
            me_shape[axis] = 1
        else:
            me_shape = -1
        if nanpol is None:
            me    = np.asarray(np.median(data, axis=axis)).reshape(me_shape)
            error = np.median(np.absolute(data - me), axis=axis)
        elif nanpol == 'omit':
            me    = np.asarray(np.nanmedian(data, axis=axis)).reshape(me_shape)
            error = np.nanmedian(np.absolute(data - me), axis=axis)
    elif stats != 'median' and stats != 'mean':
        gen_util.accepted_values_error('stats', stats, ['mean', 'median'])
    else:
        gen_util.accepted_values_error('error', error, ['std', 'sem'])
    if nanpol is not None and nanpol != 'omit':
        gen_util.accepted_values_error('nanpol', nanpol, ['[None]', 'omit'])

    error = np.asarray(error)
    if len(error.shape) == 0:
        error = error.item()

    return error


#############################################
def get_stats(data, stats='mean', error='sem', axes=None, nanpol=None,
              qu=[25, 75]):
    """
    get_stats(data)

    Returns statistics calculated as follows: means/medians are calculated 
    along each axis successively, then the full statistics are calculated along 
    the last axis in the list. 
    
    Returns statistics (me, error x values) statistics as a single array.
    Note that is stats='median' and error='std', the error will be in two 
    rows/cols.
    
    Required args:
        - data (nd array): data array (at least 2D)

    Optional args:
        - stats (str)       : stats to take, i.e., 'mean' or 'median'
                              default: 'mean'
        - error (str)       : error to take, i.e., 'std' (for std or quintiles) 
                              or 'sem' (for SEM or MAD)
                              default: 'std'
        - axes (int or list): axes along which to  take statistics. If a list  
                              is passed.
                              If None, axes are ordered reverse sequentially 
                              (-1 to 0).
                              default: None
        - nanpol (str)      : policy for NaNs, 'omit' or None
                              default: None
        - qu (list)         : quintiles to take, if median and std along which 
                              to take the statistic
                              default: [25, 75]

    Returns:
        - data_stats (nd array): stats array, structured as: 
                                 stat type (me, error x values) x 
                                     remaining_dims
    """

    if data.shape == 1:
        raise ValueError('Data array must comprise at least 2 dimensions.')

    if axes is None:
        # reversed list of axes, omitting last one
        axes = list(range(0, len(data.shape)))[::-1]
    axes = gen_util.list_if_not(axes)

    # make axis numbers positive
    axes = gen_util.pos_idx(axes, len(data.shape))

    if len(axes) > len(data.shape):
        raise ValueError(('Must provide no more axes value than the number of '
                          'data axes.'))
    
    if len(axes) > 1:
        # take the mean/median successively across axes
        prev = []
        for ax in axes[:-1]:
            # update axis number based on previously removed axes
            sub = sum(p < ax for p in prev)
            prev.append(ax)
            ax = ax-sub
            data = mean_med(data, stats=stats, axis=ax, nanpol=nanpol)
        axis = axes[-1]
        if axis != -1:
            sub = sum(p < axis for p in prev)
            axis = axis-sub
    else:
        axis = axes[0]
        
    # mean/med along units axis (last)
    me  = mean_med(data, stats=stats, axis=axis, nanpol=nanpol) 
    err = error_stat(data, stats=stats, error=error, axis=axis, nanpol=nanpol, 
                     qu=qu)
    
    # ensures that these are arrays
    me = np.asarray(me)
    err = np.asarray(err)

    if stats=='median' and error=='std':
        me = np.expand_dims(me, 0)
        data_stats = np.concatenate([me, err], axis=0)
    else:
        data_stats = np.stack([me, err])

    return data_stats


#############################################
def print_stats(stats, stat_str=None, ret_str_only=False):
    """
    print_stats(stats)

    Prints the statistics.

    Required args:
        - stats (array-like): stats, structured as [me, err]

    Optional args:
        - stat_str (str)     : string associated with statistics
                               default: None
        - ret_str_only (bool): if True, string is returned instead of printed
                               default: False
    
    Returns:
        if ret_str_only:
            full_stat_str: full string associated with statistics
    """

    me = stats[0]
    err = stats[1:]
    
    err_str = '/'.join(['{:.3f}'.format(e) for e in err])

    plusmin = u'\u00B1'

    if stat_str is None:
        stat_str = ''
    else:
        stat_str = '{}: '.format(stat_str)

    full_stat_str = u'{}{:.5f} {} {}'.format(stat_str, me, plusmin, err_str)
    if ret_str_only:
        return full_stat_str
    else:
        print(full_stat_str)
        

#############################################
def integ(data, dx, axis=None, nanpol=None):
    """
    integ(data, dx)

    Returns integral of data along specified axis.

    Required args:
        - data (nd array): data on which to calculate integral
        - dx (num)       : interval between data points

    Optional args:
        - axis (int)  : axis along which to take the statistic
                        default: None
        - nanpol (str): policy for NaNs, 'omit' or None
                        default: None
    
    Returns:
        - integ_data (nd array): integral of data along specified axis
    """

    # sum * freq
    if nanpol == 'omit':
        integ_data = np.nansum(data, axis)*dx
    elif nanpol is None:
        integ_data = np.sum(data, axis)*dx
    else:
        gen_util.accepted_values_error('nanpol', nanpol, ['None', 'omit'])

    return integ_data


#############################################
def rolling_mean(vals, win=3):
    """
    rolling_mean(vals)

    Returns rolling mean over the last dimension of the input data.

    Required args:
        - vals (nd array): data array, for which rolling mean will be taken 
                           along last dimension

    Optional args:
        - win (int): length of the rolling mean window
                     default: 3

    Returns:
        - vals_out (nd array): rolling mean data array 
    """

    targ_dims = tuple([1] * (len(vals.shape) - 1) + [win])
    weights = (np.repeat(1.0, win)/win).reshape(targ_dims)
    vals_out = scn.convolve(vals, weights, mode='mirror')

    return vals_out


#############################################
def calc_op(data, op='diff', dim=0, rev=False):
    """
    calc_op(data)

    Returns result of specified operation performed on a data array defined
    by the specified dimension.

    Required args:
        - data (nd array): data on which to run operation, with length 2 along 
                           dim.

    Optional args:
        - op (str) : 'diff': index 1 - 0, or 'ratio': index 1/0.
                     If int, the corresponding data index is returned.
                     default: 'diff'
        - dim (int): dimension along which to do operation
    
    Returns:
        - data (nd array): data on which operation has been applied
    """
    
    if data.shape[dim] != 2:
        raise ValueError('Data should have length 2 along dim: {}'.format(dim))

    if isinstance(op, int):
        data_idx = gen_util.slice_idx(dim, op)
        data = data[data_idx]
    else:
        if rev:
            fir, sec = [0, 1]
        else:
            fir, sec = [1, 0]
        fir_idx = gen_util.slice_idx(dim, fir)
        sec_idx = gen_util.slice_idx(dim, sec)
        if op == 'diff':
            data = (data[fir_idx] - data[sec_idx])
        elif op == 'ratio':
            data = (data[fir_idx]/data[sec_idx])
    
    return data


#############################################
def scale_facts(data, axis=None, pos=None, sc_type='min_max', extrem='reg', 
               mult=1.0, shift=0.0, nanpol=None, allow_0=False):
    """
    scale_facts(data)

    Returns scaling factors.

    Required args:
        - data (nd array): data to scale

    Optional args:
        - axis (int)    : axis along which to calculate scaling values (if None, 
                          entire data array is used)     
        - pos (int)     : position along axis along which to calculate scaling 
                          values (if None, each position is scaled separately)
        - sc_type (str) : type of scaling to use
                          'min_max'  : (data - min)/(max - min)
                          'scale'    : (data - 0.0)/std
                          'stand'    : (data - mean)/std
                          'stand_rob': (data - median)/IQR (75-25)
                          'center'   : (data - mean)/1.0
                          'unit'     : (data - 0.0)/abs(mean)
                          default: 'min_max'
        - extrem (str)  : only needed if min_max  or stand_rob scaling is used. 
                          'reg': the minimum and maximum (min_max) or 25-75 IQR 
                                 of the data are used 
                          'perc': the 5th and 95th percentiles are used as min
                                  and max respectively (robust to outliers)
        - mult (num)    : value by which to multiply scaled data
                          default: 1.0
        - shift (num)   : value by which to shift scaled data (applied after
                          mult)
                          default: 0.0
        - nanpol (str)  : policy for NaNs, 'omit' or None
                          default: None
        - allow_0 (bool): if True, div == 0 is allowed (likely resulting from 
                          np.nans)

    Returns:
        - sub (float or list): value(s) to subtract from scaled data
        - div (float or list): value(s) by which to divide scaled data
        - mult (num)         : value by which to multiply scaled data
        - shift (num)        : value by which to shift scaled data (applied 
                               after mult)
    """  

    if pos is not None and axis is None:
        raise ValueError('Must pass an axis if passing a position.')
    
    if pos is not None:
        sc_idx = gen_util.slice_idx(axis, pos) # for a slice
        axis = None
    else:
        sc_idx = gen_util.slice_idx(None, None) # for entire data

    if sc_type == 'stand':
        sub = mean_med(data[sc_idx], stats='mean', axis=axis, nanpol=nanpol)
        div = error_stat(data[sc_idx], stats='mean', error='std', axis=axis, 
                         nanpol=nanpol)
    elif sc_type == 'stand_rob':
        sub = mean_med(data[sc_idx], stats='median', axis=axis, nanpol=nanpol)
        if extrem == 'reg':
            qus = [25, 75]
        elif extrem == 'perc':
            qus = [5, 95]
        else:
            gen_util.accepted_values_error('extrem', extrem, ['reg', 'perc'])
        qs  = error_stat(data[sc_idx], stats='median', error='std', axis=axis, 
                         qu=qus, nanpol=nanpol)
        div = qs[1] - qs[0]
    elif sc_type == 'center':
        sub = mean_med(data[sc_idx], stats='mean', axis=axis, nanpol=nanpol)
        div = 1.0
    elif sc_type == 'scale':
        sub = 0.0
        div = error_stat(data[sc_idx], stats='mean', error='std', axis=axis, 
                         nanpol=nanpol)
    elif sc_type == 'unit':
        sub = 0.0
        div = np.absolute(mean_med(data[sc_idx], stats='mean', axis=axis, 
                                   nanpol=nanpol))
    elif sc_type == 'min_max':
        if nanpol is not None and nanpol != 'omit':
            gen_util.accepted_values_error('nanpol', nanpol, ['[None]', 'omit'])
        if extrem == 'reg':
            if nanpol is None:
                minim = np.min(data[sc_idx], axis=axis)
                maxim = np.max(data[sc_idx], axis=axis)
            elif nanpol == 'omit':
                minim = np.nanmin(data[sc_idx], axis=axis)
                maxim = np.nanmax(data[sc_idx], axis=axis)
        elif extrem == 'perc':
            if nanpol is None:
                minim = np.percentile(data[sc_idx], 5, axis=axis)
                maxim = np.percentile(data[sc_idx], 95, axis=axis)
            elif nanpol == 'omit':
                minim = np.percentile(data[sc_idx], 5, axis=axis)
                maxim = np.percentile(data[sc_idx], 95, axis=axis)
        else:
            gen_util.accepted_values_error('extrem', extrem, ['reg', 'perc'])
        sub = minim
        div = maxim - minim
    else:
        gen_util.accepted_values_error('sc_type', sc_type, 
                 ['stand', 'stand_rob', 'center', 'scale', 'min_max'])
    
    if not allow_0 and (np.asarray(div) == 0).any():
        raise ValueError('Scaling cannot proceed due to division by 0.')

    if isinstance(sub, np.ndarray):
        sub = sub.tolist()
    if isinstance(div, np.ndarray):
        div = div.tolist()

    return sub, div, mult, shift


#############################################
def extrem_to_med(data, ext_p=[5, 95]):
    """
    Returns data array with values above and below the threshold percentiles 
    replaced with the median, for each channel.

    Required args:
        - data (2D array): data array, structured as vals x channels

    Optional args:
        - ext_p (list): percentile values to use [low, high]
                        default: [5, 95]
    
    Returns:
        - data (2D array): data array with extreme values replaced by median, 
                           structured as vals x channels
    """

    p_lo, p_hi = ext_p

    if p_hi < p_lo:
        raise ValueError('p_lo must be smaller than p_hi.')
    meds, lo, hi = [np.nanpercentile(data, p, axis=0).reshape([1, -1]) 
                                                   for p in [50, p_lo, p_hi]]
    modif = np.where(np.add(data < lo, data > hi))
    data[modif] = meds[:, modif[1]]

    return data


#############################################
def scale_data(data, axis=None, pos=None, sc_type='min_max', extrem='reg', 
               mult=1.0, shift=0.0, facts=None, nanpol=None):
    """
    scale_data(data)

    Returns scaled data, and factors if None are passed.

    Required args:
        - data (nd array): data to scale

    Optional args:
        - axis (int)   : axis to collapse when scaling values (if None, 
                         entire data array is collapsed)   
        - pos (int)    : position along axis to retain when calculating scaling 
                         values (if None, each position is scaled separately)
        - sc_type (str): type of scaling to use
                         'min_max'  : (data - min)/(max - min)
                         'scale'    : (data - 0.0)/std
                         'stand'    : (data - mean)/std
                         'stand_rob': (data - median)/IQR (75-25)
                         'center'   : (data - mean)/1.0
                         'unit'     : (data - 0.0)/abs(mean)
                         default: 'min_max'
        - extrem (str) : only needed if min_max scaling is used. 
                         'reg': the minimum and maximum of the data are used 
                         'perc': the 5th and 95th percentiles are used as min
                                 and max respectively (robust to outliers)
        - mult (num)   : value by which to multiply scaled data
                         default: 1.0
        - shift (num)  : value by which to shift scaled data (applied after
                         mult)
                         default: 0.0
        - facts (list) : list of sub, div, mult and shift values to use on data
                         (overrides sc_type, extrem, mult and shift), where
                         sub is the value subtracted and div is the value
                         used as divisor (before applying mult and shift)
                         default: None
        - nanpol (str) : policy for NaNs, 'omit' or None
                         default: None

    Returns:
        - sc_data (nd array): scaled data
        if facts value passed is None:
        - facts (list)      : list of sub, div, mult and shift values used on
                              data, where sub is the value(s) subtracted and 
                              div is the value(s) used as divisor(s) (before 
                              applying mult and shift)
    """  
    
    ret_facts = False
    if facts is None:
        facts = scale_facts(data, axis, pos, sc_type, extrem=extrem, mult=mult, 
                            shift=shift, nanpol=nanpol)
        ret_facts = True
    elif len(facts) != 4:
        raise ValueError(('If passing factors, must pass 4 items: '
                        'sub, div, mult and shift.'))
    
    sub, div, mult, shift = np.asarray([fact for fact in facts])

    if axis is not None:
        sub = np.expand_dims(sub, axis)
        div = np.expand_dims(div, axis)
    
    data = (data - sub)/div * mult + shift

    if ret_facts:
        return data, facts

    else:
        return data


#############################################
def calc_mag_change(data, change_dim, item_dim, order=1, op='diff', 
                    stats='mean', error='sem', scale=False, axis=0, pos=0, 
                    sc_type='unit'):
    """
    calc_mag_change(data, change_dim, item_dim)

    Returns the magnitude diff/ratio or statistics of diff/ratio between 
    dimensions.

    Required args:
        - data (nd array) : data, with at least 2 dimensions or 3 if scaling
        - change_dim (int): dimension along which to calculate change
        - item_dim (int)  : dimension along which to scale or take statistics
    
    Optional args:
        - order (int)    : order of the norm (or 'stats' to take change stats)
                           default: 1
        - op (str)       : 'diff': index 1 - 0, or 'ratio': index 1/0.
                           default: 'diff'
        - stats (str)    : stats to take, i.e., 'mean' or 'median'
                           default: 'mean'
        - error (str)    : error to take, i.e., 'std' (for std or quintiles) 
                           or 'sem' (for SEM or MAD)
                           default: 'std'
        - scale (bool)   : if True, data is scaled using axis, pos and sc_type
                           default: False
        - axis (int)     : axis along which to calculate scaling values (if  
                           None, entire data array is used)     
        - pos (int)      : position along axis along which to calculate scaling 
                           values (if None, each position is scaled separately)
        - sc_type (str)  : type of scaling to use
                           'min_max'  : (data - min)/(max - min)
                           'scale'    : (data - 0.0)/std
                           'stand'    : (data - mean)/std
                           'stand_rob': (data - median)/IQR (75-25)
                           'center'   : (data - mean)/1.0
                           'unit'     : (data - 0.0)/abs(mean)
                           default: 'min_max'

    Returns:
        if order == 'stats:
            - data_ch_stats (nd array): array of magnitude change statistics,
                                        where first dimension are the stats
                                        (me, de)
        elif order is an int:
            - data_ch_norm (nd array) : array of norm values
    """

    if op not in ['diff', 'ratio']:
        raise ValueError('op can only take values `diff` or `ratio`.')
    
    data_change = np.absolute(calc_op(data, op, dim=change_dim))

    if item_dim > change_dim: # adjust dimension if needed
        item_dim -= 1

    if scale and axis is not None and axis > change_dim: # adjust dim if needed
        axis += -1

    if order == 'stats':
        if scale:
            data_change, _ = scale_data(data_change, axis, pos, sc_type)
        data_ch_stats = get_stats(data_change, stats, error, axes=item_dim)
        return data_ch_stats
    else:
        data_ch_norm = np.linalg.norm(data_change, ord=int(order), 
                                      axis=item_dim)
        if scale:
            data_ch_norm, _ = scale_data(data_ch_norm, axis, pos, sc_type)
        return data_ch_norm


#############################################
def calc_mult_comp(n_comp, p_val=0.05, n_perms=10000, min_n=100):
    """
    calc_mult_comp(n_comp)

    Returns new p-value, based on the original p-value and a new number of 
    permutations using a Bonferroni correction.
    
    Specifically, the p-value is divided by the number of comparisons,
    and the number of permutations is increased if necessary to ensure a 
    sufficient number of permuted datapoints will be outside the CI 
    to properly measure the significance threshold.

    Required args:
        - n_comp (int): number of comparisons
    
    Optional args:
        - n_perms (int): original number of permutations
                         default: 10,000
        - p_val (num)  : original p_value
                         default: 0.05
        - min_n (int)  : minimum number of values required outside of the CI
                         default: 100
    
    Return:
        - new_p_val (num)  : new p-value
        - new_n_perms (num): new number of permutations
    """
    
    new_p_val   = float(p_val)/n_comp

    new_n_perms = int(np.ceil(np.max([n_perms, float(min_n)/new_p_val])))


    return new_p_val, new_n_perms


#############################################
def run_permute(all_data, n_perms=10000, lim_e6=350):
    """
    run_permute(all_data)

    Returns array containing data permuted the number of times requested. Will
    throw an AssertionError if permuted data array is projected to exceed 
    limit. 

    Required args:
        - all_data (2D array)  : full data on which to run permutation
                                 (items x datapoints to permute (all groups))

    Optional args:
        - n_perms  (int): nbr of permutations to run
                          default: 10000
        - lim_e6 (num)  : limit (when multiplied by 1e6) to permuted data array 
                          size at which an AssertionError is thrown. 'none' for
                          no limit
                          default: 350
    Returns:
        - permed_data (3D array): array of multiple permutation of the data, 
                                  structured as: 
                                  items x datapoints x permutations
    """

    if len(all_data.shape) > 2:
        raise NotImplementedError(('Permutation analysis only implemented for '
                                   '2D data.'))

    # checks final size of permutation array and throws an error if
    # it is bigger than accepted limit.
    perm_size = np.product(all_data.shape) * n_perms
    if lim_e6 != 'none':
        lim = int(lim_e6*1e6)
        fold = int(np.ceil(float(perm_size)/lim))
        permute_cri = ('\nPermutation array exceeds allowed size '
                    '({} * 10^6) by {} fold.').format(lim_e6, fold)
        assert (perm_size < lim), permute_cri

    # (item x datapoints (all groups))
    perms_idxs = np.argsort(np.random.rand(all_data.shape[1], n_perms), 
                            axis=0)[np.newaxis, :, :]
    dim_data   = np.arange(all_data.shape[0])[:, np.newaxis, np.newaxis]

    # generate permutation array
    permed_data = np.stack(all_data[dim_data, perms_idxs])

    return permed_data


#############################################
def permute_diff_ratio(all_data, div='half', n_perms=10000, stats='mean', 
                       nanpol=None, op='diff'):       
    """
    permute_diff_ratio(all_data)

    Returns all group mean/medians or differences/ratios between two groups 
    resulting from the permutation analysis on input data.

    Required args:
        - all_data (2D array)  : full data on which to run permutation
                                 (items x datapoints to permute (all groups))

    Optional args:
        - div (str or int)  : nbr of datapoints in first group
                              default: 'half'
        - n_perms (int)     : nbr of permutations to run
                              default: 10000
        - stats (str)       : statistic parameter, i.e. 'mean' or 'median'
                              default: 'mean'
        - nanpol (str)      : policy for NaNs, 'omit' or None when taking 
                              statistics
                              default: None
        - op (str)          : operation to use to compare groups, 
                              i.e. 'diff': grp1-grp2, or 'ratio': grp1/grp2
                              or 'none'
                              default: 'diff'

    Returns:
        - all_rand_vals (2 or 3D array): permutation results, structured as:
                                             (grps if op is 'none' x) 
                                             items x perms
    """

    if len(all_data.shape) > 2:
        raise NotImplementedError(('Significant difference/ratio analysis only '
                                   'implemented for 2D data.'))
    
    all_rand_res = []
    perm = True
    n_perms_tot = n_perms
    perms_done = 0

    if div == 'half':
        div = int(all_data.shape[1]//2)

    while perm:
        try:
            perms_rem = n_perms_tot - perms_done
            if perms_rem < n_perms:
                n_perms = perms_rem
            permed_data = run_permute(all_data, n_perms=n_perms)

            rand = np.stack([mean_med(permed_data[:, 0:div], stats, axis=1, 
                                      nanpol=nanpol), 
                             mean_med(permed_data[:, div:], stats, axis=1, 
                                      nanpol=nanpol)])
            
            if op == 'none':
                rand_res = rand
            # calculate grp1-grp2 or grp1/grp2 -> elem x perms
            else:
                rand_res = calc_op(rand, op, dim=0)
            
            del permed_data
            all_rand_res.append(rand_res)
            perms_done += n_perms

            if perms_done >= n_perms_tot:
                perm = False

        except AssertionError as err:
            print(err)
            print('Doing permutations in smaller batches.')
            # retrieve fold from error message.
            err_fold_str = str(err)[str(err).find('by '):]
            fold = int(re.findall('\d+', err_fold_str)[0])
            n_perms = int(n_perms//fold)

    all_rand_res = np.concatenate(all_rand_res, axis=-1)

    return all_rand_res


#############################################
def print_elem_list(elems, tail='up', act_vals=None):
    """
    print_elem_list(rand_vals, act_vals)

    Print numbers of elements showing significant difference in a specific tail,
    and optionally their actual values.

    Required args:
        - elems (1D array): array of elements showing significant differences

    Optional args:
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
                raise ValueError(('`elems` and `act_vals` should be the '
                                  'same length, but are of length {} and {} '
                                  'respectively.').format(len(elems), 
                                                          len(act_vals)))
            print('\tVals: {}'.format(', '.join(['{:.2f}'.format(x) 
                                                 for x in act_vals])))    


#############################################
def lin_interp_nan(data_arr):
    """
    lin_interp_nan(data_arr)

    Linearly interpolate NaNs in data array.

    Required args:
        - data_arr (1D array): data array

    Returns:
        - data_arr_interp (1D array): linearly interpolated data array
    """

    arr_len = len(data_arr)

    # get indices of non NaN values
    nan_idx = np.where(1 - np.isnan(data_arr))[0]

    arr_no_nans = data_arr[nan_idx]
    data_arr_interp = np.interp(range(arr_len), nan_idx, arr_no_nans)

    return data_arr_interp


#############################################    
def id_elem(rand_vals, act_vals, tails='2', p_val=0.05, min_n=100, 
            print_elems=False, ret_th=False):
    """
    id_elem(rand_vals, act_vals)

    Returns elements whose actual values are beyond the threshold(s) obtained 
    with distributions of randomly generated values. 
    Optionally also returns the threshold(s) for each element.
    Optionally also prints significant element indices and their values.

    Required args:
        - rand_vals (2D array): random values for each element: elem x val
                                (or 1D, but will be treated as if it were 2D
                                 with 1 element)
        - act_vals (1D array) : actual values for each element

    Optional args:
        - tails (str or int): which tail(s) to test: 'up', 'lo', '2'
                              default: '2'
        - p_val (num)       : p-value to use for significance thresholding 
                              (0 to 1)
                              default: 0.05
        - min_n (int)       : minimum number of values required outside of the 
                              CI
                              default: 100
        - print_elems (bool): if True, the indices of significant elements and
                              their actual values are printed
        - ret_th (bool)     : if True, thresholds are returned for each element

    Returns:
        - elems (list): list of elements showing significant differences, or 
                        list of lists if 2-tailed analysis [lo, up].
        if ret_th, also:
        - threshs (list): list of threshold(s) for each element, either one 
                          value per element if 1-tailed analysis, or list of 2 
                          thresholds if 2-tailed [lo, up].
    """

    act_vals  = np.asarray(act_vals)
    rand_vals = np.asarray(rand_vals)

    nan_act_vals  = np.isnan(act_vals).any()
    nan_rand_vals = np.isnan(rand_vals).any()
    
    if nan_act_vals > 0:
        raise ValueError('NaNs encountered in actual values.')
    if nan_rand_vals > 0:
        raise ValueError('NaNs encountered in random values.')

    # check whether there are enough values for determining thresholds
    out_vals = int(rand_vals.shape[-1] * p_val)
    if out_vals < min_n:
        raise ValueError(('Insufficient number of values ({}) outside the '
                          'CI (< {}).'.format(out_vals, min_n)))

    single = False
    if len(rand_vals.shape) == 1:
        single = True

    if tails == 'lo':
        threshs = np.percentile(rand_vals, p_val*100, axis=-1)
        elems = np.where(act_vals < threshs)[0]
        if print_elems:
            print_elem_list(elems, 'lo', act_vals[elems])
        elems = elems.tolist()
    elif tails == 'up':
        threshs = np.percentile(rand_vals, 100-p_val*100, axis=-1)
        elems = np.where(act_vals > threshs)[0]
        if print_elems:
            print_elem_list(elems, 'up', act_vals[elems])
        elems = elems.tolist()
    elif str(tails) == '2':
        lo_threshs = np.percentile(rand_vals, p_val*100/2., axis=-1)
        lo_elems = np.where(act_vals < lo_threshs)[0]
        up_threshs = np.percentile(rand_vals, 100-p_val*100/2., axis=-1)
        up_elems = np.where(act_vals > up_threshs)[0]
        if print_elems:
            print_elem_list(lo_elems, 'lo', act_vals[lo_elems])
            print_elem_list(up_elems, 'up', act_vals[up_elems])
        elems = [lo_elems.tolist(), up_elems.tolist()]
    else:
        gen_util.accepted_values_error('tails', tails, ['up', 'lo', '2'])
    
    if ret_th:
        if tails in ['lo', 'up']:
            if single:
                threshs = [threshs]
            else:
                threshs = threshs.tolist()
        else:
            if single:
                threshs = [[lo_threshs, up_threshs]]
            else:
                threshs = [[lo, up] for lo, up in zip(lo_threshs, up_threshs)]
        return elems, threshs

    return elems


#############################################
def get_percentiles(CI=0.95, tails=2):
    """
    get_percentiles()

    Returns percentiles and names corresponding to the confidence interval
    (centered on the median).

    Optional args:
        - CI (num)          : confidence interval
                              default: 0.95
        - tails (str or int): which tail(s) to test: 'up', 'lo', '2'
                              default: '2'

    Returns:
        - ps (list)     : list of percentile values, e.g., [2.5, 97.5]
        - p_names (list): list of percentile names, e.g., ['p2-5', 'p97-5']
    """

    if CI < 0 or CI > 1:
        raise ValueError('CI must be between 0 and 1.')

    if tails == 'up':
        ps = [0.0, CI]
    elif tails == 'lo':
        ps = [1.0 - CI, 1.0]
    elif tails in ['2', 2]:
        ps = [0.5 * (1.0 + v) for v in [-CI, CI]]
    else:
        gen_util.accepted_values_error('tails', tails, ['up', 'lo', 2])

    ps = [100.0 * p for p in ps]
    p_names = []
    for p in ps:
        p_names.append('p{}'.format(gen_util.num_to_str(p)))

    return ps, p_names


#############################################
def autocorr(data, lag):
    """
    Calculates autocorrelation on data series.

    Required args:
        - data (1D array): 1D dataseries
        - lag (int)      : lag in steps
    
    Returns:
        - autoc_snip (1D array): 1D array of autocorrelations at specified lag
    """

    autoc = np.correlate(data, data, 'full')
    mid = int((autoc.shape[0] - 1)//2)
    autoc_snip = autoc[mid - lag:mid + lag + 1]
    autoc_snip /= np.max(autoc_snip)
    return autoc_snip


#############################################
def autocorr_stats(data, lag, spu=None, byitem=True, stats='mean', error='std', 
                   nanpol=None):
    """
    autocorr_stats(data, lag)
    
    Returns average autocorrelation across data series.

    Required args:
        - data (list or 2-3D array): list of series or single series 
                                     (2D array), where autocorrelation is 
                                     calculated along the last axis. 
                                     Structured as: 
                                         (blocks x ) item x frame
        - lag (num)                : lag for which to calculate 
                                     autocorrelation (in steps ir in units 
                                     if steps per units (spu) is provided).

    Optional args:
        - spu (num)    : spu (steps per unit) value to calculate lag in steps
                         default: None
        - byitem (bool): if True, autocorrelation statistics are taken by 
                         item, else across items
                         default: True
        - stats (str)  : statistic parameter, i.e. 'mean' or 'median'
                         default: 'mean'
        - error (str)  : error statistic parameter, i.e. 'std' or 'sem'
                         default: 'std
        - nanpol (str) : policy for NaNs, 'omit' or None when taking statistics
                         default: None

    Returns:
        - xran (array-like)             : range of lag values in frames or in
                                          units if fpu is not None.
                                          (length is equal to last  
                                          dimension of autocorr_stats) 
        - autocorr_stats (2 or 3D array): autocorr statistics, structured as 
                                          follows:
                                          stats (me, de) x (item if item x) lag
    """
    
    if spu is None:
        lag_fr = int(lag)
    else:
        lag_fr = int(lag * spu)

    snip_len = 2 * lag_fr + 1

    data = gen_util.list_if_not(data)
    n_series = len(data)
    n_items  = len(data[0])

    autocorr_snips = np.empty((n_series, n_items, snip_len))

    for s, series in enumerate(data):
        sc_vals = series - np.mean(series, axis=1)[:, np.newaxis]
        for i, item in enumerate(sc_vals):
            autocorr_snips[s, i] = autocorr(item, lag_fr)

    xran = np.linspace(-lag, lag, snip_len)

    # take autocorrelations statistics for each lag across blocks
    if byitem:
        axes = 0
    else:
        axes = [0, 1]

    autocorr_stats = get_stats(autocorr_snips, stats, error, axes=axes, 
                               nanpol=nanpol)

    return xran, autocorr_stats


#############################################
def run_cv_svm(inp, target, cv=5, shuffle=False, stats='mean', error='std', 
               class_weight='balanced', n_jobs=None):
    """
    run_cv_svm(inp, target)
    
    Returns scores from running a cross-validation SVM on the input and target
    data.

    Required args:
        - inp (array-like)   : input array whose first dimension matches the 
                               target first dimension 
        - target (array-like): 1D target array

    Optional args:
        - cv (int)          : number of cross-validation folds (at least 3)
                              (stratified KFold)
                              default: 5
        - shuffle (bool)    : if True, target is shuffled
                              default: False
        - stats (str)       : statistic to return across fold scores 
                              ('mean' or 'median')  If None, all scores are 
                              returned
                              default: 'mean'
        - error (str)       : error statistic to return across fold scores. If 
                              None or if `stats` is None, no error statistic is 
                              returned.('std' for std or q1-3 and 'sem' for 
                              SEM or MAD, depending on the value or `stats`)
                              default: 'std'
        - class_weight (str): sklearn class_weight attribute
                              default: 'balanced'
        - n_jobs (int)      : number of CPUs to use (see sklearn)
                              default: None

    Returns:
        if stats is None and error is None:
        - sc (1D array): scores for each fold (accuracy or balanced accuracy if 
                         class_weight is 'balanced')
        elif only error is None:
        - me (float)   : mean/median statistic across fold scores
        else:
        - me (float)   : mean/median statistic across fold scores
        - err (float)  : std/SEM/q1-3/MAD across fold scores
    """

    clf = svm.SVC(kernel='poly', C=1, gamma='auto', class_weight=class_weight)                    
    
    # first dim must be trials
    if shuffle:
        np.random.shuffle(target)
    
    if cv < 3:
        raise ValueError('`cv` must be at least 3.')

    scoring = None
    if class_weight == 'balanced':
        scoring = 'balanced_accuracy'

    sc = cross_val_score(clf, inp, target, cv=cv, scoring=scoring, 
                         n_jobs=n_jobs)
    
    if stats is None:
        return sc
    else:
        me = mean_med(sc, stats=stats)
        if error is None:
            return me
        else:
            err = error_stat(sc, stats=stats, error=error)
            return me, err

