"""
math_util.py

This module contains basic math functions.

Authors: Colleen Gillon

Date: February 2023

Note: this code was aggregated from https://github.com/colleenjg/util.

"""

import numpy as np
import scipy.stats

from util import gen_util



############################################
def mean_med(data, stats="mean", axis=None, nanpol=None):
    """
    mean_med(data)

    Returns the mean or median of the data along a specified axis, depending on
    which statistic is requested.

    Required args:
        - data (nd array): data on which to calculate statistic

    Optional args:
        - stats (str) : "mean" or "median"
                        default: "mean"
        - axis (int)  : axis along which to take the statistic
                        default: None
        - nanpol (str): policy for NaNs, "omit" or None
                        default: None
    
    Returns:
        - me (nd array or num): mean or median of data along specified axis
    """

    if stats == "mean":
        if nanpol is None:
            me = np.mean(data, axis=axis)
        elif nanpol == "omit":
            me = np.nanmean(data, axis=axis)
    elif stats == "median":
        if nanpol is None:
            me = np.median(data, axis=axis)
        elif nanpol == "omit":
            me = np.nanmedian(data, axis=axis)
    else:
        gen_util.accepted_values_error("stats", stats, ["mean", "median"])
    if nanpol is not None and nanpol != "omit":
        gen_util.accepted_values_error("nanpol", nanpol, ["None", "omit"])

    return me
    

#############################################
def error_stat(data, stats="mean", error="sem", axis=None, nanpol=None, 
               qu=[25, 75]):
    """
    error_stat(data)

    Returns the std, SEM, quartiles or median absolute deviation (MAD) of data 
    along a specified axis, depending on which statistic is requested.

    Required args:
        - data (nd array): data on which to calculate statistic

    Optional args:
        - stats (str) : "mean" or "median"
                        default: "mean"
        - error (str) : "std" (for std or quintiles) or "sem" (for SEM or MAD) 
                        or "var" (for variance)
                        default: "sem"
        - axis (int)  : axis along which to take the statistic
                        default: None
        - nanpol (str): policy for NaNs, "omit" or None
                        default: None
        - qu (list)   : quintiles to take, if median and std along which 
                        to take the statistic
                        default: [25, 75]
    
    Returns:
        - error (nd array or num): std, SEM, quintiles or MAD of data along 
                                   specified axis
    """

    if stats == "mean" and error == "std":
        if nanpol is None:
            error = np.std(data, axis=axis)
        elif nanpol == "omit":
            error = np.nanstd(data, axis=axis)
    elif stats == "mean" and error == "sem":
        if nanpol is None:
            error = scipy.stats.sem(data, axis=axis)
        elif nanpol == "omit":
            error = scipy.stats.sem(data, axis=axis, nan_policy="omit")
    elif stats == "mean" and error == "var":
        if nanpol is None:
            error = np.var(data, axis=axis)
        elif nanpol == "omit":
            error = np.nanvar(data, axis=axis)
    elif stats == "median" and error == "std":
        if nanpol is None:
            error = [np.percentile(data, qu[0], axis=axis), 
                np.percentile(data, qu[1], axis=axis)]
        elif nanpol == "omit":
            error = [np.nanpercentile(data, qu[0], axis=axis), 
                np.nanpercentile(data, qu[1], axis=axis)]
        
    elif stats == "median" and error == "sem":
        # MAD: median(abs(x - median(x)))
        if axis is not None:
            me_shape       = list(data.shape)
            me_shape[axis] = 1
        else:
            me_shape = -1
        if nanpol is None:
            me    = np.asarray(np.median(data, axis=axis)).reshape(me_shape)
            error = np.median(np.absolute(data - me), axis=axis)
        elif nanpol == "omit":
            me    = np.asarray(np.nanmedian(data, axis=axis)).reshape(me_shape)
            error = np.nanmedian(np.absolute(data - me), axis=axis)

    elif stats == "median" and error == "var":
        raise NotImplementedError(
            "No robust equivalent for 'variance' is implemented."
            )

    elif stats != "median" and stats != "mean":
        gen_util.accepted_values_error("stats", stats, ["mean", "median"])
    else:
        gen_util.accepted_values_error("error", error, ["std", "sem", "var"])
    if nanpol is not None and nanpol != "omit":
        gen_util.accepted_values_error("nanpol", nanpol, ["[None]", "omit"])

    error = np.asarray(error)
    if len(error.shape) == 0:
        error = error.item()

    return error

  
#############################################
def get_stats(data, stats="mean", error="sem", axis=-1, nanpol=None,
              qu=[25, 75]):
    """
    get_stats(data)
    
    Returns statistics (me, error x values) statistics as a single array.
    Note that if stats="median" and error="std", the error will be in two 
    rows/cols.
    
    Required args:
        - data (nd array): data array (at least 2D)

    Optional args:
        - stats (str):  stats to take, i.e., "mean" or "median"
                        default: "mean"
        - error (str):  error to take, i.e., "std" (for std or quintiles) 
                        or "sem" (for SEM or MAD) or "var" (for variance)
                        default: "std"
        - axes (int) :  axes along which to  take statistics. If a list  
                        is passed.
                        default: -1
        - nanpol (str): policy for NaNs, "omit" or None
                        default: None
        - qu (list)   : quintiles to take, if median and std along which 
                        to take the statistic
                        default: [25, 75]

    Returns:
        - data_stats (nd array): stats array, structured as: 
                                 stat type (me, error x values) x 
                                     remaining_dims
    """

    data = np.asarray(data)

    if data.shape == 1:
        raise ValueError("Data array must comprise at least 2 dimensions.")

        
    # mean/med along units axis (last)
    me  = mean_med(data, stats=stats, axis=axis, nanpol=nanpol) 
    err = error_stat(
        data, stats=stats, error=error, axis=axis, nanpol=nanpol, qu=qu)
    
    # ensures that these are arrays
    me = np.asarray(me)
    err = np.asarray(err)

    if stats == "median" and error == "std":
        me = np.expand_dims(me, 0)
        data_stats = np.concatenate([me, err], axis=0)
    else:
        data_stats = np.stack([me, err])

    return data_stats


#############################################
def rolling_mean(vals, win=3):
    """
    rolling_mean(vals)

    Returns rolling mean over the last dimension of the input data.

    NaNs/Infs will propagate.

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
    vals_out = scipy.ndimage.convolve(vals, weights, mode="mirror")

    return vals_out


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
def get_order_of_mag(val):
    """
    get_order_of_mag(val)
    
    Returns order of magnitude for a value.

    Required args:
        - val (float): value to round
    
    Returns:
        - order (int): order of magnitude for rounding value
    """

    if val == 0:
        return 0

    order = int(np.floor(np.log10(np.absolute(val))))

    return order


#############################################
def round_by_order_of_mag(val, n_sig=1, direc="any", decimal_only=False):
    """
    round_by_order_of_mag(val)
    
    Returns value, rounded by the order of magnitude.

    Required args:
        - val (float): value to round
    
    Optional args:
        - n_sig (int)        : number of significant digits
                               default: 1
        - direc (str)        : direction in which to round value ("up", "down")
                               default: "any"
        - decimal_only (bool): if True, only decimals are rounded
                               default: False

    Returns:
        - rounded_val (float): rounded value
    """
    
    if n_sig < 1:
        raise ValueError("'n_sig' must be at least 1.")

    o = int(-get_order_of_mag(val) + n_sig - 1)

    if decimal_only and o < 0:
        o = 0

    if direc == "any":
        rounded_val = np.around(val, o)
    elif direc == "up":
        rounded_val = np.ceil(val * 10**o) / 10**o
    elif direc == "down":
        rounded_val = np.floor(val * 10**o) / 10**o
    else:
        gen_util.accepted_values_error("direc", direc, ["any", "up", "down"])

    return rounded_val


#############################################
def get_near_square_divisors(val):
    """
    get_near_square_divisors(val)

    Returns near-square divisors of a number.

    Required args:
        - val (int): value for which to get divisors
    
    Returns:
        - divs (list): list of divisor values in order [high, low]
    """

    if int(val) != float(val):
        raise TypeError("'val' must be an int.")

    i = int(np.max([np.floor(np.sqrt(val)), 1]))
    j = int(np.ceil(val / i))

    divs = [i, j]
    if j > i:
        divs = divs[::-1]

    return divs


#############################################
def calculate_snr(data, return_stats=False):
    """
    calculate_snr(data)
    
    Returns SNR for data (std of estimated noise / mean of signal).

    Required args:
        - data (1D array): data for which to calculate SNR (flattened if not 1D)

    Optional args:
        - return_stats (bool): if True, additional stats are returned
                               default: False

    Returns:
        - snr (float): SNR of data
        if return_stats:
        - data_median (float)  : median of full data
        - noise_data (1D array): noisy data
        - noise_mean (float)   : mean of the noise 
        - noise_std (float)    : standard deviation of the noise
        - noise_thr (float)    : noise threshold
        - signal_mean (float)  : mean of the signal
    """
    
    data = np.asarray(data).reshape(-1)
    data_median = np.median(data)

    lower_vals = np.where(data <= data_median)[0]
    noise_data = np.concatenate(
        [data[lower_vals], 2 * data_median - data[lower_vals]]
        )

    noise_mean = np.mean(noise_data)
    noise_std = np.std(noise_data)
    noise_thr = scipy.stats.norm.ppf(0.95, noise_mean, noise_std)
    signal_mean = np.mean(data[np.where(data > noise_thr)])

    snr = signal_mean / noise_std
    
    if return_stats:
        return [snr, data_median, noise_data, noise_mean, noise_std, 
            noise_thr, signal_mean]
    else:
        return snr