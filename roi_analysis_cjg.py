import os
import datetime

import numpy as np
import pickle
import scipy.stats as st
import pandas as pd
from matplotlib import pyplot as plt
import pdb

from analysis import session
import util.file_util


#############################################
def mean_med(data, stats='mean', axis=None):
    """
    mean_med(data)

    Returns the mean or median of the data along a specified axis, depending on
    which statistic is needed.

    Required arguments:
        - data (np.array): data on which to calculate statistic

    Optional arguments:
        - stats (str): 'mean' or 'median'
                       default: 'mean'
        - axis (int) : axis along which to take the statistic
                       default: None
    
    Returns:
        - (nd array): mean or median of data along specified axis
    """

    if stats == 'mean':
        return np.mean(data, axis=axis)
    elif stats == 'median':
        return np.median(data, axis=axis)
    else:
        accepted_values_error('stats', stats, ['mean', 'median'])

#############################################
def error_stat(data, stats='mean', error='std', axis=None):
    """
    error_stat(data)

    Returns the std, SEM, quartiles or median absolute deviation (MAD) of data 
    along a specified axis, depending on which statistic is needed.

    Required arguments:
        - data (np.array): data on which to calculate statistic

    Optional arguments:
        - stats (str): 'mean' or 'median'
                       default: 'mean'
        - error (str): 'std' (for std or quartiles) or 'sem' (for SEM or MAD)
                       default: 'std'
        - axis (int) : axis along which to take the statistic
                       default: None
    
    Returns:
        - (nd array): std, SEM, quartiles or MAD of data along specified axis
    """

    if stats == 'mean' and error == 'std':
        return np.std(data, axis=axis)
    elif stats == 'mean' and error == 'sem':
        return st.sem(data, axis=axis)
    elif stats == 'median' and error == 'std':
        return [np.percentile(data, 25, axis=axis), 
                np.percentile(data, 75, axis=axis)]
    elif stats == 'median' and error == 'sem':
        # MAD: median(abs(x - median(x)))
        return np.median(np.absolute(data - np.median(data, axis=None)), axis=None)
    elif stats != 'median' and stats != 'mean':
        accepted_values_error('stats', stats, ['mean', 'median'])
    else:
        accepted_values_error('error', error, ['std', 'sem'])


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
    nosurp_ind = [slice(None)] * surp_dim + [0]
    surp_ind = [slice(None)] * surp_dim + [1]

    if vals == 'diff':
        if op == 'diff':
            data = (data[surp_ind] - data[nosurp_ind])
        elif op == 'ratio':
            data = (data[surp_ind]/data[nosurp_ind])
        else:
            accepted_values_error('op', op, ['diff', 'ratio'])
    elif vals == 'nosurp':
        data = data[nosurp_ind]
    elif vals == 'surp':
        data = data[surp_ind]
    else:
        accepted_values_error('op', op, ['diff', 'surp', 'nosurp'])
    data = data.squeeze(surp_dim)
    
    return data


#############################################
def integ_dff(data, dx, axis=None):
    """
    integ_dff(data, dx)

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
def stat_par_str(stats='mean', error='std'):
    """
    stat_par_str(stats, error)

    Creates a string from statistical analysis parameters to print or for a 
    title.

    Optional arguments:
        - stats (str): 'mean' or 'median'
                       default: 'mean'
        - error (str): 'std' (for std or quartiles) or 'sem' (for SEM or MAD)
                       default: 'std'
    
    Returns:
        - stat_str (str): statistics combo string
    """

    if stats == 'mean':
        stat_str = '{}/{}'.format(stats, error)
    elif stats == 'median' and error == 'std':
        stat_str = '{}/qu'.format(stats)
    elif stats == 'median' and error == 'sem':
        stat_str = '{}/mad'.format(stats)
    return stat_str

#############################################
def plot_val_lab(plot_vals='diff', op='diff'):
    """
    plot_val_lab()

    Creates a list of labels for gabor frames based on values that are plotted,
    and operation on surprise v no surprise.

    Optional arguments:
        - plot_vals (str): 'surp', 'nosurp' or 'diff'
                           default: 'diff'
        - op (str)       : 'surp', 'nosurp' or 'diff'
                           default: 'diff'
    
    Returns:
        - labels (list)  : list of labels for gabor frames
    """
    if plot_vals == 'surp':
        labels = ['E']
    elif plot_vals == 'nosurp':
        labels = ['D']
    elif plot_vals == 'diff':
        if op == 'diff':
            labels = ['E-D']       
        elif op == 'ratio':
            labels = ['E/D']
        else:
            accepted_values_error('op', op, ['diff', 'ratio'])
    else:
        accepted_values_error('plot_vals', plot_vals, ['diff', 'surp', 'nosurp'])
    
    labels.extend(['gray', 'A', 'B', 'C'])

    return labels


#############################################
def op_par_str(plot_vals='diff', op='diff', area=False, str_type='file'):
    """
    op_par_str()

    Creates a string from operation parameter (e.g., surprise, non surprise 
    or the difference/ratio between the two) to print or for a title.

    Optional arguments:
        - plot_vals (str): 'surp', 'nosurp' or 'diff'
                           default: 'diff'
        - op (str)       : 'diff' or 'ratio'
                           default: 'diff'
        - str_type (str) : use of output str, i.e., for a filename ('file') or
                           to print the info to console ('print')
                           default = 'file'
    
    Returns:
        - op_str (str): operation type string
    """
    if area:
        area_str = ' area'
    else:
        area_str = ''
    if plot_vals == 'diff':
        if str_type == 'print':
            op_str = '{} in dF/F{} for surp v nosurp'.format(op, area_str)
        elif str_type == 'file':
            op_str = plot_vals
    elif plot_vals in ['surp', 'nosurp']:
        if str_type == 'print':
            op_str = 'dF/F{} for {}'.format(plot_vals, area_str)
        elif str_type == 'file':
            op_str = plot_vals
    else:
        accepted_values_error('plot_vals', plot_vals, ['diff', 'surp', 'nosurp'])
    return op_str


#############################################
def create_time_str():
    """
    create_time_str()

    Creates a string in a format appropriate for a directory or filename
    containing date and time information based on time at which the function is
    called.

    Return:
        dir_name (str): string containing date and time formatted as 
                        YYMMDD_HHMMSS
    """

    now = datetime.datetime.now()
    dir_name = ('{:02d}{:02d}{:02d}_'
                '{:02d}{:02d}{:02d}').format(now.year, now.month, now.day, 
                                             now.hour, now.minute, now.second)
    return dir_name


#############################################
def sess_par_str(sess_par, gab_k, str_type='file'):
    """
    sess_par_str(sess_par, gab_k)

    Creates a string from session and gabor kappa parameters for a filename, 
    or to print or use in a title.

    Required arguments:
        - sess_par (dict)     : dictionary containing session parameters: 
                                e.g. 'layer', 'overall_sess_n'
                ['gab_k'] (int or list) : gabor kappa parameter
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                                'L5_soma', 'L23_dend', 'L5_dend', 
                                                'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for

    Optional arguments:
        - str_type (str): use of output str, i.e., for a filename ('file') or to
                          print the info to console ('print')
                          default = 'file'
    Return:
        - sess_str (list): string containing info on session and gabor kappa 
                           parameters
    """

    gab_k_str = gab_k_par_str(gab_k)
    if str_type == 'file':
        sess_str = 'sess{}_gab{}_{}'.format(sess_par['overall_sess_n'], gab_k_str,
                                             sess_par['layer'])
    elif str_type == 'print':
        if len(gab_k_str) == 0:
            sess_str = ('gabors, ')
        else:
            sess_str = 'gabors: {}, '.format(gab_k_str)
        sess_str += 'session: {}, layer: {}'.format(sess_par['overall_sess_n'], 
                                                     sess_par['layer'])
    else:
        accepted_values_error('str_type', str_type, ['file', 'print'])
    return sess_str
    

#############################################
def gab_k_par_str(gab_k):
    """
    gab_k_par_str(gab_k)

    Creates a string from gabor kappa parameter (e.g., '4' or '16'). Returns
    an empty string if both gabor kappa parameters are passed.

    Required arguments:
        - gab_k (int or list): gabor kappa parameter

    Return:
        (list): string containing gabor kappa parameter value or empty string
                if both parameters are passed.
    """

    gab_k = list_if_not(gab_k)
    if len(gab_k) > 1:
        return ''
    else:
        return str(gab_k[0])

    
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
def init_fig(n_subplots, fig_par):
    """
    init_fig(n_subplots, fig_par)

    Creates a figure with the correct number of rows and columns for the 
    number of subplots, following the figure parameters

    Required arguments:
        - n_subplots (int): number of subplots to accomodate in the figure
        - fig_par (dict)  : dictionary containing figure parameters:
                ['ncols'] (int)        : number of columns in the figure
                ['sharey'] (bool)      : if True, y axis lims are shared across 
                                         subplots
                ['subplot_wid'] (float): width of each subplot (inches)
                ['subplot_hei'] (float): height of each subplot (inches)

    Return:
        - fig (plt fig): pyplot figure
        - ax (plt ax)  : pyplot axes
    """

    if n_subplots == 1:
        fig_par['ncols'] = 1
    elif n_subplots < fig_par['ncols']:
        fig_par['ncols'] = n_subplots
    ncols = fig_par['ncols']
    nrows = int(np.ceil(n_subplots/float(fig_par['ncols'])))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, 
                           figsize=(ncols*fig_par['subplot_wid'], 
                                    nrows*fig_par['subplot_hei']), 
                           sharey=fig_par['sharey'])
    return fig, ax

#############################################
def save_fig(fig, save_dir, save_name, fig_par):
    """
    save_fig(fig, save_dir, save_name, fig_par)

    Saves a figure under a specific directory and name, following figure
    parameters and returns final directory name.

    Required arguments:
        - fig (plt fig)  : pyplot figure
        - save_dir (str) : directory in which to save figure
        - save_name (str): name under which to save figure (WITHOUT extension)
        - fig_par (dict) : dictionary containing figure parameters:
                ['bbox'] (str)      : bbox_inches parameter for plt.savefig(),
                                      e.g., 'tight' 
                ['datetime'] (bool) : if True, figures are saved in a subfolder
                                      named based on the date and time.
                ['fig_ext'] (str)   : extension (without '.') with which to save
                                      figure
                ['overwrite'] (bool): if False, overwriting existing figures is 
                                      prevented by adding suffix numbers.
                ['prev_dt'] (str)   : datetime folder to use
    Returns:
        - save_dir (str): final name of the directory in which the figure is 
                          saved 
                          (may be different from input save_dir, as a datetime 
                          subfolder, or a suffix to prevent overwriting may have 
                          been added depending on the parameters in fig_par.)
    """

    # add subfolder with date and time
    if fig_par['datetime']:
        if fig_par['prev_dt'] is not None:
            save_dir = os.path.join(save_dir, fig_par['prev_dt'])
        else:
            datetime = create_time_str()
            save_dir = os.path.join(save_dir, datetime)
            fig_par['prev_dt'] = datetime


    # check if it exists, and if so, add number at end
    if not fig_par['overwrite']:
        if fig_par['datetime'] and fig_par['prev_dt'] is None:
            if os.path.exists(save_dir):     
                count = 1
                while os.path.exists('{}_{}'.format(save_dir, count)):
                    count += 1 
                save_dir = '{}_{}'.format(save_dir, count)
            os.makedirs(save_dir)
    
    # create directory if doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('Figures saved under {}.'.format(save_dir))

    full_save_name = '{}.{}'.format(save_name, fig_par['fig_ext'])

    fig.savefig('{}/{}'.format(save_dir, full_save_name), 
                bbox_inches=fig_par['bbox'])
    
    return save_dir


#############################################
def save_info(info_dict, full_dir, save_name='info'):
    """
    save_info(dict, full_dir)

    Pickles and saves dictionary under a specific directory and optional name.

    Required arguments:
        - info_dict (dict): dictionary to pickle and save
        - full_dir (str)  : directory in which to save pickle
        - save_name (str) : name under which to save info (WITHOUT extension)
    """

    full_name = os.path.join(full_dir, '{}.pkl'.format(save_name))

    # create directory if it doesn't exist
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    
    if isinstance(info_dict, dict):
        with open(full_name, 'w') as f:
            pickle.dump(info_dict, f)
    else:
        raise TypeError(('\'info_dict\' must be a dictionary, but is a {}.')
                        .format(type(info_dict)))


#############################################
def label_values(mouse_df, label, values='any'):
    """
    label_values(mouse_df, label)

    Either returns the specified value(s) for a specific label of a pandas 
    dataframe as a list, or if values='any', collects all different values for 
    that label and returns them in a list.

    Required arguments:
        - mouse_df (pandas df): dataframe
        - label (str)         : label of the dataframe column of interest

    Optional arguments:
        - values (str, list or item): values to return
                                      default='any'
    Return:
        - vals (list): values
    """

    if values == 'any':
        vals = mouse_df[label].unique().tolist()
    else:
        vals = list_if_not(values)
    return vals


#############################################
def depth_values(layer):
    """
    depth_values(layer)

    Returns depth values corresponding to a layer.

    Required arguments:
        - layer (str): layer (e.g., 'dend', 'soma', 'L23_all', 'L5_all', 
                                    'L23_dend', 'L23_soma', 'L5_dend', 
                                    'L5_soma', 'any')
    Return:
        - (list): depths corresponding to layer or 'any'
    """

    if layer == 'any':
        return 'any'
    layer_dict = {'L23_dend': 20,
                  'L23_soma': 175,
                  'L5_dend' : 75,
                  'L5_soma' : 375
                 }

    if layer in ['L23_dend', 'L23_soma', 'L5_dend', 'L5_soma']:
        pass
    elif layer == 'dend':
        layer_dict['dend'] = [layer_dict['L23_dend'], layer_dict['L5_dend']]
    elif layer == 'soma':
        layer_dict['soma'] = [layer_dict['L23_soma'], layer_dict['L5_soma']]
    elif layer == 'L23_all':
        layer_dict['L23_all'] = [layer_dict['L23_soma'], layer_dict['L23_soma']]
    elif layer == 'L5_all':
        layer_dict['L5_all'] = [layer_dict['L5_soma'], layer_dict['L5_soma']]
    else:
        accepted_values_error('layer', layer, ['L23_dend', 'L23_soma', 'L5_dend', 
                                        'L5_soma', 'dend', 'soma', 'L23_all', 
                                        'L5_all', 'any'])

    return layer_dict[layer]


#############################################
def sess_values(mouse_df, returnlab, mouseid, sessid, depth, pass_fail, 
                  all_files, any_files, overall_sess_n, min_rois, sort=False):
    """
    sess_values(mouse_df, returnlab, mouseid, sessid, depth, pass_fail, 
                all_files, any_files, overall_sess_n, min_rois)

    Returns values from dataframe under a specified label that fit the 
    criteria.

    Required arguments:
        - mouse_df (pandas df)       : dataframe containing parameters for each 
                                       session.
        - returnlab ('str')          : label from which to return values
        - mouseid (int or list)      : mouse id value(s) of interest         
        - sessid (int or list)       : session id value(s) of interest
        - depth (str or list)        : depth value(s) of interest (20, 75, 175, 
                                       375)
        - pass_fail (str or list)    : pass/fail values of interest ('P', 'F')
        - all_files (int or list)    : all_files values of interest (0, 1)
        - any_files (int or list)    : any_files values of interest (0, 1)
        - overall_sess_n (int or str): overall_sess_n values of interest
        - min_rois (int)             : min number of ROIs
    
    Optional arguments:
        - sort (bool): whether to sort output values
                       default: False
                                    
    Return:
        - sess_vals (list): list of values under the specified label that fit
                            the criteria
    """

    sess_vals = mouse_df.loc[(mouse_df['mouseid'] == mouseid) & 
                            (mouse_df['sessionid'].isin(sessid)) &
                            (mouse_df['depth'].isin(depth)) &
                            (mouse_df['pass_fail'].isin(pass_fail)) &
                            (mouse_df['all_files'].isin(all_files)) &
                            (mouse_df['any_files'].isin(any_files)) &
                            (mouse_df['overall_sess_n'].isin(overall_sess_n)) &
                            (mouse_df['n_rois'] >= min_rois)][returnlab].tolist()
    if sort:
        sess_vals = sorted(sess_vals)
    return sess_vals


#############################################
def sess_per_mouse(mouse_df, sessid='any', layer='any', pass_fail='any', 
                   all_files='any', any_files='any',overall_sess_n=1, 
                   omit_sess=[], omit_mice=[], min_rois=1):
    """
    Returns list of session IDs (up to 1 per mouse) that fit the specified
    criteria, IDs of mice for which a session was found and actual overall 
    session numbers.

    Required arguments:
        - mouse_df (pandas df): dataframe containing parameters for each session.
        
    Optional arguments:
        - sessid (int or list)       : session id value(s) of interest
                                       (default: 'any')
        - layer (str or list)        : layer value(s) of interest
                                       ('soma', 'dend', 'L5', 'L23', etc.)
                                       (default: 'any')
        - pass_fail (str or list)    : pass/fail values to pick from 
                                       ('P', 'F')
                                       (default: 'any')
        - all_files (int or list)    : all_files values to pick from (0, 1)
                                       (default: 'any')
        - any_files (int or list)    : any_files values to pick from (0, 1)
                                       (default: 'any')
        - overall_sess_n (int or str): overall_sess_n value to aim for
                                       (1, 2, 3, ... or 'last')
                                       (default: 1)
        - sess_omit (list)           : sessions to omit
                                       (default: [])
        - mice_omit (list)           : mice to omit
                                       (default: [])
        - min_rois (int)             : min number of ROIs
                                       (default: 1)
     
    Returns:
        - sess_ns (list)    : sessions to analyse (1 per mouse)
        - mouse_ns (list)   : mouse numbers corresponding to sessions
        - act_sess_ns (list): actual overall session number for each mouse
    """

    # get depth values corresponding to the layer
    depth = depth_values(sess_par['layer'])

    params = [sessid, depth, pass_fail, all_files, any_files]
    param_names = ['sessionid', 'depth', 'pass_fail', 'all_files', 'any_files']
    # for each label, collect values of that fit criteria in a list
    for i in range(len(params)):
        params[i] = label_values(mouse_df, param_names[i], params[i])
    [sessid, depth, pass_fail, all_files, any_files] = params
    overall_sess_any = label_values(mouse_df, 'overall_sess_n', 'any')
    # remove omitted sessions from the session id list
    sessid = remove_if(sessid, omit_sess)
    # collect all mouse IDs and remove omitted mice
    mouseids = remove_if(sorted(label_values(mouse_df, 'mouseid', values='any')), omit_mice)

    # get session ID, mouse ID and actual session numbers for each mouse based 
    # on criteria 
    sess_ns = []
    mouse_ns = []
    act_sess_ns = []
    for i in mouseids:
        sessions = sess_values(mouse_df, 'overall_sess_n', i, sessid, depth, 
                               pass_fail, all_files, any_files, overall_sess_any, 
                               min_rois, sort=True)
        # skip mouse if no sessions meet criteria
        if len(sessions) == 0:
            continue
        elif overall_sess_n == 'last':
            sess_n = sessions[-1]
        # find closest sess number among possible sessions
        else:
            sess_n = sessions[np.argmin(np.absolute([x-overall_sess_n
                                                     for x in sessions]))]
        sess = sess_values(mouse_df, 'sessionid', i, sessid, depth, pass_fail, 
                           all_files, any_files, [sess_n], min_rois)[0]
        act_n = mouse_df.loc[(mouse_df['sessionid'] == 
                              sess)]['overall_sess_n'].tolist()[0]

        sess_ns.append(sess)
        mouse_ns.append(i)
        act_sess_ns.extend([act_n])
    
    return sess_ns, mouse_ns, act_sess_ns


#############################################
def init_sessions(sess_ns):
    """
    init_sess_dict(sess_ns)

    Creates list of Session objects for each session ID passed 

    Required arguments:
        - sess_ns (int or list): ID or list of IDs of sessions
                                    
    Returns:
        - sessions (list): list of Session objects
    """

    sessions = []
    sess_ns = list_if_not(sess_ns)
    for sess_n in sess_ns:
        print('\nCreating session {}...'.format(sess_n))
        sess = session.Session(maindir, sess_n) # creates a session object to work with
        sess.extract_info()                     # extracts necessary info for analysis
        print('Finished session {}.'.format(sess_n))
        sessions.append(sess)
    return sessions


#############################################
def gab_mice_omit(gab_k):
    """
    gab_mice_omit(gab_k)

    Returns IDs of mice to omit based on gabor kappa values to include.

    Required arguments:
        - gab_k (int or list): gabor kappa values
                                    
    Return:
        - omit_mice (list): list IDs of mice to omit
    """
    gab_k = list_if_not(gab_k)
    if 4 not in gab_k:
        omit_mice = [1] # mouse 1 only got K=4
    elif 16 not in gab_k:
        omit_mice = [3] # mouse 3 only got K=16
    else: 
        omit_mice = []
    return omit_mice


#############################################
def quint_par(gabors, analys_par):
    """
    quint_par(gabors, analys_par)

    Returns dictionary containing parameters for breaking segments into 
    quintiles.
    
    Required arguments:
        - gabors (Gabor object): gabors object
        - analys_par (dict)    : dictionary containing relevant parameters
                                 to extracting segment numbers and dividing
                                 them into quintiles
                ['gab_k'] (int or list) : gabor kappa values to include 
                                          (e.g., 4, 16 or [4, 16])
                ['gab_fr'] (int or list): gabor frame values to include
                                         (e.g., 0, 1, 2, 3)
                ['n_quints'] (int)      : number of quintiles

    Return:
        - qu_info (dict): dictionary containing parameters for breaking 
                          segments into quintiles
                ['seg_min'] (int): minimum segment number
                ['seg_max'] (int): maximum segment number
                ['len'] (float)  : length of each quintile in seg numbers
                                   (can be a decimal)
    """

    # get all seg values
    all_segs = gabors.get_segs_by_criteria(stimPar2=analys_par['gab_k'], 
                                           gaborframe=analys_par['gab_fr'], 
                                           by='seg')
    # get the min and max seg numbers for stimulus
    qu_info = {'seg_min': min(all_segs),
               'seg_max': max(all_segs)+1
               }
    # check for breaks in stimulus presentation
    diff     = np.diff(all_segs)
    min_diff = np.min(np.diff(all_segs))

    # identify larger intervals
    high_interv = np.where(diff > min_diff)[0].tolist()
    if len(high_interv) != 0:
        high_interv_str = ', '.join(['{} ({})'.format(diff[x], all_segs[x]) 
                                     for x in high_interv])
        print('The list of segments retrieved contains breaks. \n    Expected '
              'seg intervals: {}. \n    Actual seg intervals, i.e. interval, '
              '(preceeding seg n): {}'.format(min_diff, high_interv_str))
        raise NotImplementedError(('quint_par() should not be used to '
                                   'determine length of quintiles if the '
                                   'segments values have breaks in them. '
                                   'Instead, retrieve quintile values for '
                                   'Each set of segments separately.'))

    # calculate number of segments in each quintile (ok if not round number)
    qu_info['len'] = (qu_info['seg_max'] - 
                      qu_info['seg_min'])/float(analys_par['n_quints'])
    return qu_info


#############################################
def quint_segs(gabors, analys_par, qu_info, surp='any'):
    """
    quint_segs(gabors, analys_par)

    Returns dictionary containing parameters for breaking segments into 
    quintiles.
    
    Required arguments:
        - gabors (Gabor object): gabors object
        - analys_par (dict)    : dictionary containing relevant parameters
                                 to extracting segment numbers and dividing
                                 them into quintiles
                ['gab_fr'] (int or list): gabor frame values to include
                                         (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list) : gabor kappa values to include 
                                          (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)      : number of quintiles
        - quint_info (dict)    : dictionary containing parameters for breaking 
                                 segments into quintiles
                ['seg_min'] (int): minimum segment number
                ['seg_max'] (int): maximum segment number
                ['len'] (float)  : length of each quintile in seg numbers
                                   (can be a decimal)
    Optional arguments:
        - surp (int or list)     : surprise values to include (e.g., 0 or 1)
                                   default: 'any'
    Returns:
        - qu_segs (list) : list of sublists for each quintile, each containing 
                           segment numbers for that quintile
        - qu_count (list): list of number of segments in each quintile
    """

    # get all seg values
    all_segs = gabors.get_segs_by_criteria(stimPar2=analys_par['gab_k'], surp=surp, 
                                           gaborframe=analys_par['gab_fr'], by='seg')
    # get seg ranges for each quintile [[start, end], [start, end], etc.] 
    qu_segs = []
    qu_count = []
    for i in range(analys_par['n_quints']):
        qu_segs.append([seg for seg in all_segs 
                           if (seg>=i*qu_info['len']+qu_info['seg_min'] and 
                               seg < (i+1)*qu_info['len']+qu_info['seg_min'])])
        qu_count.extend([len(qu_segs[i])])
    return qu_segs, qu_count


#############################################
def chunk_stats_by_qu(gabors, qu_seg, pre, post, byroi=True, dfoverf=True, 
                      remnans='per', rand=False, stats='mean', error='std',
                      data='all'):
    """
    chunk_stats_by_qu(gabors, qu_seg, pre, post)

    Returns chunk statistics for the quintiles of interest. 

    Required arguments:
        - gabors (Gabor object): gabors object
        - qu_seg (dict)        : list of sublists for each quintile, each 
                                 containing segment numbers for that quintile
        - pre (float)          : range of frames to include before each frame 
                                 reference (in s)
        - post (float)         : range of frames to include after each frame 
                                 reference (in s)
    
    Optional arguments:
        - byroi (bool)  : if True, returns statistics for each ROI. If False,
                          returns statistics across ROIs.
                          default: True 
        - dfoverf (bool): if True, dF/F is used instead of raw ROI traces.
                          default: True
        - remnans (str)  : if 'per', removes ROIs with NaN/Inf values, for each
                          subdivision (quintile/surprise). If 'across', removes
                          ROIs with NaN/Inf values across subdivisions. If 'no',
                          ROIs with NaN/Inf values are not remomved.
                          default: 'per'
        - rand (bool)   : if True, also return statistics for a random 
                          permutation of the traces (not implemented).
                          default: False
        - stats (str)   : if 'mean', mean is used. If 'median', median
                          is used.
                          default: 'mean'
        - error (str)   : if 'std', std or [25th and 75th quartiles] are used.
                          If 'sem', SEM or median absolute deviation (MAD) are
                          used.
                          default: 'std'
        - data (str)    : if 'all', full statistics are returned (e.g., mean + 
                          std). If 'me', only mean/median is returned.
                          default: 'all'

    Returns:
        - x_ran (1D array)           : array of time values for the frame 
                                       chunks
        - qu_stats (2, 3 or 4D array): nd array of statistics for chunks with
                                       dimensions:
                                            quintiles x
                                            (ROIs if byroi x)
                                            (statistic if data == 'all' x)
                                            frames
    Optional returns (if remnans in ['per', 'across']):
            (list) containing:
                - nan_rois (list): if remnans is 'per', list of sublists for 
                                    each quintile, each containing numbers of 
                                    ROIs removed for containing NaN/Infs. If
                                    remnans == 'across', list of ROIs 
                                    with NaN/Infs across quintiles.
                                    
                - ok_rois (list) : if remnans is 'per', list of sublists for 
                                    each quintile, each containing numbers of 
                                    ROIs without NaN/Infs. If remnans is 
                                    'across', list of ROIs without NaN/Infs 
                                    across quintiles.    
    """

    if rand:
        raise NotImplementedError(('Retrieving stats for random data using '
                                   '\'chunk_stats_by_qu()\' not implemented.'))

    if remnans in ['per', 'across']:
        nan_rois = []
        ok_rois = []
        if remnans == 'across' and byroi:
            nans = 'list' 
        else:
            nans = 'rem'
            if not byroi:
                print(('WARNING: Removing ROIs with NaNs and Infs across '
                        'quintiles and other subdivisions when averaging across '
                        'ROIs not implemented yet. Removing per subdivision '
                        'instead.'))
                remnans = 'per'
    elif remnans == 'no':
        nans = 'no'
    else:
        accepted_values_error('remnans', remnans, ['per', 'across', 'no'])
    
    qu_stats = []
    for qu, segs in enumerate(qu_seg):
        print('\tQuintile {}'.format(qu+1))
        # get the stats for ROI traces for these segs 
        # returns x_ran, [mean/median, std/quartiles] for each ROI or across ROIs
        chunk_info = gabors.get_roi_chunk_stats(gabors.get_2pframes_by_seg(segs, 
                                                 first=True), 
                        pre, post, byroi=byroi, dfoverf=dfoverf, 
                        nans=nans, rand=False, stats=stats, error=error)
        x_ran = chunk_info[0]
        chunk_stats = chunk_info[1]
        if remnans == 'per':
            nan_rois.append(chunk_info[2][0])
            ok_rois.append(chunk_info[2][1])
        elif remnans == 'across':
            nan_rois.extend(chunk_info[2][0])
            ok_rois.extend(chunk_info[2][1])
        # convert to nd array and store: (ROI if byroi) x stat x frame
        if data == 'all':
            if 'stats' == 'median' and 'error' == 'std':
                if byroi:
                    chunk_stats = [np.concatenate([roi[0][np.newaxis, :], roi[1]], axis=0) 
                                   for roi in chunk_stats]
                    chunk_stats = np.asarray(stats)
                else: 
                    chunk_stats = np.concatenate([chunk_stats[0][np.newaxis, :], 
                                                  chunk_stats[1]], axis=0)
            else:
                chunk_stats = np.asarray(chunk_stats)
        # retrieve the mean/median, convert to nd array and store: 
        # (ROI if byroi) x frame
        elif data == 'me':
            if byroi:
                chunk_stats = np.asarray([roi[0] for roi in chunk_stats])
            else:
                chunk_stats = chunk_stats[0]
        else:
            accepted_values_error('data', data, ['me', 'all'])
        qu_stats.append(chunk_stats)
    qu_stats = np.asarray(qu_stats)
    if remnans == 'across':
        nan_rois = sorted(list(set(nan_rois)))
        ok_rois  = sorted(list(set(ok_rois) - set(nan_rois)))
    
    returns = [x_ran, qu_stats]
    if remnans in ['per', 'across']:
        returns.append([nan_rois, ok_rois])
    return returns


#############################################
def chunk_stats_by_qu_sess(sessions, analys_par, basic_par, byroi=True, 
                           data='all', bysurp=False, twop_fps=False):
    """
    chunk_stats_by_qu_sess(sessions, analys_par, basic_par)

    Returns chunk statistics for the quintiles of interest.
    Also, added to analys_par: 
        ['all_counts'] (list): list of sublists structured as follows:
                               session x (surp value if bysurp x) quint,
                               containing number of segments for each
                               quintile
        if basic_par['remnans'] is ['across', 'per']:
            ['nan_rois'] (list): list of sublists each containing numbers of 
                                 ROIs removed, structured as follows:
                                    if 'per': session x (surp value if bysurp x) 
                                              quint
                                    if 'across': session
            ['ok_rois'] (list) : list of sublists each containing numbers of 
                                 ROIs retained, structured as follows:
                                    if 'per': session x (surp value if bysurp x) 
                                              quint
                                    if 'across': session

    Required arguments:
        - sessions (list): list of Session objects
        - analys_par (dict): dictionary containing relevant parameters
                             to extracting chunk statistics by quintile
                ['gab_k'] (int or list) : gabor kappa values to include 
                                          (e.g., 4, 16 or [4, 16])
                ['gab_fr'] (int or list): gabor frame values to include
                                         (e.g., 0, 1, 2, 3)
                ['n_quints'] (int)      : number of quintiles
                ['pre'] (float)         : range of frames to include before each 
                                          frame reference (in s)
                ['post'] (float)        : range of frames to include after each 
                                          frame reference (in s)
        - basic_par (dict): dictionary containing additional parameters 
                            relevant to analysis
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                                    default: 'std'
                ['rand'] (bool)   : if True, also includes statistics for a 
                                    random permutation of the traces (not 
                                    implemented).
                                    default: False
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'across', removes ROIs with NaN/Inf values 
                                    across subdivisions. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                                    default: 'per'
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'
                                    default: 'mean'
    Optional arguments:
        - byroi (bool)   : if True, returns statistics for each ROI. If False,
                           returns statistics across ROIs.
                           default: True 
        - data (str)     : if 'all', full statistics are returned (e.g., mean + 
                           std). If 'me', only mean/median is returned.
                           default: 'all'
        - bysurp (bool)  : if True, quintiles are separated into surprise and 
                           no surprise groups.
        - twop_fps (bool): if True, a list 2P FPS for each session is also 
                           returned

    Returns:
        - x_ran (1D array) : array of time values for the frame chunks
        - all_stats (list) : list of 2 to 5D arrays of statistics for chunks for 
                             each session:
                                (surp if bysurp x)
                                quintiles x
                                (ROIs if byroi x)
                                (statistic if data == 'all' x)
                                frames
    Optional returns:
        if twop_fps: 
            all_twop_fps (list): list of 2-photon fps values for each 
                                    session
    """

    if basic_par['rand']:
        raise NotImplementedError(('Retrieving stats for random data using '
                                  '\'chunk_stats_by_qu_sess()\' '
                                  'not implemented.'))
    if bysurp:
        surp_vals = [0, 1]
    else:
        surp_vals = ['any']

    print('Getting ROI trace stats for each session.')
    all_counts = []
    all_stats = []
    all_twop_fps = [] # 2p fps by session
    if basic_par['remnans'] in ['per', 'across']:
        all_nan_rois = []
        all_ok_rois = []
    for sess in sessions:
        print('Session {}'.format(sess.session))
        # get length of each quintile, seg_min and seg_max in dictionary
        quint_info = quint_par(sess.gabors, analys_par)
        # retrieve list of segs per quintile
        sess_counts = []
        sess_stats = []
        sess_nan_rois = []
        sess_ok_rois = []
        for surp in surp_vals:
            if surp == 0:
                print('    Non surprise')
            elif surp == 1:
                print('    Surprise')
            qu_seg, qu_count = quint_segs(sess.gabors, analys_par, quint_info, surp)
            sess_counts.append(qu_count)
            chunk_info = chunk_stats_by_qu(sess.gabors, qu_seg, 
                                           analys_par['pre'],
                                           analys_par['post'],
                                           byroi=byroi, data=data, **basic_par)
            x_ran = chunk_info[0]
            sess_stats.append(chunk_info[1])
            # if ROIs removed per subdivision
            if basic_par['remnans'] == 'per' or (basic_par['remnans'] == 'across'
                                                 and not byroi):
                sess_nan_rois.append(chunk_info[2][0])
                sess_ok_rois.append(chunk_info[2][1])
            # if ROIs to be removed across subdivisions
            elif basic_par['remnans'] == 'across' and byroi:
                sess_nan_rois.extend(chunk_info[2][0])
                sess_ok_rois.extend(chunk_info[2][1])
            if twop_fps:
                all_twop_fps.append(sess.twop_fps)
        # store by session
        sess_stats = np.asarray(sess_stats)
        # remove ROIs across subdivisions for session
        if byroi and basic_par['remnans'] == 'across':
            roi_dim = 1+bysurp # ROI dimension in array ((surp x) qu x ROI ...)
            [sess_stats, sess_nan_rois, 
                          sess_ok_rois] = remove_nan_rois(sess_stats, sess_nan_rois, 
                                                          sess_ok_rois, roi_dim)
        if basic_par['remnans'] in ['per', 'across']:
            all_nan_rois.append(sess_nan_rois)
            all_ok_rois.append(sess_ok_rois)
        if twop_fps:
            all_twop_fps.extend([sess.twop_fps])
        all_counts.append(sess_counts)
        all_stats.append(sess_stats)
    
    analys_par['seg_per_quint'] = all_counts
    if basic_par['remnans'] in ['per', 'across']:
        analys_par['nan_rois'] = all_nan_rois
        analys_par['ok_rois'] = all_ok_rois

    returns = [x_ran, all_stats]
    if twop_fps:
        returns.append(all_twop_fps)
    return returns


#############################################
def remove_nan_rois(stat_data, nan_rois, ok_rois, roi_dim):
    """
    remove_nan_rois(stats, nan_rois, ok_rois, roi_dim)

    Remove ROIs containing NaN/Infs from stats. 

    Required arguments:
        - stat_data (2 to 5D array): nd array, typically of statistics for 
                                     chunks, with dimensions:
                                         (surp if bysurp x)
                                         quintiles x
                                         (ROIs if byroi x)
                                         (statistic if data == 'all' x)
                                         frames
        - nan_rois (list)          : flat list of ROIs containing NaN/Infs, 
                                     arranged by surp, if bysurp
        - ok_rois (list)           : flat list of ROIs not containing NaN/Infs,
                                     arranged by surp, if bysurp
        - roi_dim (int)            : dimension of stats corresponding to ROIs
    
    Return:
        - stat_data (2 to 5D array): nd array, typically of statistics for  
                                     chunks, with dimensions:
                                        (surp if bysurp x)
                                        quintiles x
                                        (ROIs if byroi x)
                                        (statistic if data == 'all' x)
                                        frames
        - all_nans (list)          : flat list of ROIs containing NaN/Infs
        - all_rois (list)          : flat list of ROIs not containing NaN/Infs
    """

    # flatten nan_rois
    all_nans = sorted(list(set(nan_rois)))
    # remove any ROIs containing NaN/Infs from ok_rois, and flatten
    all_ok = sorted(list(set(ok_rois) - set(all_nans)))
    n_rois = len(all_nans) + len(all_ok)
    if len(all_nans) != 0:
        idx = [slice(None)] * roi_dim + [all_ok]
        stat_data = stat_data[idx]
        print('Removing {}/{} ROIs across quintiles and divisions: {}'
              .format(len(all_nans), n_rois, ', '.join(map(str, all_nans))))
    
    return stat_data, all_nans, all_ok


#############################################
def add_labels(ax, labels, xpos, t_hei=0.9, col='k'):
    """
    add_labels(ax, labels, xpos)

    Adds labels to a subplot.

    Required arguments:
        - ax (pyplot Axis object): pyplot Axis (subplot) object
        - labels (list or str) : list of labels to add to to axis
        - xpos (list or float) : list of x coordinates at which to add labels
                                 (same length as labels)
      

    Optional arguments:
        - t_hei (float)        : relative height (from 0 to 1) at which to 
                                 place labels, with respect to y limits
                                 default: 0.9  
        - col (str)            : color to use
                                 default: 'k'
    """

    labels = list_if_not(labels)
    xpos = list_if_not(xpos)
    if len(labels) != len(xpos):
        raise IOError(('Arguments \'labels\' and \'xpos\' must be of '
                        'the same length.'))
    ymin, ymax = ax.get_ylim()
    ypos = (ymax-ymin)*t_hei+ymin
    for l, x in zip(labels, xpos):
        ax.text(x, ypos, l, ha='center', fontsize=15, color=col)


#############################################
def add_bars(ax, hbars=None, bars=None, col='k'):
    """
    add_bars(ax)

    Adds dashed vertical bars to a subplot.

    Required arguments:
        - ax (pyplot Axis object): pyplot Axis (subplot) object

    Optional arguments:
        - hbars (list or float): list of x coordinates at which to add 
                                 heavy dashed vertical bars
                                 default: None
        - bars (list or float) : list of x coordinates at which to add 
                                 dashed vertical bars
                                 default: None
        - col (str)            : color to use
                                 default: 'k'
    """

    torem = []
    if hbars is not None:
        hbars = list_if_not(hbars)
        torem = hbars
        for b in hbars:
            ax.axvline(x=b, ls='dashed', c='k', lw='{}'.format(2, alpha=0.5)) 
    if bars is not None:
        bars = remove_if(bars, torem)
        for b in bars:
            ax.axvline(x=b, ls='dashed', c='k', lw='{}'.format(1, alpha=0.5)) 


#############################################
def plot_traces(ax, chunk_val, stats='mean', error='std', title='', lw=1.5, 
                col=None, alpha=0.5, plot_err=True, xticks=None, yticks=None):
    """
    plot_traces(ax, chunk_val)

    Plot traces (mean/median with shaded error bars) on axis (ax).

    Required arguments:
        - ax (pyplot axis object): pyplot axis
        - chunk_val (2D array)   : array of chunk statistics, with the first
                                   dimension corresponds to the statistics 
                                   (x_ran [0], mean/median [1], deviation [2] or
                                   [2:3] if quartiles)
    Optional arguments:
        - stats (str)          : statistic parameter, i.e. 'mean' or 'median'
                                 default: 'mean'
        - error (str)          : error statistic parameter, i.e. 'std' or 'sem'
                                 default: 'std'
        - title (str)          : axis title
        - lw (float)           : pyplot line weight variable
        - alpha (float)        : pyplot alpha variable controlling shading 
                                 transparency (from 0 to 1)
    """
    
    ax.plot(chunk_val[0], chunk_val[1], lw=lw, color=col)
    col = ax.lines[-1].get_color()
    if plot_err:
        # only condition where pos and neg error are different
        if stats == 'median' and error == 'std': 
            ax.fill_between(chunk_val[0], chunk_val[2], chunk_val[3], 
                            facecolor=col, alpha=alpha)
        else:
            ax.fill_between(chunk_val[0], chunk_val[1] - chunk_val[2], 
                            chunk_val[1] + chunk_val[2], 
                            facecolor=col, alpha=alpha)

    ax.set_title(title)
    ax.set_ylabel('dF/F')
    ax.set_xlabel('Time (s)')
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)


#############################################
def plot_traces_by_qu_surp_sess(sessions, analys_par, basic_par, fig_par,
                                sess_par):
    """
    plot_traces_by_qu_surp_sess(sessions, analys_par, basic_par, fig_par,
                                sess_par)

    Retrieves chunk statistics by session x surp val x quintile and
    plots traces by quintile/surprise with each session in a separate subplot.
    Saves parameter dictionaries relevant to analyses in a pickle.

    Required arguments:
        - sessions (list)  : list of Session objects
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['act_sess_ns'] (list)  : actual overall session number for
                                          each session
                ['gab_fr'] (int or list): gabor frame values to include
                                         (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list) : gabor kappa values to include 
                                          (e.g., 4, 16 or [4, 16])
                ['mouse_ns'] (list)     : mouse numbers corresponding to 
                                          sessions
                ['n_quints'] (int)      : number of quintiles
                ['pre'] (float)         : range of frames to include before each 
                                          frame reference (in s)
                ['post'] (float)        : range of frames to include after each 
                                          frame reference (in s)
                ['sess_ns'] (list)      : list of session IDs

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                                    default: 'std'
                ['rand'] (bool)   : if True, also includes statistics for a 
                                    random permutation of the traces (not 
                                    implemented).
                                    default: False
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'across', removes ROIs with NaN/Inf values 
                                    across subdivisions. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                                    default: 'per'
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'
                                    default: 'mean'
        - fig_par (dict)   : dictionary containing figure parameters:
                ['bbox'] (str)           : bbox_inches parameter for 
                                           plt.savefig(), e.g., 'tight'
                ['datetime'] (bool)      : if True, figures are saved in a 
                                           subfolder named based on the date 
                                           and time.
                ['fig_ext'] (str)        : extension (without '.') with which to
                                           save figure
                ['figdir_prel_roi'] (str): main folder in which to save figure
                ['ncols'] (int)          : number of columns in the figure
                ['overwrite'] (bool)     : if False, overwriting existing 
                                           figures is prevented by adding 
                                           suffix numbers.
                ['sharey'] (bool)        : if True, y axis lims are shared 
                                           across subplots
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['surp_quint'] (str)     : specific subfolder in which to save 
                                           folder
        - sess_par (dict)  : dictionary containing session parameters:
                ['gab_k'] (int or list) : gabor kappa parameter
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                               'L5_soma', 'L23_dend', 'L5_dend', 
                                               'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    """

    print(('\nAnalysing and plotting surprise vs non surprise ROI traces '
           'by quintile ({}) \n({}).').format(analys_par['n_quints'],
                                              sess_par_str(sess_par, 
                                                           analys_par['gab_k'],
                                                           str_type='print')))
    
    seg_bars = [0.3, 0.6, 0.9, 1.2] # light lines
    xpos = [0.15, 0.45, 0.75, 1.05, 1.35] # seg labels
    t_heis = [0.85, 0.95]
    labels_nosurp = ['D', 'gray', 'A', 'B', 'C']
    labels_surp = ['E', 'gray', 'A', 'B', 'C']
    if analys_par['n_quints'] <= 7:
        col_nosurp = ['steelblue', 'dodgerblue', 'blue', 'darkblue',  
                      'mediumblue', 'royalblue', 
                      'cornflowerblue'][0:analys_par['n_quints']]
        col_surp   = ['coral', 'darkorange', 'sandybrown', 'chocolate', 
                      'orange', 'peru', 
                      'sienna'][0:analys_par['n_quints']]
    else:
        raise NotImplementedError(('Not enough colors preselected for more '
                                    'than 4 quintiles.'))

    # get the stats (all) separating by session, surprise and quintiles
    chunk_info = chunk_stats_by_qu_sess(sessions, analys_par, basic_par, 
                                        byroi=False, data='all', bysurp=True)
    x_ran = chunk_info[0]
    all_stats = chunk_info[1]
    fig, ax = init_fig(len(sessions), fig_par)
    n = fig_par['ncols']

    for i, sess in enumerate(sessions):
        leg = []
        for s, [col, leg_ext] in enumerate(zip([col_nosurp, col_surp],
                                               ['nosurp', 'surp'])):
            for q in range(analys_par['n_quints']):
                if basic_par['remnans'] in ['per', 'across']:
                    n_rois = len(analys_par['ok_rois'][i][s][q])
                else:
                    n_rois = sess.nroi
                chunk_stats = np.concatenate([x_ran[np.newaxis, :], 
                                              all_stats[i][s][q]], axis=0)
                plot_traces(ax[i/n][i%n], chunk_stats, stats=basic_par['stats'], 
                            error=basic_par['error'], col=col[q], 
                            alpha=0.4/analys_par['n_quints'],
                            title=('Mouse {} - gab{} {} dF/F across gabor seqs' 
                                   '\n(sess {}, {}, n={})')
                                .format(analys_par['mouse_ns'][i], 
                                        gab_k_par_str(analys_par['gab_k']),
                                        stat_par_str(basic_par['stats'], 
                                                 basic_par['error']),
                                        analys_par['act_sess_ns'][i], 
                                        sess_par['layer'], n_rois))
                leg.extend(['{}-{} ({})'.format(q+1, leg_ext, 
                           analys_par['seg_per_quint'][i][s][q])])
        for s, [lab, col, t_hei] in enumerate(zip([labels_nosurp, labels_surp], 
                                                  [col_nosurp, col_surp], 
                                                  t_heis)):
            add_labels(ax[i/n][i%n], lab, xpos, t_hei, col=col[0])
        add_bars(ax[i/n][i%n], bars=seg_bars)
        ax[i/n][i%n].legend(leg)
    
    save_dir = '{}/{}'.format(fig_par['figdir_prel_roi'], fig_par['surp_quint'])
    save_name = 'roi_av_{}_{}quint'.format(sess_par_str(sess_par, analys_par['gab_k']),
                                             analys_par['n_quints'])
    full_dir = save_fig(fig, save_dir, save_name, fig_par)

    info = {'sess_par': sess_par,
            'basic_par': basic_par,
            'analys_par': analys_par,
            }
    save_info(info, full_dir, save_name)


#############################################
def run_permute(all_data, act_diff, div='half', stats='mean', op='diff', 
                tails=2, n_perms=10000, p_val=0.05):       
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
        - tails (str or int): which tail(s) to test: 'up', 'lo', 2
                              default: 2
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
def id_elem(rand_vals, act_vals, tails=2, p_val=0.05, print_elems=False):
    """
    id_elem(rand_vals, act_vals)

    Identify elements whose actual values are beyond the threshold(s) obtained 
    with distributions of randomly generated values.

    Required arguments:
        - rand_vals (2D array): random values for each element: elem x val
        - act_vals (1D array) : actual values for each element

    Optional arguments:
        - tails (str or int): which tail(s) to test: 'up', 'lo', 2
                              default: 2
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
    elif tails == 2:
        lo_threshs = np.percentile(rand_vals, p_val*100/2.0, axis=1)
        lo_elems = np.where(act_vals < lo_threshs)[0]
        up_threshs = np.percentile(rand_vals, 100-p_val*100/2.0, axis=1)
        up_elems = np.where(act_vals > up_threshs)[0]
        if print_elems:
            print_elem_list(lo_elems, 'lo', act_vals[lo_elems])
            print_elem_list(up_elems, 'up', act_vals[up_elems])
        elems = [lo_elems, up_elems]
    else:
        accepted_values_error('tails', tails, ['up', 'lo', 2])
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
        print('\tSign ROIs {}: None'.format(tail))
    else:
        print('\tSign ROIs {}: {}'.format(tail, ', '.join('{}'.format(x) 
            for x in elems)))
        if act_vals is not None:
            if len(act_vals) != len(elems):
                raise ValueError(('\'elems\' and \'act_vals\' should be the '
                                  'same length, but are of length {} and {} '
                                  'respectively.').format(len(elems), len(act_vals)))
            print('\tdiffs: {}'.format(', '.join(['{:.2f}'.format(x) 
                                        for x in act_vals])))    

#############################################
def sep_grps(sign_rois, n_rois, grps='all', tails=2, add_nosurp=False):
    """
    sep_grps(sign_rois, n_rois)

    Separate ROIs into groups based on whether their first/last quintile was
    significant in a specific tail.

    Required arguments:
        - sign_rois (list): list of sublists per quintile ['first', 'last'],
                            each containing numbers of ROIs showing 
                            significant differences within the quintile, 
                            separated into tails ['lo', 'up'] if applicable
        - n_rois (int)    : total number of ROIs in data (signif or not)
    Optional arguments:
        - grps (str or list): set of groups or list of sets of groups to 
                              return, e.g., 'all', 'change', 'no_change', 
                              'reduc', 'incr'
                              default: 'all'
        - tails (str)       : tail(s) used in analysis: 'up', 'lo' or 2
                              default: 2
        - add_nosurp (bool) : if True, group of ROIs showing no significance in 
                              either is included in the groups returned
                              default: False 
    Returns:
        - roi_grps (list)   : lists structured as follows:
                              if grp parameter includes only one group, 
                                  ROIs per roi group
                              otherwise: sets x roi grps
                              numbers included in the group
        - grp_names (list)  : if grp parameter includes only one group, list of 
                              names of roi grps (order preserved)
                              otherwise: list of sublists per set, each 
                              containing names of roi grps per set
    """
    grps = list_if_not(grps)
    # get ROI numbers for each group
    if tails in ['up', 'lo']:
        # sign_rois[first/last]
        all_rois      = range(n_rois)
        surp_surp     = list(set(sign_rois[0]) & set(sign_rois[1]))
        surp_nosurp   = list(set(sign_rois[0]) - set(sign_rois[1]))
        nosurp_surp   = list(set(sign_rois[1]) - set(sign_rois[0]))
        nosurp_nosurp = list(set(all_rois) - set(surp_surp) - 
                                set(surp_nosurp) - set(nosurp_surp))
        # to store stats
        roi_grps = [surp_surp, surp_nosurp, nosurp_surp, nosurp_nosurp]
        # group names
        grp_names = ['surp_surp', 'surp_nosurp', 'nosurp_surp', 'nosurp_nosurp']
        nosurp_ind = 3
        grp_inds = []
        for i, g in enumerate(grps):
            if g == 'all':
                grp_ind = range(len(roi_grps))
            elif g == 'change':
                grp_ind = [1, 2]
            elif g == 'no_change':
                grp_ind = [0, 3]
            elif g == 'reduc':
                grp_ind = [1]
            elif g == 'incr':
                grp_ind = [2]
            else:
                accepted_values_error('grps', g, ['all', 'change', 'no_change', 
                                                    'reduc', 'incr'])
            if add_nosurp and nosurp_ind not in grp_ind:
                grp_ind.extend([nosurp_ind])
            grp_inds.append(sorted(grp_ind))

    elif tails == 2:
        # sign_rois[first/last][lo/up]
        all_rois = range(n_rois)         
        surp_up_surp_up = list(set(sign_rois[0][1]) & set(sign_rois[1][1]))
        surp_up_surp_lo = list(set(sign_rois[0][1]) & set(sign_rois[1][0]))
        surp_lo_surp_up = list(set(sign_rois[0][0]) & set(sign_rois[1][1]))
        surp_lo_surp_lo = list(set(sign_rois[0][0]) & set(sign_rois[1][0]))

        surp_up_nosurp = list((set(sign_rois[0][1]) - set(sign_rois[1][1]) - 
                                set(sign_rois[1][0])))
        surp_lo_nosurp = list((set(sign_rois[0][0]) - set(sign_rois[1][1]) -
                                set(sign_rois[1][0])))
        
        nosurp_surp_up = list((set(sign_rois[1][1]) - set(sign_rois[0][1]) - 
                                set(sign_rois[0][0])))
        nosurp_surp_lo = list((set(sign_rois[1][0]) - set(sign_rois[0][1]) -
                                set(sign_rois[0][0])))
        
        nosurp_nosurp = list((set(all_rois) - set(sign_rois[0][1]) -
                                set(sign_rois[1][1]) - set(sign_rois[0][0]) -
                                set(sign_rois[1][0])))
        # to store stats
        roi_grps = [surp_up_surp_up, surp_up_surp_lo, surp_lo_surp_up, 
                    surp_lo_surp_lo, surp_up_nosurp, surp_lo_nosurp, 
                    nosurp_surp_up, nosurp_surp_lo, nosurp_nosurp]
        nosurp_ind = 8 # index of nosurp_nosurp
        # group names 
        grp_names = ['surpup_surpup', 'surpup_surplo', 'surplo_surpup', 
                     'surplo_surplo', 'surpup_nosurp', 'surplo_nosurp', 
                     'nosurp_surpup', 'nosurp_surplo', 'nosurp_nosurp']
        nosurp_ind = 8
        grp_inds = []
        for i, g in enumerate(grps):
            if g == 'all':
                grp_ind = range(len(roi_grps))
            elif g == 'change':
                grp_ind = [1, 2, 4, 5, 6, 7]
            elif g == 'no_change':
                grp_ind = [0, 3, 8]
            elif g == 'reduc':
                grp_ind = [1, 4, 7]
            elif g == 'incr':
                grp_ind = [2, 5, 6]
            else:
                accepted_values_error('grps', grps, ['all', 'change', 'no_change', 
                                                    'reduc', 'incr'])
            if add_nosurp and nosurp_ind not in grp_ind:
                    grp_ind.extend([nosurp_ind])
            grp_inds.append(sorted(grp_ind))

    all_roi_grps = [[roi_grps[i] for i in grp_ind] for grp_ind in grp_inds]
    all_grp_names = [[grp_names[i] for i in grp_ind] for grp_ind in grp_inds]
    if len(grps) == 1:
        all_roi_grps = all_roi_grps[0]
        all_grp_names = all_grp_names[0] 

    return all_roi_grps, all_grp_names


#############################################
def grp_stats(integ_data, grps, op='diff', stats='mean', error='std'):
    """
    grp_stats(all_data, grps)

    Calculate statistics (e.g. mean + std) across quintiles for each group 
    and session.

    Required arguments:
        - integ_data (list): list of 3D arrays of mean/medians integrated 
                             across chunks, for each session:
                                 surp if bysurp x
                                 quintiles x
                                 ROIs if byroi

        - grps (list)      : list of sublists per session, each containing
                             sublists per roi grp with ROI numbers included in 
                             the group: session x roi_grp
    Optional arguments:
        - op (str)   : operation to use to compare groups, 
                       i.e. 'diff': grp1-grp2, or 'ratio': grp1/grp2
                       default: 'diff'
        - stats (str): statistic parameter, i.e. 'mean' or 'median'
                       default: 'mean'
        - error (str): error statistic parameter, i.e. 'std' or 'sem'
                       default: 'std'
    Returns:
        - all_grp_st (4D arrays): array of group stats (mean/median, error) 
                                  structured as:
                                  session x quintile x grp x stat 
        - all_ns (2D array)     : array of group ns, structured as:
                                  session x grp
    """

    if stats == 'median' and error == 'std':
        n_stats = 3
    else:
        n_stats = 2
    # array to collect group stats, structured as sessions x quintile x grp x stat
    all_grp_st = np.empty([len(integ_data), integ_data[0].shape[1], 
                           len(grps[0]), n_stats])
    # array to collect number of ROIs per group, structured as 
    # sessions x grp
    all_ns = np.empty([len(integ_data), len(grps[0])], dtype=int)
    for i, [sess_data, sess_grps] in enumerate(zip(integ_data, grps)):
        sess_data = calc_op(sess_data, 'diff', op, surp_dim=0)
        # take mean/median and error for each group
        for g, grp in enumerate(sess_grps):
            all_ns[i, g] = len(grp)
            for q, qu_data in enumerate(sess_data):
                all_grp_st[i, q, g] = np.nan
                if len(grp) != 0:
                    all_grp_st[i, q, g, 0] = mean_med(qu_data[grp], stats)
                    all_grp_st[i, q, g, 1:] = np.asarray(error_stat(qu_data[grp], stats, error))

    return all_grp_st, all_ns


#############################################
def integ_per_qu_surp_sess(data_me, twop_fps, op='diff'):
    """
    integ_per_qu_surp_sess(data_me, twop_fps)

    Takes integral over data from each session, structured as surprise value x 
    quintile x ROI.

    Required arguments:
        - data_me (list) : list of 3 to 4D array of mean/medians for chunks, 
                           for each session:
                                surp if bysurp x
                                quintiles x
                                (ROIs if byroi x)
                                frames
        - twop_fps (list): list of 2-photon fps values for each session
    Optional arguments:
        - op (str): operation to use to compare groups, i.e. 'diff': grp1-grp2, 
                    or 'ratio': grp1/grp2 or None
                    default: 'diff'
    Returns:
        - integ_dffs (list)    : list of 2 to 3D arrays of mean/medians 
                                 integrated across chunks, for each session:
                                    surp if bysurp x
                                    quintiles x
                                    (ROIs if byroi)
    Optional returns (if op is not None)
            - integ_dffs_rel (list): list of 1 to 2D array of differences/ratios 
                                     between surprise and non surprise conditions 
                                     of mean/medians integrated across chunks, for 
                                     each session:
                                        quintiles x
                                        (ROIs if byroi)
    """

    integ_dffs = []
    integ_dffs_rel = []
    for s, data in enumerate(data_me):
        all_integ = integ_dff(data, twop_fps[s], axis=-1)
        if op == 'diff':
            integ_dffs_rel.append(all_integ[1] - all_integ[0])
        elif op == 'ratio':
            integ_dffs_rel.append(all_integ[1]/all_integ[0])
        integ_dffs.append(all_integ)
    if op is not None:
        return integ_dffs, integ_dffs_rel
    else:
        return integ_dffs


#############################################
def integ_per_grp_qu_sess(sessions, analys_par, basic_par, roi_grp_par):
    """
    integ_per_grp_qu_sess(sessions, analys_par, basic_par, roi_grp_par)

    Takes integral over data, across sessions and calculates number of ROIs
    for each session.

    Required arguments:
        - sessions (list)  : list of Session objects
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['gab_k'] (int or list) : gabor kappa values to include 
                                          (e.g., 4, 16 or [4, 16])
                ['gab_fr'] (int or list): gabor frame values to include
                                         (e.g., 0, 1, 2, 3)
                ['n_quints'] (int)      : number of quintiles
                ['pre'] (float)         : range of frames to include before each 
                                          frame reference (in s)
                ['post'] (float)        : range of frames to include after each 
                                          frame reference (in s)
        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                                    default: 'std'
                ['rand'] (bool)   : if True, also includes statistics for a 
                                    random permutation of the traces (not 
                                    implemented).
                                    default: False
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'across', removes ROIs with NaN/Inf values 
                                    across subdivisions. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                                    default: 'per'
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'
                                    default: 'mean'
        - roi_grp_par (dict) : dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
                                       default: 'diff'
    Returns:
        - integ_dffs (list)    : list of 3D arrays of mean/medians integrated 
                                 across chunks, for each session:
                                    surp if bysurp x
                                    quintiles x
                                    ROIs
        - integ_dffs_rel (list): list 2D array of differences/ratios between 
                                 surprise and non surprise conditions 
                                 of mean/medians integrated across chunks, for 
                                 each session:
                                    quintiles x
                                    ROIs
        - n_rois (1D array)    : number of ROIs retained in each session
    """

    if basic_par['remnans'] == 'per':
        print('NaNs must be removed for across subgroups, not per.')
        basic_par['remnans'] == 'across'
    
    chunk_info = chunk_stats_by_qu_sess(sessions, analys_par, basic_par, 
                                        byroi=True, data='me', bysurp=True, 
                                        twop_fps=True)

    all_me= chunk_info[1]
    twop_fps = chunk_info[-1]

    integ_dffs, integ_dffs_rel = integ_per_qu_surp_sess(all_me, twop_fps, 
                                                        op=roi_grp_par['op'])
    n_rois = np.empty(len(sessions), dtype=int)
    for s in range(len(sessions)):
        # integ_dffs[s] structure: quint x ROIs
        n_rois[s] = integ_dffs_rel[s].shape[1]

    return integ_dffs, integ_dffs_rel, n_rois


#############################################
def signif_rois_by_grp_sess(sessions, integ_dffs_rel, analys_par, basic_par, 
                            perm_par, roi_grp_par):
    """
    plot_rois_by_grp_qu_sess(sessions, analys_par, basic_par, fig_par, perm_par, 
                             roi_grp_par, sess_par)

    Identifies ROIs showing significant surprise in first and/or last quintile,
    groups accordingly and retrieves statistics for each group.

    Required arguments:
        - sessions (list)      : list of Session objects
        - integ_dffs_rel (list): list of 2D array of differences/ratios 
                                 between surprise and non surprise conditions 
                                 of mean/medians integrated across chunks, for 
                                 each session:
                                    quintiles x
                                    ROIs
        - analys_par (dict)    : dictionary containing specific analysis 
                                 parameters:
                ['gab_fr'] (int or list): gabor frame values to include
                                         (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list) : gabor kappa values to include 
                                          (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)      : number of quintiles
                ['pre'] (float)         : range of frames to include before each 
                                          frame reference (in s)
                ['post'] (float)        : range of frames to include after each 
                                          frame reference (in s)
        - basic_par (dict)     : dictionary containing basic analysis 
                                 parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'across', removes ROIs with NaN/Inf values 
                                    across subdivisions. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                                    default: 'per'
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'
                                    default: 'mean'
        - perm_par (dict)    : dictionary containing permutation analysis 
                               parameters:
                ['n_perms'] (int)     : nbr of permutations to run
                                        default: 10000
                ['p_val'] (float)     : p-value to use for significance thresholding 
                                        (0 to 1)
                ['tails'] (str or int): which tail(s) to test: 'up', 'lo', 2
                                        default: 2
        - roi_grp_par (dict) : dictionary containing ROI grouping parameters:
                ['op'] (str)          : operation to use to compare groups, 
                                        i.e. 'diff': grp1-grp2, or 'ratio': 
                                        grp1/grp2
                                        default: 'diff'
                ['grps'] (str or list): set or sets of groups to return, 
                                        e.g., 'all', 'change', 'no_change', 
                                        'reduc', 'incr'.
                                        default: 'all'
                ['add_nosurp'] (bool) : if True, group of ROIs showing no 
                                        significance in either is included in the 
                                        groups returned
                                        default: False   
    Returns:
        - all_roi_grps (list): list of sublists per session, containing ROI  
                               numbers included in each  group, structured as 
                               follows:
                               if sets of groups are passed: session x set x roi_grp
                               if one group is passed: session x roi_grp
        - grp_names (list )  : list of names of the ROI groups in roi grp lists
                               (order preserved)   
    """

    all_roi_grps = []
    for s, sess in enumerate(sessions):
        print('\nSession {}'.format(sess.session))
        fps = sess.twop_fps
        quint_info = quint_par(sess.gabors, analys_par)
        qu_seg, _ = quint_segs(sess.gabors, analys_par, quint_info, surp='any')
        all_rois = []
        # Run permutation test for first and last quintiles
        for q, pos in zip([0, analys_par['n_quints']-1], ['First', 'Last']):
            print('\n{} quintile'.format(pos))
            # get dF/F for each segment and each ROI
            qu_twop_fr = sess.gabors.get_2pframes_by_seg(qu_seg[q], first=True)
            _, roi_traces = sess.gabors.get_roi_trace_chunks(qu_twop_fr, 
                                                             analys_par['pre'], 
                                                             analys_par['post'], 
                                                             dfoverf=basic_par['dfoverf'])
            # remove previously removed ROIs if applicable 
            if basic_par['remnans'] == 'across':
                ok_rois    = analys_par['ok_rois'][s]
                roi_traces = roi_traces[ok_rois]
            # get area under the curve
            roi_dffs = integ_dff(roi_traces, fps, axis=1)
            # run permutation test on dataset (dffs: ROI x seg)
            sign_rois = run_permute(roi_dffs, integ_dffs_rel[s][q], 
                                    div=analys_par['seg_per_quint'][s][1][q], 
                                    stats=basic_par['stats'],  
                                    op=roi_grp_par['op'], **perm_par)
            all_rois.append(sign_rois)

        roi_grps, grp_names = sep_grps(all_rois, n_rois=roi_dffs.shape[0], 
                                       grps=roi_grp_par['grps'], 
                                       tails=perm_par['tails'],
                                       add_nosurp=roi_grp_par['add_nosurp'])
        
        all_roi_grps.append(roi_grps)
    
    return all_roi_grps, grp_names

        
#############################################
def plot_rois_by_grp_qu_sess(sessions, analys_par, basic_par, fig_par, perm_par, 
                             roi_grp_par, sess_par):

    """
    plot_rois_by_grp_qu_sess(sessions, analys_par, basic_par, fig_par, perm_par, 
                             roi_grp_par, sess_par)

    Identify ROIs showing significant surprise in first and/or last quintile,
    group accordingly and plot average integrated surprise, no surprise or 
    difference between surprise and no surprise activity per group across
    quintiles with each session in a separate subplot.

    Required arguments:
        - sessions (list)  : list of Session objects
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['act_sess_ns'] (list)  : actual overall session number for
                                          each session
                ['gab_fr'] (int or list): gabor frame values to include
                                         (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list) : gabor kappa values to include 
                                          (e.g., 4, 16 or [4, 16])
                ['mouse_ns'] (list)     : mouse numbers corresponding to 
                                          sessions
                ['n_quints'] (int)      : number of quintiles
                ['pre'] (float)         : range of frames to include before each 
                                          frame reference (in s)
                ['post'] (float)        : range of frames to include after each 
                                          frame reference (in s)
                ['sess_ns'] (list)      : list of session IDs

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                                    default: 'std'
                ['rand'] (bool)   : if True, also includes statistics for a 
                                    random permutation of the traces (not 
                                    implemented).
                                    default: False
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'across', removes ROIs with NaN/Inf values 
                                    across subdivisions. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                                    default: 'per'
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'
                                    default: 'mean'
        - fig_par (dict)   : dictionary containing figure parameters:
                ['bbox'] (str)           : bbox_inches parameter for 
                                           plt.savefig(), e.g., 'tight'
                ['datetime'] (bool)      : if True, figures are saved in a 
                                           subfolder named based on the date 
                                           and time.
                ['fig_ext'] (str)        : extension (without '.') with which to
                                           save figure
                ['figdir_prel_roi'] (str): main folder in which to save figure
                ['ncols'] (int)          : number of columns in the figure
                ['overwrite'] (bool)     : if False, overwriting existing 
                                           figures is prevented by adding 
                                           suffix numbers.
                ['sharey'] (bool)        : if True, y axis lims are shared 
                                           across subplots
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['surp_quint'] (str)     : specific subfolder in which to save 
                                           folder
        - perm_par (dict)    : dictionary containing permutation analysis 
                               parameters:
                ['n_perms'] (int)     : nbr of permutations to run
                                        default: 10000
                ['p_val'] (float)     : p-value to use for significance  
                                        thresholding (0 to 1)
                ['tails'] (str or int): which tail(s) to test: 'up', 'lo', 2
                                        default: 2
        - roi_grp_par (dict) : dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
                                       default: 'diff'
                ['plot_vals'] (str)  : values to plot 'diff' (surp-nosurp), 
                                       'surp' or 'nosurp'
                ['grps'] (str)       : set of groups to return, e.g., 'all', 
                                       'change', 'no_change', 'reduc', 'incr'.
                                       If several sets are passed, only the
                                       first is plotted.
                                       default: 'all'
                ['add_nosurp'] (bool): if True, group of ROIs showing no 
                                       significance in either is included in the 
                                       groups returned
                                       default: False         
        - sess_par (dict)  : dictionary containing session parameters:
                ['gab_k'] (int or list) : gabor kappa parameter
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                               'L5_soma', 'L23_dend', 'L5_dend', 
                                               'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    """

    print(('\nAnalysing and plotting {} ROI surp vs nosurp responses by '
           'quintile ({}). \n{}.').format(op_par_str(roi_grp_par['plot_vals'],
                                                     roi_grp_par['op'], True, 'print'),
                                            analys_par['n_quints'],
                                            sess_par_str(sess_par, 
                                                         analys_par['gab_k'],
                                                         str_type='print')))
    
    # permutation test
    [integ_dffs, integ_dffs_rel, 
                 n_rois] = integ_per_grp_qu_sess(sessions, analys_par,  
                                                 basic_par, roi_grp_par)
    
    # make sure only one set of ROI groups is passed
    roi_grp_par['grps'] = list_if_not(roi_grp_par['grps'])
    if len(roi_grp_par['grps']) > 1:
        roi_grp_par['grps'] = roi_grp_par['grps'][0]

    # identify significant ROIs
    all_roi_grps, grp_names = signif_rois_by_grp_sess(sessions, integ_dffs_rel, 
                                                      analys_par, basic_par, 
                                                      perm_par, roi_grp_par)
    
    # get statistics per group and number of ROIs per group
    grp_st, ns = grp_stats(integ_dffs, all_roi_grps, roi_grp_par['op'], 
                           basic_par['stats'], basic_par['error'])

    x_ran = [x+1 for x in range(analys_par['n_quints'])]
    fig, ax = init_fig(len(sessions), fig_par)
    n = fig_par['ncols']
    
    for i, sess_st in enumerate(grp_st):
        act_leg = []
        sub_ax = ax[i/n][i%n]
        for g, g_n in enumerate(ns[i]):
            me = sess_st[:, g, 0]
            if basic_par['stats'] == 'median' and basic_par['error'] == 'std':
                yerr1 = me - sess_st[:, g, 1]
                yerr2 = sess_st[:, g, 2] - me
                yerr = [yerr1, yerr2]
            else:
                yerr = sess_st[:, g, 1]
            sub_ax.errorbar(x_ran, me, yerr, fmt='-o', capsize=4, capthick=2)
            act_leg.append('{} ({})'.format(grp_names[g], g_n))

        sub_ax.set_title(('Mouse {} - {} gab{} \n{} seqs \n(sess {}, {}, '
                        '{} tail (n={}))').format(analys_par['mouse_ns'][i], 
                                                stat_par_str(basic_par['stats'], 
                                                         basic_par['error']),
                                                gab_k_par_str(analys_par['gab_k']), 
                                                op_par_str(roi_grp_par['plot_vals'],
                                                           roi_grp_par['op'], True, 'print'),
                                                analys_par['act_sess_ns'][i],
                                                sess_par['layer'], 
                                                perm_par['tails'], n_rois[i]))
        sub_ax.set_xticks(x_ran)
        sub_ax.set_ylabel('dF/F')
        sub_ax.set_xlabel('Quintiles')
        sub_ax.legend(act_leg)

    save_dir = '{}/{}'.format(fig_par['figdir_prel_roi'], fig_par['surp_quint'])
    save_name = 'roi_{}_grps_{}_{}quint_{}tail'.format(sess_par_str(sess_par, 
                                                        analys_par['gab_k']),
                                                   roi_grp_par['plot_vals'],
                                                   analys_par['n_quints'],
                                                   perm_par['tails'])
    full_dir = save_fig(fig, save_dir, save_name, fig_par)

    info = {'sess_par': sess_par,
            'basic_par': basic_par,
            'analys_par': analys_par,
            'perm_par': perm_par,
            'roi_grp_par': roi_grp_par
            }
    
    save_info(info, full_dir, save_name)


#############################################
def grp_traces_by_qu_surp_sess(sessions, all_roi_grps, analys_par, basic_par, 
                               roi_grp_par, quint_ns):
    """
    Required arguments:
        - sessions (list): list of Session objects
        - all_roi_grps (list): list of sublists per session, each containing
                               sublists per roi grp with ROI numbers included in 
                               the group: session x roi_grp
        - analys_par (dict): dictionary containing relevant parameters
                             to extracting chunk statistics by quintile
                ['gab_k'] (int or list) : gabor kappa values to include 
                                          (e.g., 4, 16 or [4, 16])
                ['gab_fr'] (int or list): gabor frame values to include
                                         (e.g., 0, 1, 2, 3)
                ['n_quints'] (int)      : number of quintiles
                ['pre'] (float)         : range of frames to include before each 
                                          frame reference (in s)
                ['post'] (float)        : range of frames to include after each 
                                          frame reference (in s)
        - basic_par (dict): dictionary containing additional parameters 
                            relevant to analysis
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                                    default: 'std'
                ['rand'] (bool)   : if True, also includes statistics for a 
                                    random permutation of the traces (not 
                                    implemented).
                                    default: False
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'across', removes ROIs with NaN/Inf values 
                                    across subdivisions. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                                    default: 'per'
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'
                                    default: 'mean'
        - roi_grp_par (dict) : dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
                                       default: 'diff'
                ['plot_vals'] (str)  : values to plot 'diff' (surp-nosurp), 
                                       'surp' or 'nosurp' 
        - quint_ns (list or int): indices of the quintiles to include
    Returns:
        - x_ran (1D array)    : array of time values for the frame chunks
        - grp_stats (6D array): statistics for ROI groups structured as:
                                sess x surp x qu x ROI grp x stats x frame
    """

    # get sess x surp x quint x ROIs x frames
    chunk_info = chunk_stats_by_qu_sess(sessions, analys_par, basic_par, 
                                        byroi=True, data='me', bysurp=True)
    x_ran = chunk_info[0]

    # retain quintiles passed
    chunk_me = [sess_me[:, quint_ns] for sess_me in chunk_info[1]]
    # apply operation on surp vs nosurp data
    data_me = [calc_op(x, roi_grp_par['plot_vals'], roi_grp_par['op'], 
                       surp_dim=0) for x in chunk_me]

    if basic_par['stats'] and basic_par['error']:
        n_stats = 3
    else:
        n_stats = 2
    
    # sess x quintile (first/last) x ROI grp x stats x frame
    grp_stats = np.empty([len(sessions), len(quint_ns), len(all_roi_grps[0]), 
                          n_stats, len(x_ran)]) * np.nan
    for i, sess in enumerate(data_me):
        for q, quint in enumerate(sess): 
            for g, grp_rois in enumerate(all_roi_grps[i]):
                # leave NaNs if no ROIs in group
                if len(grp_rois) == 0:
                    continue
                grp_stats[i, q, g, 0] = mean_med(quint[grp_rois], 
                                                 basic_par['stats'], axis=0)
                grp_stats[i, q, g, 1:] = np.asarray(error_stat(quint[grp_rois], 
                                                    basic_par['stats'], 
                                                    basic_par['error'], axis=0))
    return x_ran, grp_stats


#############################################
def plot_roi_traces_by_grp(sessions, analys_par, basic_par, fig_par,  
                               perm_par, roi_grp_par, sess_par):
        """
        plot_roi_traces_by_grp(sessions, analys_par, basic_par, fig_par,  
                               perm_par, roi_grp_par, sess_par)

        Identify ROIs showing significant surprise in first and/or last quintile,
        group accordingly and plot traces across surprise, no surprise or 
        difference between surprise and no surprise activity per quintile 
        (first/last) with each group in a separate subplot and each session
        in a different figure.
        Saves parameter dictionaries relevant to analyses in a pickle.

        Required arguments:
            - sessions (list)  : list of Session objects
            - analys_par (dict): dictionary containing specific analysis parameters:
                    ['act_sess_ns'] (list)  : actual overall session number for
                                            each session
                    ['gab_fr'] (int or list): gabor frame values to include
                                            (e.g., 0, 1, 2, 3)
                    ['gab_k'] (int or list) : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                    ['mouse_ns'] (list)     : mouse numbers corresponding to 
                                            sessions
                    ['n_quints'] (int)      : number of quintiles
                    ['pre'] (float)         : range of frames to include before each 
                                            frame reference (in s)
                    ['post'] (float)        : range of frames to include after each 
                                            frame reference (in s)
                    ['sess_ns'] (list)      : list of session IDs

            - basic_par (dict) : dictionary containing basic analysis parameters:
                    ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                        traces.
                    ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                        'sem'
                                        default: 'std'
                    ['rand'] (bool)   : if True, also includes statistics for a 
                                        random permutation of the traces (not 
                                        implemented).
                                        default: False
                    ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                        for each subdivision (quintile/surprise). If 
                                        'across', removes ROIs with NaN/Inf values 
                                        across subdivisions. If 'no', ROIs with 
                                        NaN/Inf values are not removed.
                                        default: 'per'
                    ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'
                                        default: 'mean'
            - fig_par (dict)   : dictionary containing figure parameters:
                    ['bbox'] (str)           : bbox_inches parameter for 
                                               plt.savefig(), e.g., 'tight'
                    ['datetime'] (bool)      : if True, figures are saved in a 
                                               subfolder named based on the date 
                                               and time.
                    ['fig_ext'] (str)        : extension (without '.') with 
                                               which to save figure
                    ['figdir_prel_roi'] (str): main folder in which to save figure
                    ['ncols'] (int)          : number of columns in the figure
                    ['overwrite'] (bool)     : if False, overwriting existing 
                                               figures is prevented by adding 
                                               suffix numbers.
                    ['prev_dt'] (str)        : datetime folder to use 
                                               default: None
                    ['sharey'] (bool)        : if True, y axis lims are shared 
                                               across subplots
                    ['subplot_wid'] (float)  : width of each subplot (inches)
                    ['subplot_hei'] (float)  : height of each subplot (inches)
                    ['surp_quint'] (str)     : specific subfolder in which to save 
                                               folder
            - perm_par (dict)    : dictionary containing permutation analysis 
                                parameters:
                    ['n_perms'] (int)     : nbr of permutations to run
                                            default: 10000
                    ['p_val'] (float)     : p-value to use for significance  
                                            thresholding (0 to 1)
                    ['tails'] (str or int): which tail(s) to test: 'up', 'lo', 2
                                            default: 2
            - roi_grp_par (dict) : dictionary containing ROI grouping parameters:
                    ['op'] (str)         : operation to use to compare groups, 
                                        i.e. 'diff': grp1-grp2, or 'ratio': 
                                        grp1/grp2
                                        default: 'diff'
                    ['plot_vals'] (str)  : values to plot 'diff' (surp-nosurp), 
                                        'surp' or 'nosurp'
                    ['grps'] (str)       : set of groups to return, e.g., 'all', 
                                        'change', 'no_change', 'reduc', 'incr'.
                                        If several sets are passed, only the
                                        first is plotted.
                                        default: 'all'
                    ['add_nosurp'] (bool): if True, group of ROIs showing no 
                                        significance in either is included in the 
                                        groups returned
                                        default: False         
            - sess_par (dict)  : dictionary containing session parameters:
                    ['gab_k'] (int or list) : gabor kappa parameter
                    ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                                'L5_soma', 'L23_dend', 'L5_dend', 
                                                'L23_all', 'L5_all')
                    ['overall_sess_n'] (int): overall session number aimed for
        """

        print(('\nAnalysing and plotting {} ROI surp vs nosurp responses by '
           'quintile ({}). \n{}.').format(op_par_str(roi_grp_par['plot_vals'],
                                                     roi_grp_par['op'], 'print'),
                                            analys_par['n_quints'],
                                            sess_par_str(sess_par, 
                                                         analys_par['gab_k'],
                                                         str_type='print')))

        # quintiles to plot
        qu = [0, -1]
        qu_lab = ['first quint', 'last_quint']
        # permutation test
        [_, integ_dffs_rel, n_rois] = integ_per_grp_qu_sess(sessions, analys_par, 
                                                            basic_par, roi_grp_par)
            
        # make sure only one set of ROI groups is passed
        roi_grp_par['grps'] = list_if_not(roi_grp_par['grps'])
        if len(roi_grp_par['grps']) > 1:
            roi_grp_par['grps'] = roi_grp_par['grps'][0]
        
        # identify significant ROIs
        all_roi_grps, grp_names = signif_rois_by_grp_sess(sessions, integ_dffs_rel, 
                                                        analys_par, basic_par, 
                                                        perm_par, roi_grp_par)
        
        # get statistics for each group
        # sess x surp x qu x ROI grps x stats x frames
        x_ran, grp_traces = grp_traces_by_qu_surp_sess(sessions, all_roi_grps,
                                                    analys_par, basic_par, 
                                                    roi_grp_par, qu)

        labels = plot_val_lab(roi_grp_par['plot_vals'], roi_grp_par['op'])
        xpos = [0.15, 0.45, 0.75, 1.05, 1.35]
        seg_bars = [0.3, 0.6, 0.9, 1.2]
        cols = ['steelblue', 'coral'] # for quintiles (e.g., [first, last])

        # Manual y_lims
        # y lims based on previous graphs
        if fig_par['preset_ylims']:
            if sess_par['layer'] == 'dend':
                ylims = [[-0.05, 0.2], [-0.2, 0.4], [-0.1, 0.25]] # per mouse
            elif sess_par['layer'] == 'soma':
                ylims = [[-0.3, 0.8], [-0.7, 1.0], [-0.15, 0.25]] # per mouse
            else:
                print('No ylims preset for {}.'.format(sess_par['layer']))

        # figure directories
        save_dir = '{}/{}'.format(fig_par['figdir_prel_roi'], fig_par['surp_quint'])
        fig_par['mult_figs'] = True

        for i in range(len(sessions)):
            fig, ax = init_fig(len(all_roi_grps[i]), fig_par)
            n = fig_par['ncols']
            for g, [grp_nam, grp_rois] in enumerate(zip(grp_names, all_roi_grps[i])):
                title = '{} group (n={})'.format(grp_nam, len(grp_rois))
                sub_ax = ax[g/n][g%n]
                if len(grp_rois) == 0:
                    sub_ax.set_title(title)
                    continue
                for j, q in enumerate(qu):
                    trace_data = np.concatenate([x_ran[np.newaxis, :], 
                                                grp_traces[i, q, g]], axis=0)
                    plot_traces(sub_ax, trace_data, stats=basic_par['stats'], 
                                error=basic_par['error'], title=title,
                                col=cols[j], alpha=0.8/len(qu))
                if fig_par['preset_ylims']:
                    sub_ax.set_ylim(ylims[i])
                add_bars(sub_ax, bars=seg_bars)
                add_labels(sub_ax, labels, xpos, t_hei=0.9, col='k')
                sub_ax.set_ylabel('dF/F')
                sub_ax.set_xlabel('Time (s)')
                sub_ax.legend(qu_lab)
            
            fig.suptitle(('Mouse {} - {} gab{} \n{} seqs \n for diff quint '
                        '(sess {}, {}, {} tail (n={}))')
                            .format(analys_par['mouse_ns'][i], 
                                    stat_par_str(basic_par['stats'], basic_par['error']),
                                    gab_k_par_str(analys_par['gab_k']), 
                                    op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'], 'print'),
                                    analys_par['act_sess_ns'][i], sess_par['layer'], 
                                    perm_par['tails'], n_rois[i]))

            save_name = ('roi_tr_m{}_{}_grps_{}_{}quint_'
                         '{}tail').format(analys_par['mouse_ns'][i], 
                                    sess_par_str(sess_par, analys_par['gab_k'], 'file'), 
                                    op_par_str(roi_grp_par['plot_vals'], 
                                               roi_grp_par['op'], 'file'),
                                    analys_par['n_quints'], perm_par['tails'])
            full_dir = save_fig(fig, save_dir, save_name, fig_par)

        fig_par['prev_dt'] = None

        info = {'sess_par': sess_par,
                'basic_par': basic_par,
                'analys_par': analys_par,
                'perm_par': perm_par,
                'roi_grp_par': roi_grp_par
                }
        
        info_name = ('roi_tr_{}_grps_{}_{}quint_'
                     '{}tail').format(sess_par_str(sess_par, analys_par['gab_k'], 'file'), 
                                      op_par_str(roi_grp_par['plot_vals'], 
                                                 roi_grp_par['op'], 'file'),
                                      analys_par['n_quints'], perm_par['tails'])

        save_info(info, full_dir, info_name)


#############################################
def plot_mag_change(sessions, analys_par, basic_par, fig_par, sess_par):
    """
    plot_mag_change(sessions, analys_par, basic_par, fig_par,  
                            perm_par, roi_grp_par, sess_par)

    Plots the magnitude of change in activity of ROIs between the first and
    last quintile for non surprise vs surprise segments.
    Saves parameter dictionaries relevant to analyses in a pickle.

    Required arguments:
        - sessions (list)  : list of Session objects
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['act_sess_ns'] (list)  : actual overall session number for
                                        each session
                ['gab_fr'] (int or list): gabor frame values to include
                                        (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list) : gabor kappa values to include 
                                        (e.g., 4, 16 or [4, 16])
                ['mouse_ns'] (list)     : mouse numbers corresponding to 
                                        sessions
                ['n_quints'] (int)      : number of quintiles
                ['pre'] (float)         : range of frames to include before each 
                                        frame reference (in s)
                ['post'] (float)        : range of frames to include after each 
                                        frame reference (in s)
                ['sess_ns'] (list)      : list of session IDs

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                                    default: 'std'
                ['rand'] (bool)   : if True, also includes statistics for a 
                                    random permutation of the traces (not 
                                    implemented).
                                    default: False
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'across', removes ROIs with NaN/Inf values 
                                    across subdivisions. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                                    default: 'per'
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'
                                    default: 'mean'
        - fig_par (dict)   : dictionary containing figure parameters:
                ['bbox'] (str)           : bbox_inches parameter for 
                                            plt.savefig(), e.g., 'tight'
                ['datetime'] (bool)      : if True, figures are saved in a 
                                            subfolder named based on the date 
                                            and time.
                ['fig_ext'] (str)        : extension (without '.') with 
                                            which to save figure
                ['figdir_prel_roi'] (str): main folder in which to save figure
                ['ncols'] (int)          : number of columns in the figure
                ['overwrite'] (bool)     : if False, overwriting existing 
                                            figures is prevented by adding 
                                            suffix numbers.
                ['prev_dt'] (str)        : datetime folder to use 
                                            default: None
                ['sharey'] (bool)        : if True, y axis lims are shared 
                                            across subplots
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['surp_quint'] (str)     : specific subfolder in which to save 
                                            folder      
        - sess_par (dict)  : dictionary containing session parameters:
                ['gab_k'] (int or list) : gabor kappa parameter
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                            'L5_soma', 'L23_dend', 'L5_dend', 
                                            'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    """
    # get sess x surp x quint x ROIs x frames
    chunk_info = chunk_stats_by_qu_sess(sessions, analys_par, basic_par, 
                                        byroi=True, data='me', bysurp=True,
                                        twop_fps=True)

    integ_data = integ_per_qu_surp_sess(chunk_info[1], chunk_info[-1], None)
    n_rois = [data.shape[2] for data in integ_data] # data is an array, not a list (ignore error)
    # get magnitude differences (magnitude of change for surprise vs non 
    # surprise segments)
    if basic_par['stats'] == 'median' and basic_par['error'] == 'std':
        stat_len = 2
    else:
        stat_len = 1
    mags = {'all_l2s': np.empty([len(sessions), 2]),
            'mag_me' : np.empty([len(sessions), 2]),
            'mag_de' : np.empty([len(sessions), 2, stat_len])
            }  
    
    for i in range(len(sessions)):
        for s in [0, 1]:
            # abs difference in average integrated areas across ROIs between last 
            # and first quintiles
            abs_diffs = np.absolute(integ_data[i][s, -1]-integ_data[i][s, 0])
            mags['all_l2s'][i, s] = np.linalg.norm(abs_diffs)

            mags['mag_me'][i, s] = mean_med(abs_diffs, basic_par['stats'])
            mags['mag_de'][i, s, 0:] = error_stat(abs_diffs, basic_par['stats'], 
                                                    basic_par['error'])
    
    print(('\nMagnitude in quintile difference per ROI per '
            'mouse ({}).').format(sess_par['layer']))
    for s, surp in enumerate(['non surprise', 'surprise']):
        l2_str = ('\n'.join(['\tMouse {}: {:.2f}'.format(i, l2) 
                                    for i, l2 in zip(analys_par['mouse_ns'], 
                                                    mags['all_l2s'][:, s])]))
        print('\n{} segs: \n{}'.format(surp, l2_str))

    # create figure
    barw = 0.15
    leg = ['nosurp', 'surp']
    col = ['steelblue', 'coral']
    fig, ax = plt.subplots()

    pos = np.arange(len(sessions))

    for s in range(len(leg)):
        xpos = pos + s*barw
        ax.bar(xpos, mags['mag_me'][:, s], width=barw, color=col[s], 
                yerr=mags['mag_de'][:, s], capsize=3)
    
    labels = ['Mouse {} (n={})'.format(analys_par['mouse_ns'][i], n_rois[i]) 
                        for i in range(len(sessions))]
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.legend(leg)
    ax.set_title(('Magnitude ({}) in quintile difference across ROIs '
                    'per mouse \n(sess {})').format(stat_par_str(basic_par['stats'], 
                            basic_par['error']), sess_par_str(sess_par, 
                            analys_par['gab_k'], 'print')))

    save_dir = '{}/{}'.format(fig_par['figdir_prel_roi'], fig_par['surp_quint'])
    save_name = ('roi_mag_diff_{}').format(sess_par_str(sess_par, analys_par['gab_k'], 'file'))
    full_dir = save_fig(fig, save_dir, save_name, fig_par)

    info = {'sess_par': sess_par,
            'basic_par': basic_par,
            'fig_par': fig_par,
            'analys_par': analys_par,
            'mags': mags
            }

    save_info(info, full_dir, save_name)


#############################################
def lfads_dict(sessions, mouse_df, gabfr, gabk=16):
    """
    lfads_dict(sessions, mouse_df, gabfr)

    Creates and saves dictionary containing information relevant to sessions.

    Arguments:
        - sessions (list)     : list of Session objects
        - mouse_df (pandas df): pandas dataframe with mouse sessions info
        - gabfr (int)         : gabor frame to include (e.g., A:0)

    Optional arguments:
        - gabk (int or list)  : gabor kappa values to include
    """
    
    all_gabfr = ['A', 'B', 'C', 'D/E']
    if isinstance(gabfr, list):
        gabfr_let = [all_gabfr[x] for x in gabfr]
    else:
        gabfr_let = all_gabfr[gabfr]
    sessions = list_if_not(sessions)
    for sess in sessions:
        sess.create_dff()
        frames = sess.gabors.get_2pframes_by_seg(sess.gabors
                        .get_segs_by_criteria(stimPar2=gabk, gaborframe=gabfr, 
                                            by='seg'), first=True)
        df_line = mouse_df.loc[(mouse_df['sessionid'] == sess.session)]
        act_n = df_line['overall_sess_n'].tolist()[0]
        depth = df_line['depth'].tolist()[0]
        mouse = df_line['mouseid'].tolist()[0]
        if depth in [20, 75]:
            layer = 'dend'
        elif depth in [175, 375]:
            layer = 'soma'
        
        sess_dict = {'sessionid'     : sess.session,
                     'mouse'         : mouse,
                     'act_sess_n'    : act_n,
                     'depth'         : depth,
                     'layer'         : layer,
                     'traces_dir'    : sess.roi_traces,
                     'dff_traces_dir': sess.roi_traces_dff,
                     'gab_k'         : gabk,
                     'gab_fr'        : [gabfr, gabfr_let], # e.g., [0, A]
                     'frames'        : frames,
                     'twop_fps'      : sess.twop_fps,
                    }
        
        name = 'sess_dict_mouse{}_sess{}_{}'.format(mouse, act_n, layer)
        save_info(sess_dict, 'session_dicts', name)
        print('Creating stimulus dictionary: {}'.format(name))

#############################################
def autocorr(data, lag_fr):
    """
    Calculates autocorrelation on data series.

    Arguments:
        - data (1D array): 1D dataseries
        - lag_fr (int)   : lag steps in frames
    
    Returns:
        - autoc_snip (1D array): 1D array of autocorrelations at specified lag
    """
    autoc = np.correlate(data, data, 'full')
    mid = int((autoc.shape[0]-1)/2)
    autoc_snip = autoc[mid-lag_fr:mid+lag_fr+1]
    autoc_snip /= np.max(autoc_snip)
    return autoc_snip


#############################################
def autocorr_rois(data, lag, fps=None, stats='mean', error='std'):
    """
    get_autocorr_rois(data, lag)
    
    Calculates average autocorrelation across data series.

    Arguments:
        - data (list or 1 or 2D array): list of series (1-2D array) or single 
                                        series. Autocorrelation is calculated 
                                        along last axis for each array.
                                        Each series can have 2D, e.g. ROI x frame
                                        and the number of ROIs in each series 
                                        must match.
        - lag (float)                 : lag in frames or in seconds if fps is 
                                        provided.
    Optional arguments:
        - axis (int) : axis along which to calculate autocorrelation
                       default: None
        - fps (float): fps value to calculate lag in frames
                       default: None
        - stats (str): statistic parameter, i.e. 'mean' or 'median'
                       default: 'mean'
        - error (str): error statistic parameter, i.e. 'std' or 'sem'
                       default: 'std
    Returns:
        - autocorr_stats (2 or 3D array): autocorr statistics, structured as 
                                          follows:
                                          (ROI x)
                                          stats (x_ran, mean/med, std/qu/sem/mad) x
                                          frame
    """
    
    if fps is not None:
        lag_fr = int(fps * lag)
    else:
        lag_fr = lag
    snip_len = 2*lag_fr+1

    data = list_if_not(data)
    if len(data[0].shape) == 1:
        mult = False
        autocorr_snips = np.empty((len(data), snip_len))
    elif len(data[0].shape) == 2:
        mult = True
        n_comp = data[0].shape[0]
        autocorr_snips = np.empty((n_comp, len(data), snip_len))

    for i, series in enumerate(data):
        if mult:
            norm_vals = series - np.mean(series, axis=1)[:, np.newaxis]
            for s, sub in enumerate(norm_vals):
                autocorr_snips[s, i] = autocorr(sub, lag_fr)
        else:
            norm_vals = series - np.mean(series)
            autocorr_snips[i] = autocorr(norm_vals, lag_fr)

    # average autocorrelations for each lag across blocks
    autocorr_me = mean_med(autocorr_snips, stats, axis=-2)
    autocorr_de = error_stat(autocorr_snips, stats, error, axis=-2)
    
    x_ran = np.linspace(-lag, lag, snip_len)

    if not mult:
        if not(stats == 'median' and error == 'std'):
            autocorr_de = autocorr_de[np.newaxis, :]
        autocorr_stats = np.concatenate([x_ran[np.newaxis, :], 
                                        autocorr_me[np.newaxis, :], 
                                        autocorr_de], axis=0)   
    else:
        if not(stats == 'median' and error == 'std'):
            autocorr_de = autocorr_de[:, np.newaxis, :]
        x_ran = np.repeat(x_ran[np.newaxis, np.newaxis, :], 
                          autocorr_me.shape[0], axis=0)
        autocorr_stats = np.concatenate([x_ran, autocorr_me[:, np.newaxis, :], 
                                        autocorr_de], axis=1)
    
    return autocorr_stats


#############################################
def plot_gab_autocorr(sessions, analys_par, basic_par, fig_par):
    """
    plot_mag_change(sessions, analys_par, basic_par, fig_par,  
                            perm_par, roi_grp_par, sess_par)

    Plots the magnitude of change in activity of ROIs between the first and
    last quintile for non surprise vs surprise segments.
    Saves parameter dictionaries relevant to analyses in a pickle.

    Required arguments:
        - sessions (list)  : list of Session objects
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['act_sess_ns'] (list)  : actual overall session number for
                                        each session
                ['gab_k'] (int or list) : gabor kappa values to include 
                                        (e.g., 4, 16 or [4, 16])
                ['lag_s'] (float)       : lag in seconds with which to calculate
                                          autocorrelation
                ['mouse_ns'] (list)     : mouse numbers corresponding to 
                                        sessions
                ['sess_ns'] (list)      : list of session IDs

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                                    default: 'std'
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'across', removes ROIs with NaN/Inf values 
                                    across subdivisions. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                                    default: 'per'
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'
                                    default: 'mean'
        - fig_par (dict)   : dictionary containing figure parameters:
                ['autocorr'] (str)       : specific subfolder in which to save 
                                            folder  
                ['bbox'] (str)           : bbox_inches parameter for 
                                            plt.savefig(), e.g., 'tight'
                ['datetime'] (bool)      : if True, figures are saved in a 
                                            subfolder named based on the date 
                                            and time.
                ['fig_ext'] (str)        : extension (without '.') with 
                                            which to save figure
                ['figdir_prel_roi'] (str): main folder in which to save figure
                ['ncols'] (int)          : number of columns in the figure
                ['overwrite'] (bool)     : if False, overwriting existing 
                                            figures is prevented by adding 
                                            suffix numbers.
                ['prev_dt'] (str)        : datetime folder to use 
                                            default: None
                ['sharey'] (bool)        : if True, y axis lims are shared 
                                            across subplots
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['subplot_hei'] (float)  : height of each subplot (inches)    
        - sess_par (dict)  : dictionary containing session parameters:
                ['gab_k'] (int or list) : gabor kappa parameter
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                            'L5_soma', 'L23_dend', 'L5_dend', 
                                            'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    """

    print(('\nAnalysing and plotting ROI autocorrelations ' 
          '({}).').format(sess_par_str(sess_par, analys_par['gab_k'], 
                                str_type='print')))

    fig, ax = init_fig(len(sessions), fig_par)
    n = fig_par['ncols']
    lag_s = analys_par['lag_s']
    xticks = np.linspace(-lag_s, lag_s, lag_s*4+1)
    yticks = np.linspace(0, 1, 6)
    seg_bars = [-1.5, 1.5] # light lines
    if basic_par['dfoverf']:
        title_str = 'dF/F autocorr'
    else:
        title_str = autocorr
    if basic_par['remnans'] == 'per':
        print('NaNs must be removed for across series, not per.')
        basic_par['remnans'] == 'across'
    nan_rois = []
    ok_rois = []
    for i, sess in enumerate(sessions):
        sub_ax = ax[i/n][i%n]
        all_segs = sess.gabors.get_segs_by_criteria(stimPar2=analys_par['gab_k'], 
                                                    by='block')
        sess_nans = []
        sess_ok = []
        sess_traces = []
        nrois = sess.nroi
        for segs in all_segs:
            if len(segs) == 0:
                continue
            # check that segs are contiguous
            if max(np.diff(segs)) > 1:
                raise ValueError('Segments used for autocorrelation are not '
                                 'contiguous.')
            frame_edges = sess.gabors.get_2pframes_by_seg([min(segs), max(segs)])
            frames = range(min(frame_edges[0]), max(frame_edges[1])+1)
            traces = sess.get_roi_traces(frames, dfoverf=basic_par['dfoverf'])
            if basic_par['remnans'] == ['across']:
                nans = []
                oks = range(nrois)
                if sum(sum(np.isnan(traces))) != 0:
                    nans = np.where(sum(np.isnan(traces)) != 0)[0].tolist()
                    oks = sorted(list(set(oks)-set(nans)))
                if basic_par['remnans'] == 'across':
                    sess_nans.extend(nans)
                    sess_ok.extend(oks)
            sess_traces.append(traces)
        if basic_par['remnans'] == 'across':
            sess_nans = sorted(list(set(sess_nans)))
            sess_oks = sorted(list(set(range(nrois))-set(sess_nans)))
            nrois = len(sess_oks)
            sess_traces = [traces[sess_oks] for traces in sess_traces]
            nan_rois.append(sess_nans)
            ok_rois.append(sess_oks)
        autocorr_stats = autocorr_rois(sess_traces, analys_par['lag_s'], sess.twop_fps, 
                                      basic_par['stats'], basic_par['error'])
        # add each ROI
        for roi_stats in autocorr_stats:
            plot_traces(sub_ax, roi_stats, basic_par['stats'], 
                        basic_par['error'], alpha=0.5/len(sessions), 
                        xticks=xticks, yticks=yticks)
        add_bars(sub_ax, bars=seg_bars)
        
        sub_ax.set_title(('Mouse {} - {} gab{} {}\n(sess {}, {}, '
                          '(n={}))').format(analys_par['mouse_ns'][i], 
                                            stat_par_str(basic_par['stats'], 
                                                         basic_par['error']),
                                            gab_k_par_str(analys_par['gab_k']), 
                                            title_str, analys_par['act_sess_ns'][i],
                                            sess_par['layer'], nrois))
        sub_ax.set_ylim([0, 1])
    
    if basic_par['remnans'] == 'across':
        analys_par['nan_rois'] = nan_rois
        analys_par['ok_rois'] = ok_rois    

    save_dir = '{}/{}'.format(fig_par['figdir_prel_roi'], fig_par['autocorr'])

    save_name = ('roi_autocorr_{}').format(sess_par_str(sess_par, analys_par['gab_k'], 'file'))

    full_dir = save_fig(fig, save_dir, save_name, fig_par)

    info = {'sess_par': sess_par,
            'basic_par': basic_par,
            'analys_par': analys_par,
            'fig_par': fig_par
            }

    save_info(info, full_dir, save_name)



if __name__ == "__main__":

    # set the main data directory (this needs to be changed by each user)
    maindir = '/media/colleen/LaCie/CredAssign/pilot_data'
    mouse_df_dir = 'mouse_df.pkl'

    # SPECIFIC PARAMETERS
    # session selection parameters (fixed keys)
    sess_par = {'layer'         : 'dend', # 'soma', 'dend', 'L23_soma', 'L5_soma', 
                                          # 'L23_dend', 'L5_dend', 'L23_all', 'L5_all'
                'overall_sess_n': 1, # 1 for first, etc. or 'last'
                'min_rois'      : 15, # min n of ROIs for inclusion
                }
    # additional analysis parameters
    analys_par = {'n_quints': 4, # min 1
                  'lag_s'   : 4, # sec
                 }
    # grouped ROI analysis parameters
    roi_grp_par = {'op'        : 'diff', # calculate 'diff' or 'ratio' of surp 
                                         # to nosurp 
                   'plot_vals' : 'nosurp', # plot 'diff' (surp-nosurp), 'surp' or 
                                         # 'nosurp'
                   'grps'      : 'all', # plot 'all' ROI grps or grps with 
                                        # 'change' or 'no_change'
                   'add_nosurp': True, # add nosurp_nosurp to ROI grp plots
                  }
    # permutation analysis parameters (fixed keys)
    perm_par = {'n_perms': 10000, # n of permutations for permutation analysis
                'p_val'   : 0.05, # p-value for permutation analysis
                'tails'   : 2 # 'up' (1 tail, upper), 'lo' (1 tail, lower) or 
                              # 2 (2 tailed test)
                }
    # output parameters for figures
    fig_par = {'figdir_prel_roi': 'figures/prelim_roi',
               'fig_ext'        : 'svg', # 'svg' or 'png'
               'surp_quint'     : 'surp_nosurp_quint', # subfolder
               'autocorr'       : 'autocorr', # subfolder
               'datetime'       : True, # create a datetime subfolder
               'overwrite'      : False, # allow previous subfolder to be overwritten
               'ncols'          : 2, # number of columns per figure
               'subplot_wid'    : 7.5,
               'subplot_hei'    : 7.5,
               'bbox'           : 'tight', # wrapping around figures
               'preset_ylims'   : True,
               'sharey'         : False # share y axis lims within figures
              }

    # TYPICALLY FIXED PARAMETERS
    # additional analysis parameters
    analys_par['gab_k']  = [4, 16] # kappa value(s) to use (either 4, 16 or [4, 16])
    analys_par['gab_fr'] = 3 # gabor frame to retrieve
    analys_par['pre']    = 0 # sec before frame
    analys_par['post']   = 1.5 # sec before frame

    # basic analysis parameters (fixed keys)
    basic_par = {'rand'   : False, # produce plots from randomized data 
                                   #(mostly not implemented yet)
                 'remnans': 'across', # remove ROIs containing NaNs or Infs
                 'dfoverf': True, # use dfoverf instead of raw ROI traces
                 'stats'  : 'mean', # plot mean or median
                 'error'  : 'sem', #'sem' for SEM/MAD, 'std' for std/qu
                 }
    # fairly fixed parameters (fixed keys)
    sess_par['omit_sess'] = 721038464 # alignment didn't work
    sess_par['pass_fail'] = 'P' # 'P' to only take passed sessions
    sess_par['omit_mice'] = gab_mice_omit(analys_par['gab_k'])
    
    # allows reusing datetime folder (if multiple figures created by one function)
    fig_par['prev_dt'] = None


    ### CODE STARTS HERE ###
    mouse_df = util.file_util.open_file(mouse_df_dir, 'pandas')

    # get session numbers
    [analys_par['sess_ns'], analys_par['mouse_ns'], 
            analys_par['act_sess_ns']] = sess_per_mouse(mouse_df, **sess_par)

    # create a dictionary with Session objects prepared for analysis
    sessions = init_sessions(analys_par['sess_ns'])

    # 0. Create dictionary including frame numbers for LFADS analysis
    # lfads_dict(sessions, mouse_df, 0, 16)
            

    # 1. Plot average traces by quintile x surprise for each session 
    # plot_traces_by_qu_surp_sess(sessions, analys_par, basic_par, fig_par, 
    #                             sess_par)
    
    # 2. Plot average dF/F area for each ROI group across quintiles for each session 
    # plot_rois_by_grp_qu_sess(sessions, analys_par, basic_par, fig_par, perm_par, 
    #                          roi_grp_par, sess_par)

    # 3. Plot average traces by suprise for each ROI group, for each session
    # plot_roi_traces_by_grp(sessions, analys_par, basic_par, fig_par, perm_par, 
    #                        roi_grp_par, sess_par)

    # 4. Plot magnitude of change in dF/F area from first to last quintile of 
    # surprise vs no surprise segments, for each session
    # plot_mag_change(sessions, analys_par, basic_par, fig_par, sess_par)

    # 5. Run autocorrelation analysis
    plot_gab_autocorr(sessions, analys_par, basic_par, fig_par)

