import os
import datetime
import argparse

import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
import pdb

from analysis import session
from util import file_util, gen_util, math_util, plot_util, str_util


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
        vals = gen_util.list_if_not(values)
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
    layer_dict = {'L23_dend': [50, 75],
                  'L23_soma': [175],
                  'L5_dend' : [20],
                  'L5_soma' : [375]
                 }

    if layer in ['L23_dend', 'L23_soma', 'L5_dend', 'L5_soma']:
        pass
    elif layer == 'dend':
        layer_dict['dend'] = layer_dict['L23_dend'] + layer_dict['L5_dend']
    elif layer == 'soma':
        layer_dict['soma'] = layer_dict['L23_soma'] + layer_dict['L5_soma']
    elif layer == 'L23_all':
        layer_dict['L23_all'] = layer_dict['L23_soma'] + layer_dict['L23_soma']
    elif layer == 'L5_all':
        layer_dict['L5_all'] = layer_dict['L5_soma'] + layer_dict['L5_soma']
    else:
        gen_util.accepted_values_error('layer', layer, ['L23_dend', 'L23_soma', 
                                       'L5_dend', 'L5_soma', 'dend', 'soma',  
                                       'L23_all', 'L5_all', 'any'])

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
        - depth (str or list)        : depth value(s) of interest (20, 50, 75,  
                                       175, 375)
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
                   all_files=[1], any_files=[1], overall_sess_n=1, 
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
                                       (default: [1])
        - any_files (int or list)    : any_files values to pick from (0, 1)
                                       (default: [1])
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
    sessid = gen_util.remove_if(sessid, omit_sess)
    
    # collect all mouse IDs and remove omitted mice
    mouseids = gen_util.remove_if(sorted(label_values(mouse_df, 'mouseid', 
                                                      values='any')), omit_mice)

    # get session ID, mouse ID and actual session numbers for each mouse based 
    # on criteria 
    sess_ns = []
    mouse_ns = []
    act_sess_ns = []
    lines = []
    for i in mouseids:
        sessions = sess_values(mouse_df, 'overall_sess_n', i, sessid, depth, 
                               pass_fail, all_files, any_files, 
                               overall_sess_any, min_rois, sort=True)
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
        line = mouse_df.loc[(mouse_df['sessionid'] == 
                              sess)]['line'].tolist()[0]

        sess_ns.append(sess)
        mouse_ns.append(i)
        act_sess_ns.append(act_n)
        lines.append(line)
    
    if len(sess_ns) == 0:
        raise ValueError('No sessions meet the criteria.')

    return sess_ns, mouse_ns, act_sess_ns, lines


#############################################
def init_sessions(sess_ns, datadir, runtype='prod'):
    """
    init_sess_dict(sess_ns)

    Creates list of Session objects for each session ID passed 

    Required arguments:
        - sess_ns (int or list): ID or list of IDs of sessions
        - datadir (str)        : directory where sessions are stored

    Optional arguments:
        - runtype (string): the type of run, either 'pilot' or 'prod'
                            default = 'prod'              
    Returns:
        - sessions (list): list of Session objects
    """

    sessions = []
    sess_ns = gen_util.list_if_not(sess_ns)
    for sess_n in sess_ns:
        print('\nCreating session {}...'.format(sess_n))
        # creates a session object to work with
        sess = session.Session(datadir, sess_n, runtype=runtype) 
        # extracts necessary info for analysis
        sess.extract_info()
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
    gab_k = gen_util.list_if_not(gab_k)
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
        gen_util.accepted_values_error('remnans', remnans, 
                                       ['per', 'across', 'no'])
    
    qu_stats = []
    for qu, segs in enumerate(qu_seg):
        print('\tQuintile {}'.format(qu+1))
        # get the stats for ROI traces for these segs 
        # returns x_ran, [mean/median, std/quartiles] for each ROI or across ROIs
        chunk_info = gabors.get_roi_chunk_stats(gabors.get_2pframes_by_seg(segs, 
                                                 first=True), 
                            pre, post, byroi=byroi, dfoverf=dfoverf, nans=nans, 
                            rand=False, stats=stats, error=error)
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
                    chunk_stats = [np.concatenate([roi[0][np.newaxis, :], 
                                                   roi[1]], axis=0) 
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
            gen_util.accepted_values_error('data', data, ['me', 'all'])
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
            qu_seg, qu_count = quint_segs(sess.gabors, analys_par, quint_info, 
                                          surp)
            sess_counts.append(qu_count)
            chunk_info = chunk_stats_by_qu(sess.gabors, qu_seg, 
                                           analys_par['pre'],
                                           analys_par['post'],
                                           byroi=byroi, data=data, **basic_par)
            x_ran = chunk_info[0]
            sess_stats.append(chunk_info[1])
            # if ROIs removed per subdivision
            if (basic_par['remnans'] == 'per' or 
                (basic_par['remnans'] == 'across' and not byroi)):
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
            # ROI dimension in array ((surp x) qu x ROI ...)
            roi_dim = 1+bysurp 
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
                ['lines'] (list)        : transgenic line for each session
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
                ['figdir_roi'] (str)     : main folder in which to save figure
                ['mult'] (bool)          : if True, prev_dt is created or used.
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
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                               'L5_soma', 'L23_dend', 'L5_dend', 
                                               'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    """
    gabkstr = str_util.gab_k_par_str(analys_par['gab_k'])
    statstr = str_util.stat_par_str(basic_par['stats'], basic_par['error'])
    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'])
    sessstr_pr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                       str_type='print')

    print(('\nAnalysing and plotting surprise vs non surprise ROI traces '
           'by quintile ({}) \n({}).').format(analys_par['n_quints'], 
                                              sessstr_pr))
    
    [xpos, labels_nosurp, h_bars, 
        seg_bars] = plot_util.plot_seg_comp(analys_par, 'nosurp')
    _, labels_surp, _, _ = plot_util.plot_seg_comp(analys_par, 'surp')

    t_heis = [0.85, 0.95]
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
    fig, ax, ncols, nrows = plot_util.init_fig(len(sessions), fig_par)

    for i, sess in enumerate(sessions):
        if nrows == 1:
            sub_ax = ax[i%ncols]
        else:
            sub_ax = ax[i/ncols][i%ncols]
        for s, [col, leg_ext] in enumerate(zip([col_nosurp, col_surp],
                                               ['nosurp', 'surp'])):
            for q in range(analys_par['n_quints']):
                if basic_par['remnans'] in ['per', 'across']:
                    n_rois = len(analys_par['ok_rois'][i][s][q])
                else:
                    n_rois = sess.nroi
                title=('Mouse {} - gab{} {} dF/F across gabor seqs\n(sess {}, '
                       '{} {}, n={})').format(analys_par['mouse_ns'][i], gabkstr, 
                                           statstr, analys_par['act_sess_ns'][i], 
                                           analys_par['lines'][i], 
                                           sess_par['layer'], n_rois)
                chunk_stats = np.concatenate([x_ran[np.newaxis, :], 
                                              all_stats[i][s][q]], axis=0)
                leg = '{}-{} ({})'.format(q+1, leg_ext, 
                                          analys_par['seg_per_quint'][i][s][q])
                plot_util.plot_traces(sub_ax, chunk_stats, 
                                      stats=basic_par['stats'], 
                                      error=basic_par['error'], col=col[q], 
                                      alpha=0.4/analys_par['n_quints'],
                                      title=title, label=leg,
                                      dff=basic_par['dfoverf'])
        for s, [lab, col, t_hei] in enumerate(zip([labels_nosurp, labels_surp], 
                                                  [col_nosurp, col_surp], 
                                                  t_heis)):
            plot_util.add_labels(sub_ax, lab, xpos, t_hei, col=col[0])
        plot_util.add_bars(sub_ax, hbars=h_bars, bars=seg_bars)
    
    save_dir = os.path.join(fig_par['figdir_roi'], fig_par['surp_quint'])
    save_name = 'roi_av_{}_{}quint'.format(sessstr, analys_par['n_quints'])
    full_dir = plot_util.save_fig(fig, save_dir, save_name, fig_par)

    info = {'sess_par': sess_par,
            'basic_par': basic_par,
            'analys_par': analys_par,
            }

    file_util.save_info(info, save_name, full_dir, 'json')


#############################################
def sep_grps(sign_rois, n_rois, grps='all', tails='2', add_nosurp=False):
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
                              if grp parameter includes only one set, 
                                  ROIs per roi group
                              otherwise: sets x roi grps
                              numbers included in the group
        - grp_names (list)  : if grp parameter includes only one set, list of 
                              names of roi grps (order preserved)
                              otherwise: list of sublists per set, each 
                              containing names of roi grps per set
    """
    grps = gen_util.list_if_not(grps)
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
                gen_util.accepted_values_error('grps', g, ['all', 'change', 
                                               'no_change', 'reduc', 'incr'])
            if add_nosurp and nosurp_ind not in grp_ind:
                grp_ind.extend([nosurp_ind])
            grp_inds.append(sorted(grp_ind))

    elif str(tails) == '2':
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
                gen_util.accepted_values_error('grps', grps, ['all', 'change', 
                                               'no_change', 'reduc', 'incr'])
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
def grp_stats(integ_data, grps, plot_vals='diff', op='diff', stats='mean', 
              error='std', norm=False):
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
        - plt_vals (str): 'surp', 'nosurp' or 'diff'
                           default: 'diff'
        - op (str)      : operation to use to compare groups, 
                          i.e. 'diff': grp1-grp2, or 'ratio': grp1/grp2
                          default: 'diff'
        - stats (str)   : statistic parameter, i.e. 'mean' or 'median'
                          default: 'mean'
        - error (str)   : error statistic parameter, i.e. 'std' or 'sem'
                          default: 'std'
       - norm (bool)    : if True, data is normalized by first quintile
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
    # array to collect group stats, structured as sessions x quintile x grp x 
    # stat
    all_grp_st = np.empty([len(integ_data), integ_data[0].shape[1], 
                           len(grps[0]), n_stats])
    # array to collect number of ROIs per group, structured as 
    # sessions x grp
    all_ns = np.empty([len(integ_data), len(grps[0])], dtype=int)
    for i, [sess_data, sess_grps] in enumerate(zip(integ_data, grps)):
        sess_data = math_util.calc_op(sess_data, plot_vals, op, surp_dim=0)
        # take mean/median and error for each group
        for g, grp in enumerate(sess_grps):
            all_ns[i, g] = len(grp)
            for q in range(len(sess_data)):
                all_grp_st[i, q, g] = np.nan
                if len(grp) != 0:
                    if norm:
                        grp_data = math_util.calc_norm(sess_data[:, grp], 
                                                       [0, 0], out_range='one')
                    else:
                        grp_data = sess_data[:, grp]
                    all_grp_st[i, q, g, 0] = math_util.mean_med(grp_data[q], 
                                                                stats)
                    all_grp_st[i, q, g, 1:] = np.asarray(math_util
                                                .error_stat(grp_data[q], stats, 
                                                            error))

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
        all_integ = math_util.integ(data, twop_fps[s], axis=-1)
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
    signif_rois_by_grp_qu_sess(sessions, analys_par, basic_par, fig_par, perm_par, 
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
                ['p_val'] (float)     : p-value to use for significance  
                                        thresholding (0 to 1)
                ['tails'] (str or int): which tail(s) to test: 'up', 'lo', '2'
                                        default: '2'
        - roi_grp_par (dict) : dictionary containing ROI grouping parameters:
                ['op'] (str)          : operation to use to compare groups, 
                                        i.e. 'diff': grp1-grp2, or 'ratio': 
                                        grp1/grp2
                                        default: 'diff'
                ['grps'] (str or list): set or sets of groups to return, 
                                        e.g., 'all', 'change', 'no_change', 
                                        'reduc', 'incr'.
                                        If several sets are passed, each set 
                                        will be collapsed as one group and
                                        'add_nosurp' will be set to False.
                                        default: 'all'
                ['add_nosurp'] (bool) : if True, group of ROIs showing no 
                                        significance in either is included in  
                                        the groups returned
                                        default: False   
    Returns:
        - roi_grps (dict): dictionary containing:
                ['all_roi_grps'] (list): list of sublists per session,  
                                         containing ROI numbers included in each  
                                         group, structured as follows:
                                            if sets of groups are passed: 
                                                session x set x roi_grp
                                            if one group is passed: 
                                                session x roi_grp
                ['grp_names'] (list)   : list of names of the ROI groups in roi 
                                         grp lists (order preserved)
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
                                analys_par['pre'], analys_par['post'], 
                                dfoverf=basic_par['dfoverf'])
            # remove previously removed ROIs if applicable 
            if basic_par['remnans'] == 'across':
                ok_rois    = analys_par['ok_rois'][s]
                roi_traces = roi_traces[ok_rois]
            # get area under the curve
            roi_dffs = math_util.integ(roi_traces, fps, axis=1)
            # run permutation test on dataset (dffs: ROI x seg)
            sign_rois = math_util.run_permute(roi_dffs, integ_dffs_rel[s][q], 
                            div=analys_par['seg_per_quint'][s][1][q], 
                            stats=basic_par['stats'], op=roi_grp_par['op'], 
                            **perm_par)
            all_rois.append(sign_rois)

        roi_grp_par['grps'] = gen_util.list_if_not(roi_grp_par['grps'])

        if len(roi_grp_par['grps']) == 1:
            roi_grps, grp_names = sep_grps(all_rois, n_rois=roi_dffs.shape[0], 
                                        grps=roi_grp_par['grps'], 
                                        tails=perm_par['tails'],
                                        add_nosurp=roi_grp_par['add_nosurp'])
        else:
            roi_grps = []
            for grp_set in roi_grp_par['grps']:
                roi_grps_set, _ = sep_grps(all_rois, n_rois=roi_dffs.shape[0], 
                                           grps=grp_set, 
                                           tails=perm_par['tails'],
                                           add_nosurp=False)
                # flat, without duplicates    
                flat_grp = sorted(list(set([roi for grp in roi_grps_set 
                                                for roi in grp])))
                roi_grps.append(flat_grp)
                
            grp_names = roi_grp_par['grps']
        
        all_roi_grps.append(roi_grps)

    roi_grp_dict = {'all_roi_grps': all_roi_grps,
                    'grp_names': grp_names,
                    }

    return roi_grp_dict

        
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
                ['lines'] (list)        : transgenic line for each session
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
                ['figdir_roi'] (str)     : main folder in which to save figure
                ['mult'] (bool)          : if True, prev_dt is created or used.
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
                ['tails'] (str or int): which tail(s) to test: 'up', 'lo', '2'
                                        default: '2'
        - roi_grp_par (dict) : dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
                                       default: 'diff'
                ['plot_vals'] (str)  : values to plot 'diff' (surp-nosurp), 
                                       'surp' or 'nosurp'
                ['grps'] (str)       : set of groups to return, e.g., 'all', 
                                       'change', 'no_change', 'reduc', 'incr'.
                                       If several sets are passed, each set 
                                        will be collapsed as one group and
                                        'add_nosurp' will be set to False.
                                       default: 'all'
                ['add_nosurp'] (bool): if True, group of ROIs showing no 
                                       significance in either is included in the 
                                       groups returned
                                       default: False         
        - sess_par (dict)  : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                               'L5_soma', 'L23_dend', 'L5_dend', 
                                               'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    """

    plotvalstr_pr = str_util.op_par_str(roi_grp_par['plot_vals'], 
                                        roi_grp_par['op'], True, 'print')
    sessstr_pr = str_util.sess_par_str(sess_par, analys_par['gab_k'],
                                       str_type='print')
    opstr_pr = str_util.op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'], 
                                   True, 'print')
    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'])
    statstr = str_util.stat_par_str(basic_par['stats'], basic_par['error'])
    gabkstr = str_util.gab_k_par_str(analys_par['gab_k'])

    print(('\nAnalysing and plotting {} ROI surp vs nosurp responses by '
           'quintile ({}). \n{}.').format(plotvalstr_pr, analys_par['n_quints'],
                                          sessstr_pr))
    
    # permutation test
    [integ_dffs, integ_dffs_rel, 
                 n_rois] = integ_per_grp_qu_sess(sessions, analys_par,  
                                                 basic_par, roi_grp_par)

    # identify significant ROIs
    roi_grps = signif_rois_by_grp_sess(sessions, integ_dffs_rel, analys_par, 
                                       basic_par, perm_par, roi_grp_par)
    
    # get statistics per group and number of ROIs per group
    grp_st, ns = grp_stats(integ_dffs, roi_grps['all_roi_grps'], 
                           roi_grp_par['plot_vals'], roi_grp_par['op'], 
                           basic_par['stats'], basic_par['error'])

    x_ran = [x+1 for x in range(analys_par['n_quints'])]
    fig, ax, ncols, nrows = plot_util.init_fig(len(sessions), fig_par)
    
    for i, sess_st in enumerate(grp_st):
        if nrows == 1:
            sub_ax = ax[i%ncols]
        else:
            sub_ax = ax[i/ncols][i%ncols]
        for g, g_n in enumerate(ns[i]):
            me = sess_st[:, g, 0]
            if basic_par['stats'] == 'median' and basic_par['error'] == 'std':
                yerr1 = me - sess_st[:, g, 1]
                yerr2 = sess_st[:, g, 2] - me
                yerr = [yerr1, yerr2]
            else:
                yerr = sess_st[:, g, 1]
            leg = '{} ({})'.format(roi_grps['grp_names'][g], g_n)
            sub_ax.errorbar(x_ran, me, yerr, fmt='-o', capsize=4, capthick=2, 
                            label=leg)

        title=('Mouse {} - {} gab{} \n{} seqs \n(sess {}, {} {}, {} tail '
               '(n={}))').format(analys_par['mouse_ns'][i], statstr, gabkstr, 
                                 opstr_pr, analys_par['act_sess_ns'][i],
                                 analys_par['lines'][i], sess_par['layer'], 
                                 perm_par['tails'], n_rois[i])
        sub_ax.set_title(title)
        sub_ax.set_xticks(x_ran)
        sub_ax.set_ylabel('dF/F')
        sub_ax.set_xlabel('Quintiles')

    save_dir = os.path.join(fig_par['figdir_roi'], fig_par['surp_quint'])
    save_name = 'roi_{}_grps_{}_{}quint_{}tail'.format(sessstr, 
                    roi_grp_par['plot_vals'], analys_par['n_quints'], 
                    perm_par['tails'])
    full_dir = plot_util.save_fig(fig, save_dir, save_name, fig_par)

    info = {'sess_par': sess_par,
            'basic_par': basic_par,
            'analys_par': analys_par,
            'perm_par': perm_par,
            'roi_grp_par': roi_grp_par,
            'roi_grps': roi_grps
            }
    
    file_util.save_info(info, save_name, full_dir, 'json')


#############################################
def grp_traces_by_qu_surp_sess(sessions, all_roi_grps, analys_par, basic_par, 
                               roi_grp_par, quint_ns):
    """
    grp_traces_by_qu_surp_sess(sessions, all_roi_grps, analys_par, basic_par, 
                               roi_grp_par, quint_ns)
                               
    Required arguments:
        - sessions (list)     : list of Session objects
        - all_roi_grps (list) : list of sublists per session, each containing
                               sublists per roi grp with ROI numbers included in 
                               the group: session x roi_grp
        - analys_par (dict)   : dictionary containing relevant parameters
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
        - basic_par (dict)    : dictionary containing additional parameters 
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
        - roi_grp_par (dict)   : dictionary containing ROI grouping parameters:
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
    data_me = [math_util.calc_op(x, roi_grp_par['plot_vals'], roi_grp_par['op'], 
                       surp_dim=0) for x in chunk_me]

    if basic_par['stats'] and basic_par['error']:
        n_stats = 3
    else:
        n_stats = 2
    
    # sess x quintile (first/last) x ROI grp x stats
    grp_stats = np.empty([len(sessions), len(quint_ns), len(all_roi_grps[0]), 
                          n_stats, len(x_ran)]) * np.nan

    for i, sess in enumerate(data_me):
        for q, quint in enumerate(sess): 
            for g, grp_rois in enumerate(all_roi_grps[i]):
                # leave NaNs if no ROIs in group
                if len(grp_rois) == 0:
                    continue
                grp_stats[i, q, g, 0] = math_util.mean_med(quint[grp_rois], 
                                                 basic_par['stats'], axis=0)
                grp_stats[i, q, g, 1:] = np.asarray(math_util.error_stat(quint[grp_rois], 
                                                    basic_par['stats'], 
                                                    basic_par['error'], axis=0))
    return x_ran, grp_stats


#############################################
def plot_roi_traces_by_grp(sessions, quint_plot, roi_grps, n_rois, 
                           analys_par, basic_par, fig_par, roi_grp_par, 
                           sess_par, save_dict=True):
    """
    plot_roi_traces_by_grp(sessions, quint_plot, roi_grps, n_rois, 
                           analys_par, basic_par, fig_par, roi_grp_par, sess_par)

    Plots ROI traces by group across surprise, no surprise or difference between 
    surprise and no surprise activity per quintile (first/last) with each group 
    in a separate subplot and each session in a different figure.

    Optionally saves parameter dictionaries relevant to analyses in a pickle.

    Required arguments:
        - sessions (list)  : list of Session objects
        - quint_plot (dict): dictionary containing information on quintiles to
                             plot:
                ['qu'] (list)    : list of quintile indices to plot,
                ['qu_lab'] (list): list of quintile labels,
                ['cols'] (list)  : list of quintile colors
        - roi_grps (dict)  : dictionary containing ROI groups:
                ['all_roi_grps'] (list): list of sublists per session,  
                                         containing ROI numbers included in each  
                                         group, structured as follows:
                                            if sets of groups are passed: 
                                                session x set x roi_grp
                                            if one group is passed: 
                                                session x roi_grp
                ['grp_names'] (list)   : list of names of the ROI groups in roi 
                                         grp lists (order preserved)
        - n_rois (1D array): number of ROIs retained in each session
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['act_sess_ns'] (list)  : actual overall session number for
                                          each session
                ['gab_fr'] (int or list): gabor frame values to include
                                          (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list) : gabor kappa values to include 
                                          (e.g., 4, 16 or [4, 16])
                ['lines'] (list)        : transgenic line for each session
                ['mouse_ns'] (list)     : mouse numbers corresponding to 
                                          sessions
                ['n_quints'] (int)      : number of quintiles
                ['pre'] (float)         : range of frames to include before each 
                                          frame reference (in s)
                ['post'] (float)        : range of frames to include after each 
                                          frame reference (in s)
                ['sess_ns'] (list)      : list of session IDs

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                                    default: 'std'
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
                ['figdir_roi'] (str)     : main folder in which to save figure
                ['mult'] (bool)          : if True, prev_dt is created or used.
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
        - roi_grp_par (dict): dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
                                       default: 'diff'
                ['plot_vals'] (str)  : values to plot 'diff' (surp-nosurp), 
                                       'surp' or 'nosurp'     
        - sess_par (dict)   : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                          'L5_soma', 'L23_dend', 'L5_dend', 
                                          'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    
    Optional arguments:
        - save_dict (bool): if True, dictionaries containing parameters used
                            for analysis are saved to pickle (sess_par, 
                            basic_par, analys_par, perm_par, roi_grp_par, 
                            roi_grps.
    Returns:
        - full_dir (str): final name of the directory in which the figures are 
                          saved 
    """
    opstr_pr = str_util.op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'], 
                                   'print')
    sessstr_pr = str_util.sess_par_str(sess_par, analys_par['gab_k'],
                                       str_type='print')
    statstr = str_util.stat_par_str(basic_par['stats'], basic_par['error'])
    gabkstr = str_util.gab_k_par_str(analys_par['gab_k'])
    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'])
    opstr = str_util.op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'])

    print(('\nAnalysing and plotting {} ROI surp vs nosurp responses by '
           'quintile ({}). \n{}.').format(opstr_pr, analys_par['n_quints'], 
                                          sessstr_pr))

    # get statistics for each group
    # sess x surp x qu x ROI grps x stats x frames
    x_ran, grp_traces = grp_traces_by_qu_surp_sess(sessions, 
                                                   roi_grps['all_roi_grps'],
                                                   analys_par, basic_par, 
                                                   roi_grp_par, quint_plot['qu'])

    xpos, labels, h_bars, seg_bars = plot_util.plot_seg_comp(analys_par, 
                                                   roi_grp_par['plot_vals'], 
                                                   roi_grp_par['op'])

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
    save_dir = os.path.join(fig_par['figdir_roi'], fig_par['surp_quint'])

    if fig_par['mult']:
        reset_mult = False
    else:
        fig_par['mult'] = True
        reset_mult = True

    for i in range(len(sessions)):
        fig, ax, ncols, nrows = plot_util.init_fig(len(roi_grps['all_roi_grps'][i]), fig_par)
        for g, [grp_nam, grp_rois] in enumerate(zip(roi_grps['grp_names'], 
                                                    roi_grps['all_roi_grps'][i])):
            title = '{} group (n={})'.format(grp_nam, len(grp_rois))
            if nrows == 1:
                sub_ax = ax[g%ncols]
            else:
                sub_ax = ax[g/ncols][g%ncols]
            if len(grp_rois) == 0:
                sub_ax.set_title(title)
                continue
            for j, q in enumerate(quint_plot['qu']):
                trace_data = np.concatenate([x_ran[np.newaxis, :], 
                                            grp_traces[i, q, g]], axis=0)
                plot_util.plot_traces(sub_ax, trace_data, stats=basic_par['stats'], 
                            error=basic_par['error'], title=title,
                            col=quint_plot['cols'][j], 
                            alpha=0.8/len(quint_plot['qu']), 
                            dff=basic_par['dfoverf'])
            if fig_par['preset_ylims']:
                sub_ax.set_ylim(ylims[i])
            plot_util.add_bars(sub_ax, hbars=h_bars, bars=seg_bars)
            plot_util.add_labels(sub_ax, labels, xpos, t_hei=0.9, col='k')
            sub_ax.set_ylabel('dF/F')
            sub_ax.set_xlabel('Time (s)')
            sub_ax.legend(quint_plot['qu_lab'])
        
        fig.suptitle(('Mouse {} - {} gab{} \n{} seqs for diff quint\n'
                    '(sess {}, {} {}, {} tail (n={}))')
                        .format(analys_par['mouse_ns'][i], statstr, gabkstr, 
                                opstr_pr, analys_par['act_sess_ns'][i], 
                                analys_par['lines'][i], sess_par['layer'], 
                                perm_par['tails'], n_rois[i]))

        save_name = ('roi_tr_m{}_{}_grps_{}_{}quint_'
                        '{}tail').format(analys_par['mouse_ns'][i], sessstr, 
                                         opstr, analys_par['n_quints'], 
                                         perm_par['tails'])
        full_dir = plot_util.save_fig(fig, save_dir, save_name, fig_par)

    # resetting the fig_par
    if reset_mult:
        fig_par['mult'] = False
        fig_par['prev_dt'] = None
    
    if save_dict:
        info = {'sess_par': sess_par,
                'basic_par': basic_par,
                'analys_par': analys_par,
                'perm_par': perm_par,
                'roi_grp_par': roi_grp_par,
                'roi_grps': roi_grps,
                }
        
        info_name = ('roi_tr_{}_grps_{}_{}quint_'
                        '{}tail').format(sessstr, opstr, analys_par['n_quints'], 
                                         perm_par['tails'])

        file_util.save_info(info, info_name, full_dir, 'json')

    return full_dir


#############################################
def plot_roi_areas_by_grp(sessions, integ_dffs, quint_plot, roi_grps, n_rois,
                          analys_par, basic_par, fig_par, roi_grp_par, sess_par,
                          save_dict=False):
    """
    plot_roi_traces_by_grp(sessions, integ_dffs, quint_plot, roi_grps, 
                           analys_par, basic_par, fig_par, roi_grp_par, sess_par)

    Plots ROI traces by group across surprise, no surprise or difference between 
    surprise and no surprise activity per quintile (first/last) with each group 
    in a separate subplot and each session in a different figure. Saves 
    dictionary containing group statistics as a pickle.

    Optionally also saves parameter dictionaries relevant to analyses in a pickle.

    Required arguments:
        - sessions (list)  : list of Session objects
        - integ_dffs (list): list of 3D arrays of mean/medians integrated 
                             across chunks, for each session:
                                 surp if bysurp x
                                 quintiles x
                                 ROIs
        - quint_plot (dict): dictionary containing information on quintiles to
                             plot:
                ['qu'] (list)    : list of quintile indices to plot,
                ['qu_lab'] (list): list of quintile labels,
                ['cols'] (list)  : list of quintile colors
        - roi_grps (dict)  : dictionary containing ROI groups:
                ['all_roi_grps'] (list): list of sublists per session,  
                                         containing ROI numbers included in each  
                                         group, structured as follows:
                                            if sets of groups are passed: 
                                                session x set x roi_grp
                                            if one group is passed: 
                                                session x roi_grp
                ['grp_names'] (list)   : list of names of the ROI groups in roi 
                                         grp lists (order preserved)
        - n_rois (1D array): number of ROIs retained in each session
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['act_sess_ns'] (list)  : actual overall session number for
                                          each session
                ['gab_fr'] (int or list): gabor frame values to include
                                          (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list) : gabor kappa values to include 
                                          (e.g., 4, 16 or [4, 16])
                ['lines'] (list)        : transgenic line for each session
                ['mouse_ns'] (list)     : mouse numbers corresponding to 
                                          sessions
                ['n_quints'] (int)      : number of quintiles
                ['pre'] (float)         : range of frames to include before each 
                                          frame reference (in s)
                ['post'] (float)        : range of frames to include after each 
                                          frame reference (in s)
                ['sess_ns'] (list)      : list of session IDs
        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                                    default: 'std'
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
                ['figdir_roi'] (str)     : main folder in which to save figure
                ['mult'] (bool)          : if True, prev_dt is created or used.
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
        - roi_grp_par (dict): dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
                                       default: 'diff'
                ['plot_vals'] (str)  : values to plot 'diff' (surp-nosurp), 
                                      'surp' or 'nosurp' 
        - sess_par (dict)  : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                          'L5_soma', 'L23_dend', 'L5_dend', 
                                          'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    
    Optional arguments:
        - save_dict (bool): if True, dictionaries containing parameters used
                            for analysis are saved to pickle (sess_par, 
                            basic_par, analys_par, perm_par, roi_grp_par, 
                            roi_grps.
    
    Returns:
        - full_dir (str): final name of the directory in which the figures are 
                          saved 
    """
    opstr_pr = str_util.op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'], 
                                   True, 'print')
    sessstr_pr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                       str_type='print')
    statstr = str_util.stat_par_str(basic_par['stats'], basic_par['error'])
    gabkstr = str_util.gab_k_par_str(analys_par['gab_k'])
    opstr = str_util.op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'], 
                                True)
    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'])

    print(('\nAnalysing and plotting {} ROI surp vs nosurp responses by '
        'quintile ({}). \n{}.').format(opstr_pr, analys_par['n_quints'],
                                       sessstr_pr))
    
    # get statistics per group and number of ROIs per group
    grp_st, _ = grp_stats(integ_dffs, roi_grps['all_roi_grps'], 
                          roi_grp_par['plot_vals'], roi_grp_par['op'], 
                          basic_par['stats'], basic_par['error'])
    # sess x quint x grp x stat
    grp_st_norm, _ = grp_stats(integ_dffs, roi_grps['all_roi_grps'], 
                               roi_grp_par['plot_vals'], roi_grp_par['op'], 
                               basic_par['stats'], basic_par['error'], True)

    # figure directories
    save_dir = os.path.join(fig_par['figdir_roi'], fig_par['surp_quint'])
    
    if fig_par['mult']:
        reset_mult = False
    else:
        fig_par['mult'] = True
        reset_mult = True
    
    xpos = range(len(quint_plot['qu']))
    
    for i in range(len(sessions)):
        fig, ax, _, _ = plot_util.init_fig(len(roi_grps['all_roi_grps'][i]), fig_par)
        fignorm, axnorm, ncols, nrows = plot_util.init_fig(len(roi_grps['all_roi_grps'][i]), fig_par)
        for axis, norm in zip([ax, axnorm], [False, True]):
            for g, [grp_nam, grp_rois] in enumerate(zip(roi_grps['grp_names'], 
                                                        roi_grps['all_roi_grps'][i])):
                title = '{} group (n={})'.format(grp_nam, len(grp_rois))
                if nrows == 1:
                    sub_ax = axis[g%ncols]
                else:
                    sub_ax = axis[g/ncols][g%ncols]
                if len(grp_rois) == 0:
                    if not norm:
                        sub_ax.set_title(title)
                    else:
                        sub_ax.set_title('{} (norm)'.format(title))
                    continue
                for j, q in enumerate(quint_plot['qu']):
                    if not norm:
                        sub_ax.bar(xpos[j], grp_st[i, q, g, 0], width=0.15, 
                            color=quint_plot['cols'][j], 
                            yerr=grp_st[i, q, g, 1:], capsize=3)
                    else:
                        sub_ax.bar(xpos[j], grp_st_norm[i, q, g, 0], width=0.15, 
                            color=quint_plot['cols'][j], 
                            yerr=grp_st[i, q, g, 1:], capsize=3)                    
                sub_ax.legend(quint_plot['qu_lab'])
                if not norm:
                    sub_ax.set_ylabel('dF/F area')
                else:
                    sub_ax.set_ylabel('dF/F area (norm)')
                sub_ax.set_xticks([0, 1])
                sub_ax.set_title(title)

        suptitle = ('Mouse {} - {} gab{} \n{} seqs for diff quint\n(sess {}, '
                    '{} {}, {} tail (n={}))').format(analys_par['mouse_ns'][i], 
                                                  statstr, gabkstr, opstr_pr,
                                                  analys_par['act_sess_ns'][i], 
                                                  analys_par['lines'][i], 
                                                  sess_par['layer'], 
                                                  perm_par['tails'], n_rois[i])

        fig.suptitle(suptitle)
        fignorm.suptitle('{} (norm)'.format(suptitle))

        save_name = ('roi_area_m{}_{}_grps_{}_{}quint_'
                        '{}tail').format(analys_par['mouse_ns'][i], sessstr, 
                                         opstr, analys_par['n_quints'], 
                                         perm_par['tails'])
        save_name_norm = '{}_norm'.format(save_name)

        full_dir = plot_util.save_fig(fig, save_dir, save_name, fig_par)
        _ = plot_util.save_fig(fignorm, save_dir, save_name_norm, fig_par)

    if reset_mult:
        fig_par['mult'] = False
        fig_par['prev_dt'] = None

    info_name = ('roi_area_{}_grps_{}_{}quint_'
                        '{}tail').format(sessstr, opstr, analys_par['n_quints'], 
                                         perm_par['tails'])

    if save_dict:
        info = {'sess_par': sess_par,
                'basic_par': basic_par,
                'analys_par': analys_par,
                'perm_par': perm_par,
                'roi_grp_par': roi_grp_par,
                'roi_grps': roi_grps,
                }

        file_util.save_info(info, info_name, full_dir, 'json')

    data = {'grp_st': grp_st.tolist(),
            'grp_st_norm': grp_st_norm.tolist(),
            'grp_names': roi_grps['grp_names']
            }

    data_name = '{}_data'.format(info_name)
    file_util.save_info(data, data_name, full_dir, 'json')
    
    return full_dir


#############################################
def plot_rois_by_grp(sessions, analys_par, basic_par, fig_par, perm_par, 
                     roi_grp_par, sess_par):
    """
    plot_rois_by_grp(sessions, analys_par, basic_par, fig_par, perm_par, 
                    roi_grp_par)

    Identify ROIs showing significant surprise in first and/or last quintile,
    group accordingly and plot traces and areas across surprise, no surprise or 
    difference between surprise and no surprise activity per quintile 
    (first/last) with each group in a separate subplot and each session
    in a different figure. Saves dictionary containing group statistics as a
    pickle and parameter dictionaries relevant to analyses in another pickle.

    Required arguments:
        - sessions (list)  : list of Session objects
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['act_sess_ns'] (list)  : actual overall session number for
                                          each session
                ['gab_fr'] (int or list): gabor frame values to include
                                          (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list) : gabor kappa values to include 
                                          (e.g., 4, 16 or [4, 16])
                ['lines'] (list)        : transgenic line for each session
                ['mouse_ns'] (list)     : mouse numbers corresponding to 
                                          sessions
                ['n_quints'] (int)      : number of quintiles
                ['pre'] (float)         : range of frames to include before each 
                                          frame reference (in s)
                ['post'] (float)        : range of frames to include after each 
                                          frame reference (in s)
                ['sess_ns'] (list)      : list of session IDs
        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                                    default: 'std'
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
                ['figdir_roi'] (str)     : main folder in which to save figure
                ['mult'] (bool)          : if True, prev_dt is created or used.
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
        - roi_grp_par (dict): dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
                                       default: 'diff'
                ['plot_vals'] (str)  : values to plot 'diff' (surp-nosurp), 
                                      'surp' or 'nosurp' 
        - sess_par (dict)  : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                          'L5_soma', 'L23_dend', 'L5_dend', 
                                          'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    """
    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 'file')
    opstr = str_util.op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'], 
                                str_type='file')

    # quintiles to plot
    quint_plot = {'qu': [0, -1], # must correspond to indices
                  'qu_lab': ['first quint', 'last quint'],
                  'cols': ['steelblue', 'coral']
                 }
    
    [integ_dffs, integ_dffs_rel, n_rois] = integ_per_grp_qu_sess(sessions, analys_par, 
                                                        basic_par, roi_grp_par)
    
    # permutation test 
    roi_grps = signif_rois_by_grp_sess(sessions, integ_dffs_rel, analys_par, 
                                       basic_par, perm_par, roi_grp_par)

    # ensure that plots are all saved in same file
    if fig_par['mult']:
        reset_mult = False
    else:
        fig_par['mult'] = True
        reset_mult = True

    _ = plot_roi_traces_by_grp(sessions, quint_plot, roi_grps, 
                               n_rois, analys_par, basic_par, fig_par, 
                               roi_grp_par, sess_par, save_dict=False)

    full_dir = plot_roi_areas_by_grp(sessions, integ_dffs, quint_plot, roi_grps, 
                                     n_rois, analys_par, basic_par, fig_par, 
                                     roi_grp_par, sess_par, save_dict=False)

    # resetting the fig_par
    if reset_mult:
        fig_par['mult'] = False
        fig_par['prev_dt'] = None

    info = {'sess_par': sess_par,
            'basic_par': basic_par,
            'analys_par': analys_par,
            'perm_par': perm_par,
            'roi_grp_par': roi_grp_par,
            'roi_grps': roi_grps,
            }
    
    info_name = ('roi_{}_grps_{}_{}quint_'
                    '{}tail').format(sessstr, opstr, analys_par['n_quints'], 
                                     perm_par['tails'])

    file_util.save_info(info, info_name, full_dir, 'json')


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
                ['lines'] (list)        : transgenic line for each session
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
                ['figdir_roi'] (str)     : main folder in which to save figure
                ['ncols'] (int)          : number of columns in the figure
                ['overwrite'] (bool)     : if False, overwriting existing 
                                            figures is prevented by adding 
                                            suffix numbers.
                ['mult'] (bool)          : if True, prev_dt is created or used.
                ['prev_dt'] (str)        : datetime folder to use 
                                            default: None
                ['sharey'] (bool)        : if True, y axis lims are shared 
                                            across subplots
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['surp_quint'] (str)     : specific subfolder in which to save 
                                            folder      
        - sess_par (dict)  : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                            'L5_soma', 'L23_dend', 'L5_dend', 
                                            'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    """
    statstr = str_util.stat_par_str(basic_par['stats'], basic_par['error'])
    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'])
    sessstr_pr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 'print')

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
            'mag_me_norm' : np.empty([len(sessions), 2]),
            'mag_de' : np.empty([len(sessions), 2, stat_len]).squeeze(),
            'mag_de_norm' : np.empty([len(sessions), 2, stat_len]).squeeze(),
            'p_vals': np.empty([len(sessions)])
            }  

    if fig_par['mult']:
        reset_mult = False
    else:
        fig_par['mult'] = True
        reset_mult = True

    for i in range(len(sessions)):
        abs_diffs = np.absolute(integ_data[i][:, -1] - integ_data[i][:, 0])
        abs_diffs_norm = math_util.calc_norm(abs_diffs, dimpos=[0, 0], out_range='onepos')
        for s in [0, 1]:
            # abs difference in average integrated areas across ROIs between last 
            # and first quintiles
            mags['all_l2s'][i, s] = np.linalg.norm(abs_diffs[s])
            
            mags['mag_me'][i, s] = math_util.mean_med(abs_diffs[s], basic_par['stats'])
            mags['mag_de'][i, s] = math_util.error_stat(abs_diffs[s], basic_par['stats'], 
                                                    basic_par['error'])
            mags['mag_me_norm'][i, s] = math_util.mean_med(abs_diffs_norm[s], 
                                                 basic_par['stats'])
            mags['mag_de_norm'][i, s] = math_util.error_stat(abs_diffs_norm[s], 
                                                       basic_par['stats'], 
                                                       basic_par['error'])
        # p_val test on surp vs nosurp for each session
        _, mags['p_vals'][i] = st.ttest_rel(abs_diffs[0], abs_diffs[1])
        print('Session {}: p-value={}'.format(sessions[i].session, 
                                              mags['p_vals'][i]))
    
    print(('\nMagnitude in quintile difference per ROI per '
            'mouse ({}).').format(sess_par['layer']))
    for s, surp in enumerate(['non surprise', 'surprise']):
        l2_str = ('\n'.join(['\tMouse {}, {}: {:.2f}'.format(i, l, l2) 
                                    for i, l, l2 in zip(analys_par['mouse_ns'],
                                                     analys_par['lines'],  
                                                    mags['all_l2s'][:, s])]))
        print('\n{} segs: \n{}'.format(surp, l2_str))

    # create figure
    barw = 0.15
    leg = ['nosurp', 'surp']
    col = ['steelblue', 'coral']
    fig, ax = plt.subplots()
    fignorm, axnorm = plt.subplots()

    pos = np.arange(len(sessions))

    for s in range(len(leg)):
        xpos = pos + s*barw
        ax.bar(xpos, mags['mag_me'][:, s], width=barw, color=col[s], 
                yerr=mags['mag_de'][:, s], capsize=3)
        axnorm.bar(xpos, mags['mag_me_norm'][:, s], width=barw, color=col[s], 
                   yerr=mags['mag_de_norm'][:, s], capsize=3)
    
    labels = ['Mouse {},{}\n(n={})'.format(analys_par['mouse_ns'][i], 
                                           analys_par['lines'][i], n_rois[i]) 
                        for i in range(len(sessions))]
    
    for axis in [ax, axnorm]:
        axis.set_xticks(pos)
        axis.set_xticklabels(labels)
        axis.legend(leg)

    title = (('Magnitude ({}) in quintile difference across ROIs '
              'per mouse \n(sess {})').format(statstr, sessstr_pr))
    ax.set_title(title)
    axnorm.set_title('{} (norm)'.format(title))

    save_dir = os.path.join(fig_par['figdir_roi'], fig_par['surp_quint'])
    save_name = ('roi_mag_diff_{}').format(sessstr)
    save_name_norm = '{}_norm'.format(save_name)

    full_dir = plot_util.save_fig(fig, save_dir, save_name, fig_par)
    _ = plot_util.save_fig(fignorm, save_dir, save_name_norm, fig_par)

    # resetting the fig_par
    if reset_mult:
        fig_par['mult'] = False
        fig_par['prev_dt'] = None

    # convert mags items to lists
    for key in mags.keys():
        mags[key] = mags[key].tolist()

    info = {'sess_par': sess_par,
            'basic_par': basic_par,
            'fig_par': fig_par,
            'analys_par': analys_par,
            'mags': mags,
            }

    file_util.save_info(info, save_name, full_dir, 'json')


#############################################
def lfads_dict(sessions, mouse_df, runtype, gabfr, gabk=16, output=''):
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
    sessions = gen_util.list_if_not(sessions)
    for sess in sessions:
        sess.create_dff()
        segs = sess.gabors.get_segs_by_criteria(stimPar2=gabk, 
                        gaborframe=gabfr, by='seg')
        if min(np.diff(segs)) != max(np.diff(segs)) and np.diff(segs)[0] != 4:
            raise ValueError(('Retrieving surprise values not implemented for ' 
                              'non consecutive segments.'))
        
        frames = sess.gabors.get_2pframes_by_seg(segs, first=True)
        surp_segs = sess.gabors.get_segs_by_criteria(stimPar2=gabk, 
                        gaborframe=gabfr, surp=1, by='seg')
        surp_idx = [int((seg - min(segs))/4) for seg in surp_segs]

        df_line = mouse_df.loc[(mouse_df['sessionid'] == sess.session)]
        act_n = df_line['overall_sess_n'].tolist()[0]
        depth = df_line['depth'].tolist()[0]
        mouse = df_line['mouseid'].tolist()[0]
        line = df_line['line'].tolist()[0]
        if depth in [20, 50, 75]:
            layer = 'dend'
        elif depth in [175, 375]:
            layer = 'soma'
        if runtype == 'pilot':
            roi_tr_dir = sess.roi_traces[sess.roi_traces.find('ophys'):]
            roi_dff_dir = sess.roi_traces_dff[sess.roi_traces_dff.find('ophys'):]
        elif runtype == 'prod':
            roi_tr_dir = sess.roi_traces[sess.roi_traces.find('mouse'):]
            roi_dff_dir = sess.roi_traces_dff[sess.roi_traces_dff.find('mouse'):]
        
        sess_dict = {'sessionid'     : sess.session,
                     'mouse'         : mouse,
                     'act_sess_n'    : act_n,
                     'depth'         : depth,
                     'layer'         : layer,
                     'line'          : line,
                     'traces_dir'    : roi_tr_dir,
                     'dff_traces_dir': roi_dff_dir,
                     'gab_k'         : gabk,
                     'gab_fr'        : [gabfr, gabfr_let], # e.g., [0, A]
                     'frames'        : frames.tolist(),
                     'surp_idx'      : surp_idx,    
                     'twop_fps'      : sess.twop_fps,
                    }
    
        name = 'sess_dict_mouse{}_sess{}_{}'.format(mouse, act_n, layer)
        file_util.save_info(sess_dict, name, os.path.join(output, 'session_dicts', 
                                                          runtype), 'json')
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

    data = gen_util.list_if_not(data)
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
    autocorr_me = math_util.mean_med(autocorr_snips, stats, axis=-2)
    autocorr_de = math_util.error_stat(autocorr_snips, stats, error, axis=-2)
    
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
def plot_gab_autocorr(sessions, analys_par, basic_par, fig_par, sess_par):
    """
    plot_gab_autocorr(sessions, analys_par, basic_par, fig_par)

    Plots autocorrelation during gabor blocks.

    Required arguments:
        - sessions (list)  : list of Session objects
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['act_sess_ns'] (list)  : actual overall session number for
                                        each session
                ['gab_k'] (int or list) : gabor kappa values to include 
                                        (e.g., 4, 16 or [4, 16])
                ['lag_s'] (float)       : lag in seconds with which to calculate
                                          autocorrelation
                ['lines'] (list)        : transgenic line for each session
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
                ['figdir_roi'] (str)     : main folder in which to save figure
                ['mult'] (bool)          : if True, prev_dt is created or used.
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
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                            'L5_soma', 'L23_dend', 'L5_dend', 
                                            'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    """
    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'])
    sessstr_pr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 'print')
    statstr = str_util.stat_par_str(basic_par['stats'], basic_par['error'])
    gabkstr = str_util.gab_k_par_str(analys_par['gab_k'])

    print(('\nAnalysing and plotting ROI autocorrelations ' 
          '({}).').format(sessstr_pr))

    fig, ax, ncols, nrows = plot_util.init_fig(len(sessions), fig_par)
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
        if nrows == 1:
            sub_ax = ax[i%ncols]
        else:
            sub_ax = ax[i/ncols][i%ncols]
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
            plot_util.plot_traces(sub_ax, roi_stats, basic_par['stats'], 
                        basic_par['error'], alpha=0.5/len(sessions), 
                        xticks=xticks, yticks=yticks, dff=basic_par['dfoverf'])
        plot_util.add_bars(sub_ax, bars=seg_bars)
        sub_ax.set_title(('Mouse {} - {} gab{} {}\n(sess {}, {} {}, '
                          '(n={}))').format(analys_par['mouse_ns'][i], statstr,
                                            gabkstr, title_str, 
                                            analys_par['act_sess_ns'][i],
                                            analys_par['lines'][i], 
                                            sess_par['layer'], nrois))
        sub_ax.set_ylim([0, 1])
    
    if basic_par['remnans'] == 'across':
        analys_par['nan_rois'] = nan_rois
        analys_par['ok_rois'] = ok_rois    

    save_dir = os.path.join(fig_par['figdir_roi'], fig_par['autocorr'])

    save_name = ('roi_autocorr_{}').format(sessstr)

    full_dir = plot_util.save_fig(fig, save_dir, save_name, fig_par)

    info = {'sess_par': sess_par,
            'basic_par': basic_par,
            'analys_par': analys_par,
            'fig_par': fig_par
            }

    file_util.save_info(info, save_name, full_dir, 'json')



if __name__ == "__main__":

    # typically change runtype, analyses, layer, overall_sess_n, plot_vals

    # commonly changed
        # general
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=None, 
                        help=('data directory (if None, uses a directory '
                              'defined below'))
    parser.add_argument('--output', default='', help='where to store output')
    parser.add_argument('--plt_bkend', default=None, 
                        help='switch matplotlib backend when running on server')
    parser.add_argument('--runtype', default='prod', help='prod or pilot')
    parser.add_argument('--analyses', default='t', 
                        help=('analyses to run: lfads (l), traces (t), '
                              'roi_grps_qu (q), roi_grps_ch (c), mag (m), '
                              'autocorr (a)'))
        # session parameters
    parser.add_argument('--layer', default='dend',
                        help=('soma, dend, L23_soma, L5_soma, L23_dend, '
                              'L5_dend, L23_all, L5_all'))
    parser.add_argument('--overall_sess_n', default=1, type=int,
                        help='session to aim for, e.g. 1, 2, \'last\'')
    parser.add_argument('--min_rois', default=10, type=int, 
                        help='min rois criterion')
        # roi group parameters
    parser.add_argument('--plot_vals', default='surp', 
                        help='plot diff (surp-nosurp), surp or nosurp')
    
    # generally fixed 
        # session parameters
    parser.add_argument('--pass_fail', default='P', 
                        help='P to take only passed sessions')
    parser.add_argument('--omit_sess', default=[], help='sessions to omit')  
    parser.add_argument('--omit_mice', default=[], help='mice to omit') 

        # analysis parameters
    parser.add_argument('--n_quints', default=4, type=int, help='nbr of quintiles')
    parser.add_argument('--lag_s', default=4, type=float,
                        help='lag for autocorrelation (in sec)')
    parser.add_argument('--gab_k', default=[16],
                        help='kappa value ([4], [16], or [4, 16])')    
    parser.add_argument('--gab_fr', default=3, type=int, help='gabor frame of reference')
    parser.add_argument('--pre', default=0, type=float, help='sec before frame')
    parser.add_argument('--post', default=1.5, type=float, help='sec after frame')
    
        # roi group parameters
    parser.add_argument('--op', default='diff', 
                        help='calculate diff or ratio of surp to nonsurp')
    parser.add_argument('--grps', default=['no_change', 'incr', 'reduc'], 
                        help=('plot all ROI grps or grps with change or '
                              'no_change'))
    parser.add_argument('--add_nosurp', action='store_false',
                        help='add nosurp_nosurp to ROI grp plots') # default True
    
        # permutation analysis parameters
    parser.add_argument('--n_perms', default=10000, type=int, 
                        help='nbr of permutations')
    parser.add_argument('--p_val', default=0.05, type=float,
                        help='p-val for perm analysis')
    parser.add_argument('--tails', default='2', 
                        help='nbr tails for perm analysis (2, lo, up)')
    
        # figure parameters
    parser.add_argument('--fig_ext', default='svg', help='svg or png')
    parser.add_argument('--datetime', action='store_false',
                        help='create a datetime folder') # default True
    parser.add_argument('--overwrite', action='store_true', 
                        help='allow overwriting') # default False
    parser.add_argument('--ncols', default=3, type=int, 
                        help='nbr of cols per fig')
    parser.add_argument('--subplot_wid', default=7.5, type=float, 
                        help='subplot width')
    parser.add_argument('--subplot_hei', default=7.5, type=float, 
                        help='subplot height')
    parser.add_argument('--bbox', default='tight', help='wrapping around figs')
    parser.add_argument('--preset_ylims', action='store_true',
                        help='use preset y lims') # default False
    parser.add_argument('--sharey', action='store_false',
                        help='share y axis lims within figs') # default True

        # basic parameters
    parser.add_argument('--rand', action='store_true',
                        help=('produce plots from randomized data (in many '
                              'cases, not implemented yet')) # default False
    parser.add_argument('--remnans', default='across', 
                        help=('remove ROIs containing NaNs or Infs across '
                              'quintiles (across), per quintile (per) or not '
                              '(no)'))
    parser.add_argument('--dfoverf', action='store_false',
                        help='use dfoverf instead of raw ROI traces') # default True
    parser.add_argument('--stats', default='mean', help='plot mean or median')
    parser.add_argument('--error', default='sem', 
                        help='sem for SEM/MAD, std for std/qu')

    args = parser.parse_args()
    args_dict = args.__dict__

    if args.plt_bkend is not None:
        plt.switch_backend(args.plt_bkend)

    if args.datadir is None:
        # previously: '/media/colleen/LaCie/CredAssign/pilot_data'
        args.datadir = '../data/AIBS/{}'.format(args.runtype)
    
    mouse_df_dir = 'mouse_df_{}.csv'.format(args.runtype)

    # split args dictionary keys into different dictionaries
    sess_keys    = ['layer', 'overall_sess_n', 'min_rois', 'pass_fail', 
                    'omit_sess', 'omit_mice']
    analys_keys  = ['n_quints', 'lag_s', 'gab_k', 'gab_fr', 'pre', 'post']
    roi_grp_keys = ['op', 'plot_vals', 'grps', 'add_nosurp']
    perm_keys    = ['n_perms', 'p_val', 'tails']
    fig_keys     = ['fig_ext', 'datetime', 'overwrite', 'ncols', 'subplot_wid', 
                    'subplot_hei', 'bbox', 'preset_ylims', 'sharey']
    basic_keys   = ['rand', 'remnans', 'dfoverf', 'stats', 'error']

    sess_par    = {key: args_dict[key] for key in sess_keys if key in args_dict.keys()}
    analys_par  = {key: args_dict[key] for key in analys_keys if key in args_dict.keys()}
    roi_grp_par = {key: args_dict[key] for key in roi_grp_keys if key in args_dict.keys()}
    perm_par    = {key: args_dict[key] for key in perm_keys if key in args_dict.keys()}
    fig_par     = {key: args_dict[key] for key in fig_keys if key in args_dict.keys()}
    basic_par   = {key: args_dict[key] for key in basic_keys if key in args_dict.keys()}

    fig_par['figdir_roi'] = os.path.join(args.output, 'figures', 
                                         '{}_roi'.format(args.runtype))
    # subfolders
    fig_par['surp_quint'] = 'surp_nosurp_quint' 
    fig_par['autocorr']   = 'autocorr'
    # allow reusing datetime folder (if mult figs created by one function)
    fig_par['prev_dt'] = None
    fig_par['mult']    = False

    # make sure this key contains a list of ints
    analys_par['gab_k'] = [int(val) for val in gen_util.list_if_not(analys_par['gab_k'])]
    
    if args.runtype == 'pilot':
        sess_par['omit_sess'].extend([721038464]) # alignment didn't work
        sess_par['omit_mice'].extend(gab_mice_omit(analys_par['gab_k']))
    elif args.runtype == 'prod' and 16 not in analys_par['gab_k']:
            raise ValueError(('The production data only includes gabor '
                              'stimuli with kappa=16'))

    ### CODE STARTS HERE ###
    mouse_df = file_util.load_file(mouse_df_dir, 'csv')

    # get session numbers
    [analys_par['sess_ns'], analys_par['mouse_ns'],
            analys_par['act_sess_ns'], 
            analys_par['lines']] = sess_per_mouse(mouse_df, **sess_par)

    # create a dictionary with Session objects prepared for analysis
    sessions = init_sessions(analys_par['sess_ns'], args.datadir, args.runtype)

    if args.analyses == 'all':
        args.analyses = 'ltqcma'

    # 0. Create dictionary including frame numbers for LFADS analysis
    if 'l' in args.analyses: # lfads
        lfads_dict(sessions, mouse_df, args.runtype, analys_par['gab_fr'], 
                analys_par['gab_k'], args.output)

    # 1. Plot average traces by quintile x surprise for each session 
    if 't' in args.analyses: # traces
        plot_traces_by_qu_surp_sess(sessions, analys_par, basic_par, fig_par, 
                                    sess_par)

    # 2. Plot average dF/F area for each ROI group across quintiles for each 
    # session 
    if 'q' in args.analyses: # roi_grps_qu
        plot_rois_by_grp_qu_sess(sessions, analys_par, basic_par, fig_par, 
                                 perm_par, roi_grp_par, sess_par)

    # 3. Plot average traces and trace areas by suprise for first vs last 
    # quintile, for each ROI group, for each session
    if 'c' in args.analyses: # roi_grps_ch
        plot_rois_by_grp(sessions, analys_par, basic_par, fig_par, perm_par, 
                        roi_grp_par, sess_par)

    # 4. Plot magnitude of change in dF/F area from first to last quintile of 
    # surprise vs no surprise segments, for each session
    if 'm' in args.analyses: # mag
        plot_mag_change(sessions, analys_par, basic_par, fig_par, sess_par)

    # 5. Run autocorrelation analysis
    if 'a' in args.analyses: # autocorr
        plot_gab_autocorr(sessions, analys_par, basic_par, fig_par, sess_par)


