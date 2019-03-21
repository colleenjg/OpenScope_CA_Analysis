import os
import datetime
import argparse
import glob
import multiprocessing
import re

import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
import pandas as pd
import pdb
from joblib import Parallel, delayed

from analysis import session
from util import file_util, gen_util, math_util, plot_util, str_util


#############################################
def comb_block_ran_seg(stims):
    """
    comb_block_ran_seg(stims)

    Returns the block range segments combines for all stim objects if a list
    is passed.

    Required arguments:
        - stims (list or Stim): Stim object or list of Stim objects

    Return:
        - block_ran_seg (list): segment tuples (start, end) for each block 
                                (end is EXCLUDED) each sublist contains tuples 
                                for a display sequence e.g., for 2 sequences 
                                with 2 blocks each:
                                [[[start, end], [start, end]], 
                                [[start, end], [start, end]]] 
    """

    
    if isinstance(stims, list):
        block_ran_seg = []
        for stim in stims:
            block_ran_seg.extend(stim.block_ran_seg)
    else:
        block_ran_seg = stims.block_ran_seg

    return block_ran_seg


#############################################
def get_stim(sess, stim_type='gabors', bri_idx=0):
    """
    get_stim(sess)

    Returns the Stim object and combined block range segments.

    Required arguments:
        - sess (Session): Session object

    Optional arguments:
        - stim_type (str): stimulus type to return
                           default: 'gabors'
        - bri_idx (int)  : index of brick object to return, if bricks is a list.
                           If None, the brick object is returned, even if it 
                           is a list.
                           default: 0

    Return:
        - stim (Stim)         : Stim object
        - block_ran_seg (list): segment tuples (start, end) for each block 
                                (end is EXCLUDED) each sublist contains tuples 
                                for a display sequence e.g., for 2 sequences 
                                with 2 blocks each:
                                [[[start, end], [start, end]], 
                                [[start, end], [start, end]]] 
    """

    if stim_type == 'gabors':
        stim = sess.gabors
    elif stim_type == 'bricks':
        stim = sess.bricks
    else:
        gen_util.accepted_values_error('stim_type', stim_type, ['gabors', 'bricks'])

    block_ran_seg = comb_block_ran_seg(stim)

    if stim_type == 'bricks' and isinstance(stim, list) and bri_idx is not None:
        if bri_idx < 0:
            raise IOError('\'bri_idx\' must be > 0.')
        elif bri_idx >= len(sess.bricks):
            raise IOError(('\'bri_idx\' value {} is out of range for list '
                            'of length {}.').format(bri_idx, len(sess.bricks)))
        else:
            stim = sess.bricks[0]

    return stim, block_ran_seg


#############################################
def label_values(mouse_df, label, values='any'):
    """
    label_values(mouse_df, label)

    Either returns the specified value(s) for a specific label of a pandas 
    dataframe as a list, or if values='any', collects all different values for 
    that label and returns them in a list.

    Required arguments:
        - mouse_df (pandas df): dataframe containing parameters for each 
                                session.
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
def sess_values(mouse_df, returnlab, mouseid, sessid, runtype, depth, pass_fail, 
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
        - runtype (str or list)      : runtype value(s) of interest       
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

    sess_vals = mouse_df.loc[(mouse_df['mouseid'].isin(mouseid)) & 
                             (mouse_df['sessionid'].isin(sessid)) &
                             (mouse_df['runtype'].isin(runtype)) &
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
def all_sess_ns(mouse_df, runtype='any', layer='any', pass_fail='any', 
                all_files=[1], any_files=[1], omit_sess=[], omit_mice=[], 
                min_rois=1):
    """
    all_sess_ns(mouse_df)

    Returns list of overall session numbers that correspond to specific
    criteria.

    Required arguments:
        - mouse_df (pandas df): dataframe containing parameters for each session.
        
    Optional arguments:
        - runtype (str or list)      : runtype value(s) of interest
                                       ('pilot', 'prod')
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
        - omit_sess (list)           : sessions to omit
                                       (default: [])
        - omit_mice (list)           : mice to omit
                                       (default: [])
        - min_rois (int)             : min number of ROIs
                                       (default: 1)
     
    Returns:
        - all_sess (list): overall session numbers that correspond to criteria
    """

     # get depth values corresponding to the layer
    depth = depth_values(layer)

    mouseid, sessid, sess_n = ['any', 'any', 'any']
    params      = [mouseid, sessid, runtype, depth, pass_fail, all_files,  
                   any_files, sess_n]
    param_names = ['mouseid', 'sessionid', 'runtype', 'depth', 'pass_fail', 
                   'all_files', 'any_files', 'overall_sess_n']
    
    # for each label, collect values of that fit criteria in a list
    for i in range(len(params)):
        params[i] = label_values(mouse_df, param_names[i], params[i])
    [mouseid, sessid, runtype, depth, pass_fail, 
                    all_files, any_files, sess_n] = params

    # remove omitted sessions from the session id list
    sessid = gen_util.remove_if(sessid, omit_sess)
    
    # collect all mouse IDs and remove omitted mice
    mouseid = gen_util.remove_if(mouseid, omit_mice)

    all_sess = sess_values(mouse_df, 'overall_sess_n', mouseid, sessid, runtype,
                           depth, pass_fail, all_files, any_files, sess_n, 
                           min_rois, sort=True)

    all_sess = list(set(all_sess)) # get unique

    return all_sess


#############################################
def sess_per_mouse(mouse_df, sessid='any', runtype='any', layer='any', 
                   pass_fail='any', all_files=[1], any_files=[1], 
                   overall_sess_n=1, closest=False, omit_sess=[], omit_mice=[], 
                   min_rois=1):
    """
    sess_per_mouse(mouse_df)
    
    Returns list of session IDs (up to 1 per mouse) that fit the specified
    criteria, IDs of mice for which a session was found and actual overall 
    session numbers.

    Required arguments:
        - mouse_df (pandas df): dataframe containing parameters for each session.
        
    Optional arguments:
        - sessid (int or list)       : session id value(s) of interest
                                       (default: 'any')
        - runtype (str or list)      : runtype value(s) of interest
                                       ('pilot', 'prod')
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
        - closest (bool)             : if False, only exact session number is 
                                       retained, otherwise the closest
                                       (default: False)
        - omit_sess (list)           : sessions to omit
                                       (default: [])
        - omit_mice (list)           : mice to omit
                                       (default: [])
        - min_rois (int)             : min number of ROIs
                                       (default: 1)
     
    Returns:
        - sess_ns (list)    : sessions to analyse (1 per mouse)
    """

    # get depth values corresponding to the layer
    depth = depth_values(layer)

    params = [sessid, runtype, depth, pass_fail, all_files, any_files]
    param_names = ['sessionid', 'runtype', 'depth', 'pass_fail', 'all_files', 
                   'any_files']
    
    # for each label, collect values of that fit criteria in a list
    for i in range(len(params)):
        params[i] = label_values(mouse_df, param_names[i], params[i])
    [sessid, runtype, depth, pass_fail, all_files, any_files] = params

    if closest or overall_sess_n == 'last':
        overall_sess_any = label_values(mouse_df, 'overall_sess_n', 'any')
    else:
        overall_sess_any = gen_util.list_if_not(overall_sess_n)
    
    # remove omitted sessions from the session id list
    sessid = gen_util.remove_if(sessid, omit_sess)
    
    # collect all mouse IDs and remove omitted mice
    mouseids = gen_util.remove_if(sorted(label_values(mouse_df, 'mouseid', 
                                                      values='any')), omit_mice)

    # get session ID, mouse ID and actual session numbers for each mouse based 
    # on criteria 
    sess_ns = []
    for i in mouseids:
        sessions = sess_values(mouse_df, 'overall_sess_n', [i], sessid, 
                               runtype, depth, pass_fail, all_files, any_files, 
                               overall_sess_any, min_rois, sort=True)
        # skip mouse if no sessions meet criteria
        if len(sessions) == 0:
            continue
        elif overall_sess_n == 'last' or not closest:
            sess_n = sessions[-1]
        # find closest sess number among possible sessions
        else:
            sess_n = sessions[np.argmin(np.absolute([x-overall_sess_n
                                                     for x in sessions]))]
        sess = sess_values(mouse_df, 'sessionid', [i], sessid, runtype, 
                           depth, pass_fail, all_files, any_files, [sess_n], 
                           min_rois)[0]
        sess_ns.append(sess)
    
    if len(sess_ns) == 0:
        raise ValueError('No sessions meet the criteria.')

    return sess_ns


#############################################
def init_sessions(sess_ns, datadir, mouse_df, runtype='prod', full_dict=True, 
                  load_run=True):
    """
    init_sessions(sess_ns, datadir)

    Creates list of Session objects for each session ID passed.

    Required arguments:
        - sess_ns (int or list): ID or list of IDs of sessions
        - datadir (str)        : directory where sessions are stored
        - mouse_df (pandas df) : dataframe containing information for 
                                 each session

    Optional arguments:
        - runtype (string): the type of run, either 'pilot' or 'prod'
                            default = 'prod' 
        - load_run (bool) : if True, session run info is loaded
                            default = True           
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
        sess.extract_sess_attribs(mouse_df)
        sess.extract_info(full_dict=full_dict, load_run=load_run)
        print('Finished session {}.'.format(sess_n))
        sessions.append(sess)

    return sessions


#############################################
def get_sess_info(sessions, dfoverf=True):
    """
    get_sess_info(sessions)

    Puts information from each session into a dictionary.

    Required arguments:
        - sessions (list): ordered list of Session objects
    
    Optional arguments:
        - dfoverf (bool): if True, an error is thrown if the list of ROIs with
                          NaNs/Infs in dF/F traces cannot be extracted

    Returns:
        - sess_info (dict): dictionary containing information from each
                            session 
                ['mouse_ns'] (list)       : mouse numbers
                ['overall_sess_ns'] (list): overall session numbers  
                ['lines'] (list)          : mouse lines
                ['layers'] (list)         : imaging layers
                ['n_rois'] (list)         : number of ROIs in session
                ['nanrois'] (list)        : list of ROIs with NaNs/Infs in raw
                                            traces
                ['nanrois_dff'] (list)    : list of ROIs with NaNs/Infs in dF/F
                                            traces, for sessions for which this 
                                            attribute exists
    """

    sess_info = dict()
    keys = ['mouse_ns', 'overall_sess_ns', 'lines', 'layers', 'n_rois', 
            'nanrois']
    
    if dfoverf:
        keys.append('nanrois_dff')

    for key in keys:
        sess_info[key] = []

    for i, sess in enumerate(sessions):
        sess_info['mouse_ns'].append(sess.mouse_n)
        sess_info['overall_sess_ns'].append(sess.sess_overall)
        sess_info['lines'].append(sess.line)
        sess_info['layers'].append(sess.layer)
        sess_info['n_rois'].append(sess.nroi)
        sess_info['nanrois'].append(sess.nanrois)
        if hasattr(sess, 'nanrois_dff'):
            sess_info['nanrois_dff'].append(sess.nanrois_dff)
        elif dfoverf:
            # try to extract information
            sess.get_nanrois(dfoverf=True)
            sess_info['nanrois_dff'].append(sess.nanrois_dff)
            
    return sess_info


#############################################
def pilot_gab_omit(gab_k):
    """
    pilot_gab_omit(gab_k)

    Returns IDs of pilot mice to omit based on gabor kappa values to include.

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
def pilot_bri_omit(bri_dir, bri_size):
    """
    pilot_bri_omit(bri_dir, bri_size)

    Returns IDs of pilot mice to omit based on brick direction and size values 
    to include.

    Required arguments:
        - bri_dir (str or list): brick direction values
        - bri_size (int or list): brick size values
                                    
    Return:
        - omit_mice (list): list IDs of mice to omit
    """

    bri_dir = gen_util.list_if_not(bri_dir)
    bri_size = gen_util.list_if_not(bri_size)
    omit_mice = []

    if 'right' not in bri_dir:
        omit_mice.extend([3]) # mouse 3 only got bri_dir='right'
        if 128 not in bri_size:
            omit_mice.extend([1]) # mouse 1 only got bri_dir='left' with bri_size=128
    elif 'left' not in bri_dir and 256 not in bri_size:
        omit_mice.extend([1]) # mouse 1 only got bri_dir='right' with bri_size=256
    return omit_mice


#############################################
def format_args(args):
    """
    format_args(args)

    Reformats args correctly.

    Required arguments:
        - args (Argument parser): parser with arguments as attributes listed
                                  in the _keys lists, as well as 
                bri_dir (str)  : brick direction values to include
                                 (e.g., 'right', 'left' or 'both')
                bri_size (str) : brick size values to include
                                 (e.g., 128, 256, 'both')
                gab_fr (int)   : gabor frame value to start segments at
                                 (e.g., 0, 1, 2, 3)
                gab_k (str)    : gabor kappa values to include 
                                 (e.g., 4, 16 or 'both')
                omit_sess (str): sess to omit
                omit_mice (str): mice to omit
                stim (str)     : stimulus to analyse (bricks or gabors)
                grps (str)     : set or sets of groups to plot, 
                                 e.g., 'all change no_change reduc incr'.
    """

    # make sure these keys contains a list of ints/strs, and are set to 'none'
    # if irrelevant to the stimulus
    if args.stim == 'gabors':
        args.bri_size = 'none'
        args.bri_dir = 'none'
        if args.gab_k == 'both':
            args.gab_k = [4, 16]
        else:
            args.gab_k = int(args.gab_k)

    elif args.stim == 'bricks':
        args.gab_fr = 'none'
        args.gab_k = 'none'
        if args.bri_size == 'both':
            args.bri_size = [128, 256]
        else:
            args.bri_size = int(args.bri_size)
        if args.bri_dir == 'both':
            args.bri_dir = ['right', 'left']
    else:
        gen_util.accepted_values_error('stim argument', args.stim, 
                                       ['gabors', 'bricks'])

    # convert string args to lists
    args.grps = gen_util.str_to_list(args.grps)
    args.omit_sess = gen_util.str_to_list(args.omit_sess, only_int=True)
    args.omit_mice = gen_util.str_to_list(args.omit_mice, only_int=True)


#############################################
def update_args(args):
    """
    update_args(args)

    Updates mice and sessions to omit based on analysis parameters (runtype, 
    gab_k, bri_dir and bri_size) and throws an error if the parameter 
    combination does not occur in the dataset.

    Required arguments:
        - args (Argument parser): parser with arguments as attributes listed
                                  in the _keys lists, as well as 
                bri_size (int or list): brick size values to include
                                        (e.g., 128, 256, [128, 256])
                gab_k (int or list)   : gabor kappa values to include 
                                        (e.g., 4, 16 or [4, 16])
                omit_sess (list)      : sess to omit
                omit_mice (list)      : mice to omit
                runtype (str)         : runtype value(s) of interest
                                        e.g., ('pilot', 'prod')
                stim (str)            : stimulus to analyse, e.g. bricks or 
                                        gabors
    """

    if args.runtype == 'pilot':
        args.omit_sess.extend([721038464]) # alignment didn't work
        if args.stim == 'gabors':
            args.omit_mice.extend(pilot_gab_omit(args.gab_k))
        elif args.stim == 'bricks':
            args.omit_mice.extend(pilot_bri_omit(args.bri_dir, args.bri_size))
    
    # verify whether the parameters are present in the runtype data
    elif args.runtype == 'prod':
        if args.stim == 'gabors': 
            if 16 not in gen_util.list_if_not(args.gab_k):
                raise ValueError(('The production data only includes gabor '
                                  'stimuli with kappa=16'))
        elif args.stim == 'bricks':
            if 128 not in gen_util.list_if_not(args.bri_size):
                raise ValueError(('The production data only includes bricks '
                                  'stimuli with size=128'))


#############################################
def quint_par(stim, analys_par, block_ran_seg=None):
    """
    quint_par(stim, analys_par)

    Returns dictionary containing parameters for breaking segments into 
    quintiles.
    
    Required arguments:
        - stim (Stim object): stim object
        - analys_par (dict) : dictionary containing relevant parameters
                              to extracting segment numbers and dividing
                              them into quintiles
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                          (e.g., 128, 256 or [128, 256])
                ['gab_fr'] (int)          : gabor frame at which segments start 
                                            (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list) : gabor kappa values to include 
                                          (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)      : number of quintiles

    Return:
        - qu_info (dict): dictionary containing parameters for breaking 
                          segments into quintiles
                ['seg_min'] (int): minimum segment number
                ['seg_max'] (int): maximum segment number
                ['len'] (float)  : length of each quintile in seg numbers
                                   (can be a decimal)
    """

    # get all seg values (for all gabor frames)
    all_segs = stim.get_segs_by_criteria(gab_k=analys_par['gab_k'], 
                                         bri_dir=analys_par['bri_dir'],
                                         bri_size=analys_par['bri_size'],
                                         by='seg', block_ran_seg=block_ran_seg)

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
def quint_segs(stim, analys_par, qu_info, surp='any', block_ran_seg=None,):
    """
    quint_segs(stim, analys_par)

    Returns dictionary containing parameters for breaking segments into 
    quintiles.
    
    Required arguments:
        - stim (Stim object): stim object
        - analys_par (dict)    : dictionary containing relevant parameters
                                 to extracting segment numbers and dividing
                                 them into quintiles
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                          (e.g., 128, 256 or [128, 256])
                ['gab_fr'] (int)          : gabor frame at which segments start 
                                            (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles

        - quint_info (dict)    : dictionary containing parameters for breaking 
                                 segments into quintiles
                ['seg_min'] (int): minimum segment number
                ['seg_max'] (int): maximum segment number
                ['len'] (float)  : length of each quintile in seg numbers
                                   (can be a decimal)
    Optional arguments:
        - surp (int or list)  : surprise values to include (e.g., 0 or 1)
                                default: 'any'
        - block_ran_seg (list): segment tuples (start, end) for each block 
                                (end is EXCLUDED) each sublist contains tuples 
                                for a display sequence e.g., for 2 sequences 
                                with 2 blocks each:
                                [[[start, end], [start, end]], 
                                [[start, end], [start, end]]] 
                                default: None
    Returns:
        - qu_segs (list) : list of sublists for each quintile, each containing 
                           segment numbers for that quintile
        - qu_count (list): list of number of segments in each quintile
    """

    # get all seg values
    all_segs = stim.get_segs_by_criteria(gaborframe=analys_par['gab_fr'],
                                         gab_k=analys_par['gab_k'], 
                                         bri_dir=analys_par['bri_dir'],
                                         bri_size=analys_par['bri_size'],
                                         surp=surp, by='seg', 
                                         block_ran_seg=block_ran_seg)
                                         
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
def chunk_stats_by_qu(stim, qu_seg, pre, post, byroi=True, dfoverf=True, 
                      remnans='per', rand=False, stats='mean', error='std',
                      data='all'):
    """
    chunk_stats_by_qu(stim, qu_seg, pre, post)

    Returns chunk statistics for the quintiles of interest. 

    Required arguments:
        - stim (Stim object): stim object
        - qu_seg (dict)     : list of sublists for each quintile, each 
                              containing segment numbers for that quintile
        - pre (float)       : range of frames to include before each frame 
                              reference (in s)
        - post (float)      : range of frames to include after each frame 
                              reference (in s)
    
    Optional arguments:
        - byroi (bool)  : if True, returns statistics for each ROI. If False,
                          returns statistics across ROIs.
                          default: True 
        - dfoverf (bool): if True, dF/F is used instead of raw ROI traces.
                          default: True
        - remnans (str)  : if 'per', removes ROIs with NaN/Inf values, for each
                          subdivision (quintile/surprise). If 'all', removes
                          ROIs with NaN/Inf values across entire session. If 'no',
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
    Optional returns (if remnans in ['per', 'all']):
            (list) containing:
                - nan_rois (list): if remnans is 'per', list of sublists for 
                                    each quintile, each containing numbers of 
                                    ROIs removed for containing NaN/Infs. If
                                    remnans == 'all', list of ROIs 
                                    with NaN/Infs across entire session.
                                    
                - ok_rois (list) : if remnans is 'per', list of sublists for 
                                    each quintile, each containing numbers of 
                                    ROIs without NaN/Infs. If remnans is 
                                    'all', list of ROIs without NaN/Infs 
                                    across entire session.    
    """

    if rand:
        raise NotImplementedError(('Retrieving stats for random data using '
                                   '\'chunk_stats_by_qu()\' not implemented.'))

    if remnans == 'per':
        nans = 'rem'
        nan_rois = []
    elif remnans == 'all':
        nans = 'rem_all'
    elif remnans == 'no':
        nans = 'no'
    else:
        gen_util.accepted_values_error('remnans', remnans, ['per', 'all', 'no'])
    
    qu_stats = []
    for qu, segs in enumerate(qu_seg):
        if remnans == 'per':
            print('\tQuintile {}'.format(qu+1))
        # get the stats for ROI traces for these segs 
        # returns x_ran, [mean/median, std/quartiles] for each ROI or across ROIs
        chunk_info = stim.get_roi_chunk_stats(stim.get_2pframes_by_seg(segs, 
                                              first=True), pre, post, 
                                              byroi=byroi, dfoverf=dfoverf, 
                                              nans=nans, rand=False, 
                                              stats=stats, error=error)
        x_ran = chunk_info[0]
        chunk_stats = chunk_info[1]
        if remnans == 'per':
            nan_rois.append(chunk_info[2][0])
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
    
    returns = [x_ran, qu_stats]
    if remnans == 'per':
        returns.append(nan_rois)
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
        if basic_par['remnans'] is ['all', 'per']:
            ['nan_rois'] (list): list of sublists each containing numbers of 
                                 ROIs removed, structured as follows:
                                    if 'per': session x (surp value if bysurp x) 
                                              quint
                                    if 'all': session
            ['ok_rois'] (list) : list of sublists each containing numbers of 
                                 ROIs retained, structured as follows:
                                    if 'per': session x (surp value if bysurp x) 
                                              quint
                                    if 'all': session

    Required arguments:
        - sessions (list): list of Session objects
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_fr'] (int)          : gabor frame at which segments start 
                                            (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['pre'] (float)           : range of frames to include before 
                                            each frame reference (in s)
                ['post'] (float)          : range of frames to include after  
                                            each frame reference (in s)
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)
        - basic_par (dict): dictionary containing additional parameters 
                            relevant to analysis
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['rand'] (bool)   : if True, also includes statistics for a 
                                    random permutation of the traces (not 
                                    implemented).
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'all', removes ROIs with NaN/Inf values 
                                    across entire session. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'

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

    print('\nGetting ROI trace stats for each session.')
    all_counts = []
    all_stats = []
    all_twop_fps = [] # 2p fps by session
    if basic_par['remnans'] == 'per':
        all_nan_rois = []
    for sess in sessions:
        print('Session {}'.format(sess.session))
        stim, block_ran_seg = get_stim(sess, analys_par['stim'])
        # get length of each quintile, seg_min and seg_max in dictionary
        quint_info = quint_par(stim, analys_par, block_ran_seg=block_ran_seg)
        # retrieve list of segs per quintile
        sess_counts = []
        sess_stats = []
        sess_nan_rois = []
        for surp in surp_vals:
            if basic_par['remnans'] == 'per':
                if surp == 0:
                    print('    Non surprise')
                elif surp == 1:
                    print('    Surprise')
            qu_seg, qu_count = quint_segs(stim, analys_par, quint_info, surp, 
                                          block_ran_seg)
            sess_counts.append(qu_count)
            chunk_info = chunk_stats_by_qu(stim, qu_seg, analys_par['pre'],
                                           analys_par['post'], byroi=byroi, 
                                           data=data, **basic_par)
            x_ran = chunk_info[0]
            sess_stats.append(chunk_info[1])
            # if ROIs were removed per subdivision
            if basic_par['remnans'] == 'per':
                sess_nan_rois = sess_nan_rois.append(chunk_info[2][0])
            if twop_fps:
                all_twop_fps.append(sess.twop_fps)
        # store by session
        sess_stats = np.asarray(sess_stats)
        if basic_par['remnans'] == 'per':
            all_nan_rois.append(sess_nan_rois)
        if twop_fps:
            all_twop_fps.extend([sess.twop_fps])
        all_counts.append(sess_counts)
        all_stats.append(sess_stats)
    
    analys_par['seg_per_quint'] = all_counts
    if basic_par['remnans'] == 'per':
        analys_par['nan_rois'] = all_nan_rois

    returns = [x_ran, all_stats]
    if twop_fps:
        returns.append(all_twop_fps)
    return returns


#############################################
def remove_nan_rois(data, roi_dim=0, rem_rois=None):
    """
    remove_nan_rois(data)

    Remove ROIs from a data array, either specified or containing NaN/Infs . 

    Required arguments:
        - data (2 to 5D array): nd array from which to remove ROIs
    
    Optional arguments:
        - roi_dim (int)       : dimension of data corresponding to ROIs
                                default: 0
        - rem_rois (list)     : flat list of ROIs to remove. If None, ROIs
                                containing NaNs or Infs in the data are removed.
                                default: None
    
    Return:
        - data (2 to 5D array): nd array from which ROIs have been removed
        - rem_rois (list)     : list of ROIs removed
    """

    print_rem = False

    n_rois = data.shape[roi_dim]
    
    if rem_rois is None:
        print_rem = True
        rem_arr = np.isnan(data).any(axis=roi_dim) + \
                  np.isinf(data).any(axis=roi_dim)
        rem_rois = np.where(rem_arr)[0].tolist()
    
    rem_rois = sorted(list(set(rem_rois)))
    keep_rois = sorted(list(set(range(n_rois)) - set(rem_rois)))
    
    # remove any ROIs containing NaN/Infs from ok_rois, and flatten
    if len(rem_rois) != 0:
        idx = [slice(None)] * roi_dim + [keep_rois]
        data = data[idx]
        if print_rem:
            print('Removing {}/{} ROIs across quintiles and divisions: {}'
                  .format(len(rem_rois), n_rois, ', '.join(map(str, rem_rois))))
    
    return data, rem_rois


#############################################
def plot_traces_by_qu_surp_sess_from_dicts(analys_par, basic_par, chunk_stats, 
                                           fig_par, sess_info, sess_par, 
                                           save_dir=None):
    """
    plot_traces_by_qu_surp_sess_from_dicts(analys_par, basic_par, data, fig_par,
                                           sess_info, sess_par)

    From dictionaries, plots traces by quintile/surprise with each session in a 
    separate subplot.
    
    Required arguments:
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_fr'] (int)          : gabor frame at which segments start 
                                            (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['ok_rois'] (list)        : list of ROIs included in analysis,
                                            required if basic_par['remnans'] is
                                            'per' or 'all', structured as:
                                                'per': sess x surp x quint
                                                'all': sess
                ['pre'] (float)           : range of frames to include before 
                                            each frame reference (in s)
                ['post'] (float)          : range of frames to include after  
                                            each frame reference (in s)
                ['segs_per_quint'] (list) : number of segs per quintile, 
                                            structured as:
                                                sess x surp x quint
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'all', removes ROIs with NaN/Inf values 
                                    across entire session. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'

        - chunk_stats (dict) : dictionary containing data to plot.
                ['x_ran'] (list)     : list of time values for the frame chunks
                ['all_stats'] (list) : list of 2 to 5D arrays (or nested lists) 
                                       of statistics for chunks for each 
                                       session:
                                            (surp if bysurp x)
                                            quintiles x
                                            (ROIs if byroi x)
                                            (statistic if data == 'all' x)
                                            frames
        
        - fig_par (dict)   : dictionary containing figure parameters:
                ['bbox'] (str)           : bbox parameter for plt.savefig(), 
                                           e.g., 'tight'
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
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['surp_quint'] (str)     : specific subfolder in which to save 
                                           folder

        - sess_info (dict)  : dictionary containing information from each
                              session 
                ['mouse_ns'] (list)       : mouse numbers
                ['overall_sess_ns'] (list): overall session numbers  
                ['lines'] (list)          : mouse lines
                ['layers'] (list)         : imaging layers
                ['n_rois'] (list)         : number of ROIs in session
                ['nanrois'] (list)        : list of ROIs with NaNs/Infs in raw
                                            traces
                ['nanrois_dff'] (list)    : list of ROIs with NaNs/Infs in dF/F
                                            traces, for sessions for which this 
                                            attribute exists

        - sess_par (dict)  : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                               'L5_soma', 'L23_dend', 'L5_dend', 
                                               'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
        
        - save_dir (str) : path of directory in which to save plots.
                           default: None
    
    Returns:
        - full_dir (str):  final name of the directory in which the figure is 
                           saved
                           (may be different from input save_dir, as a datetime 
                           subfolder may have been added depending on the 
                           parameters in fig_par.)
        - save_name (str): name under which the figure is saved, excluding 
                           extension
    """

    stimstr = str_util.stim_par_str(analys_par['gab_k'], analys_par['bri_dir'], 
                                    analys_par['bri_size'], analys_par['stim'],
                                    'print')
    statstr = str_util.stat_par_str(basic_par['stats'], basic_par['error'])
    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], analys_par['bri_size'], 
                                    analys_par['stim'], 'file')
    fluorstr_pr = str_util.fluor_par_str(type_str='print', dff=basic_par['dfoverf'])

    if analys_par['stim'] == 'gabors':
        [xpos, labels_nosurp, h_bars, 
            seg_bars] = plot_util.plot_seg_comp(analys_par, 'nosurp')
        _, labels_surp, _, _ = plot_util.plot_seg_comp(analys_par, 'surp')
        t_heis = [0.85, 0.75]

    if analys_par['n_quints'] <= 7:
        col_nosurp = ['#50a2d5', 'cornflowerblue', 'steelblue', 'dodgerblue', 
                      'mediumblue', 'darkblue', 'royalblue'][0:analys_par['n_quints']]
        col_surp   = ['#eb3920', 'tomato', 'salmon', 'coral', 'orangered', 
                     'indianred', 'firebrick', ][0:analys_par['n_quints']]
    else:
        raise NotImplementedError(('Not enough colors preselected for more '
                                    'than 4 quintiles.'))

    # extract some info from dictionaries
    mouse_ns         = sess_info['mouse_ns']
    overall_sess_ns  = sess_info['overall_sess_ns']
    lines            = sess_info['lines']
    layers           = sess_info['layers']
    n_rois           = sess_info['n_rois']
    n_sess = len(mouse_ns)

    x_ran = np.asarray(chunk_stats['x_ran'])
    all_stats = [np.asarray(sess_stats) for sess_stats in chunk_stats['all_stats']]

    fig, ax = plot_util.init_fig(n_sess, fig_par)

    for i in range(n_sess):
        sub_ax = plot_util.get_subax(ax, i)
        for s, [col, leg_ext] in enumerate(zip([col_nosurp, col_surp],
                                               ['nosurp', 'surp'])):
            for q in range(analys_par['n_quints']):
                n_roi = n_rois[i]
                if basic_par['remnans'] == 'per':
                    n_roi = n_roi - len(analys_par['nan_rois'][i][s][q])
                elif basic_par['remnans'] == 'all':
                    if basic_par['dfoverf']:
                        n_roi = n_roi - len(sess_info['nanrois_dff'][i])
                    else:
                        n_roi = n_roi - len(sess_info['nanrois'][i])
                title=(u'Mouse {} - {}{} {} {} across seqs\n(sess {}, '
                       '{} {}, n={})').format(mouse_ns[i], 
                                              analys_par['stim'][0:3], stimstr, 
                                              statstr, fluorstr_pr, 
                                              overall_sess_ns[i], lines[i], 
                                              layers[i], n_roi)
                chunk_stats = np.concatenate([x_ran[np.newaxis, :], 
                                              all_stats[i][s][q]], axis=0)
                leg = '{}-{} ({})'.format(q+1, leg_ext, 
                                          analys_par['seg_per_quint'][i][s][q])
                plot_util.plot_traces(sub_ax, chunk_stats, 
                                      stats=basic_par['stats'], 
                                      error=basic_par['error'], col=col[q], 
                                      alpha=0.8/analys_par['n_quints'],
                                      title=title, label=leg,
                                      dff=basic_par['dfoverf'])
        if analys_par['stim'] == 'gabors':
            plot_util.add_bars(sub_ax, hbars=h_bars, bars=seg_bars)

    if analys_par['stim'] == 'gabors': 
        plot_util.incr_ymax(ax, incr=1.05/min(t_heis), sharey=fig_par['sharey'])
        for i in range(n_sess):
            sub_ax = plot_util.get_subax(ax, i)
            for s, (lab, col, t_hei) in enumerate(zip([labels_nosurp, labels_surp], 
                                                    [col_nosurp, col_surp], 
                                                    t_heis)):
                plot_util.add_labels(sub_ax, lab, xpos, t_hei, col[0])

    if save_dir is None:
        save_dir = os.path.join(fig_par['figdir_roi'], fig_par['surp_quint'])

    save_name = 'roi_av_{}_{}quint'.format(sessstr, analys_par['n_quints'])
    full_dir = plot_util.save_fig(fig, save_dir, save_name, fig_par)

    return full_dir, save_name


#############################################
def plot_traces_by_qu_surp_sess(sessions, analys_par, basic_par, fig_par,
                                sess_par):
    """
    plot_traces_by_qu_surp_sess(sessions, analys_par, basic_par, fig_par,
                                sess_par)

    Retrieves chunk statistics by session x surp val x quintile and
    plots traces by quintile/surprise with each session in a separate subplot.
    Saves results and parameters relevant to analysis in a dictionary.

    Required arguments:
        - sessions (list)  : list of Session objects
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_fr'] (int)          : gabor frame at which segments start 
                                            (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['pre'] (float)           : range of frames to include before 
                                            each frame reference (in s)
                ['post'] (float)          : range of frames to include after  
                                            each frame reference (in s)
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['rand'] (bool)   : if True, also includes statistics for a 
                                    random permutation of the traces (not 
                                    implemented).
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'all', removes ROIs with NaN/Inf values 
                                    across subdivisions. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'

        - fig_par (dict)   : dictionary containing figure parameters:
                ['bbox'] (str)           : bbox parameter for plt.savefig(), 
                                           e.g., 'tight'
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
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['surp_quint'] (str)     : specific subfolder in which to save 
                                           folder

        - sess_par (dict)  : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                               'L5_soma', 'L23_dend', 'L5_dend', 
                                               'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    """

    sessstr_pr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], analys_par['bri_size'], 
                                    analys_par['stim'], 'print')

    print(('\nAnalysing and plotting surprise vs non surprise ROI traces '
           'by quintile ({}) \n({}).').format(analys_par['n_quints'], 
                                              sessstr_pr))
    # get the stats (all) separating by session, surprise and quintiles
    chunk_info = chunk_stats_by_qu_sess(sessions, analys_par, basic_par, 
                                        byroi=False, data='all', bysurp=True)

    all_stats = [sess_stats.tolist() for sess_stats in chunk_info[1]]

    chunk_stats = {'x_ran'    : chunk_info[0].tolist(),
                   'all_stats': all_stats
                  }

    info = {'sess_par'   : sess_par,
            'basic_par'  : basic_par,
            'analys_par' : analys_par,
            'sess_info'  : get_sess_info(sessions, basic_par['dfoverf']),
            'chunk_stats': chunk_stats
            }

    [full_dir, 
        save_name] = plot_traces_by_qu_surp_sess_from_dicts(fig_par=fig_par, 
                                                            **info)

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
        - all_grp_st (4D array): array of group stats (mean/median, error) 
                                 structured as:
                                  session x quintile x grp x stat 
        - all_ns (2D array)    : array of group ns, structured as:
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
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_fr'] (int)          : gabor frame at which segments start 
                                            (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['pre'] (float)           : range of frames to include before 
                                            each frame reference (in s)
                ['post'] (float)          : range of frames to include after  
                                            each frame reference (in s)
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['rand'] (bool)   : if True, also includes statistics for a 
                                    random permutation of the traces (not 
                                    implemented).
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'all', removes ROIs with NaN/Inf values 
                                    across entire session. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'

        - roi_grp_par (dict) : dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2

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
        print('NaNs should be removed across subgroups, not per.')
        basic_par['remnans'] == 'all'
    
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
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_fr'] (int)          : gabor frame at which segments start 
                                            (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['pre'] (float)           : range of frames to include before 
                                            each frame reference (in s)
                ['post'] (float)          : range of frames to include after  
                                            each frame reference (in s)
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)

        - basic_par (dict)     : dictionary containing basic analysis 
                                 parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'all', removes ROIs with NaN/Inf values 
                                    across entire session. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'

        - perm_par (dict)    : dictionary containing permutation analysis 
                               parameters:
                ['n_perms'] (int)     : nbr of permutations to run
                ['p_val'] (float)     : p-value to use for significance  
                                        thresholding (0 to 1)
                ['tails'] (str or int): which tail(s) to test: 'up', 'lo', '2'

        - roi_grp_par (dict) : dictionary containing ROI grouping parameters:
                ['op'] (str)          : operation to use to compare groups, 
                                        i.e. 'diff': grp1-grp2, or 'ratio': 
                                        grp1/grp2
                ['grps'] (str or list): set or sets of groups to return, 
                                        e.g., 'all', 'change', 'no_change', 
                                        'reduc', 'incr'.
                                        If several sets are passed, each set 
                                        will be collapsed as one group and
                                        'add_nosurp' will be set to False.
                ['add_nosurp'] (bool) : if True, group of ROIs showing no 
                                        significance in either is included in  
                                        the groups returned

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
    print(('\nIdentifying ROIs showing significant surprise in first and/or '
           'last quintile.'))

    for s, sess in enumerate(sessions):
        print('\nSession {}'.format(sess.session))
        fps = sess.twop_fps
        stim, block_ran_seg = get_stim(sess, analys_par['stim'])
        quint_info = quint_par(stim, analys_par, block_ran_seg=block_ran_seg)
        qu_seg, _ = quint_segs(stim, analys_par, quint_info, surp='any', 
                               block_ran_seg=block_ran_seg)
        all_rois = []
        # Run permutation test for first and last quintiles
        for q, pos in zip([0, analys_par['n_quints']-1], ['First', 'Last']):
            print('    {} quintile'.format(pos))
            # get dF/F for each segment and each ROI
            qu_twop_fr = stim.get_2pframes_by_seg(qu_seg[q], first=True)
            _, roi_traces = stim.get_roi_trace_chunks(qu_twop_fr, 
                                analys_par['pre'], analys_par['post'], 
                                dfoverf=basic_par['dfoverf'])
            # remove previously removed ROIs if applicable 
            if basic_par['remnans'] in ['per', 'all']:
                if basic_par['remnans'] == 'all':
                    if basic_par['dfoverf']:
                        nan_rois = sess.nanrois_dff
                    else:
                        nan_rois = sess.nanrois
                else:
                    nan_rois = analys_par['nan_rois'][s]
                roi_traces, _ = remove_nan_rois(roi_traces, roi_dim=0, 
                                                rem_rois=nan_rois)
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
def plot_rois_by_grp_qu_sess_from_dicts(analys_par, basic_par, perm_par, 
                                        roi_grp_par, roi_grps, sess_info, 
                                        fig_par, sess_par=None, save_dir=None):
    """
    plot_rois_by_grp_qu_sess_from_dicts(analys_par, basic_par, fig_par, perm_par, 
                                        roi_grp_par, roi_grps, sess_par)

    From dictionaries, plots average integrated surprise, no surprise or 
    difference between surprise and no surprise activity per group of ROIs 
    showing significant surprise in first and/or last quintile. Each session is 
    in a different plot.

    Required arguments:
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['rand'] (bool)   : if True, also includes statistics for a 
                                    random permutation of the traces (not 
                                    implemented).
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'all', removes ROIs with NaN/Inf values 
                                    across entire session. If 'no', ROIs with 
                                    NaN/Inf values are not removed.

        - fig_par (dict)   : dictionary containing figure parameters:
                ['bbox'] (str)           : bbox parameter for plt.savefig(), 
                                           e.g., 'tight'
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
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['surp_quint'] (str)     : specific subfolder in which to save 
                                           folder

        - perm_par (dict)    : dictionary containing permutation analysis 
                               parameters:
                ['tails'] (str or int): which tail(s) to test: 'up', 'lo', '2'

        - roi_grp_par (dict) : dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
                ['plot_vals'] (str)  : values to plot 'diff' (surp-nosurp), 
                                       'surp' or 'nosurp'

         - roi_grps (dict): dictionary containing:
                ['grp_names'] (list)     : list of names of the ROI groups in 
                                           roi grp lists (order preserved)
                ['grp_stats'] (4D arrays): array (or nested list) of group stats 
                                           (mean/median, error) structured as:
                                               session x quintile x grp x stat
                ['ns'] (2D array)        : array (or nested list) of group ns, 
                                           structured as:
                                                session x grp

        - sess_info (dict): dictionary containing information from each
                            session 
                ['mouse_ns'] (list)       : mouse numbers
                ['overall_sess_ns'] (list): overall session numbers  
                ['lines'] (list)          : mouse lines
                ['layers'] (list)         : imaging layers
                ['n_rois'] (list)         : number of ROIs in session
                ['nanrois'] (list)        : list of ROIs with NaNs/Infs in raw
                                            traces
                ['nanrois_dff'] (list)    : list of ROIs with NaNs/Infs in dF/F
                                            traces, for sessions for which this 
                                            attribute exists

        - sess_par (dict): ignored (typically saved dictionary that is not 
                           needed for plotting)
                           default: None
        - save_dir (str) : path of directory in which to save plots.
                           default: None
    Returns:
        - full_dir (str):  final name of the directory in which the figure is 
                           saved
                           (may be different from input save_dir, as a datetime 
                           subfolder may have been added depending on the 
                           parameters in fig_par.)
        - save_name (str): name under which the figure is saved, excluding 
                           extension
    """


    # gather mouse info 
    opstr_pr = str_util.op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'], 
                                   True)
    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], 
                                    analys_par['bri_size'], 
                                    analys_par['stim'], 'file')
    statstr = str_util.stat_par_str(basic_par['stats'], basic_par['error'])
    stimstr = str_util.stim_par_str(analys_par['gab_k'], analys_par['bri_dir'], 
                                    analys_par['bri_size'], analys_par['stim'],
                                    'print')
    fluorstr_pr = str_util.fluor_par_str(type_str='print', 
                                         dff=basic_par['dfoverf'])

    # extract some info from dictionaries
    mouse_ns         = sess_info['mouse_ns']
    overall_sess_ns  = sess_info['overall_sess_ns']
    lines            = sess_info['lines']
    layers           = sess_info['layers']
    n_rois           = sess_info['n_rois']
    n_sess = len(mouse_ns)

    if basic_par['remnans'] == 'all':
        if basic_par['dfoverf']:
            n_nans = [len(sess_nans) for sess_nans in sess_info['nanrois_dff']]
        else:
            n_nans = [len(sess_nans) for sess_nans in sess_info['nanrois']]
        n_rois = [n_roi - n_nan for n_roi, n_nan in zip(n_rois, n_nans)]

    grp_st = np.asarray(roi_grps['grp_st'])
    ns     = np.asarray(roi_grps['ns'])

    x_ran = [x+1 for x in range(analys_par['n_quints'])]
    fig, ax = plot_util.init_fig(n_sess, fig_par)
    
    for i, sess_st in enumerate(grp_st):
        sub_ax = plot_util.get_subax(ax, i)
        for g, g_n in enumerate(ns[i]):
            me = sess_st[:, g, 0]
            if basic_par['stats'] == 'median' and basic_par['error'] == 'std':
                yerr1 = me - sess_st[:, g, 1]
                yerr2 = sess_st[:, g, 2] - me
                yerr = [yerr1, yerr2]
            else:
                yerr = sess_st[:, g, 1]
            leg = '{} ({})'.format(roi_grps['grp_names'][g], g_n)
            sub_ax.errorbar(x_ran, me, yerr, fmt='-o', label=leg, alpha=0.8)

        title=(u'Mouse {} - {} {}{} across {} seqs \n(sess {}, {} {}, {} tail '
               '(n={}))').format(mouse_ns[i], statstr, 
                                 analys_par['stim'][0:3], stimstr, 
                                 opstr_pr, overall_sess_ns[i], 
                                 lines[i], layers[i], perm_par['tails'], 
                                 n_rois[i])
        sub_ax.set_title(title)
        sub_ax.set_xticks(x_ran)
        sub_ax.set_ylabel(fluorstr_pr)
        sub_ax.set_xlabel('Quintiles')
        sub_ax.legend()

    if save_dir is None:
        save_dir = os.path.join(fig_par['figdir_roi'], fig_par['surp_quint'])
    save_name = 'roi_{}_grps_{}_{}quint_{}tail'.format(sessstr, 
                    roi_grp_par['plot_vals'], analys_par['n_quints'], 
                    perm_par['tails'])

    full_dir = plot_util.save_fig(fig, save_dir, save_name, fig_par)

    return full_dir, save_name

        
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
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_fr'] (int)          : gabor frame at which segments start 
                                            (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['pre'] (float)           : range of frames to include before 
                                            each frame reference (in s)
                ['post'] (float)          : range of frames to include after  
                                            each frame reference (in s)
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['rand'] (bool)   : if True, also includes statistics for a 
                                    random permutation of the traces (not 
                                    implemented).
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'all', removes ROIs with NaN/Inf values 
                                    across entire session. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'

        - fig_par (dict)   : dictionary containing figure parameters:
                ['bbox'] (str)           : bbox parameter for plt.savefig(), 
                                           e.g., 'tight'
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
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['surp_quint'] (str)     : specific subfolder in which to save 
                                           folder

        - perm_par (dict)    : dictionary containing permutation analysis 
                               parameters:
                ['n_perms'] (int)     : nbr of permutations to run
                ['p_val'] (float)     : p-value to use for significance  
                                        thresholding (0 to 1)
                ['tails'] (str or int): which tail(s) to test: 'up', 'lo', '2'

        - roi_grp_par (dict) : dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
                ['plot_vals'] (str)  : values to plot 'diff' (surp-nosurp), 
                                       'surp' or 'nosurp'
                ['grps'] (str)       : set of groups to return, e.g., 'all', 
                                       'change', 'no_change', 'reduc', 'incr'.
                                       If several sets are passed, each set 
                                        will be collapsed as one group and
                                        'add_nosurp' will be set to False.
                ['add_nosurp'] (bool): if True, group of ROIs showing no 
                                       significance in either is included in the 
                                       groups returned
 
        - sess_par (dict)  : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                               'L5_soma', 'L23_dend', 'L5_dend', 
                                               'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    """

    plotvalstr_pr = str_util.op_par_str(roi_grp_par['plot_vals'], 
                                        roi_grp_par['op'], True)
    sessstr_pr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], analys_par['bri_size'], 
                                    analys_par['stim'], 'print')

    print(('\nAnalysing and plotting {} ROI surp vs nosurp average responses '
           'by quintile ({}). \n{}.').format(plotvalstr_pr, 
                                             analys_par['n_quints'],
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

    roi_grps['grp_st'] = grp_st.tolist()
    roi_grps['ns']     = ns.tolist()
    
    info = {'sess_info': get_sess_info(sessions, dfoverf=basic_par['dfoverf']),
            'sess_par': sess_par,
            'basic_par': basic_par,
            'analys_par': analys_par,
            'perm_par': perm_par,
            'roi_grp_par': roi_grp_par,
            'roi_grps': roi_grps
            }
    
    # plot
    full_dir, save_name = plot_rois_by_grp_qu_sess_from_dicts(fig_par=fig_par, 
                                                              **info)
    
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
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_fr'] (int)          : gabor frame at which segments start 
                                            (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['pre'] (float)           : range of frames to include before 
                                            each frame reference (in s)
                ['post'] (float)          : range of frames to include after  
                                            each frame reference (in s)
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)

        - basic_par (dict)    : dictionary containing additional parameters 
                                relevant to analysis
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['rand'] (bool)   : if True, also includes statistics for a 
                                    random permutation of the traces (not 
                                    implemented).
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'all', removes ROIs with NaN/Inf values 
                                    across entire session. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'

        - roi_grp_par (dict)   : dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
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
def plot_roi_traces_by_grp_from_dicts(analys_par, basic_par, fig_par, perm_par, 
                                      quint_plot, roi_grp_par, roi_grps, 
                                      sess_info, sess_par, traces_data,
                                      save_dir=None):
    """
    plot_roi_traces_by_grp_from_dicts(analys_par, basic_par, fig_par, perm_par, 
                                      quint_plot, roi_grp_par, roi_grps, 
                                      sess_info, sess_par, traces_data)

    From dictionaries, plots ROI traces by group across surprise, no surprise or 
    difference between surprise and no surprise activity per quintile 
    (first/last) with each group in a separate subplot and each session in a 
    different figure.

    Required arguments:
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_fr'] (int)          : gabor frame at which segments start 
                                            (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['pre'] (float)           : range of frames to include before 
                                            each frame reference (in s)
                ['post'] (float)          : range of frames to include after  
                                            each frame reference (in s)
                ['sess_ns'] (list)        : list of session IDs
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['rand'] (bool)   : if True, also includes statistics for a 
                                    random permutation of the traces (not 
                                    implemented).
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'all', removes ROIs with NaN/Inf values 
                                    across entire session. If 'no', ROIs with 
                                    NaN/Inf values are not removed.


        - fig_par (dict)   : dictionary containing figure parameters:
                ['bbox'] (str)           : bbox parameter for plt.savefig(), 
                                           e.g., 'tight'
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
                ['preset_ylims']         : if True, preset y lims are used
                ['sharey'] (bool)        : if True, y axis lims are shared 
                                           across subplots
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['surp_quint'] (str)     : specific subfolder in which to save 
                                           folder

        - perm_par (dict)    : dictionary containing permutation analysis 
                               parameters:
                ['tails'] (str or int): which tail(s) to test: 'up', 'lo', '2'
  
        - quint_plot (dict): dictionary containing information on quintiles to
                             plot:
                ['qu'] (list)    : list of quintile indices to plot,
                ['qu_lab'] (list): list of quintile labels,

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

        - roi_grp_par (dict): dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
                ['plot_vals'] (str)  : values to plot 'diff' (surp-nosurp), 
                                       'surp' or 'nosurp'     
  
        - sess_info (dict): dictionary containing information from each
                            session 
                ['mouse_ns'] (list)       : mouse numbers
                ['overall_sess_ns'] (list): overall session numbers  
                ['lines'] (list)          : mouse lines
                ['layers'] (list)         : imaging layers
                ['n_rois'] (list)         : number of ROIs in session
                ['nanrois'] (list)        : list of ROIs with NaNs/Infs in raw
                                            traces
                ['nanrois_dff'] (list)    : list of ROIs with NaNs/Infs in dF/F
                                            traces, for sessions for which this 
                                            attribute exists

        - sess_par (dict)   : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                          'L5_soma', 'L23_dend', 'L5_dend', 
                                          'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
        
        - traces_data (dict): dictionary containing traces data to plot:
                ['x_ran'] (1D array)    : array or list of time values for the 
                                          frame chunks
                ['grp_stats'] (6D array): array or nested list of statistics f
                                          or ROI groups structured as:
                                                sess x surp x qu x ROI grp x 
                                                    stats x frame
    
    Optional arguments:
        - save_dir (str): path of directory in which to save plots.
                          default: None

    Returns:
        - full_dir (str): final name of the directory in which the figures are 
                          saved 
    """

    opstr_pr = str_util.op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'])
    statstr = str_util.stat_par_str(basic_par['stats'], basic_par['error'])
    stimstr = str_util.stim_par_str(analys_par['gab_k'], analys_par['bri_dir'], 
                                    analys_par['bri_size'], analys_par['stim'],
                                    'print')
    opstr = str_util.op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'], 
                                str_type='file')
    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], analys_par['bri_size'], 
                                    analys_par['stim'], 'file')
    fluorstr_pr = str_util.fluor_par_str(type_str='print', dff=basic_par['dfoverf'])

    if analys_par['stim'] == 'gabors':
        xpos, labels, h_bars, seg_bars = plot_util.plot_seg_comp(analys_par, 
                                                    roi_grp_par['plot_vals'], 
                                                    roi_grp_par['op'])

    # extract some info from dictionaries
    mouse_ns         = sess_info['mouse_ns']
    overall_sess_ns  = sess_info['overall_sess_ns']
    lines            = sess_info['lines']
    layers           = sess_info['layers']
    n_rois           = sess_info['n_rois']
    n_sess = len(mouse_ns)

    if basic_par['remnans'] == 'all':
        if basic_par['dfoverf']:
            n_nans = [len(sess_nans) for sess_nans in sess_info['nanrois_dff']]
        else:
            n_nans = [len(sess_nans) for sess_nans in sess_info['nanrois']]
        n_rois = [n_roi - n_nan for n_roi, n_nan in zip(n_rois, n_nans)]

    x_ran      = np.asarray(traces_data['x_ran'])
    grp_traces = np.asarray(traces_data['grp_traces'])
    
    # Manual y_lims
    # y lims determined from graphs
    if fig_par['preset_ylims']:
        if sess_par['layer'] == 'dend':
            ylims = [[-0.05, 0.2], [-0.2, 0.4], [-0.1, 0.25]] # per mouse
        elif sess_par['layer'] == 'soma':
            ylims = [[-0.3, 0.8], [-0.7, 1.0], [-0.15, 0.25]] # per mouse
        else:
            print('No ylims preset for {}.'.format(sess_par['layer']))

    # figure directories
    if save_dir is None:
        save_dir = os.path.join(fig_par['figdir_roi'], fig_par['surp_quint'], 
                   'grped')

    if fig_par['mult']:
        reset_mult = False
    else:
        fig_par['mult'] = True
        reset_mult = True

    print_dir = True
    for i in range(n_sess):
        fig, ax = plot_util.init_fig(len(roi_grps['all_roi_grps'][i]), fig_par)
        for g, [grp_nam, grp_rois] in enumerate(zip(roi_grps['grp_names'], 
                                                    roi_grps['all_roi_grps'][i])):
            title = '{} group (n={})'.format(grp_nam, len(grp_rois))
            sub_ax = plot_util.get_subax(ax, g)

            if len(grp_rois) == 0:
                sub_ax.set_title(title)
                plot_util.set_ticks(sub_ax, 'x', min(x_ran), max(x_ran))
                continue
            for q in quint_plot['qu']:
                trace_data = np.concatenate([x_ran[np.newaxis, :], 
                                            grp_traces[i, q, g]], axis=0)
                plot_util.plot_traces(sub_ax, trace_data, 
                                      stats=basic_par['stats'], 
                                      error=basic_par['error'], title=title,
                                      alpha=0.8/len(quint_plot['qu']), 
                                      dff=basic_par['dfoverf'])
            if fig_par['preset_ylims']:
                sub_ax.set_ylim(ylims[i])
            if analys_par['stim'] == 'gabors':
                plot_util.add_bars(sub_ax, hbars=h_bars, bars=seg_bars)
            sub_ax.set_ylabel(fluorstr_pr)
            sub_ax.set_xlabel('Time (s)')
            sub_ax.legend(quint_plot['qu_lab'])
    
        if analys_par['stim'] == 'gabors':
            t_hei = 0.85
            plot_util.incr_ymax(ax, incr=1.05/t_hei, sharey=fig_par['sharey'])
            for g, grp_rois in enumerate(roi_grps['all_roi_grps'][i]):
                if len(grp_rois) == 0:
                    continue
                sub_ax = plot_util.get_subax(ax, g)
                plot_util.add_labels(sub_ax, labels, xpos, t_hei, 'k')
            
        fig.suptitle((u'Mouse {} - {} {}{} - {} seqs for diff quint\n'
                    '(sess {}, {} {}, {} tail (n={}))')
                        .format(mouse_ns[i], statstr, analys_par['stim'][0:3], 
                                stimstr, opstr_pr, overall_sess_ns[i], lines[i],
                                layers[i], perm_par['tails'], n_rois[i]))

        save_name = ('roi_tr_m{}_{}_grps_{}_{}quint_'
                        '{}tail').format(mouse_ns[i], sessstr, opstr, 
                                         analys_par['n_quints'], 
                                         perm_par['tails'])
        
        full_dir = plot_util.save_fig(fig, save_dir, save_name, fig_par, 
                                      print_dir=print_dir)
        print_dir = False

    # resetting the fig_par
    if reset_mult:
        fig_par['mult'] = False
        fig_par['prev_dt'] = None

    return full_dir


#############################################
def plot_roi_traces_by_grp(sessions, quint_plot, roi_grps, n_rois, 
                           analys_par, basic_par, fig_par, perm_par, 
                           roi_grp_par, sess_par, save_dict=True):
    """
    plot_roi_traces_by_grp(sessions, quint_plot, roi_grps, n_rois, 
                           analys_par, basic_par, fig_par, roi_grp_par, sess_par)

    Plots ROI traces by group across surprise, no surprise or difference between 
    surprise and no surprise activity per quintile (first/last) with each group 
    in a separate subplot and each session in a different figure.

    Optionally saves results and parameters relevant to analysis in a 
    dictionary.

    Required arguments:
        - sessions (list)  : list of Session objects
        - quint_plot (dict): dictionary containing information on quintiles to
                             plot:
                ['qu'] (list)    : list of quintile indices to plot,
                ['qu_lab'] (list): list of quintile labels,

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
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_fr'] (int)          : gabor frame at which segments start 
                                            (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['pre'] (float)           : range of frames to include before 
                                            each frame reference (in s)
                ['post'] (float)          : range of frames to include after  
                                            each frame reference (in s)
                ['sess_ns'] (list)        : list of session IDs
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'

        - fig_par (dict)   : dictionary containing figure parameters:
                ['bbox'] (str)           : bbox parameter for plt.savefig(), 
                                           e.g., 'tight'
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
                ['preset_ylims']         : if True, preset y lims are used
                ['sharey'] (bool)        : if True, y axis lims are shared 
                                           across subplots
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['surp_quint'] (str)     : specific subfolder in which to save 
                                           folder

        - perm_par (dict)    : dictionary containing permutation analysis 
                               parameters:
                ['tails'] (str or int): which tail(s) to test: 'up', 'lo', '2'
  
        - roi_grp_par (dict): dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
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
        - traces_data (dict): dictionary containing traces data to plot:
                ['x_ran'] (1D array)    : array or list of time values for the 
                                          frame chunks
                ['grp_stats'] (6D array): array or nested list of statistics f
                                          or ROI groups structured as:
                                                sess x surp x qu x ROI grp x 
                                                    stats x frame
    """

    opstr_pr = str_util.op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'])
    sessstr_pr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], analys_par['bri_size'], 
                                    analys_par['stim'], 'print')
    opstr = str_util.op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'], 
                                str_type='file')
    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], analys_par['bri_size'], 
                                    analys_par['stim'], 'file')

    print(('\nAnalysing and plotting {} ROI surp vs nosurp traces by '
           'quintile ({}). \n{}.').format(opstr_pr, analys_par['n_quints'], 
                                          sessstr_pr))

    # get statistics for each group
    # sess x surp x qu x ROI grps x stats x frames
    x_ran, grp_traces = grp_traces_by_qu_surp_sess(sessions, 
                                                   roi_grps['all_roi_grps'],
                                                   analys_par, basic_par, 
                                                   roi_grp_par, quint_plot['qu'])

    traces_data = {'x_ran'     : x_ran.tolist(),
                   'grp_traces': grp_traces.tolist()
                  }

    info = {'analys_par' : analys_par,
            'basic_par'  : basic_par,
            'perm_par'   : perm_par,
            'quint_plot' : quint_plot,
            'roi_grp_par': roi_grp_par,
            'roi_grps'   : roi_grps,
            'sess_info'  : get_sess_info(sessions, dfoverf=basic_par['dfoverf']),
            'sess_par'   : sess_par,
            'traces_data': traces_data
            }

    full_dir = plot_roi_traces_by_grp_from_dicts(fig_par=fig_par, **info)

    if save_dict:
        info_name = ('roi_tr_{}_grps_{}_{}quint_'
                        '{}tail').format(sessstr, opstr, analys_par['n_quints'], 
                                         perm_par['tails'])

        file_util.save_info(info, info_name, full_dir, 'json')

    return full_dir, traces_data


#############################################
def plot_roi_areas_by_grp_from_dicts(analys_par, areas_data, basic_par, fig_par, 
                                     perm_par, quint_plot, roi_grp_par, 
                                     roi_grps, sess_info, sess_par, 
                                     save_dir=None):
    """
    plot_roi_areas_by_grp_from_dicts(analys_par, areas_data, basic_par, fig_par, 
                                     perm_par, quint_plot, roi_grp_par, 
                                     roi_grps, sess_info, sess_par)

    From dictionaries, plots ROI traces by group across surprise, no surprise 
    or difference between surprise and no surprise activity per quintile 
    (first/last) with each group in a separate subplot and each session in a 
    different figure.

    Required arguments:
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)

        - areas_data (dict): dictionary containing data to plot:
                ['all_grp_st'] (4D array)     : array or nested list of group 
                                                stats (mean/median, error) 
                                                structured as:
                                                    session x quintile x grp x 
                                                    stat
                ['all_grp_st_norm'] (4D array): same as 'all_grp_st', but with
                                                normalized group stats 
                                                
        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'
                                    
        - fig_par (dict)   : dictionary containing figure parameters:
                ['bbox'] (str)           : bbox parameter for plt.savefig(), 
                                           e.g., 'tight'
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
                ['sharey'] (bool)        : if True, y axis lims are shared 
                                           across subplots
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['surp_quint'] (str)     : specific subfolder in which to save 
                                           folder

        - perm_par (dict)    : dictionary containing permutation analysis 
                               parameters:
                ['n_perms'] (int)     : nbr of permutations to run

        - quint_plot (dict): dictionary containing information on quintiles to
                             plot:
                ['qu'] (list)    : list of quintile indices to plot,
                ['qu_lab'] (list): list of quintile labels,

        - roi_grp_par (dict): dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
                ['plot_vals'] (str)  : values to plot 'diff' (surp-nosurp), 
                                      'surp' or 'nosurp' 

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

        - sess_info (dict): dictionary containing information from each
                            session 
                ['mouse_ns'] (list)       : mouse numbers
                ['overall_sess_ns'] (list): overall session numbers  
                ['lines'] (list)          : mouse lines
                ['layers'] (list)         : imaging layers
                ['n_rois'] (list)         : number of ROIs in session
                ['nanrois'] (list)        : list of ROIs with NaNs/Infs in raw
                                            traces
                ['nanrois_dff'] (list)    : list of ROIs with NaNs/Infs in dF/F
                                            traces, for sessions for which this 
                                            attribute exists

        - sess_par (dict)  : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                          'L5_soma', 'L23_dend', 'L5_dend', 
                                          'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    
    Returns:
        - full_dir (str): final name of the directory in which the figures are 
                          saved 
    """
    opstr_pr = str_util.op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'], 
                                   True)
    statstr = str_util.stat_par_str(basic_par['stats'], basic_par['error'])
    stimstr = str_util.stim_par_str(analys_par['gab_k'], analys_par['bri_dir'], 
                                    analys_par['bri_size'], analys_par['stim'],
                                    'print')
    opstr = str_util.op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'], 
                                True, str_type='file')
    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], analys_par['bri_size'], 
                                    analys_par['stim'], 'file')
    fluorstr_pr = str_util.fluor_par_str(type_str='print', dff=basic_par['dfoverf'])


    # extract some info from dictionaries
    mouse_ns         = sess_info['mouse_ns']
    overall_sess_ns  = sess_info['overall_sess_ns']
    lines            = sess_info['lines']
    layers           = sess_info['layers']
    n_rois           = sess_info['n_rois']
    n_sess = len(mouse_ns)

    if basic_par['remnans'] == 'all':
        if basic_par['dfoverf']:
            n_nans = [len(sess_nans) for sess_nans in sess_info['nanrois_dff']]
        else:
            n_nans = [len(sess_nans) for sess_nans in sess_info['nanrois']]
        n_rois = [n_roi - n_nan for n_roi, n_nan in zip(n_rois, n_nans)]

    grp_st = np.asarray(areas_data['grp_st'])
    grp_st_norm = np.asarray(areas_data['grp_st_norm'])
    
    # figure directories
    if save_dir is None:
        save_dir = os.path.join(fig_par['figdir_roi'], fig_par['surp_quint'], 
                                'grped')
    
    # for spacing the bars on the graph
    xpos = [2*x+1 for x in range(len(quint_plot['qu']))]
    xlims = [0, len(xpos)*2]

    if fig_par['mult']:
        reset_mult = False
    else:
        fig_par['mult'] = True
        reset_mult = True

    print_dir = True
    for i in range(n_sess):
        fig, ax = plot_util.init_fig(len(roi_grps['all_roi_grps'][i]), fig_par, 
                                     div=2.0)
        fignorm, axnorm = plot_util.init_fig(len(roi_grps['all_roi_grps'][i]), 
                                             fig_par, div=2.0)
        for axis, norm in zip([ax, axnorm], [False, True]):
            for g, [grp_nam, grp_rois] in enumerate(zip(roi_grps['grp_names'], 
                                                        roi_grps['all_roi_grps'][i])):
                title = '{} group (n={})'.format(grp_nam, len(grp_rois))
                sub_ax = plot_util.get_subax(axis, g)
                sub_ax.tick_params(labelbottom=False)
                if len(grp_rois) == 0:
                    sub_ax.tick_params(axis='x', which='both', bottom=False) 
                    if not norm:
                        sub_ax.set_title(title)
                    else:
                        sub_ax.set_title('{} (norm)'.format(title))
                    continue
                for j, q in enumerate(quint_plot['qu']):
                    if not norm:
                        vals = grp_st
                    else:
                        vals = grp_st_norm

                    plot_util.plot_bars(sub_ax, xpos[j], vals[i, q, g, 0], 
                                        vals[i, q, g, 1:], title, alpha=0.5, 
                                        xticks='None', xlims=xlims, 
                                        label=quint_plot['qu_lab'][j], 
                                        hline=0, dff=basic_par['dfoverf'])
                if not norm:
                    sub_ax.set_ylabel(u'{} area'.format(fluorstr_pr))
                else:
                    sub_ax.set_ylabel(u'{} area (norm)'.format(fluorstr_pr))

        suptitle = (u'Mouse {} - {} {}{} - {} seqs for diff quint\n(sess {}, '
                    '{} {}, {} tail (n={}))').format(mouse_ns[i], statstr, 
                                                     analys_par['stim'][0:3],
                                                     stimstr, opstr_pr,
                                                     overall_sess_ns[i], 
                                                     lines[i], layers[i],
                                                     perm_par['tails'], 
                                                     n_rois[i])

        fig.suptitle(suptitle)
        fignorm.suptitle(u'{} (norm)'.format(suptitle))

        save_name = ('roi_area_m{}_{}_grps_{}_{}quint_'
                        '{}tail').format(mouse_ns[i], sessstr, opstr, 
                                         analys_par['n_quints'], 
                                         perm_par['tails'])
        save_name_norm = '{}_norm'.format(save_name)

        full_dir = plot_util.save_fig(fig, save_dir, save_name, fig_par, 
                                      print_dir=print_dir)
        print_dir = False

        _ = plot_util.save_fig(fignorm, save_dir, save_name_norm, fig_par, 
                                      print_dir=print_dir)
    
    if reset_mult:
        fig_par['mult'] = False
        fig_par['prev_dt'] = None

    return full_dir


#############################################
def plot_roi_areas_by_grp(sessions, integ_dffs, quint_plot, roi_grps, n_rois,
                          analys_par, basic_par, fig_par, perm_par, roi_grp_par, 
                          sess_par, save_dict=False):
    """
    plot_roi_traces_by_grp(sessions, integ_dffs, quint_plot, roi_grps, 
                           analys_par, basic_par, fig_par, roi_grp_par, sess_par)

    Plots ROI traces by group across surprise, no surprise or difference between 
    surprise and no surprise activity per quintile (first/last) with each group 
    in a separate subplot and each session in a different figure. 

    Optionally saves results and parameters relevant to analysis in a 
    dictionary.

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
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_fr'] (int)          : gabor frame at which segments start 
                                            (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['pre'] (float)           : range of frames to include before 
                                            each frame reference (in s)
                ['post'] (float)          : range of frames to include after  
                                            each frame reference (in s)
                ['sess_ns'] (list)        : list of session IDs
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'
                                    
        - fig_par (dict)   : dictionary containing figure parameters:
                ['bbox'] (str)           : bbox parameter for plt.savefig(), 
                                           e.g., 'tight'
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
                ['sharey'] (bool)        : if True, y axis lims are shared 
                                           across subplots
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['surp_quint'] (str)     : specific subfolder in which to save 
                                           folder

        - perm_par (dict)    : dictionary containing permutation analysis 
                               parameters:
                ['n_perms'] (int)     : nbr of permutations to run
                ['p_val'] (float)     : p-value to use for significance  
                                        thresholding (0 to 1)
                ['tails'] (str or int): which tail(s) to test: 'up', 'lo', '2'
                                        
        - roi_grp_par (dict): dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
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
                                   True)
    sessstr_pr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], analys_par['bri_size'], 
                                    analys_par['stim'], 'print')
    opstr = str_util.op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'], 
                                True, str_type='file')
    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], analys_par['bri_size'], 
                                    analys_par['stim'], 'file')

    print(('\nAnalysing and plotting {} ROI surp vs nosurp average responses '
           'by quintile ({}). \n{}.').format(opstr_pr, analys_par['n_quints'],
                                             sessstr_pr))
    
    # get statistics per group and number of ROIs per group
    grp_st, _ = grp_stats(integ_dffs, roi_grps['all_roi_grps'], 
                          roi_grp_par['plot_vals'], roi_grp_par['op'], 
                          basic_par['stats'], basic_par['error'])
    # sess x quint x grp x stat
    grp_st_norm, _ = grp_stats(integ_dffs, roi_grps['all_roi_grps'], 
                               roi_grp_par['plot_vals'], roi_grp_par['op'], 
                               basic_par['stats'], basic_par['error'], True)

    areas_data = {'grp_st': grp_st.tolist(),
                  'grp_st_norm': grp_st_norm.tolist(),
                  }

    info = {'analys_par' : analys_par,
            'basic_par'  : basic_par,
            'perm_par'   : perm_par,
            'quint_plot' : quint_plot,
            'roi_grp_par': roi_grp_par,
            'roi_grps'   : roi_grps,
            'sess_info'  : get_sess_info(sessions, dfoverf=basic_par['dfoverf']),
            'sess_par'   : sess_par,
            'areas_data' : areas_data
            }
        
    full_dir = plot_roi_areas_by_grp_from_dicts(fig_par=fig_par, **info)

    if save_dict:
        info_name = ('roi_area_{}_grps_{}_{}quint_'
                        '{}tail').format(sessstr, opstr, analys_par['n_quints'], 
                                         perm_par['tails'])
        file_util.save_info(info, info_name, full_dir, 'json')
    
    return full_dir, areas_data


#############################################
def plot_rois_by_grp_from_dicts(analys_par, basic_par, fig_par, perm_par, 
                                quint_plot, roi_grp_par, roi_grps, sess_info,
                                sess_par, traces_data=None, areas_data=None, 
                                save_dir=None):
    """
    plot_rois_by_grp_from_dicts(analys_par, basic_par, fig_par, perm_par, 
                               quint_plot, roi_grp_par, roi_grps, sess_info, 
                               sess_par)

    From dictionaries, plots ROI data by group across surprise, no surprise 
    or difference between surprise and no surprise activity per quintile 
    (first/last) with each group in a separate subplot and each session in a 
    different figure.

    Two types of ROI data are plotted:
        1. ROI traces, if traces_data is passed
        2. ROI areas, if areas_data is passed 

    Required arguments:
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)
                                        
        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'
                                    
        - fig_par (dict)   : dictionary containing figure parameters:
                ['bbox'] (str)           : bbox parameter for plt.savefig(), 
                                           e.g., 'tight'
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
                ['sharey'] (bool)        : if True, y axis lims are shared 
                                           across subplots
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['surp_quint'] (str)     : specific subfolder in which to save 
                                           folder

        - perm_par (dict)    : dictionary containing permutation analysis 
                               parameters:
                ['n_perms'] (int)     : nbr of permutations to run

        - quint_plot (dict): dictionary containing information on quintiles to
                             plot:
                ['qu'] (list)    : list of quintile indices to plot,
                ['qu_lab'] (list): list of quintile labels,

        - roi_grp_par (dict): dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
                ['plot_vals'] (str)  : values to plot 'diff' (surp-nosurp), 
                                      'surp' or 'nosurp' 

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

        - sess_info (dict): dictionary containing information from each
                            session 
                ['mouse_ns'] (list)       : mouse numbers
                ['overall_sess_ns'] (list): overall session numbers  
                ['lines'] (list)          : mouse lines
                ['layers'] (list)         : imaging layers
                ['n_rois'] (list)         : number of ROIs in session
                ['nanrois'] (list)        : list of ROIs with NaNs/Infs in raw
                                            traces
                ['nanrois_dff'] (list)    : list of ROIs with NaNs/Infs in dF/F
                                            traces, for sessions for which this 
                                            attribute exists

        - sess_par (dict)  : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                          'L5_soma', 'L23_dend', 'L5_dend', 
                                          'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    
    Optional arguments:
        - areas_data (dict): dictionary containing data to plot:
                             default: None
                ['all_grp_st'] (4D array)     : array or nested list of group 
                                                stats (mean/median, error) 
                                                structured as:
                                                    session x quintile x grp x 
                                                    stat
                ['all_grp_st_norm'] (4D array): same as 'all_grp_st', but with
                                                normalized group stats 

        - traces_data (dict): dictionary containing traces data to plot:
                ['x_ran'] (1D array)    : array or list of time values for the 
                                          frame chunks
                ['grp_stats'] (6D array): array or nested list of statistics f
                                          or ROI groups structured as:
                                                sess x surp x qu x ROI grp x 
                                                    stats x frame
                              default: None
        - save_dir (str)    : path of directory in which to save plots.
                              default: None   
    """

    # ensure that plots are all saved in same file
    if fig_par['mult']:
        reset_mult = False
    else:
        fig_par['mult'] = True
        reset_mult = True

    comm_info = {'analys_par' : analys_par,
                 'basic_par'  : basic_par,
                 'fig_par'    : fig_par,
                 'perm_par'   : perm_par,
                 'quint_plot' : quint_plot,
                 'roi_grp_par': roi_grp_par,
                 'roi_grps'   : roi_grps,
                 'sess_info'  : sess_info,
                 'sess_par'   : sess_par
                 }

    if traces_data is not None:
        plot_roi_traces_by_grp_from_dicts(traces_data=traces_data, 
                                          save_dir=save_dir, **comm_info)

    if areas_data is not None:
        plot_roi_areas_by_grp_from_dicts(areas_data=areas_data, 
                                         save_dir=save_dir, **comm_info)

    # resetting the fig_par
    if reset_mult:
        fig_par['mult'] = False
        fig_par['prev_dt'] = None


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
    in a different figure. Saves results and parameters relevant to analysis
    in a dictionary.

    Required arguments:
        - sessions (list)  : list of Session objects
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_fr'] (int)          : gabor frame at which segments start 
                                            (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['pre'] (float)           : range of frames to include before 
                                            each frame reference (in s)
                ['post'] (float)          : range of frames to include after  
                                            each frame reference (in s)
                ['sess_ns'] (list)        : list of session IDs
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'

        - fig_par (dict)   : dictionary containing figure parameters:
                ['bbox'] (str)           : bbox parameter for plt.savefig(), 
                                           e.g., 'tight'
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
                ['sharey'] (bool)        : if True, y axis lims are shared 
                                           across subplots
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['surp_quint'] (str)     : specific subfolder in which to save 
                                           folder

        - perm_par (dict)    : dictionary containing permutation analysis 
                               parameters:
                ['n_perms'] (int)     : nbr of permutations to run
                ['p_val'] (float)     : p-value to use for significance  
                                        thresholding (0 to 1)
                ['tails'] (str or int): which tail(s) to test: 'up', 'lo', '2'

        - roi_grp_par (dict): dictionary containing ROI grouping parameters:
                ['op'] (str)         : operation to use to compare groups, 
                                       i.e. 'diff': grp1-grp2, or 'ratio': 
                                       grp1/grp2
                ['plot_vals'] (str)  : values to plot 'diff' (surp-nosurp), 
                                      'surp' or 'nosurp' 

        - sess_par (dict)  : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                          'L5_soma', 'L23_dend', 'L5_dend', 
                                          'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    """

    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], analys_par['bri_size'], 
                                    analys_par['stim'], 'file')

    opstr = str_util.op_par_str(roi_grp_par['plot_vals'], roi_grp_par['op'], 
                                str_type='file')

    # quintiles to plot
    quint_plot = {'qu': [0, -1], # must correspond to indices
                  'qu_lab': ['first quint', 'last quint'],
                 }
    
    [integ_dffs, integ_dffs_rel, 
                         n_rois] = integ_per_grp_qu_sess(sessions, analys_par, 
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

    _, traces_data = plot_roi_traces_by_grp(sessions, quint_plot, roi_grps, 
                                            n_rois, analys_par, basic_par, 
                                            fig_par, perm_par, roi_grp_par, 
                                            sess_par, save_dict=False)

    full_dir, areas_data = plot_roi_areas_by_grp(sessions, integ_dffs, 
                                                 quint_plot, roi_grps, 
                                                 n_rois, analys_par, basic_par, 
                                                 fig_par, perm_par, roi_grp_par, 
                                                 sess_par, save_dict=False)

    # resetting the fig_par
    if reset_mult:
        fig_par['mult'] = False
        fig_par['prev_dt'] = None

    info = {'sess_par'   : sess_par,
            'sess_info'  : get_sess_info(sessions, dfoverf=basic_par['dfoverf']),
            'basic_par'  : basic_par,
            'analys_par' : analys_par,
            'perm_par'   : perm_par,
            'quint_plot' : quint_plot,
            'roi_grp_par': roi_grp_par,
            'roi_grps'   : roi_grps,
            'traces_data': traces_data,
            'areas_data' : areas_data
            }
    
    info_name = ('roi_{}_grps_{}_{}quint_'
                    '{}tail').format(sessstr, opstr, analys_par['n_quints'], 
                                     perm_par['tails'])

    file_util.save_info(info, info_name, full_dir, 'json')


############################################
def plot_mag_change_from_dicts(analys_par, basic_par, fig_par, mags, sess_info, 
                               sess_par, save_dir=None):
    """
    From dictionaries, plots autocorrelation during stimulus blocks.

    Required arguments:
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                          (e.g., 128, 256 or [128, 256])
                ['gab_k'] (int or list) : gabor kappa values to include 
                                        (e.g., 4, 16 or [4, 16])
                ['sess_ns'] (list)      : list of session IDs
                ['stim'] (str)          : stimulus to analyse (bricks or gabors)

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'all', removes ROIs with NaN/Inf values 
                                    across entire session. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'

        - fig_par (dict)   : dictionary containing figure parameters:
                ['autocorr'] (str)       : specific subfolder in which to save 
                                           folder  
                ['bbox'] (str)           : bbox parameter for plt.savefig(), 
                                           e.g., 'tight'
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
                ['sharey'] (bool)        : if True, y axis lims are shared 
                                           across subplots
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['subplot_wid'] (float)  : width of each subplot (inches)

        - mags (dict): dictionary containing magnitude data to plot
                ['all_L2s'] (2d array)     : array or nested list containing L2 
                                             norms, structured as: 
                                                sess x surp
                ['mag_me'] (2d array)      : array or nested list containing 
                                             magnitude means or medians, 
                                             structured as: 
                                                sess x surp
                ['mag_me_norm'] (2d array) : array or nested list containing 
                                             means or medians of normalized 
                                             magnitude, structured as: 
                                                sess x surp
                ['mag_de'] (3d array)      : array or nested list containing
                                             error for mean or median 
                                             magnitudes, structured as:
                                                sess x surp x stats
                ['mag_de_norm'] (3d array) : array or nested list containing
                                             error for mean or median of
                                             normalized magnitudes, structured 
                                             as:
                                                sess x surp x stats
                ['p_vals'] (1D array)      : array or list with p_values for 
                                             each session

        - sess_info (dict): dictionary containing information from each
                            session 
                ['mouse_ns'] (list)       : mouse numbers
                ['lines'] (list)          : mouse lines
                ['n_rois'] (list)         : number of ROIs in session
                ['nanrois'] (list)        : list of ROIs with NaNs/Infs in raw
                                            traces
                ['nanrois_dff'] (list)    : list of ROIs with NaNs/Infs in dF/F
                                            traces, for sessions for which this 
                                            attribute exists    

        - sess_par (dict)  : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                            'L5_soma', 'L23_dend', 'L5_dend', 
                                            'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    
    Optional arguments:
            - save_dir (str)    : path of directory in which to save plots.
                              default: None  

    Returns:
        - full_dir (str):  final name of the directory in which the figure is 
                           saved
                           (may be different from input save_dir, as a datetime 
                           subfolder may have been added depending on the 
                           parameters in fig_par.)
        - save_name (str): name under which the figure is saved, excluding 
                           extension 
    """
    
    statstr = str_util.stat_par_str(basic_par['stats'], basic_par['error'])

    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], analys_par['bri_size'], 
                                    analys_par['stim'], 'file')
    sessstr_pr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], analys_par['bri_size'], 
                                    analys_par['stim'], 'print')
    fluorstr_pr = str_util.fluor_par_str(type_str='print', dff=basic_par['dfoverf'])

    # extract some info from dictionaries
    mouse_ns         = sess_info['mouse_ns']
    lines            = sess_info['lines']
    n_rois           = sess_info['n_rois']
    n_sess = len(mouse_ns)

    if basic_par['remnans'] == 'all':
        if basic_par['dfoverf']:
            n_nans = [len(sess_nans) for sess_nans in sess_info['nanrois_dff']]
        else:
            n_nans = [len(sess_nans) for sess_nans in sess_info['nanrois']]
        n_rois = [n_roi - n_nan for n_roi, n_nan in zip(n_rois, n_nans)]

    # convert mags items to lists
    np_mags = dict()
    for key in mags.keys():
        np_mags[key] = np.asarray(mags[key])

    # create figure
    barw = 0.75
    leg = ['nosurp', 'surp']
    div = 2.0/n_sess
    fig, ax = plt.subplots(figsize=(fig_par['subplot_wid']/div, 
                                    fig_par['subplot_hei']))
    fignorm, axnorm = plt.subplots(figsize=(fig_par['subplot_wid']/div, 
                                   fig_par['subplot_hei']))

    pos = [3*len(leg)*x+1 + len(leg)/2.0 for x in range(n_sess)]
    xlims = [0, 3*len(leg)*len(pos)-1.5]

    if fig_par['mult']:
        reset_mult = False
    else:
        fig_par['mult'] = True
        reset_mult = True

    print_dir = True
    for s, lab in enumerate(leg):
        xpos = [x - len(leg)/2.0 + 2*s for x in pos]
        
        plot_util.plot_bars(ax, xpos, np_mags['mag_me'][:, s], 
                            err=np_mags['mag_de'][:, s], width=barw, 
                            xlims=xlims, xticks='None', label=lab, capsize=4)

        plot_util.plot_bars(axnorm, xpos, np_mags['mag_me_norm'][:, s], 
                            err=np_mags['mag_de_norm'][:, s], width=barw, 
                            xlims=xlims, xticks='None', label=lab, capsize=4)

    labels = ['Mouse {}, {}\n(n={})'.format(mouse_ns[i], lines[i], n_rois[i]) 
               for i in range(n_sess)]

    title = ((u'Magnitude ({}) of difference in activity across ROIs per '
               'mouse \n({})').format(statstr, sessstr_pr))

    for axis, normstr in zip([ax, axnorm], ['', ' (norm)']):
        axis.set_xticks(pos)
        axis.set_xticklabels(labels)
        axis.set_title(u'{}{}'.format(title, normstr))
        axis.set_ylabel(u'Magnitude difference in {}{}'.format(fluorstr_pr, 
                                                               normstr))

    if save_dir is None:
        save_dir = os.path.join(fig_par['figdir_roi'], fig_par['surp_quint'], 'mags')
    save_name = ('roi_mag_diff_{}').format(sessstr)
    save_name_norm = '{}_norm'.format(save_name)

    full_dir = plot_util.save_fig(fig, save_dir, save_name, fig_par, 
                                  print_dir=print_dir)
    print_dir = False

    _ = plot_util.save_fig(fignorm, save_dir, save_name_norm, fig_par, 
                           print_dir=print_dir)

    # resetting the fig_par
    if reset_mult:
        fig_par['mult'] = False
        fig_par['prev_dt'] = None

    return full_dir, save_name


#############################################
def plot_mag_change(sessions, analys_par, basic_par, fig_par, sess_par):
    """
    plot_mag_change(sessions, analys_par, basic_par, fig_par,  
                            perm_par, roi_grp_par, sess_par)

    Plots the magnitude of change in activity of ROIs between the first and
    last quintile for non surprise vs surprise segments.
    Saves results and parameters relevant to analysis in a dictionary.

    Required arguments:
        - sessions (list)  : list of Session objects
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_fr'] (int)          : gabor frame at which segments start 
                                            (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['pre'] (float)           : range of frames to include before 
                                            each frame reference (in s)
                ['post'] (float)          : range of frames to include after  
                                            each frame reference (in s)
                ['sess_ns'] (list)        : list of session IDs
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['rand'] (bool)   : if True, also includes statistics for a 
                                    random permutation of the traces (not 
                                    implemented).
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'all', removes ROIs with NaN/Inf values 
                                    across entire session. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'

        - fig_par (dict)   : dictionary containing figure parameters:
                ['bbox'] (str)           : bbox parameter for plt.savefig(), 
                                           e.g., 'tight'
                ['datetime'] (bool)      : if True, figures are saved in a 
                                            subfolder named based on the date 
                                            and time.
                ['fig_ext'] (str)        : extension (without '.') with 
                                            which to save figure
                ['figdir_roi'] (str)     : main folder in which to save figure
                ['overwrite'] (bool)     : if False, overwriting existing 
                                            figures is prevented by adding 
                                            suffix numbers.
                ['mult'] (bool)          : if True, prev_dt is created or used.
                ['prev_dt'] (str)        : datetime folder to use 
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['surp_quint'] (str)     : specific subfolder in which to save 
                                            folder 

        - sess_par (dict)  : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                            'L5_soma', 'L23_dend', 'L5_dend', 
                                            'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    """

    sessstr_pr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], analys_par['bri_size'], 
                                    analys_par['stim'], 'print')
    
    print(('Calculating and plotting the magnitude changes in ROI activity '
           'across quintiles \n({})').format(sessstr_pr))

    # get sess x surp x quint x ROIs x frames
    chunk_info = chunk_stats_by_qu_sess(sessions, analys_par, basic_par, 
                                        byroi=True, data='me', bysurp=True,
                                        twop_fps=True)

    integ_data = integ_per_qu_surp_sess(chunk_info[1], chunk_info[-1], None)
    
    sess_info = get_sess_info(sessions, dfoverf=basic_par['dfoverf'])

    # get magnitude differences (magnitude of change for surprise vs non 
    # surprise segments)
    if basic_par['stats'] == 'median' and basic_par['error'] == 'std':
        stat_len = 2
    else:
        stat_len = 1

    mags = {'all_L2s': np.empty([len(sessions), 2]),
            'mag_me' : np.empty([len(sessions), 2]),
            'mag_me_norm' : np.empty([len(sessions), 2]),
            'mag_de' : np.empty([len(sessions), 2, stat_len]).squeeze(axis=-1),
            'mag_de_norm' : np.empty([len(sessions), 2, stat_len]).squeeze(axis=-1),
            'p_vals': np.empty([len(sessions)])
            }  

    print('\nMagnitude of the difference in activity across ROIs per mouse.')
    for i in range(len(sessions)):
        print('\nMouse {}, {}:'.format(sess_info['mouse_ns'][i], 
                                         sess_info['lines'][i]))
        abs_diffs = np.absolute(integ_data[i][:, -1] - integ_data[i][:, 0]) # /integ_data[i][:, 0])
        abs_diffs_norm = math_util.calc_norm(abs_diffs, dimpos=[0, 0], out_range='onepos')
        for s in [0, 1]:
            # abs difference in average integrated areas across ROIs between last 
            # and first quintiles
            mags['all_L2s'][i, s] = np.linalg.norm(abs_diffs[s], ord=2) # take L2 norm
            
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

        for val, val_name in zip(['all_L2s', 'mag_me'], ['L2 norm', 'Mag']):
            for s, surp in zip([0, 1], ['(non surp)', '(surp)    ']):
                print('    {} {}: {:.2f}'.format(val_name, surp, 
                                                 mags[val][i, s]))
        print('    Mag p-value: {:.4f}'.format(mags['p_vals'][i]))
    
    # convert mags items to lists
    for key in mags.keys():
        mags[key] = mags[key].tolist()

    info = {'analys_par': analys_par,
            'basic_par': basic_par,
            'mags': mags,
            'sess_info': sess_info,
            'sess_par': sess_par,
            }
    
    full_dir, save_name = plot_mag_change_from_dicts(fig_par=fig_par, **info)

    file_util.save_info(info, save_name, full_dir, 'json')


#############################################
def lfads_dict(sessions, mouse_df, runtype, output=''):
    """
    lfads_dict(sessions, mouse_df, gabfr)

    Creates and saves dictionary containing information relevant to sessions.
    Currently set to use only specific gabfr, gabk and bri_size values.

    Arguments:
        - sessions (list)     : list of Session objects
        - mouse_df (pandas df): dataframe containing parameters for each 
                                session.

    Optional arguments:
        - output (str)          : output directory
                                  default: ''

    """
    
    gabfr = 0
    gabk = 16
    bri_size = 128

    stim_types = ['gabors', 'bricks']
    stim_dirs  = [[''], ['right', 'left']]
    stim_diffs = [4, 1]
    stim_keys  = ['gab', 'bri_']
    stim_seps  = ['', ' ']
    stim_omit  = [[pilot_gab_omit(gabk)], 
                  [pilot_bri_omit(stim_dirs[1][0], bri_size), 
                   pilot_bri_omit(stim_dirs[1][1], bri_size)]]

    all_gabfr = ['A', 'B', 'C', 'D/E']

    if isinstance(gabfr, list):
        gabfr_let = [all_gabfr[x] for x in gabfr]
    else:
        gabfr_let = all_gabfr[gabfr]
    sessions = gen_util.list_if_not(sessions)
    for sess in sessions:
        name = 'sess_dict_mouse{}_sess{}_{}'.format(sess.mouse_n, 
                                                    sess.sess_overall, sess.layer)
        print('\nCreating stimulus dictionary: {}'.format(name))

        sess.create_dff()
        if runtype == 'pilot':
            roi_tr_dir = sess.roi_traces[sess.roi_traces.find('ophys'):]
            roi_dff_dir = sess.roi_traces_dff[sess.roi_traces_dff.find('ophys'):]
        elif runtype == 'prod':
            roi_tr_dir = sess.roi_traces[sess.roi_traces.find('mouse'):]
            roi_dff_dir = sess.roi_traces_dff[sess.roi_traces_dff.find('mouse'):]
        
        # get first and last left and first right seg for bricks


        sess_dict = {'sessionid'     : sess.session,
                     'mouse'         : sess.mouse_n,
                     'act_sess_n'    : sess.sess_overall,
                     'depth'         : sess.depth,
                     'layer'         : sess.layer,
                     'line'          : sess.line,
                     'traces_dir'    : roi_tr_dir,
                     'dff_traces_dir': roi_dff_dir,
                     'twop_fps'      : sess.twop_fps,
                     'nanrois'       : sess.nanrois,
                     'nanrois_dff'   : sess.nanrois_dff,
                     'gab_k'         : gabk,
                     'gab_fr'        : [gabfr, gabfr_let], # e.g., [0, A]
                     'bri_size'      : bri_size
                    }

        for stim_type, dirs, diff, key, sep, omits in zip(stim_types, stim_dirs, 
                                                          stim_diffs, stim_keys, 
                                                          stim_seps, stim_omit):
            for direc, omit in zip(dirs, omits):
                stim, block_ran_seg = get_stim(sess, stim_type)
                if stim.sess.mouse_n in omit:
                    print(('    {}{}{} surprise indices and frames '
                           'omitted.').format(stim_type, sep, direc))
                    continue
                segs = stim.get_segs_by_criteria(gab_k=gabk, gaborframe=gabfr, 
                                                 bri_dir=direc, 
                                                 bri_size=bri_size, by='seg', 
                                                 block_ran_seg=block_ran_seg)
                frames = stim.get_2pframes_by_seg(segs, first=True)

                if (min(np.diff(segs)) != max(np.diff(segs)) or 
                    np.diff(segs)[0] != diff):
                    raise ValueError(('Retrieving surprise values not ' 
                                      'implemented for non consecutive '
                                      'segments.'))
                                
                surp_segs = stim.get_segs_by_criteria(gab_k=gabk, gaborframe=gabfr, 
                                                      bri_dir=direc, 
                                                      bri_size=bri_size, surp=1, 
                                                      by='seg', 
                                                      block_ran_seg=block_ran_seg)
                surp_idx = [int((seg - min(segs))/diff) for seg in surp_segs]

                frames_key   = '{}{}_frames'.format(key, direc)
                surp_idx_key = '{}{}_surp_idx'.format(key, direc)
                sess_dict[frames_key]   = frames.tolist()
                sess_dict[surp_idx_key] = surp_idx

        file_util.save_info(sess_dict, name, os.path.join(output, 'session_dicts', 
                                                          runtype), 'json')


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
def plot_autocorr_from_dicts(analys_par, autocorr_stats, basic_par, fig_par, 
                             sess_info, sess_par, save_dir=None):
    """
    From dictionaries, plots autocorrelation during stimulus blocks.

    Required arguments:
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                          (e.g., 128, 256 or [128, 256])
                ['gab_k'] (int or list) : gabor kappa values to include 
                                        (e.g., 4, 16 or [4, 16])
                ['lag_s'] (float)       : lag in seconds with which to calculate
                                          autocorrelation
                ['sess_ns'] (list)      : list of session IDs
                ['stim'] (str)          : stimulus to analyse (bricks or gabors)

        - autocorr_stats (dict): dictionary containing data to plot:
                ['data'] (list): list of 2 or 3D arrays (or nested lists) of
                                 autocorrelation statistics, structured as:
                                     sessions (x ROI) x stats (x_ran, mean/med, 
                                        std/qu/sem/mad) x frame

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'all', removes ROIs with NaN/Inf values 
                                    across entire session. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'

        - fig_par (dict)   : dictionary containing figure parameters:
                ['autocorr'] (str)       : specific subfolder in which to save 
                                           folder  
                ['bbox'] (str)           : bbox parameter for plt.savefig(), 
                                           e.g., 'tight'
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
                ['sharey'] (bool)        : if True, y axis lims are shared 
                                           across subplots
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['subplot_wid'] (float)  : width of each subplot (inches)

        - sess_info (dict): dictionary containing information from each
                            session 
                ['mouse_ns'] (list)       : mouse numbers
                ['overall_sess_ns'] (list): overall session numbers  
                ['lines'] (list)          : mouse lines
                ['layers'] (list)         : imaging layers
                ['n_rois'] (list)         : number of ROIs in session
                ['nanrois'] (list)        : list of ROIs with NaNs/Infs in raw
                                            traces
                ['nanrois_dff'] (list)    : list of ROIs with NaNs/Infs in dF/F
                                            traces, for sessions for which this 
                                            attribute exists    

        - sess_par (dict)  : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                            'L5_soma', 'L23_dend', 'L5_dend', 
                                            'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    
    Optional arguments:
            - save_dir (str)    : path of directory in which to save plots.
                              default: None  

    Returns:
        - full_dir (str):  final name of the directory in which the figure is 
                           saved
                           (may be different from input save_dir, as a datetime 
                           subfolder may have been added depending on the 
                           parameters in fig_par.)
        - save_name (str): name under which the figure is saved, excluding 
                           extension 
    """

    stimstr = str_util.stim_par_str(analys_par['gab_k'], analys_par['bri_dir'], 
                                    analys_par['bri_size'], analys_par['stim'],
                                    'print')
    sessstr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], analys_par['bri_size'], 
                                    analys_par['stim'], 'file')
    statstr = str_util.stat_par_str(basic_par['stats'], basic_par['error'])
    fluorstr_pr = str_util.fluor_par_str(type_str='print', dff=basic_par['dfoverf'])

    title_str = u'{} autocorr'.format(fluorstr_pr)

    if analys_par['stim'] == 'gabors':
        seg_bars = [-1.5, 1.5] # light lines
    else:
        seg_bars = [-1.0, 1.0] # light lines

    # extract some info from dictionaries
    mouse_ns         = sess_info['mouse_ns']
    overall_sess_ns  = sess_info['overall_sess_ns']
    lines            = sess_info['lines']
    layers           = sess_info['layers']
    n_rois           = sess_info['n_rois']
    n_sess = len(mouse_ns)

    if basic_par['remnans'] == 'all':
        if basic_par['dfoverf']:
            n_nans = [len(sess_nans) for sess_nans in sess_info['nanrois_dff']]
        else:
            n_nans = [len(sess_nans) for sess_nans in sess_info['nanrois']]
        n_rois = [n_roi - n_nan for n_roi, n_nan in zip(n_rois, n_nans)]

    stats = [np.asarray(stat) for stat in autocorr_stats['data']]

    lag_s = analys_par['lag_s']
    xticks = np.linspace(-lag_s, lag_s, lag_s*2+1)
    yticks = np.linspace(0, 1, 6)

    fig, ax = plot_util.init_fig(n_sess, fig_par)

    for i in range(n_sess):
        sub_ax = plot_util.get_subax(ax, i)
        # add each ROI
        for roi_stats in stats[i]:
            plot_util.plot_traces(sub_ax, roi_stats, basic_par['stats'], 
                        basic_par['error'], alpha=0.5/n_sess, 
                        xticks=xticks, yticks=yticks, dff=basic_par['dfoverf'])
        plot_util.add_bars(sub_ax, hbars=seg_bars)

        sub_ax.set_title((u'Mouse {} - {} {}{} {}\n(sess {}, {} {}, '
                          '(n={}))').format(mouse_ns[i], statstr, 
                                            analys_par['stim'][0:3], stimstr, 
                                            title_str, overall_sess_ns[i], 
                                            lines[i], layers[i], n_rois[i]))
        sub_ax.set_ylim([0, 1])

    save_dir = os.path.join(fig_par['figdir_roi'], fig_par['autocorr'])

    save_name = ('roi_autocorr_{}').format(sessstr)

    full_dir = plot_util.save_fig(fig, save_dir, save_name, fig_par)

    return full_dir, save_name


#############################################
def plot_autocorr(sessions, analys_par, basic_par, fig_par, sess_par):
    """
    plot_autocorr(sessions, analys_par, basic_par, fig_par)

    Plots autocorrelation during stimulus blocks.

    Required arguments:
        - sessions (list)  : list of Session objects
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                          (e.g., 128, 256 or [128, 256])
                ['gab_k'] (int or list) : gabor kappa values to include 
                                        (e.g., 4, 16 or [4, 16])
                ['lag_s'] (float)       : lag in seconds with which to calculate
                                          autocorrelation
                ['sess_ns'] (list)      : list of session IDs
                ['stim'] (str)          : stimulus to analyse (bricks or gabors)

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'all', removes ROIs with NaN/Inf values 
                                    across entire session. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'

        - fig_par (dict)   : dictionary containing figure parameters:
                ['autocorr'] (str)       : specific subfolder in which to save 
                                           folder  
                ['bbox'] (str)           : bbox parameter for plt.savefig(), 
                                           e.g., 'tight'
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
                ['sharey'] (bool)        : if True, y axis lims are shared 
                                           across subplots
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['subplot_wid'] (float)  : width of each subplot (inches)

        - sess_par (dict)  : dictionary containing session parameters:
                ['layer'] (str)         : layer ('soma', 'dend', 'L23_soma',  
                                            'L5_soma', 'L23_dend', 'L5_dend', 
                                            'L23_all', 'L5_all')
                ['overall_sess_n'] (int): overall session number aimed for
    """
    sessstr_pr = str_util.sess_par_str(sess_par, analys_par['gab_k'], 
                                    analys_par['bri_dir'], analys_par['bri_size'], 
                                    analys_par['stim'], 'print')

    if basic_par['remnans'] == 'per':
        print('NaNs remove across entire session, not per subsection.')
        basic_par['remnans'] == 'all'

    print(('\nAnalysing and plotting ROI autocorrelations ' 
          '({}).').format(sessstr_pr))
    
    stats = []
    for sess in sessions:
        stim, block_ran_seg = get_stim(sess, analys_par['stim'])
        all_segs = stim.get_segs_by_criteria(gab_k=analys_par['gab_k'], 
                                             bri_dir=analys_par['bri_dir'],
                                             bri_size=analys_par['bri_size'],
                                             by='block', 
                                             block_ran_seg=block_ran_seg)
        sess_traces = []
        for segs in all_segs:
            if len(segs) == 0:
                continue
            # check that segs are contiguous
            if max(np.diff(segs)) > 1:
                raise ValueError('Segments used for autocorrelation are not '
                                 'contiguous.')
            frame_edges = stim.get_2pframes_by_seg([min(segs), max(segs)])
            frames = range(min(frame_edges[0]), max(frame_edges[1])+1)
            traces = sess.get_roi_traces(frames, dfoverf=basic_par['dfoverf'])
            if basic_par['remnans'] == 'all':
                if basic_par['dfoverf']:
                    nan_rois = sess.nanrois_dff
                else:
                    nan_rois = sess.nanrois
                traces, _ = remove_nan_rois(traces, roi_dim=0, 
                                            rem_rois=nan_rois)
            sess_traces.append(traces)
        autocorr = autocorr_rois(sess_traces, analys_par['lag_s'], 
                                 sess.twop_fps, basic_par['stats'], 
                                 basic_par['error'])
        stats.append(autocorr)

    autocorr_stats = {'data': [stat.tolist() for stat in stats]
                      }

    info = {'analys_par': analys_par,
            'autocorr_stats': autocorr_stats,
            'basic_par': basic_par,
            'sess_info': get_sess_info(sessions, dfoverf=basic_par['dfoverf']),
            'sess_par': sess_par
            }

    full_dir, save_name = plot_autocorr_from_dicts(fig_par=fig_par, **info)

    file_util.save_info(info, save_name, full_dir, 'json')


#############################################
def create_arg_dicts(args, fig_only=False):
    """
    create_arg_dicts(args)

    Split args into different dictionaries

    Required arguments:
        - args (Argument parser): parser with arguments as attributes listed
                                  in the _keys lists, as well as 
                no_add_nosurp (bool): if True, group of ROIs showing no 
                                      significance in either is not added to   
                                      the groups returned
                no_datetime (bool)  : if True, figures are not saved in a 
                                      subfolder named based on the date and 
                                      time.
                no_sharey (bool)    : if True, y axis lims are not shared 
                                      across subplots
                output (str)        : general path to save output
                raw (bool)          : if True, raw ROI traces is used.

    Optional arguments:
        - fig_only (bool): if True, only the figure arguments are compiled
                           into a dictionary and returned

    Returns:
        - analys_par (dict): dictionary containing specific analysis parameters:
                ['bri_dir'] (str or list) : brick direction values to include
                                            (e.g., 'right', 'left')
                ['bri_size'] (int or list): brick size values to include
                                            (e.g., 128, 256 or [128, 256])
                ['gab_fr'] (int)          : gabor frame at which segments start 
                                            (e.g., 0, 1, 2, 3)
                ['gab_k'] (int or list)   : gabor kappa values to include 
                                            (e.g., 4, 16 or [4, 16])
                ['n_quints'] (int)        : number of quintiles
                ['pre'] (float)           : range of frames to include before 
                                            each frame reference (in s)
                ['post'] (float)          : range of frames to include after  
                                            each frame reference (in s)
                ['stim'] (str)            : stimulus to analyse (bricks or 
                                            gabors)

        - basic_par (dict) : dictionary containing basic analysis parameters:
                ['dfoverf'] (bool): if True, dF/F is used instead of raw ROI 
                                    traces.
                ['error'] (str)   : error statistic parameter, i.e. 'std' or 
                                    'sem'
                ['rand'] (bool)   : if True, also includes statistics for a 
                                    random permutation of the traces (not 
                                    implemented).
                ['remnans'] (str) : if 'per', removes ROIs with NaN/Inf values, 
                                    for each subdivision (quintile/surprise). If 
                                    'all', removes ROIs with NaN/Inf values 
                                    across entire session. If 'no', ROIs with 
                                    NaN/Inf values are not removed.
                ['stats'] (str)   : statistic parameter, i.e. 'mean' or 'median'

        - fig_par (dict)   : dictionary containing figure parameters:
                ['autocorr'] (str)       : specific subfolder in which to save 
                                           autocorrelation results  
                ['bbox'] (str)           : bbox parameter for plt.savefig(), 
                                           e.g., 'tight'
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
                ['preset_ylims']         : if True, preset y lims are used
                ['prev_dt'] (str)        : datetime folder to use 
                ['sharey'] (bool)        : if True, y axis lims are shared 
                                           across subplots
                ['subplot_hei'] (float)  : height of each subplot (inches)
                ['subplot_wid'] (float)  : width of each subplot (inches)
                ['surp_quint'] (str)     : specific folder in which to save 
                                           surprise quintile results

        - perm_par (dict)    : dictionary containing permutation analysis 
                               parameters:
                ['n_perms'] (int)     : nbr of permutations to run
                ['p_val'] (float)     : p-value to use for significance  
                                        thresholding (0 to 1)
                ['tails'] (str or int): which tail(s) to test: 'up', 'lo', '2'

        - roi_grp_par (dict) : dictionary containing ROI grouping parameters:
                ['add_nosurp'] (bool) : if True, group of ROIs showing no 
                                        significance in either is included in  
                                        the groups returned
                ['grps'] (str or list): set or sets of groups to return, 
                                        e.g., 'all', 'change', 'no_change', 
                                        'reduc', 'incr'.
                                        If several sets are passed, each set 
                                        will be collapsed as one group and
                                        'add_nosurp' will be set to False.
                ['op'] (str)          : operation to use to compare groups, 
                                        i.e. 'diff': grp1-grp2, or 'ratio': 
                                        grp1/grp2

        - sess_par (dict)  : dictionary containing session parameters:
                ['closest'] (bool)         : if False, only exact session
                                             number is retained, otherwise the 
                                             closest.
                ['layer'] (str)            : layer ('soma', 'dend', 'L23_soma',  
                                             'L5_soma', 'L23_dend', 'L5_dend', 
                                             'L23_all', 'L5_all')
                ['min_rois'] (int)         : min number of ROIs
                ['omit_mice'] (list)       : mice to omit
                ['omit_sess'] (list)       : sessions to omit
                ['overall_sess_n'] (int)   : overall session number aimed for
                ['pass_fail'] (str or list): pass/fail values of interest 
                                             ('P', 'F')
                ['runtype'] (str or list)  : runtype value(s) of interest
                                             ('pilot', 'prod')
    """

    fig_keys     = ['bbox', 'fig_ext', 'ncols', 'overwrite', 'preset_ylims', 
                    'subplot_hei', 'subplot_wid']
    fig_par     = {key: args_dict[key] for key in fig_keys if key in args_dict.keys()}
    
    # set keys that are the inverse of args
    fig_par['datetime']  = not(args.no_datetime)
    fig_par['sharey']    = not(args.no_sharey)

    # subfolders
    fig_par['figdir_roi'] = os.path.join(args.output, 'results', 'figures', 
                                         '{}_roi'.format(args.runtype))
    fig_par['surp_quint'] = 'surp_nosurp_quint' 
    fig_par['autocorr']   = 'autocorr'
    
    # allow reusing datetime folder (if mult figs created by one function)
    fig_par['prev_dt'] = None
    fig_par['mult']    = False

    if fig_only:
        return fig_par

    analys_keys  = ['bri_dir', 'bri_size', 'gab_fr', 'gab_k', 'lag_s',  
                    'n_quints', 'post', 'pre', 'stim']
    basic_keys   = ['error', 'rand', 'remnans', 'stats']
    perm_keys    = ['n_perms', 'p_val', 'tails']
    roi_grp_keys = ['grps', 'op', 'plot_vals']
    sess_keys    = ['closest', 'layer', 'min_rois', 'omit_mice', 'omit_sess', 
                    'overall_sess_n', 'pass_fail', 'runtype']

    analys_par  = {key: args_dict[key] for key in analys_keys if key in args_dict.keys()}
    basic_par   = {key: args_dict[key] for key in basic_keys if key in args_dict.keys()}
    perm_par    = {key: args_dict[key] for key in perm_keys if key in args_dict.keys()}
    roi_grp_par = {key: args_dict[key] for key in roi_grp_keys if key in args_dict.keys()}
    sess_par    = {key: args_dict[key] for key in sess_keys if key in args_dict.keys()}
    
    # set keys that are the inverse of args
    basic_par['dfoverf']      = not(args.raw)
    roi_grp_par['add_nosurp'] = not(args.no_add_nosurp)
    
    return analys_par, basic_par, fig_par, perm_par, roi_grp_par, sess_par


#############################################
def plot_from_dict(dict_path, args):
    """
    plot_from_dict(info_path, args)

    Plots data from dictionaries containing analysis parameters and results.

    Required arguments:
        - dict_path (str)       : path to dictionary to plot data from
        - args (Argument parser): parser containing all parameters
        - mouse_df (pandas df)  : dataframe containing parameters for each 
                                  session.
    """

    plot_util.linclab_plt_defaults(font=['Arial', 'Liberation Sans'], 
                                   font_dir='../tools/fonts', example=True)

    if args.parallel and args.plt_bkend is not None:
        plt.switch_backend(args.plt_bkend) # needs to be repeated within joblib

    fig_par = create_arg_dicts(args, fig_only=True)

    info_dict = file_util.load_file(dict_path)
    save_dir  = os.path.dirname(dict_path)

    analysis = info_dict['analys_par']['analysis']

    # 1. Plot average traces by quintile x surprise for each session 
    if 't' in args.analyses: # traces
        plot_traces_by_qu_surp_sess_from_dicts(fig_par=fig_par, 
                                               save_dir=save_dir, **info_dict)

    # 2. Plot average dF/F area for each ROI group across quintiles for each 
    # session 
    if analysis == 'q': # roi_grps_qu
        plot_rois_by_grp_qu_sess_from_dicts(fig_par=fig_par, save_dir=save_dir,
                                            **info_dict)

    # # 3. Plot average traces and trace areas by suprise for first vs last 
    # # quintile, for each ROI group, for each session
    if analysis == 'c': # roi_grps_ch
        plot_rois_by_grp_from_dicts(fig_par=fig_par, save_dir=save_dir,
                                    **info_dict)

    # # 4. Plot magnitude of change in dF/F area from first to last quintile of 
    # # surprise vs no surprise segments, for each session
    if analysis == 'm': # mag
        plot_mag_change_from_dicts(fig_par=fig_par, save_dir=save_dir, **info_dict)

    # # 5. Run autocorrelation analysis
    if analysis == 'a': # autocorr
        plot_autocorr_from_dicts(fig_par=fig_par, save_dir=save_dir, 
                                 **info_dict)


#############################################
def run_analyses(sess, args, mouse_df):
    """
    run_analyses(sess, args, mouse_df)

    Runs analyses on the sessions corresponding to the overall session numbers
    passed.

    Required arguments:
        - sess (int)            : overall session number to run analyses on
        - args (Argument parser): parser containing all parameters
        - mouse_df (pandas df)  : dataframe containing parameters for each 
                                  session.
    """

    plot_util.linclab_plt_defaults(font=['Arial', 'Liberation Sans'], 
                                   font_dir='../tools/fonts', example=True)

    if args.parallel and args.plt_bkend is not None:
        plt.switch_backend(args.plt_bkend) # needs to be repeated within joblib

    [analys_par, basic_par, fig_par, perm_par, 
                        roi_grp_par, sess_par] = create_arg_dicts(args)

    print(('\nAnalysis of {} responses to {} stimuli '
           '({} data)').format(sess_par['layer'], analys_par['stim'], 
                               sess_par['runtype']))

    print('\nOverall_sess_n: {}'.format(sess))
    sess_par['overall_sess_n'] = int(sess)

    analys_par['sess_ns'] = sess_per_mouse(mouse_df, **sess_par)

    # create a dictionary with Session objects prepared for analysis
    sessions = init_sessions(analys_par['sess_ns'], args.datadir, mouse_df,
                             args.runtype, full_dict=False, load_run=False)

    if args.analyses == 'all':
        args.analyses = 'ltqcma'

    # 0. Create dictionary including frame numbers for LFADS analysis
    if 'l' in args.analyses: # lfads
        analys_par['analysis'] = 'l'
        lfads_dict(sessions, mouse_df, args.runtype, args.output)

    # 1. Plot average traces by quintile x surprise for each session 
    if 't' in args.analyses: # traces
        analys_par['analysis'] = 't'
        plot_traces_by_qu_surp_sess(sessions, analys_par, basic_par, fig_par, 
                                    sess_par)

    # 2. Plot average dF/F area for each ROI group across quintiles for each 
    # session 
    if 'q' in args.analyses: # roi_grps_qu
        analys_par['analysis'] = 'q'
        plot_rois_by_grp_qu_sess(sessions, analys_par, basic_par, fig_par, 
                                perm_par, roi_grp_par, sess_par)

    # 3. Plot average traces and trace areas by suprise for first vs last 
    # quintile, for each ROI group, for each session
    if 'c' in args.analyses: # roi_grps_ch
        analys_par['analysis'] = 'c'
        plot_rois_by_grp(sessions, analys_par, basic_par, fig_par, perm_par, 
                        roi_grp_par, sess_par)

    # 4. Plot magnitude of change in dF/F area from first to last quintile of 
    # surprise vs no surprise segments, for each session
    if 'm' in args.analyses: # mag
        analys_par['analysis'] = 'm'
        plot_mag_change(sessions, analys_par, basic_par, fig_par, sess_par)

    # 5. Run autocorrelation analysis
    if 'a' in args.analyses: # autocorr
        analys_par['analysis'] = 'a'
        plot_autocorr(sessions, analys_par, basic_par, fig_par, sess_par)


if __name__ == "__main__":

    # typically change runtype, analyses, layer, overall_sess_n, plot_vals

    parser = argparse.ArgumentParser()

        # general parameters
    parser.add_argument('--datadir', default=None, 
                        help=('data directory (if None, uses a directory '
                              'defined below'))
    parser.add_argument('--output', default='', help='where to store output')
    parser.add_argument('--plt_bkend', default=None, 
                        help='switch matplotlib backend when running on server')
    parser.add_argument('--analyses', default='all', 
                        help=('analyses to run: lfads (l), traces (t), '
                              'roi_grps_qu (q), roi_grps_ch (c), mag (m), '
                              'autocorr (a) or \'all\''))
    parser.add_argument('--parallel', action='store_true', 
                        help='do overall_sess_n\'s in parallel.')
    parser.add_argument('--dict_path', default='', 
                        help='path to info dictionary to plot data from.')

        # session parameters
    parser.add_argument('--runtype', default='prod', help='prod or pilot')
    parser.add_argument('--layer', default='soma',
                        help=('soma, dend, L23_soma, L5_soma, L23_dend, '
                              'L5_dend, L23_all, L5_all'))
    parser.add_argument('--overall_sess_n', default='all',
                        help='session to aim for, e.g. 1, 2, last, all')
    parser.add_argument('--closest', action='store_true', 
                        help=('if True, the closest session number is used.'
                              ' Otherwise, only exact.'))
    parser.add_argument('--min_rois', default=5, type=int, 
                        help='min rois criterion')
        # analysis parameters
    parser.add_argument('--bri_dir', default='right', help='brick dir (right, left, or both)')   
    parser.add_argument('--post', default=1.5, type=float, help='sec after frame')
    parser.add_argument('--stim', default='gabors', help='stimulus to analyse')   
        # roi group parameters
    parser.add_argument('--plot_vals', default='surp', 
                        help='plot diff (surp-nosurp), surp or nosurp')
    
    # generally fixed 
        # session parameters
    parser.add_argument('--pass_fail', default='P', 
                        help='P to take only passed sessions')
    parser.add_argument('--omit_sess', default='', help='sessions to omit, separated by spaces')  
    parser.add_argument('--omit_mice', default='', help='mice to omit, separated by spaces') 

        # analysis parameters
    parser.add_argument('--bri_size', default=128, help='brick size (128, 256, or both)')
    parser.add_argument('--n_quints', default=4, type=int, help='nbr of quintiles')
    parser.add_argument('--lag_s', default=4, type=float,
                        help='lag for autocorrelation (in sec)')
    parser.add_argument('--gab_k', default=16,
                        help='kappa value (4, 16, or both)')    
    parser.add_argument('--gab_fr', default=3, type=int, 
                        help='gabor frame to start segments at')
    parser.add_argument('--pre', default=0, type=float, help='sec before frame')
    
        # roi group parameters
    parser.add_argument('--op', default='diff', 
                        help='calculate diff or ratio of surp to nonsurp')
    parser.add_argument('--grps', default='reduc incr no_change', 
                        help=('plot all ROI grps or grps with change or '
                              'no_change'))
    parser.add_argument('--no_add_nosurp', action='store_true',
                        help='do not add nosurp_nosurp to ROI grp plots')
    
        # permutation analysis parameters
    parser.add_argument('--n_perms', default=5000, type=int, 
                        help='nbr of permutations')
    parser.add_argument('--p_val', default=0.05, type=float,
                        help='p-val for perm analysis')
    parser.add_argument('--tails', default='2', 
                        help='nbr tails for perm analysis (2, lo, up)')
    
        # figure parameters
    parser.add_argument('--fig_ext', default='svg', help='svg or png')
    parser.add_argument('--no_datetime', action='store_true',
                        help='create a datetime folder')
    parser.add_argument('--overwrite', action='store_true', 
                        help='allow overwriting')
    parser.add_argument('--ncols', default=3, type=int, 
                        help='nbr of cols per fig')
    parser.add_argument('--subplot_hei', default=7.5, type=float, 
                        help='subplot height')
    parser.add_argument('--subplot_wid', default=7.5, type=float, 
                        help='subplot width')
    parser.add_argument('--bbox', default='tight', help='wrapping around figs')
    parser.add_argument('--preset_ylims', action='store_true',
                        help='use preset y lims')
    parser.add_argument('--no_sharey', action='store_true',
                        help='do not share y axis lims within figs')

        # basic parameters
    parser.add_argument('--rand', action='store_true',
                        help=('produce plots from randomized data (in many '
                              'cases, not implemented yet'))
    parser.add_argument('--remnans', default='all', 
                        help=('remove ROIs containing NaNs or Infs across '
                              'entire session (all), per subsection (per) or not '
                              '(no)'))
    parser.add_argument('--raw', action='store_true',
                        help='use raw instead of dfoverf ROI traces')
    parser.add_argument('--stats', default='mean', help='plot mean or median')
    parser.add_argument('--error', default='sem', 
                        help='sem for SEM/MAD, std for std/qu')


    args = parser.parse_args()
    args_dict = args.__dict__

    if args.plt_bkend is not None: # necessary for running on server
        plt.switch_backend(args.plt_bkend)

    ##### DICTIONARY TO PLOT ####
    args.dict_path = 'results/figures/prod_roi/surp_nosurp_quint/mags/20190321_073821/roi_mag_diff_sess1_bri128_right_soma.json'
    args.analyses = 'm'

    if args.dict_path is not '':
        plot_from_dict(args.dict_path, args)
    else:
        if args.datadir is None:
            args.datadir = '../data/AIBS/{}'.format(args.runtype)
        
        mouse_df_dir = 'mouse_df.csv'
        mouse_df = file_util.load_file(mouse_df_dir)

        format_args(args) # reformats some of the args
        update_args(args)  # updates args based on dataset properties

        # get numbers of sessions to analyse
        if args.overall_sess_n == 'all':
            args.closest = False
            all_sesses = all_sess_ns(mouse_df, args.runtype, args.layer, 
                                    args.pass_fail, omit_sess=args.omit_sess, 
                                    omit_mice=args.omit_mice, 
                                    min_rois=args.min_rois)
        else:
            all_sesses = gen_util.list_if_not(args.overall_sess_n)

        # run through all sessions
        if args.parallel:
            num_cores = multiprocessing.cpu_count()
            Parallel(n_jobs=num_cores)(delayed(run_analyses)
                    (sess, args, mouse_df) for sess in all_sesses)
        else:
            for sess in all_sesses:
                run_analyses(sess, args, mouse_df)

