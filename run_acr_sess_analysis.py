"""
run_acr_sess_analysis.py

This script runs across session analyses using a Session object with data 
generated by the AIBS experiments for the Credit Assignment Project.

Authors: Colleen Gillon

Date: October, 2019

Note: this code uses python 3.7.

"""

import argparse
import copy
import glob
import os
import re

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import pdb

from util import file_util, gen_util, math_util, plot_util
from sess_util import sess_gen_util, sess_ntuple_util, sess_plot_util, \
                      sess_str_util
from analysis import session, acr_sess_analys
from plot_fcts import plot_from_dicts_tool as plot_dicts



#############################################
def reformat_args(args):
    """
    reformat_args(args)

    Returns reformatted args for analyses, specifically 
        - Sets stimulus parameters to 'none' if they are irrelevant to the 
          stimtype
        - Changes stimulus parameters from 'both' to actual values
        - Modifies the session number parameter

    Adds the following args:
        - dend (str)     : type of dendrites to use ('aibs' or 'extr')
        - omit_sess (str): sess to omit
        - omit_mice (str): mice to omit

    Required args:
        - args (Argument parser): parser with the following attributes: 
            bri_dir (str)        : brick direction values to include
                                   (e.g., 'right', 'left' or 'both')
            bri_size (int or str): brick size values to include
                                   (e.g., 128, 256, 'both')
            gabfr (int)          : gabor frame value to start sequences at
                                   (e.g., 0, 1, 2, 3)
            gabk (int or str)    : gabor kappa values to include 
                                   (e.g., 4, 16 or 'both')
            gab_ori (int or str) : gabor orientation values to include
                                   (e.g., 0, 45, 90, 135 or 'all')
            runtype (str)        : runtype ('pilot' or 'prod')
            sess_n (str)         : session number range
            stimtype (str)       : stimulus to analyse (bricks or gabors)
    
    Returns:
        - args (Argument parser): input parser, with the following attributes
                                  modified: 
                                      bri_dir, bri_size, gabfr, gabk, gab_ori, 
                                      sess_n
                                  and the following attributes added:
                                      omit_sess, omit_mice, dend
    """
    args = copy.deepcopy(args)

    if args.plane == 'soma':
        args.dend = 'aibs'

    [args.bri_dir, args.bri_size, args.gabfr, 
     args.gabk, args.gab_ori] = sess_gen_util.get_params(args.stimtype, 
                                              args.bri_dir, args.bri_size, 
                                              args.gabfr, args.gabk, 
                                              args.gab_ori)

    args.omit_sess, args.omit_mice = sess_gen_util.all_omit(args.stimtype, 
                                                    args.runtype, args.bri_dir, 
                                                    args.bri_size, args.gabk)
    
    if '-' in str(args.sess_n):
        split = str(args.sess_n).find('-')
        st = int(str(args.sess_n)[ : split])
        end = int(str(args.sess_n)[split + 1:]) + 1
        args.sess_n = list(range(st, end))

    if args.method == 'ratio':
        args.p_val_thr = None
    elif args.method == 'ttest':
        args.rel_std = None

    return args


#############################################
def init_param_cont(args):
    """
    init_param_cont(args)

    Returns args:
        - in the following nametuples: analyspar, sesspar, stimpar, autocorr, 
                                       permpar, quintpar, roigrppar, tcurvpar
        - in the following dictionary: figpar 

    Required args:
        - args (Argument parser): parser with the following attributes:

            base (float)           : baseline value to use
            bri_dir (str or list)  : brick direction values to include
                                     ('right', 'left', ['right', 'left'])
            bri_size (int or list) : brick size values to include
                                     (128, 256 or [128, 256])
            dend (str)             : type of dendrites to use ('aibs' or 'dend')
            error (str)            : error statistic parameter ('std' or 'sem')
            fluor (str)            : if 'raw', raw ROI traces are used. If 
                                     'dff', dF/F ROI traces are used.
            fontdir (str)          : path to directory containing additional 
                                     fonts
            gabfr (int)            : gabor frame at which sequences start 
                                     (0, 1, 2, 3)
            gabk (int or list)     : gabor kappa values to include 
                                     (4, 16 or [4, 16])
            gab_ori (int or list)  : gabor orientation values to include
                                     ([0, 45, 90, 135])
            incl (str)             : sessions to include ('yes', 'no', 'all') 
            keepnans (str)         : if True, ROIs with NaN/Inf values are 
                                     kept in the analyses.
            lag_s (num)            : lag for autocorrelation (in sec)
            line (str)             : 'L23', 'L5', 'any'
            method (str)           : latency calculation method (ratio or ttest)
            min_rois (int)         : min number of ROIs
            n_perms (int)          : nbr of permutations to run
            n_quints (int)         : number of quintiles
            ncols (int)            : number of columns
            no_datetime (bool)     : if True, figures are not saved in a 
                                     subfolder named based on the date and time.
            no_scale (bool)        : if True, data is not scaled
            not_surp_rois (bool)   : if False, only surprise responsive ROIs 
                                      are used for latency analysis
            output (str)           : general directory in which to save output
            overwrite (bool)       : if False, overwriting existing figures 
                                     is prevented by adding suffix numbers.
            pass_fail (str or list): pass/fail values of interest ('P', 'F')
            p_val_thr (float)      : p-value threshold for ttest latency method
            plane (str)            : plane ('soma', 'dend', 'any')
            plt_bkend (str)        : mpl backend to use
            post (num)             : range of frames to include after each 
                                     reference frame (in s)
            pre (num)              : range of frames to include before each 
                                     reference frame (in s)
            rel_std (float)        : relative st. dev. threshold for ratio 
                                     latency method
            runtype (str or list)  : runtype ('pilot' or 'prod')
            sess_n (int)           : session number
            stats (str)            : statistic parameter ('mean' or 'median')
            stimtype (str)         : stimulus to analyse ('bricks' or 'gabors')
            tails (str or int)     : which tail(s) to test ('up', 'lo', 2)

    Returns:
        - analyspar (AnalysPar): named tuple of analysis parameters
        - sesspar (SessPar)    : named tuple of session parameters
        - stimpar (StimPar)    : named tuple of stimulus parameters
        - permpar (PermPar)    : named tuple of permutation parameters
        - latpar (LatPar)      : named tuple of latency parameters
        - figpar (dict)        : dictionary containing following 
                                 subdictionaries:
            ['init']: dict with following inputs as attributes:
                                'ncols', 'sharey', as well as
                ['ncols'] (int)      : number of columns in the figures
                ['sharex'] (bool)    : if True, x axis lims are shared across
                                       subplots
                ['sharey'] (bool)    : if True, y axis lims are shared across
                                       subplots
                ['subplot_hei'] (num): height of each subplot (inches)
                ['subplot_wid'] (num): width of each subplot (inches)

            ['save']: dict with the following inputs as attributes:
                                'overwrite', as well as
                ['datetime'] (bool): if True, figures are saved in a subfolder 
                                     named based on the date and time.
                ['use_dt'] (str)   : datetime folder to use
                ['fig_ext'] (str)  : figure extension

            ['dirs']: dict with the following attributes:
                ['figdir'] (str)   : main folder in which to save figures
                ['roi'] (str)      : subdirectory name for ROI analyses
                ['run'] (str)      : subdirectory name for running analyses
                ['autocorr'] (str) : subdirectory name for autocorrelation 
                                     analyses
                ['locori'] (str)   : subdirectory name for location and 
                                     orientation responses
                ['oridir'] (str)   : subdirectory name for 
                                     orientation/direction analyses
                ['surp_qu'] (str)  : subdirectory name for surprise, quintile 
                                     analyses
                ['tune_curv'] (str): subdirectory name for tuning curves
                ['grped'] (str)    : subdirectory name for ROI grps data
                ['mags'] (str)     : subdirectory name for magnitude analyses
            
            ['mng']: dict with the following attributes:
                ['plt_bkend'] (str): mpl backend to use
                ['linclab'] (bool) : if True, Linclab mpl defaults are used
                ['fontdir'] (str)  : path to directory containing additional 
                                     fonts
    """

    args = copy.deepcopy(args)

    # analysis parameters
    analyspar = sess_ntuple_util.init_analyspar(args.fluor, not(args.keepnans), 
                                 args.stats, args.error, 
                                 scale=not(args.no_scale), dend=args.dend)

    # session parameters
    sesspar = sess_ntuple_util.init_sesspar(args.sess_n, False, 
                               args.plane, args.line, args.min_rois, 
                               args.pass_fail, args.incl, args.runtype)
    
    # stimulus parameters
    stimpar = sess_ntuple_util.init_stimpar(args.stimtype, args.bri_dir, 
                                            args.bri_size, args.gabfr, 
                                            args.gabk, args.gab_ori, 
                                            args.pre, args.post)

    # SPECIFIC ANALYSES
    # permutation parameters
    permpar = sess_ntuple_util.init_permpar(args.n_perms, 0.05, args.tails, 
                                            False)

    basepar = sess_ntuple_util.init_basepar(args.base)

    latpar = sess_ntuple_util.init_latpar(args.method, args.p_val_thr, 
                                          args.rel_std, not(args.not_surp_resp))


    # figure parameters
    figpar = sess_plot_util.init_figpar(ncols=int(args.ncols), 
                            datetime=not(args.no_datetime), 
                            overwrite=args.overwrite, runtype=args.runtype, 
                            output=args.output, plt_bkend=args.plt_bkend, 
                            fontdir=args.fontdir)

    return [analyspar, sesspar, stimpar, permpar, basepar, latpar, figpar]


#############################################
def init_mouse_sess(mouse_n, all_sess_ns, sesspar, mouse_df, datadir, 
                    omit_sess=[], dend='extr'):

    """
    init_mouse_sess(mouse_n, all_sess_ns, sesspar, mouse_df, datadir)

    Required args:
        - mouse_n (int)       : mouse number
        - all_sess_ns (list)  : list of all sessions to include
        - sesspar (SessPar)   : named tuple containing session parameters
        - mouse_df (pandas df): path name of dataframe containing information 
                                  on each session
        - datadir (str)       : path to data directory
    
    Optional args:
        - omit_sess (list): list of sessions to omit
        - dend (str)      : type of dendrites to use ('aibs' or 'dend')

    Returns:
        - mouse_sesses (list): list of Session objects for the specified mouse, 
                               with None in the position of missing sessions 
    """

    sesspar_dict = sesspar._asdict()
    sesspar_dict.pop('closest')

    mouse_sesses = []
    for sess_n in all_sess_ns:
        sesspar_dict['sess_n'] = sess_n
        sesspar_dict['mouse_n'] = mouse_n
        sessid = sess_gen_util.get_sess_vals(mouse_df, 'sessid', 
                               omit_sess=omit_sess, **sesspar_dict)
        if len(sessid) == 0:
            sess = [None]
        elif len(sessid) > 1:
            raise ValueError('Unexpected error. Should not give more '
                             'than 1 session.')
        else:
            sess = sess_gen_util.init_sessions(sessid[0], datadir, mouse_df, 
                                 sesspar.runtype, fulldict=False,
                                 dend=dend, omit=True)
            if len(sess) == 0:
                sess = [None]
        mouse_sesses.append(sess[0])

    return mouse_sesses


#############################################
def prep_analyses(sess_n, args, mouse_df, parallel=False):
    """
    prep_analyses(sess_n, args, mouse_df)

    Prepares named tuples and sessions for which to run analyses, based on the 
    arguments passed.

    Required args:
        - sess_n (int)          : session number to run analyses on, or 
                                  combination of session numbers to compare, 
                                  e.g. '1v2'
        - args (Argument parser): parser containing all parameters
        - mouse_df (pandas df)  : path name of dataframe containing information 
                                  on each session
    Optional args:
        - parallel (bool): if True, sessions are initialized in parallel 
                           across CPU cores 
                           default: False

    Returns:
        - sessions (list)      : list of sessions, or nested list per mouse 
                                 if sess_n is a combination
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - permpar (PermPar)    : named tuple of permutation parameters
        - figpar (dict)        : dictionary containing following 
                                 subdictionaries:
            ['init']: dict with following inputs as attributes:
                                'ncols', 'sharey', as well as
                ['ncols'] (int)      : number of columns in the figures
                ['sharex'] (bool)    : if True, x axis lims are shared across
                                       subplots
                ['sharey'] (bool)    : if True, y axis lims are shared across
                                       subplots
                ['subplot_hei'] (num): height of each subplot (inches)
                ['subplot_wid'] (num): width of each subplot (inches)

            ['save']: dict with the following inputs as attributes:
                                'overwrite', as well as
                ['datetime'] (bool): if True, figures are saved in a subfolder 
                                     named based on the date and time.
                ['use_dt'] (str)   : datetime folder to use
                ['fig_ext'] (str)  : figure extension

            ['dirs']: dict with the following attributes:
                ['figdir']   (str) : main folder in which to save figures
                ['roi']      (str) : subdirectory name for ROI analyses
                ['run']      (str) : subdirectory name for running analyses
                ['grp']      (str) : main folder in which to save ROI grps data
                ['autocorr'] (str) : subdirectory name for autocorrelation 
                                     analyses
                ['mags']     (str) : subdirectory name for magnitude analyses
                ['oridir']  (str)  : subdirectory name for  
                                     orientation/direction analyses
                ['surp_qu']  (str) : subdirectory name for surprise, quintile 
                                     analyses
                ['tune_curv'] (str): subdirectory name for tuning curves
            
            ['mng']: dict with the following attributes:
                ['plt_bkend'] (str): mpl backend to use
                ['linclab'] (bool) : if True, Linclab mpl defaults are used
                ['fontdir'] (str)  : path to directory containing additional 
                                     fonts
        - seed (int)               : seed to use
    """

    args = copy.deepcopy(args)

    # chose a seed if none is provided (i.e., args.seed=-1), but seed later
    seed = gen_util.seed_all(args.seed, 'cpu', print_seed=False, 
                             seed_now=False)

    args.sess_n = sess_n

    [analyspar, sesspar, stimpar, permpar, 
                  basepar, latpar, figpar] = init_param_cont(args)
    
    sesspar_dict = sesspar._asdict()
    _ = sesspar_dict.pop('closest')

    [all_mouse_ns, all_sess_ns] = sess_gen_util.get_sess_vals(mouse_df, 
                            ['mouse_n', 'sess_n'], omit_sess=args.omit_sess, 
                            omit_mice=args.omit_mice, **sesspar_dict)

    if args.sess_n in ['any', 'all']:
        all_sess_ns = [n + 1 for n in range(max(all_sess_ns))]
    else:
        all_sess_ns = args.sess_n

    # get session IDs and create Sessions
    all_mouse_ns = sorted(set(all_mouse_ns))
    n_jobs = gen_util.get_n_jobs(len(all_mouse_ns), parallel)
    if parallel:
        sessions = Parallel(n_jobs=n_jobs)(delayed(init_mouse_sess)
                   (mouse_n, all_sess_ns, sesspar, mouse_df, args.datadir, 
                    args.omit_sess, analyspar.dend) for mouse_n in all_mouse_ns)
    else:
        sessions = []
        for mouse_n in all_mouse_ns:
            sessions.append(init_mouse_sess(mouse_n, all_sess_ns, sesspar, 
                            mouse_df, args.datadir, args.omit_sess, 
                            analyspar.dend))

    check_all = set([sess for m_sess in sessions for sess in m_sess])
    if len(sessions) == 0 or check_all == {None}:
        raise ValueError('No sessions meet the criteria.')

    print(f'\nAnalysis of {sesspar.plane} responses to {stimpar.stimtype[:-1]} '
          f'stimuli ({sesspar.runtype} data)\nSessions: {args.sess_n}')

    return [sessions, analyspar, sesspar, stimpar, permpar, 
            basepar, latpar, figpar, seed]

    
#############################################
def run_analyses(sessions, analyspar, sesspar, stimpar, permpar, basepar, 
                 latpar, figpar, seed=None, analyses='all', parallel=False):
    """
    run_analyses(sessions, analyspar, sesspar, stimpar, autocorrpar, 
                 permpar, quintpar, roigrppar, tcurvpar, figpar)

    Run requested analyses on sessions using the named tuples passed.
    Some analyses can be skipped (e.g., to be launched in a non parallel
    process instead.)

    Required args:
        - sessions (list)      : nested list of sessions
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - permpar (PermPar)    : named tuple of permutation parameters
        - basepar (BasePar)    : named tuple of baseline parameters
        - latpar (LatPar)      : named tuple of latency parameters
        - figpar (dict)        : dictionary containing figure parameters
    
    Optional args:
        - seed (int)     : seed to use
                           default: None
        - analyses (str) : analyses to run
                           default: 'all'
        - parallel (bool): if True, some analyses are parallelized 
                           across CPU cores 
                           default: False
    """

    if len(sessions) == 0:
        print('No sessions fit these criteria.')
        return

    all_analyses = 'sltrup'
    all_check = ''

    if 'all' in analyses:
        if '_' in analyses:
            excl = analyses.split('_')[1]
            analyses, _ = gen_util.remove_lett(all_analyses, excl)
        else:
            analyses = all_analyses

    # changes backend and defaults
    plot_util.manage_mpl(cmap=False, **figpar['mng'])

    # 0. Plots the difference between surprise and regular across sessions
    if 's' in analyses:
        acr_sess_analys.run_surp_area_diff(sessions, 's', analyspar, sesspar, 
                        stimpar, basepar, permpar, figpar, parallel=parallel)
    all_check += 's'

    # 1. Plots the difference between surprise and regular locked to surprise
    # across sessions
    if 'l' in analyses:
        acr_sess_analys.run_lock_area_diff(sessions, 'l', analyspar, sesspar, 
                        stimpar, basepar, permpar, figpar, parallel=parallel)
    all_check += 'l'

    # 2. Plots the surprise and regular traces across sessions
    if 't' in analyses:
        acr_sess_analys.run_surp_traces(sessions, 't', analyspar, sesspar, 
                        stimpar, basepar, figpar, parallel=parallel)
    all_check += 't'

    # 3. Plots the surprise and regular traces locked to surprise
    # across sessions
    if 'r' in analyses:
        acr_sess_analys.run_lock_traces(sessions, 'r', analyspar, sesspar, 
                        stimpar, basepar, figpar, parallel=parallel)
    all_check += 'r'

    # 4. Plots the surprise latencies across sessions
    if 'u' in analyses:
        acr_sess_analys.run_surp_latency(sessions, 'u', analyspar, 
                        sesspar, stimpar, latpar, figpar, permpar, 
                        parallel=parallel)
    all_check += 'u'

    # 5. Plots proportion of ROIs responses to both surprise types
    if 'p' in analyses:
        acr_sess_analys.run_resp_prop(sessions, 'c', analyspar, 
                        sesspar, stimpar, latpar, figpar, permpar, 
                        parallel=parallel)
    all_check += 'p'



    ################## RATIO ANALYSIS ########################


    if set(all_analyses) != set(all_check):
        raise ValueError('all_analyses variable is missing some analysis '
                         'letters!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

        # general parameters
    parser.add_argument('--datadir', default=None, 
                        help=('data directory (if None, uses a directory '
                              'defined below'))
    parser.add_argument('--output', default='', help='where to store output')
    parser.add_argument('--analyses', default='all', 
                        help=('analyses to run: traces (t), locked traces (l), '
                              'roi_grps_qu (q), roi_grps_ch (c), mag (m), '
                              'autocorr (a), ori/dir (o), tuning curves (c) '
                              'or `all` or `all_m` to, for example, '
                              'run all analyses except m'))
    parser.add_argument('--sess_n', default='1-3',
                        help='session range to include, where last value is '
                             'included, e.g. 1-3, all')
    parser.add_argument('--dict_path', default='', 
                        help=('path to info dictionary or directory of '
                              'dictionaries from which to plot data.'))

        # technical parameters
    parser.add_argument('--plt_bkend', default=None, 
                        help='switch mpl backend when running on server')
    parser.add_argument('--parallel', action='store_true', 
                        help='do runs in parallel.')
    parser.add_argument('--seed', default=-1, type=int, 
                        help='random seed (-1 for None)')

        # session parameters
    parser.add_argument('--runtype', default='prod', help='prod or pilot')
    parser.add_argument('--plane', default='any', help='soma, dend, any')
    parser.add_argument('--min_rois', default=5, type=int, 
                        help='min rois criterion')
        # stimulus parameters
    parser.add_argument('--bri_dir', default='both', 
                        help='brick dir (right, left, or both)') 
    parser.add_argument('--gabfr', default=3, type=int, 
                        help='gabor frame at which to start sequences')  
    parser.add_argument('--post', default=0.45, type=float, 
                        help='sec after reference frames')
    parser.add_argument('--stimtype', default='gabors', 
                        help='stimulus to analyse')   
        # roi group parameters
    parser.add_argument('--plot_vals', default='surp', 
                        help='plot both (with op applied), surp or reg')
    
    # generally fixed 
        # analysis parameters
    parser.add_argument('--keepnans', action='store_true', 
                        help='keep ROIs containing NaNs or Infs in session.')
    parser.add_argument('--fluor', default='dff', help='raw or dff')
    parser.add_argument('--stats', default='mean', help='plot mean or median')
    parser.add_argument('--error', default='sem', 
                        help='sem for SEM/MAD, std for std/qu')    
    parser.add_argument('--dend', default='extr', help='aibs, extr')
    parser.add_argument('--no_scale', action='store_true', help='scale ROIs')
        # session parameters
    parser.add_argument('--line', default='any', help='L23, L5')
    parser.add_argument('--closest', action='store_true', 
                        help=('if True, the closest session number is used. '
                              'Otherwise, only exact.'))
    parser.add_argument('--pass_fail', default='P', 
                        help='P to take only passed sessions')
    parser.add_argument('--incl', default='any',
                        help='include only `yes`, `no` or `any`')
        # stimulus parameters
    parser.add_argument('--bri_size', default=128, 
                        help='brick size (128, 256, or both)')
    parser.add_argument('--gabk', default=16,
                        help='kappa value (4, 16, or both)')    
    parser.add_argument('--gab_ori', default='all',
                        help='gabor orientation values (0, 45, 90, 135, all)')    
    parser.add_argument('--pre', default=0, type=float, help='sec before frame')
        # permutation parameters
    parser.add_argument('--n_perms', default=10000, type=int, 
                        help='nbr of permutations')
    parser.add_argument('--tails', default='2', 
                        help='nbr tails for perm analysis (2, lo, up)')
        # baseline parameter
    parser.add_argument('--base', default=0, type=float,
                        help='baseline for surprise difference calculations.')
 
        # latency parameters
    parser.add_argument('--method', default='ttest', 
                        help='latency calculation method (`ratio` or `ttest`)')
    parser.add_argument('--p_val_thr', default='0.005', type=float,
                        help='p-value threshold for ttest method')
    parser.add_argument('--rel_std', default='0.5', type=float,
                        help='relative st. dev. threshold for ratio method')
    parser.add_argument('--not_surp_resp', action='store_true', 
                        help='don\'t use only surprise responsive ROIs')

        # figure parameters
    parser.add_argument('--ncols', default=4, help='number of columns')
    parser.add_argument('--no_datetime', action='store_true',
                        help='create a datetime folder')
    parser.add_argument('--overwrite', action='store_true', 
                        help='allow overwriting')
        # plot using modif_analys_plots (only if plotting from dictionary)
    parser.add_argument('--modif', action='store_true', 
                        help=('plot from dictionary using modified plot '
                              'functions'))

    args = parser.parse_args()

    args.fontdir = os.path.join('..', 'tools', 'fonts')

    if args.dict_path is not '':
        source = 'acr_sess'
        if args.modif:
            source = 'modif'
        plot_dicts.plot_from_dicts(args.dict_path, source=source, 
                   plt_bkend=args.plt_bkend, fontdir=args.fontdir, 
                   parallel=args.parallel)

    else:
        if args.datadir is None:
            args.datadir = os.path.join('..', 'data', 'AIBS')

        mouse_df = 'mouse_df.csv'
        args = reformat_args(args)

        analys_pars = prep_analyses(args.sess_n, args, mouse_df, args.parallel)
        run_analyses(*analys_pars, analyses=args.analyses, 
                     parallel=args.parallel)

                