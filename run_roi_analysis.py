"""
run_roi_analysis.py

This script runs ROI trace analyses using a Session object with data generated 
by the AIBS experiments for the Credit Assignment Project.

Authors: Colleen Gillon

Date: October, 2018

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
from analysis import session, roi_analys, gen_analys
from plot_fcts import plot_from_dicts_tool as plot_dicts



#############################################
def reformat_args(args):
    """
    reformat_args(args)

    Returns reformatted args for analyses, specifically 
        - Sets stimulus parameters to 'none' if they are irrelevant to the 
          stimtype
        - Changes stimulus parameters from 'both' to actual values
        - Changes grps string values to a list

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
            stimtype (str)       : stimulus to analyse (bricks or gabors)
            grps (str)           : set or sets of groups to plot, 
                                   (e.g., 'all change no_change reduc incr').
    
    Returns:
        - args (Argument parser): input parser, with the following attributes
                                  modified: 
                                      bri_dir, bri_size, gabfr, gabk, gab_ori, 
                                      grps 
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

    args.grps = gen_util.str_to_list(args.grps)

    args.omit_sess, args.omit_mice = sess_gen_util.all_omit(args.stimtype, 
                                                    args.runtype, args.bri_dir, 
                                                    args.bri_size, args.gabk)
    
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

            bri_dir (str or list)  : brick direction values to include
                                     ('right', 'left', ['right', 'left'])
            bri_size (int or list) : brick size values to include
                                     (128, 256 or [128, 256])
            closest (bool)         : if False, only exact session number is 
                                     retained, otherwise the closest.
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
            grps (str or list)     : set or sets of groups to return, 
                                     ('all', 'change', 'no_change', 'reduc', 
                                     'incr'.)
            incl (str)             : sessions to include ('yes', 'no', 'all') 
            keepnans (str)         : if True, ROIs with NaN/Inf values are 
                                     kept in the analyses.
            lag_s (num)            : lag for autocorrelation (in sec)
            line (str)             : 'L23', 'L5', 'any'
            min_rois (int)         : min number of ROIs
            n_perms (int)          : nbr of permutations to run
            n_quints (int)         : number of quintiles
            ncols (int)            : number of columns
            no_add_reg (bool)      : if True, the group of ROIs showing no 
                                     significance in either is not added to   
                                     the groups returned
            no_datetime (bool)     : if True, figures are not saved in a 
                                     subfolder named based on the date and time.
            not_byitem (bool)      : if True, autocorrelation statistics are
                                     taken across items (e.g., ROIs)
            op (str)               : operation on values, if plotvals if 'both' 
                                     ('ratio' or 'diff') 
            output (str)           : general directory in which to save output
            overwrite (bool)       : if False, overwriting existing figures 
                                     is prevented by adding suffix numbers.
            pass_fail (str or list): pass/fail values of interest ('P', 'F')
            plot_vals (str)        : values to plot ('surp', 'reg', 'both')
            plane (str)            : plane ('soma', 'dend', 'any')
            plt_bkend (str)        : mpl backend to use
            post (num)             : range of frames to include after each 
                                     reference frame (in s)
            pre (num)              : range of frames to include before each 
                                     reference frame (in s)
            runtype (str or list)  : runtype ('pilot' or 'prod')
            sess_n (int)           : session number
            stats (str)            : statistic parameter ('mean' or 'median')
            stimtype (str)         : stimulus to analyse ('bricks' or 'gabors')
            tails (str or int)     : which tail(s) to test ('up', 'lo', 2)
            tc_gabfr (int or str)  : gabor frame at which sequences start 
                                     (0, 1, 2, 3) for tuning curve analysis
                                     (x_x, interpreted as 2 gabfrs)
            tc_grp2 (str)          : second group: either surp, reg or rand 
                                     (random subsample of reg, the size of 
                                     surp)
            tc_post (num)          : range of frames to include after each 
                                     reference frame (in s) for tuning curve 
                                     analysis
            tc_prev (bool)         : runs analysis using previous parameter 
                                     estimation method
            tc_test (bool)         : if True, tuning curve analysis is run on a 
                                     small subset of ROIs and gabors

    Returns:
        - analyspar (AnalysPar)    : named tuple of analysis parameters
        - sesspar (SessPar)        : named tuple of session parameters
        - stimpar (StimPar)        : named tuple of stimulus parameters
        - autocorrpar (AutocorrPar): named tuple of autocorrelation 
                                     parameters
        - permpar (PermPar)        : named tuple of permutation parameters
        - quintpar (QuintPar)      : named tuple of quintile parameters
        - roigrppar (RoiGrpPar)    : named tuple of roi grp parameters
        - tcurvpar (TCurvPar)      : named tuple of tuning curve parameters
        - figpar (dict)            : dictionary containing following 
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
                                                dend=args.dend)

    # session parameters
    sesspar = sess_ntuple_util.init_sesspar(args.sess_n, args.closest, 
                               args.plane, args.line, args.min_rois, 
                               args.pass_fail, args.incl, args.runtype)
    
    # stimulus parameters
    stimpar = sess_ntuple_util.init_stimpar(args.stimtype, args.bri_dir, 
                                            args.bri_size, args.gabfr, 
                                            args.gabk, args.gab_ori, 
                                            args.pre, args.post)

    # SPECIFIC ANALYSES    
    # autocorrelation parameters
    autocorrpar = sess_ntuple_util.init_autocorrpar(args.lag_s, 
                                   not(args.not_byitem))
    
    # permutation parameters
    permpar = sess_ntuple_util.init_permpar(args.n_perms, 0.05, args.tails)
    
    # quintile parameters
    quintpar = sess_ntuple_util.init_quintpar(args.n_quints, [0, -1])

    # roi grp parameters
    roigrppar = sess_ntuple_util.init_roigrppar(args.grps, not(args.no_add_reg), 
                                 args.op, args.plot_vals)

    # tuning curve parameters
    tcurvpar = sess_ntuple_util.init_tcurvpar(args.tc_gabfr, 0, args.tc_post, 
                                args.tc_grp2, args.tc_test, args.tc_prev)

    # figure parameters
    figpar = sess_plot_util.init_figpar(ncols=int(args.ncols), 
                            datetime=not(args.no_datetime), 
                            overwrite=args.overwrite, runtype=args.runtype, 
                            output=args.output, plt_bkend=args.plt_bkend, 
                            fontdir=args.fontdir)

    return [analyspar, sesspar, stimpar, autocorrpar, permpar, quintpar, 
            roigrppar, tcurvpar, figpar]


#############################################
def prep_analyses(sess_n, args, mouse_df):
    """
    prep_analyses(sess_n, args, mouse_df)

    Prepares named tuples and sessions for which to run analyses, based on the 
    arguments passed.

    Required args:
        - sess_n (int)          : session number to run analyses on, or 
                                  session numbers to compare, 
                                  e.g. '1v2'
        - args (Argument parser): parser containing all parameters
        - mouse_df (pandas df)  : path name of dataframe containing information 
                                  on each session

    Returns:
        - sessions (list)          : list of sessions, or nested list per mouse 
                                     if sess_n is a combination to compare
        - analyspar (AnalysPar)    : named tuple containing analysis parameters
        - sesspar (SessPar)        : named tuple containing session parameters
        - stimpar (StimPar)        : named tuple containing stimulus parameters
        - autocorrpar (AutocorrPar): named tuple containing autocorrelation 
                                     parameters
        - permpar (PermPar)        : named tuple containing permutation 
                                     parameters
        - quintpar (QuintPar)      : named tuple containing quintile 
                                     parameters
        - roigrppar (RoiGrpPar)    : named tuple containing ROI group 
                                     parameters
        - tcurvpar (TCurvPar)      : named tuple containing tuning curve 
                                     parameters
        - figpar (dict)            : dictionary containing following 
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

    comp = True
    if 'v' not in str(sess_n):
        comp = False
        if sess_n not in ['first', 'last']:
            sess_n = int(sess_n)
        
    args.sess_n = sess_n

    [analyspar, sesspar, stimpar, autocorrpar, permpar,
          quintpar, roigrppar, tcurvpar, figpar] = init_param_cont(args)
    
    # get session IDs and create Sessions
    if comp:
        sessids = sess_gen_util.sess_comp_per_mouse(mouse_df, 
                                omit_sess=args.omit_sess, 
                                omit_mice=args.omit_mice, **sesspar._asdict())
        sessions = []
        for ids in sessids:
            subs = sess_gen_util.init_sessions(ids, args.datadir, mouse_df, 
                                 sesspar.runtype, fulldict=False,
                                 dend=analyspar.dend, omit=True)
            if len(subs) == 2:
                sessions.append(subs)
            else:
                print(f'Omitting session {subs[0].sessid} due to incomplete '
                       'pair.')
    else:
        sessids = sess_gen_util.sess_per_mouse(mouse_df, 
                                omit_sess=args.omit_sess, 
                                omit_mice=args.omit_mice, **sesspar._asdict())
        sessions = sess_gen_util.init_sessions(sessids, args.datadir, mouse_df, 
                                               sesspar.runtype, fulldict=False, 
                                               dend=analyspar.dend, omit=True)

    if len(sessids) == 0:
        raise ValueError('No sessions meet the criteria.')

    print(f'\nAnalysis of {sesspar.plane} responses to {stimpar.stimtype[:-1]} '
          f'stimuli ({sesspar.runtype} data)\nSession {sesspar.sess_n}')

    return [sessions, analyspar, sesspar, stimpar, autocorrpar, permpar, 
            quintpar, roigrppar, tcurvpar, figpar, seed]

    
#############################################
def run_analyses(sessions, analyspar, sesspar, stimpar, autocorrpar, 
                 permpar, quintpar, roigrppar, tcurvpar, figpar, seed=None, 
                 analyses='all', skip='', parallel=False, plot_tc=True):
    """
    run_analyses(sessions, analyspar, sesspar, stimpar, autocorrpar, 
                 permpar, quintpar, roigrppar, tcurvpar, figpar)

    Run requested analyses on sessions using the named tuples passed.
    Some analyses can be skipped (e.g., to be launched in a non parallel
    process instead.)

    Required args:
        - sessions (list)          : list of sessions, possibly nested
        - analyspar (AnalysPar)    : named tuple containing analysis parameters
        - sesspar (SessPar)        : named tuple containing session parameters
        - stimpar (StimPar)        : named tuple containing stimulus parameters
        - autocorrpar (AutocorrPar): named tuple containing autocorrelation 
                                     parameters
        - permpar (PermPar)        : named tuple containing permutation 
                                     parameters
        - quintpar (QuintPar)      : named tuple containing quintile 
                                     parameters
        - roigrppar (RoiGrpPar)    : named tuple containing ROI group 
                                     parameters
        - tcurvpar (TCurvPar)      : named tuple containing tuning curve 
                                     parameters
        - figpar (dict)            : dictionary containing figure parameters
    
    Optional args:
        - seed (int)     : seed to use
                           default: None
        - analyses (str) : analyses to run
                           default: 'all'
        - skip (str)     : analyses to skip
                           default: ''
        - parallel (bool): if True, some analyses are parallelized 
                           across CPU cores 
                           default: False
        - plot_tc (bool) : if True, tuning curves are plotted for each ROI  
                           default: True
    
    Returns:
        - skipped (str): any analyses skipped
    """

    all_analyses = 'ftlmagocpr'
    all_check = ''

    if 'all' in analyses:
        if '_' in analyses:
            excl = analyses.split('_')[1]
            analyses, _ = gen_util.remove_lett(all_analyses, excl)
        else:
            analyses = all_analyses
    
    analyses, skipped = gen_util.remove_lett(analyses, skip)

    if len(sessions) == 0:
        print('No sessions fit these criteria.')
        return skipped

    comp = False
    if isinstance(sessions[0], list):
        comp = True

    # changes backend and defaults
    plot_util.manage_mpl(cmap=False, **figpar['mng'])

    # 0. Plots the full traces for each session
    if 'f' in analyses and not comp: # full traces
        gen_analys.run_full_traces(sessions, 'f', analyspar, sesspar, figpar)
    all_check += 'f'

    # 1. Analyses and plots average traces by quintile x surprise for each 
    # session 
    if 't' in analyses and not comp: # traces
        gen_analys.run_traces_by_qu_surp_sess(sessions, 't', analyspar, 
                                              sesspar, stimpar, quintpar, 
                                              figpar)
    all_check += 't'

    # 2. Analyses and plots average traces locked to surprise by quintile x 
    # surprise for each session 
    if 'l' in analyses and not comp: # surprise-locked traces
        gen_analys.run_traces_by_qu_lock_sess(sessions, 'l', seed, analyspar, 
                                              sesspar, stimpar, quintpar, 
                                              figpar)
    all_check += 'l'

    # 3. Analyses and plots magnitude of change in dF/F area from first to last 
    # quintile of surprise vs no surprise sequences, for each session
    if 'm' in analyses and not comp: # mag
        gen_analys.run_mag_change(sessions, 'm', seed, analyspar, sesspar, 
                                  stimpar, permpar, quintpar, figpar)
    all_check += 'm'

    # 4. Analyses and plots autocorrelation
    if 'a' in analyses and not comp: # autocorr
        gen_analys.run_autocorr(sessions, 'a', analyspar, sesspar, stimpar, 
                                autocorrpar, figpar)
    all_check += 'a'

    # 5. Analyses and plots: a) trace areas by quintile, b) average traces, 
    # c) trace areas by suprise for first vs last quintile, for each ROI group, 
    # for each session
    if 'g' in analyses and not comp: # roi_grps_ch
        roi_analys.run_rois_by_grp(sessions, 'g', seed, analyspar, sesspar, 
                                   stimpar, permpar, quintpar, roigrppar, 
                                   figpar)
    all_check += 'g'

    # 6. Analyses and plots colormaps by orientation or direction, as well as 
    # average traces
    if 'o' in analyses and not comp: # colormaps and traces
        roi_analys.run_oridirs(sessions, 'o', analyspar, sesspar, stimpar, 
                               quintpar, figpar, parallel)
    all_check += 'o'

    # 7. Analyses and plots ROI tuning curves for gabor orientation
    if 'c' in analyses and not comp: # tune curves
        roi_analys.run_tune_curves(sessions, 'c', seed, analyspar, sesspar, 
                                   stimpar, tcurvpar, figpar, parallel, 
                                   plot_tc)
    all_check += 'c'

    # 8. Analyses and plots ROI responses for positions and mean gabor 
    # orientations
    if 'p' in analyses and not comp: # position orientation resp
        roi_analys.run_posori_resp(sessions, 'p', analyspar, sesspar, stimpar, 
                                   figpar, parallel)
    all_check += 'p'

    # 9. Analyses and plots ROI responses for positions and mean gabor 
    # orientations
    if 'r' in analyses and comp: # correlation
        gen_analys.run_trace_corr_acr_sess(sessions, 'r', analyspar, sesspar, 
                                           stimpar, figpar)
    all_check += 'r'


    ################## RATIO ANALYSIS ########################


    if set(all_analyses) != set(all_check):
        raise ValueError('all_analyses variable is missing some analysis '
                         'letters!')

    return skipped


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
    parser.add_argument('--sess_n', default='all',
                        help='session to aim for, e.g. 1, 2, first, last, all')
    parser.add_argument('--dict_path', default='', 
                        help=('path to info dictionary or directory of '
                              'dictionaries from which to plot data.'))
    parser.add_argument('--no_plot_tc', action='store_true', 
                        help='no tuning curve plots are generated')

        # technical parameters
    parser.add_argument('--plt_bkend', default=None, 
                        help='switch mpl backend when running on server')
    parser.add_argument('--parallel', action='store_true', 
                        help='do runs in parallel.')
    parser.add_argument('--seed', default=-1, type=int, 
                        help='random seed (-1 for None)')

        # session parameters
    parser.add_argument('--runtype', default='prod', help='prod or pilot')
    parser.add_argument('--plane', default='soma', help='soma, dend')
    parser.add_argument('--min_rois', default=5, type=int, 
                        help='min rois criterion')
        # stimulus parameters
    parser.add_argument('--bri_dir', default='right', 
                        help='brick dir (right, left, or both)') 
    parser.add_argument('--gabfr', default=3, type=int, 
                        help='gabor frame at which to start sequences')  
    parser.add_argument('--post', default=1.5, type=float, 
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
        # quintile parameters
    parser.add_argument('--n_quints', default=4, type=int, 
                        help='nbr of quintiles')
        # autocorrelation parameters
    parser.add_argument('--lag_s', default=4, type=float,
                        help='lag for autocorrelation (in sec)')
    parser.add_argument('--not_byitem', action='store_true',
                        help=('if True, autocorrelation stats are taken '
                              'across ROIs'))
        # roi grp parameters
    parser.add_argument('--op', default='diff', 
                        help='calculate diff or ratio of surp to nonsurp')
    parser.add_argument('--grps', default='reduc incr no_change', 
                        help=('plot all ROI grps or grps with change or '
                              'no_change'))
    parser.add_argument('--no_add_reg', action='store_true',
                        help='do not add reg_reg to ROI grp plots')
        # tuning curve parameters
    parser.add_argument('--tc_gabfr', default=3, 
                        help='gabor frame at which to start sequences (if '
                             'x_x, interpreted as 2 gabfrs)')  
    parser.add_argument('--tc_post', default=0.6, type=float, 
                        help='sec after reference frames')
    parser.add_argument('--tc_grp2', default='surp', 
                        help=('second group: either surp, reg or rand '
                              '(random subsample of reg, the size of surp)'))
    parser.add_argument('--tc_test', action='store_true',
                        help='tests code on a small number of gabors and ROIs')
    parser.add_argument('--tc_prev', action='store_true',
                        help=('runs analysis using previous parameter '
                              'estimation method'))
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
        source = 'roi'
        if args.modif:
            source = 'modif'
        plot_dicts.plot_from_dicts(args.dict_path, source=source, 
                   plt_bkend=args.plt_bkend, fontdir=args.fontdir, 
                   plot_tc=not(args.no_plot_tc), parallel=args.parallel)

    else:
        if args.datadir is None:
            args.datadir = os.path.join('..', 'data', 'AIBS')

        mouse_df = 'mouse_df.csv'
        args = reformat_args(args)

        # get numbers of sessions to analyse
        if args.sess_n == 'all':
            all_sess_ns = sess_gen_util.get_sess_vals(mouse_df, 'sess_n', 
                            runtype=args.runtype, plane=args.plane, 
                            line=args.line, min_rois=args.min_rois, 
                            pass_fail=args.pass_fail, incl=args.incl, 
                            omit_sess=args.omit_sess, omit_mice=args.omit_mice)
        else:
            all_sess_ns = gen_util.list_if_not(args.sess_n)

        # run through all sessions
        if args.parallel:
            n_jobs = gen_util.get_n_jobs(len(all_sess_ns))
            # initialize sessions and collect analysis params
            all_analys_pars = Parallel(n_jobs=n_jobs)(delayed(prep_analyses)
                                      (sess_n, args, mouse_df) 
                                      for sess_n in all_sess_ns)
            # run analyses, and record any skipped analyses (to be run in 
            # sequential, as they themselves have been parallelized)
            run_seq = 'oc'
            if not(set(args.analyses).issubset(set(run_seq))):
                skipped = Parallel(n_jobs=n_jobs)(delayed(run_analyses)
                                  (*analys_pars, analyses=args.analyses, 
                                   skip=run_seq, parallel=False, 
                                   plot_tc=not(args.no_plot_tc))
                                   for analys_pars in all_analys_pars)[0]
            else:
                skipped = args.analyses
            
            # run skipped analyses in sequential
            if len(skipped) != 0:
                for analys_pars in all_analys_pars:
                    run_analyses(*analys_pars, analyses=skipped, parallel=True, 
                                plot_tc=not(args.no_plot_tc))
            
        else:
            for sess_n in all_sess_ns:
                analys_pars = prep_analyses(sess_n, args, mouse_df)
                run_analyses(*analys_pars, analyses=args.analyses, 
                parallel=False, plot_tc=not(args.no_plot_tc))

                