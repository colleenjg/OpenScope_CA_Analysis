"""
run_run_analysis.py

This script runs running analyses using a Session object with data generated 
by the AIBS experiments for the Credit Assignment Project.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import argparse
import copy
import multiprocessing
import os
import re

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import pdb

from util import file_util, gen_util, math_util, plot_util
from sess_util import sess_gen_util, sess_ntuple_util, sess_plot_util, \
                      sess_str_util
from analysis import session, gen_analys
from plot_fcts import gen_analysis_plots as gen_plots
from plot_fcts import modif_analysis_plots as mod_plots


#############################################
def reformat_args(args):
    """
    reformat_args(args)

    Returns reformatted args for analyses, specifically 
        - Sets stimulus parameters to 'none' if they are irrelevant to the 
          stimtype
        - Changes stimulus parameters from 'both' to actual values

    Adds the following args:
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
    
    Returns:
        - args (Argument parser): input parser, with the following attributes
                                  modified: 
                                      bri_dir, bri_size, gabfr, gabk, grps, 
                                  and the following attributes added:
                                      omit_sess, omit_mice
    """
    args = copy.deepcopy(args)

    [args.bri_dir, args.bri_size, 
          args.gab_fr, args.gabk, 
                     args.gab_ori] = sess_gen_util.get_params(args.stimtype, 
                                                   args.bri_dir, args.bri_size, 
                                                   args.gabfr, args.gabk, 
                                                   args.gab_ori)

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
                                       permpar, quintpar
        - in the following dictionary: figpar 

    Required args:
        - args (Argument parser): parser with the following attributes:

            bri_dir (str or list)  : brick direction values to include
                                     ('right', 'left', ['right', 'left'])
            bri_size (int or list) : brick size values to include
                                     (128, 256 or [128, 256])
            closest (bool)         : if False, only exact session number is 
                                     retained, otherwise the closest.
            error (str)            : error statistic parameter ('std' or 'sem')
            fontdir (str)          : path to directory containing additional 
                                     fonts
            gabfr (int)            : gabor frame at which sequences start 
                                     (0, 1, 2, 3)
            gabk (int or list)     : gabor kappa values to include 
                                     (4, 16 or [4, 16])
            gab_ori (int or list)  : gabor orientation values to include
                                     ([0, 45, 90, 135])
            keepnans (str)         : if True, ROIs with NaN/Inf values are 
                                     kept in the analyses.
            lag_s (num)            : lag for autocorrelation (in sec)
            layer (str)            : layer ('soma', 'dend', 'L23_soma', 
                                     'L5_soma', 'L23_dend', 'L5_dend', 
                                     'L23_all', 'L5_all')
            min_rois (int)         : min number of ROIs
            n_perms (int)          : nbr of permutations to run
            n_quints (int)         : number of quintiles
            no_datetime (bool)     : if True, figures are not saved in a 
                                     subfolder named based on the date and time.
            output (str)           : general directory in which to save output
            overwrite (bool)       : if False, overwriting existing figures 
                                     is prevented by adding suffix numbers.
            pass_fail (str or list): pass/fail values of interest ('P', 'F')
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

    Returns:
        - sesspar (SessPar)        : named tuple of session parameters
        - stimpar (StimPar)        : named tuple of stimulus parameters
        - analyspar (AnalysPar)    : named tuple of analysis parameters
        - autocorrpar (AutocorrPar): named tuple of autocorrelation 
                                     parameters
        - permpar (PermPar)        : named tuple of permutation parameters
        - quintpar (QuintPar)      : named tuple of quintile parameters
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

    # analysis parameters
    analyspar = sess_ntuple_util.init_analyspar('n/a', not(args.keepnans), 
                                                args.stats, args.error)

    # session parameters
    sesspar = sess_ntuple_util.init_sesspar(args.sess_n, args.closest, 
                                            args.layer, 'any', args.min_rois, 
                                            args.pass_fail, args.runtype)

    # stimulus parameters
    stimpar = sess_ntuple_util.init_stimpar(args.bri_dir, args.bri_size, 
                                            args.gabfr, args.gabk, 
                                            args.gab_ori, args.pre, 
                                            args.post, args.stimtype)

    # SPECIFIC ANALYSES    
    # autocorrelation parameters
    autocorrpar = sess_ntuple_util.init_autocorrpar(args.lag_s, byitem=False)
    
    # permutation parameters
    permpar = sess_ntuple_util.init_permpar(args.n_perms, 0.05, args.tails)
    
    # quintile parameters
    quintpar = sess_ntuple_util.init_quintpar(args.n_quints, [0, -1])

    # figure parameters
    figpar = sess_plot_util.init_figpar(datetime=not(args.no_datetime), 
                                        overwrite=args.overwrite, 
                                        runtype=args.runtype, 
                                        output=args.output, 
                                        plt_bkend=args.plt_bkend, 
                                        fontdir=args.fontdir)

    return [analyspar, sesspar, stimpar, autocorrpar, permpar, quintpar, figpar]


#############################################
def prep_analyses(sess_n, args, mouse_df):
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

    Returns:
        - sessions (list)          : list of sessions, or nested list per mouse 
                                     if sess_n is a combination
        - analyspar (AnalysPar)    : named tuple containing analysis parameters
        - sesspar (SessPar)        : named tuple containing session parameters
        - stimpar (StimPar)        : named tuple containing stimulus parameters
        - autocorrpar (AutocorrPar): named tuple containing autocorrelation 
                                     parameters
        - permpar (PermPar)        : named tuple containing permutation 
                                     parameters
        - quintpar (QuintPar)      : named tuple containing quintile 
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
        - seed (int)               : seed to use
    """

    args = copy.deepcopy(args)

    # chose a seed if none is provided (i.e., args.seed=-1), but seed later
    seed = gen_util.seed_all(args.seed, 'cpu', print_seed=False, 
                             seed_now=False)

    args.sess_n = sess_n

    [analyspar, sesspar, stimpar, autocorrpar,           
                    permpar, quintpar, figpar] = init_param_cont(args)
    
    # get session IDs and create Sessions
    sessids = sess_gen_util.sess_per_mouse(mouse_df, 
                                            omit_sess=args.omit_sess, 
                                            omit_mice=args.omit_mice, 
                                            **sesspar._asdict())
    sessions = sess_gen_util.init_sessions(sessids, args.datadir, mouse_df, 
                                            sesspar.runtype, fulldict=False)

    print(('\nAnalysis of {} responses to {} stimuli ({} data)'
           '\nSession {}').format(sesspar.layer, stimpar.stimtype[:-1],
                                  sesspar.runtype, sesspar.sess_n))

    return [sessions, analyspar, sesspar, stimpar, autocorrpar, permpar, 
            quintpar, figpar, seed]

    
#############################################
def run_analyses(sessions, analyspar, sesspar, stimpar, autocorrpar, 
                 permpar, quintpar, figpar, seed=None, analyses='all', 
                 skip='', parallel=False):
    """
    run_analyses(sessions, analyses, analyspar, sesspar, stimpar, autocorrpar, 
                 permpar, quintpar)

    Run requested analyses on sessions using the named tuples passed.
    Some analyses can be skipped (e.g., to be launched in a non parallel
    process instead.)

    Required args:
        - sessions (list)          : list of sessions
        - analyspar (AnalysPar)    : named tuple containing analysis parameters
        - sesspar (SessPar)        : named tuple containing session parameters
        - stimpar (StimPar)        : named tuple containing stimulus parameters
        - autocorrpar (AutocorrPar): named tuple containing autocorrelation 
                                     parameters
        - permpar (PermPar)        : named tuple containing permutation 
                                     parameters
        - quintpar (QuintPar)      : named tuple containing quintile 
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
    
    Returns:
        - skipped (str): any analyses skipped
    """

    all_analyses = 'tlma'
    all_check = ''

    if 'all' in analyses:
        if '_' in analyses:
            excl = analyses.split('_')[1]
            analyses, _ = gen_util.remove_lett(all_analyses, excl)
        else:
            analyses = all_analyses
    
    analyses, skipped = gen_util.remove_lett(analyses, skip)

    # changes backend and defaults
    plot_util.manage_mpl(cmap=False, **figpar['mng'])

    # 1. Analyses and plots average running by quintile x surprise for each 
    # session 
    if 't' in analyses: # traces
        gen_analys.run_traces_by_qu_surp_sess(sessions, 't', analyspar, 
                                              sesspar, stimpar, quintpar, 
                                              figpar, datatype='run')
    all_check += 't'

    # 2. Analyses and plots average running locked to surprise by quintile x 
    # surprise for each session 
    if 'l' in analyses: # surprise-locked traces
        gen_analys.run_traces_by_qu_lock_sess(sessions, 'l', seed, analyspar, 
                                              sesspar, stimpar, quintpar, 
                                              figpar, datatype='run')
    all_check += 'l'

    # 3. Analyses and plots magnitude of change in dF/F area from first to last 
    # quintile of surprise vs no surprise sequences, for each session
    if 'm' in analyses: # mag
        gen_analys.run_mag_change(sessions, 'm', seed, analyspar, sesspar, 
                                  stimpar, permpar, quintpar, figpar, 
                                  datatype='run')
    all_check += 'm'

    # 4. Analyses and plots autocorrelation
    if 'a' in analyses: # autocorr
        gen_analys.run_autocorr(sessions, 'a', analyspar, sesspar, stimpar, 
                                autocorrpar, figpar, datatype='run')
    all_check += 'a'


    if set(all_analyses) != set(all_check):
        raise ValueError(('all_analyses variable is missing some analysis '
                          'letters!'))

    return skipped


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

        # general parameters
    parser.add_argument('--datadir', default=None, 
                        help=('data directory (if None, uses a directory '
                              'defined below'))
    parser.add_argument('--output', default='', help='where to store output')
    parser.add_argument('--plt_bkend', default=None, 
                        help='mpl backend to use, e.g. when running on server')
    parser.add_argument('--analyses', default='all', 
                        help=('analyses to run: traces (t), locked traces (l), '
                              'mag (m), autocorr (a) or `all` or `all_m` '
                              'to, for example, run all analyses except m'))
    parser.add_argument('--sess_n', default='all',
                        help='session to aim for, e.g. 1, 2, last, all')
    parser.add_argument('--parallel', action='store_true', 
                        help='do sess_n\'s in parallel.')
    parser.add_argument('--dict_path', default='', 
                        help=('path to info dictionary from which to plot '
                              'data.'))
    parser.add_argument('--seed', default=-1, type=int, 
                        help='random seed (-1 for None)')

        # session parameters
    parser.add_argument('--runtype', default='prod', help='prod or pilot')
    parser.add_argument('--layer', default='soma', help='soma, dend')
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
    
    # generally fixed 
        # analysis parameters
    parser.add_argument('--keepnans', action='store_true', 
                        help='keep ROIs containing NaNs or Infs in session.')
    parser.add_argument('--stats', default='mean', help='plot mean or median')
    parser.add_argument('--error', default='sem', 
                        help='sem for SEM/MAD, std for std/qu')    
        # session parameters
    parser.add_argument('--closest', action='store_true', 
                        help=('if True, the closest session number is used. '
                              'Otherwise, only exact.'))
    parser.add_argument('--pass_fail', default='P', 
                        help='P to take only passed sessions')
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
        # figure parameters
    parser.add_argument('--no_datetime', action='store_true',
                        help='create a datetime folder')
    parser.add_argument('--overwrite', action='store_true', 
                        help='allow overwriting')
        # plot using modif_analys_plots (if plotting from dictionary)
    parser.add_argument('--modif', action='store_true', 
                        help=('plot from dictionary using modified plot '
                              'functions'))
    args = parser.parse_args()





    args.runtype = 'pilot'
    args.layer = 'dend'
    args.bri_dir = 'left'
    args.analyses = 'a'







    args.fontdir = os.path.join('..', 'tools', 'fonts')

    if args.dict_path is not '':
        main_dir  = os.path.join('results', 'figures')
        dict_path = os.path.join(main_dir, args.dict_path)
        if args.modif:
            mod_plots.plot_from_dict(dict_path, args.parallel, args.plt_bkend, 
                                     args.fontdir)
        else:
            gen_plots.plot_from_dict(dict_path, args.parallel, args.plt_bkend, 
                                     args.fontdir)

    else:
        if args.datadir is None:
            args.datadir = os.path.join('..', 'data', 'AIBS')

        mouse_df = 'mouse_df.csv'
        args = reformat_args(args)

        # get numbers of sessions to analyse
        if args.sess_n == 'all':
            all_sess_ns = sess_gen_util.get_sess_vals(mouse_df, 'sess_n', 
                            runtype=args.runtype, layer=args.layer, 
                            min_rois=args.min_rois, pass_fail=args.pass_fail, 
                            omit_sess=args.omit_sess, omit_mice=args.omit_mice)
        else:
            all_sess_ns = gen_util.list_if_not(args.sess_n)

        # run through all sessions
        if args.parallel:
            n_cores = multiprocessing.cpu_count()
            n_jobs = min(n_cores, len(all_sess_ns))

            # initialize sessions and collect analysis params
            all_analys_pars = Parallel(n_jobs=n_jobs)(delayed(prep_analyses)
                                      (sess_n, args, mouse_df) 
                                      for sess_n in all_sess_ns)
            # run analyses, and record any skipped analyses (to be run in 
            # sequential, as they themselves have been parallelized)
            run_seq = ''
            if not(set(args.analyses).issubset(set(run_seq))):
                skipped = Parallel(n_jobs=n_jobs)(delayed(run_analyses)
                                  (*analys_pars, analyses=args.analyses, 
                                   skip=run_seq, parallel=False)
                                   for analys_pars in all_analys_pars)[0]
            else:
                skipped = args.analyses
            
            # run skipped analyses in sequential
            if len(skipped) != 0:
                for analys_pars in all_analys_pars:
                    run_analyses(*analys_pars, analyses=skipped, parallel=True)
            
        else:
            for sess_n in all_sess_ns:
                analys_pars = prep_analyses(sess_n, args, mouse_df)
                run_analyses(*analys_pars, analyses=args.analyses, 
                parallel=False)

                