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
import os

import numpy as np

from util import file_util, gen_util, math_util, plot_util
from sess_util import sess_gen_util, sess_ntuple_util, sess_plot_util, \
                      sess_str_util
from analysis import session, gen_analys
from plot_fcts import gen_analysis_plots as gen_plots


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
            incl (str)             : sessions to include ('yes', 'no', 'all')     
            keepnans (str)         : if True, the original running array is 
                                     used instead of the one where NaNs
                                     are interpolated.
            layer (str)            : layer ('soma', 'dend', 'L23_soma', 
                                     'L5_soma', 'L23_dend', 'L5_dend', 
                                     'L23_all', 'L5_all')
            min_rois (int)         : min number of ROIs
            ncols (int)            : number of columns
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

    Returns:
        - analyspar (AnalysPar)    : named tuple of analysis parameters
        - sesspar (SessPar)        : named tuple of session parameters
        - stimpar (StimPar)        : named tuple of stimulus parameters
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
    analyspar = sess_ntuple_util.init_analyspar(args.fluor, not(args.keepnans), 
                                                args.stats, args.error, 
                                                dend=args.dend)

    # session parameters
    sesspar = sess_ntuple_util.init_sesspar(args.sess_n, args.closest, 
                                            args.layer, 'any', args.min_rois, 
                                            args.pass_fail, args.incl, 
                                            args.runtype)

    # stimulus parameters
    stimpar = sess_ntuple_util.init_stimpar(args.stimtype, args.bri_dir, 
                                            args.bri_size, args.gabfr, 
                                            args.gabk, args.gab_ori, 
                                            args.pre, args.post)

    # figure parameters
    figpar = sess_plot_util.init_figpar(ncols=int(args.ncols),
                                        datetime=not(args.no_datetime), 
                                        overwrite=args.overwrite, 
                                        runtype=args.runtype, 
                                        output=args.output, 
                                        plt_bkend=args.plt_bkend, 
                                        fontdir=args.fontdir)

    return [analyspar, sesspar, stimpar, figpar]


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

    args.sess_n = sess_n

    # initializes named tuples to hold the analysis parameters (like 
    # dictionaries, but immutable)
    [analyspar, sesspar, stimpar, figpar] = init_param_cont(args)
    
    # GETS THE IDS OF SESSIONS THAT FIT THE CRITERIA
    sessids = sess_gen_util.sess_per_mouse(mouse_df, 
                                            omit_sess=args.omit_sess, 
                                            omit_mice=args.omit_mice, 
                                            **sesspar._asdict())
    
    # INITIALIZE THE SESSION OBJECTS -> CHECK THIS OUT
    # When you intialize a session, you need to run 
    #   sess.extract_sess_attribs() and
    #   sess.extract_info()
    # to get all the info you need loaded up
    sessions = sess_gen_util.init_sessions(sessids, args.datadir, mouse_df, 
                                            sesspar.runtype, fulldict=False)

    print(('\nAnalysis of {} responses to {} stimuli ({} data)'
           '\nSession {}').format(sesspar.layer, stimpar.stimtype[:-1],
                                  sesspar.runtype, sesspar.sess_n))

    return [sessions, analyspar, sesspar, stimpar, figpar]



#############################################
def run_full_traces(sessions, analysis, analyspar, sesspar, stimpar, figpar):
    """
    run_full_traces(sessions, analysis, analyspar, sesspar, stimpar, figpar)

    Plots full traces across an entire session. If ROI traces are plotted,
    each ROI is scaled and plotted separately and an average is plotted.
    
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., 't')
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)        : named tuple containing stimulus parameters
        - figpar (dict)        : dictionary containing figure parameters
    """

    # THESE JUST GENERATE STRINGS RELATED TO THE ANALYSIS PARAMETERS 
    # (FOR PRINTING TO CONSOLE, FOR EXAMPLE)
    dendstr_pr = sess_str_util.dend_par_str(analyspar.dend, sesspar.layer, 
                                            'roi', 'print')    
    sessstr_pr = 'session: {}, layer: {}{}'.format(sesspar.sess_n, 
                                                   sesspar.layer, dendstr_pr)
    datastr = sess_str_util.datatype_par_str('roi')

    print(('\nPlotting {} traces across an entire '
           'session\n({}).').format(datastr, sessstr_pr))

    # CREATES A NAME FOR THE DATE/TIME FOLDER IN WHICH THE PLOTS WILL BE SAVED 
    # I always copy a dictionary before modifying to avoid unexpected 
    # repercussions 
    figpar = copy.deepcopy(figpar) 
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()
    

    # CREATE EMPTY LISTS TO COLLECT VALUES
    all_tr, roi_tr, all_edges, all_pars = [], [], [], []
    
    # COLLECT INFO FOR EACH SESSIONS
    for sess in sessions:
        edge_fr, par_descs = [], []

        # EACH SESSION OBJECT HAS THE ATTRIBUTE *stims* WHICH IS A LIST OF
        # THE STIMULUS OBJECTS, IN ORDER
        for stim in sess.stims:
            # FOR EACH STIMULUS, GETS THE 2-PHOTON FRAMES AT THE BEGINNING AND 
            # END (TO DRAW LINES ON THE GRAPHS)
            edges = [fr for d_fr in stim.block_ran_twop_fr for fr in d_fr]
            edge_fr.extend(edges)
            
            # COLLECTS THE STIMULUS PARAMETER VALUES FOR EACH BLOCK TO WRITE 
            # THEM ONTO THE GRAPHS
            # * USES THE BLOCK_PARAMS ATTRIBUTE *block_params* THAT LISTS THE 
            # PARAMETERS FOR EACH BLOCK
            params = [pars for d_pars in stim.block_params for pars in d_pars]
            par_desc = []

            # FORMATS THE DESCRIPTIONS FOR EACH PARAMETER (TO PRINT ON GRAPHS)
            for pars in params:
                pars = [str(par) for par in pars]
                par_desc.append(sess_str_util.pars_to_desc('{}\n{}'.format(
                            stim.stimtype.capitalize(), ', '.join(pars[0:2]))))
            par_descs.extend(par_desc)
        
        
        nanpol = None
        if not analyspar.remnans:
            nanpol = 'omit'
        
        # METHOD GETS THE WHOLE TRACES
        all_rois = sess.get_roi_traces(None, analyspar.fluor, 
                                        analyspar.remnans)
        # FUNCTION CALCULATES STATISTICS FOR THE TRACS
        full_tr = math_util.get_stats(all_rois, analyspar.stats, 
                                        analyspar.error, axes=0, 
                                        nanpol=nanpol).tolist()
        


        ###########################################
        # THIS PORTION HERE IS NOT NEEDED, BUT I USE THIS A TON FOR ANALYSES 
        # WHERE ONLY SPECIFIC STIMULUS PARAMETERS ARE INCLUDED IN THE ANALYSIS
        
        # IT PICKS OUT THE SEGMENTS, GETS THE CORRESPONDING 2P FRAMES, AND 
        # THEN GETS THE STATS FOR THOSE SEGMENTS

        # THIS CAN ALSO BE DONE FOR PUPIL OR RUNNING DATA INSTEAD OF ROI DATA
        
        # returns the requested stimulus object (brick or gabor)
        stim = sess.get_stim(stimpar.stimtype) 
        # returns all the segment numbers corresponding to the criteria 
        segs = stim.get_segs_by_criteria(bri_dir=stimpar.bri_dir, 
                                         bri_size=stimpar.bri_size, 
                                         gabk=stimpar.gabk, by='seg')
        # gets the first 2-photon frame for each segment
        twopfr = stim.get_twop_fr_by_seg(segs, first=True)
        # gets the trace statistics for each ROI centered around the
        # specified 2-photon frames (stats) and the second values for each 
        # frame (xran)
        xran, stats = stim.get_roi_trace_stats(twopfr, stimpar.pre, 
                           stimpar.post, byroi=True, fluor=analyspar.fluor, 
                           remnans=analyspar.remnans, stats=analyspar.stats, 
                           error=analyspar.error)
        ###########################################

        
        
        # FILLS UP THE LISTS
        roi_tr.append(all_rois.tolist())
        all_tr.append(full_tr)
        all_edges.append(edge_fr)
        all_pars.append(par_descs)

    # CREATES DICTIONARIES FOR PLOTTING
    extrapar = {'analysis': analysis,
                'datatype': 'roi',
                }

    trace_info = {'all_tr'   : all_tr,
                  'all_edges': all_edges,
                  'all_pars' : all_pars
                  }

    sess_info = sess_gen_util.get_sess_info(sessions, analyspar.fluor)

    info = {'analyspar' : analyspar._asdict(),
            'sesspar'   : sesspar._asdict(),
            'extrapar'  : extrapar,
            'sess_info' : sess_info,
            'trace_info': trace_info
            }

    # PLOTS PURELY FROM THE DICTIONARIES. NO NEED TO CHECK IT OUT
    fulldir, savename = gen_plots.plot_full_traces(roi_tr=roi_tr, 
                                                   figpar=figpar, **info)
    # SAVES THE DICTIONARIES
    file_util.saveinfo(info, savename, fulldir, 'json')
    
 
#############################################
def run_analyses(sessions, analyspar, sesspar, stimpar, figpar, analyses='all'):
    """
    run_analyses(sessions, analyspar, sesspar, stimpar, figpar)

    Run requested analyses on sessions using the named tuples passed.

    Required args:
        - sessions (list)          : list of sessions
        - analyspar (AnalysPar)    : named tuple containing analysis parameters
        - sesspar (SessPar)        : named tuple containing session parameters
        - stimpar (StimPar)        : named tuple containing stimulus parameters
        - figpar (dict)            : dictionary containing figure parameters
    
    Optional args:
        - analyses (str) : analyses to run
                           default: 'all'
    """

    all_analyses = 'f'

    # this is only useful if several different analysis options exist
    if 'all' in analyses:
        analyses = all_analyses
    
    # sets some pyplot parameters, including the backend ('agg' needed to run 
    # on computer servers)
    plot_util.manage_mpl(cmap=False, **figpar['mng'])

    # Runs the analysis the full traces for each session
    if 'f' in analyses: # full traces

        # note that the stimpar is not actually needed for this analysis
        run_full_traces(sessions, 'f', analyspar, sesspar, stimpar, figpar)



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
                        help=('analyses to run: all or f'))
    parser.add_argument('--sess_n', default=1,
                        help='session to aim for, e.g. 1, 2, last, all')
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
    parser.add_argument('--fluor', default='dff', help='raw or dff')
    parser.add_argument('--stats', default='mean', help='plot mean or median')
    parser.add_argument('--error', default='sem', 
                        help='sem for SEM/MAD, std for std/qu')    
    parser.add_argument('--dend', default='extr', help='aibs, extr')
        # session parameters
    parser.add_argument('--closest', action='store_true', 
                        help=('if True, the closest session number is used. '
                              'Otherwise, only exact.'))
    parser.add_argument('--pass_fail', default='P', 
                        help='P to take only passed sessions')
    parser.add_argument('--incl', default='yes',
                        help='include only `yes`, `no` or `all`')
        # stimulus parameters
    parser.add_argument('--bri_size', default=128, 
                        help='brick size (128, 256, or both)')
    parser.add_argument('--gabk', default=16,
                        help='kappa value (4, 16, or both)')    
    parser.add_argument('--gab_ori', default='all',
                        help='gabor orientation values (0, 45, 90, 135, all)')    
    parser.add_argument('--pre', default=0, type=float, help='sec before frame')
        # figure parameters
    parser.add_argument('--ncols', default=4, help='number of columns')
    parser.add_argument('--no_datetime', action='store_true',
                        help='create a datetime folder')
    parser.add_argument('--overwrite', action='store_true', 
                        help='allow overwriting')

    args = parser.parse_args()


    # where I store extra fonts
    args.fontdir = os.path.join('..', 'tools', 'fonts')

    # default data folder
    if args.datadir is None:
        args.datadir = os.path.join('..', 'data', 'AIBS')

    # path to mouse dataframe
    mouse_df = 'mouse_df.csv'



    # this function just reformats the arguments, e.g. replacing 'both' by the 
    # actual parameter values, etc.
    args = reformat_args(args)



    # get numbers of sessions to analyse
    if args.sess_n == 'all':
        # this function looks at the mouse dataframe to find all the session
        # number (e.g., 1, 2, 3, 4...) and only retain the ones for which
        # there are sessions that fit the criteria
        all_sess_ns = sess_gen_util.get_sess_vals(mouse_df, 'sess_n', 
                        runtype=args.runtype, layer=args.layer, 
                        min_rois=args.min_rois, pass_fail=args.pass_fail, 
                        incl=args.incl, omit_sess=args.omit_sess, 
                        omit_mice=args.omit_mice)
    else:
        # if there's just one session number, puts it into a list
        all_sess_ns = gen_util.list_if_not(args.sess_n)


    # goes through each session number
    for sess_n in all_sess_ns:
        # puts the analysis parameters into named tuples (fixed dictionaries) 
        # and INITIALIZES THE SESSION OBJECTS!! -> CHECK THIS OUT
        analys_pars = prep_analyses(sess_n, args, mouse_df)
        
        
        run_analyses(*analys_pars, analyses=args.analyses)

            