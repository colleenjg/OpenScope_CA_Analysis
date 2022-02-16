#!/usr/bin/env python

"""
run_roi_analysis.py

This script runs ROI trace analyses using a Session object with data generated 
by the Allen Institute OpenScope experiments for the Credit Assignment Project.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import argparse
import copy
import inspect
from pathlib import Path

import numpy as np

# try to set cache/config as early as possible (for clusters)
from util import gen_util 
gen_util.CC_config_cache()

gen_util.extend_sys_path(__file__, parents=2)
from util import gen_util, logger_util, plot_util, rand_util
from sess_util import sess_gen_util, sess_ntuple_util, sess_plot_util
from extra_analysis import roi_analys, gen_analys
from extra_plot_fcts import plot_from_dicts_tool as plot_dicts


DEFAULT_DATADIR = Path("..", "data", "OSCA")
DEFAULT_MOUSE_DF_PATH = Path("mouse_df.csv")
DEFAULT_FONTDIR = Path("..", "tools", "fonts")


ANALYSIS_DESCR = {
    "f": "full ROI traces",
    "t": "ROI traces by session quantile, split by unexpected and expected",
    "l": "ROI traces by session quantile, locked to unexpected or expected onset",
    "m": "magnitude of ROI differences between unexpected and expected",
    "a": "ROI autocorrelation",
    "g": "ROIs grouped by change in unexpected sensitivity in session",
    "o": "orientation or direction-locked response colormaps",
    "c": "ROI tuning curves for Gabor orientations",
    "p": "ROI responses by Gabor locations and mean orientation ",
    "v": "Trial PCA trajectories (UNDER DEVELOPMENT)",
    "r": "Correlations between responses in different sessions",
}


logger = logger_util.get_module_logger(name=__name__)


#############################################
def reformat_args(args):
    """
    reformat_args(args)

    Returns reformatted args for analyses, specifically 
        - Sets stimulus parameters to "none" if they are irrelevant to the 
          stimtype
        - Changes stimulus parameters from "both" to actual values
        - Changes grps string values to a list
        - Sets seed, though doesn't seed
        - Modifies analyses (if "all" or "all_" in parameter)

    Adds the following args:
        - dend (str)     : type of dendrites to use ("allen" or "extr")
        - omit_sess (str): sess to omit
        - omit_mice (str): mice to omit

    Required args:
        - args (Argument parser): parser with the following attributes: 
            visflow_dir (str)    : visual flow direction values to include
                                   (e.g., "right", "left" or "both")
            visflow_size (int or str): visual flow size values to include
                                   (e.g., 128, 256, "both")
            gabfr (int)          : gabor frame value to start sequences at
                                   (e.g., 0, 1, 2, 3)
            gabk (int or str)    : gabor kappa values to include 
                                   (e.g., 4, 16 or "both")
            gab_ori (int or str) : gabor orientation values to include
                                   (e.g., 0, 45, 90, 135, 180, 225 or "all")
            runtype (str)        : runtype ("pilot" or "prod")
            stimtype (str)       : stimulus to analyse (visflow or gabors)
            grps (str)           : set or sets of groups to plot, 
                                   (e.g., "all change no_change reduc incr").
    
    Returns:
        - args (Argument parser): input parser, with the following attributes
                                  modified: 
                                      visflow_dir, visflow_size, gabfr, gabk, 
                                      gab_ori, grps, analyses, seed
                                  and the following attributes added:
                                      omit_sess, omit_mice, dend
    """

    args = copy.deepcopy(args)

    if args.plane == "soma": args.dend = "allen"

    [args.visflow_dir, args.visflow_size, args.gabfr, 
    args.gabk, args.gab_ori] = sess_gen_util.get_params(
        args.stimtype, args.visflow_dir, args.visflow_size, args.gabfr, 
        args.gabk, args.gab_ori)

    args.grps = gen_util.str_to_list(args.grps)

    args.omit_sess, args.omit_mice = sess_gen_util.all_omit(
        args.stimtype, args.runtype, args.visflow_dir, args.visflow_size, 
        args.gabk)

    # choose a seed if none is provided (i.e., args.seed=-1), but seed later
    args.seed = rand_util.seed_all(
        args.seed, "cpu", log_seed=False, seed_now=False)

    # collect analysis letters
    all_analyses = "".join(get_analysis_fcts().keys())
    if "all" in args.analyses:
        if "_" in args.analyses:
            excl = args.analyses.split("_")[1]
            args.analyses, _ = gen_util.remove_lett(all_analyses, excl)
        else:
            args.analyses = all_analyses
    elif "_" in args.analyses:
        raise ValueError("Use '_' in args.analyses only with 'all'.")

    return args


#############################################
def init_param_cont(args):
    """
    init_param_cont(args)

    Initializes parameter containers.

    Returns args:
        - in the following nametuples: analyspar, sesspar, stimpar, autocorr, 
                                       permpar, quantpar, roigrppar, tcurvpar
        - in the following dictionary: figpar 

    Required args:
        - args (Argument parser): parser with the following attributes:

            visflow_dir (str or list): visual flow direction values to include
                                     ("right", "left", ["right", "left"])
            visflow_size (int or list): visual flow size values to include
                                     (128, 256 or [128, 256])
            closest (bool)         : if False, only exact session number is 
                                     retained, otherwise the closest.
            dend (str)             : type of dendrites to use ("allen" or "dend")
            error (str)            : error statistic parameter ("std" or "sem")
            fluor (str)            : if "raw", raw ROI traces are used. If 
                                     "dff", dF/F ROI traces are used.
            fontdir (str)          : path to directory containing additional 
                                     fonts
            gabfr (int)            : gabor frame at which sequences start 
                                     (0, 1, 2, 3)
            gabk (int or list)     : gabor kappa values to include 
                                     (4, 16 or [4, 16])
            gab_ori (int or list)  : gabor orientation values to include
                                     ([0, 45, 90, 135, 180, 225])
            grps (str or list)     : set or sets of groups to return, 
                                     ("all", "change", "no_change", "reduc", 
                                     "incr".)
            incl (str)             : sessions to include ("yes", "no", "all") 
            keepnans (str)         : if True, ROIs with NaN/Inf values are 
                                     kept in the analyses.
            lag_s (num)            : lag for autocorrelation (in sec)
            line (str)             : "L23", "L5", "any"
            min_rois (int)         : min number of ROIs
            n_perms (int)          : nbr of permutations to run
            n_quants (int)         : number of quantiles
            ncols (int)            : number of columns
            no_add_exp (bool)      : if True, the group of ROIs showing no 
                                     significance in either is not added to   
                                     the groups returned
            no_datetime (bool)     : if True, figures are not saved in a 
                                     subfolder named based on the date and time.
            not_byitem (bool)      : if True, autocorrelation statistics are
                                     taken across items (e.g., ROIs)
            not_save_fig (bool)    : if True, figures are not saved
            op (str)               : operation on values, if plotvals if "both" 
                                     ("ratio" or "diff") 
            output (str)           : general directory in which to save output
            overwrite (bool)       : if False, overwriting existing figures 
                                     is prevented by adding suffix numbers.
            pass_fail (str or list): pass/fail values of interest ("P", "F")
            plot_vals (str)        : values to plot ("unexp", "exp", "both")
            plane (str)            : plane ("soma", "dend", "any")
            plt_bkend (str)        : mpl backend to use
            post (num)             : range of frames to include after each 
                                     reference frame (in s)
            pre (num)              : range of frames to include before each 
                                     reference frame (in s)
            runtype (str or list)  : runtype ("pilot" or "prod")
            scale (bool)           : whether to scale ROI data
            sess_n (int)           : session number
            stats (str)            : statistic parameter ("mean" or "median")
            stimtype (str)         : stimulus to analyse ("visflow" or "gabors")
            tails (str or int)     : which tail(s) to test ("hi", "lo", 2)
            tc_gabfr (int or str)  : gabor frame at which sequences start 
                                     (0, 1, 2, 3) for tuning curve analysis
                                     (x_x, interpreted as 2 gabfrs)
            tc_grp2 (str)          : second group: either unexp, exp or rand 
                                     (random subsample of exp, the size of 
                                     unexp)
            tc_post (num)          : range of frames to include after each 
                                     reference frame (in s) for tuning curve 
                                     analysis
            tc_vm_estim (bool)     : runs analysis using von Mises parameter 
                                     estimation method
            tc_test (bool)         : if True, tuning curve analysis is run on a 
                                     small subset of ROIs and gabors

    Returns:
        - analysis_dict (dict): dictionary of analysis parameters
            ["analyspar"] (AnalysPar)    : named tuple of analysis parameters
            ["sesspar"] (SessPar)        : named tuple of session parameters
            ["stimpar"] (StimPar)        : named tuple of stimulus parameters
            ["autocorrpar"] (AutocorrPar): named tuple of autocorrelation 
                                           parameters
            ["permpar"] (PermPar)        : named tuple of permutation parameters
            ["quantpar"] (QuantPar)      : named tuple of quantile parameters
            ["roigrppar"] (RoiGrpPar)    : named tuple of roi grp parameters
            ["tcurvpar"] (TCurvPar)      : named tuple of tuning curve 
                                           parameters
            ["figpar"] (dict)            : dictionary containing following 
                                           subdictionaries:
                ["init"]: dict with following inputs as attributes:
                    ["ncols"] (int)      : number of columns in the figures
                    ["sharex"] (bool)    : if True, x axis lims are shared 
                                           across subplots
                    ["sharey"] (bool)    : if True, y axis lims are shared 
                                           across subplots
                    ["subplot_hei"] (num): height of each subplot (inches)
                    ["subplot_wid"] (num): width of each subplot (inches)

                ["save"]: dict with the following inputs as attributes:
                    ["datetime"] (bool) : if True, figures are saved in a  
                                          subfolder named based on the date and 
                                          time.
                    ["fig_ext"] (str)   : figure extension
                    ["overwrite"] (bool): if True, existing figures can be 
                                          overwritten
                    ["save_fig"] (bool) : if True, figures are saved
                    ["use_dt"] (str)    : datetime folder to use

                ["dirs"]: dict with the following attributes:
                    ["figdir"] (str)   : main folder in which to save figures
                    ["roi"] (str)      : subdirectory name for ROI analyses
                    ["run"] (str)      : subdirectory name for running analyses
                    ["autocorr"] (str) : subdirectory name for autocorrelation 
                                         analyses
                    ["locori"] (str)   : subdirectory name for location and 
                                         orientation responses
                    ["oridir"] (str)   : subdirectory name for 
                                         orientation/direction analyses
                    ["unexp_qu"] (str)  : subdirectory name for unexpected, 
                                         quantile analyses
                    ["tune_curv"] (str): subdirectory name for tuning curves
                    ["grped"] (str)    : subdirectory name for ROI grps data
                    ["mags"] (str)     : subdirectory name for magnitude 
                                         analyses
                
                ["mng"]: dict with the following attributes:
                    ["plt_bkend"] (str): mpl backend to use
                    ["linclab"] (bool) : if True, Linclab mpl defaults are used
                    ["fontdir"] (str)  : path to directory containing 
                                         additional fonts
    """

    args = copy.deepcopy(args)

    analysis_dict = dict()

    # analysis parameters
    analysis_dict["analyspar"] = sess_ntuple_util.init_analyspar(
        args.fluor, not(args.keepnans), args.stats, args.error, args.scale, 
        dend=args.dend)

    # session parameters
    analysis_dict["sesspar"] = sess_ntuple_util.init_sesspar(
        args.sess_n, args.closest, args.plane, args.line, args.min_rois, 
        args.pass_fail, args.incl, args.runtype)
    
    # stimulus parameters
    analysis_dict["stimpar"] = sess_ntuple_util.init_stimpar(
        args.stimtype, args.visflow_dir, args.visflow_size, args.gabfr, args.gabk, 
        args.gab_ori, args.pre, args.post)

    # SPECIFIC ANALYSES    
    # autocorrelation parameters
    analysis_dict["autocorrpar"] = sess_ntuple_util.init_autocorrpar(
        args.lag_s, not(args.not_byitem))
    
    # permutation parameters
    analysis_dict["permpar"] = sess_ntuple_util.init_permpar(
        args.n_perms, 0.05, args.tails)
    
    # quantile parameters
    analysis_dict["quantpar"] = sess_ntuple_util.init_quantpar(
        args.n_quants, [0, -1])

    # roi grp parameters
    analysis_dict["roigrppar"] = sess_ntuple_util.init_roigrppar(
        args.grps, not(args.no_add_exp), args.op, args.plot_vals)

    # tuning curve parameters
    analysis_dict["tcurvpar"] = sess_ntuple_util.init_tcurvpar(
        args.tc_gabfr, 0, args.tc_post, args.tc_grp2, args.tc_test, 
        args.tc_vm_estim)

    # figure parameters
    analysis_dict["figpar"] = sess_plot_util.init_figpar(
        ncols=int(args.ncols), datetime=not(args.no_datetime), 
        overwrite=args.overwrite, save_fig=not(args.not_save_fig), 
        runtype=args.runtype, output=args.output, plt_bkend=args.plt_bkend, 
        fontdir=args.fontdir)

    return analysis_dict


#############################################
def init_sessions(sessids, datadir, mouse_df, analyspar, runtype="prod", 
                  full_table=False, parallel=False):
    """
    init_sessions(sessids, datadir, mouse_df, analyspar)

    Initializes sessions.

    Required args:
        - sessids (list)       : IDs of sessions to load
        - datadir (Path)       : data directory
        - mouse_df (pandas df) : path name of dataframe containing information 
                                 on each session
        - analyspar (AnalysPar): named tuple containing analysis parameters

    Optional args:
        - runtype (str)    : runtype ("pilot" or "prod")
        - full_table (bool): if True, full stimulus dataframe is loaded
                             default: False
        - parallel (bool)  : if True, some analyses are parallelized 
                             across CPU cores 
                             default: False

    Returns:
        - sessions (list): list of sessions
    """

    args_dict = {
        "datadir"   : datadir,
        "mouse_df"  : mouse_df,
        "runtype"   : runtype,
        "full_table": full_table,
        "fluor"     : analyspar.fluor,
        "dend"      : analyspar.dend,
        "omit"      : True,
        "temp_log"  : "warning",
    }

    sessions = gen_util.parallel_wrap(
        sess_gen_util.init_sessions, sessids, args_dict=args_dict, 
        parallel=parallel, use_tqdm=True
        )

    # flatten list of sessions
    sessions = [sess for singles in sessions for sess in singles]

    return sessions


#############################################
def prep_analyses(sess_n, args, mouse_df):
    """
    prep_analyses(sess_n, args, mouse_df)

    Prepares named tuples and sessions for which to run analyses, based on the 
    arguments passed.

    Required args:
        - sess_n (int)          : session number to run analyses on, or 
                                  session numbers to compare, e.g. "1v2"
        - args (Argument parser): parser containing all parameters
        - mouse_df (pandas df)  : path name of dataframe containing information 
                                  on each session

    Returns:
        - sessions (list)     : list of sessions, or nested list per mouse 
                                if sess_n is a combination to compare
        - analysis_dict (dict): dictionary of analysis parameters 
                                (see init_param_cont())
    """

    args = copy.deepcopy(args)

    comp = True
    if "v" not in str(sess_n):
        comp = False
        if sess_n not in ["first", "last"]:
            sess_n = int(sess_n)
        
    args.sess_n = sess_n

    analysis_dict = init_param_cont(args)
    analyspar, sesspar, stimpar = [analysis_dict[key] for key in 
        ["analyspar", "sesspar", "stimpar"]]
    
    # get session IDs and create Sessions
    if comp:
        sessids = sess_gen_util.sess_comp_per_mouse(mouse_df, 
            omit_sess=args.omit_sess, omit_mice=args.omit_mice, 
            **sesspar._asdict())
        n_sess = np.sum([len(ids) for ids in sessids])
        logger.info(f"Loading {n_sess} session(s)...", extra={"spacing": "\n"})
        sessions = []
        for ids in sessids:
            subs = init_sessions(
                ids, args.datadir, mouse_df, analyspar, 
                runtype=sesspar.runtype, full_table=args.tc_vm_estim,
                parallel=args.parallel
                )
            if len(subs) == 2:
                sessions.append(subs)
            else:
                logger.warning(
                    f"Omitting session {subs[0].sessid} due to incomplete "
                    "pair.")
    else:
        sessids = sess_gen_util.sess_per_mouse(mouse_df, 
            omit_sess=args.omit_sess, omit_mice=args.omit_mice, 
            **sesspar._asdict())
        logger.info(
            f"Loading {len(sessids)} session(s)...", extra={"spacing": "\n"}
            )
        sessions = init_sessions(
            sessids, args.datadir, mouse_df, analyspar, 
            runtype=sesspar.runtype, full_table=args.tc_vm_estim,
            parallel=args.parallel
            )

    if len(sessids) == 0:
        raise RuntimeError("No sessions meet the criteria.")

    runtype_str = ""
    if sesspar.runtype != "prod":
        runtype_str = f" ({sesspar.runtype} data)"

    stim_str = stimpar.stimtype
    if stimpar.stimtype == "gabors":
        stim_str = "gabor"
    elif stimpar.stimtype == "visflow":
        stim_str = "visual flow"

    logger.info(
        f"Analysis of {sesspar.plane} responses to {stim_str} "
        f"stimuli{runtype_str}.\nSession {sesspar.sess_n}", 
        extra={"spacing": "\n"})

    return sessions, analysis_dict


#############################################
def get_analysis_fcts():
    """
    get_analysis_fcts()

    Returns dictionary of analysis functions.

    Returns:
        - fct_dict (dict): dictionary where each key is an analysis letter, and
                           records the corresponding function and 'comp' 
                           boolean value 
    """

    fct_dict = dict()

    # 0. Plots the full traces for each session
    fct_dict["f"] = [gen_analys.run_full_traces, False]

    # 1. Analyses and plots average traces by quantile x unexpected for each 
    # session
    fct_dict["t"] = [gen_analys.run_traces_by_qu_unexp_sess, False]

    # 2. Analyses and plots average traces locked to unexpected by quantile x 
    # unexpected for each session 
    fct_dict["l"] = [gen_analys.run_traces_by_qu_lock_sess, False]

    # 3. Analyses and plots magnitude of change in dF/F area from first to last 
    # quantile of unexpected vs expected sequences, for each session
    fct_dict["m"] = [gen_analys.run_mag_change, False]

    # 4. Analyses and plots autocorrelation
    fct_dict["a"] = [gen_analys.run_autocorr, False]

    # 5. Analyses and plots: a) trace areas by quantile, b) average traces, 
    # c) trace areas by suprise for first vs last quantile, for each ROI group, 
    # for each session
    fct_dict["g"] = [roi_analys.run_rois_by_grp, False]

    # 6. Analyses and plots colormaps by orientation or direction, as well as 
    # average traces
    fct_dict["o"] = [roi_analys.run_oridirs, False]

    # 7. Analyses and plots ROI tuning curves for gabor orientation
    fct_dict["c"] = [roi_analys.run_tune_curves, False]

    # 8. Analyses and plots ROI responses for Gabor locations and mean 
    # orientations
    fct_dict["p"] = [roi_analys.run_loc_ori_resp, False]

    # 9. Plots trials as trajectories in 2 principal components
    #### UNDER DEVELOPMENT ####
    fct_dict["v"] = [roi_analys.run_trial_pc_traj, False]

    # 10. Analyses and logs correlations between sessions to console
    fct_dict["r"] = [gen_analys.run_trace_corr_acr_sess, True]

    return fct_dict


#############################################
def run_analyses(sessions, analysis_dict, analyses, seed=None,
                 parallel=False, plot_tc=True):
    """
    run_analyses(sessions, analysis_dict, analyses)

    Runs requested analyses on sessions using the parameters passed.

    Required args:
        - sessions (list)     : list of sessions, possibly nested
        - analysis_dict (dict): analysis parameter dictionary 
                                (see init_param_cont())
        - analyses (str)      : analyses to run

    Optional args:
        - seed (int)     : seed to use
                           default: None
        - parallel (bool): if True, some analyses are parallelized 
                           across CPU cores 
                           default: False
        - plot_tc (bool) : if True, tuning curves are plotted for each ROI  
                           default: True
    """

    if len(sessions) == 0:
        logger.warning("No sessions meet these criteria.")
        return

    comp = True if isinstance(sessions[0], list) else False

    # changes backend and defaults
    plot_util.manage_mpl(cmap=False, **analysis_dict["figpar"]["mng"])

    fct_dict = get_analysis_fcts()

    args_dict = copy.deepcopy(analysis_dict)
    for key, item in zip(["seed", "parallel", "plot_tc"], 
        [seed, parallel, plot_tc]):
        args_dict[key] = item

    # run through analyses
    for analysis in analyses:
        if analysis not in fct_dict.keys():
            raise ValueError(f"{analysis} analysis not found.")
        fct, comp_req = fct_dict[analysis]
        if comp_req != comp:
            continue
        args_dict_use = gen_util.keep_dict_keys(
            args_dict, inspect.getfullargspec(fct).args)
        fct(sessions=sessions, analysis=analysis, **args_dict_use)

        plot_util.cond_close_figs()


#############################################
def main(args):
    """
    main(args)

    Runs analyses with parser arguments.

    Required args:
        - args (dict): parser argument dictionary
    """

    # set logger to the specified level
    logger_util.set_level(level=args.log_level)
    
    args.fontdir = DEFAULT_FONTDIR if DEFAULT_FONTDIR.is_dir() else None

    if args.dict_path is not None:
        source = "modif" if args.modif else "roi"
        plot_dicts.plot_from_dicts(
            Path(args.dict_path), source=source, plt_bkend=args.plt_bkend, 
            fontdir=args.fontdir, plot_tc=not(args.no_plot_tc), 
            parallel=args.parallel, datetime=not(args.no_datetime), 
            overwrite=args.overwrite)

    else:
        args = reformat_args(args)
        if args.datadir is None: 
            args.datadir = DEFAULT_DATADIR
        else:
            args.datadir = Path(args.datadir)
        mouse_df = DEFAULT_MOUSE_DF_PATH

        # get numbers of sessions to analyse
        if args.sess_n == "all":
            all_sess_ns = sess_gen_util.get_sess_vals(
                mouse_df, "sess_n", runtype=args.runtype, plane=args.plane, 
                line=args.line, min_rois=args.min_rois, 
                pass_fail=args.pass_fail, incl=args.incl, 
                omit_sess=args.omit_sess, omit_mice=args.omit_mice, sort=True)
        else:
            all_sess_ns = gen_util.list_if_not(args.sess_n)

        # get analysis parameters for each session number
        all_analys_pars = gen_util.parallel_wrap(
            prep_analyses, all_sess_ns, args_list=[args, mouse_df], 
            parallel=args.parallel)

        # split parallel from sequential analyses
        args.parallel = bool(args.parallel  * (not args.debug))
        if args.parallel:
            run_seq = "oc" # should be run parallel within analysis
            all_analyses = gen_util.remove_lett(args.analyses, run_seq)
            sess_parallels = [True, False]
            analyses_parallels = [False, True]
        else:
            all_analyses = [args.analyses]
            sess_parallels, analyses_parallels = [False], [False]

        for analyses, sess_parallel, analyses_parallel in zip(
            all_analyses, sess_parallels, analyses_parallels):
            if len(analyses) == 0:
                continue
            args_dict = {
                "analyses": analyses,
                "seed"    : args.seed,
                "parallel": analyses_parallel,
                "plot_tc" : not(args.no_plot_tc),
                }

            # run analyses for each parameter set
            gen_util.parallel_wrap(run_analyses, all_analys_pars, 
                args_dict=args_dict, parallel=sess_parallel, mult_loop=True)


#############################################
def parse_args():
    """
    parse_args()

    Returns parser arguments.

    Returns:
        - args (dict): parser argument dictionary
    """

    parser = argparse.ArgumentParser()

    ANALYSIS_STR = " || ".join(
        [f"{key}: {item}" for key, item in ANALYSIS_DESCR.items()])

        # general parameters
    parser.add_argument("--datadir", default=None, 
        help="data directory (if not provided, uses a default directory)")
    parser.add_argument("--output", default=".", type=Path, 
        help="main directory in which to store output")
    parser.add_argument("--analyses", default="all", 
        help=("analyses to run, e.g. 'ftl', 'all' or 'all_f' (all, save 'f'). "
            f"ANALYSES: {ANALYSIS_STR}"))
    parser.add_argument("--sess_n", default="all",
        help="session to aim for, e.g. 1, 2, first, last, all")
    parser.add_argument("--dict_path", default=None, 
        help=("path to info dictionary or directory of dictionaries from "
            "which to plot data."))
    parser.add_argument("--no_plot_tc", action="store_true", 
        help="no tuning curve plots are generated")

        # technical parameters
    parser.add_argument("--plt_bkend", default=None, 
        help="switch mpl backend when running on server")
    parser.add_argument("--parallel", action="store_true", 
        help="do runs in parallel.")
    parser.add_argument("--debug", action="store_true", 
        help="only enable session loading in parallel")
    parser.add_argument("--seed", default=-1, type=int, 
        help="random seed (-1 for None)")
    parser.add_argument("--log_level", default="info", help="logging level")

        # session parameters
    parser.add_argument("--runtype", default="prod", help="prod or pilot")
    parser.add_argument("--plane", default="soma", help="soma, dend")
    parser.add_argument("--min_rois", default=5, type=int, 
        help="min rois criterion")
        # stimulus parameters
    parser.add_argument("--visflow_dir", default="both", 
        help="visual flow dir (right, left, or both)") 
    parser.add_argument("--gabfr", default=3, type=int, 
        help="gabor frame at which to start sequences")  
    parser.add_argument("--post", default=1.5, type=float, 
        help="sec after reference frames")
    parser.add_argument("--stimtype", default="gabors", 
        help="stimulus to analyse")   
        # roi group parameters
    parser.add_argument("--plot_vals", default="unexp", 
        help="plot both (with op applied), unexp or exp")
    
    # generally fixed 
        # analysis parameters
    parser.add_argument("--keepnans", action="store_true", 
        help="keep ROIs containing NaNs or Infs in session.")
    parser.add_argument("--fluor", default="dff", help="raw or dff")
    parser.add_argument("--stats", default="mean", help="plot mean or median")
    parser.add_argument("--error", default="sem", 
        help="sem for SEM/MAD, std for std/qu")    
    parser.add_argument("--scale", action="store_true", 
        help="whether to scale ROI data")    
    parser.add_argument("--dend", default="extr", help="allen, extr")
        # session parameters
    parser.add_argument("--line", default="any", help="L23, L5")
    parser.add_argument("--closest", action="store_true", 
        help=("if True, the closest session number is used. "
            "Otherwise, only exact."))
    parser.add_argument("--pass_fail", default="P", 
        help="P to take only passed sessions")
    parser.add_argument("--incl", default="any",
        help="include only 'yes', 'no' or 'any'")
        # stimulus parameters
    parser.add_argument("--visflow_size", default=128, 
        help="visual flow size (128, 256, or both)")
    parser.add_argument("--gabk", default=16,
        help="kappa value (4, 16, or both)")    
    parser.add_argument("--gab_ori", default="all",
        help="gabor orientation values (0, 45, 90, 135, 180, 225, all)")    
    parser.add_argument("--pre", default=0, type=float, help="sec before frame")
        # permutation parameters
    parser.add_argument("--n_perms", default=10000, type=int, 
        help="nbr of permutations")
    parser.add_argument("--tails", default="2", 
        help="tails for perm analysis (2, lo, up)")
        # quantile parameters
    parser.add_argument("--n_quants", default=4, type=int, 
        help="nbr of quantiles")
        # autocorrelation parameters
    parser.add_argument("--lag_s", default=4, type=float,
        help="lag for autocorrelation (in sec)")
    parser.add_argument("--not_byitem", action="store_true",
        help="if True, autocorrelation stats are taken across ROIs")
        # roi grp parameters
    parser.add_argument("--op", default="diff", 
        help="calculate diff or ratio of unexp to exp")
    parser.add_argument("--grps", default="reduc incr no_change", 
        help="plot all ROI grps or grps with change or no_change")
    parser.add_argument("--no_add_exp", action="store_true",
        help="do not add exp_exp to ROI grp plots")
        # tuning curve parameters
    parser.add_argument("--tc_gabfr", default=3, 
        help="gabor frame at which to start sequences (if x_x, interpreted as "
            "2 gabfrs)")  
    parser.add_argument("--tc_post", default=0.6, type=float, 
        help="sec after reference frames")
    parser.add_argument("--tc_grp2", default="unexp", 
        help=("second group: either unexp, exp or rand (random subsample of "
            "exp, the size of unexp)"))
    parser.add_argument("--tc_test", action="store_true",
        help="tests code on a small number of gabors and ROIs")
    parser.add_argument("--tc_vm_estim", action="store_true",
        help="runs analysis using von Mises parameter estimation method")
        # figure parameters
    parser.add_argument("--ncols", default=4, help="number of columns")
    parser.add_argument("--no_datetime", action="store_true",
        help="create a datetime folder")
    parser.add_argument("--overwrite", action="store_true", 
        help="allow overwriting")
    parser.add_argument("--not_save_fig", action="store_true", 
        help="don't save figures")
        # plot using modif_analys_plots (only if plotting from dictionary)
    parser.add_argument("--modif", action="store_true", 
        help="plot from dictionary using modified plot functions")

    args = parser.parse_args()

    return args


#############################################
if __name__ == "__main__":

    args = parse_args()

    logger_util.format_all(level=args.log_level)

    main(args)

