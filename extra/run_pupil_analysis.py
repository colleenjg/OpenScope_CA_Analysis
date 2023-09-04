#!/usr/bin/env python

"""
run_pupil_analysis.py

This script runs pupil analyses using a Session object with data generated 
by the Allen Institute OpenScope experiments for the Credit Assignment Project.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import argparse
import copy
import inspect
from pathlib import Path

# try to set cache/config as early as possible (for clusters)
from util import gen_util 
gen_util.CC_config_cache()

gen_util.extend_sys_path(__file__, parents=2)
from util import gen_util, logger_util, plot_util, rand_util
from sess_util import sess_gen_util, sess_ntuple_util, sess_plot_util
from extra_analysis import pup_analys
from extra_plot_fcts import plot_from_dicts_tool as plot_dicts


DEFAULT_DATADIR = Path("..", "data", "OSCA")
DEFAULT_MOUSE_DF_PATH = Path("mouse_df.csv")

DEFAULT_FONTDIR = Path("..", "tools", "fonts")
if not DEFAULT_FONTDIR.is_dir():
    DEFAULT_FONTDIR = Path("..", "..", "tools", "fonts")


ANALYSIS_DESCR = {
    "c": "correlation between pupil and roi or run unexpected-locked changes",
    "r": "unexpected-locked change correlation per ROI between stimuli",
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
        - Sets seed, though doesn't seed
        - Modifies analyses (if "all" or "all_" in parameter)
        
    Adds the following args:
        - dend (str)     : type of dendrites to use ("allen", "extr")
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
    
    Returns:
        - args (Argument parser): input parser, with the following attributes
                                  modified: 
                                      visflow_dir, visflow_size, gabfr, gabk, 
                                      gab_ori, grps, analyses, seed
                                  and the following attributes added:
                                      omit_sess, omit_mice, dend
    """
    args = copy.deepcopy(args)

    [args.visflow_dir, args.visflow_size, args.gabfr, 
        args.gabk, args.gab_ori] = sess_gen_util.get_params(
            args.stimtype, args.visflow_dir, args.visflow_size, args.gabfr, 
            args.gabk, args.gab_ori)

    if args.datatype == "run":
        args.fluor = "n/a"
    if args.plane == "soma":
        args.dend = "allen"

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

    Returns args:
        - in the following nametuples: analyspar, sesspar, stimpar, autocorr, 
                                       permpar, quantpar
        - in the following dictionary: figpar 

    Required args:
        - args (Argument parser): parser with the following attributes:

            visflow_dir (str or list): visual flow direction values to include
                                     ("right", "left", ["right", "left"])
            visflow_size (int or list) : visual flow size values to include
                                     (128, 256 or [128, 256])
            closest (bool)         : if False, only exact session number is 
                                     retained, otherwise the closest.
            dend (str)             : type of dendrites to use ("allen" or "dend")
            error (str)            : error statistic parameter ("std" or "sem")
            fontdir (str)          : path to directory containing additional 
                                     fonts
            fluor (str)            : if "raw", raw ROI traces are used. If 
                                     "dff", dF/F ROI traces are used.
            gabfr (int)            : gabor frame at which sequences start 
                                     (0, 1, 2, 3)
            gabk (int or list)     : gabor kappa values to include 
                                     (4, 16 or [4, 16])
            gab_ori (int or list)  : gabor orientation values to include
                                     ([0, 45, 90, 135, 180, 225])
            incl (str)             : 
            keepnans (str)         : if True, ROIs with NaN/Inf values are 
                                     kept in the analyses and the original 
                                     running array is used instead of the one 
                                     where NaNs are interpolated..
            lag_s (num)            : lag for autocorrelation (in sec)
            line (str)             : line ("L23", "L5", "any")
            min_rois (int)         : min number of ROIs
            n_perms (int)          : nbr of permutations to run
            n_quants (int)         : number of quantiles
            ncols (int)            : number of columns
            no_datetime (bool)     : if True, figures are not saved in a 
                                     subfolder named based on the date and time.
            not_save_fig (bool)    : if True, figures are not saved
            output (str)           : general directory in which to save output
            overwrite (bool)       : if False, overwriting existing figures 
                                     is prevented by adding suffix numbers.
            pass_fail (str or list): pass/fail values of interest ("P", "F")
            plt_bkend (str)        : mpl backend to use
            plane (str)            : plane ("soma", "dend", "any")
            post (num)             : range of frames to include after each 
                                     reference frame (in s)
            pre (num)              : range of frames to include before each 
                                     reference frame (in s)
            runtype (str or list)  : runtype ("pilot" or "prod")
            scale (bool)           : whether to scale data (pupil, running, ROI)
            sess_n (int)           : session number
            stats (str)            : statistic parameter ("mean" or "median")
            stimtype (str)         : stimulus to analyse ("visflow" or "gabors")
            tails (str or int)     : which tail(s) to test ("hi", "lo", 2)

    Returns:
        - analysis_dict (dict): dictionary of analysis parameters
            ["analyspar"] (AnalysPar)    : named tuple of analysis parameters
            ["sesspar"] (SessPar)        : named tuple of session parameters
            ["stimpar"] (StimPar)        : named tuple of stimulus parameters
            ["autocorrpar"] (AutocorrPar): named tuple of autocorrelation 
                                           parameters
            ["permpar"] (PermPar)        : named tuple of permutation parameters
            ["quantpar"] (QuantPar)      : named tuple of quantile parameters
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
        args.stimtype, args.visflow_dir, args.visflow_size, args.gabfr, 
        args.gabk, args.gab_ori, args.pre, args.post)

    # SPECIFIC ANALYSES    
    # autocorrelation parameters
    analysis_dict["autocorrpar"] = sess_ntuple_util.init_autocorrpar(
        args.lag_s, byitem=False)
    
    # permutation parameters
    analysis_dict["permpar"] = sess_ntuple_util.init_permpar(
        args.n_perms, 0.05, args.tails)
    
    # quantile parameters
    analysis_dict["quantpar"] = sess_ntuple_util.init_quantpar(
        args.n_quants, [0, -1])

    # figure parameters
    analysis_dict["figpar"] = sess_plot_util.init_figpar(
        ncols=int(args.ncols), datetime=not(args.no_datetime), 
        overwrite=args.overwrite, save_fig=not(args.not_save_fig), 
        runtype=args.runtype, output=args.output, plt_bkend=args.plt_bkend, 
        fontdir=args.fontdir)

    return analysis_dict


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
    
    args.sess_n = sess_n

    analysis_dict = init_param_cont(args)
    analyspar, sesspar, stimpar = [analysis_dict[key] for key in 
        ["analyspar", "sesspar", "stimpar"]]
    
    roi = (args.datatype == "roi")
    run = (args.datatype == "run")

    # get session IDs and create Sessions
    sessids = sess_gen_util.sess_per_mouse(
        mouse_df, omit_sess=args.omit_sess, omit_mice=args.omit_mice, 
        **sesspar._asdict())

    if sesspar.runtype == "pilot":
        raise RuntimeError(
            "No sessions meet the criteria, as pupil data is not available "
            "for pilot data."
            )
    
    logger.info(
        f"Loading {len(sessids)} session(s)...", extra={"spacing": "\n"}
        )
        
    args_dict = {
        "datadir"   : args.datadir,
        "mouse_df"  : mouse_df,
        "runtype"   : sesspar.runtype,
        "full_table": False,
        "fluor"     : analyspar.fluor,
        "dend"      : analyspar.dend,
        "roi"       : roi,
        "run"       : run,
        "pupil"     : True,
        "omit"      : roi,
        "temp_log"  : "warning",
    }

    sessions = gen_util.parallel_wrap(
        sess_gen_util.init_sessions, sessids, args_dict=args_dict, 
        parallel=args.parallel, use_tqdm=True
        )

    # flatten list of sessions
    sessions = [sess for singles in sessions for sess in singles]

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
                           records the corresponding function and list of
                           acceptable 'datatype' values
    """

    fct_dict = dict()

    # 0. Plots the correlation between pupil and roi/run unexpected-locked 
    # changes for each session
    fct_dict["c"] = [pup_analys.run_pupil_diff_corr, ["roi", "run"]]

    # 1. Calculates Calculates unexpected-locked chnge correlation per ROI 
    # between stimuli
    fct_dict["r"] = [pup_analys.run_pup_roi_stim_corr, ["roi"]]

    return fct_dict


#############################################
def run_analyses(sessions, analysis_dict, analyses, datatype="roi", seed=None, 
                 parallel=False):
    """
    run_analyses(sessions, analysis_dict, analyses)

    Runs requested analyses on sessions using the parameters passed.

    Required args:
        - sessions (list)     : list of sessions, possibly nested
        - analysis_dict (dict): analysis parameter dictionary 
                                (see init_param_cont())
        - analyses (str)      : analyses to run
    
    Optional args:
        - datatype (str) : datatype ("run", "roi")
                           default: "roi"
        - seed (int)     : seed to use
                           default: None
        - parallel (bool): if True, some analyses are parallelized 
                           across CPU cores 
                           default: False
    """

    if len(sessions) == 0:
        logger.warning("No sessions meet these criteria.")
        return

    # changes backend and defaults
    plot_util.manage_mpl(cmap=False, **analysis_dict["figpar"]["mng"])

    fct_dict = get_analysis_fcts()

    args_dict = copy.deepcopy(analysis_dict)
    for key, item in zip(["seed", "parallel", "datatype"], 
        [seed, parallel, datatype]):
        args_dict[key] = item

    # run through analyses
    for analysis in analyses:
        if analysis not in fct_dict.keys():
            raise ValueError(f"{analysis} analysis not found.")
        fct, datatype_req = fct_dict[analysis]
        if datatype not in datatype_req:
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
        source = "modif" if args.modif else "pup"
        plot_dicts.plot_from_dicts(
            Path(args.dict_path), source=source, plt_bkend=args.plt_bkend, 
            fontdir=args.fontdir, parallel=args.parallel, 
            datetime=not(args.no_datetime), overwrite=args.overwrite)

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
            run_seq = "r" # should be run parallel within analysis
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
                "datatype": args.datatype,
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
        help=("analyses to run, e.g. 'cr', 'all' or 'all_c' (all, save 'c'). "
            f"ANALYSES: {ANALYSIS_STR}"))
    parser.add_argument("--datatype", default="roi", 
        help="datatype to use (roi or run)")          
    parser.add_argument("--sess_n", default="all",
        help="session to aim for, e.g. 1, 2, last, all")
    parser.add_argument("--dict_path", default=None, 
        help=("path to info dictionary from which to plot data."))

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
    parser.add_argument("--pre", default=1.5, type=float, 
        help="sec before reference frames")
    parser.add_argument("--post", default=1.5, type=float, 
        help="sec after reference frames")
    parser.add_argument("--stimtype", default="gabors", 
        help="stimulus to analyse")   
    
    # generally fixed 
        # analysis parameters
    parser.add_argument("--keepnans", action="store_true", 
        help=("use running array in which NaN values have not been "
            "interpolated and include ROIs with NaN/Inf values."))
    parser.add_argument("--fluor", default="dff", help="raw or dff")
    parser.add_argument("--stats", default="mean", help="plot mean or median")
    parser.add_argument("--error", default="sem", 
        help="sem for SEM/MAD, std for std/qu")
    parser.add_argument("--scale", action="store_true", 
        help="whether to scale data (pupil, ROIs, running data)") 
    parser.add_argument("--dend", default="extr", help="allen, extr")
        # session parameters
    parser.add_argument("--line", default="any", help="L23, L5")
    parser.add_argument("--closest", action="store_true", 
        help=("if True, the closest session number is used. Otherwise, "
            "only exact."))
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
        # figure parameters
    parser.add_argument("--ncols", default=4, help="number of columns")
    parser.add_argument("--no_datetime", action="store_true",
        help="create a datetime folder")
    parser.add_argument("--overwrite", action="store_true", 
        help="allow overwriting")
    parser.add_argument("--not_save_fig", action="store_true", 
        help="don't save figures")
        # plot using modif_analys_plots (if plotting from dictionary)
    parser.add_argument("--modif", action="store_true", 
        help=("plot from dictionary using modified plot functions"))
    
    args = parser.parse_args()

    return args


#############################################
if __name__ == "__main__":

    args = parse_args()

    logger_util.format_all(level=args.log_level)

    main(args)

