"""
run_run_analysis.py

This script runs running analyses using a Session object with data generated 
by the Allen Institute OpenScope experiments for the Credit Assignment Project.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import argparse
import copy
import inspect
import logging
from pathlib import Path

# try to set cache/config as early as possible (for clusters)
from util import gen_util 
gen_util.CC_config_cache()

gen_util.extend_sys_path(__file__, parents=2)
from util import gen_util, logger_util, plot_util
from sess_util import sess_gen_util, sess_ntuple_util, sess_plot_util
from extra_analysis import gen_analys
from extra_plot_fcts import plot_from_dicts_tool as plot_dicts


DEFAULT_DATADIR = Path("..", "data", "OSCA")
DEFAULT_MOUSE_DF_PATH = Path("mouse_df.csv")
DEFAULT_FONTDIR = Path("..", "tools", "fonts")

logger = logging.getLogger(__name__)


ANALYSIS_DESCR = {
    "f": "full running traces",
    "t": "running traces by session quintile, split by surprise and regular",
    "l": "running traces by session quintile, locked to surprise or regular onset",
    "m": "magnitude of running differences between surprise and regular",
    "a": "running autocorrelation",
}


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
        - omit_sess (str): sess to omit
        - omit_mice (str): mice to omit

    Required args:
        - args (Argument parser): parser with the following attributes: 
            bri_dir (str)        : brick direction values to include
                                   (e.g., "right", "left" or "both")
            bri_size (int or str): brick size values to include
                                   (e.g., 128, 256, "both")
            gabfr (int)          : gabor frame value to start sequences at
                                   (e.g., 0, 1, 2, 3)
            gabk (int or str)    : gabor kappa values to include 
                                   (e.g., 4, 16 or "both")
            gab_ori (int or str) : gabor orientation values to include
                                   (e.g., 0, 45, 90, 135 or "all")
            runtype (str)        : runtype ("pilot" or "prod")
            stimtype (str)       : stimulus to analyse (bricks or gabors)
    
    Returns:
        - args (Argument parser): input parser, with the following attributes
                                  modified: 
                                      bri_dir, bri_size, gabfr, gabk, gab_ori, 
                                      grps, analyses, seed
                                  and the following attributes added:
                                      omit_sess, omit_mice
    """
    args = copy.deepcopy(args)

    [args.bri_dir, args.bri_size, args.gabfr, 
        args.gabk, args.gab_ori] = sess_gen_util.get_params(
            args.stimtype, args.bri_dir, args.bri_size, args.gabfr, args.gabk, 
            args.gab_ori)

    args.omit_sess, args.omit_mice = sess_gen_util.all_omit(
        args.stimtype, args.runtype, args.bri_dir, args.bri_size, args.gabk)

    # chose a seed if none is provided (i.e., args.seed=-1), but seed later
    args.seed = gen_util.seed_all(
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
                                       permpar, quintpar
        - in the following dictionary: figpar 

    Required args:
        - args (Argument parser): parser with the following attributes:

            bri_dir (str or list)  : brick direction values to include
                                     ("right", "left", ["right", "left"])
            bri_size (int or list) : brick size values to include
                                     (128, 256 or [128, 256])
            closest (bool)         : if False, only exact session number is 
                                     retained, otherwise the closest.
            error (str)            : error statistic parameter ("std" or "sem")
            fontdir (str)          : path to directory containing additional 
                                     fonts
            gabfr (int)            : gabor frame at which sequences start 
                                     (0, 1, 2, 3)
            gabk (int or list)     : gabor kappa values to include 
                                     (4, 16 or [4, 16])
            gab_ori (int or list)  : gabor orientation values to include
                                     ([0, 45, 90, 135])
            incl (str)             : sessions to include ("yes", "no", "all")     
            keepnans (str)         : if True, the original running array is 
                                     used instead of the one where NaNs
                                     are interpolated.
            lag_s (num)            : lag for autocorrelation (in sec)
            line (str)             : line ("L23", "L5", "any")
            min_rois (int)         : min number of ROIs
            n_perms (int)          : nbr of permutations to run
            n_quints (int)         : number of quintiles
            ncols (int)            : number of columns
            no_datetime (bool)     : if True, figures are not saved in a 
                                     subfolder named based on the date and time.
            no_sharey (bool)       : if True, sharey figure parameter is set to 
                                     False.
            not_save_fig (bool)    : if True, figures are not saved
            output (str)           : general directory in which to save output
            overwrite (bool)       : if False, overwriting existing figures 
                                     is prevented by adding suffix numbers.
            pass_fail (str or list): pass/fail values of interest ("P", "F")
            plane (str)            : plane ("soma", "dend", "any")
            plt_bkend (str)        : mpl backend to use
            post (num)             : range of frames to include after each 
                                     reference frame (in s)
            pre (num)              : range of frames to include before each 
                                     reference frame (in s)
            runtype (str or list)  : runtype ("pilot" or "prod")
            scale (bool)           : whether to scale running data
            sess_n (int)           : session number
            stats (str)            : statistic parameter ("mean" or "median")
            stimtype (str)         : stimulus to analyse ("bricks" or "gabors")
            tails (str or int)     : which tail(s) to test ("hi", "lo", 2)

    Returns:
        - analysis_dict (dict): dictionary of analysis parameters
            ["analyspar"] (AnalysPar)    : named tuple of analysis parameters
            ["sesspar"] (SessPar)        : named tuple of session parameters
            ["stimpar"] (StimPar)        : named tuple of stimulus parameters
            ["autocorrpar"] (AutocorrPar): named tuple of autocorrelation 
                                           parameters
            ["permpar"] (PermPar)        : named tuple of permutation parameters
            ["quintpar"] (QuintPar)      : named tuple of quintile parameters
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
                    ["surp_qu"] (str)  : subdirectory name for surprise, 
                                         quintile analyses
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
        "n/a", not(args.keepnans), args.stats, args.error, args.scale)

    # session parameters
    analysis_dict["sesspar"] = sess_ntuple_util.init_sesspar(
        args.sess_n, args.closest, args.plane, args.line, args.min_rois, 
        args.pass_fail, args.incl, args.runtype)

    # stimulus parameters
    analysis_dict["stimpar"] = sess_ntuple_util.init_stimpar(
        args.stimtype, args.bri_dir, args.bri_size, args.gabfr, args.gabk, 
        args.gab_ori, args.pre, args.post)

    # SPECIFIC ANALYSES    
    # autocorrelation parameters
    analysis_dict["autocorrpar"] = sess_ntuple_util.init_autocorrpar(
        args.lag_s, byitem=False)
    
    # permutation parameters
    analysis_dict["permpar"] = sess_ntuple_util.init_permpar(
        args.n_perms, 0.05, args.tails, False)
    
    # quintile parameters
    analysis_dict["quintpar"] = sess_ntuple_util.init_quintpar(
        args.n_quints, [0, -1])

    # figure parameters
    analysis_dict["figpar"] = sess_plot_util.init_figpar(
        ncols=int(args.ncols), datetime=not(args.no_datetime), 
        overwrite=args.overwrite, save_fig=not(args.not_save_fig), 
        runtype=args.runtype, output=args.output, plt_bkend=args.plt_bkend, 
        fontdir=args.fontdir, sharey=not(args.no_sharey))

    return analysis_dict


#############################################
def prep_analyses(sess_n, args, mouse_df):
    """
    prep_analyses(sess_n, args, mouse_df)

    Prepares named tuples and sessions for which to run analyses, based on the 
    arguments passed.

    Required args:
        - sess_n (int)          : session number to run analyses on, or 
                                  combination of session numbers to compare, 
                                  e.g. "1v2"
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
    sesspar, stimpar = [analysis_dict[key] for key in ["sesspar", "stimpar"]]
    
    # get session IDs and create Sessions
    sessids = sess_gen_util.sess_per_mouse(
        mouse_df, omit_sess=args.omit_sess, omit_mice=args.omit_mice, 
        **sesspar._asdict())
    sessions = sess_gen_util.init_sessions(
        sessids, args.datadir, mouse_df, sesspar.runtype, roi=False, run=True, 
        fulldict=False)

    runtype_str = ""
    if sesspar.runtype != "prod":
        runtype_str = f" ({sesspar.runtype} data)"

    logger.info(
        f"Analysis of {sesspar.plane} responses to {stimpar.stimtype[:-1]} "
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
                           records the corresponding function
    """

    fct_dict = dict()

    # 0. Plots the full traces for each session
    fct_dict["f"] = gen_analys.run_full_traces

    # 1. Analyses and plots average running by quintile x surprise for each 
    # session 
    fct_dict["t"] = gen_analys.run_traces_by_qu_surp_sess

    # 2. Analyses and plots average running locked to surprise by quintile x 
    # surprise for each session 
    fct_dict["l"] = gen_analys.run_traces_by_qu_lock_sess

    # 3. Analyses and plots magnitude of change in dF/F area from first to last 
    # quintile of surprise vs no surprise sequences, for each session
    fct_dict["m"] = gen_analys.run_mag_change

    # 4. Analyses and plots autocorrelation
    fct_dict["a"] = gen_analys.run_autocorr

    return fct_dict


#############################################
def run_analyses(sessions, analysis_dict, analyses, seed=None, parallel=False):
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
    """

    if len(sessions) == 0:
        logger.warning("No sessions meet these criteria.")
        return

    # changes backend and defaults
    plot_util.manage_mpl(cmap=False, **analysis_dict["figpar"]["mng"])

    fct_dict = get_analysis_fcts()

    args_dict = copy.deepcopy(analysis_dict)
    for key, item in zip(["seed", "parallel", "datatype"], 
        [seed, parallel, "run"]):
        args_dict[key] = item

    # run through analyses
    for analysis in analyses:
        if analysis not in fct_dict.keys():
            raise ValueError(f"{analysis} analysis not found.")
        fct = fct_dict[analysis]
        args_dict_use = gen_util.keep_dict_keys(
            args_dict, inspect.getfullargspec(fct).args)
        fct(sessions=sessions, analysis=analysis, **args_dict_use)


#############################################
def main(args):
    """
    main(args)

    Runs analyses with parser arguments.

    Required args:
        - args (dict): parser argument dictionary
    """

    logger_util.set_level(level=args.log_level)

    args.fontdir = DEFAULT_FONTDIR

    if args.dict_path is not None:
        source = "modif" if args.modif else "run"
        plot_dicts.plot_from_dicts(
            Path(args.dict_path), source=source, plt_bkend=args.plt_bkend, 
            fontdir=args.fontdir, parallel=args.parallel, 
            datetime=not(args.no_datetime))
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
                omit_sess=args.omit_sess, omit_mice=args.omit_mice)
        else:
            all_sess_ns = gen_util.list_if_not(args.sess_n)

        # get analysis parameters for each session number
        all_analys_pars = gen_util.parallel_wrap(
            prep_analyses, all_sess_ns, args_list=[args, mouse_df], 
            parallel=args.parallel)

        # split analyses between parallel and sequential
        if args.parallel:
            run_seq = "" # should be run parallel within analysis
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
        help="data directory (if None, uses a directory defined below)")
    parser.add_argument("--output", default=".", type=Path, 
        help="where to store output")
    parser.add_argument("--analyses", default="all", 
        help=("analyses to run, e.g. 'ftl', 'all' or 'all_f' (all, save 'f'). "
            f"ANALYSES: {ANALYSIS_STR}"))
    parser.add_argument("--sess_n", default="all",
        help="session to aim for, e.g. 1, 2, last, all")
    parser.add_argument("--dict_path", default=None, 
        help=("path to info dictionary from which to plot data."))

        # technical parameters
    parser.add_argument("--plt_bkend", default=None, 
        help="switch mpl backend when running on server")
    parser.add_argument("--parallel", action="store_true", 
        help="do runs in parallel.")
    parser.add_argument("--seed", default=-1, type=int, 
        help="random seed (-1 for None)")
    parser.add_argument("--log_level", default="info", 
        help="logging level (does not work with --parallel)")

        # session parameters
    parser.add_argument("--runtype", default="prod", help="prod or pilot")
    parser.add_argument("--plane", default="soma", help="soma, dend")
    parser.add_argument("--min_rois", default=5, type=int, 
        help="min rois criterion")
        # stimulus parameters
    parser.add_argument("--bri_dir", default="right", 
        help="brick dir (right, left, or both)") 
    parser.add_argument("--gabfr", default=3, type=int, 
        help="gabor frame at which to start sequences")  
    parser.add_argument("--post", default=1.5, type=float, 
        help="sec after reference frames")
    parser.add_argument("--stimtype", default="gabors", 
        help="stimulus to analyse")   
    
    # generally fixed 
        # analysis parameters
    parser.add_argument("--keepnans", action="store_true", 
        help=("use running array in which NaN values have not been "
            "interpolated."))
    parser.add_argument("--stats", default="mean", help="plot mean or median")
    parser.add_argument("--error", default="sem", 
        help="sem for SEM/MAD, std for std/qu")
    parser.add_argument("--scale", action="store_true", 
        help="whether to scale running data")    

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
    parser.add_argument("--bri_size", default=128, 
        help="brick size (128, 256, or both)")
    parser.add_argument("--gabk", default=16,
        help="kappa value (4, 16, or both)")    
    parser.add_argument("--gab_ori", default="all",
        help="gabor orientation values (0, 45, 90, 135, all)")    
    parser.add_argument("--pre", default=0, type=float, help="sec before frame")
        # permutation parameters
    parser.add_argument("--n_perms", default=10000, type=int, 
        help="nbr of permutations")
    parser.add_argument("--tails", default="2", 
        help="tails for perm analysis (2, lo, up)")
        # quintile parameters
    parser.add_argument("--n_quints", default=4, type=int, 
        help="nbr of quintiles")
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
    parser.add_argument("--no_sharey", action="store_true", 
        help="don't share y axis")
        # plot using modif_analys_plots (if plotting from dictionary)
    parser.add_argument("--modif", action="store_true", 
        help=("plot from dictionary using modified plot functions"))
        
    args = parser.parse_args()

    return args


#############################################
if __name__ == "__main__":

    args = parse_args()
    main(args)

