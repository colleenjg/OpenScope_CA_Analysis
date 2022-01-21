#!/usr/bin/env python

"""
run_paper_figures.py

This script produces paper figures for this project.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.

"""

import argparse
import copy
import inspect
import logging
from pathlib import Path
import time
import warnings

from matplotlib import pyplot as plt

# try to set cache/config as early as possible (for clusters)
from util import gen_util 
gen_util.CC_config_cache()

from util import logger_util, plot_util
from sess_util import sess_gen_util, sess_ntuple_util, sess_plot_util
from paper_fig_util import paper_organization, helper_fcts


DEFAULT_DATADIR = Path("..", "data", "OSCA")
DEFAULT_MOUSE_DF_PATH = Path("mouse_df.csv")
DEFAULT_FONTDIR = Path("..", "tools", "fonts")

logger = logging.getLogger(__name__)

SEP = f"\n{'=' * 80}\n"
DOUBLE_SEP = f"\n{'_' * 80}\n{'=' * 80}\n"


#############################################
def reformat_sess_n(sess_n):
    """
    reformat_sess_n(sess_n)

    Returns reformatted sess_n argument, converting ranges to lists.

    Required args:
        - sess_n (str): 
            session number or range (e.g., "1-1", "all")
    
    Returns:
        - sess_n (str or list): 
            session number or range (e.g., [1, 2, 3], "all")
    """

    if "-" in str(sess_n):
        vals = str(sess_n).split("-")
        if len(vals) != 2:
            raise ValueError("If sess_n is a range, must have format 1-3.")
        st = int(vals[0])
        end = int(vals[1]) + 1
        sess_n = list(range(st, end))
    
    elif sess_n not in ["any", "all"]:
        sess_n = gen_util.list_if_not(sess_n)

    return sess_n


#############################################
def init_analysis(args):
    """
    init_analysis(args)

    Initializes analysis parameters based on input arguments containers.

    Required args:
        - args (dict): 
            parser argument dictionary

    Returns:
        - analysis_dict (dict): 
            dictionary of analysis parameters
            ["analyspar"] (AnalysPar): named tuple of analysis parameters
            ["sesspar"] (SessPar): named tuple with session parameters
            ["stimpar"] (StimPar): named tuple with stimulus parameters
            ["basepar"] (LatPar): named tuple with latency parameters
            ["idxpar"] (PermPar): named tuple with unexpected event index 
                parameters
            ["logregpar"] (LogRegPar): 
                named tuple with logistic regression parameters
            ["permpar"] (PermPar): named tuple with permutation parameters
            ["figpar"] (dict): dictionary containing subdictionaries 
                (see sess_plot_util.init_figpar), with fig_panel_analysis 
                added under the "fig_panel_analysis" key.
    """
    
    args = copy.deepcopy(args)

    fig_panel_analysis = paper_organization.FigurePanelAnalysis(
        figure=args.figure, 
        panel=args.panel, 
        datadir=args.datadir,
        mouse_df_path=args.mouse_df_path,
        output=args.output,
        full_power=args.full_power, 
        seed=args.seed, 
        parallel=args.parallel,
        plt_bkend=args.plt_bkend,
        fontdir=args.fontdir,
        )

    specific_params = fig_panel_analysis.specific_params
    sess_n = reformat_sess_n(specific_params["sess_n"])

    analysis_dict = dict()

    # analysis parameters
    analysis_dict["analyspar"] = sess_ntuple_util.init_analyspar(
        fluor="dff", # type of fluorescence data to use (dF/F)
        rem_bad=specific_params["rem_bad"], # whether to remove bad ROIs OR interpolate bad values in run or pupil data
        stats="mean", # type of statistic to measure (mean/median)
        error=specific_params["error"], # type of error to measure (std/SEM)
        scale=specific_params["scale"], # whether to scale ROIs (robust scaling)
        tracked=specific_params["tracked"], # whether to use only tracked ROIs
        )

    # session inclusion parameters
    analysis_dict["sesspar"] = sess_ntuple_util.init_sesspar(
        sess_n=sess_n, # session number(s)
        plane=specific_params["plane"], # recording plane(s)
        line=specific_params["line"], # mouse line(s)
        pass_fail="P", # include sessions that passed QC
        incl="all", # include all remaining sessions
        runtype="prod", # production run data
        mouse_n=specific_params["mouse_n"], # mouse numbers
        )

    # stimulus analysis parameters
    analysis_dict["stimpar"] = sess_ntuple_util.init_stimpar(
        stimtype=specific_params["stimtype"], # stimulus to analyse
        visflow_dir=specific_params["visflow_dir"], # visual flow directions
        visflow_size=specific_params["visflow_size"], # visual flow square sizes
        gabfr=specific_params["gabfr"], # Gabor frame to center analyses on
        gabk=specific_params["gabk"], # Gabor orientation kappas
        gab_ori=specific_params["gab_ori"], # mean Gabor orientations
        pre=specific_params["pre"], # number of seconds pre reference frame
        post=specific_params["post"] # number of seconds post reference frame
        )

    # baseline parameters
    analysis_dict["basepar"] = sess_ntuple_util.init_basepar(
        baseline=0, # sequence baselining (None)
        )

    # USI analysis parameters
    analysis_dict["idxpar"] = sess_ntuple_util.init_idxpar(
        op="d-prime", # USI measure
        feature=specific_params["idx_feature"], # how to select sequences
        )

    # logistic regression parameters
    analysis_dict["logregpar"] = sess_ntuple_util.init_logregpar(
        comp=specific_params["comp"], # classes
        ctrl=True, # control for dataset size
        n_epochs=1000, # number of training epochs
        batchsize=200, # batch size
        lr=0.0001, # learning rate
        train_p=0.75, # train:test split
        wd=0, # weight decay to use (None)
        )

    # permutation analysis parameters
    analysis_dict["permpar"] = sess_ntuple_util.init_permpar(
        n_perms=fig_panel_analysis.n_perms, # number of permutations to run
        p_val=0.05, # significance threshold to consider
        tails=specific_params["tails"], # number of tails
        multcomp=False # multiple comparisons
        )

    # figure plotting parameters
    analysis_dict["figpar"] = sess_plot_util.init_figpar(
        datetime=False, 
        overwrite=args.overwrite, 
        runtype="prod",
        output=args.output, 
        plt_bkend=args.plt_bkend, 
        fontdir=args.fontdir,
        paper=True,
        )

    analysis_dict["figpar"]["fig_panel_analysis"] = fig_panel_analysis

    return analysis_dict


#############################################
def init_sessions(analyspar, sesspar, mouse_df, datadir, sessions=None, 
                  roi=True, run=False, pupil=False, parallel=False):
    """
    init_sessions(sesspar, mouse_df, datadir)

    Initializes sessions.

    Required args:
        - analyspar (AnalysPar): 
            named tuple containing session parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - mouse_df (pandas df): 
            path name of dataframe containing information on each session
        - datadir (Path): 
            path to data directory
    
    Optional args:
        - sessions (list): 
            preloaded sessions
            default: None
        - roi (bool): 
            if True, ROI data is loaded
            default: True
        - run (bool): 
            if True, running data is loaded
            default: False
        - pupil (bool): 
            if True, pupil data is loaded
            default: False

    Returns:
        - sessions (list): 
            Session objects 
    """

    sesspar_dict = sesspar._asdict()
    sesspar_dict.pop("closest")

    # identify sessions needed
    sessids = sorted(
        sess_gen_util.get_sess_vals(mouse_df, "sessid", **sesspar_dict)
    )

    if len(sessids) == 0:
        raise ValueError("No sessions meet the criteria.")

    # check for preloaded sessions, and only load new ones
    if sessions is not None:
        loaded_sessids = [session.sessid for session in sessions]
        ext_str = " additional"
    else:
        sessions = []
        loaded_sessids = []
        ext_str = ""

    # identify new sessions to load
    load_sessids = list(
        filter(lambda sessid: sessid not in loaded_sessids, sessids)
        )

    # remove sessions that are not needed
    if len(sessions):
        sessions = [
            session for session in sessions if session.sessid in sessids
            ]

        # check that previously loaded sessions have roi/run/pupil data loaded
        args_list = [roi, run, pupil, analyspar.fluor, analyspar.dend]
        with logger_util.TempChangeLogLevel(level="warning"):
            sessions = gen_util.parallel_wrap(
                sess_gen_util.check_session, sessions, args_list=args_list, 
                parallel=parallel)

    # load new sessions
    if len(load_sessids):
        logger.info(
            f"Loading {len(load_sessids)}{ext_str} session(s)...", 
            extra={"spacing": "\n"}
            )

        args_dict = {
            "datadir"   : datadir,
            "mouse_df"  : mouse_df,
            "runtype"   : sesspar.runtype,
            "full_table": False,
            "fluor"     : analyspar.fluor,
            "dend"      : analyspar.dend,
            "roi"       : roi,
            "run"       : run,
            "pupil"     : pupil,
            "temp_log"  : "critical" # suppress almost all logs 
        }

        new_sessions = gen_util.parallel_wrap(
            sess_gen_util.init_sessions, load_sessids, args_dict=args_dict, 
            parallel=parallel, use_tqdm=True
            )

        # flatten list of new sessions, and add to full sessions list
        new_sessions = [sess for singles in new_sessions for sess in singles]
        sessions = sessions + new_sessions

    # combine session lists, and sort
    sorter = [sessids.index(session.sessid) for session in sessions]
    sessions = [sessions[i] for i in sorter]

    # update ROI tracking parameters
    for sess in sessions:
        sess.set_only_tracked_rois(analyspar.tracked)

    return sessions


#############################################
def run_single_panel(args, sessions=None, new_fig=False):
    """
    run_single_panel(args, sessions=None, new_fig=False)

    Runs analyses and plots a single panel.

    Required args:
        - args (dict): 
            parser argument dictionary

    Optional args:
        - sessions (list):
            preloaded Session objects
            default: None
        - new_fig (bool):
            if True, a new figure is being plotted 
            (additional separator is logged)
            default: False

    Returns:
        - sessions (list):
            loaded Session objects
    """

    analysis_dict = init_analysis(args)
    fig_panel_analysis = analysis_dict["figpar"]["fig_panel_analysis"]

    # changes backend and defaults
    plot_util.manage_mpl(cmap=False, **analysis_dict["figpar"]["mng"])
    sess_plot_util.update_plt_linpla()

    action = "Running analysis and producing plot"
    if args.plot_only:
        action = "Producing plot"

    sep = DOUBLE_SEP if new_fig else SEP
    logger.info(
        f"{sep}Fig. {fig_panel_analysis.figure}{fig_panel_analysis.panel}. "
        f"{action}: {fig_panel_analysis.description}", 
        extra={"spacing": "\n"}
        )

    # Log any relevant warnings to the console
    fig_panel_analysis.log_warnings(plot_only=args.plot_only)

    # Check if analysis needs to be rerun, and if not, replots only.
    run_analysis, data_path = \
        helper_fcts.check_if_data_exists(
            analysis_dict["figpar"], overwrite_plot_only=args.plot_only, 
            raise_no_data=False
            )
    
    if not run_analysis:
        return
    elif args.plot_only:
        logger.warning(
            f"Skipping plot, as no analysis data was found under {data_path}.", 
            extra={"spacing": "\n"}
        )
        return

    sessions = init_sessions(
        analyspar=analysis_dict["analyspar"], 
        sesspar=analysis_dict["sesspar"], 
        mouse_df=args.mouse_df_path, 
        datadir=args.datadir, 
        sessions=sessions,
        roi=fig_panel_analysis.specific_params["roi"], 
        run=fig_panel_analysis.specific_params["run"], 
        pupil=fig_panel_analysis.specific_params["pupil"], 
        parallel=args.parallel,
    )

    analysis_dict["seed"] = fig_panel_analysis.seed
    analysis_dict["parallel"] = bool(args.parallel  * (not args.debug))

    analysis_fct = fig_panel_analysis.analysis_fct
    analysis_dict_use = gen_util.keep_dict_keys(
        analysis_dict, inspect.getfullargspec(analysis_fct).args)
    
    analysis_fct(sessions=sessions, **analysis_dict_use)

    return sessions


#############################################
def main(args):
    """
    main(args)

    Runs analyses with parser arguments.

    Required args:
        - args (dict): 
            parser argument dictionary
    """

    logger_util.set_level(level=args.log_level)

    if args.datadir is None: 
        args.datadir = DEFAULT_DATADIR
    else:
        args.datadir = Path(args.datadir)
    args.mouse_df_path = DEFAULT_MOUSE_DF_PATH

    # Directory with additional fonts
    args.fontdir = DEFAULT_FONTDIR if DEFAULT_FONTDIR.exists() else None

    # warn if parallel is not used
    if args.overwrite and not(args.plot_only):
        if not args.parallel:
            warnings.warn(
                "Unless memory demands are too high for the machine being "
                "used, it is strongly recommended that paper analyses be run "
                "with the '--parallel' argument (enables computations to be "
                "distributed across available CPU cores). Otherwise, analyses "
                "may be very slow.", category=UserWarning, stacklevel=1
                )
            time.sleep(paper_organization.WARNING_SLEEP)


    # run through figure(s) and panel(s)
    if args.figure == "all":
        figures = paper_organization.get_all_figures()
    else:
        figures = [args.figure]

    sessions = None
    panel = args.panel
    for args.figure in figures:
        if panel == "all":
            panels = paper_organization.get_all_panels(args.figure)
        else:
            panels = [panel]

        for p, args.panel in enumerate(panels):
            new_fig = (p==0)
            try: 
                with gen_util.TimeIt():
                    sessions = run_single_panel(
                        args, sessions=sessions, new_fig=new_fig
                        )
            except Exception as err:
                sep = DOUBLE_SEP if new_fig else SEP
                if "Cannot plot figure panel" in str(err):
                    lead = f"{sep}Fig. {args.figure}{args.panel.upper()}"
                    logger.info(f"{lead}. {err}")
                else:
                    raise err

            plot_util.cond_close_figs()


#############################################
def parse_args():
    """
    parse_args()

    Returns parser arguments.

    Returns:
        - args (dict): 
            parser argument dictionary
    """

    parser = argparse.ArgumentParser()

        # data parameters
    parser.add_argument("--datadir", default=None, 
        help="data directory (if not provided, uses a default directory)")
    parser.add_argument("--output", default=".", type=Path,
        help="main directory in which to store output")
    parser.add_argument("--overwrite", action="store_true", 
        help=("overwrite figure, and analysis files if args.plot_only is False"
        "(figures are always overwritten)"))
    parser.add_argument("--plot_only", action="store_true", 
        help=("only replots panels for which analysis files exist in the "
            "specified directory."))

        # analysis parameter
    parser.add_argument("--full_power", action="store_true", 
        help=("run analyses with all permutations (much slower for local "
        "computing)"))
    parser.add_argument("--figure", default="1", 
        help="figure for which to plot results")
    parser.add_argument("--panel", default="all", 
        help="specific panel for which to plot results or 'all'")

        # technical parameters
    parser.add_argument("--plt_bkend", default=None, 
        help="switch mpl backend when running on server")
    parser.add_argument("--parallel", action="store_true", 
        help="do analyses in parallel.")
    parser.add_argument("--debug", action="store_true", 
        help="only enable session loading in parallel")
    parser.add_argument("--seed", default="paper", 
        help="paper random seed, a different value or -1 for a random seed")
    parser.add_argument("--log_level", default="info", 
        help="logging level (does not work with --parallel)")

    args = parser.parse_args()

    return args


#############################################
if __name__ == "__main__":

    args = parse_args()

    main(args)

