"""
run_paper_figures.py

This script produces paper figures for this project.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.

"""

import argparse
import copy
import logging
import os

# try to set cache/config as early as possible (for clusters)
from util import gen_util 
gen_util.CC_config_cache()

from util import logger_util
from sess_util import sess_ntuple_util, sess_plot_util
from paper_fig_util import paper_organization 


DEFAULT_DATADIR = os.path.join("..", "data", "OSCA")
DEFAULT_MOUSE_DF_PATH = "mouse_df.csv"
DEFAULT_FONTDIR = os.path.join("..", "tools", "fonts")

logger = logging.getLogger(__name__)



def init_analysis(args):


    fig_panel_analysis = paper_organization.FigurePanelAnalysis(
        figure=args.figure, 
        panel=args.panel, 
        datadir=args.datadir,
        mouse_df_path=args.mouse_df_path,
        output=args.output,
        full_power=args.full_power, 
        seed=args.seed, 
        rerun_local=args.rerun_local,
        parallel=args.parallel,
        plt_bkend=args.plt_bkend,
        fontdir=args.fontdir,
        )
    
    specific_params, plot_fcts = fig_panel_analysis.get_plot_info()

    # lower power: produce a warning and amends intermediate dictionary name

    args = copy.deepcopy(args)

    analysis_dict = dict()

    # analysis parameters
    analysis_dict["analyspar"] = sess_ntuple_util.init_analyspar(
        fluor="dff", # type of fluorescence data to use (dF/F)
        remnans=True, # whether to ROIs with NaNs/Infs
        stats="mean", # type of statistic to measure (mean/median)
        error="sem", # type of error to measure (std/SEM)
        scale=args.scale # whether to scale ROIs (robust scaling)
        )

    # session inclusion parameters
    analysis_dict["sesspar"] = sess_ntuple_util.init_sesspar(
        args.sess_n, # session number(s)
        plane=args.plane, # recording plane(s)
        line=args.line, # mouse line(s)
        pass_fail="P", # include sessions that passed QC
        incl="all", # include all remaining sessions
        runtype="prod" # production run data
        )

    # stimulus analysis parameters
    analysis_dict["stimpar"] = sess_ntuple_util.init_stimpar(
        stimtype=args.stimtype, # stimulus to analyse
        bri_dir=args.bri_dir, # brick directions
        gab_fr=args.gabfr, # Gabor frame to center analyses on
        gab_ori=args.gab_ori, # mean Gabor orientations
        pre=args.pre, # number of seconds pre reference frame
        post=args.post # number of seconds post reference frame
        )
    
    # permutation analysis parameters
    analysis_dict["permpar"] = sess_ntuple_util.init_permpar(
        n_perms=args.n_perms, # number of permutations to run
        p_val=0.05, # significance threshold to consider
        tails=2 # number of tails
        )

    # figure plotting parameters
    analysis_dict["figpar"] = sess_plot_util.init_figpar(
        ncols=int(args.ncols), 
        datetime=not(args.no_datetime), 
        overwrite=args.overwrite, 
        runtype="prod",
        output=args.output, 
        plt_bkend=args.plt_bkend, 
        fontdir=args.fontdir
        )

    return analysis_dict


#############################################
def main(args):
    """
    main(args)

    Runs analyses with parser arguments.

    Required args:
        - args (dict): parser argument dictionary
    """

    logger_util.set_level(level=args.log_level)

    if args.datadir is None: args.datadir = DEFAULT_DATADIR
    args.fontdir = DEFAULT_FONTDIR
    args.mouse_df_path = DEFAULT_MOUSE_DF_PATH

    analysis_ntuples = get_analysis_ntuples(
        args.figure, args.panel, args.lower_power)


#############################################
def parse_args():
    """
    parse_args()

    Returns parser arguments.

    Returns:
        - args (dict): parser argument dictionary
    """

    parser = argparse.ArgumentParser()

        # data parameters
    parser.add_argument("--datadir", default=None, 
        help=("data directory (if None, uses a directory defined below)"))
    parser.add_argument("--output", default=".", help="where to store output")
    parser.add_argument("--rerun_local", action="store_true", 
        help="rerun and overwrite intermediate analysis files")

        # analysis parameter
    parser.add_argument("--full_power", default="store_true", 
        help=("run analyses with all permutations (much slower for local "
        "computing)"))
    parser.add_argument("--figure", default="1", 
        help="figure for which to plot results")
    parser.add_argument("--panel", default="A", 
        help="panel for which to plot results")

        # technical parameters
    parser.add_argument("--plt_bkend", default=None, 
        help="switch mpl backend when running on server")
    parser.add_argument("--parallel", action="store_true", 
        help="do analyses in parallel.")
    parser.add_argument("--seed", default="paper", 
        help="paper random seed or a different value")
    parser.add_argument("--log_level", default="info", 
        help="logging level (does not work with --parallel)")

    args = parser.parse_args()

    return args


#############################################
if __name__ == "__main__":

    args = parse_args()
    main(args)

