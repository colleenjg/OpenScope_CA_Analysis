"""
run_paper_figures.py

This script produces paper figures for this project.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.

"""

import argparse
import logging

# try to set cache/config as early as possible (for clusters)
from util import gen_util 
gen_util.CC_config_cache()

from util import logger_util
from sess_util import sess_ntuple_util
import paper_organization 


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
    
    # lower power: produce a warning and amend intermediate dictionary name

    args = copy.deepcopy(args)

    analysis_dict = dict()

    # analysis parameters
    analysis_dict["analyspar"] = sess_ntuple_util.init_analyspar(
        "dff", not(args.keepnans), args.stats, args.error, args.scale)

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
        overwrite=args.overwrite, runtype=args.runtype, output=args.output, 
        plt_bkend=args.plt_bkend, fontdir=args.fontdir)

    return analysis_dict











if __name__ == "__main__":

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
        help="do runs in parallel.")
    parser.add_argument("--seed", default="paper", 
        help="paper random seed or a different value")
    parser.add_argument("--log_level", default="info", 
        help="logging level (does not work with --parallel)")

    logger_util.set_level(level=args.log_level)

    if args.datadir is None: args.datadir = DEFAULT_DATADIR
    args.fontdir = DEFAULT_FONTDIR
    args.mouse_df_path = DEFAULT_MOUSE_DF_PATH

    analysis_ntuples = get_analysis_ntuples(
        args.figure, args.panel, args.lower_power)