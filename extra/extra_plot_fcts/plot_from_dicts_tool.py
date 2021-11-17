"""
plot_from_dicts_tool.py

This script contains functions to plot from dictionaries.

Authors: Colleen Gillon

Date: October, 2019

Note: this code uses python 3.7.

"""

import argparse
import glob
import inspect
import logging
import sys
from pathlib import Path

# try to set cache/config as early as possible (for clusters)
from util import gen_util 
gen_util.CC_config_cache()

from matplotlib import pyplot as plt

sys.path.extend([".", ".."])
from util import file_util, gen_util, logger_util
from extra_plot_fcts import roi_analysis_plots as roi_plots
from extra_plot_fcts import gen_analysis_plots as gen_plots
from extra_plot_fcts import pup_analysis_plots as pup_plots
from extra_plot_fcts import modif_analysis_plots as mod_plots
from extra_plot_fcts import acr_sess_analysis_plots as acr_sess_plots
from extra_plot_fcts import logreg_plots, glm_plots

logger = logging.getLogger(__name__)


DEFAULT_FONTDIR = Path("..", "tools", "fonts")


#############################################
def plot_from_dicts(direc, source="roi", plt_bkend=None, fontdir=None, 
                    plot_tc=True, parallel=False, datetime=True, 
                    overwrite=False, pattern="", depth=10):
    """
    plot_from_dicts(direc)

    Plots data from dictionaries containing analysis parameters and results, or 
    path to results.

    Required args:
        - direc (Path): path to directory in which dictionaries to plot data 
                        from are located or path to a single json file
    
    Optional_args:
        - source (str)    : plotting source ("roi", "run", "gen", "pup", 
                            "modif", "logreg", "glm")
        - plt_bkend (str) : mpl backend to use for plotting (e.g., "agg")
                            default: None
        - fontdir (Path)  : directory in which additional fonts are stored
                            default: None
        - plot_tc (bool)  : if True, tuning curves are plotted for each ROI 
                            default: True
        - parallel (bool) : if True, some of the analysis is parallelized 
                            across CPU cores
                            default: False
        - datetime (bool) : figpar["save"] datatime parameter (whether to 
                            place figures in a datetime folder)
                            default: True
        - overwrite (bool): figpar["save"] overwrite parameter (whether to 
                            overwrite figures)
                            default: False
        - pattern (str)   : pattern based on which to include json files in 
                            direc if direc is a directory
                            default: ""
        - depth (int)     : maximum depth at which to check for json files if 
                            direc is a directory
                            default: 0
    """
    
    file_util.checkexists(direc)

    direc = Path(direc)

    if direc.is_dir():
        if source == "logreg": 
            targ_file = "hyperparameters.json"
        else:
            targ_file = "*.json"
        
        dict_paths = []
        for d in range(depth + 1):
            dict_paths.extend(
                glob.glob(str(Path(direc, *(["*"] * d), targ_file)))
                )

        dict_paths = list(filter(lambda x : pattern in str(x), dict_paths))            

        if source == "logreg":
            dict_paths = [Path(dp).parent for dp in dict_paths]

        if len(dict_paths) == 0:
            raise OSError(f"No jsons found in {direc} at "
                f"depth {depth} with pattern '{pattern}'.")
    
    elif ".json" not in str(direc):
        raise ValueError("If providing a file, must be a json file.")
    else:
        if (source == "logreg" and 
            not str(direc).endswith("hyperparameters.json")):
            raise ValueError("For logreg source, must provide path to "
                "a hyperparameters json file.")

        dict_paths = [direc]

    if len(dict_paths) > 1:
        logger.info(f"Plotting from {len(dict_paths)} dictionaries.")

    fontdir = Path(fontdir) if fontdir is not None else fontdir
    args_dict = {
        "plt_bkend": plt_bkend, 
        "fontdir"  : fontdir,
        "plot_tc"  : plot_tc,
        "datetime" : datetime,
        "overwrite": overwrite,
        }

    pass_parallel = True
    sources = ["roi", "run", "gen", "modif", "pup", "logreg", "glm", "acr_sess"]
    if source == "roi":
        fct = roi_plots.plot_from_dict
    elif source in ["run", "gen"]:
        fct = gen_plots.plot_from_dict
    elif source in ["pup", "pupil"]:
        fct = pup_plots.plot_from_dict
    elif source == "modif":
        fct = mod_plots.plot_from_dict
    elif source == "logreg":
        pass_parallel = False
        fct = logreg_plots.plot_from_dict
    elif source == "glm":
        pass_parallel = False
        fct = glm_plots.plot_from_dict
    elif source == "acr_sess":
        fct = acr_sess_plots.plot_from_dict
    else:
        gen_util.accepted_values_error("source", source, sources)

    args_dict = gen_util.keep_dict_keys(
        args_dict, inspect.getfullargspec(fct).args)
    gen_util.parallel_wrap(
        fct, dict_paths, args_dict=args_dict, parallel=parallel, 
        pass_parallel=pass_parallel)

    plt.close("all")


#############################################
def main(args):
    """
    main(args)

    Runs analyses with parser arguments.

    Required args:
        - args (dict): parser argument dictionary
    """

    plot_from_dicts(
        args.direc, 
        source=args.source, 
        plt_bkend=args.plt_bkend, 
        fontdir=args.fontdir, 
        plot_tc=not(args.not_plot_tc), 
        parallel=args.parallel, 
        datetime=not(args.no_datetime), 
        pattern=args.pattern, 
        depth=args.depth
        )


#############################################
def parse_args():
    """
    parse_args()

    Returns parser arguments.

    Returns:
        - args (dict): parser argument dictionary
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--direc",  
        help="directory in which jsons are stored or path to a specific json")

    parser.add_argument("--source", default="roi", help="plotting source")
    parser.add_argument("--plt_bkend", default=None, 
        help="switch mpl backend when running on server")
    parser.add_argument("--fontdir", default=DEFAULT_FONTDIR, 
        help="directory in which additional fonts are stored")
    parser.add_argument("--not_plot_tc", action="store_true", 
        help="don't plot tuning curves for individual ROIs, if applicable")  
    parser.add_argument("--parallel", action="store_true", 
        help="do runs in parallel.")
    parser.add_argument("--no_datetime", action="store_true",
        help="create a datetime folder")
    parser.add_argument("--pattern", default="",
        help="pattern based on which to include json files in directory")
    parser.add_argument("--depth", default=0, type=int,
        help="maximum depth at which to check for json files in directory")

    args = parser.parse_args()

    return args


#############################################
if __name__ == "__main__":

    args = parse_args()
    main(args)

