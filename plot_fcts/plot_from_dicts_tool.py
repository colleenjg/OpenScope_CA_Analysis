"""
plot_from_dicts_tool.py

This script contains functions to plot from dictionaries.

Authors: Colleen Gillon

Date: October, 2019

Note: this code uses python 3.7.

"""

import glob
import inspect
import logging
import os

from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from util import file_util, gen_util, logger_util
from plot_fcts import roi_analysis_plots as roi_plots
from plot_fcts import gen_analysis_plots as gen_plots
from plot_fcts import pup_analysis_plots as pup_plots
from plot_fcts import modif_analysis_plots as mod_plots
from plot_fcts import acr_sess_analysis_plots as acr_sess_plots
from plot_fcts import logreg_plots, glm_plots

logger = logging.getLogger(__name__)


#############################################
def plot_from_dicts(direc, source="roi", plt_bkend=None, fontdir=None, 
                    plot_tc=True, parallel=False, datetime=True, pattern="", 
                    depth=0):
    """
    plot_from_dicts(direc)

    Plots data from dictionaries containing analysis parameters and results, or 
    path to results.

    Required args:
        - direc (str): path to directory in which dictionaries to plot data 
                       from are located or path to a single json file
    
    Optional_args:
        - source (str)   : plotting source ("roi", "run", "gen", "pup", 
                           "modif", "logreg", "glm")
        - plt_bkend (str): mpl backend to use for plotting (e.g., "agg")
                           default: None
        - fontdir (str)  : directory in which additional fonts are stored
                           default: None
        - plot_tc (bool) : if True, tuning curves are plotted for each ROI 
                           default: True
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
                           default: False
        - datetime (bool): figpar["save"] datatime parameter (whether to 
                           place figures in a datetime folder)
                           default: True
        - pattern (str)  : pattern based on which to include json files in 
                           direc if direc is a directory
                           default: ""
        - depth (int)    : maximum depth at which to check for json files if 
                           direc is a directory
                           default: 0
    """
    
    file_util.checkexists(direc)

    if os.path.isdir(direc):
        if source == "logreg": 
            fn = "hyperparameters.json"
            all_paths = glob.glob(os.path.join(direc, fn)) + \
                glob.glob(os.path.join(direc, "*", fn))
            dict_paths = [os.path.dirname(dp) for dp in all_paths]
        else:
            dict_paths = []
            for d in range(depth + 1):
                dict_paths.extend(
                    glob.glob(os.path.join(direc, *(["*"] * d), "*.json")))
            
            dict_paths = list(filter(lambda x : pattern in x, dict_paths))

        if len(dict_paths) == 0:
            raise ValueError(f"No jsons found in {direc} at "
                f"depth {depth} with pattern '{pattern}'.")
    
    elif ".json" not in direc:
        raise ValueError("If providing a file, must be a json file.")
    else:
        if source == "logreg" and not direc.endswith("hyperparameters.json"):
            raise ValueError("For logreg source, must provide path to "
                "a hyperparameters json file.")

        dict_paths = [direc]

    if len(dict_paths) > 1:
        logger.info(f"Plotting from {len(dict_paths)} dictionaries.")

    sub_parallel = parallel * (len(dict_paths) == 1)

    args_dict = {
        "plt_bkend": plt_bkend, 
        "fontdir"  : fontdir,
        "plot_tc"  : plot_tc,
        "parallel" : sub_parallel,
        "datetime" : datetime,
        }

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
        fct = logreg_plots.plot_from_dict
    elif source == "glm":
        fct = glm_plots.plot_from_dict
    elif source == "acr_sess":
        fct = acr_sess_plots.plot_from_dict
    else:
        gen_util.accepted_values_error("source", source, sources)

    args_dict = gen_util.keep_dict_keys(
        args_dict, inspect.getfullargspec(fct).args)
    gen_util.parallel_wrap(
        fct, dict_paths, args_dict=args_dict, parallel=parallel)

    plt.close("all")
