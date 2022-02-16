"""
modif_analysis_plots.py

This script is used to modify plot designs on the fly for ROI and running 
analyses from dictionaries, e.g. for presentations or papers.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

from pathlib import Path
import warnings

from matplotlib import pyplot as plt

from util import file_util, logger_util, plot_util
from sess_util import sess_plot_util
from extra_plot_fcts import gen_analysis_plots as gen_plots


TAB = "    "


logger = logger_util.get_module_logger(name=__name__)


#############################################
def plot_from_dict(dict_path, plt_bkend=None, fontdir=None, plot_tc=True, 
                   parallel=False, datetime=True, overwrite=False):
    """
    plot_from_dict(dict_path)

    Plots data from dictionaries containing analysis parameters and results.

    Required args:
        - dict_path (Path): path to dictionary to plot data from
    
    Optional_args:
        - plt_bkend (str) : mpl backend to use for plotting (e.g., "agg")
                            default: None
        - fontdir (Path)  : path to directory where additional fonts are stored
                            default: None
        - plot_tc (bool)  : if True, tuning curves are plotted for each ROI 
                            (dummy argument)
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
    """
    
    logger.info(f"Plotting from dictionary: {dict_path}", 
        extra={"spacing": "\n"})
    
    figpar = sess_plot_util.init_figpar(
        plt_bkend=plt_bkend, fontdir=fontdir, datetime=datetime,
        overwrite=overwrite
        )
    plot_util.manage_mpl(cmap=False, **figpar["mng"])

    plt.rcParams["figure.titlesize"] = "xx-large"
    plt.rcParams["axes.titlesize"] = "xx-large"

    dict_path = Path(dict_path)

    info = file_util.loadfile(dict_path)
    savedir = dict_path.parent

    analysis = info["extrapar"]["analysis"]

    # 1. Plot average traces by quantile x unexpected for each session 
    if analysis == "t": # traces
        gen_plots.plot_traces_by_qu_unexp_sess(
            figpar=figpar, savedir=savedir, modif=True, **info
            )

    # 2. Plot average traces by quantile, locked to unexpected for each session 
    elif analysis == "l": # unexpected locked traces
        gen_plots.plot_traces_by_qu_lock_sess(
            figpar=figpar, savedir=savedir, modif=True, **info
            )

    else:
        warnings.warn(f"No modified plotting option for analysis {analysis}", 
            category=UserWarning, stacklevel=1)

    plot_util.cond_close_figs()



