"""
helper_fcts.py

This script contains helper functions for paper analysis plotting.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

from pathlib import Path

from util import file_util, logger_util


TAB = "    "

logger = logger_util.get_module_logger(name=__name__)


#############################################
def get_datafile_save_path(direc):
    """
    get_datafile_save_path(direc)

    Returns directory path amended with data file subdirectory.

    Required args:
        - direc (Path): directory path
    
    Returns:
        - direc (Path): amended directory path
    """

    direc = direc.joinpath("data_files")

    return direc
    

#############################################
def get_save_path(fig_panel_analysis, main_direc=None):
    """
    get_save_path(fig_panel_analysis)

    Returns directory path amended with data file subdirectory.

    Required args:
        - fig_panel_analysis (FigPanelAnalysis): 
            figure/panel analysis object
    
    Optional args:
        - main_direc (Path): 
            main directory to save file in
            default: None
    
    Returns:
        - savedir (Path): 
            specific directory to save file in
        - savename (str): 
            name under with to save file
    """

    savedir = Path(
        f"{fig_panel_analysis.paper}_paper", f"Fig{fig_panel_analysis.figure}"
        )
    savename = f"panel_{fig_panel_analysis.panel}"

    if main_direc is not None:
        savedir = main_direc.joinpath(savedir)

    if not (fig_panel_analysis.full_power and fig_panel_analysis.paper_seed):
        savedir = savedir.joinpath("panels_with_diff_params")
        if not (fig_panel_analysis.full_power):
            savename = f"{savename}_lower_power"
        if not (fig_panel_analysis.paper_seed):
            savename = f"{savename}_seed{fig_panel_analysis.seed}"
            
    return savedir, savename
    

#############################################
def check_if_data_exists(figpar, filetype="json", overwrite_plot_only=False, 
                         raise_no_data=True):
    """
    check_if_data_exists(figpar)

    Returns whether to rerun analysis, depending on whether data file already 
    exists and fipar["save"]["overwrite"] is True or False.

    Required args:
        - figpar (dict): 
            dictionary containing figure parameters
            ["fig_panel_analysis"] (FigPanelAnalysis): figure/panel analysis 
                object
            ["dirs"]["figdir"] (Path): figure directory
            ["save"]["overwrite"] (bool): whether to overwrite data and figure 
                files

    Optional args:
        - filetype (str): 
            type of data file expected
            default: "json"
        - overwrite_plot_only (bool): 
            if True, data is replotted only. 
            default: False
        - raise_no_data (bool):
            if True, an error is raised if overwrite_plot_only is True, but no 
            analysis data is found.
            default: True

    Returns:
        - run_analysis (bool): 
            if True, analysis should be run
        - data_path (Path): 
            path to data (whether it exists, or not)
    """
    
    fig_panel_analysis = figpar["fig_panel_analysis"]
    savedir, savename = get_save_path(
        fig_panel_analysis, main_direc=figpar["dirs"]["figdir"]
    )
    
    datadir = get_datafile_save_path(savedir)

    data_path = file_util.add_ext(datadir.joinpath(savename), filetype)[0]

    run_analysis = True

    if data_path.is_file():
        warn_str = f"Analysis data already exists under {data_path}."
        if figpar["save"]["overwrite"] and not overwrite_plot_only:
            warn_str = f"{warn_str}\nFile will be overwritten."
            logger.warning(warn_str, extra={"spacing": "\n"})
        else:
            warn_str = (f"{warn_str}\nReplotting from existing file.\n"
                "To overwrite file, run script with the '--overwrite' "
                "argument, and without --plot_only.")
            logger.warning(warn_str, extra={"spacing": "\n"})

            info = file_util.loadfile(data_path)
            fig_panel_analysis.plot_fct(figpar=figpar, **info)
            run_analysis = False

    elif overwrite_plot_only and raise_no_data:
        raise RuntimeError(
            "overwrite_plot_only is True, but no analysis data was found "
            f"under {data_path}"
            )

    return run_analysis, data_path


#############################################
def plot_save_all(info, figpar):
    """
    plot_save_all(info, figpar)

    Plots figure and saves information dictionary.

    Required args:
        - info (dict): 
            dictionary with all analysis information
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
            ["fig_panel_analysis"] (FigPanelAnalysis): figure/panel analysis 
                object
    """

    fig_plot_fct = figpar["fig_panel_analysis"].plot_fct


    fulldir, savename = fig_plot_fct(figpar=figpar, **info)
    
    overwrite = figpar["save"]["overwrite"]
    fulldir = get_datafile_save_path(fulldir)
    file_util.saveinfo(info, savename, fulldir, "json", overwrite=overwrite)

