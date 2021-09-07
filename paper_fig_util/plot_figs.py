"""
plot_figs.py

This script contains functions defining figure panel plotting.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

import numpy as np
import pandas as pd

from util import logger_util, plot_util
from paper_fig_util import helper_fcts
from plot_fcts import corr_plots, misc_plots, seq_plots, usi_plots

logger = logging.getLogger(__name__)


#############################################
def plot_roi_tracking(figpar, **kwargs):
    """
    """

    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_example_roi_usis(analyspar, sesspar, stimpar, basepar, permpar, 
                                idxpar, extrapar, chosen_rois_df, figpar):
    """
    plot_gabor_example_roi_usis(analyspar, sesspar, stimpar, basepar, permpar, 
                                idxpar, extrapar, chosen_rois_info, figpar)

    From dictionaries, plots traces for example ROIs.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - basepar (dict): 
            dictionary with keys of BasePar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["seed"] (int): seed
        - chosen_rois_df (pd.DataFrame in dict format):
            dataframe with a row for each ROI retained, and the following 
            columns, in addition to the basic sess_df columns: 
            - "target_idxs" (str): index values and significance aimed for
            - "roi_idxs" (float): ROI index
            - "roi_idx_percs" (float): ROI index percentile
            - "roi_ns" (int): ROI number with in its session
            - "roi_trace_stats" (list): 
                ROI trace stats (split x frames x stat (me, err))
            - time_values (list):
                values for each frame, in seconds
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  
    
    Returns:
        - fulldir (Path): 
            final path of the directory in which the figure is saved
        - savename (str): 
            name under which the figure is saved
    """
    
    title = "Gabor sequences (example ROIs)"

    chosen_rois_df = pd.DataFrame.from_dict(chosen_rois_df)

    # identify target idx vals and sigs
    target_idxs = chosen_rois_df["target_idxs"].unique()
    target_idx_vals = [float(idx.split("_")[0]) for idx in target_idxs]
    sorted_target_idxs = [
        target_idxs[i] for i in np.argsort(target_idx_vals)[::-1]
        ]

    ax = seq_plots.plot_sess_traces(
        chosen_rois_df, 
        analyspar=analyspar, 
        sesspar=sesspar,
        stimpar=stimpar, 
        figpar=figpar, 
        trace_col="roi_trace_stats",
        row_col="target_idxs",
        row_order=sorted_target_idxs, 
        title=title, 
        lock=False,
        size="small"
        )

    usi_plots.add_USI_boxes(ax, chosen_rois_df, sorted_target_idxs)

    fig = ax.reshape(-1)[0].figure

    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_example_roi_usi_sig(analyspar, sesspar, stimpar, basepar, 
                                   permpar, idxpar, extrapar, ex_idx_df, 
                                   figpar):
    """
    plot_gabor_example_roi_usi_sig(analyspar, sesspar, stimpar, basepar, 
                                   permpar, idxpar, extrapar, ex_idx_df, 
                                   figpar)

    From dictionaries, plots traces for example ROIs.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - basepar (dict): 
            dictionary with keys of BasePar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["seed"] (int): seed
        - ex_idx_df (pd.DataFrame):
            dataframe with a row for the example ROI, and the following 
            columns, in addition to the basic sess_df columns:
            - roi_ns (int): ROI number in session
            - roi_idxs (int): ROI feature index
            - roi_idx_percs (float): ROI feature index percentile
            - rand_idx_binned (list): bin counts for the random ROI indices
            - bin_edges (list): first and last bin edge
            - CI_lims (list): confidence interval limits (lo, hi)
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  
    
    Returns:
        - fulldir (Path): 
            final path of the directory in which the figure is saved
        - savename (str): 
            name under which the figure is saved
    """

    title = "Gabor sequences: example sig. ROI"

    ex_idx_df = pd.DataFrame.from_dict(ex_idx_df)

    ax = usi_plots.plot_ex_roi_hists(
        ex_idx_df, sesspar, permpar, figpar, title=title
        )
    fig = ax.reshape(-1)[0].figure

    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_roi_usi_distr(analyspar, sesspar, stimpar, basepar, permpar, 
                             idxpar, extrapar, idx_df, figpar):
    """
    plot_gabor_roi_usi_distr(analyspar, sesspar, stimpar, basepar, permpar, 
                             idxpar, extrapar, idx_df, figpar)

    From dictionaries, plots ROI index percentile distributions.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - basepar (dict): 
            dictionary with keys of BasePar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["seed"] (int): seed
        - idx_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns:
            - bin_edges (list): first and last bin edge
            - CI_perc (list): confidence interval percentile limits (lo, hi)
            - CI_edges (list): confidence interval limit values (lo, hi)
            - n_pos (int): number of positive ROIs
            - n_signif_lo (int): number of significant ROIs (low) 
            - n_signif_hi (int): number of significant ROIs (high)
            - roi_idx_binned (list): bin counts for the ROI indices
            - rand_idx_binned (list): bin counts for the random ROI indices
            - perc_idx_binned (list): bin counts for the ROI index percentiles
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  
    
    Returns:
        - fulldir (Path): 
            final path of the directory in which the figure is saved
        - savename (str): 
            name under which the figure is saved
    """

    title = "Gabor sequence percentiles"

    idx_df = pd.DataFrame.from_dict(idx_df)
    figpar["init"]["sharey"] = "row"

    ax = usi_plots.plot_idxs(
        idx_df, sesspar, figpar, plot="percs", title=title, size="small", 
        density=True, n_bins=40
        )
    fig = ax.reshape(-1)[0].figure

    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_roi_usi_sig(analyspar, sesspar, stimpar, basepar, permpar, 
                           idxpar, extrapar, perc_sig_df, figpar, 
                           common_oris=False):
    """
    plot_gabor_roi_usi_sig(analyspar, sesspar, stimpar, basepar, permpar, 
                           idxpar, extrapar, perc_sig_df, figpar)

    From dictionaries, plots percentage of significant ROIs.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters
        - idxpar (IdxPar): 
            named tuple containing index parameters
        - figpar (dict): 
            dictionary containing figure parameters
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["seed"] (int): seed
        - perc_sig_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns:
            - n_pos (int): number of positive ROIs
            - n_signif_lo (int): number of significant ROIs (low) 
            - n_signif_hi (int): number of significant ROIs (high)

            - perc_pos_idxs (num): percent positive ROIs (0-100)
            - perc_pos_idxs_stds (num): bootstrapped standard deviation 
                over percent significant ROIs
            - perc_pos_idxs_CIs (list): adjusted confidence interval for 
                percent pos. ROIs 
            - perc_pos_idxs_null_CIs (list): adjusted null CI for percent pos. 
                ROIs
            - perc_pos_idxs_raw_p_vals (num): unadjusted p-value for percent 
                pos. ROIs
            - perc_pos_idxs_p_vals (num): p-value for percent pos. ROIs, 
                adjusted for multiple comparisons and tails

            for sig in ["lo", "hi"]: for low vs high ROI indices
            - perc_sig_{sig}_idxs (num): percent significant ROIs (0-100)
            - perc_sig_{sig}_idxs_stds (num): bootstrapped standard deviation 
                over percent significant ROIs
            - perc_sig_{sig}_idxs_CIs (list): adjusted CI for percent sig. ROIs 
            - perc_sig_{sig}_idxs_null_CIs (list): adjusted null CI for percent 
                sig. ROIs
            - perc_sig_{sig}_idxs_raw_p_vals (num): unadjusted p-value for 
                percent sig. ROIs
            - perc_sig_{sig}_idxs_p_vals (num): p-value for percent sig. 
                ROIs, adjusted for multiple comparisons and tails

        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  
    
    Optional args:
        - common_oris (bool): 
            if True, data is for common orientations
            default: False

    Returns:
        - fulldir (Path): 
            final path of the directory in which the figure is saved
        - savename (str): 
            name under which the figure is saved
    """

    title = "Gabor sequence sig USIs"
    if common_oris:
        title = f"{title} (common oris)".replace("sequence", "seq")
    
    perc_sig_df = pd.DataFrame.from_dict(perc_sig_df)

    ax = usi_plots.plot_perc_sig_usis(
        perc_sig_df, 
        analyspar=analyspar,
        permpar=permpar,
        figpar=figpar, 
        title=title,
        )
    fig = ax.reshape(-1)[0].figure
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_roi_usi_sig_common_oris(**kwargs):
    """
    plot_gabor_roi_usi_sig_common_oris(**kwargs)

    From dictionaries, plots percentage of significant ROIs, for data with 
    common orientations.
    
    Returns figure name and save directory path.
    
    Required args: see plot_gabor_roi_usi_sig()

    Returns:
        - fulldir (Path): final path of the directory in which the figure 
                          is saved
        - savename (str): name under which the figure is saved
    """

    fulldir, savename = plot_gabor_roi_usi_sig(common_oris=True, **kwargs)

    return fulldir, savename


#############################################
def plot_pupil_run_responses(figpar, **kwargs):
    """
    """

    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_pupil_run_diffs(figpar, **kwargs):
    """
    """

    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_sequences_sess123(analyspar, sesspar, stimpar, basepar, 
                                 extrapar, trace_df, figpar):
    """
    plot_gabor_sequences_sess123(analyspar, sesspar, stimpar, basepar, 
                                 extrapar, trace_df, figpar)

    From dictionaries, plots Gabor sequences across sessions. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["seed"] (int): seed
        - trace_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - trace_stats (list): 
                trace stats (split x frames x stat (me, err))
            - time_values (list):
                values for each frame, in seconds

        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  

    Returns:
        - fulldir (Path): final path of the directory in which the figure 
                          is saved
        - savename (str): name under which the figure is saved
    """

    title = "Visual flow sequences"
    
    trace_df = pd.DataFrame.from_dict(trace_df)

    ax = seq_plots.plot_sess_traces(
        trace_df, 
        analyspar=analyspar, 
        sesspar=sesspar,
        stimpar=stimpar, 
        figpar=figpar, 
        trace_col="trace_stats",
        row_col="sess_ns",
        title=title, 
        lock=False,
        size="wide"
        )
    fig = ax.reshape(-1)[0].figure
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_sequence_diffs_sess123(analyspar, sesspar, stimpar, basepar, 
                                      permpar, extrapar, diffs_df, figpar):
    """
    plot_gabor_sequence_diffs_sess123(analyspar, sesspar, stimpar, basepar, 
                                      permpar, extrapar, diffs_df, figpar)

    From dictionaries, plots Gabor sequences across sessions. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["seed"] (int): seed
        - diffs_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - diff_stats (list): split difference stats (me, err)
            - null_CIs (list): adjusted null CI for split differences
            - raw_p_vals (float): unadjusted p-value for differences within 
                sessions
            - p_vals (float): p-value for differences within sessions, 
                adjusted for multiple comparisons and tails
            for session comparisons, e.g. 1v2:
            - raw_p_vals_{}v{} (float): unadjusted p-value for differences
                between sessions 
            - p_vals_{}v{} (float): p-value for differences between sessions, 
                adjusted for multiple comparisons and tails

        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  

    Returns:
        - fulldir (Path): final path of the directory in which the figure 
                          is saved
        - savename (str): name under which the figure is saved
    """

    title = "Visual flow sequence differences"
    
    diffs_df = pd.DataFrame.from_dict(diffs_df)

    # ax = seq_plots.plot_sess_diffs(
    #     diffs_df, 
    #     analyspar=analyspar, 
    #     sesspar=sesspar, 
    #     permpar=permpar, 
    #     figpar=figpar, 
    #     title=title, 
    #     wide=True
    #     )
    # fig = ax.reshape(-1)[0].figure

    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_rel_resp_sess123(figpar, **kwargs):
    """
    """

    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_tracked_roi_usis_sess123(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_tracked_roi_usi_means_sess123(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_decoding_sess123(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_norm_res_corr_example(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_norm_res_corrs_sess123_comps(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_roi_overlays_sess123(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_snrs_sess123(analyspar, sesspar, extrapar, snr_df, figpar):
    """
    plot_snrs_sess123(analyspar, sesspar, extrapar, snr_df, figpar)

    Plots ROI SNRs.

    Returns figure name and save directory path.

    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["seed"] (int): seed
        - snr_df (pd.DataFrame in dict format):
            dataframe with ROI SNRs under "snrs" for each 
            session, in addition to the basic sess_df columns
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  
    
    Returns:
        - fulldir (Path): 
            final path of the directory in which the figure is saved
        - savename (str): 
            name under which the figure is saved
    """

    title = "ROI SNR"
    snr_df = pd.DataFrame.from_dict(snr_df)
    ax = misc_plots.plot_snr_sigmeans_nrois(
        snr_df, figpar, datatype="snrs", title=title
        )
    fig = ax.reshape(-1)[0].figure
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_mean_signal_sess123(analyspar, sesspar, extrapar, sig_mean_df, figpar):
    """
    plot_mean_signal_sess123(analyspar, sesspar, extrapar, sig_mean_df, figpar)
    
    Plots ROI signal means.

    Returns figure name and save directory path.

    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["seed"] (int): seed
        - sig_mean_df (pd.DataFrame in dict format):
            dataframe with ROI signal means under "signal_means" for each 
            session, in addition to the basic sess_df columns
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  
    
    Returns:
        - fulldir (Path): 
            final path of the directory in which the figure is saved
        - savename (str): 
            name under which the figure is saved
    """

    title = "Mean ROI signal"
    sig_mean_df = pd.DataFrame.from_dict(sig_mean_df)
    ax = misc_plots.plot_snr_sigmeans_nrois(
        sig_mean_df, figpar, datatype="signal_means", title=title
        )
    fig = ax.reshape(-1)[0].figure

    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_nrois_sess123(analyspar, sesspar, extrapar, nrois_df, figpar):
    """
    plot_nrois_sess123(analyspar, sesspar, extrapar, nrois_df, figpar)
    
    Plots number of ROIs.

    Returns figure name and save directory path.

    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["seed"] (int): seed
        - nrois_df (pd.DataFrame in dict format):
            dataframe with number of ROIs, in addition to the basic sess_df 
            columns.
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  
    
    Returns:
        - fulldir (Path): 
            final path of the directory in which the figure is saved
        - savename (str): 
            name under which the figure is saved
    """

    title = "Number of ROIs"
    nrois_df = pd.DataFrame.from_dict(nrois_df)
    ax = misc_plots.plot_snr_sigmeans_nrois(
        nrois_df, figpar, datatype="nrois", title=title
        )
    fig = ax.reshape(-1)[0].figure

    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_stimulus_onset_sess123(analyspar, sesspar, stimpar, basepar, extrapar, 
                                trace_df, figpar):
    """
    plot_stimulus_onset_sess123(analyspar, sesspar, stimpar, basepar, extrapar, 
                                trace_df, figpar)

    From dictionaries, plots stimulus onset sequences across sessions. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["seed"] (int): seed
        - trace_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - trace_stats (list): 
                trace stats (split x frames x stat (me, err))
            - time_values (list):
                values for each frame, in seconds

        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  

    Returns:
        - fulldir (Path): final path of the directory in which the figure 
                          is saved
        - savename (str): name under which the figure is saved
    """

    title = "Stimulus onset sequences"
    
    trace_df = pd.DataFrame.from_dict(trace_df)

    ax = seq_plots.plot_sess_traces(
        trace_df, 
        analyspar=analyspar, 
        sesspar=sesspar,
        stimpar=stimpar, 
        figpar=figpar, 
        trace_col="trace_stats",
        row_col="sess_ns",
        title=title, 
        lock='stim_onset',
        size="reg"
        )
    fig = ax.reshape(-1)[0].figure
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_ex_roi_responses_sess1(analyspar, sesspar, stimpar, basepar, 
                                      extrapar, ex_traces_df, figpar):

    """
    plot_gabor_ex_roi_responses_sess1(sessions, analyspar, sesspar, stimpar, 
                                      basepar, extrapar, ex_traces_df, figpar)

    From dictionaries, plots example ROI responses to Gabor sequences. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["seed"] (int): seed
        - ex_traces_df (pd.DataFrame):
            dataframe with a row for each ROI, and the following columns, 
            in addition to the basic sess_df columns: 
            - time_values (list): values for each frame, in seconds
            - roi_ns (list): selected ROI number
            - traces (list): selected ROI sequence traces, dims: seq x frames
            - trace_stat (list): selected ROI trace mean or median
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  

    Returns:
        - fulldir (Path): final path of the directory in which the figure 
                          is saved
        - savename (str): name under which the figure is saved
    """

    title = "Example ROI responses to Gabor sequences"
    
    ex_traces_df = pd.DataFrame.from_dict(ex_traces_df)

    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename                                  


#############################################
def plot_gabor_roi_usi_sig_by_mouse(analyspar, sesspar, stimpar, basepar, 
                                    permpar, idxpar, extrapar, perc_sig_df, 
                                    figpar):
    """
    plot_gabor_roi_usi_sig_by_mouse(analyspar, sesspar, stimpar, basepar, 
                                    permpar, idxpar, extrapar, perc_sig_df, 
                                    figpar)

    Required args:
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["seed"] (int): seed
        - perc_sig_df (pd.DataFrame):
            dataframe with one row per mouse/session/line/plane, and the 
            following columns, in addition to the basic sess_df columns:
            - n_pos (int): number of positive ROIs
            - n_signif_lo (int): number of significant ROIs (low) 
            - n_signif_hi (int): number of significant ROIs (high)

            - perc_pos_idxs (num): percent positive ROIs (0-100)
            - perc_pos_idxs_stds (num): bootstrapped standard deviation 
                over percent significant ROIs
            - perc_pos_idxs_CIs (list): adjusted confidence interval for 
                percent pos. ROIs 
            - perc_pos_idxs_null_CIs (list): adjusted null CI for percent pos. 
                ROIs
            - perc_pos_idxs_raw_p_vals (num): unadjusted p-value for percent 
                pos. ROIs
            - perc_pos_idxs_p_vals (num): p-value for percent pos. ROIs, 
                adjusted for multiple comparisons and tails

            for sig in ["lo", "hi"]: for low vs high ROI indices
            - perc_sig_{sig}_idxs (num): percent significant ROIs (0-100)
            - perc_sig_{sig}_idxs_stds (num): bootstrapped standard deviation 
                over percent significant ROIs
            - perc_sig_{sig}_idxs_CIs (list): adjusted CI for percent sig. ROIs 
            - perc_sig_{sig}_idxs_null_CIs (list): adjusted null CI for percent 
                sig. ROIs
            - perc_sig_{sig}_idxs_raw_p_vals (num): unadjusted p-value for 
                percent sig. ROIs
            - perc_sig_{sig}_idxs_p_vals (num): p-value for percent sig. 
                ROIs, adjusted for multiple comparisons and tails

        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters

    Returns:
        - fulldir (Path): 
            final path of the directory in which the figure is saved
        - savename (str): 
            name under which the figure is saved
    """
 
    title = "Gabor sequence sig USIs by mouse"
    
    perc_sig_df = pd.DataFrame.from_dict(perc_sig_df)

    ax = usi_plots.plot_perc_sig_usis(
        perc_sig_df, 
        analyspar=analyspar,
        permpar=permpar,
        figpar=figpar, 
        title=title, 
        by_mouse=True,
        )
    fig = ax.reshape(-1)[0].figure
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_rel_resp_tracked_rois_sess123(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_gabor_tracked_roi_means_sess123_by_mouse(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_visual_flow_sequences_sess123(analyspar, sesspar, stimpar, basepar, 
                                       extrapar, trace_df, figpar):
    """
    plot_visual_flow_sequences_sess123(analyspar, sesspar, stimpar, basepar, 
                                       extrapar, trace_df, figpar)

    From dictionaries, plots visual flow sequences across sessions. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["seed"] (int): seed
        - trace_df (pd.DataFrame):
            dataframe with a row for each session, and the following 
            columns, in addition to the basic sess_df columns: 
            - trace_stats (list): 
                trace stats (split x frames x stat (me, err))
            - time_values (list):
                values for each frame, in seconds

        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  

    Returns:
        - fulldir (Path): final path of the directory in which the figure 
                          is saved
        - savename (str): name under which the figure is saved
    """

    title = "Visual flow sequences"
    
    trace_df = pd.DataFrame.from_dict(trace_df)

    ax = seq_plots.plot_sess_traces(
        trace_df, 
        analyspar=analyspar, 
        sesspar=sesspar,
        stimpar=stimpar, 
        figpar=figpar, 
        trace_col="trace_stats",
        row_col="sess_ns",
        title=title, 
        lock='unexp_lock',
        size="reg"
        )
    fig = ax.reshape(-1)[0].figure
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_visual_flow_diffs_sess123(analyspar, sesspar, stimpar, basepar, 
                                   permpar, extrapar, diffs_df, figpar):
    """
    plot_visual_flow_diffs_sess123(analyspar, sesspar, stimpar, basepar, 
                                   permpar, extrapar, diffs_df, figpar)

    From dictionaries, plots visual flow sequences across sessions. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["seed"] (int): seed
        - diffs_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - diff_stats (list): split difference stats (me, err)
            - null_CIs (list): adjusted null CI for split differences
            - raw_p_vals (float): unadjusted p-value for differences within 
                sessions
            - p_vals (float): p-value for differences within sessions, 
                adjusted for multiple comparisons and tails
            for session comparisons, e.g. 1v2:
            - raw_p_vals_{}v{} (float): unadjusted p-value for differences
                between sessions 
            - p_vals_{}v{} (float): p-value for differences between sessions, 
                adjusted for multiple comparisons and tails

        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  

    Returns:
        - fulldir (Path): final path of the directory in which the figure 
                          is saved
        - savename (str): name under which the figure is saved
    """

    title = "Visual flow sequence differences"
    
    diffs_df = pd.DataFrame.from_dict(diffs_df)

    # ax = seq_plots.plot_sess_diffs(
    #     diffs_df, 
    #     analyspar=analyspar, 
    #     sesspar=sesspar, 
    #     permpar=permpar, 
    #     figpar=figpar, 
    #     title=title, 
    #     wide=True
    #     )
    # fig = ax.reshape(-1)[0].figure
    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename


#############################################
def plot_visual_flow_rel_resp_sess123(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename

    
#############################################
def plot_rel_resp_stimulus_comp_sess1v3(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename

    
#############################################
def plot_visual_flow_tracked_roi_usis_sess123(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename

    
#############################################
def plot_visual_flow_tracked_roi_usi_means_sess123_by_mouse(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename

    
#############################################
def plot_tracked_roi_usis_stimulus_comp_sess1v3(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename

    
#############################################
def plot_visual_flow_norm_res_corrs_sess123_comps(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename

    
#############################################
def plot_dendritic_roi_tracking_example(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename

    
#############################################
def plot_somatic_roi_tracking_example(figpar, **kwargs):
    """
    """
 
    title = ""
    
    # df = pd.DataFrame.from_dict(df)


    fig = None
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename
