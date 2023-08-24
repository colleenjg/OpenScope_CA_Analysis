"""
plot_figs.py

This script contains functions defining figure panel plotting.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import numpy as np
import pandas as pd

from util import plot_util
from paper_fig_util import helper_fcts
from plot_fcts import behav_plots, corr_plots, misc_plots, seq_plots, \
    stim_plots, roi_plots, usi_plots


#############################################
def plot_imaging_planes(sesspar, extrapar, imaging_plane_df, figpar):
    """
    plot_imaging_planes(sesspar, extrapar, imaging_plane_df, figpar)

    From dictionaries, plots imaging planes.
    
    Returns figure name and save directory path.
    
    Required args:
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
        - imaging_plane_df (pd.DataFrame in dict format):
            dataframe with a row for each mouse, and the following 
            columns, in addition to the basic sess_df columns: 
            - "max_projections" (list): pixel intensities of maximum projection 
                for the plane (hei x wid)
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

    title = "Imaging plane examples"
    
    imaging_plane_df = pd.DataFrame.from_dict(imaging_plane_df)

    ax = roi_plots.plot_imaging_planes(
        imaging_plane_df, 
        figpar, 
        title=title
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
def plot_roi_tracking(analyspar, sesspar, extrapar, roi_mask_df, figpar):
    """
    plot_roi_tracking(analyspar, sesspar, extrapar, roi_mask_df, figpar)

    From dictionaries, plots tracked ROI masks.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
        - roi_mask_df (pd.DataFrame in dict format):
            dataframe with a row for each mouse, and the following 
            columns, in addition to the basic sess_df columns: 

            - "max_projections" (list): pixel intensities of maximum projection 
                for the plane (hei x wid)
            - "registered_roi_mask_idxs" (list): list of mask indices, 
                registered across sessions, for each session 
                (flattened across ROIs) ((sess, hei, wid) x val)
            - "roi_mask_idxs" (list): list of mask indices for each session, 
                and each ROI (sess x ((ROI, hei, wid) x val)) (not registered)
            - "roi_mask_shapes" (list): shape into which ROI mask indices index 
                (sess x hei x wid)

            - "crop_fact" (num): factor by which to crop masks (> 1) 
            - "shift_prop_hei" (float): proportion by which to shift cropped 
                mask center vertically from left edge [0, 1]
            - "shift_prop_wid" (float): proportion by which to shift cropped 
                mask center horizontally from left edge [0, 1]

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

    title = "Tracked ROI examples"
    
    roi_mask_df = pd.DataFrame.from_dict(roi_mask_df)

    ax = roi_plots.plot_roi_masks_overlayed_with_proj(
        roi_mask_df, 
        figpar, 
        title=title
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
def plot_gabor_example_roi_usis(analyspar, sesspar, stimpar, basepar, idxpar, 
                                permpar, extrapar, chosen_rois_df, figpar):
    """
    plot_gabor_example_roi_usis(analyspar, sesspar, stimpar, basepar, idxpar, 
                                permpar, extrapar, chosen_rois_info, figpar)

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
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
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
        figpar=figpar, 
        trace_col="roi_trace_stats",
        row_col="target_idxs",
        row_order=sorted_target_idxs, 
        split=idxpar["feature"],
        title=title, 
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
                                   idxpar, permpar, extrapar, ex_idx_df, 
                                   figpar):
    """
    plot_gabor_example_roi_usi_sig(analyspar, sesspar, stimpar, basepar, 
                                   idxpar, permpar, extrapar, ex_idx_df, 
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
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
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
        ex_idx_df, 
        sesspar=sesspar, 
        permpar=permpar, 
        figpar=figpar, 
        title=title
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
def plot_gabor_roi_usi_distr(analyspar, sesspar, stimpar, basepar, idxpar, 
                             permpar, extrapar, idx_df, figpar):
    """
    plot_gabor_roi_usi_distr(analyspar, sesspar, stimpar, basepar, idxpar, 
                             permpar, extrapar, idx_df, figpar)

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
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["n_bins"] (int): n_bins
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

    ax = usi_plots.plot_idxs(
        idx_df, 
        sesspar=sesspar, 
        figpar=figpar, 
        plot="percs", 
        density=True, 
        n_bins=extrapar["n_bins"],
        title=title, 
        size="small", 
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
def plot_gabor_roi_usi_sig(analyspar, sesspar, stimpar, basepar, idxpar, 
                           permpar, extrapar, perc_sig_df, figpar):
    """
    plot_gabor_roi_usi_sig(analyspar, sesspar, stimpar, basepar, idxpar, 
                           permpar, extrapar, perc_sig_df, figpar)

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
        - idxpar (IdxPar): 
            named tuple containing index parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["common_oris"] (bool): whether common orientations are used
            ["by_mouse"] (bool): whether data is by mouse
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
            - perc_pos_idxs_raw_p_vals (num): uncorrected p-value for percent 
                pos. ROIs
            - perc_pos_idxs_p_vals (num): p-value for percent pos. ROIs, 
                corrected for multiple comparisons and tails

            for sig in ["lo", "hi"]: for low vs high ROI indices
            - perc_sig_{sig}_idxs (num): percent significant ROIs (0-100)
            - perc_sig_{sig}_idxs_stds (num): bootstrapped standard deviation 
                over percent significant ROIs
            - perc_sig_{sig}_idxs_CIs (list): adjusted CI for percent sig. ROIs 
            - perc_sig_{sig}_idxs_null_CIs (list): adjusted null CI for percent 
                sig. ROIs
            - perc_sig_{sig}_idxs_raw_p_vals (num): uncorrected p-value for 
                percent sig. ROIs
            - perc_sig_{sig}_idxs_p_vals (num): p-value for percent sig. 
                ROIs, corrected for multiple comparisons and tails

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

    title = "Gabor sequence sig USIs"
    if extrapar["common_oris"]:
        title = f"{title} (common oris)".replace("sequence", "seq")
    
    perc_sig_df = pd.DataFrame.from_dict(perc_sig_df)

    ax = usi_plots.plot_perc_sig_usis(
        perc_sig_df, 
        analyspar=analyspar,
        permpar=permpar,
        figpar=figpar, 
        by_mouse=extrapar["by_mouse"],
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

    fulldir, savename = plot_gabor_roi_usi_sig(**kwargs)

    return fulldir, savename


#############################################
def plot_pupil_run_responses(analyspar, sesspar, stimpar, basepar, extrapar, 
                             trace_df, figpar):
    """
    plot_pupil_run_responses(analyspar, sesspar, stimpar, basepar, extrapar, 
                             trace_df, figpar)

    From dictionaries, plots running and pupil responses.
    
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
            ["split"] (str): data split
        - trace_df (pd.DataFrame):
            dataframe with one row per session number, and the following 
            columns, in addition to the basic sess_df columns: 
            - run_trace_stats (list): 
                running velocity trace stats (split x frames x stats (me, err))
            - run_time_values (list):
                values for each frame, in seconds
                (only 0 to stimpar.post, unless split is "by_exp")
            - pupil_trace_stats (list): 
                pupil diameter trace stats (split x frames x stats (me, err))
            - pupil_time_values (list):
                values for each frame, in seconds
                (only 0 to stimpar.post, unless split is "by_exp")    

        - figpar (dict): 
            dictionary containing figure parameters
    """

    title = "Running and pupil responses"
    
    trace_df = pd.DataFrame.from_dict(trace_df)

    ax = behav_plots.plot_pupil_run_trace_stats(
        trace_df, 
        analyspar=analyspar,
        figpar=figpar, 
        split=extrapar["split"],
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
def plot_pupil_run_block_diffs(analyspar, sesspar, stimpar, permpar, extrapar, 
                               block_df, figpar):
    """
    plot_pupil_run_block_diffs(analyspar, sesspar, stimpar, permpar, extrapar, 
                               block_df, figpar)

    From dictionaries, plots running and pupil block response differences.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["seed"] (int): seed
        - block_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - run_block_diffs (list): 
                running velocity differences per block
            - run_raw_p_vals (float):
                uncorrected p-value for differences within sessions
            - run_p_vals (float):
                p-value for differences within sessions, 
                corrected for multiple comparisons and tails
            - pupil_block_diffs (list): 
                for pupil diameter differences per block
            - pupil_raw_p_vals (list):
                uncorrected p-value for differences within sessions
            - pupil_p_vals (list):
                p-value for differences within sessions, 
                corrected for multiple comparisons and tails

        - figpar (dict): 
            dictionary containing figure parameters
    """

    title = "Running and pupil block response differences"
    
    block_df = pd.DataFrame.from_dict(block_df)

    ax = behav_plots.plot_pupil_run_block_diffs(
        block_df, 
        analyspar=analyspar,
        permpar=permpar,
        figpar=figpar, 
        title=title,
        seed=extrapar["seed"],
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
def plot_pupil_run_full(analyspar, sesspar, extrapar, sess_df, figpar):
    """
    plot_pupil_run_full(analyspar, sesspar, extrapar, sess_df, figpar)

    From dictionaries, plots running and pupil responses for a full session.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - extrapar (dict): 
            dictionary containing additional analysis parameters
        - sess_df (pd.DataFrame):
            dataframe with one row per session, and the following columns, in 
            addition to the basic sess_df columns:
            - duration_sec (float):
                duration of the session in seconds
            - pup_data (list):
                pupil data
            - pup_frames (list):
                start and stop pupil frame numbers for each stimulus type
            - run_data (list):
                running velocity data
            - run_frames (list):
                start and stop running frame numbers for each stimulus type
            - stims (list):
                stimulus types

        - figpar (dict): 
            dictionary containing figure parameters
    """

    title = "Full session running and pupil responses"
    
    sess_df = pd.DataFrame.from_dict(sess_df)

    ax = behav_plots.plot_pupil_run_full(
        sess_df, 
        analyspar=analyspar,
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
def plot_pupil_run_histograms(analyspar, sesspar, extrapar, hist_df, figpar):
    """
    plot_pupil_run_histograms(analyspar, sesspar, extrapar, hist_df, figpar)

    From dictionaries, plots running and pupil histograms across sessions.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - extrapar (dict): 
            dictionary containing additional analysis parameters
        - hist_df (pd.DataFrame):
            dataframe with one row per session, and the following columns, in 
            addition to the basic sess_df columns:
            - pupil_bin_edges (float):
                bin edges for the pupil diameter data
            - pupil_vals_binned (list):
                pupil diameter values by bin, for each mouse
            - run_bin_edges (float):
                bin edges for the running data
            - run_vals_binned (list):
                run values by bin, for each mouse

        - figpar (dict): 
            dictionary containing figure parameters
    """

    title = "Running and pupil histograms"
    
    hist_df = pd.DataFrame.from_dict(hist_df)

    ax = behav_plots.plot_pupil_run_histograms(
        hist_df, 
        analyspar=analyspar,
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
def plot_gabor_sequences_sess123(analyspar, sesspar, stimpar, basepar, 
                                 extrapar, trace_df, figpar):
    """
    plot_gabor_sequences_sess123(analyspar, sesspar, stimpar, basepar, 
                                 extrapar, trace_df, figpar)

    From dictionaries, plots Gabor sequences across sessions. 
    
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
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["split"] (str): data split
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

    title = "Gabor sequences"
    
    trace_df = pd.DataFrame.from_dict(trace_df)

    ax = seq_plots.plot_sess_traces(
        trace_df, 
        analyspar=analyspar, 
        sesspar=sesspar,
        figpar=figpar, 
        trace_col="trace_stats",
        row_col="sess_ns",
        split=extrapar["split"],
        title=title, 
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
def plot_gabor_sequence_diffs_sess123(analyspar, sesspar, stimpar, basepar, 
                                      permpar, extrapar, diffs_df, figpar):
    """
    plot_gabor_sequence_diffs_sess123(analyspar, sesspar, stimpar, basepar, 
                                      permpar, extrapar, diffs_df, figpar)

    From dictionaries, plots Gabor sequence differences across sessions. 
    
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
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["split"] (str): data split
            ["seed"] (int): seed
        - diffs_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - diff_stats (list): split difference stats (me, err)
            - null_CIs (list): adjusted null CI for split differences
            - raw_p_vals (float): uncorrected p-value for differences within 
                sessions
            - p_vals (float): p-value for differences within sessions, 
                corrected for multiple comparisons and tails
            for session comparisons, e.g. 1v2:
            - raw_p_vals_{}v{} (float): uncorrected p-value for differences
                between sessions 
            - p_vals_{}v{} (float): p-value for differences between sessions, 
                corrected for multiple comparisons and tails

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

    title = "Gabor sequence differences"
    
    diffs_df = pd.DataFrame.from_dict(diffs_df)

    ax = seq_plots.plot_sess_data(
        diffs_df, 
        analyspar=analyspar, 
        sesspar=sesspar, 
        permpar=permpar, 
        figpar=figpar, 
        title=title, 
        wide=True
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
def plot_gabor_sequences_early_late_sess123(analyspar, sesspar, stimpar, 
                                            basepar, extrapar, trace_df, 
                                            figpar):
    """
    plot_gabor_sequences_early_late_sess123(analyspar, sesspar, stimpar, 
                                            basepar, extrapar, trace_df, 
                                            figpar)

    From dictionaries, plots Gabor sequences early and late across sessions. 
    
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
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["split"] (str): data split
            ["thirds"] (bool): if True, data is split into early and late thirds
        - trace_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - third_0_trace_stats (list): 
                ROI trace stats for each third 
                (split x frames x stat (me, err))
            - third_2_trace_stats (list): 
                ROI trace stats for each third 
                (split x frames x stat (me, err))
            - time_values (list):
                values for each frame, in seconds
              (only 0 to stimpar.post, unless split is "by_exp")

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

    title = "Gabor sequences (early and late)"
    
    trace_df = pd.DataFrame.from_dict(trace_df)

    ax = seq_plots.plot_early_late_sess_traces(
        trace_df, 
        analyspar=analyspar, 
        sesspar=sesspar,
        figpar=figpar, 
        split=extrapar["split"],
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
def plot_gabor_sequence_diffs_early_late_sess123(analyspar, sesspar, stimpar, 
                                                 basepar, permpar, extrapar, 
                                                 diffs_df, figpar):
    """
    plot_gabor_sequence_diffs_early_late_sess123(analyspar, sesspar, stimpar, 
                                                 basepar, permpar, extrapar, 
                                                 diffs_df, figpar)

    From dictionaries, plots Gabor sequence differences early and late, for 
    each session. 
    
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
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["split"] (str): data split
            ["seed"] (int): seed
            ["thirds"] (bool): if True, data is split into early and late thirds
        - diffs_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - third_{third}_diff_stats (list): split difference stats (me, err)
            for early vs late comparisons, e.g. 0v1:
            - raw_p_vals_{}v{} (float): uncorrected p-value for differences
                between sessions 
            - p_vals_{}v{} (float): p-value for differences between sessions, 
                corrected for multiple comparisons and tails

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

    title = "Gabor sequence differences (early vs late)"
    
    diffs_df = pd.DataFrame.from_dict(diffs_df)

    ax = seq_plots.plot_early_late_sess_data(
        diffs_df, 
        analyspar=analyspar, 
        sesspar=sesspar, 
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
def plot_gabor_rel_resp_sess123(analyspar, sesspar, stimpar, permpar, extrapar, 
                                rel_resp_df, figpar):
    """
    plot_gabor_rel_resp_sess123(analyspar, sesspar, stimpar, permpar, extrapar, 
                                rel_resp_df, figpar)

    From dictionaries, plots regular and unexpected Gabor responses, relative 
    to session 1. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["rel_sess"] (int): base session
            ["seed"] (int): seed
        - rel_resp_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - rel_reg (list): data stats for regular data (me, err)
            - rel_unexp (list): data stats for unexpected data (me, err)
            for reg/exp/unexp data types, session comparisons, e.g. 1v2:
            - {data_type}_raw_p_vals_{}v{} (float): uncorrected p-value for 
                data differences between sessions 
            - {data_type}_p_vals_{}v{} (float): p-value for data between 
                sessions, corrected for multiple comparisons and tails

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

    title = "Gabor ABC vs UG, relative to session 1"
    
    rel_resp_df = pd.DataFrame.from_dict(rel_resp_df)

    ax = seq_plots.plot_rel_resp_data(
        rel_resp_df, 
        analyspar=analyspar, 
        sesspar=sesspar, 
        stimpar=stimpar, 
        permpar=permpar, 
        figpar=figpar, 
        title=title, 
        small=False,
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
def plot_gabor_tracked_roi_usis_sess123(analyspar, sesspar, stimpar, basepar, 
                                        idxpar, extrapar, idx_only_df, figpar):
    """
    plot_gabor_tracked_roi_usis_sess123(analyspar, sesspar, stimpar, basepar, 
                                        idxpar, extrapar, idx_only_df, figpar)

    From dictionaries, plots tracked Gabor USIs across sessions. 
    
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
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
        - idx_only_df (pd.DataFrame):
            dataframe with one row per session, and the following columns, in 
            addition to the basic sess_df columns:
            - roi_idxs (list): index for each ROI

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
 

    title = "Tracked Gabor USIs"

    idx_only_df = pd.DataFrame.from_dict(idx_only_df)
    
    ax = usi_plots.plot_tracked_idxs(
        idx_only_df, 
        sesspar=sesspar, 
        figpar=figpar, 
        title=title, 
        wide=True
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
def plot_gabor_tracked_roi_abs_usi_means_sess123(
        analyspar, sesspar, stimpar, basepar, idxpar, permpar, extrapar, 
        idx_stats_df, figpar):
    """
    plot_gabor_tracked_roi_abs_usi_means_sess123(
        analyspar, sesspar, stimpar, basepar, idxpar, permpar, extrapar, 
        idx_stats_df, figpar)

    From dictionaries, plots statistics for tracked Gabor USIs across sessions. 
    
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
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["absolute"] (bool): whether statistics are for absolute USIs
            ["by_mouse"] (bool): whether data is by mouse
            ["seed"] (int): seed
        - idx_stats_df (pd.DataFrame):
            dataframe with one row per (mouse/)session/line/plane, and the 
            following columns, in addition to the basic sess_df columns:
            - roi_idxs or abs_roi_idxs (list): 
                USI statistics or absolute USI statistics
            for session comparisons, e.g. 1v2 (if permpar is not None):
            - raw_p_vals_{}v{} (float): uncorrected p-value for differences
                between sessions 
            - p_vals_{}v{} (float): p-value for differences between sessions, 
                corrected for multiple comparisons and tails

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

    title = "Means of absolute tracked Gabor USIs"

    idx_stats_df = pd.DataFrame.from_dict(idx_stats_df)
    
    ax = usi_plots.plot_tracked_idx_stats(
        idx_stats_df, 
        sesspar=sesspar, 
        figpar=figpar, 
        permpar=permpar,
        absolute=extrapar["absolute"],
        by_mouse=extrapar["by_mouse"],
        title=title, 
        wide=True
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
def plot_gabor_tracked_roi_usi_variances_sess123(
        analyspar, sesspar, stimpar, basepar, idxpar, permpar, extrapar, 
        idx_stats_df, figpar):
    """
    plot_gabor_tracked_roi_usi_variances_sess123(
        analyspar, sesspar, stimpar, basepar, idxpar, permpar, extrapar, 
        idx_stats_df, figpar)

    From dictionaries, plots variances for tracked Gabor USIs across sessions. 
    
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
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["absolute"] (bool): whether statistics are for absolute USIs
            ["by_mouse"] (bool): whether data is by mouse
            ["seed"] (int): seed
            ["stat"] (str): statistic used
        - idx_stats_df (pd.DataFrame):
            dataframe with one row per (mouse/)session/line/plane, and the 
            following columns, in addition to the basic sess_df columns:
            - roi_idxs or abs_roi_idxs (list): 
                USI statistics or absolute USI statistics
            for session comparisons, e.g. 1v2 (if permpar is not None):
            - raw_p_vals_{}v{} (float): uncorrected p-value for differences
                between sessions 
            - p_vals_{}v{} (float): p-value for differences between sessions, 
                corrected for multiple comparisons and tails

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

    title = "Variances of tracked Gabor USIs"

    idx_stats_df = pd.DataFrame.from_dict(idx_stats_df)
    
    ax = usi_plots.plot_tracked_idx_stats(
        idx_stats_df, 
        sesspar=sesspar, 
        figpar=figpar, 
        permpar=permpar,
        absolute=extrapar["absolute"],
        by_mouse=extrapar["by_mouse"],
        bootstr_err=True,
        title=title, 
        wide=True
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
def plot_gabor_Aori_decoding_sess123(analyspar, sesspar, stimpar, logregpar, 
                                     permpar, extrapar, scores_df, figpar):
    """
    plot_gabor_Aori_decoding_sess123(analyspar, sesspar, stimpar, logregpar, 
                                     permpar, extrapar, scores_df, figpar)

    From dictionaries, plots Gabor A orientation decoding scores across 
    sessions. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - logregpar (dict): 
            dictionary with keys of LogRegPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["n_splits"] (int): number of data splits
            ["seed"] (int): seed
        - scores_df (pd.DataFrame):
            dataframe with logistic regression score statistics, shuffled score 
            confidence intervals, and test set p-values for each 
            line/plane/session, in addition to the basic sess_df columns

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
 
    comp_str = logregpar["comp"].replace("ori", " orientation")
    title = f"{comp_str} decoding (test set)"

    scores_df = pd.DataFrame.from_dict(scores_df)

    ax = misc_plots.plot_decoder_data(
        scores_df, 
        analyspar=analyspar, 
        sesspar=sesspar, 
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
def plot_gabor_Bori_decoding_sess123(analyspar, sesspar, stimpar, logregpar, 
                                     permpar, extrapar, scores_df, figpar):
    """
    plot_gabor_Bori_decoding_sess123(analyspar, sesspar, stimpar, logregpar, 
                                     permpar, extrapar, scores_df, figpar)

    From dictionaries, plots Gabor B orientation decoding scores across 
    sessions. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - logregpar (dict): 
            dictionary with keys of LogRegPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["n_splits"] (int): number of data splits
            ["seed"] (int): seed
        - scores_df (pd.DataFrame):
            dataframe with logistic regression score statistics, shuffled score 
            confidence intervals, and test set p-values for each 
            line/plane/session, in addition to the basic sess_df columns

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
 
    comp_str = logregpar["comp"].replace("ori", " orientation")
    title = f"{comp_str} decoding (test set)"

    scores_df = pd.DataFrame.from_dict(scores_df)

    ax = misc_plots.plot_decoder_data(
        scores_df, 
        analyspar=analyspar, 
        sesspar=sesspar, 
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
def plot_gabor_Cori_decoding_sess123(analyspar, sesspar, stimpar, logregpar, 
                                     permpar, extrapar, scores_df, figpar):
    """
    plot_gabor_Cori_decoding_sess123(analyspar, sesspar, stimpar, logregpar, 
                                     permpar, extrapar, scores_df, figpar)

    From dictionaries, plots Gabor C orientation decoding scores across 
    sessions. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - logregpar (dict): 
            dictionary with keys of LogRegPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["n_splits"] (int): number of data splits
            ["seed"] (int): seed
        - scores_df (pd.DataFrame):
            dataframe with logistic regression score statistics, shuffled score 
            confidence intervals, and test set p-values for each 
            line/plane/session, in addition to the basic sess_df columns

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
 
    comp_str = logregpar["comp"].replace("ori", " orientation")
    title = f"{comp_str} decoding (test set)"

    scores_df = pd.DataFrame.from_dict(scores_df)

    ax = misc_plots.plot_decoder_data(
        scores_df, 
        analyspar=analyspar, 
        sesspar=sesspar, 
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
def plot_gabor_Dori_decoding_sess123(analyspar, sesspar, stimpar, logregpar, 
                                     permpar, extrapar, scores_df, figpar):
    """
    plot_gabor_Dori_decoding_sess123(analyspar, sesspar, stimpar, logregpar, 
                                     permpar, extrapar, scores_df, figpar)

    From dictionaries, plots Gabor D orientation decoding scores across 
    sessions. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - logregpar (dict): 
            dictionary with keys of LogRegPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["n_splits"] (int): number of data splits
            ["seed"] (int): seed
        - scores_df (pd.DataFrame):
            dataframe with logistic regression score statistics, shuffled score 
            confidence intervals, and test set p-values for each 
            line/plane/session, in addition to the basic sess_df columns

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
 
    comp_str = logregpar["comp"].replace("ori", " orientation")
    title = f"{comp_str} decoding (test set)"

    scores_df = pd.DataFrame.from_dict(scores_df)

    ax = misc_plots.plot_decoder_data(
        scores_df, 
        analyspar=analyspar, 
        sesspar=sesspar, 
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
def plot_gabor_Uori_decoding_sess123(analyspar, sesspar, stimpar, logregpar, 
                                     permpar, extrapar, scores_df, figpar):
    """
    plot_gabor_Uori_decoding_sess123(analyspar, sesspar, stimpar, logregpar, 
                                     permpar, extrapar, scores_df, figpar)

    From dictionaries, plots Gabor U orientation decoding scores across 
    sessions. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - logregpar (dict): 
            dictionary with keys of LogRegPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["n_splits"] (int): number of data splits
            ["seed"] (int): seed
        - scores_dfs (pd.DataFrame):
            dataframe with logistic regression score statistics, shuffled score 
            confidence intervals, and test set p-values for each 
            line/plane/session, in addition to the basic sess_df columns

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
 
    comp_str = logregpar["comp"].replace("ori", " orientation")
    title = f"{comp_str} decoding (test set)"

    scores_df = pd.DataFrame.from_dict(scores_df)

    ax = misc_plots.plot_decoder_data(
        scores_df, 
        analyspar=analyspar, 
        sesspar=sesspar, 
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
def plot_gabor_timecourse_decoding_sess1(analyspar, sesspar, stimpar, logregpar, 
                                         permpar, extrapar, scores_df, figpar):
    """
    plot_gabor_timecourse_decoding_sess1(analyspar, sesspar, stimpar, logregpar, 
                                         permpar, extrapar, scores_df, figpar)

    From dictionaries, plots timecourse orientation decoding scores for 
    session 1. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - logregpar (dict): 
            dictionary with keys of LogRegPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["n_splits"] (int): number of data splits
            ["seed"] (int): seed
        - scores_df (pd.DataFrame):
            dataframe with logistic regression score statistics over time, 
            shuffled score confidence intervals, and test set p-values for each 
            line/plane/session, in addition to the basic sess_df columns

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

    title = "Session 1 orientation decoding (test set)"

    scores_df = pd.DataFrame.from_dict(scores_df)

    ax = misc_plots.plot_decoder_timecourse_data(
        scores_df, 
        analyspar=analyspar, 
        sesspar=sesspar, 
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
def plot_gabor_timecourse_decoding_sess2(analyspar, sesspar, stimpar, logregpar, 
                                         permpar, extrapar, scores_df, figpar):
    """
    plot_gabor_timecourse_decoding_sess2(analyspar, sesspar, stimpar, logregpar, 
                                         permpar, extrapar, scores_df, figpar)

    From dictionaries, plots timecourse orientation decoding scores for 
    session 2. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - logregpar (dict): 
            dictionary with keys of LogRegPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["n_splits"] (int): number of data splits
            ["seed"] (int): seed
        - scores_df (pd.DataFrame):
            dataframe with logistic regression score statistics over time, 
            shuffled score confidence intervals, and test set p-values for each 
            line/plane/session, in addition to the basic sess_df columns

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

    title = "Session 2 orientation decoding (test set)"

    scores_df = pd.DataFrame.from_dict(scores_df)

    ax = misc_plots.plot_decoder_timecourse_data(
        scores_df, 
        analyspar=analyspar, 
        sesspar=sesspar, 
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
def plot_gabor_timecourse_decoding_sess3(analyspar, sesspar, stimpar, logregpar, 
                                         permpar, extrapar, scores_df, figpar):
    """
    plot_gabor_timecourse_decoding_sess3(analyspar, sesspar, stimpar, logregpar, 
                                         permpar, extrapar, scores_df, figpar)

    From dictionaries, plots timecourse orientation decoding scores for 
    session 3. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - logregpar (dict): 
            dictionary with keys of LogRegPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["n_splits"] (int): number of data splits
            ["seed"] (int): seed
        - scores_df (pd.DataFrame):
            dataframe with logistic regression score statistics over time, 
            shuffled score confidence intervals, and test set p-values for each 
            line/plane/session, in addition to the basic sess_df columns

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

    title = "Session 3 orientation decoding (test set)"

    scores_df = pd.DataFrame.from_dict(scores_df)

    ax = misc_plots.plot_decoder_timecourse_data(
        scores_df, 
        analyspar=analyspar, 
        sesspar=sesspar, 
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
def plot_gabor_corrs_sess123_comps(analyspar, sesspar, stimpar, basepar, 
                                   idxpar, permpar, extrapar, idx_corr_df, 
                                   figpar):
    """
    plot_gabor_corrs_sess123_comps(analyspar, sesspar, stimpar, basepar, 
                                   idxpar, permpar, extrapar, idx_corr_df, 
                                   figpar))

    From dictionaries, plots ROI USI correlations across sessions for tracked 
    Gabor USIs. 
    
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
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["consec_only"] (bool): if True, only consecutive sessions are  
                correlated
            ["corr_type"] (str): type of correlation ("corr" or "R_sqr")
            ["permute"] (str): type of permutations used 
                ("sess", "tracking", "all")
            ["seed"] (int): seed
            ["sig_only"] (bool): if True, only ROIs with significant USIs are 
                retained
        - idx_corr_df (pd.DataFrame):
            dataframe with one row per line/plane, and the 
            following columns, in addition to the basic sess_df columns:

            for session comparisons, e.g. 1v2
            - {}v{}{norm_str}_corrs (float): intersession ROI index correlations
            - {}v{}{norm_str}_corr_stds (float): bootstrapped intersession ROI 
                index correlation standard deviation
            - {}v{}_null_CIs (list): adjusted null CI for intersession ROI 
                index correlations
            - {}v{}_raw_p_vals (float): p-value for intersession correlations
            - {}v{}_p_vals (float): p-value for intersession correlations, 
                corrected for multiple comparisons and tails

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
 
    title = "Correlations for Gabor USIs"
    
    idx_corr_df = pd.DataFrame.from_dict(idx_corr_df)
    
    ax = corr_plots.plot_idx_correlations(
        idx_corr_df, 
        permpar=permpar,
        figpar=figpar, 
        permute=extrapar["permute"],
        corr_type=extrapar["corr_type"], 
        title=title,
        small=True,
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
def plot_roi_overlays_sess123(analyspar, sesspar, extrapar, roi_mask_df, figpar):
    """
    plot_roi_overlays_sess123(analyspar, sesspar, extrapar, roi_mask_df, figpar)

    From dictionaries, plots tracked ROI masks.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            - "mark_crop_only" (bool): if True, crop information is used to 
                mark images, but not to crop
        - roi_mask_df (pd.DataFrame in dict format):
            dataframe with a row for each mouse, and the following 
            columns, in addition to the basic sess_df columns: 
            - "registered_roi_mask_idxs" (list): list of mask indices, 
                registered across sessions, for each session 
                (flattened across ROIs) ((sess, hei, wid) x val)
            - "roi_mask_shapes" (list): shape into which ROI mask indices index 
                (sess x hei x wid)
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

    title = "Tracked ROI examples (full)"
    
    roi_mask_df = pd.DataFrame.from_dict(roi_mask_df)

    ax = roi_plots.plot_roi_masks_overlayed(
        roi_mask_df, 
        figpar=figpar, 
        title=title,
        mark_crop_only=extrapar["mark_crop_only"],
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
def plot_roi_overlays_sess123_enlarged(analyspar, sesspar, extrapar, 
                                       roi_mask_df, figpar):
    """
    plot_roi_overlays_sess123_enlarged(analyspar, sesspar, extrapar, 
                                       roi_mask_df, figpar)

    From dictionaries, plots enlarged tracked ROI masks.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
        - roi_mask_df (pd.DataFrame in dict format):
            dataframe with a row for each mouse, and the following 
            columns, in addition to the basic sess_df columns: 
            - "registered_roi_mask_idxs" (list): list of mask indices, 
                registered across sessions, for each session 
                (flattened across ROIs) ((sess, hei, wid) x val)
            - "roi_mask_shapes" (list): shape into which ROI mask indices index 
                (sess x hei x wid)
            - "crop_fact" (num): factor by which to crop masks (> 1) 
            - "shift_prop_hei" (float): proportion by which to shift cropped 
                mask center vertically from left edge [0, 1]
            - "shift_prop_wid" (float): proportion by which to shift cropped 
                mask center horizontally from left edge [0, 1]

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

    title = "Tracked ROI examples (enlarged)"
    
    roi_mask_df = pd.DataFrame.from_dict(roi_mask_df)

    ax = roi_plots.plot_roi_masks_overlayed(
        roi_mask_df, 
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

    title = "ROI SNRs"
    snr_df = pd.DataFrame.from_dict(snr_df)
    ax = misc_plots.plot_snr_sigmeans_nrois(
        snr_df, 
        figpar=figpar, 
        datatype="snrs", 
        title=title
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

    title = "Mean ROI signals"
    sig_mean_df = pd.DataFrame.from_dict(sig_mean_df)
    ax = misc_plots.plot_snr_sigmeans_nrois(
        sig_mean_df, 
        figpar=figpar, 
        datatype="signal_means", 
        title=title
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
        nrois_df, 
        figpar=figpar, 
        datatype="nrois", 
        title=title
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
def plot_roi_corr_sess123(analyspar, sesspar, extrapar, corr_df, figpar):
    """
    plot_roi_corr_sess123(analyspar, sesspar, extrapar, corr_df, figpar)
    
    Plots ROI correlations.

    Returns figure name and save directory path.

    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["rolling_win"] (int): window used in rolling mean over individual 
                traces 
        - corr_df (pd.DataFrame in dict format):
            dataframe with one row per session/line/plane, and the 
            following columns, in addition to the basic sess_df columns:
            - bin_edges (list): first and last bin edge
            - corrs_binned (list): number of correlation values per bin

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
 
    title = "ROI correlations"
    
    corr_df = pd.DataFrame.from_dict(corr_df)

    ax = misc_plots.plot_roi_correlations(
        corr_df, 
        figpar=figpar, 
        title=title
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
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - basepar (dict): 
            dictionary with keys of BasePar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["split"] (str): data split
        - trace_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - trace_stats (list): 
                trace stats (split x frames x stat (me, err))
            - time_values (list):
                values for each frame, in seconds (only 0 to stimpar.post)

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
        figpar=figpar, 
        trace_col="trace_stats",
        row_col="sess_ns",
        split=extrapar["split"],
        title=title, 
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
def plot_stimulus_offset_sess123(analyspar, sesspar, stimpar, basepar, extrapar, 
                                 trace_df, figpar):
    """
    plot_stimulus_offset_sess123(analyspar, sesspar, stimpar, basepar, extrapar, 
                                 trace_df, figpar)

    From dictionaries, plots stimulus offset sequences across sessions. 
    
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
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["split"] (str): data split
        - trace_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - trace_stats (list): 
                trace stats (split x frames x stat (me, err))
            - time_values (list):
                values for each frame, in seconds (only 0 to stimpar.post)

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

    title = "Stimulus offset sequences"
    
    trace_df = pd.DataFrame.from_dict(trace_df)

    ax = seq_plots.plot_sess_traces(
        trace_df, 
        analyspar=analyspar, 
        sesspar=sesspar,
        figpar=figpar, 
        trace_col="trace_stats",
        row_col="sess_ns",
        split=extrapar["split"],
        title=title, 
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
def plot_gabor_ex_roi_exp_responses_sess1(analyspar, sesspar, stimpar, basepar, 
                                          extrapar, ex_traces_df, figpar):

    """
    plot_gabor_ex_roi_exp_responses_sess1(analyspar, sesspar, stimpar, basepar, 
                                          extrapar, ex_traces_df, figpar)

    From dictionaries, plots example ROI responses to expected Gabor sequences. 
    
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
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["n_ex"] (int): number of example ROIs retained per line/plane
            ["rolling_win"] (int): window used in rolling mean over individual 
                trial traces 
            ["seed"] (int): seed
            ["unexp"] (int): whether sequences are expected or unexpected
        - ex_traces_df (pd.DataFrame):
            dataframe with a row for each ROI, and the following columns, 
            in addition to the basic sess_df columns: 
            - time_values (list): values for each frame, in seconds
            - roi_ns (list): selected ROI number
            - traces_sm (list): selected ROI sequence traces, smoothed, with 
                dims: seq x frames
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

    # For dataset paper, so using "consistent" instead of "expected"
    title = "Example ROI responses to consistent Gabor sequences"
    
    ex_traces_df = pd.DataFrame.from_dict(ex_traces_df)

    ax = seq_plots.plot_ex_traces(
        ex_traces_df, 
        stimpar=stimpar, 
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
def plot_gabor_ex_roi_unexp_responses_sess1(analyspar, sesspar, stimpar, 
                                            basepar, extrapar, ex_traces_df, 
                                            figpar):

    """
    plot_gabor_ex_roi_unexp_responses_sess1(analyspar, sesspar, stimpar, 
                                            basepar, extrapar, ex_traces_df, 
                                            figpar)

    From dictionaries, plots example ROI responses to unexpected Gabor 
    sequences. 
    
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
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["n_ex"] (int): number of example ROIs retained per line/plane
            ["rolling_win"] (int): window used in rolling mean over individual 
                trial traces 
            ["seed"] (int): seed
            ["unexp"] (int): whether sequences are expected or unexpected
        - ex_traces_df (pd.DataFrame):
            dataframe with a row for each ROI, and the following columns, 
            in addition to the basic sess_df columns: 
            - time_values (list): values for each frame, in seconds
            - roi_ns (list): selected ROI number
            - traces_sm (list): selected ROI sequence traces, smoothed, with 
                dims: seq x frames
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

    # For dataset paper, so using "inconsistent" instead of "unexpected"
    title = "Example ROI responses to inconsistent Gabor sequences"
    
    ex_traces_df = pd.DataFrame.from_dict(ex_traces_df)

    ax = seq_plots.plot_ex_traces(
        ex_traces_df, 
        stimpar=stimpar, 
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
def plot_visflow_ex_roi_nasal_responses_sess1(analyspar, sesspar, stimpar, 
                                              basepar, extrapar, ex_traces_df, 
                                              figpar):

    """
    plot_visflow_ex_roi_nasal_responses_sess1(analyspar, sesspar, stimpar, 
                                              basepar, extrapar, ex_traces_df, 
                                              figpar)

    From dictionaries, plots example ROI responses to onset of unexpected flow 
    during nasal (leftward) visual flow. 
    
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
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["n_ex"] (int): number of example ROIs retained per line/plane
            ["rolling_win"] (int): window used in rolling mean over individual 
                trial traces 
            ["seed"] (int): seed
        - ex_traces_df (pd.DataFrame):
            dataframe with a row for each ROI, and the following columns, 
            in addition to the basic sess_df columns: 
            - time_values (list): values for each frame, in seconds
            - roi_ns (list): selected ROI number
            - traces_sm (list): selected ROI sequence traces, smoothed, with 
                dims: seq x frames
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

    # For dataset paper, so using "inconsistent" instead of "unexpected"
    title = (
        "Example ROI responses to inconsistent flow during nasal visual flow"
    )
    
    ex_traces_df = pd.DataFrame.from_dict(ex_traces_df)

    ax = seq_plots.plot_ex_traces(
        ex_traces_df, 
        stimpar=stimpar, 
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
def plot_visflow_ex_roi_temp_responses_sess1(analyspar, sesspar, stimpar, 
                                             basepar, extrapar, ex_traces_df, 
                                             figpar):

    """
    plot_visflow_ex_roi_temp_responses_sess1(analyspar, sesspar, stimpar, 
                                             basepar, extrapar, ex_traces_df, 
                                             figpar)

    From dictionaries, plots example ROI responses to onset of unexpected flow 
    during temporal (rightward) visual flow. 
    
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
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["n_ex"] (int): number of example ROIs retained per line/plane
            ["rolling_win"] (int): window used in rolling mean over individual 
                trial traces 
            ["seed"] (int): seed
        - ex_traces_df (pd.DataFrame):
            dataframe with a row for each ROI, and the following columns, 
            in addition to the basic sess_df columns: 
            - time_values (list): values for each frame, in seconds
            - roi_ns (list): selected ROI number
            - traces_sm (list): selected ROI sequence traces, smoothed, with 
                dims: seq x frames
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

    # For dataset paper, so using "inconsistent" instead of "unexpected"
    title = (
        "Example ROI responses to inconsistent flow during temporal visual flow"
    )
    
    ex_traces_df = pd.DataFrame.from_dict(ex_traces_df)

    ax = seq_plots.plot_ex_traces(
        ex_traces_df, 
        stimpar=stimpar, 
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
def plot_gabor_roi_usi_sig_by_mouse(analyspar, sesspar, stimpar, basepar, 
                                    idxpar, permpar, extrapar, perc_sig_df, 
                                    figpar):
    """
    plot_gabor_roi_usi_sig_by_mouse(analyspar, sesspar, stimpar, basepar, 
                                    idxpar, permpar, extrapar, perc_sig_df, 
                                    figpar)

    From dictionaries, plots percentage significant Gabor USIs per mouse. 
    
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
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["by_mouse"] (bool): whether data is by mouse
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
            - perc_pos_idxs_raw_p_vals (num): uncorrected p-value for percent 
                pos. ROIs
            - perc_pos_idxs_p_vals (num): p-value for percent pos. ROIs, 
                corrected for multiple comparisons and tails

            for sig in ["lo", "hi"]: for low vs high ROI indices
            - perc_sig_{sig}_idxs (num): percent significant ROIs (0-100)
            - perc_sig_{sig}_idxs_stds (num): bootstrapped standard deviation 
                over percent significant ROIs
            - perc_sig_{sig}_idxs_CIs (list): adjusted CI for percent sig. ROIs 
            - perc_sig_{sig}_idxs_null_CIs (list): adjusted null CI for percent 
                sig. ROIs
            - perc_sig_{sig}_idxs_raw_p_vals (num): uncorrected p-value for 
                percent sig. ROIs
            - perc_sig_{sig}_idxs_p_vals (num): p-value for percent sig. 
                ROIs, corrected for multiple comparisons and tails

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
 
    title = "Gabor sequence sig USIs (by mouse)"
    
    perc_sig_df = pd.DataFrame.from_dict(perc_sig_df)

    ax = usi_plots.plot_perc_sig_usis(
        perc_sig_df, 
        analyspar=analyspar,
        permpar=permpar,
        figpar=figpar, 
        by_mouse=extrapar["by_mouse"],
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
def plot_gabor_rel_resp_tracked_rois_sess123(analyspar, sesspar, stimpar, 
                                             permpar, extrapar, rel_resp_df, 
                                             figpar):
    """
    plot_gabor_rel_resp_tracked_rois_sess123(analyspar, sesspar, stimpar, 
                                             permpar, extrapar, rel_resp_df, 
                                             figpar)

    From dictionaries, plots regular and unexpected Gabor responses, relative 
    to session 1, for tracked ROIs. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["rel_sess"]: base session
            ["seed"] (int): seed
        - rel_resp_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - rel_reg (list): data stats for regular data (me, err)
            - rel_unexp (list): data stats for unexpected data (me, err)
            for reg/exp/unexp data types, session comparisons, e.g. 1v2:
            - {data_type}_raw_p_vals_{}v{} (float): uncorrected p-value for 
                data differences between sessions 
            - {data_type}_p_vals_{}v{} (float): p-value for data between 
                sessions, corrected for multiple comparisons and tails

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

    title = "Gabor ABC vs UG, relative to session 1\n(tracked ROIs)"
    
    rel_resp_df = pd.DataFrame.from_dict(rel_resp_df)

    ax = seq_plots.plot_rel_resp_data(
        rel_resp_df, 
        analyspar=analyspar, 
        sesspar=sesspar, 
        stimpar=stimpar, 
        permpar=permpar, 
        figpar=figpar, 
        title=title, 
        small=True
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
def plot_gabor_tracked_roi_abs_usi_means_sess123_by_mouse(
        analyspar, sesspar, stimpar, basepar, idxpar, extrapar, 
        idx_stats_df, figpar):
    """
    plot_gabor_tracked_roi_abs_usi_means_sess123_by_mouse(
        analyspar, sesspar, stimpar, basepar, idxpar, extrapar, 
        idx_stats_df, figpar)

    From dictionaries, plots statistics for tracked Gabor USIs across sessions. 
    
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
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["absolute"] (bool): whether statistics are for absolute USIs
            ["by_mouse"] (bool): whether data is by mouse
            ["seed"] (int): seed
        - idx_stats_df (pd.DataFrame):
            dataframe with one row per (mouse/)session/line/plane, and the 
            following columns, in addition to the basic sess_df columns:
            - roi_idxs or abs_roi_idxs (list): 
                USI statistics or absolute USI statistics
            for session comparisons, e.g. 1v2 (if permpar is not None):
            - raw_p_vals_{}v{} (float): uncorrected p-value for differences
                between sessions 
            - p_vals_{}v{} (float): p-value for differences between sessions, 
                corrected for multiple comparisons and tails

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
 

    title = "Means of absolute tracked Gabor USIs\n(by mouse)"

    idx_stats_df = pd.DataFrame.from_dict(idx_stats_df)
    
    ax = usi_plots.plot_tracked_idx_stats(
        idx_stats_df, 
        sesspar=sesspar, 
        figpar=figpar, 
        absolute=extrapar["absolute"],
        title=title, 
        between_sess_sig=False,
        by_mouse=extrapar["by_mouse"],
        wide=False
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
def plot_visual_flow_sequences_sess123(analyspar, sesspar, stimpar, basepar, 
                                       extrapar, trace_df, figpar):
    """
    plot_visual_flow_sequences_sess123(analyspar, sesspar, stimpar, basepar, 
                                       extrapar, trace_df, figpar)

    From dictionaries, plots visual flow sequences across sessions. 
    
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
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["split"] (str): data split
        - trace_df (pd.DataFrame):
            dataframe with a row for each session, and the following 
            columns, in addition to the basic sess_df columns: 
            - trace_stats (list): 
                trace stats (split x frames x stat (me, err))
            - time_values (list):
                values for each frame, in seconds (only 0 to stimpar.post)

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
        figpar=figpar, 
        trace_col="trace_stats",
        row_col="sess_ns",
        split=extrapar["split"],
        title=title, 
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
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - basepar (dict): 
            dictionary with keys of BasePar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["split"] (str): data split
            ["seed"] (int): seed
        - diffs_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - diff_stats (list): split difference stats (me, err)
            - null_CIs (list): adjusted null CI for split differences
            - raw_p_vals (float): uncorrected p-value for differences within 
                sessions
            - p_vals (float): p-value for differences within sessions, 
                corrected for multiple comparisons and tails
            for session comparisons, e.g. 1v2:
            - raw_p_vals_{}v{} (float): uncorrected p-value for differences
                between sessions 
            - p_vals_{}v{} (float): p-value for differences between sessions, 
                corrected for multiple comparisons and tails

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

    ax = seq_plots.plot_sess_data(
        diffs_df, 
        analyspar=analyspar, 
        sesspar=sesspar, 
        permpar=permpar, 
        figpar=figpar, 
        title=title, 
        wide=False
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
def plot_visual_flow_rel_resp_sess123(analyspar, sesspar, stimpar, permpar, 
                                      extrapar, rel_resp_df, figpar):
    """
    plot_visual_flow_rel_resp_sess123(analyspar, sesspar, stimpar, permpar, 
                                      extrapar, rel_resp_df, figpar)

    From dictionaries, plots expected and unexpected visual flow responses, 
    relative to session 1. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict): 
            dictionary with keys of StimPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["rel_sess"] (int): base session
            ["seed"] (int): seed
        - rel_resp_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - rel_exp (list): data stats for expected data (me, err)
            - rel_unexp (list): data stats for unexpected data (me, err)
            for reg/exp/unexp data types, session comparisons, e.g. 1v2:
            - {data_type}_raw_p_vals_{}v{} (float): uncorrected p-value for 
                data differences between sessions 
            - {data_type}_p_vals_{}v{} (float): p-value for data between 
                sessions, corrected for multiple comparisons and tails

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
 
    # For analysis paper, so using "uniform" and "counter-flow" instead of "expected" and "unexpected flow"
    title = "Uniform flow vs counter-flow, relative to session 1"
    
    rel_resp_df = pd.DataFrame.from_dict(rel_resp_df)

    ax = seq_plots.plot_rel_resp_data(
        rel_resp_df, 
        analyspar=analyspar, 
        sesspar=sesspar, 
        stimpar=stimpar, 
        permpar=permpar, 
        figpar=figpar, 
        title=title, 
        small=True,
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
def plot_unexp_resp_stimulus_comp_sess1v3(analyspar, sesspar, stimpar, permpar, 
                                          extrapar, unexp_comp_df, figpar):
    """
    plot_unexp_resp_stimulus_comp_sess1v3(analyspar, sesspar, stimpar, permpar, 
                                          extrapar, unexp_comp_df, figpar)

    From dictionaries, plots statistics comparing changes across session in 
    tracked ROI unexpected responses for different stimuli. 
    
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
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["comp_sess"] (list): sessions for which absolute fractional 
                changes were calculated, with [x, y] => |(y - x) / x|
            ["datatype"] (str): type of data to retrieved
            ["pop_stats"] (bool): whether population stats were analysed, 
                instead of individual ROI data
            ["seed"] (int): seed
        - unexp_comp_df (pd.DataFrame):
            dataframe with one row per line/plane and one for all line/planes 
            together, and the basic sess_df columns, in addition to, 
            for each stimtype:
            - stimtype (list): absolute fractional change statistics (me, err)
            - raw_p_vals (float): uncorrected p-value for data differences 
                between stimulus types 
            - p_vals (float): p-value for data differences between stimulus 
                types, corrected for multiple comparisons and tails
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
 
    # For analysis paper, so using "pattern-violating" instead of "unexpected"
    title = "Pattern-viol. resp. changes from session 1 to 3"
    
    unexp_comp_df = pd.DataFrame.from_dict(unexp_comp_df)

    ax = stim_plots.plot_stim_data_df(
        unexp_comp_df, 
        stimpar=stimpar,
        permpar=permpar, 
        figpar=figpar, 
        pop_stats=extrapar["pop_stats"],
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
def plot_visual_flow_tracked_roi_usis_sess123(analyspar, sesspar, stimpar, 
                                              basepar, idxpar, extrapar, 
                                              idx_only_df, figpar):
    """
    plot_visual_flow_tracked_roi_usis_sess123(analyspar, sesspar, stimpar, 
                                              basepar, idxpar, extrapar, 
                                              idx_only_df, figpar)

    From dictionaries, plots tracked visual flow USIs across sessions. 
    
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
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
        - idx_only_df (pd.DataFrame):
            dataframe with one row per session, and the following columns, in 
            addition to the basic sess_df columns:
            - roi_idxs (list): index for each ROI

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
 

    title = "Tracked visual flow USIs"

    idx_only_df = pd.DataFrame.from_dict(idx_only_df)
    
    ax = usi_plots.plot_tracked_idxs(
        idx_only_df, 
        sesspar=sesspar, 
        figpar=figpar, 
        title=title, 
        wide=False
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
def plot_visual_flow_tracked_roi_abs_usi_means_sess123(
        analyspar, sesspar, stimpar, basepar, idxpar, permpar, extrapar, 
        idx_stats_df, figpar):
    """
    plot_gabor_tracked_roi_abs_usi_means_sess123(
        analyspar, sesspar, stimpar, basepar, idxpar, permpar, extrapar, 
        idx_stats_df, figpar)

    From dictionaries, plots statistics for tracked Gabor USIs across sessions. 
    
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
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["absolute"] (bool): whether statistics are for absolute USIs
            ["by_mouse"] (bool): whether data is by mouse
            ["seed"] (int): seed
        - idx_stats_df (pd.DataFrame):
            dataframe with one row per (mouse/)session/line/plane, and the 
            following columns, in addition to the basic sess_df columns:
            - roi_idxs or abs_roi_idxs (list): 
                USI statistics or absolute USI statistics
            for session comparisons, e.g. 1v2 (if permpar is not None):
            - raw_p_vals_{}v{} (float): uncorrected p-value for differences
                between sessions 
            - p_vals_{}v{} (float): p-value for differences between sessions, 
                corrected for multiple comparisons and tails

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
 

    title = "Means of absolute tracked visual flow USIs"

    idx_stats_df = pd.DataFrame.from_dict(idx_stats_df)
    
    ax = usi_plots.plot_tracked_idx_stats(
        idx_stats_df, 
        sesspar=sesspar, 
        figpar=figpar, 
        permpar=permpar,
        absolute=extrapar["absolute"],
        by_mouse=extrapar["by_mouse"],
        title=title, 
        wide=False
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
def plot_tracked_roi_usis_stimulus_comp_sess1v3(analyspar, sesspar, stimpar, 
                                                basepar, idxpar, permpar, 
                                                extrapar, usi_comp_df, figpar):
    """
    plot_tracked_roi_usis_stimulus_comp_sess1v3(analyspar, sesspar, stimpar, 
                                                basepar, idxpar, permpar, 
                                                extrapar, usi_comp_df, figpar)

    From dictionaries, plots statistics comparing changes across session in 
    tracked USIs for different stimuli. 
    
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
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["comp_sess"] (list): sessions for which absolute fractional 
                changes were calculated, with [x, y] => |(y - x) / x|
            ["datatype"] (str): type of data to retrieved
            ["rel_sess"] (int): number of session relative to which data is 
                scaled, for each mouse
            ["pop_stats"] (bool): whether population stats were analysed, 
                instead of individual ROI data
            ["seed"] (int): seed
        - usi_comp_df (pd.DataFrame):
            dataframe with one row per line/plane and one for all line/planes 
            together, and the basic sess_df columns, in addition to, 
            for each stimtype:
            - stimtype (list): absolute fractional change statistics (me, err)
            - raw_p_vals (float): uncorrected p-value for data differences 
                between stimulus types 
            - p_vals (float): p-value for data differences between stimulus 
                types, corrected for multiple comparisons and tails
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
 
    title = "USI changes from session 1 to 3"
    
    usi_comp_df = pd.DataFrame.from_dict(usi_comp_df)

    ax = stim_plots.plot_stim_data_df(
        usi_comp_df, 
        stimpar=stimpar,
        permpar=permpar, 
        figpar=figpar, 
        pop_stats=extrapar["pop_stats"],
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
def plot_visual_flow_corrs_sess123_comps(analyspar, sesspar, stimpar, basepar, 
                                         idxpar, permpar, extrapar, idx_corr_df, 
                                         figpar):
    """
    plot_visual_flow_corrs_sess123_comps(analyspar, sesspar, stimpar, basepar, 
                                         idxpar, permpar, extrapar, idx_corr_df, 
                                         figpar)

    From dictionaries, plots ROI USI correlations across sessions for tracked 
    visual flow USIs. 
    
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
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["consec_only"] (bool): if True, only consecutive sessions are  
                correlated
            ["corr_type"] (str): type of correlation ("corr" or "R_sqr")
            ["permute"] (str): type of permutations used 
                ("sess", "tracking", "all")
            ["seed"] (int): seed
            ["sig_only"] (bool): if True, only ROIs with significant USIs are 
                retained
        - idx_corr_df (pd.DataFrame):
            dataframe with one row per line/plane, and the 
            following columns, in addition to the basic sess_df columns:

            for session comparisons, e.g. 1v2
            - {}v{}{norm_str}_corrs (float): intersession ROI index correlations
            - {}v{}{norm_str}_corr_stds (float): bootstrapped intersession ROI 
                index correlation standard deviation
            - {}v{}_null_CIs (list): adjusted null CI for intersession ROI 
                index correlations
            - {}v{}_raw_p_vals (float): p-value for intersession correlations
            - {}v{}_p_vals (float): p-value for intersession correlations, 
                corrected for multiple comparisons and tails

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
 

    title = "Correlations for visual flow USIs"

    idx_corr_df = pd.DataFrame.from_dict(idx_corr_df)
    
    ax = corr_plots.plot_idx_correlations(
        idx_corr_df, 
        permpar=permpar,
        figpar=figpar, 
        permute=extrapar["permute"],
        corr_type=extrapar["corr_type"], 
        title=title, 
        small=True,
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
def plot_scatterplots(analyspar, sesspar, stimpar, basepar, idxpar, permpar, 
                      extrapar, idx_corr_df, figpar, title=None):
    """
    plot_scatterplots(analyspar, sesspar, stimpar, basepar, idxpar, permpar, 
                      extrapar, idx_corr_df, figpar)

    From dictionaries, plots ROI USI correlation scatterplots. 
    
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
        - idxpar (dict): 
            dictionary with keys of IdxPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
            ["permute"] (str): type of permutations used 
                ("sess", "tracking", "all")
            ["seed"] (int): seed
            ["sig_only"] (bool): if True, only ROIs with significant USIs are 
                retained
        - idx_corr_df (pd.DataFrame):
            dataframe with one row per line/plane, and the 
            following columns, in addition to the basic sess_df columns:

            for correlation data (normalized if corr_type is "diff_corr") for 
            session comparisons (x, y), e.g. 1v2
            - binned_rand_stats (list): number of random correlation values per 
                bin (xs x ys)
            - corr_data_xs (list): USI values for x
            - corr_data_ys (list): USI values for y
            - corrs (float): correlation between session data (x and y)
            - p_vals (float): p-value for correlation, corrected for 
                multiple comparisons and tails
            - rand_corr_meds (float): median of the random correlations
            - raw_p_vals (float): p-value for intersession correlations
            - regr_coefs (float): regression correlation coefficient (slope)
            - regr_intercepts (float): regression correlation intercept
            - x_bin_mids (list): x mid point for each random correlation bin
            - y_bin_mids (list): y mid point for each random correlation bin

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

    idx_corr_df = pd.DataFrame.from_dict(idx_corr_df)

    ax = corr_plots.plot_idx_corr_scatterplots(
        idx_corr_df, 
        permpar=permpar, 
        figpar=figpar, 
        permute=extrapar["permute"], 
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
def plot_gabor_corr_scatterplots_sess12(analyspar, sesspar, stimpar, basepar, 
                                        idxpar, permpar, extrapar, idx_corr_df, 
                                        figpar):
    """
    plot_gabor_corr_scatterplots_sess12(analyspar, sesspar, stimpar, basepar, 
                                        idxpar, permpar, extrapar, idx_corr_df, 
                                        figpar)

    Retrieves tracked ROI Gabor USI correlation scatterplot data for 
    sessions 1 and 2.
        
    Saves results and parameters relevant to analysis in a dictionary.

    See plot_scatterplots().
    """
 
    title = "Correlation scatterplot for Gabor USIs (session 1 vs 2)"

    fulldir, savename = plot_scatterplots(
        analyspar=analyspar, 
        sesspar=sesspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar, 
        extrapar=extrapar, 
        idx_corr_df=idx_corr_df, 
        figpar=figpar, 
        title=title
        )

    return fulldir, savename


#############################################
def plot_gabor_corr_scatterplots_sess23(analyspar, sesspar, stimpar, basepar, 
                                        idxpar, permpar, extrapar, idx_corr_df, 
                                        figpar):
    """
    plot_gabor_corr_scatterplots_sess23(analyspar, sesspar, stimpar, basepar, 
                                        idxpar, permpar, extrapar, idx_corr_df, 
                                        figpar)

    Retrieves tracked ROI Gabor USI correlation scatterplot data for 
    sessions 2 and 3.
        
    Saves results and parameters relevant to analysis in a dictionary.

    See plot_scatterplots().
    """
 
    title = "Correlation scatterplot for Gabor USIs (session 2 vs 3)"

    fulldir, savename = plot_scatterplots(
        analyspar=analyspar, 
        sesspar=sesspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar, 
        extrapar=extrapar, 
        idx_corr_df=idx_corr_df, 
        figpar=figpar, 
        title=title
        )

    return fulldir, savename

    
#############################################
def plot_visual_flow_corr_scatterplots_sess12(
        analyspar, sesspar, stimpar, basepar, idxpar, permpar, extrapar, 
        idx_corr_df, figpar):
    """
    plot_visual_flow_corr_scatterplots_sess12(
        analyspar, sesspar, stimpar, basepar, idxpar, permpar, extrapar, 
        idx_corr_df, figpar)

    Retrieves tracked ROI visual flow USI correlation scatterplot data for 
    sessions 1 and 2.
        
    Saves results and parameters relevant to analysis in a dictionary.

    See plot_scatterplots().
    """
 
    title = "Correlation scatterplot for visual flow USIs (session 1 vs 2)"

    fulldir, savename = plot_scatterplots(
        analyspar=analyspar, 
        sesspar=sesspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar, 
        extrapar=extrapar, 
        idx_corr_df=idx_corr_df, 
        figpar=figpar, 
        title=title
        )

    return fulldir, savename

    
#############################################
def plot_visual_flow_corr_scatterplots_sess23(
        analyspar, sesspar, stimpar, basepar, idxpar, permpar, extrapar, 
        idx_corr_df, figpar):
    """
    plot_visual_flow_corr_scatterplots_sess23(
        analyspar, sesspar, stimpar, basepar, idxpar, permpar, extrapar, 
        idx_corr_df, figpar)

    Retrieves tracked ROI visual flow USI correlation scatterplot data for 
    sessions 2 and 3.
        
    Saves results and parameters relevant to analysis in a dictionary.

    See plot_scatterplots().
    """
 
    title = "Correlation scatterplot for visual flow USIs (session 2 vs 3)"

    fulldir, savename = plot_scatterplots(
        analyspar=analyspar, 
        sesspar=sesspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar, 
        extrapar=extrapar, 
        idx_corr_df=idx_corr_df, 
        figpar=figpar, 
        title=title
        )

    return fulldir, savename


#############################################
def plot_dendritic_roi_tracking_example(analyspar, sesspar, extrapar, 
                                        roi_mask_df, figpar):
    """
    plot_dendritic_roi_tracking_example(analyspar, sesspar, extrapar, 
                                        roi_mask_df, figpar)

    From dictionaries, plots tracked ROI masks for different session orderings 
    for an example dendritic session.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
        - roi_mask_df (pd.DataFrame in dict format):
            dataframe with a row for each mouse, and the following 
            columns, in addition to the basic sess_df columns: 
            - "roi_mask_shapes" (list): shape into which ROI mask indices index 
                (sess x hei x wid)
            - "union_n_conflicts" (int): number of conflicts after union
            for "union", "fewest" and "most" tracked ROIs:
            - "{}_registered_roi_mask_idxs" (list): list of mask indices, 
                registered across sessions, for each session 
                (flattened across ROIs) ((sess, hei, wid) x val),
                ordered by {}_sess_ns if "fewest" or "most"
            - "{}_n_tracked" (int): number of tracked ROIs
            for "fewest", "most" tracked ROIs:
            - "{}_sess_ns" (list): ordered session number 

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

    title = "Dendritic ROI tracking example"

    roi_mask_df = pd.DataFrame.from_dict(roi_mask_df)

    ax = roi_plots.plot_roi_tracking(
        roi_mask_df, 
        figpar=figpar, 
        title=title
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
def plot_somatic_roi_tracking_example(analyspar, sesspar, extrapar, roi_mask_df, 
                                      figpar):
    """
    plot_somatic_roi_tracking_example(analyspar, sesspar, extrapar, roi_mask_df, 
                                      figpar)

    From dictionaries, plots tracked ROI masks for different session orderings 
    for an example somatic session.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - extrapar (dict): 
            dictionary containing additional analysis parameters
        - roi_mask_df (pd.DataFrame in dict format):
            dataframe with a row for each mouse, and the following 
            columns, in addition to the basic sess_df columns: 
            - "roi_mask_shapes" (list): shape into which ROI mask indices index 
                (sess x hei x wid)
            - "union_n_conflicts" (int): number of conflicts after union
            for "union", "fewest" and "most" tracked ROIs:
            - "{}_registered_roi_mask_idxs" (list): list of mask indices, 
                registered across sessions, for each session 
                (flattened across ROIs) ((sess, hei, wid) x val), 
                ordered by {}_sess_ns if "fewest" or "most"
            - "{}_n_tracked" (int): number of tracked ROIs
            for "fewest", "most" tracked ROIs:
            - "{}_sess_ns" (list): ordered session number 

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

    title = "Somatic ROI tracking example"

    roi_mask_df = pd.DataFrame.from_dict(roi_mask_df)

    ax = roi_plots.plot_roi_tracking(
        roi_mask_df, 
        figpar=figpar, 
        title=title
        )
    fig = ax.reshape(-1)[0].figure
    
    savedir, savename = helper_fcts.get_save_path(
        figpar['fig_panel_analysis'], main_direc=figpar["dirs"]["figdir"]
    )

    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=True, **figpar["save"]
    )

    return fulldir, savename
