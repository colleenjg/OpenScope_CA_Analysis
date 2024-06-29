"""
plot_figs.py

This script contains functions defining figure panel plotting.

Authors: Colleen Gillon

Date: February 2023

Note: this code was aggregated from https://github.com/colleenjg/OpenScope_CA_Analysis.
"""

import pandas as pd

from credassign.util import plot_util
from credassign.paper_fig_util import helper_fcts
from credassign.plot_fcts import behav_plots, misc_plots, seq_plots, roi_plots


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

