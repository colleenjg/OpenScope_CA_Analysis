"""
corr_plots.py

This script contains functions for plotting correlation analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

import numpy as np

from util import plot_util, logger_util
from sess_util import sess_plot_util
from analysis import misc_analys
from plot_fcts import plot_helper_fcts

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def get_sorted_sess_pairs(idx_corr_df):
    """
    get_sorted_sess_pairs(idx_corr_df)

    Returns list of session pairs, sorted by session number

    Required args:
        - idx_corr_df (pd.DataFrame):
            dataframe with one row per line/plane, and the 
            following columns, in addition to the basic sess_df columns:

            for session comparisons, e.g. 1v2
            - {}v{}_norm_corrs (float): normalized intersession ROI index 
                correlations

    Returns:
        - sorted_pairs (list):
            sorted session number pairs, e.g. [[s1, s2], [s2, s3], ...]
    """

    sess_pairs = [
        [int(n) for n in col.replace("_norm_corrs", "").split("v")] 
        for col in idx_corr_df.columns
        if "_norm_corrs" in col
        ]

    sorted_pairs = [
        sess_pairs[i] for i in np.argsort(
            [float(f"{s1}.{s2}") for s1, s2 in sess_pairs]
            )
        ]
    
    return sorted_pairs


#############################################
def get_idx_corr_ylims(idx_corr_df):
    """
    get_idx_corr_ylims(idx_corr_df)

    Returns data edges (min and max data to be plotted).

    Required args:
        - idx_corr_df (pd.DataFrame):
            dataframe with one row per line/plane, and the 
            following columns, in addition to the basic sess_df columns:

            for session comparisons, e.g. 1v2
            - {}v{}_norm_corrs (float): normalized intersession ROI index 
                correlations
            - {}v{}_norm_corr_stds (float): bootstrapped normalized 
                intersession ROI index correlation standard deviation
            - {}v{}_null_CIs (list): adjusted null CI for normalized 
                intersession ROI index correlations
        
    Returns:
        - plane_pts (list):
            [low_pt, high_pt] for each plane, in plane order, based on plane 
            indices
    """  

    sess_pairs = get_sorted_sess_pairs(idx_corr_df)

    plane_pts = []
    plane_idxs = []
    for plane, plane_df in idx_corr_df.groupby("planes"):
        _, pl, _, _ = plot_helper_fcts.get_line_plane_idxs(plane=plane)
        plane_idxs.append(pl)

        low_pts, high_pts = [], []

        for sess_pair in sess_pairs:
            base = f"{sess_pair[0]}v{sess_pair[1]}"

            # get null_CIs low
            null_CI_low = np.min(
                [null_CI[0] for null_CI in plane_df[f"{base}_null_CIs"]]
            )
            null_CI_high = np.max(
                [null_CI[2] for null_CI in plane_df[f"{base}_null_CIs"]]
            )

            # get data
            data_low = (
                plane_df[f"{base}_norm_corrs"] - 
                plane_df[f"{base}_norm_corr_stds"]
                ).min()
            
            data_high = (
                plane_df[f"{base}_norm_corrs"] + 
                plane_df[f"{base}_norm_corr_stds"]
                ).max()

            low_pts.extend([null_CI_low, data_low])
            high_pts.extend([null_CI_high, data_high])

        low_pt = np.min(low_pts)
        high_pt = np.max(high_pts)

        pt_range = high_pt - low_pt
        low_pt -= pt_range / 10
        high_pt += pt_range / 10

        plane_pts.append([low_pt, high_pt])

    plane_pts = [plane_pts[i] for i in np.argsort(plane_idxs)]

    return plane_pts


#############################################
def plot_idx_correlations(idx_corr_df, permpar, figpar, title=None, small=True):
    """
    plot_idx_correlations(idx_corr_df, permpar, figpar)

    Plots ROI USI index correlations across sessions.

    Required args:
        - idx_corr_df (pd.DataFrame):
            dataframe with one row per line/plane, and the 
            following columns, in addition to the basic sess_df columns:

            for session comparisons, e.g. 1v2
            - {}v{}_norm_corrs (float): normalized intersession ROI index 
                correlations
            - {}v{}_norm_corr_stds (float): bootstrapped normalized 
                intersession ROI index correlation standard deviation
            - {}v{}_null_CIs (list): adjusted null CI for normalized 
                intersession ROI index correlations
            - {}v{}_raw_p_vals (float): p-value for normalized intersession 
                correlations
            - {}v{}_p_vals (float): p-value for normalized intersession 
                correlations, corrected for multiple comparisons and tails

        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters

    Optional args:
        - title (str):
            plot title
            default: None
        - small (bool):
            if True, smaller subplots are plotted
            default: True

    Returns:
        - ax (2D array): 
            array of subplots
    """

    sess_pairs = get_sorted_sess_pairs(idx_corr_df)
    n_pairs = int(np.ceil(len(sess_pairs) / 2) * 2) # multiple of 2

    figpar = sess_plot_util.fig_init_linpla(
        figpar, kind="reg", n_sub=int(n_pairs / 2)
        )

    figpar["init"]["ncols"] = n_pairs
    figpar["init"]["sharey"] = "row"

    if small:
        figpar["init"]["subplot_wid"] = 3.0
        figpar["init"]["subplot_hei"] = 4.0
    else:
        figpar["init"]["subplot_wid"] = 3.5
        figpar["init"]["subplot_hei"] = 5.0

    fig, ax = plot_util.init_fig(n_pairs * 2, **figpar["init"])
    if title is not None:
        fig.suptitle(title, y=1.0, weight="bold")

    plane_pts = get_idx_corr_ylims(idx_corr_df)
    lines = [None, None]

    comp_info = misc_analys.get_comp_info(permpar)
    logger.info(f"Corrected p-values ({comp_info}):", extra={"spacing": "\n"})
    for (line, plane), lp_df in idx_corr_df.groupby(["lines", "planes"]):
        li, pl, col, _ = plot_helper_fcts.get_line_plane_idxs(line, plane)
        linpla_name = plot_helper_fcts.get_line_plane_name(line, plane)
        lines[li] = line.split("-")[0].replace("23", "2/3")

        if len(lp_df) != 1:
            raise RuntimeError("Expected only one row per line/plane.")
        row = lp_df.loc[lp_df.index[0]]
        
        lp_sig_str = f"{linpla_name:6}:"
        for s, sess_pair in enumerate(sess_pairs):
            sub_ax = ax[pl, s]
            if s == 0:
                sub_ax.set_ylim(plane_pts[pl])

            col_base = f"{sess_pair[0]}v{sess_pair[1]}"
            
            CI = row[f"{col_base}_null_CIs"]
            extr = np.asarray([CI[0], CI[2]])
            plot_util.plot_CI(
                sub_ax, extr, med=CI[1], x=li, width=0.45, med_rat=0.025
                )

            y = row[f"{col_base}_norm_corrs"]
            err = row[f"{col_base}_norm_corr_stds"]
            plot_util.plot_ufo(
                sub_ax, x=li, y=y, err=err, color=col, capsize=8
                )

            # add significance markers
            p_val = row[f"{col_base}_p_vals"]
            side = np.sign(y - CI[1])
            sensitivity = misc_analys.get_sensitivity(permpar)          
            sig_str = misc_analys.get_sig_symbol(
                p_val, sensitivity=sensitivity, side=side, 
                tails=permpar["tails"], p_thresh=permpar["p_val"]
                )

            if len(sig_str):
                high = np.max([CI[-1], y + err])
                plot_util.add_signif_mark(sub_ax, li, high, 
                    rel_y=0.1, color=col, fontsize=24, mark=sig_str) 

            sess_str = f"S{sess_pair[0]}v{sess_pair[1]}: "
            lp_sig_str = f"{lp_sig_str}{TAB}{sess_str}{p_val:.5f}{sig_str:3}"

        logger.info(lp_sig_str, extra={"spacing": TAB})

    # Add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(
        ax, datatype="roi", lines=["", ""], xticks=[0, 1], ylab="", 
        xlab="", kind="traces"
        )

    xs = np.arange(len(lines))
    pad_x = 0.8 * (xs[1] - xs[0])
    for row_n in range(len(ax)):
        for col_n in range(len(ax[row_n])):
            sub_ax = ax[row_n, col_n]
            sub_ax.tick_params(axis="x", which="both", bottom=False) 
            sub_ax.set_xticks(xs)
            sub_ax.set_xticklabels(lines, weight="bold")
            sub_ax.set_xlim([xs[0] - pad_x, xs[-1] + pad_x])

            sub_ax.set_ylim(plane_pts[row_n])
            sub_ax.set_yticks(plane_pts[row_n])

            if row_n == 0:
                if col_n < len(sess_pairs):
                    s1, s2 = sess_pairs[col_n]
                    sess_pair_title = f"Session {s1} v {s2}"
                    sub_ax.set_title(
                        sess_pair_title, fontweight="bold", y=1.07
                        )

        plot_util.set_interm_ticks(
            ax[row_n], 3, dim="y", weight="bold", share=True
            )
    
    return ax

