"""
corr_plots.py

This script contains functions for plotting correlation analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

from matplotlib import pyplot as plt
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
def plot_corr_ex_data_scatterplot(sub_ax, idx_corr_norm_row, corr_name="1v2", 
                                  col="k"):
    """
    plot_corr_ex_data_scatterplot(sub_ax, idx_corr_norm_row)

    Plots example random correlation data in a scatterplot showing real versus
    example randomly generated data.

    Required args:
        - sub_ax (plt subplot): 
            subplot
        - idx_corr_norm_df (pd.Series):
            dataframe series with the following columns, in addition to the 
            basic sess_df columns:

            for a specific session comparison, e.g. 1v2
            - {}v{}_corrs (float): unnormalized intersession ROI index 
                correlations
            - {}v{}_norm_corrs (float): normalized intersession ROI index 
                correlations
            - {}v{}_rand_corr_meds (float): median of randomized correlations

            - {}v{}_rand_corrs_binned (list): binned random unnormalized 
                intersession ROI index correlations
            - {}v{}_rand_corrs_bin_edges (list): bins edges
    
    Optional args:
        - corr_name (str):
            session pair correlation name, used in series columns
            default: "1v2"
        - col (str):
            color for real data
    """

    sess_pair = corr_name.split("v")

    x_perm, y_perm = np.asarray(idx_corr_norm_row[f"{corr_name}_rand_ex"])
    raw_rand_corr = idx_corr_norm_row[f"{corr_name}_rand_ex_corrs"]
    sub_ax.scatter(
        x_perm, y_perm, color="gray", alpha=0.4, marker="d", lw=2, 
        label=f"Raw random corr: {raw_rand_corr:.2f}"
        )

    x_data, y_data = np.asarray(idx_corr_norm_row[f"{corr_name}_corr_data"])
    raw_corr = idx_corr_norm_row[f"{corr_name}_corrs"]
    sub_ax.scatter(
        x_data, y_data, color=col, alpha=0.4, lw=2, 
        label=f"Raw corr: {raw_corr:.2f}"
        )
    
    sub_ax.set_ylabel(
        f"USI diff. between\nsession {sess_pair[0]} and {sess_pair[1]}", 
        fontweight="bold"
        )
    sub_ax.set_xlabel(f"Session {sess_pair[0]} USIs", fontweight="bold"
        )
    sub_ax.legend()
    
    
#############################################
def plot_corr_ex_data_histogram(sub_ax, idx_corr_norm_row, corr_name="1v2", 
                                col="k"):
    """
    plot_corr_ex_data_histogram(sub_ax, idx_corr_norm_row)

    Plots example random correlation data in a histogram show how normalized 
    residual correlations are calculated.

    Required args:
        - sub_ax (plt subplot): 
            subplot
        - idx_corr_norm_df (pd.Series):
            dataframe series with the following columns, in addition to the 
            basic sess_df columns:

            for a specific session comparison, e.g. 1v2
            - {}v{}_corrs (float): unnormalized intersession ROI index 
                correlations
            - {}v{}_norm_corrs (float): normalized intersession ROI index 
                correlations
            - {}v{}_rand_corr_meds (float): median of randomized correlations

            - {}v{}_rand_corrs_binned (list): binned random unnormalized 
                intersession ROI index correlations
            - {}v{}_rand_corrs_bin_edges (list): bins edges
    
    Optional args:
        - corr_name (str):
            session pair correlation name, used in series columns
            default: "1v2"
        - col (str):
            color for real data
    """

    med = idx_corr_norm_row[f"{corr_name}_rand_corr_meds"]
    raw_corr = idx_corr_norm_row[f"{corr_name}_corrs"]
    norm_corr = idx_corr_norm_row[f"{corr_name}_norm_corrs"]
    binned_corrs = \
        np.asarray(idx_corr_norm_row[f"{corr_name}_rand_corrs_binned"])
    bin_edges = idx_corr_norm_row[f"{corr_name}_rand_corrs_bin_edges"]
    bin_edges = np.linspace(bin_edges[0], bin_edges[1], len(binned_corrs) + 1)

    sub_ax.hist(
        bin_edges[:-1], bin_edges, weights=binned_corrs, color="gray", 
        alpha=0.45, density=True
        )
    # median line
    sub_ax.axvline(x=med, ls=plot_helper_fcts.VDASH, c="k", lw=3.0, alpha=0.5)
    
    # corr line
    sub_ax.axvline(
        x=raw_corr, ls=plot_helper_fcts.VDASH, c=col, lw=3.0, alpha=0.7
        )
    
    # adjust axes so that at least 1/5 of the graph is beyond the correlation value
    xlims = list(sub_ax.get_xlim())
    if raw_corr < med:
        leave_space = np.absolute(np.diff([raw_corr, xlims[1]]))[0] / 3
        xlims[0] = np.min([xlims[0], -1.08, leave_space])
        edge = -1
    else:
        leave_space = np.absolute(np.diff([raw_corr, xlims[0]]))[0] / 3
        xlims[1] = np.max([xlims[1], 1.08, leave_space])
        edge = 1
    
    # edge line
    sub_ax.axvline(x=edge, ls=plot_helper_fcts.VDASH, c="k", lw=3.0, alpha=0.5)

    # shift limits
    sub_ax.set_xlim(xlims)
    ylims = list(sub_ax.get_ylim())
    sub_ax.set_ylim(ylims[0], ylims[1] * 1.3)

    sub_ax.set_ylabel("Density", fontweight="bold", labelpad=10)
    sub_ax.set_xlabel("Raw correlations", fontweight="bold")
    sub_ax.set_title(
        f"Normalized residual\ncorrelation: {norm_corr:.2f}", 
        fontweight="bold", y=1.07, fontsize=20
    )


#############################################
def plot_rand_corr_ex_data(idx_corr_norm_df, title=None):
    """
    plot_rand_corr_ex_data(idx_corr_norm_df)

    Plots example random correlation data in a scatterplot and histogram to 
    show how normalized residual correlations are calculated.

    Required args:
        - idx_corr_norm_df (pd.DataFrame):
            dataframe with one row for a line/plane, and the 
            following columns, in addition to the basic sess_df columns:

            for a specific session comparison, e.g. 1v2
            - {}v{}_corrs (float): unnormalized intersession ROI index 
                correlations
            - {}v{}_norm_corrs (float): normalized intersession ROI index 
                correlations
            - {}v{}_rand_ex_corrs (float): unnormalized intersession 
                ROI index correlations for an example of randomized data
            - {}v{}_rand_corr_meds (float): median of randomized correlations

            - {}v{}_corr_data (list): intersession values to correlate
            - {}v{}_rand_ex (list): intersession values for an example of 
                randomized data
            - {}v{}_rand_corrs_binned (list): binned random unnormalized 
                intersession ROI index correlations
            - {}v{}_rand_corrs_bin_edges (list): bins edges

    Optional args:
        - title (str):
            plot title
            default: None

    Returns:
        - ax (2D array): 
            array of subplots
    """

    plot_types = ["scatter", "hist"]
    fig, ax = plt.subplots(
        nrows=len(plot_types), figsize=[8.7, 9.3], gridspec_kw={"hspace": 0.7}
        ) 

    if len(idx_corr_norm_df) != 1:
        raise ValueError("Expected idx_corr_norm_df to contain only one row.")
    
    sorted_pairs = get_sorted_sess_pairs(idx_corr_norm_df)

    if len(sorted_pairs) != 1:
        raise RuntimeError(
            "Expected to find only one pair of sessions for which to plot data."
            )
    sess_pair = sorted_pairs[0]
    
    row = idx_corr_norm_df.loc[idx_corr_norm_df.index[0]]
    corr_name = f"{sess_pair[0]}v{sess_pair[1]}"
    _, _, col, _ = plot_helper_fcts.get_line_plane_idxs(
        row["lines"], row["planes"]
        )

    if title is not None:
        fig.suptitle(title, y=0.95, weight="bold")

    # plot scatterplot
    scatt_ax = ax[0]
    plot_corr_ex_data_scatterplot(scatt_ax, row, corr_name=corr_name, col=col)

    # plot histogram
    hist_ax = ax[1]
    plot_corr_ex_data_histogram(hist_ax, row, corr_name=corr_name, col=col)

    plot_util.set_interm_ticks(
        ax, n_ticks=3, dim="x", share=False, fontweight="bold"
        )
    plot_util.set_interm_ticks(
        ax, n_ticks=4, dim="y", share=False, fontweight="bold"
        )

    return ax


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

        if low_pt < -1:
            low_pt = -1

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

    figpar["init"]["gs"] = {"hspace": 0.25}
    if small:
        figpar["init"]["subplot_wid"] = 2.7
        figpar["init"]["subplot_hei"] = 4.2
        figpar["init"]["gs"]["wspace"] = 0.2
    else:
        figpar["init"]["subplot_wid"] = 3.3
        figpar["init"]["subplot_hei"] = 5.2
        figpar["init"]["gs"]["wspace"] = 0.3 
        

    fig, ax = plot_util.init_fig(n_pairs * 2, **figpar["init"])
    if title is not None:
        fig.suptitle(title, y=0.98, weight="bold")

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
    pad_x = 0.6 * (xs[1] - xs[0])
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
                sub_ax.spines["bottom"].set_visible(True)

        plot_util.set_interm_ticks(
            ax[row_n], 3, dim="y", weight="bold", share=False, update_ticks=True
            )
    
    return ax

