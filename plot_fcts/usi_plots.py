"""
usi_plots.py

This script contains functions for plotting USI analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import copy
import logging

import numpy as np

from util import logger_util, gen_util, plot_util, math_util
from sess_util import sess_plot_util
from analysis import misc_analys
from plot_fcts import plot_helper_fcts, seq_plots

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def add_USI_boxes(ax, chosen_rois_df, sorted_target_idxs): 
    """
    add_USI_boxes(ax, chosen_rois_df, sorted_target_idxs)

    Adds boxes with USI values to individual plots (e.g., trace plots).

    Required args:
        - ax (subplot array):
            pyplot axis array
        - chosen_rois_df (pd.DataFrame):
            chosen ROIs dataframe with, in addition to the basic sess_df 
            columns, "target_idxs".
        - sorted_target_idxs (list): 
            order in which different target_idxs should be added to subplots
    """
    props = dict(
        boxstyle="round", facecolor="white", edgecolor="black", alpha=0.5, 
        lw=1.5)

    for (line, plane), lp_df in chosen_rois_df.groupby(["lines", "planes"]):
        li, pl, _, _ = plot_helper_fcts.get_line_plane_idxs(line, plane)

        for r, row_val in enumerate(sorted_target_idxs):
            rows = lp_df.loc[lp_df["target_idxs"] == row_val]
            if len(rows) == 0:
                continue
            elif len(rows) > 1:
                raise RuntimeError(
                    "Expected row_order instances to be unique per line/plane."
                    )

            row = rows.loc[rows.index[0]]
            sub_ax = ax[r + pl * len(sorted_target_idxs), li]

            # place a text box in upper left in axes coords
            sub_ax.text(0.1, 0.9, f"USI = {row['roi_idxs']:.2f}", 
                transform=sub_ax.transAxes, fontsize=15, va="center", 
                bbox=props)
            


#############################################
def plot_stim_idx_hist(sub_ax, data, CI_lims, n_bins=None, rand_data=None, 
                       orig_edges=None, plot="items", col="b", density=False, 
                       perc_label=None):
    """
    plot_stim_idx_hist(sub_ax, percs, CI_lims)

    Plots histograms of stimulus indices.

    Required args:
        - sub_ax (plt subplot): 
            subplot
        - data (1D array):
            binned index data (ROIs or percentiles, based on 'plot')
        - CI_lims (list): 
            confidence interval limits (index or percentile values)

    Optional args:
        - n_bins (int): 
            number of histogram bins to plot (must be a divisor of original 
            number of bins and compatible with separately binning the data 
            outside the CI). If None, the minimum accepted binning value is 
            selected.
            default: None
        - data (array-like): 
            item index bin counts, required if plot == "items"
            default: None
        - rand_data (array_like): 
            randomly calculated item index bin counts, required if 
            plot == "items"
            default: None 
        - orig_edges (None): 
            origin edges used to generate counts [min, max], required if 
            plot == "items"
            default: None
        - plot (str): 
            type of plot ("items" for ROI indices,"percs" for index percentiles) 
            default: "items"
        - col (str): 
            color for indices (only significant indices, if plot == "data")
            default: "b"
        - density (bool): 
            if True, densities are plotted instead of frequencies.
            default: False
        - perc_label (str): percent significant ROIs label
            default: None
    """

    orig_n_bins = len(data)

    if plot == "percs":
        CI_wid = np.max(np.absolute([CI_lims[0] - 0, 100 - CI_lims[1]]))
        min_n_bins = int(np.around(100 / CI_wid))
        bin_edges = np.linspace(0, 100, n_bins + 1).tolist()
        poss_b_bins = list(range(min_n_bins, orig_n_bins + 1, min_n_bins))
        if len(poss_b_bins) == 0:
            raise RuntimeError("Original binning of the data is incompatible "
                f"with the CI_lims: {CI_lims[0]} to ({CI_lims[1]}. ")
        elif n_bins is None:
            n_bins = poss_b_bins[0]
        elif n_bins not in poss_b_bins:
            raise RuntimeError("Target binning value is incompatible with "
                "other parameters. Must be among {}.".join(
                    [str(val) for val in poss_b_bins]))

    elif plot == "items":
        if rand_data is not None and len(data) != len(rand_data):
            raise ValueError(
                "'data' and 'rand_data' must have the same length."
                )
        if (orig_edges is None) or (len(orig_edges) != 2):
            raise ValueError("Must provide 'orig_edges' if plot is 'items', "
                "and of length 2 [min, max].")
        if n_bins is None:
            n_bins = orig_n_bins
        elif (n_bins / orig_n_bins) != (n_bins // orig_n_bins):
            raise ValueError(f"Target number of bins ({n_bins}) is "
                f"incompatible with original number of bins ({orig_n_bins}).")
        bin_edges = np.linspace(orig_edges[0], orig_edges[1], n_bins + 1)

    else:
        gen_util.accepted_values_error("plot", plot, ["percs", "items"])

    # rebin and collate data
    join_bins = int(orig_n_bins / n_bins)
    rebinned_data = [np.sum(np.asarray(data).reshape(-1, join_bins), axis=1)]
    colors = [col]
    alphas = [0.6] if plot == "items" else [0.7]

    if plot == "items" and rand_data is not None:
        rand_data = np.sum(np.asarray(rand_data).reshape(-1, join_bins), axis=1)
        rebinned_data.insert(0, rand_data)
        colors.insert(0, "gray")
        alphas.insert(0, 0.45)

    elif plot == "percs": # add red markers for significant zones
        for c, CI_lim in enumerate(CI_lims):
            if c == 0:
                sub_ax.axvspan(
                    0, CI_lim, color=plot_helper_fcts.DARKRED, alpha=0.15, 
                    lw=80./n_bins)
            elif c == 1:
                sub_ax.axvspan(
                    CI_lim, 100, color=plot_helper_fcts.DARKRED, alpha=0.15, 
                    lw=80./n_bins)

    for sub, col, alp in zip(rebinned_data, colors, alphas):
        sub_ax.hist(
            bin_edges[:-1], bin_edges, weights=sub, color=col, alpha=alp, 
            density=density)

    # add a chance line
    if plot == "percs":
        if density:
            n_rand = 1 / n_bins
        else:
            n_rand = np.mean(data) / n_bins
        sub_ax.axhline(
            y=n_rand, ls=plot_helper_fcts.HDASH, c="k", lw=3.0, alpha=0.5
            )

    # add label
    if perc_label is not None:
        sub_ax.plot([], label=perc_label, color=col)
        sub_ax.legend(fontsize="large")
    
    
#############################################
def plot_idxs(idx_df, sesspar, figpar, plot="items", density=True, n_bins=40, 
              title=None, size="reg"):
    """
    plot_idxs(idx_df, sesspar, figpar)

    Returns exact color for a specific line.

    Required args:
        - idx_df (pd.DataFrame):
            dataframe with indices for different line/plane combinations, and 
            the following columns, in addition to the basic sess_df columns:
            - rand_idx_binned (list): bin counts for the random ROI indices
            - bin_edges (list): first and last bin edge
            - CI_edges (list): confidence interval limit values
            - CI_perc (list): confidence interval percentile limits
            if plot == "items":
            - roi_idx_binned (list): bin counts for the ROI indices
            if plot == "percs":
            - perc_idx_binned (list): bin counts for the ROI index percentiles
            optionally:
            - n_signif_lo (int): number of significant ROIs (low) 
            - n_signif_hi (int): number of significant ROIs (high)

        - sesspar (dict): 
            dictionary with keys of SessPar namedtuple
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters

    Optional args:
        - plot (str): 
            type of data to plot ("items" or "percs")
            default: "items"
        - density (bool): 
            if True, histograms are plotted as densities
            default: True
        - n_bins (int): 
            number of bins to use in histograms
            default: 40
        - title (str): 
            plot title
            default: None
        - size (str): 
            plot size ("reg", "small" or "tall")
            default: "reg"
        
    Returns:
        - ax (2D array): 
            array of subplots
    """

    if plot == "items":
        data_key = "roi_idx_binned"
        CI_key = "CI_edges"
    elif plot == "percs":
        data_key = "perc_idx_binned"
        CI_key = "CI_perc"
    else:
        gen_util.accepted_values_error("plot", plot, ["items", "percs"])

    sess_ns = misc_analys.get_sess_ns(sesspar, idx_df)

    n_plots = len(sess_ns) * 4
    figpar["init"]["sharey"] = "row"
    figpar = sess_plot_util.fig_init_linpla(figpar, kind="idx", 
        n_sub=len(sess_ns), sharex=(plot == "percs"))

    if size == "reg":
        subplot_hei = 3.2
        subplot_wid = 5.5
    elif size == "small":
        subplot_hei = 2.40
        subplot_wid = 3.75
        figpar["init"]["gs"] = {"hspace": 0.25, "wspace": 0.30}
    elif size == "tall":
        subplot_hei = 5.3
        subplot_wid = 5.55
    else:
        gen_util.accepted_values_error("size", size, ["reg", "small", "tall"])
    
    figpar["init"]["subplot_hei"] = subplot_hei
    figpar["init"]["subplot_wid"] = subplot_wid
    figpar["init"]["sharey"] = "row"
    
    fig, ax = plot_util.init_fig(n_plots, **figpar["init"])
    if title is not None:
        fig.suptitle(title, y=1.0, weight="bold")

    for (line, plane), lp_df in idx_df.groupby(["lines", "planes"]):
        li, pl, col, _ = plot_helper_fcts.get_line_plane_idxs(line, plane)

        for s, sess_n in enumerate(sess_ns):
            rows = lp_df.loc[lp_df["sess_ns"] == sess_n]
            if len(rows) == 0:
                continue
            elif len(rows) > 1:
                raise RuntimeError(
                    "Expected sess_ns to be unique per line/plane."
                    )
            row = rows.loc[rows.index[0]]

            sub_ax = ax[s + pl * len(sess_ns), li]

            # get percentage significant label
            perc_label = None
            if "n_signif_lo" in row.keys() and "n_signif_hi" in row.keys():
                n_sig_lo, n_sig_hi = row["n_signif_lo"], row["n_signif_hi"]
                nrois = np.sum(row["nrois"])
                perc_signif = np.sum([n_sig_lo, n_sig_hi]) / nrois * 100
                perc_label = (f"{perc_signif:.2f}% sig\n"
                    f"({n_sig_lo}-/{n_sig_hi}+ of {nrois})")                

            plot_stim_idx_hist(
                sub_ax, row[data_key], row[CI_key], n_bins=n_bins, 
                rand_data=row["rand_idx_binned"], 
                orig_edges=row["bin_edges"], 
                plot=plot, col=col, density=density, perc_label=perc_label)
            
            if size == "small":
                sub_ax.legend(fontsize="small")

    if plot == "percs":
        nticks = 5
        xticks = [int(np.around(x, 0)) for x in np.linspace(0, 100, nticks)]
        for sub_ax in ax[-1]:
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks, weight="bold")
    
    elif plot == "items":
        nticks = 3
        plot_util.set_interm_ticks(
            ax, nticks, dim="x", weight="bold", share=False, skip=False
            )
    
    else:
        gen_util.accepted_values_error("plot", plot, ["items", "percs"])

    # Add plane, line info to plots
    y_lab = "Density" if density else f"N ROIs" 
    sess_plot_util.format_linpla_subaxes(ax, datatype="roi", ylab=y_lab, 
        xticks=None, sess_ns=None, kind="idx", modif_share=False, 
        single_lab=True)
        
    return ax


#############################################
def get_perc_sig_ylims(perc_sig_df, low_pt_max=90, high_pt_min=10):
    """
    get_perc_sig_ylims(perc_sig_df)

    Returns data edges (min and max data to be plotted).

    Required args:
        - perc_sig_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns:
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

    Optional args:
        - low_pt_max (num):
            maximum value for the low point
            default: 90
        - high_pt_min (num):
            minimum value for the high point
            default: 10
        
    Returns:
        - low_pt (num):
            lowest data edge
        - high_pt (num):
            highest data edge
    """  

    low_pts, high_pts = [], []
    for col in ["lo", "hi"]:
        data_col = f"perc_sig_{col}_idxs"
        
        null_CI_low = np.min(
            [null_CI[0] for null_CI in perc_sig_df[f"{data_col}_null_CIs"]]
        )
        null_CI_high = np.max(
            [null_CI[2] for null_CI in perc_sig_df[f"{data_col}_null_CIs"]]
        )

        perc_low = (
            perc_sig_df[data_col] - perc_sig_df[f"{data_col}_stds"]
            ).min()
        perc_high = (
            perc_sig_df[data_col] + perc_sig_df[f"{data_col}_stds"]
            ).max()

        low_pts.extend([null_CI_low, perc_low])
        high_pts.extend([null_CI_high, perc_high])

    low_pt = np.min(low_pts)
    high_pt = np.max(high_pts)

    pt_range = high_pt - low_pt
    low_pt -= pt_range / 30 
    high_pt += pt_range / 30

    low_pt = np.min([low_pt_max, low_pt])
    high_pt = np.max([high_pt_min, high_pt])

    return low_pt, high_pt


#############################################
def plot_perc_sig_usis(perc_sig_df, analyspar, permpar, figpar, by_mouse=False, 
                       title=None):
    """
    plot_perc_sig_usis(perc_sig_df, analyspar, figpar)

    Plots percentage of significant USIs.

    Required args:
        - perc_sig_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns:
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

        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters

    Optional args:
        - by_mouse (bool):
            if True, plotting is done per mouse
            default: False
        - title (str):
            plot title
            default: None
        
    Returns:
        - ax (2D array): 
            array of subplots
    """  

    perc_sig_df = perc_sig_df.copy(deep=True)

    nanpol = None if analyspar["remnans"] else "omit"

    sess_ns = perc_sig_df["sess_ns"].unique()
    if len(sess_ns) != 1:
        raise NotImplementedError(
            "Plotting function implemented for 1 session only."
            )

    figpar = sess_plot_util.fig_init_linpla(figpar, kind="idx", n_sub=1, 
        sharex=True, sharey=True)

    figpar["init"]["sharey"] = True
    figpar["init"]["subplot_wid"] = 3.4
    figpar["init"]["gs"] = {"wspace": 0.18}
    if by_mouse:
        figpar["init"]["subplot_hei"] = 8.4
    else:
        figpar["init"]["subplot_hei"] = 3.5

    fig, ax = plot_util.init_fig(2, **figpar["init"])
    if title is not None:
        fig.suptitle(title, y=0.98, weight="bold")

    tail_order = ["Low tail", "High tail"]
    tail_keys = ["lo", "hi"]
    chance = permpar["p_val"] / 2 * 100

    ylims = get_perc_sig_ylims(perc_sig_df, high_pt_min=40)
    n_linpla = plot_helper_fcts.N_LINPLA

    comp_info = misc_analys.get_comp_info(permpar)
    logger.info(f"Corrected p-values ({comp_info}):", extra={"spacing": "\n"})
    for t, (tail, key) in enumerate(zip(tail_order, tail_keys)):
        sub_ax = ax[0, t]
        sub_ax.set_title(tail, fontweight="bold")
        sub_ax.set_ylim(ylims)

        # replace bottom spine with line at 0
        sub_ax.spines['bottom'].set_visible(False)
        sub_ax.axhline(y=0, c="k", lw=4.0)

        data_key = f"perc_sig_{key}_idxs"

        CIs = np.full((plot_helper_fcts.N_LINPLA, 2), np.nan)
        CI_meds = np.full(plot_helper_fcts.N_LINPLA, np.nan)

        tail_sig_str = f"{tail:9}:"
        linpla_names = []
        for (line, plane), lp_df in perc_sig_df.groupby(["lines", "planes"]):
            li, pl, col, _ = plot_helper_fcts.get_line_plane_idxs(line, plane)
            x_index = 2 * li + pl
            linpla_name = plot_helper_fcts.get_line_plane_name(line, plane)
            linpla_names.append(linpla_name)
            
            if len(lp_df) == 0:
                continue
            elif len(lp_df) > 1 and not by_mouse:
                raise RuntimeError("Expected a single row per line/plane.")

            df_indices = lp_df.index.tolist()

            if by_mouse:
                # plot means or medians per mouse
                mouse_data = lp_df[data_key].to_numpy()
                mouse_data_mean = math_util.mean_med(
                    mouse_data, stats=analyspar["stats"], nanpol=nanpol
                    )
                CI_dummy = np.repeat(mouse_data_mean, 2)
                plot_util.plot_CI(sub_ax, CI_dummy, med=mouse_data_mean, 
                    x=x_index, width=0.4, med_col=col, med_rat=0.01)
            else:
                # collect confidence interval data
                row = lp_df.loc[df_indices[0]]
                CIs[x_index] = np.asarray(row[f"{data_key}_null_CIs"])[
                    np.asarray([0, 2])
                    ]
                CI_meds[x_index] = row[f"{data_key}_null_CIs"][1]

            if by_mouse:
                perc_p_vals = []
                rel_y = 0.05
            else:
                tail_sig_str = f"{tail_sig_str}{TAB}{linpla_name}: "
                rel_y = 0.1

            for df_i in df_indices:
                # plot UFOs
                err = None
                no_line = True
                if not by_mouse:
                    err = perc_sig_df.loc[df_i, f"{data_key}_stds"]
                    no_line = False
                # indicate bootstrapped error with wider capsize
                plot_util.plot_ufo(
                    sub_ax, x_index, perc_sig_df.loc[df_i, data_key], err,
                    color=col, capsize=8, no_line=no_line
                    )

                # add significance markers
                p_val = perc_sig_df.loc[df_i, f"{data_key}_p_vals"]
                perc = perc_sig_df.loc[df_i, data_key]
                nrois = np.sum(perc_sig_df.loc[df_i, "nrois"])
                side = np.sign(perc - chance)
                sensitivity = misc_analys.get_binom_sensitivity(
                    nrois, null_perc=chance, side=side
                    )                
                sig_str = misc_analys.get_sig_symbol(
                    p_val, sensitivity=sensitivity, side=side, 
                    tails=permpar["tails"], p_thresh=permpar["p_val"]
                    )

                if len(sig_str):
                    perc_high = perc + err if err is not None else perc
                    plot_util.add_signif_mark(sub_ax, x_index, perc_high, 
                        rel_y=rel_y, color=col, fontsize=24, mark=sig_str) 

                if by_mouse:
                    perc_p_vals.append(
                        (int(np.around(perc)), p_val, sig_str)
                    )
                else:
                    tail_sig_str = (
                        f"{tail_sig_str}{p_val:.5f}{sig_str:3}"
                        )

            if by_mouse: # sort p-value logging by percentage value
                tail_sig_str = f"{tail_sig_str}\n\t{linpla_name:6}: "
                order = np.argsort([vals[0] for vals in perc_p_vals])
                for i in order:
                    perc, p_val, sig_str = perc_p_vals[i]
                    perc_str = f"(~{perc}%)"
                    tail_sig_str = (
                        f"{tail_sig_str}{TAB}{perc_str:6} "
                        f"{p_val:.5f}{sig_str:3}"
                        )
                
        # add chance information
        if by_mouse:
            sub_ax.axhline(
                y=chance, ls=plot_helper_fcts.VDASH, c="k", lw=3.0, alpha=0.5, 
                zorder=-12
                )
        else:
            plot_util.plot_CI(sub_ax, CIs.T, med=CI_meds, 
                x=np.arange(n_linpla), width=0.45, med_rat=0.025, zorder=-12)

        logger.info(tail_sig_str, extra={"spacing": TAB})
    
    for sub_ax in fig.axes:
        sub_ax.tick_params(axis="x", which="both", bottom=False) 
        plot_util.set_ticks(
            sub_ax, min_tick=0, max_tick=n_linpla - 1, n=n_linpla, pad_p=0.2)
        sub_ax.set_xticklabels(linpla_names, rotation=90, weight="bold")

    ax[0, 0].set_ylabel("%", fontweight="bold")
    plot_util.set_interm_ticks(ax, 3, dim="y", weight="bold", share=True)

    # adjustment if tick interval is repeated in the negative
    if ax[0, 0].get_ylim()[0] < 0:
        ax[0, 0].set_ylim([ylims[0], ax[0, 0].get_ylim()[1]])

    return ax


#############################################
def plot_ex_roi_hists(ex_idx_df, sesspar, permpar, figpar, title=None):
    """
    plot_ex_roi_hists(ex_idx_df, sesspar, permpar, figpar)

    Plot example ROI histograms.

    Required args:
        - ex_idx_df (pd.DataFrame):
            dataframe with a row for the example ROI, and the following 
            columns, in addition to the basic sess_df columns:
            - rand_idx_binned (list): bin counts for the random ROI indices
            - bin_edges (list): first and last bin edge
            - CI_edges (list): confidence interval limit values
            - CI_perc (list): confidence interval percentile limits
        - sesspar (dict): 
            dictionary with keys of SessPar namedtuple
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters

    Optional args:
        - plot (str): 
            type of data to plot ("items" or "percs")
            default: "items"

    Returns:
        - ax (2D array): 
            array of subplots
    """
    
    ex_idx_df = copy.deepcopy(ex_idx_df) # add dummy binned_roi_idxs
    ex_idx_df["roi_idx_binned"] = [
        np.zeros_like(rand_idx_binned) 
        for rand_idx_binned in ex_idx_df["rand_idx_binned"].tolist()
    ]
    
    with gen_util.TempWarningFilter("invalid value", RuntimeWarning):
        ax = plot_idxs(
            ex_idx_df, sesspar, figpar, plot="items", title=title, size="tall", 
            density=True, n_bins=40)

    # adjust x axes
    for sub_ax in ax.reshape(-1):
        sub_ax.set_xticks([-0.5, 0, 0.5])
        sub_ax.set_xticklabels(["-0.5", "0", "0.5"])

    # add lines and labels
    for (line, plane), lp_df in ex_idx_df.groupby(["lines", "planes"]):
        li, pl, col, _ = plot_helper_fcts.get_line_plane_idxs(line, plane)

        if len(lp_df) == 0:
            continue
        elif len(lp_df) > 1:
            raise RuntimeError("Expected at most one row per line/plane.")
        row = lp_df.loc[lp_df.index[0]]

        sub_ax = ax[pl, li]
        xlims = sub_ax.get_xlim()

        # add CI markers
        for c, (CI_val, CI_perc) in enumerate(
            zip(row["CI_edges"], row["CI_perc"])
            ):

            sub_ax.axvline(
                CI_val, ls=plot_helper_fcts.VDASH, c="red", lw=3.0, alpha=1.0, 
                label=f"p{CI_perc:0.2f}")
            sub_ax.axvspan(
                CI_val, xlims[c], color=plot_helper_fcts.DARKRED, alpha=0.1, 
                lw=0, zorder=-13
                )
        
        ex_perc = row["roi_idx_percs"]

        sensitivity = misc_analys.get_sensitivity(permpar)
        sig_str = misc_analys.get_sig_symbol(
            ex_perc, percentile=True, sensitivity=sensitivity, 
            p_thresh=permpar["p_val"]
            )

        sub_ax.axvline(
            x=row["roi_idxs"], ls=plot_helper_fcts.VDASH, c=col, lw=3.0, 
            alpha=0.8, label=f"p{ex_perc:0.2f}{sig_str}"
            )

        sub_ax.axvline(
            x=0, ls=plot_helper_fcts.VDASH, c="k", lw=3.0, alpha=0.5
            )


        # reset the x limits
        sub_ax.set_xlim(xlims)

        sub_ax.legend()
    
    return ax


#############################################
def plot_tracked_idxs(idx_only_df, sesspar, figpar, title=None, wide=False):
    """
    plot_tracked_idxs(idx_only_df, sesspar, figpar)

    Plots tracked ROI USIs as individual lines.

    Required args:
        - idx_only_df (pd.DataFrame):
            dataframe with one row per (mouse/)session/line/plane, and the 
            following columns, in addition to the basic sess_df columns:
            - roi_idxs (list): index for each ROI

        - sesspar (dict): 
            dictionary with keys of SessPar namedtuple
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters

    Optional args:
        - title (str):
            plot title
            default: None
        - wide (bool):
            if True, subplots are wider
            default: False

    Returns:
        - ax (2D array): 
            array of subplots    
    """

    sess_ns = misc_analys.get_sess_ns(sesspar, idx_only_df)

    figpar = sess_plot_util.fig_init_linpla(figpar)

    figpar["init"]["sharey"] = "row"
    figpar["init"]["subplot_hei"] = 4.1
    figpar["init"]["subplot_wid"] = 2.5
    figpar["init"]["gs"] = {"wspace": 0.25, "hspace": 0.2}
    if wide:
        figpar["init"]["subplot_wid"] = 3.3
        figpar["init"]["gs"]["wspace"] = 0.25

    fig, ax = plot_util.init_fig(plot_helper_fcts.N_LINPLA, **figpar["init"])

    if title is not None:
        fig.suptitle(title, y=0.98, weight="bold")

    for (line, plane), lp_df in idx_only_df.groupby(["lines", "planes"]):
        li, pl, col, _ = plot_helper_fcts.get_line_plane_idxs(line, plane)
        sub_ax = ax[pl, li]

        # mouse_ns
        lp_mouse_ns = sorted(lp_df["mouse_ns"].unique())

        lp_data = []
        for mouse_n in lp_mouse_ns:
            mouse_df = lp_df.loc[lp_df["mouse_ns"] == mouse_n]
            nrois = mouse_df["nrois"].unique()
            if len(nrois) != 1:
                raise RuntimeError(
                    "Each mouse in idx_stats_df should retain the same number "
                    " of ROIs across sessions.")
            
            mouse_data = np.full((len(sess_ns), nrois[0]), np.nan)
            for s, sess_n in enumerate(sess_ns):
                rows = mouse_df.loc[mouse_df["sess_ns"] == sess_n]
                if len(rows) == 1:
                    mouse_data[s] = rows.loc[rows.index[0], "roi_idxs"]
                elif len(rows) > 1:
                    raise RuntimeError(
                        "Expected 1 row per line/plane/session/mouse."
                        )
            lp_data.append(mouse_data)

        lp_data = np.concatenate(lp_data, axis=1)

        sub_ax.axhline(
            y=0, ls=plot_helper_fcts.HDASH, c="k", lw=3.0, alpha=0.5, 
            zorder=-13
            )
        sub_ax.plot(sess_ns, lp_data, color=col, lw=2, alpha=0.3)
    
    # Add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(
        ax, datatype="roi", xticks=sess_ns, ylab="", kind="reg"
        )

    for sub_ax in ax.reshape(-1):
        xticks = sub_ax.get_xticks()
        plot_util.set_ticks(
            sub_ax, "x", np.min(xticks), np.max(xticks), n=len(xticks), 
            pad_p=0.2
            )
    
    return ax


#############################################
def plot_tracked_idx_stats(idx_stats_df, sesspar, figpar, permpar=None,
                           absolute=True, between_sess_sig=True, 
                           by_mouse=False, title=None, wide=False):
    """
    plot_tracked_idx_stats(idx_stats_df, sesspar, figpar)

    Plots tracked ROI USI statistics.

    Required args:
        - idx_stats_df (pd.DataFrame):
            dataframe with one row per session, and the following columns, in 
            addition to the basic sess_df columns:
            - roi_idxs (list): index statistics
            or
            - abs_roi_idxs (list): absolute index statistics
        - sesspar (dict): 
            dictionary with keys of SessPar namedtuple
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters

    Optional args:
        - permpar (dict): 
            dictionary with keys of PermPar namedtuple. Required if 
            between_sess_sig is True.
            default: None
        - absolute (bool):
            if True, data statistics are on absolute ROI indices
            default: True
        - between_sess_sig (bool):
            if True, significance between sessions is logged and plotted
            default: True
        - by_mouse (bool):
            if True, plotting is done per mouse
            default: False
        - title (str):
            plot title
            default: None
        - wide (bool):
            if True, subplots are wider
            default: False

    Returns:
        - ax (2D array): 
            array of subplots    
    """

    sess_ns = misc_analys.get_sess_ns(sesspar, idx_stats_df)

    figpar = sess_plot_util.fig_init_linpla(figpar)

    figpar["init"]["sharey"] = "row"
    figpar["init"]["subplot_hei"] = 4.1
    figpar["init"]["subplot_wid"] = 2.6
    figpar["init"]["gs"] = {"wspace": 0.25, "hspace": 0.2}
    if wide:
        figpar["init"]["subplot_wid"] = 3.3
        figpar["init"]["gs"]["wspace"] = 0.25

    fig, ax = plot_util.init_fig(plot_helper_fcts.N_LINPLA, **figpar["init"])

    if title is not None:
        fig.suptitle(title, y=0.98, weight="bold")
    
    data_col = "roi_idx_stats"
    if absolute:
        data_col = f"abs_{data_col}"
    
    if data_col not in idx_stats_df.columns:
        raise KeyError(f"Expected to find {data_col} in idx_stats_df columns.")

    for (line, plane), lp_df in idx_stats_df.groupby(["lines", "planes"]):
        li, pl, col, dash = plot_helper_fcts.get_line_plane_idxs(line, plane)
        sub_ax = ax[pl, li]

        sub_ax.axhline(
            y=0, ls=plot_helper_fcts.HDASH, c="k", lw=3.0, alpha=0.5, 
            zorder=-13
            )

        mouse_ns = ["any"]
        if by_mouse:
            mouse_ns = sorted(lp_df["mouse_ns"].unique())

        for mouse_n in mouse_ns:
            sub_df = lp_df
            if by_mouse:
                sub_df = lp_df.loc[lp_df["mouse_ns"] == mouse_n]
            
            sess_indices = []
            sub_sess_ns = []

            for sess_n in sess_ns:
                rows = sub_df.loc[sub_df["sess_ns"] == sess_n]
                if len(rows) == 1:
                    sess_indices.append(rows.index[0])
                    sub_sess_ns.append(sess_n)

            data = np.asarray([sub_df.loc[i, data_col] for i in sess_indices])

            # plot errorbars
            alpha = 0.6 if by_mouse else 0.8
            plot_util.plot_errorbars(
                sub_ax, data[:, 0], data[:, 1:].T, sub_sess_ns, color=col, 
                alpha=alpha, xticks="auto", line_dash=dash
                )

    if between_sess_sig:
        if permpar is None:
            raise ValueError(
                "If 'between_sess_sig' is True, must provide permpar."
                )
        if by_mouse:
            raise NotImplementedError(
                "Plotting between session statistical signifiance is not "
                "implemented if 'by_mouse' if True."
                )

        seq_plots.add_between_sess_sig(
            ax, idx_stats_df, permpar, data_col=data_col
            )
    
    # Add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(
        ax, datatype="roi", xticks=sess_ns, ylab="", kind="reg"
        )
    
    return ax

