"""
seq_plots.py

This script contains functions for plotting sequence analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

import numpy as np

from util import logger_util, gen_util, plot_util, math_util
from sess_util import sess_plot_util
from analysis import misc_analys
from plot_fcts import plot_helper_fcts

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def plot_traces(sub_ax, time_values, trace_st, split="by_exp", col="k", 
                lab=True, ls=None, exp_col="gray", hline=True):
    """
    plot_traces(sub_ax, time_values, trace_st)

    Plot data trace splits (a single set).

    Required args:
        - sub_ax (plt Axis subplot): 
            subplot
        - time_values (array-like): 
            values for each frame, in seconds (only from 0, if lock)
        - trace_st (3D array): 
            trace statistics
            dims: split x frame x stats

    Optional args:
        - split (str):
            data split, e.g. "exp_lock", "unexp_lock", "stim_onset" or 
            "stim_offset"
            default: False
        - col (str): 
            colour for non-regular/non-locked data
            default: "k"
        - lab (bool): 
            if True, data label is included for legend
            default: True
        - ls (float): 
            trace line style
            default: None
        - exp_col (str): 
            color for expected data
            default: "gray"
        - hline (bool):
            if True, horizontal line at y=0 is added
            default: False
    """

    lock = (split != "by_exp")

    names = ["exp", "unexp"]
    if lock and "stim" in split:
        names = ["off", "on"]

    if lock:
        cols = [col, col]
    else:
        cols = [exp_col, col]

    alphas_line = [0.8, 0.8]
    alphas_shade = [0.5, 0.5]
    
    # alpha exceptions for red
    if col in ["red", plot_util.LINCLAB_COLS["red"]]: 
        updates = [0, 1] if lock else [1]
        for update in updates:        
            alphas_line[update] = 0.75
            alphas_shade[update] = 0.2
    
    time_values = np.asarray(time_values)

    # horizontal 0 line
    if hline:
        all_rows = True
        if not sub_ax.is_last_row() or all_rows:
            alpha = 0.5
        else:
            # to prevent infinite expansion bug
            alpha = 0

        sub_ax.axhline(
            y=0, ls=plot_helper_fcts.HDASH, c="k", lw=3.0, alpha=alpha, 
            zorder=-13
            )

    # vertical 0 line    
    if lock or (time_values.min() <= 0 and time_values.max() >= 0):
        alpha = 0.5
    else:
        alpha = 0 # same as before - prevents infinite expansion
    sub_ax.axvline(x=0, ls=plot_helper_fcts.VDASH, c="k", lw=3.0, alpha=alpha, 
        zorder=-13)

    # x ticks
    xticks = "auto"
    if lock:
        xticks = np.linspace(-np.max(time_values), np.max(time_values), 5)
        rev_time_values = time_values[::-1] * -1

    trace_st = np.asarray(trace_st)
    for i, (col, name) in enumerate(zip(cols, names)):
        label = name if lab and not lock else None

        if lock and f"_{name}" not in f"_{split}": # hack, since "exp" is in "unexp"
            time_values_use = rev_time_values
        else:
            time_values_use = time_values

        if split in ["exp_lock", "stim_offset"]:
            i = 1 - i # data ordered as [non-reg, reg] instead of vv
        
        plot_util.plot_traces(sub_ax, time_values_use, trace_st[i, :, 0], 
            trace_st[i, :, 1:], label=label, alpha_line=alphas_line[i], 
            alpha=alphas_shade[i], color=col, xticks=xticks, ls=ls)
            
            
#############################################
def plot_sess_traces(data_df, analyspar, sesspar, figpar, 
                     trace_col="trace_stats", row_col="sess_ns", 
                     row_order=None, split="by_exp", title=None, size="reg"):
    """
    plot_sess_traces(data_df, analyspar, sesspar, figpar) 
    
    Plots traces from dataframe.

    Required args:
        - data_df (pd.DataFrame):
            traces data frame with, in addition to the basic sess_df columns, 
            columns specified by trace_col, row_col, and a "time_values" column
        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters

    Optional args:
        - trace_col (str):
             dataframe column containing trace statistics, as 
             split x ROIs x frames x stats 
             default: "trace_stats"
        - row_col (str):
            dataframe column specifying the variable that defines rows 
            within each line/plane
            default: "sess_ns"
        - row_order (list):
            ordered list specifying the order in which to plot from row_col.
            If None, automatic sorting order is used.
            default: None 
        - split (str):
            data split, e.g. "exp_lock", "unexp_lock", "stim_onset" or 
            "stim_offset"
            default: False
        - title (str):
            plot title
            default: None
        - size (str):
            subplot sizes
            default: "reg"

    Returns:
        - ax (2D array): 
            array of subplots
    """
    
    # retrieve session numbers, and infer row_order, if necessary
    sess_ns = None
    if row_col == "sess_ns":
        sess_ns = row_order
        if row_order is None:
            row_order = misc_analys.get_sess_ns(sesspar, data_df)

    elif row_order is None:
        row_order = data_df[row_col].unique()

    figpar = sess_plot_util.fig_init_linpla(
        figpar, kind="traces", n_sub=len(row_order), sharey=False
        )

    if size == "small":
        figpar["init"]["subplot_hei"] = 1.5
        figpar["init"]["subplot_wid"] = 3.5
    elif size == "wide":
        figpar["init"]["subplot_wid"] = 4
    elif size != "reg":
        gen_util.accepted_values_error("size", size, ["small", "wide", "reg"])

    fig, ax = plot_util.init_fig(len(row_order) * 4, **figpar["init"])
    if title is not None:
        fig.suptitle(title, y=1.0, weight="bold")

    for (line, plane), lp_df in data_df.groupby(["lines", "planes"]):
        li, pl, col, dash = plot_helper_fcts.get_line_plane_idxs(line, plane)

        for r, row_val in enumerate(row_order):
            rows = lp_df.loc[lp_df[row_col] == row_val]
            if len(rows) == 0:
                continue
            elif len(rows) > 1:
                raise RuntimeError(
                    "Expected row_order instances to be unique per line/plane."
                    )
            row = rows.loc[rows.index[0]]

            sub_ax = ax[r + pl * len(row_order), li]

            if line == "L2/3-Cux2":
                exp_col = "darkgray" # oddly, lighter than gray
            else:
                exp_col = "gray"

            plot_traces(
                sub_ax, row["time_values"], row[trace_col], split=split, 
                col=col, ls=dash, exp_col=exp_col, lab=False
                )
            
    for sub_ax in ax.reshape(-1):
        plot_util.set_minimal_ticks(sub_ax, dim="y")

    sess_plot_util.format_linpla_subaxes(ax, fluor=analyspar["fluor"], 
        area=False, datatype="roi", sess_ns=sess_ns, xticks=None, 
        kind="traces", modif_share=False)

   # fix x ticks and lims
    plot_util.set_interm_ticks(ax, 3, dim="x", fontweight="bold")
    xlims = [np.min(row["time_values"]), np.max(row["time_values"])]
    if split != "by_exp":
        xlims = [-xlims[1], xlims[1]]
    sub_ax.set_xlim(xlims)

    return ax


#############################################
def add_between_sess_sig(ax, data_df, permpar, data_col="diff_stats", 
                         highest=None, ctrl=False, p_val_prefix=False, 
                         dry_run=False):
    """
    add_between_sess_sig(ax, data_df, permpar)

    Plot significance markers for significant comparisons between sessions.

    Required args:
        - ax (plt Axis): 
            axis
        - data_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - {data_col} (list): data stats (me, err)
            for session comparisons, e.g. 1v2:
            - p_vals_{}v{} (float): p-value for differences between sessions, 
                corrected for multiple comparisons and tails

        - permpar (dict): 
            dictionary with keys of PermPar namedtuple

    Optional args:
        - data_col (str):
            data column name in data_df
            default: "diff_stats"
        - highest (list):
            highest point for each line/plane, in order
            default: None
        - ctrl (bool): 
            if True, significance symbols should use control colour and symbol
            default: False
        - p_val_prefix (bool):
            if True, p-value columns start with data_col as a prefix 
            "{data_col}_p_vals_{}v{}".
            default: False
        - dry_run (bool):
            if True, a dry-run is done to get highest values, but nothing is 
            plotted or logged.
            default: False

    Returns:
    - highest (list):
        highest point for each line/plane, in order, after plotting
    """

    sensitivity = misc_analys.get_sensitivity(permpar)
    comp_info = misc_analys.get_comp_info(permpar)

    prefix = f"{data_col}_" if p_val_prefix else ""

    if not dry_run:
        logger.info(
            f"Corrected p-values ({comp_info}):", extra={"spacing": "\n"}
            )

    for pass_n in [0, 1]: # add significance markers on the second pass
        linpla_grps = list(data_df.groupby(["lines", "planes"]))
        if highest is None:
            highest = [0] * len(linpla_grps)
        elif len(highest) != len(linpla_grps):
            raise ValueError("If highest is provided, it must contain as "
                "many values as line/plane groups in data_df.")

        for l, ((line, plane), lp_df) in enumerate(linpla_grps):
            li, pl, col, _ = plot_helper_fcts.get_line_plane_idxs(line, plane)
            line_plane_name = plot_helper_fcts.get_line_plane_name(line, plane)
            sub_ax = ax[pl, li]

            if ctrl:
                col = "gray"

            lp_sess_ns = np.sort(lp_df["sess_ns"].unique())
            for sess_n in lp_sess_ns:
                rows = lp_df.loc[lp_df["sess_ns"] == sess_n]
                if len(rows) != 1:
                    raise RuntimeError("Expected 1 row per line/plane/session.")    

            sig_p_vals, sig_strs, sig_xs = [], [], []
            lp_sig_str = f"{line_plane_name:6} (between sessions):"
            for i, sess_n1 in enumerate(lp_sess_ns):
                row_1s = lp_df.loc[lp_df["sess_ns"] == sess_n1]
                for sess_n2 in lp_sess_ns[i + 1: ]:
                    row_2s = lp_df.loc[lp_df["sess_ns"] == sess_n2]
                    if len(row_1s) != 1 or len(row_2s) != 1:
                        raise RuntimeError(
                            "Expected exactly one row per session."
                            )
                    row1 = row_1s.loc[row_1s.index[0]]
                    row2 = row_2s.loc[row_2s.index[0]]

                    row1_highest = row1[data_col][0] + row1[data_col][-1]
                    row2_highest = row2[data_col][0] + row2[data_col][-1]      
                    highest[l] = np.nanmax(
                        [highest[l], row1_highest, row2_highest]
                        )
                    
                    if dry_run:
                        continue
                    
                    p_val = row1[
                        f"{prefix}p_vals_{int(sess_n1)}v{int(sess_n2)}"
                        ]
                    side = np.sign(row2[data_col][0] - row1[data_col][0])

                    sig_str = misc_analys.get_sig_symbol(
                        p_val, sensitivity=sensitivity, side=side, 
                        tails=permpar["tails"], ctrl=ctrl)

                    if len(sig_str):
                        sig_p_vals.append(p_val)
                        sig_strs.append(sig_str)
                        sig_xs.append([sess_n1, sess_n2])
                    
                    lp_sig_str = (
                        f"{lp_sig_str}{TAB} S{sess_n1}v{sess_n2}: "
                        f"{p_val:.5f}{sig_str:3}"
                        )

            if dry_run:
                continue

            n = len(sig_p_vals)
            ylims = sub_ax.get_ylim()
            prop = np.diff(ylims)[0] / 8.0

            if pass_n == 0:
                logger.info(lp_sig_str, extra={"spacing": TAB})
                if n == 0:
                    continue

                # count number of significant comparisons, and adjust y limits
                ylims = [
                    ylims[0], np.nanmax(
                        [ylims[1], highest[l] + prop * (n + 1)])
                        ]
                sub_ax.set_ylim(ylims)
            else:
                if n == 0:
                    continue

                if ctrl:
                    mark_rel_y = 0.22
                    fontsize = 14
                else:
                    mark_rel_y = 0.18
                    fontsize = 20

                # add significance markers sequentially, on second pass
                y_pos = highest[l]
                for s, (p_val, sig_str, sig_x) in enumerate(
                    zip(sig_p_vals, sig_strs, sig_xs)
                ):  
                    y_pos = highest[l] + (s + 1) * prop
                    plot_util.plot_barplot_signif(
                        sub_ax, sig_x, y_pos, rel_y=0.11, color=col, lw=3, 
                        mark_rel_y=mark_rel_y, mark=sig_str, fontsize=fontsize
                        )
                highest[l] = np.nanmax([highest[l], y_pos])

                if y_pos > ylims[1]:
                    sub_ax.set_ylim([ylims[0], y_pos * 1.1])

    return highest


#############################################
def plot_sess_data(data_df, analyspar, sesspar, permpar, figpar, 
                   between_sess_sig=True, data_col="diff_stats", 
                   decoder_data=False, title=None, wide=False):
    """
    plot_sess_data(data_df, analyspar, sesspar, permpar, figpar)

    Plots errorbar data across sessions.

    Required args:
        - data_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - {data_key} (list): data stats (me, err)
            - null_CIs (list): adjusted null CI for data
            - raw_p_vals (float): uncorrected p-value for data within 
                sessions
            - p_vals (float): p-value for data within sessions, 
                corrected for multiple comparisons and tails
            for session comparisons, e.g. 1v2:
            - raw_p_vals_{}v{} (float): uncorrected p-value for data 
                differences between sessions 
            - p_vals_{}v{} (float): p-value for data between sessions, 
                corrected for multiple comparisons and tails

        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
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
        - between_sess_sig (bool):
            if True, significance between sessions is logged and plotted
            default: True         
        - data_col (str):
            dataframe column in which data to plot is stored
            default: "diff_stats"
        - decoder_data (bool):
            if True, data plotted is decoder data
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

    sess_ns = misc_analys.get_sess_ns(sesspar, data_df)

    figpar = sess_plot_util.fig_init_linpla(figpar)
    
    sharey = True if decoder_data else "row"
    figpar["init"]["sharey"] = sharey
    figpar["init"]["subplot_hei"] = 4.0
    figpar["init"]["subplot_wid"] = 2.8
    if wide:
        figpar["init"]["subplot_wid"] = 3.7

    fig, ax = plot_util.init_fig(plot_helper_fcts.N_LINPLA, **figpar["init"])

    if title is not None:
        fig.suptitle(title, y=0.97, weight="bold")

    sensitivity = misc_analys.get_sensitivity(permpar)
    comp_info = misc_analys.get_comp_info(permpar)

    for pass_n in [0, 1]: # add significance markers on the second pass
        if pass_n == 1:
            logger.info(
                f"Corrected p-values ({comp_info}):", 
                extra={"spacing": "\n"}
                )
        for (line, plane), lp_df in data_df.groupby(["lines", "planes"]):
            li, pl, col, dash = plot_helper_fcts.get_line_plane_idxs(
                line, plane
                )
            line_plane_name = plot_helper_fcts.get_line_plane_name(line, plane)
            sub_ax = ax[pl, li]

            sess_indices = []
            lp_sess_ns = []
            for sess_n in sess_ns:
                rows = lp_df.loc[lp_df["sess_ns"] == sess_n]
                if len(rows) == 1:
                    sess_indices.append(rows.index[0])
                    lp_sess_ns.append(sess_n)
                elif len(rows) > 1:
                    raise RuntimeError("Expected 1 row per line/plane/session.")

            data = np.asarray([lp_df.loc[i, data_col] for i in sess_indices])

            if pass_n == 0:
                # plot errorbars
                plot_util.plot_errorbars(
                    sub_ax, data[:, 0], data[:, 1:].T, lp_sess_ns, color=col, 
                    alpha=0.8, xticks="auto", line_dash=dash
                    )

            if pass_n == 1:
                # plot CIs
                CIs = np.asarray(
                    [lp_df.loc[i, "null_CIs"] for i in sess_indices]
                    )
                CI_meds = CIs[:, 1]
                CIs = CIs[:, np.asarray([0, 2])]

                plot_util.plot_CI(sub_ax, CIs.T, med=CI_meds, x=lp_sess_ns, 
                    width=0.45, color="lightgrey", med_col="gray", med_rat=0.03, 
                    zorder=-12)

                # add significance markers within sessions
                y_maxes = data[:, 0] + data[:, -1]
                sides = [
                    np.sign(sub[0] - CI_med) 
                    for sub, CI_med in zip(data, CI_meds)
                    ]
                p_vals_corr = [lp_df.loc[i, "p_vals"] for i in sess_indices]
                lp_sig_str = f"{line_plane_name:6} (within session):"
                for s, sess_n in enumerate(lp_sess_ns):
                    sig_str = misc_analys.get_sig_symbol(
                        p_vals_corr[s], sensitivity=sensitivity, side=sides[s], 
                        tails=permpar["tails"])

                    if len(sig_str):
                        plot_util.add_signif_mark(sub_ax, sess_n, y_maxes[s], 
                            rel_y=0.15, color=col, mark=sig_str)  

                    lp_sig_str = (
                        f"{lp_sig_str}{TAB} S{sess_n}: "
                        f"{p_vals_corr[s]:.5f}{sig_str:3}"
                        )

                logger.info(lp_sig_str, extra={"spacing": TAB})
        
    if between_sess_sig:
        add_between_sess_sig(ax, data_df, permpar, data_col=data_col)

    area, ylab = True, None
    if decoder_data:
        area = False
        if "balanced" in data_col:
            ylab = "Balanced accuracy (%)" 
        else:
            ylab = "Accuracy %"

    sess_plot_util.format_linpla_subaxes(ax, fluor=analyspar["fluor"], 
        area=area, ylab=ylab, datatype="roi", sess_ns=sess_ns, kind="reg", 
        xticks=sess_ns, modif_share=False)

    return ax


#############################################
def plot_ex_gabor_roi_traces(sub_ax, df_row, col="k", dash=None):
    """
    plot_ex_gabor_roi_traces(sub_ax, df_row)

    Plots example Gabor traces in a subplot.

    Required args:
        - sub_ax (plt subplot):
            subplot
        - df_row (pd.Series): 
            pandas DataFrame row to plot from

    Optional args:
        - col (str):
            trace statistic color
            default: "k"
        - dash (tuple or None):
            dash pattern to use for trace statistic
            default: None

    Returns:
        - ylims (list): 
            rounded y limits to use for the subplot
    """

    sub_ax.axvline(x=0, ls=plot_helper_fcts.VDASH, c="k", lw=3.0, 
        alpha=0.5, zorder=-12)

    time_values = df_row["time_values"]
    traces = np.asarray(df_row["traces"])
    trace_stat = df_row["trace_stat"]

    alpha = 2 / np.log(len(traces))

    sub_ax.plot(
        time_values, traces.T, lw=1, color="gray", alpha=alpha, zorder=-13
        )

    sub_ax.plot(
        time_values, trace_stat, lw=None, color=col, alpha=0.8, ls=dash
        )
    
    # use percentiles for bounds, asymmetrically
    ymin = np.min([np.min(trace_stat), np.percentile(traces, 0.01)])
    ymax = np.max([np.max(trace_stat), np.percentile(traces, 99.99)])

    yrange = ymax - ymin
    ymin_use = ymin - yrange * 0.05
    ymax_use = ymax + yrange * 0.05

    ylims = plot_util.rounded_lims([ymin_use, ymax_use])

    return ylims



#############################################
def plot_ex_gabor_traces(ex_traces_df, stimpar, figpar, title=None):
    """
    plot_ex_gabor_traces(ex_traces_df, stimpar, figpar)

    Plots example Gabor traces.

    Required args:
        - ex_traces_df (pd.DataFrame):
            dataframe with a row for each ROI, and the following columns, 
            in addition to the basic sess_df columns: 
            - time_values (list): values for each frame, in seconds
            - roi_ns (list): selected ROI number
            - traces (list): selected ROI sequence traces, dims: seq x frames
            - trace_stat (list): selected ROI trace mean or median
        - stimpar (dict):
            dictionary with keys of StimPar namedtuple
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
    
    Optional args:
        - title (str):
            plot title
            default: None

    Returns:
        - ax (2D array): 
            array of subplots
    """

    if stimpar["stimtype"] != "gabors":
        raise ValueError("Expected stimpar['stimtype'] to be 'gabors'.")

    group_columns = ["lines", "planes"]
    n_per = np.max(
        [len(lp_df) for _, lp_df in ex_traces_df.groupby(group_columns)]
        )
    per_rows, per_cols = math_util.get_near_square_divisors(n_per)
    n_per = per_rows * per_cols

    figpar = sess_plot_util.fig_init_linpla(
        figpar, kind="traces", n_sub=per_rows
        )
    figpar["init"]["subplot_hei"] = 1.45
    figpar["init"]["subplot_wid"] = 2.85
    figpar["init"]["ncols"] = per_cols * 2

    fig, ax = plot_util.init_fig(
        plot_helper_fcts.N_LINPLA * n_per, **figpar["init"]
        )
    if title is not None:
        fig.suptitle(title, y=1.03, weight="bold", fontsize=22)

    ylims = np.full(ax.shape + (2, ), np.nan)
    logger.info("Plotting individual traces...", extra={"spacing": TAB})
    for (line, plane), lp_df in ex_traces_df.groupby(["lines", "planes"]):
        li, pl, col, dash = plot_helper_fcts.get_line_plane_idxs(line, plane)
        for i, idx in enumerate(lp_df.index):
            row_idx = int(pl * per_rows + i % per_rows)
            col_idx = int(li * per_cols + i // per_rows)
            sub_ax = ax[row_idx, col_idx]

            ylims[row_idx, col_idx] = plot_ex_gabor_roi_traces(
                sub_ax, 
                lp_df.loc[idx],
                col=col,
                dash=dash
            )

        time_values = np.asarray(lp_df.loc[lp_df.index[-1], "time_values"])
        
    sess_plot_util.format_linpla_subaxes(ax, fluor="dff", 
        area=False, datatype="roi", sess_ns=None, xticks=None, kind="traces", 
        modif_share=False)

   # fix x ticks and lims
    plot_util.set_interm_ticks(ax, 3, dim="x", fontweight="bold")
    sub_ax.set_xlim([time_values.min(), time_values.max()])

    # reset y limits
    for r in range(ax.shape[0]):
        for c in range(ax.shape[1]):
            if not np.isfinite(ylims[r, c].sum()):
                continue
            ax[r, c].set_ylim(ylims[r, c])
            ax[r, c].set_yticks(ylims[r, c])

    plot_util.set_interm_ticks(ax, 2, dim="y", share=False, weight="bold")  

    # rasterize the gray lines
    logger.info("Rasterizing individual traces...", extra={"spacing": TAB})
    for sub_ax in ax.reshape(-1):
        sub_ax.set_rasterization_zorder(-12)

    return ax


#############################################
def plot_rel_resp_data(rel_resp_df, analyspar, sesspar, stimpar, permpar, 
                       figpar, title=None, wide=False):
    """
    plot_rel_resp_data((rel_resp_df, analyspar, sesspar, stimpar, permpar, 
                       figpar)

    Plots relative response errorbar data across sessions.

    Required args:
        - rel_resp_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - rel_reg or rel_exp (list): 
                data stats for regular or expected data (me, err)
            - rel_unexp (list): data stats for unexpected data (me, err)
            for reg/exp/unexp data types, session comparisons, e.g. 1v2:
            - {data_type}_raw_p_vals_{}v{} (float): uncorrected p-value for 
                data differences between sessions 
            - {data_type}_p_vals_{}v{} (float): p-value for data between 
                sessions, corrected for multiple comparisons and tails

        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - sesspar (dict):
            dictionary with keys of SessPar namedtuple
        - stimpar (dict):
            dictionary with keys of StimPar namedtuple
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
        - wide (bool):
            if True, subplots are wider
            default: False

    Returns:
        - ax (2D array): 
            array of subplots
    """

    sess_ns = misc_analys.get_sess_ns(sesspar, rel_resp_df)

    figpar = sess_plot_util.fig_init_linpla(figpar)
    
    figpar["init"]["sharey"] = "row"
    figpar["init"]["subplot_hei"] = 4.0
    figpar["init"]["subplot_wid"] = 3.0
    if wide:
        figpar["init"]["subplot_wid"] = 3.7

    fig, ax = plot_util.init_fig(plot_helper_fcts.N_LINPLA, **figpar["init"])

    if title is not None:
        fig.suptitle(title, y=1.0, weight="bold")

    if stimpar["stimtype"] == "gabors":
        data_types = ["rel_reg", "rel_unexp"]
    elif stimpar["stimtype"] == "bricks":
        data_types = ["rel_exp", "rel_unexp"]
    else:
        gen_util.accepted_values_error(
            "stimpar['stimtype']", stimpar["stimtype"], ["gabors", "bricks"]
            )

    for (line, plane), lp_df in rel_resp_df.groupby(["lines", "planes"]):
        li, pl, col, dash = plot_helper_fcts.get_line_plane_idxs(line, plane)
        sub_ax = ax[pl, li]

        sess_indices = []
        lp_sess_ns = []
        for sess_n in sess_ns:
            rows = lp_df.loc[lp_df["sess_ns"] == sess_n]
            if len(rows) == 1:
                sess_indices.append(rows.index[0])
                lp_sess_ns.append(sess_n)
            elif len(rows) > 1:
                raise RuntimeError("Expected 1 row per line/plane/session.")

        sub_ax.axhline(
            y=1, ls=plot_helper_fcts.HDASH, c="k", lw=3.0, alpha=0.5, zorder=-13
            )

        colors = ["gray", col]
        fmts = ["-d", "-o"]
        alphas = [0.6, 0.8]
        ms = [12, None]
        for d, data_type in enumerate(data_types):
            data = np.asarray([lp_df.loc[i, data_type] for i in sess_indices])
            plot_util.plot_errorbars(
                sub_ax, data[:, 0], data[:, 1:].T, lp_sess_ns, color=colors[d], 
                alpha=alphas[d], ms=ms[d], fmt=fmts[d], line_dash=dash)
            
    highest = None
    for dry_run in [True, False]: # to get correct data heights
        for data_type in data_types:
            ctrl = ("unexp" not in data_type)
            highest = add_between_sess_sig(
                ax, rel_resp_df, permpar, data_col=data_type, highest=highest, 
                ctrl=ctrl, p_val_prefix=True, dry_run=dry_run)
            # print(highest)
            if not dry_run:
                highest = [val * 1.05 for val in highest] # increment a bit
            

    sess_plot_util.format_linpla_subaxes(ax, fluor=analyspar["fluor"], 
        datatype="roi", sess_ns=sess_ns, kind="reg", xticks=sess_ns, 
        modif_share=False)

    return ax

    