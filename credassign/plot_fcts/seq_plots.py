"""
seq_plots.py

This script contains functions for plotting sequence analyses.

Authors: Colleen Gillon

Date: February 2023

Note: this code was aggregated from https://github.com/colleenjg/OpenScope_CA_Analysis.
"""

import numpy as np

from credassign.util import gen_util, logger_util, plot_util, math_util
from credassign.analysis import misc_analys
from credassign.plot_fcts import plot_helper_fcts


TAB = "    "


logger = logger_util.get_module_logger(name=__name__)


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
    if col in ["red", "#eb3920"]: 
        updates = [0, 1] if lock else [1]
        for update in updates:        
            alphas_line[update] = 0.75
            alphas_shade[update] = 0.2
    
    time_values = np.asarray(time_values)

    # horizontal 0 line
    if hline:
        all_rows = True
        if not plot_util.is_last_row(sub_ax) or all_rows:
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
            subplot sizes ("reg", "wide", "small")
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

    figpar = plot_util.fig_init_linpla(
        figpar, kind="traces", n_sub=len(row_order), sharey=False
        )

    if size == "small":
        figpar["init"]["subplot_hei"] = 1.51
        figpar["init"]["subplot_wid"] = 3.7
    elif size == "wide":
        figpar["init"]["subplot_hei"] = 1.36
        figpar["init"]["subplot_wid"] = 4.8
        figpar["init"]["gs"] = {"wspace": 0.3, "hspace": 0.5}
    elif size == "reg":
        figpar["init"]["subplot_hei"] = 1.36
        figpar["init"]["subplot_wid"] = 3.4
    else:
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
        plot_util.set_minimal_ticks(sub_ax, axis="y")

    plot_util.format_linpla_subaxes(ax, fluor=analyspar["fluor"], 
        area=False, datatype="roi", sess_ns=sess_ns, xticks=None, 
        kind="traces", modif_share=False)

   # fix x ticks and lims
    plot_util.set_interm_ticks(ax, 3, axis="x", fontweight="bold")
    xlims = [np.min(row["time_values"]), np.max(row["time_values"])]
    if split != "by_exp":
        xlims = [-xlims[1], xlims[1]]
    sub_ax.set_xlim(xlims)

    return ax


#############################################
def plot_ex_roi_traces(sub_ax, df_row, col="k", dash=None, zorder=-13):
    """
    plot_ex_roi_traces(sub_ax, df_row)

    Plots example traces in a subplot.

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
        - zorder (int):
            zorder for the individual traces 
            (zorder + 1 used for vertical dash line at 0)
            default: -13

    Returns:
        - ylims (list): 
            rounded y limits to use for the subplot
    """

    sub_ax.axvline(x=0, ls=plot_helper_fcts.VDASH, c="k", lw=3.0, 
        alpha=0.5, zorder=zorder + 1)

    time_values = df_row["time_values"]
    traces_sm = np.asarray(df_row["traces_sm"])
    trace_stat = df_row["trace_stats"]

    alpha = 2 / np.log(len(traces_sm))

    sub_ax.plot(
        time_values, traces_sm.T, lw=1, color="gray", alpha=alpha, 
        zorder=zorder
        )

    sub_ax.plot(
        time_values, trace_stat, lw=None, color=col, alpha=0.8, ls=dash
        )
    
    # use percentiles for bounds, asymmetrically
    ymin_near = np.min([np.min(trace_stat), np.percentile(traces_sm, 0.01)])
    ymax_near = np.max([np.max(trace_stat), np.percentile(traces_sm, 99.99)])

    yrange_near = ymax_near - ymin_near
    ymin = ymin_near - yrange_near * 0.05
    ymax = ymax_near + yrange_near * 0.05

    ylims = [ymin, ymax]

    return ylims


#############################################
def plot_ex_traces(ex_traces_df, stimpar, figpar, title=None):
    """
    plot_ex_traces(ex_traces_df, stimpar, figpar)

    Plots example traces.

    Required args:
        - ex_traces_df (pd.DataFrame):
            dataframe with a row for each ROI, and the following columns, 
            in addition to the basic sess_df columns: 
            - time_values (list): values for each frame, in seconds
            - roi_ns (list): selected ROI number
             - traces_sm (list): selected ROI sequence traces, smoothed, with 
                dims: seq x frames
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

    group_columns = ["lines", "planes"]
    n_per = np.max(
        [len(lp_df) for _, lp_df in ex_traces_df.groupby(group_columns)]
        )
    per_rows, per_cols = math_util.get_near_square_divisors(n_per)
    n_per = per_rows * per_cols

    figpar = plot_util.fig_init_linpla(
        figpar, kind="traces", n_sub=per_rows
        )
    figpar["init"]["subplot_hei"] = 1.25
    figpar["init"]["subplot_wid"] = 2.2
    figpar["init"]["ncols"] = per_cols * 2

    fig, ax = plot_util.init_fig(
        plot_helper_fcts.N_LINPLA * n_per, **figpar["init"]
        )
    if title is not None:
        fig.suptitle(title, y=1.03, weight="bold")

    ylims = np.full(ax.shape + (2, ), np.nan)
    
    logger.info("Plotting individual traces...", extra={"spacing": TAB})
    raster_zorder = -12
    for (line, plane), lp_df in ex_traces_df.groupby(["lines", "planes"]):
        li, pl, col, dash = plot_helper_fcts.get_line_plane_idxs(line, plane)
        for i, idx in enumerate(lp_df.index):
            row_idx = int(pl * per_rows + i % per_rows)
            col_idx = int(li * per_cols + i // per_rows)
            sub_ax = ax[row_idx, col_idx]

            ylims[row_idx, col_idx] = plot_ex_roi_traces(
                sub_ax, 
                lp_df.loc[idx],
                col=col,
                dash=dash,
                zorder=raster_zorder - 1
            )

        time_values = np.asarray(lp_df.loc[lp_df.index[-1], "time_values"])
        
    plot_util.format_linpla_subaxes(ax, fluor="dff", 
        area=False, datatype="roi", sess_ns=None, xticks=None, kind="traces", 
        modif_share=False)

   # fix x ticks and lims
    for sub_ax in ax.reshape(-1):
        xlims = [time_values[0], time_values[-1]]
        xticks = np.linspace(*xlims, 6)
        sub_ax.set_xticks(xticks)
    plot_util.set_interm_ticks(ax, 3, axis="x", fontweight="bold", skip=False)
    for sub_ax in ax.reshape(-1):
        sub_ax.set_xlim(xlims)
    
    # reset y limits
    for r in range(ax.shape[0]):
        for c in range(ax.shape[1]):
            if not np.isfinite(ylims[r, c].sum()):
                continue
            ax[r, c].set_ylim(ylims[r, c])

    plot_util.set_interm_ticks(
        ax, 2, axis="y", share=False, weight="bold", update_ticks=True
        )  

    # rasterize the gray lines
    logger.info("Rasterizing individual traces...", extra={"spacing": TAB})
    for sub_ax in ax.reshape(-1):
        sub_ax.set_rasterization_zorder(raster_zorder)

    return ax

