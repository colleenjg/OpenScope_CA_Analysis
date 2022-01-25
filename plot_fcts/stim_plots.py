"""
stim_plots.py

This script contains functions for plotting stimulus comparison analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

import numpy as np

from util import logger_util, plot_util
from sess_util import sess_plot_util
from analysis import misc_analys
from plot_fcts import plot_helper_fcts

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def add_between_stim_sig(ax, sub_ax_all, data_df, permpar):
    """
    add_between_stim_sig(ax, sub_ax_all, data_df, permpar)

    Plot significance markers for significant comparisons between stimulus 
    types.

    Required args:
        - ax (plt Axis): 
            axis
        - sub_ax_all (plt subplot): 
            all line/plane data subplot
        - data_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - {data_col} (list): data stats (me, err)
            for session comparisons, e.g. 1v2:
            - p_vals_{}v{} (float): p-value for differences between sessions, 
                corrected for multiple comparisons and tails

        - permpar (dict): 
            dictionary with keys of PermPar namedtuple
    """

    sensitivity = misc_analys.get_sensitivity(permpar)
    comp_info = misc_analys.get_comp_info(permpar)

    logger.info(f"{comp_info}:", extra={"spacing": "\n"})

    stimtypes = ["gabors", "visflow"]
    stim_sig_str = f"Gabors vs visual flow: "
    for (line, plane), lp_df in data_df.groupby(["lines", "planes"]):

        if len(lp_df) != 1:
            raise RuntimeError("Expected 1 row per line/plane/session.") 
        row_idx = lp_df.index[0]   

        x = [0, 1]
        data = np.vstack(
            [lp_df[stimtypes[0]].tolist(), lp_df[stimtypes[1]].tolist()]
            ).T
        y = data[0]
        err = data[1:]
        highest = np.max(y + err[-1])

        if line != "all" and plane != "all":
            li, pl, col, dash = plot_helper_fcts.get_line_plane_idxs(
                line, plane
                )
            linpla_name = plot_helper_fcts.get_line_plane_name(line, plane)
            sub_ax = ax[pl, li]
            mark_rel_y = 0.18
        else:
            col = plot_helper_fcts.NEARBLACK
            linpla_name = "All"
            sub_ax = sub_ax_all
            all_data_max = np.concatenate(
                [data_df[stimtypes[0]].tolist(), 
                data_df[stimtypes[1]].tolist()], 
                axis=0
                )[:, 0].max()
            highest = np.max([data[0].max(), all_data_max])
            mark_rel_y = 0.15

        p_val = lp_df.loc[row_idx, "p_vals"]
        side = np.sign(y[1] - y[0])

        sig_str = misc_analys.get_sig_symbol(
            p_val, sensitivity=sensitivity, side=side, tails=permpar["tails"], 
            p_thresh=permpar["p_val"]
            )
        stim_sig_str = \
            f"{stim_sig_str}{TAB}{linpla_name}: {p_val:.5f}{sig_str:3}"
        
        if len(sig_str):
            plot_util.plot_barplot_signif(
                sub_ax, x, highest, rel_y=0.11, color=col, lw=3, 
                mark_rel_y=mark_rel_y, mark=sig_str, 
            )
    
    logger.info(stim_sig_str, extra={"spacing": TAB})

    
#############################################
def plot_stim_data_df(stim_data_df, stimpar, permpar, figpar, pop_stats=True, 
                      title=None):
    """
    plot_stim_data_df(stim_data_df, stimpar, permpar, figpar)

    Plots stimulus comparison data.

    Required args:
        - stim_stats_df (pd.DataFrame):
            dataframe with one row per line/plane and one for all line/planes 
            together, and the basic sess_df columns, in addition to, 
            for each stimtype:
            - stimtype (list): absolute fractional change statistics (me, err)
            - raw_p_vals (float): uncorrected p-value for data differences 
                between stimulus types 
            - p_vals (float): p-value for data differences between stimulus 
                types, corrected for multiple comparisons and tails
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
        - pop_stats (bool):
            if True, analyses are run on population statistics, and not 
            individual tracked ROIs
            default: True
        - title (str):
            plot title
            default: None
    
    Returns:
        - ax (2D array): 
            array of subplots 
            (does not include added subplot for all line/plane data together)
    """

    figpar = sess_plot_util.fig_init_linpla(figpar, kind="reg")

    figpar["init"]["subplot_wid"] = 2.1
    figpar["init"]["subplot_hei"] = 4.2
    figpar["init"]["gs"] = {"hspace": 0.20, "wspace": 0.3}
    figpar["init"]["sharey"] = "row"
    
    fig, ax = plot_util.init_fig(plot_helper_fcts.N_LINPLA, **figpar["init"])
    fig.suptitle(title, y=0.98, weight="bold")

    sub_ax_all = fig.add_axes([1.05, 0.11, 0.3, 0.77])

    stimtypes = stimpar["stimtype"][:] # deep copy

    # indicate bootstrapped error with wider capsize
    capsize = 8 if pop_stats else 6

    lp_data = []
    cols = []
    for (line, plane), lp_df in stim_data_df.groupby(["lines", "planes"]):
        x = [0, 1]
        data = np.vstack(
            [lp_df[stimtypes[0]].tolist(), lp_df[stimtypes[1]].tolist()]
            ).T
        y = data[0]
        err = data[1:]

        if line != "all" and plane != "all":
            li, pl, col, dash = plot_helper_fcts.get_line_plane_idxs(
                line, plane
                )
            alpha = 0.5
            sub_ax = ax[pl, li]
            lp_data.append(y)
            cols.append(col)
        else:
            col = plot_helper_fcts.NEARBLACK
            dash = None
            alpha = 0.2
            sub_ax = sub_ax_all
            sub_ax.set_title("all", fontweight="bold")
    
        plot_util.plot_bars(
            sub_ax, x, y=y, err=err, width=0.5, lw=None, alpha=alpha, 
            color=col, ls=dash, capsize=capsize
            )

    # add dots to the all subplot
    x_vals = np.asarray([-0.17, 0.25, -0.25, 0.17]) # to spread dots out
    lw = 4
    ms = 200
    for s, _ in enumerate(stimtypes):
        lp_stim_data = [data[s] for data in lp_data]
        sorter = np.argsort(lp_stim_data)
        for i, idx in enumerate(sorter):
            x_val = s + x_vals[i]
            # white behind
            sub_ax_all.scatter(
                x=x_val, y=lp_stim_data[idx], s=ms, linewidth=lw, alpha=0.8, 
                color="white", zorder=10
                )
            
            # colored dots
            sub_ax_all.scatter(
                x=x_val, y=lp_stim_data[idx], s=ms, alpha=0.6, linewidth=0, 
                color=cols[idx], zorder=11
                )
            
            # dot borders
            sub_ax_all.scatter(
                x=x_val, y=lp_stim_data[idx], s=ms, color="None", 
                edgecolor=cols[idx], linewidth=lw, alpha=1, zorder=12
                )

    # add between stim significance 
    add_between_stim_sig(ax, sub_ax_all, stim_data_df, permpar)

    # add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(ax, datatype="roi", lines=None, 
        planes=["", ""], xticks=[0, 1], ylab="Absolute fractional change", 
        kind="reg", xlab=""
        )
    
    # adjust plot details
    stimtype_names = stimtypes[:]
    stimtype_names[stimtypes.index("visflow")] = "visual\nflow"
    for sub_ax in fig.axes:
        y_max = sub_ax.get_ylim()[1]
        sub_ax.set_ylim([0, y_max])
        sub_ax.set_xticks([0, 1])
        sub_ax.set_xticklabels(
            stimtypes, weight="bold", rotation=45, ha="right"
            )
        sub_ax.tick_params(axis="x", bottom=False)
    sub_ax_all.set_xlim(ax[0, 0].get_xlim())
        
    plot_util.set_interm_ticks(
        np.asarray(sub_ax_all), 4, axis="y", share=False, weight="bold"
        )

    return ax

