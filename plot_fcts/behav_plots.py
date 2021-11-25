"""
behav_plots.py

This script contains functions for plotting pupil and running analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

import numpy as np
from matplotlib import pyplot as plt
import seaborn

from util import logger_util, plot_util, math_util, rand_util
from analysis import misc_analys
from plot_fcts import plot_helper_fcts, seq_plots

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def plot_pupil_run_trace_stats(trace_df, analyspar, figpar, split="by_exp", 
                               title=None):
    """
    plot_pupil_run_trace_stats(trace_df, analyspar, figpar)

    Plots pupil and running trace statistics.

    Required args:
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

        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters

    Optional args:
        - split (str):
            data split, e.g. "exp_lock", "unexp_lock", "stim_onset" or 
            "stim_offset"
            default: False
        - title (str):
            plot title
            default: None

    Returns:
        - ax (2D array): 
            array of subplots
    """

    if split != "by_exp":
        raise NotImplementedError("Only implemented split 'by_exp'.")

    if analyspar["scale"]:
        raise NotImplementedError(
            "Expected running and pupil data to not be scaled."
            )

    datatypes = ["run", "pupil"]

    figpar["init"]["subplot_wid"] = 4.2
    figpar["init"]["subplot_hei"] = 2.2
    figpar["init"]["gs"] = {"hspace": 0.3}
    figpar["init"]["ncols"] = 1
    figpar["init"]["sharey"] = False
    figpar["init"]["sharex"] = True

    fig, ax = plot_util.init_fig(len(datatypes), **figpar["init"])

    if title is not None:
        fig.suptitle(title, weight="bold", y=1.0)

    if len(trace_df) != 1:
        raise NotImplementedError(
            "Only implemented for a trace_df with one row."
            )
    row_idx = trace_df.index[0]

    exp_col = plot_util.LINCLAB_COLS["gray"]
    unexp_col = plot_util.LINCLAB_COLS["red"]
    for d, datatype in enumerate(datatypes):
        sub_ax = ax[d, 0]

        time_values = trace_df.loc[row_idx, f"{datatype}_time_values"]
        trace_stats = trace_df.loc[row_idx, f"{datatype}_trace_stats"]

        seq_plots.plot_traces(
            sub_ax, time_values, trace_stats, split=split, col=unexp_col, 
            lab=False, exp_col=exp_col, hline=False
            )
    
        if datatype == "run":
            ylabel = "Running\nvelocity\n(cm/s)"
        elif datatype == "pupil":
            ylabel = "Pupil\ndiameter\n(mm)"
        sub_ax.set_ylabel(ylabel, weight="bold")

   # fix x ticks and lims
    plot_util.set_interm_ticks(ax, 3, axis="x", fontweight="bold")
    xlims = [np.min(time_values), np.max(time_values)]
    if split != "by_exp":
        xlims = [-xlims[1], xlims[1]]
    sub_ax.set_xlim(xlims)
    sub_ax.set_xlabel("Time (s)", weight="bold")

    # expand y lims a bit and fix y ticks
    for sub_ax in ax.reshape(-1):
        plot_util.expand_lims(sub_ax, axis="y", prop=0.2)

    plot_util.set_interm_ticks(
        ax, 2, axis="y", share=False, weight="bold", update_ticks=True
        )

    return ax


#############################################
def plot_violin_data(sub_ax, xs, all_data, palette=None, dashes=None, 
                     seed=None):
    """
    plot_violin_data(sub_ax, xs, all_data)

    Plots violin data for each data group.

    Required args:
        - sub_ax (plt subplot):
            subplot
        - xs (list):
            x value for each data group
        - all_data (list):
            data for each data group


    Optional args:
        - palette (list)
            colors for each data group
            default: None
        - dashes (list): 
            dash patterns for each data group
            default: None
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None
    """

    # seed for scatterplot
    rand_util.seed_all(seed, log_seed=False)

    # checks
    if len(xs) != len(all_data):
        raise ValueError("xs must have the same length as all_data.")
    
    if palette is not None and len(xs) != len(palette):
        raise ValueError(
            "palette, if provided, must have the same length as xs."
            )

    if dashes is not None and len(xs) != len(dashes):
        raise ValueError(
            "dashes, if provided, must have the same length as xs."
            )

    xs_arr = np.concatenate([
        np.full_like(data, x) for x, data in zip(xs, all_data)
    ])
    data_arr = np.concatenate(all_data)

    # add violins
    bplot = seaborn.violinplot(
        x=xs_arr, y=data_arr, inner=None, linewidth=3.5, color="white", 
        ax=sub_ax
    )

    # edit contours
    for c, collec in enumerate(bplot.collections):
        collec.set_edgecolor(plot_helper_fcts.NEARBLACK)

        if dashes is not None and dashes[c] is not None:
            collec.set_linestyle(plot_helper_fcts.VDASH)

    # add data dots
    seaborn.stripplot(
        x=xs_arr, y=data_arr, size=9, jitter=0.2, alpha=0.3, palette=palette, 
        ax=sub_ax
        )
    

#############################################
def plot_pupil_run_block_diffs(block_df, analyspar, permpar, figpar, 
                               title=None, seed=None):
    """
    plot_pupil_run_trace_stats(trace_df, analyspar, permpar, figpar)

    Plots pupil and running block differences.

    Required args:
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
        - title (str):
            plot title
            default: None
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None

    Returns:
        - ax (2D array): 
            array of subplots
    """

    if analyspar["scale"]:
        raise NotImplementedError(
            "Expected running and pupil data to not be scaled."
            )

    if len(block_df["sess_ns"].unique()) != 1:
        raise NotImplementedError(
            "'block_df' should only contain one session number."
        )

    nanpol = None if analyspar["rem_bad"] else "omit"
    
    sensitivity = misc_analys.get_sensitivity(permpar)
    comp_info = misc_analys.get_comp_info(permpar)

    datatypes = ["run", "pupil"]
    datatype_strs = ["Running velocity", "Pupil diameter"]
    n_datatypes = len(datatypes)

    fig, ax = plt.subplots(
        1, n_datatypes, figsize=(12.7, 4), squeeze=False, 
        gridspec_kw={"wspace": 0.22}
        )

    if title is not None:
        fig.suptitle(title, y=1.2, weight="bold")

    logger.info(f"Corrected p-values ({comp_info}):", extra={"spacing": "\n"})
    
    for d, datatype in enumerate(datatypes):
        datatype_sig_str = f"{datatype_strs[d]:16}:"
        sub_ax = ax[0, d]

        lp_names = [None for _ in range(plot_helper_fcts.N_LINPLA)]
        xs, all_data, cols, dashes, p_val_texts = [], [], [], [], []
        for (line, plane), lp_df in block_df.groupby(["lines", "planes"]):
            x, col, dash = plot_helper_fcts.get_line_plane_idxs(
                line, plane, flat=True
                )
            line_plane_name = plot_helper_fcts.get_line_plane_name(
                line, plane
                )
            lp_names[int(x)] = line_plane_name

            if len(lp_df) == 1:
                row_idx = lp_df.index[0]
            elif len(lp_df) > 1:
                raise RuntimeError("Expected 1 row per line/plane/session.")
        
            lp_data = lp_df.loc[row_idx, f"{datatype}_block_diffs"]
            
            # get p-value information
            p_val_corr = lp_df.loc[row_idx, f"{datatype}_p_vals"]

            side = np.sign(
                math_util.mean_med(
                    lp_data, stats=analyspar["stats"], nanpol=nanpol
                    )
                )
            sig_str = misc_analys.get_sig_symbol(
                p_val_corr, sensitivity=sensitivity, side=side, 
                tails=permpar["tails"], p_thresh=permpar["p_val"]
                )

            p_val_text = f"{p_val_corr:.2f}{sig_str}"

            datatype_sig_str = (
                f"{datatype_sig_str}{TAB}{line_plane_name}: "
                f"{p_val_corr:.5f}{sig_str:3}"
                )

            # collect information
            xs.append(x)
            all_data.append(lp_data)
            cols.append(col)
            dashes.append(dash)
            p_val_texts.append(p_val_text)

        plot_violin_data(
            sub_ax, xs, all_data, palette=cols, dashes=dashes, seed=seed
            )
        
        # edit ticks
        sub_ax.set_xticks(range(plot_helper_fcts.N_LINPLA))
        sub_ax.set_xticklabels(lp_names, fontweight="bold")
        sub_ax.tick_params(axis="x", which="both", bottom=False) 

        plot_util.expand_lims(sub_ax, axis="y", prop=0.1)

        plot_util.set_interm_ticks(
            np.asarray(sub_ax), n_ticks=3, axis="y", share=False, 
            fontweight="bold", update_ticks=True
            )

        for i, (x, p_val_text) in enumerate(zip(xs, p_val_texts)):
            ylim_range = np.diff(sub_ax.get_ylim())
            y = sub_ax.get_ylim()[1] + ylim_range * 0.08
            ha = "center"
            if d == 0 and i == 0:
                x += 0.2
                p_val_text = f"raw p-val. {p_val_text}"
                ha = "right"
            sub_ax.text(
                x, y, p_val_text, fontsize=20, weight="bold", ha=ha
                )

        logger.info(datatype_sig_str, extra={"spacing": TAB})

    # add labels/titles
    for d, datatype in enumerate(datatypes):
        sub_ax = ax[0, d]
        sub_ax.axhline(
            y=0, ls=plot_helper_fcts.HDASH, c="k", lw=3.0, alpha=0.5
            )
     
        if d == 0:
            ylabel = "Trial differences\nU-G - D-G"
            sub_ax.set_ylabel(ylabel, weight="bold")    
        
        if datatype == "run":
            title = "Running velocity (cm/s)"
        elif datatype == "pupil":
            title = "Pupil diameter (mm)"

        sub_ax.set_title(title, weight="bold", y=1.2)
        
    return ax

