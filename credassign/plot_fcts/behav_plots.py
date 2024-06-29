"""
behav_plots.py

This script contains functions for plotting pupil and running analyses.

Authors: Colleen Gillon

Date: February 2023

Note: this code was aggregated from https://github.com/colleenjg/OpenScope_CA_Analysis.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker, patches
import seaborn

from credassign.util import gen_util, logger_util, math_util, plot_util
from credassign.analysis import misc_analys
from credassign.plot_fcts import misc_plots, plot_helper_fcts, seq_plots


TAB = "    "


logger = logger_util.get_module_logger(name=__name__)


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

    exp_col = "#969696"
    unexp_col = "#eb3920"
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
        plot_util.expand_lims(sub_ax, axis="y", prop=0.21)

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
    gen_util.seed_all(seed)

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

    logger.info(f"{comp_info}:", extra={"spacing": "\n"})
    corr_str = "corr." if permpar["multcomp"] else "raw"

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
                p_val_text = f"{corr_str} p-val. {p_val_text}"
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


#############################################
def plot_pupil_run_full(sess_df, analyspar, figpar, title=None):
    """
    plot_pupil_run_full(sess_df, analyspar, figpar)

    Plots pupil and running data for a full session.

    Required args:
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

        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
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

    datatypes = ["run", "pupil"]
    datatype_strs = ["Running velocity (cm/s)", "Pupil diameter (mm)"]
    n_datatypes = len(datatypes)
    
    height_ratios = [0.12] + [0.83 / len(datatypes) for _ in datatypes] + [0.05]

    fig, ax = plt.subplots(
        n_datatypes + 2, 1, figsize=(14, 6), squeeze=False, 
        gridspec_kw={"height_ratios": height_ratios}, sharex=True
        )
    if title is not None:
        fig.suptitle(title, y=0.98, weight="bold")

    if len(sess_df) != 1:
        raise ValueError("Expected sess_df to have one row.")
    sess_row = sess_df.loc[sess_df.index[0]]
    duration_sec = sess_row["duration_sec"]

    for d, datatype in enumerate(datatypes):
        sub_ax = ax[d + 1, 0]
        data = sess_row[f"{datatype}_data"]
        x = np.linspace(0, duration_sec, len(data))
        sub_ax.plot(x, data, color="k", alpha=0.8, lw=1.5)
        sub_ax.set_ylabel(
            datatype_strs[d].replace(" ", "\n"), labelpad=12, weight="bold"
            )
        sub_ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
        for label in sub_ax.get_yticklabels():
            label.set_fontweight("bold")

    for sub_ax in ax.ravel():
        sub_ax.xaxis.set_visible(False)
        sub_ax.spines["bottom"].set_visible(False)

    for row in [0, -1]:
        ax[row, 0].yaxis.set_visible(False)
        ax[row, 0].spines["left"].set_visible(False)
        pad = duration_sec / 50
        ax[row, 0].set_xlim(-pad, duration_sec + pad)

    # add frames ranges
    data = sess_row[f"{datatype}_data"]
    fr_ranges = sess_row[f"{datatype}_frames"]
    x = np.linspace(0, duration_sec, len(data))
    ax[0, 0].set_ylim([-0.1, 0.5])
    for s, stim in enumerate(sess_row["stims"]):
        sec_ranges = [x[fr] for fr in fr_ranges[s]]
        ax[0, 0].plot(sec_ranges, [0, 0], lw=2.5, color="k")
        stim_str = stim.replace("right ", "").replace("left ", "")
        stim_str = stim_str.replace(" (", "\n").replace("))", ")")
        ax[0, 0].text(
            np.mean(sec_ranges), 0.45, stim_str, color="k", fontweight="bold", 
            fontsize="x-large", ha="center", va="center"
            )

    # add scale bar
    ax[-1, 0].plot([0, 60 * 5], [0, 0], lw=2.5, color="k")
    ax[-1, 0].set_ylim([-0.8, 0.1])
    ax[-1, 0].text(
        0, -1, "5 min", color="k", fontweight="bold", fontsize="x-large", 
        ha="left", va="center"
        )
    
    return ax


#############################################
def plot_pupil_run_histograms(hist_df, analyspar, figpar, title=None, 
                              log_scale=False):
    """
    plot_pupil_run_histograms(hist_df, analyspar, figpar)

    Plots pupil and running data histograms across sessions.

    Required args:
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

        - analyspar (dict): 
            dictionary with keys of AnalysPar namedtuple
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters

    Optional args:
        - title (str):
            plot title
            default: None
        - log_scale (bool):
            if True, a near logarithmic scale is used for the y axis, but for 
            the running data only (with a linear range to reach 0, and break 
            marks to mark the transition from linear to log range)
            default: False

    Returns:
        - ax (2D array): 
            array of subplots
    """

    datatypes = ["run", "pupil"]
    datatype_strs = ["Running velocity (cm/s)", "Pupil diameter (mm)"]
    n_datatypes = len(datatypes)
    
    sess_ns = sorted(hist_df["sess_ns"].unique())
    mouse_ns = sorted(np.unique(np.concatenate(hist_df["mouse_ns"].tolist())))
    colors = plot_util.get_cmap_colors("Greys", len(mouse_ns) + 1)[1:]
    
    n_sess = len(sess_ns)
    fig, ax = plt.subplots(
        n_datatypes, n_sess, figsize=(5.5 * n_sess, 3.8 * n_datatypes), 
        squeeze=False, sharex="row", sharey="row"
        )

    if title is not None:
        fig.suptitle(title, y=1, weight="bold")

    log_base = 2
    inset_data = []
    for s, sess_n in enumerate(sess_ns):
        sess_lines = hist_df.loc[hist_df["sess_ns"] == sess_n]
        if len(sess_lines) != 1:
            raise RuntimeError(
                "Expected exactly one line per session number."
                )
        sess_line = sess_lines.loc[sess_lines.index[0]]
        mouse_colors = [
            colors[mouse_ns.index(mouse_n)] for mouse_n in sess_line["mouse_ns"]
            ]
        n_mice = len(mouse_colors)

        for d, datatype in enumerate(datatypes):
            sub_ax = ax[d, s]
            if s == 0:
                sub_ax.set_ylabel("Density", fontweight="bold")
            if s == n_sess // 2:
                sub_ax.set_xlabel(datatype_strs[d], fontweight="bold")

            binned_data = np.asarray(sess_line[f"{datatype}_vals_binned"])
            bin_edges = sess_line[f"{datatype}_bin_edges"]
            bins = np.linspace(*bin_edges, len(binned_data[0]) + 1)
            x = np.repeat(bins[:-1], n_mice).reshape(-1, n_mice)
            density_vals = sub_ax.hist(
                x, bins, density=True, histtype="bar", stacked=True, 
                color=mouse_colors, weights=binned_data.T,
                )[0]
                     
            if not log_scale and d == 0: # make inset
                threshold = density_vals[-1].max() / 4
                first = np.where(density_vals[-1] > threshold)[0][-1] + 1
                last_vals = np.where(
                    np.cumsum(density_vals[-1] == 0) == 0
                    )[0]
                last = density_vals.shape[1]
                if len(last_vals):
                    last = last_vals[-1] + 1
                
                if first < density_vals.shape[1]:
                    sub_ax_in = sub_ax.inset_axes([0.5, 0.5, 0.45, 0.45])
                    half_bin = np.diff(bins)[-1] / 3
                    inset_bin_edges = bins[first] - half_bin, bins[last]
                    density_vals = np.diff(
                        np.insert(density_vals, 0, 0, axis=0), axis=0
                        )
                    sub_ax_in.hist(
                        x[first : last], bins[first : last + 1], 
                        density=False, histtype="bar", stacked=True, 
                        color=mouse_colors, 
                        weights=density_vals.T[first : last],
                        )
                    inset_data.append((sub_ax_in, inset_bin_edges))

            if s == 0 and d == 0 and log_scale:
                sub_ax.set_ylabel("Density (log. scale)", fontweight="bold")
                sub_ax.set_yscale("log", base=log_base)
        
        ax[0, s].set_title(f"Session {sess_n}", fontweight="bold", y=1.1)

    if log_scale: # update y ticks, if applicable
        misc_plots.set_symlog_scale(
            ax[0:1], log_base=log_base, col_per_grp=n_sess, n_ticks=4
            )

    for d in range(len(datatypes)):
        sub_ax = ax[d, 0]
        for label in sub_ax.get_yticklabels():
            label.set_fontweight("bold")
        plot_util.set_interm_ticks(
            ax[d : d + 1, :], 3, axis="x", share=True, update_ticks=True, 
            fontweight="bold"
        )
    
    if len(inset_data): # format inset plots
        y_lim_bot = np.inf
        x_lim_top, y_lim_top = -np.inf, -np.inf
        for sub_ax_in, bin_edges in inset_data:
            x_lim_top = np.max([x_lim_top, bin_edges[-1]])
            y_lim_bot = np.min([y_lim_bot, sub_ax_in.get_ylim()[0]])
            y_lim_top = np.max([y_lim_top, sub_ax_in.get_ylim()[1] * 1.1])

        for s, (sub_ax_in, bin_edges) in enumerate(inset_data):
            sub_ax_in.set_xlim(bin_edges[0], x_lim_top)
            sub_ax_in.set_ylim(y_lim_bot, y_lim_top)
            if s != 0:
                sub_ax_in.set_yticklabels("")
            for spine in ["top", "right"]:
                sub_ax_in.spines[spine].set_visible(True)
            labels = sub_ax_in.get_xticklabels() + sub_ax_in.get_yticklabels()
            for label in labels:
                label.set_fontweight("bold")

            # add dashed rectangle
            x, y = bin_edges[0], y_lim_bot
            x_wid = x_lim_top - x
            y_wid = y_lim_top - y_lim_bot
            rect = patches.Rectangle(
                (x, y), x_wid, y_wid, lw=3, edgecolor="k", facecolor="none", 
                ls=(3, (3, 3))
                )
            ax[0, s].add_patch(rect)


    return ax


