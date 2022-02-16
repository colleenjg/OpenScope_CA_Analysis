"""
misc_plots.py

This script contains miscellanous plotting functions.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import numpy as np

from util import gen_util, plot_util
from sess_util import sess_plot_util
from plot_fcts import plot_helper_fcts, seq_plots


TAB = "    "


#############################################
def plot_decoder_data(scores_df, analyspar, sesspar, permpar, figpar, 
                      title=None):
    """
    plot_decoder_data(scores_df, analyspar, sesspar, permpar, figpar)

    Plots Gabor decoding scores across sessions. 
    
    Required args:
        - scores_dfs (pd.DataFrame):
            dataframe with logistic regression score statistics, shuffled score 
            confidence intervals, and test set p-values for each 
            line/plane/session, in addition to the basic sess_df columns
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  

    Returns:
        - ax (2D array): 
            array of subplots
    """

    score_col = "test_balanced_accuracy"

    # add expected columns, and convert to percentage
    scores_df = scores_df.copy(deep=True)
    stat = scores_df[f"{score_col}_stat"].to_numpy().reshape(-1, 1) * 100
    error =  np.asarray(scores_df[f"{score_col}_err"].tolist()) * 100
    error = error.reshape(len(error), -1)
    scores_df[score_col] = np.concatenate([stat, error], axis=1).tolist()
    
    null_CI_perc = np.asarray(scores_df[f"{score_col}_null_CIs"].tolist()) * 100
    scores_df["null_CIs"] = null_CI_perc.tolist()
    scores_df["p_vals"] = scores_df[f"{score_col}_p_vals"]

    ax = seq_plots.plot_sess_data(
        scores_df, 
        analyspar=analyspar, 
        sesspar=sesspar, 
        permpar=permpar, 
        figpar=figpar, 
        title=title, 
        wide=True,
        between_sess_sig=False,
        data_col=score_col,
        decoder_data=True,
        )

    return ax


#############################################
def plot_snr_sigmeans_nrois(data_df, figpar, datatype="snrs", title="ROI SNRs"):
    """
    plot_snr_sigmeans_nrois(data_df, figpar)

    Plots SNR, signal means or number of ROIs, depending on the case.

    Required args:
        - data_df (pd.DataFrame):
            dataframe with SNR, signal mean or number of ROIs data for each 
            session, in addition to the basic sess_df columns
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters

    Optional args:
        - datatype (str):
            type of data to plot, also corresponding to column name
            default: "snrs"
        - title (str):
            plot title
            default: "ROI SNRs"

    Returns:
        - ax (2D array): 
            array of subplots
    """

    sess_ns = np.arange(data_df.sess_ns.min(), data_df.sess_ns.max() + 1)

    figpar = sess_plot_util.fig_init_linpla(figpar, kind="reg")
    figpar["init"]["sharey"] = "row"
    
    figpar["init"]["subplot_hei"] = 4.4
    figpar["init"]["gs"] = {"wspace": 0.2, "hspace": 0.2}
    if datatype != "nrois":
        figpar["init"]["subplot_wid"] = 3.2        
    else:
        figpar["init"]["subplot_wid"] = 2.5
        
    fig, ax = plot_util.init_fig(4, **figpar["init"])
    if title is not None:
        fig.suptitle(title, y=0.97, weight="bold")

    for (line, plane), lp_df in data_df.groupby(["lines", "planes"]):
        li, pl, col, dash = plot_helper_fcts.get_line_plane_idxs(line, plane)
            
        sub_ax = ax[pl, li]

        if datatype == "snrs":
            sub_ax.axhline(y=1, ls=plot_helper_fcts.HDASH, c="k", lw=3.0, 
            alpha=0.5)
        elif datatype == "signal_means":
            sub_ax.axhline(y=0, ls=plot_helper_fcts.HDASH, c="k", lw=3.0, 
            alpha=0.5)
        elif datatype != "nrois":
            gen_util.accepted_values_error(
                "datatype", datatype, ["snrs", "signal_means", "nrois"]
                )

        if datatype == "nrois":
            plot_nrois(sub_ax, lp_df, sess_ns=sess_ns, col=col, dash=dash)
            continue

        data = []
        use_sess_ns = []
        for sess_n in sess_ns:
            rows = lp_df.loc[lp_df["sess_ns"] == sess_n]
            if len(rows) > 0:
                use_sess_ns.append(sess_n)
                data.append(np.concatenate(rows[datatype].tolist()))

        sub_ax.boxplot(
            data, positions=use_sess_ns, notch=True, patch_artist=True, 
            whis=[5, 95], widths=0.6,
            boxprops=dict(facecolor="white", color=col, linewidth=3.0), 
            capprops=dict(color=col, linewidth=3.0),
            whiskerprops=dict(color=col, linewidth=3.0),
            flierprops=dict(color=col, markeredgecolor=col, markersize=8),
            medianprops=dict(color=col, linewidth=3.0)
            )
    
    sess_plot_util.format_linpla_subaxes(ax, datatype="roi", 
        ylab="", xticks=sess_ns, kind="reg", single_lab=True)       
        
    return ax


#############################################
def plot_nrois(sub_ax, sess_df, sess_ns=None, col="k", dash=None): 
    """
    plot_nrois(sub_ax, sess_df)

    Plots number of ROIs per mice, across sessions.

    Required args:
        - sub_ax (plt subplot): 
            subplot
        - sess_df (pd.DataFrame)
            dataframe with the basic sess_df columns
    
    Optional args:
        - sess_ns (array-like): 
            session numbers to use as x values. Inferred if None.
            default: None
        - col (str):
            plotting color
            default: "k"
        - dash (str or tuple):
            dash style
            default: None
    """
    
    if sess_ns is None:
        sess_ns = np.arange(sess_df.sess_ns.min(), sess_df.sess_ns.max() + 1)
    unique_mice = sorted(sess_df["mouse_ns"].unique())

    mouse_cols = plot_util.get_hex_color_range(
        len(unique_mice), col=col, interval=plot_helper_fcts.MOUSE_COL_INTERVAL
        )
    for mouse, mouse_col in zip(unique_mice, mouse_cols):
        nrois = []
        for sess_n in sess_ns:
            row = sess_df.loc[
                (sess_df["mouse_ns"] == mouse) & 
                (sess_df["sess_ns"] == sess_n)]
            if len(row):
                if len(row) > 1:
                    raise RuntimeError("No more than one match expected.")
                else:
                    nrois.append(row["nrois"].values[0])
            else:
                nrois.append(np.nan)

        plot_util.plot_errorbars(
            sub_ax, nrois, x=sess_ns, color=mouse_col, alpha=0.6, ls=dash, lw=5, 
            mew=5, markersize=8, xticks="auto"
            ) 


#############################################
def set_symlog_scale(ax, log_base=2, col_per_grp=3, n_ticks=4):
    """
    set_symlog_scale(ax)

    Converts y axis to symmetrical log scale (log scale with a linear range 
    to 0), and updates ticks.

    Required args:
        - ax (2D array): 
            array of subplots

    Optional args:
        - log_base (int):
            log base for the log scale
            default: 2
        - col_per_grp (int):
            number of columns in subplot axis groups that share y axis markers
            default: 3
        - n_ticks (int):
            number of log ticks
            default: 4
    """

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            sub_ax = ax[i, j]
            if i == 0 and j == 0:
                # adjust to avoid bins going to negative infinity
                base_lin = sub_ax.get_ylim()[0]
                sub_ax.set_yscale(
                    "symlog", base=log_base, linthresh=base_lin, linscale=0.5
                    )
                yticks = sub_ax.get_yticks()
                n_ticks = 4
                n = len(yticks) // n_ticks
                yticks = [
                    ytick for y, ytick in enumerate(yticks) if not(y % n)
                    ]
                sub_ax.set_yticks(yticks)

            if not (j % col_per_grp):
                # use minor ticks to create a break in the axis between the 
                # linear and log ranges
                yticks = sub_ax.get_yticks()
                low = yticks[0] + (base_lin - yticks[0]) * 0.55
                high = base_lin
                sub_ax.set_yticks([low, high], minor=True)
                sub_ax.set_yticklabels(["", ""], minor=True)
                sub_ax.tick_params(
                    axis="y", direction="inout", which="minor", length=16, 
                    width=3
                    )
                
            if i == 1:
                xticks = sub_ax.get_xticks()
                xticks = [int(t) if int(t) == t else t for t in xticks]
                sub_ax.set_xticks(xticks)
                sub_ax.set_xticklabels(xticks, fontweight="bold")


#############################################
def plot_roi_correlations(corr_df, figpar, title=None, log_scale=True):
    """
    plot_roi_correlations(corr_df, figpar)

    Plots correlation histograms.

    Required args:
        - corr_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the 
            following columns, in addition to the basic sess_df columns:
            - bin_edges (list): first and last bin edge
            - corrs_binned (list): number of correlation values per bin
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
            if True, a near logarithmic scale is used for the y axis (with a 
            linear range to reach 0, and break marks to mark the transition 
            from linear to log range)
            default: True

    Returns:
        - ax (2D array): 
            array of subplots
    """

    sess_ns = np.arange(corr_df.sess_ns.min(), corr_df.sess_ns.max() + 1)
    n_sess = len(sess_ns)

    figpar = sess_plot_util.fig_init_linpla(
        figpar, kind="prog", n_sub=len(sess_ns)
        )
    figpar["init"]["subplot_hei"] = 3.0
    figpar["init"]["subplot_wid"] = 2.8
    figpar["init"]["sharex"] = log_scale
    if log_scale:
        figpar["init"]["sharey"] = True

    fig, ax = plot_util.init_fig(4 * len(sess_ns), **figpar["init"])
    if title is not None:
        fig.suptitle(title, y=1.02, weight="bold")

    sess_plot_util.format_linpla_subaxes(ax, datatype="roi", 
        ylab="Density", xlab="Correlation", sess_ns=sess_ns, kind="prog", 
        single_lab=True)

    log_base = 2
    for (line, plane), lp_df in corr_df.groupby(["lines", "planes"]):
        li, pl, col, _ = plot_helper_fcts.get_line_plane_idxs(line, plane)
        for s, sess_n in enumerate(sess_ns):
            sess_rows = lp_df.loc[lp_df["sess_ns"] == sess_n]
            if len(sess_rows) == 0:
                continue
            elif len(sess_rows) > 1:
                raise RuntimeError("Expected exactly one row.")
            sess_row = sess_rows.loc[sess_rows.index[0]]

            sub_ax = ax[pl, s + li * n_sess]
 
            weights = np.asarray(sess_row["corrs_binned"])

            bin_edges = np.linspace(*sess_row["bin_edges"], len(weights) + 1)

            sub_ax.hist(
                bin_edges[:-1], bin_edges, weights=weights, color=col, 
                alpha=0.6, density=True
                )
            sub_ax.axvline(
                0, ls=plot_helper_fcts.VDASH,  c="k", lw=3.0, alpha=0.5
                )
            
            sub_ax.spines["bottom"].set_visible(True)
            sub_ax.tick_params(axis="x", which="both", bottom=True, top=False)
            
            if log_scale:
                sub_ax.set_yscale("log", base=log_base)
                sub_ax.set_xlim(-1, 1)
            else:
                sub_ax.autoscale(axis="x", tight=True)

            sub_ax.autoscale(axis="y", tight=True)

    if log_scale: # update x ticks
        set_symlog_scale(ax, log_base=log_base, col_per_grp=n_sess, n_ticks=4)
                    
    else: # update x and y ticks
        for i in range(ax.shape[0]):
            for j in range(int(ax.shape[1] / n_sess)):
                sub_axes = ax[i, j * n_sess : (j + 1) * n_sess]
                plot_util.set_interm_ticks(
                    sub_axes, 4, axis="y", share=True, update_ticks=True
                )

    plot_util.set_interm_ticks(
        ax, 4, axis="x", share=log_scale, update_ticks=True, fontweight="bold"
    )

    return ax

