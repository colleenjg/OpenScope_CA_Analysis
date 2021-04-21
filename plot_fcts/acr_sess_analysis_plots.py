"""
acr_sess_analysis_plots.py

This script contains functions to plot results of across sessions analyses 
(acr_sess_analys.py) from dictionaries.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import copy
import logging
import os
import warnings

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.stats as st

from util import file_util, gen_util, logger_util, math_util, plot_util
from sess_util import sess_gen_util, sess_plot_util, sess_str_util

logger = logging.getLogger(__name__)

# skip tight layout warning
warnings.filterwarnings("ignore", message="This figure includes*")

VDASH = (0, (3, 2))
HDASH = (0, (4, 2))
DARKRED = "#871719"


#############################################
def plot_from_dict(dict_path, plt_bkend=None, fontdir=None, parallel=False, 
                   datetime=True):
    """
    plot_from_dict(dict_path)

    Plots data from dictionaries containing analysis parameters and results.

    Required args:
        - dict_path (str): path to dictionary to plot data from
    
    Optional_args:
        - plt_bkend (str): mpl backend to use for plotting (e.g., "agg")
                           default: None
        - fontdir (str)  : path to directory where additional fonts are stored
                           default: None
        - parallel (bool): if True, some of the plotting is parallelized across 
                           CPU cores
                           default: False
        - datetime (bool): figpar["save"] datatime parameter (whether to 
                           place figures in a datetime folder)
                           default: True
    """

    logger.info(f"Plotting from dictionary: {dict_path}", 
        extra={"spacing": "\n"})
    
    figpar = sess_plot_util.init_figpar(
        plt_bkend=plt_bkend, fontdir=fontdir, datetime=datetime)
    plot_util.manage_mpl(cmap=False, **figpar["mng"])

    sess_plot_util.update_plt_linpla()

    info = file_util.loadfile(dict_path)
    savedir = os.path.dirname(dict_path)

    analysis = info["extrapar"]["analysis"]

    # 0. Plots the difference between surprise and regular across sessions
    if analysis == "s": 
        plot_surp_area_diff(figpar=figpar, savedir=savedir, **info)

    # 1. Plots the difference between surprise and regular locked to surprise
    # across sessions
    elif analysis == "l":
        plot_lock_area_diff(figpar=figpar, savedir=savedir, **info)

    # 2. Plots the difference between stimulus and grayscreen
    elif analysis == "a":
        plot_stim_grayscr_diff(figpar=figpar, savedir=savedir, **info)

    # 3. Plots the surprise and regular traces across sessions
    elif analysis == "t":
        plot_surp_traces(figpar=figpar, savedir=savedir, **info)

    # 4. Plots the surprise and regular locked to surprise traces
    # across sessions
    elif analysis == "r":
        plot_lock_traces(figpar=figpar, savedir=savedir, **info)

    # 5. Plots the stimulus and grayscreen onset/offset traces
    elif analysis == "b":
        plot_stim_grayscr_traces(figpar=figpar, savedir=savedir, **info)

    # 6. Plots the progression surprises/regular sequences across sessions
    elif analysis == "g":
        plot_prog(figpar=figpar, savedir=savedir, **info)

    # 7. Plots the difference in position surprises/regular sequences across 
    # sessions
    elif analysis == "o":
        plot_position(figpar=figpar, savedir=savedir, **info)

    # 8. Plots the surprise indices across sessions
    elif analysis == "i":
        plot_surp_idx(figpar=figpar, savedir=savedir, **info)

    # 9. Plots the surprise indices with matching orientations across sessions
    elif analysis == "m":
        plot_surp_idx_match_oris(figpar=figpar, savedir=savedir, **info)

    # 10. Plots the direction indices across sessions
    elif analysis == "d":
        plot_direction_idx(figpar=figpar, savedir=savedir, **info)

    # 11. Plots surprise index colormaps across sessions 
    elif analysis == "c":
        plot_surp_idx_cms(figpar=figpar, savedir=savedir, **info)

    # 12. Plots the surprise latency across sessions
    elif analysis == "u":
        plot_surp_latency(figpar=figpar, savedir=savedir, **info)

    # 13. Plots proportion of ROIs responses to both surprise types
    elif analysis == "p":
        plot_resp_prop(figpar=figpar, savedir=savedir, **info)

    else:
        warnings.warn(f"No plotting function for analysis {analysis}")

    plt.close("all")


#############################################
def get_linpla_idx(linpla_ord, line="L2/3", plane="soma", verbose=False, 
                   newline=False):
    """
    get_linpla_idx(linpla_ord)


    Required args:
        - linpla_ord (list): ordered list of line/plane combinations formatted 

    Optional args:
        - line (str)    : line (e.g., L2/3, L5)
                          default: "L2/3"
        - plane (str)   : plane (e.g., soma, dend)
                          default: "soma"
        - verbose (bool): if True and no data is found, this is logged
                          default: False
        - newline (bool): if True, text is logged on a new line
                          default: False

    Returns:
        - l_idx (int or None): line/plane combination index in linpla_ord or 
                               None if not found
    """

    data_name = f"{line} {plane[:4]}" # get data name
    if data_name not in linpla_ord:
        add = ""
        if newline:
            add = "\n"
        if verbose:
            logger.info(f"No data for {data_name}.", extra={"spacing": add})
        l_idx = None
    else:
        l_idx = linpla_ord.index(data_name)

    return l_idx


#############################################
def plot_data_signif(ax, sess_ns, sig_comps, lin_p_vals, maxes, 
                     p_val_thr=0.05, n_comps=1):
    """
    plot_data_signif(ax, sess_ns, sig_comps, lin_p_vals, maxes)

    Plot significance markers for significant session comparisons within and 
    across lines/plane combinations.

    Required args:
        - ax (plt Axis)         : axis
        - sess_ns (array-like)  : session numbers
        - sig_comps (array_like): list of session pair comparisons that are 
                                  significant (where the second session is 
                                  cycled in the inner loop, e.g., 0-1, 0-2, 
                                  1-2, including None sessions)
        - lin_p_vals            : p-values for each line comparison, 
                                  structured as line x session (np.nan for 
                                  sessions  missing in either plane)
        - maxes                 : max values used to adjust ylims, structured 
                                  as plane/line x session

    Optional args:
        - p_val_thr (float): p value threshold
                             default: 0.05
        - n_comps (int)    : total number of comparisons (used to modify p 
                             value threshold using Bonferroni correction)
                             default: 1
    """

    lines, planes, linpla_iter, pla_cols, _ , _ = \
        sess_plot_util.fig_linpla_pars()

    if n_comps == 0:
        return
    else:
        p_val_thr_corr = p_val_thr/n_comps

    shared = False
    if len(plot_util.get_shared_axes(ax, axis="y")) == 1:
        shared = True

    if shared:
        all_max = np.nanmax(np.asarray(maxes))
        if not np.isnan(all_max):
            ylims = ax[0, 0].get_ylim()
            ax[0, 0].set_ylim([ylims[0], np.max([ylims[1], all_max * 1.20])])
        prop = np.diff(ax[0, 0].get_ylim())[0]/10.0 # star position above data

    n_sess = len(sess_ns)
    n = 0
    # comparison number: first session start pts
    st_s1 = [sum(list(reversed(range(n_sess)))[:v]) for v in range(n_sess-1)] 
    highest = [0] * ax.size
    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        sub_ax = ax[pl, li]
        flat_idx = np.where(ax.reshape(-1) == sub_ax)[0][0]
        if not shared:
            ylims = sub_ax.get_ylim()
            if np.isfinite(ylims).all():
                sub_ax.set_ylim(
                    [ylims[0], 
                    np.nanmax([ylims[1], np.nanmax(maxes[i]) * 1.20])])
                # star position above data
                prop = np.diff(sub_ax.get_ylim())[0]/10.0
        if not(sig_comps[i] is None or len(sig_comps[i]) == 0):
            n = 0
            for p in sig_comps[i]:
                n += 1
                # get corresponding session numbers
                s1 = np.where(np.asarray(st_s1) <= p)[0][-1] 
                s2 = s1 + p - st_s1[s1] + 1
                y_pos = np.nanmax([maxes[i]]) + n * prop
                highest[flat_idx] = np.max([highest[flat_idx], y_pos])
                plot_util.plot_barplot_signif(
                    sub_ax, [sess_ns[s1], sess_ns[s2]], [y_pos], rel_y=0.03, 
                    color=pla_cols[pl], lw=3, mark_rel_y=0.09)

    # adjust for number of significance lines plotted
    for high_val, sub_ax in zip(highest, ax.reshape(-1)):
        if not np.isnan(high_val):
            ylims = sub_ax.get_ylim()
            if not np.isfinite(ylims).all():
                continue
            sub_ax.set_ylim(ylims[0], np.max([ylims[1], high_val * 1.1]))

    yposes = []
    for c in range(ax.shape[-1]):
        ylims = ax[1, c].get_ylim()
        ypos = ylims[1] + (ylims[1] - ylims[0])/10.0 # star position (btw plots)
        yposes.append(ypos)

    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        if pl == 1:
            sub_ax = ax[pl, li]
            plot_util.set_ticks(sub_ax, min_tick=min(sess_ns), 
                max_tick=max(sess_ns), n=len(sess_ns), pad_p=1.0/(len(sess_ns)))
            for s, p in enumerate(lin_p_vals[li]):
                if not np.isnan(p) and p < p_val_thr_corr:
                    # between subplots
                    plot_util.add_signif_mark(sub_ax, sess_ns[s], yposes[li], 
                        rel_y=0, color="k", fig_coord=True)


#############################################
def plot_per_mouse(sub_ax, mouse_st, sess_info, sess_ns=None, col=None, 
                   use_lab=True):
    """
    plot_per_mouse(sub_ax, mouse_st, sess_info)

    Plots statistics for each mouse.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - mouse_st (3D array)      : statistics across ROIs or seqs, 
                                     structured as mouse x session x stats
        - sess_info (list)         : list of dictionaries for each mouse 
                                     containing information from each session, 
                                     with None for missing sessions
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")

    Optional args:
        - sess_ns (array-like): session numbers for each session
                                default: None
        - col (str)           : plotting color
                                default: None
        - use_lab (bool)      : if True, label with mouse numbers is added
                                default: True
    """

    mouse_st = np.asarray(mouse_st)
    
    lab = "M"
    for m, m_info in enumerate(sess_info):
        lab_use = None
        
        # get this mouse's ns
        mouse_ns = set([n for n in m_info["mouse_ns"] if n is not None])
        if len(mouse_ns) != 1:
            raise ValueError("Should not be more than 1 mouse.")
        mouse_n = list(mouse_ns)[0]

        # extend label or create final version
        if use_lab:
            if m != len(sess_info) - 1:
                lab = f"{lab}{mouse_n}, "
            else: # if label to be included
                lab_use = f"{lab}{mouse_n}"

        # get non NaNs
        keep_idx = np.where(np.isfinite(mouse_st[m, :, 0]))[0]
        
        if sess_ns is None:
            sess_ns = np.asarray(range(len(mouse_st[m])))
        elif len(sess_ns) != len(mouse_st[m]):
            raise ValueError("Not as many session numbers as sessions.")

        plot_util.plot_errorbars(sub_ax, mouse_st[m, keep_idx, 0], 
            mouse_st[m, keep_idx, 1:].T, sess_ns[keep_idx], color=col, 
            label=lab_use, alpha=0.6)


#############################################
def plot_area_diff_stats(sub_ax, all_diff_st, sess_ns=None, mouse_mes=None, 
                         col=None):
    """
    plot_area_diff_stats(sub_ax, all_diff_st, sess_info)

    Plots area differences statistics across mice.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - all_diff_st (2D array)   : difference statistics across mice, 
                                     structured as session x stats

    Optional args:
        - sess_ns (array-like)  : session numbers for each session
                                  default: None
        - mouse_mes (array-like): mouse mean/median data (for legend), 
                                  structured as mouse x session
                                  default: None
        - col (str)             : plotting color
                                  default: None
    """
    
    all_diff_st = np.asarray(all_diff_st)

    if sess_ns is None:
        sess_ns = range(all_diff_st.shape[0])
    sess_ns = np.asarray(sess_ns)

    diff_st_nan = np.isfinite(all_diff_st[:, 0])
    keep_idx = np.where(diff_st_nan)[0]
    
    lab = None
    # calculate number of mice for legend
    if mouse_mes is not None:
        if mouse_mes.shape[1] != len(sess_ns):
            raise ValueError("'mouse_mes' second dimension should be "
                f"length {len(sess_ns)}.")
        n_mice_per = np.sum(np.isfinite(mouse_mes[:, keep_idx]), axis=0)
        min_mice, max_mice = np.min(n_mice_per), np.max(n_mice_per)
        lab = f"n={min_mice}"
        if min_mice != max_mice:
            lab = f"{lab}-{max_mice}"

    plot_util.plot_errorbars(sub_ax, all_diff_st[keep_idx, 0], 
        all_diff_st[keep_idx, 1:].T, sess_ns[keep_idx], color=col, 
        label=lab, alpha=0.8)


#############################################
def plot_signif_from_mouse_diffs(sub_ax, signif_idx, st_data, signs, 
                                 sess_ns=None, col=None):
    """
    plot_signif_from_mouse_diffs(sub_ax, signif_idx, st_data, signs)

    Plots and positions significance markers based on the mouse statistics.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - signif_idx (list)        : list of significant sessions, possibly by 
                                     tail
        - st_data (3D array)       : statistics for each mouse, structured as
                                     mouse x sess x stats
        - signs (array-like)       : sign of difference between actual and 
                                     shuffled mean

    Optional args:
        - sess_ns (array-like): session numbers for each session
                                default: None
        - col (str)           : plotting color
                                default: None
    """

    st_data = np.asarray(st_data)

    if sess_ns is None:
        sess_ns = range(st_data.shape[1])
    sess_ns = np.asarray(sess_ns)

    if len(signif_idx) == 2: # flatten if 2 tailed                
        signif_idx = [i for sub in signif_idx for i in sub]
    
    for idx in signif_idx: # each star separately
        x_val = sess_ns[idx].reshape([1])
        # get high value or low depending on tail
        if signs[idx] == 1:
            ys    = np.nansum([st_data[:, idx, 0], st_data[:, idx, -1]], axis=0)
            y_val = np.nanmax(ys, axis=0).reshape([1])
            rel_y = 0.09
        elif signs[idx] == -1:
            ys    = np.nansum([st_data[:, idx, 0], -st_data[:, idx, 1]], axis=0)
            y_val = np.nanmin(ys, axis=0).reshape([1])
            rel_y = -0.03
        plot_util.add_signif_mark(sub_ax, x_val, y_val, rel_y=rel_y, color=col)


#############################################
def plot_area_diff_per_linpla(sub_ax, sess_ns, diff_st, sess_info, 
                              mouse_diff_st=None, CI_vals=None, 
                              sign_sess=None, plot="tog", d="data", 
                              col="k", zero_line=False):
    """
    plot_area_diff_per_linpla(sub_ax, sess_ns, diff_st, sess_info)

    Plots data or CIs for a specific plane/line combination, and returns
    max y values for each session

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - sess_ns (array-like)     : session numbers
        - diff_st (2D array)       : difference stats across mice or ROIs, 
                                     structured 
                                         as session x stats
        - sess_info (list)         : list of dictionaries for each mouse 
                                     containing information from each session, 
                                     with None for missing sessions
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")

    Optional args:
        - mouse_diff_st (3D array): difference statistics across ROIs or 
                                    seqs, required if plot == "sep", 
                                    structured as 
                                        mouse x session x stats
                                    default: None
        - CI_vals (2D array)      : CIs values, required if d == "CI", 
                                    structured as 
                                      session x perc (med, lo, high) 
                                    (can be None if d == "data")
                                    default: None
        - sign_sess (list)        : list of significant sessions, possibly by 
                                    tail, used if d == "CI",
                                    default: None
        - plot (str)              : type of plot ("sep" for mice separated or 
                                    "tog" or "grped" for all mice/ROIs grouped)
                                    default: "tog"
        - d (str)                 : type of data to plot ("data" or "CIs")
                                    default: "data"
        - col (str)               : color to use for data
                                    default: "k"
        - zero_line (bool)        : if True, a horizontal line is plotted at 0 
                                    for data
                                    default: False

    Returns:
        if d == "data":
        - maxes (1D array): max values y value for each session
    """

    sess_ns = np.asarray(sess_ns)
    diff_st = np.asarray(diff_st)
    if mouse_diff_st is None:
        if plot == "sep":
            raise ValueError("Must provide 'mouse_diff_st' if 'plot' is 'sep'.")
    else:
        mouse_diff_st = np.asarray(mouse_diff_st)
    if d == "CIs" and CI_vals is None:
        raise ValueError("Must provide 'CI_vals' if 'd' is 'CIs'.")
        
    # plot the mouse lines
    if d == "data":
        if plot == "sep":
            maxes = np.nanmax(
                np.sum([mouse_diff_st[..., 0], mouse_diff_st[..., 1]], axis=0), 
                axis=0)
            plot_per_mouse(
                sub_ax, mouse_diff_st, sess_info, sess_ns, col, use_lab=True)
        elif plot in ["tog", "grped"]:
            maxes = np.sum([diff_st[..., 0], diff_st[..., 1]], axis=0)
            mouse_mes = None
            if mouse_diff_st is not None:
                mouse_mes = mouse_diff_st[:, :, 0]
            plot_area_diff_stats(sub_ax, diff_st, sess_ns, mouse_mes, col)
        else:
            gen_util.accepted_values_error("plot", plot, ["sep", "tog"])
        if zero_line:
            sub_ax.axhline(
                y=0, ls=HDASH, c="k", lw=3.0, alpha=0.5, zorder=-13)   
        return maxes
    
    elif d == "CIs":
        CI_vals  = np.asarray(CI_vals)
        keep_idx = np.where(np.isfinite(CI_vals[:, 0]))[0] # get mask
        
        # Add CIs
        width = 0.45
        plot_util.plot_CI(sub_ax, CI_vals[keep_idx, 1:].T, CI_vals[keep_idx, 0], 
            sess_ns[keep_idx], med_rat=0.03, width=width, zorder=-12)
        if plot == "sep": # Add mean/median lines
            plot_util.plot_lines(
                sub_ax, diff_st[keep_idx, 0], sess_ns[keep_idx], y_rat=0.025, 
                color=col, width=width)
            ypos_data = np.asarray(mouse_diff_st)
        else:
            ypos_data = np.expand_dims(diff_st, axis=0) 

        # plot significance markers
        if sign_sess:
            signs = np.sign(diff_st[:, 0] - CI_vals[:, 0]).astype(int)
            plot_signif_from_mouse_diffs(
                sub_ax, sign_sess, ypos_data, signs, sess_ns, col)
    else:
        gen_util.accepted_values_error("d", d, ["data", "CIs"])


#############################################
def plot_area_diff_acr_sess(analyspar, sesspar, stimpar, permpar, extrapar, 
                            sess_info, diff_info, figpar=None, lock=False, 
                            plot="tog"):
    """
    plot_area_diff_acr_sess(analyspar, sesspar, stimpar, permpar, extrapar, 
                            sess_info, diff_info)

    From dictionaries, plots statistics across ROIs or mice of difference 
    between splits.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - permpar (dict)  : dictionary with keys of PermPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                              parameters
            ["analysis"] (str): analysis type (e.g., "t")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - diff_info (dict)       : dictionary with difference info
            ["all_diff_stats"] (list)  : difference stats across mice, 
                                         structured as plane/line x session 
                                                                  x stats
            ["mouse_diff_stats"] (list): difference statistics across ROIs or 
                                         seqs, structured as 
                                             plane/line x mouse x session 
                                                        x stats
            ["CI_vals"] (list)         : CIs values, structured as
                                             plane/line x session 
                                                        x perc (med, lo, high)
            ["sign_sess"] (list)       : significant session indices, 
                                         structured as plane/line (x tails)
            ["p_vals_sess"] (list)     : p values for each session, structured 
                                         as plane/line x session
            ["linpla_ord"] (list)      : order list of planes/lines
            if extrapar["datatype"] == "roi":
                ["all_diff_st_grped"] (list): difference stats across ROIs 
                                              (grouped across mice), 
                                              structured as 
                                                plane/line x session x stats
                ["CI_vals_grped"] (list)    : CIs values across ROIs, 
                                              structured as 
                                                plane/line x session 
                                                    x perc (med, lo, high)
                ["lin_p_vals"] (list)       : p-values for each line comparison, 
                                              structured as line x session 
                                              (np.nan for sessions  missing in 
                                              either plane)
                ["max_comps_per"] (int)     : total number of comparisons
                ["p_vals_grped"] (list)     : p values for each comparison, 
                                              organized by session pairs (where 
                                              the second session is cycled in 
                                              the inner loop, e.g., 0-1, 0-2, 
                                              1-2, including empty groups)
                ["sign_sess_grped"] (list)  : significant session indices, 
                                              structured as plane/line (x tails)
                ["tot_n_comps"] (int)       : total number of comparisons

        - sess_info (nested list): nested list of dictionaries for each 
                                   line/plane x mouse containing information 
                                   from each session, with None for missing 
                                   sessions
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
                
    Optional args:
        - figpar (dict)     : dictionary containing the following figure 
                              parameter dictionaries
                              default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - lock (bool or str): if "surplock" or "reglock", differences being
                              plotted are surp or reg-locked, correspondingly. 
                              default: False    
        - plot (str)        : if "tog", average is taken across mice, otherwise, 
                              if "sep", each mouse is plotted separately
                              default: "tog"

    Returns:
        - fig (plt Figure) : pyplot figure
    """
 
    datatype = extrapar["datatype"]
    error = analyspar["error"]
    if datatype == "run":
        if plot == "sep":
            error = "None"
        elif plot == "grped":
            raise ValueError("grped plot types only supported for ROI datatype.")
    
    dimstr = ""
    if plot == "tog":
        dimstr = " across mice"
    elif datatype == "roi" and plot in ["sep", "grped"]:
        dimstr = f" across {sess_str_util.datatype_dim_str(datatype)}"
    
    grp_str, grp_str_pr = "", ""
    if plot == "grped":
        grp_str = "_grped"
        grp_str_pr = " (grouped)"
        
    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["bri_dir"], stimpar["bri_size"],
        stimpar["gabk"], "print")
    statstr_pr = sess_str_util.stat_par_str(analyspar["stats"], error, "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")
    
    sess_ns = np.asarray(sesspar["sess_n"])
    if sesspar["sess_n"] in ["any", "all"]:
        sess_ns = np.arange(len(sess_info[0][0]["sess_ns"])) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.reshape(-1).tolist())

    [lines, planes, linpla_iter, 
     pla_cols, _, n_plots] = sess_plot_util.fig_linpla_pars( 
        n_grps=len(diff_info["linpla_ord"]))
    linpla_iter = [[d, ll] for d in ["data", "CIs"] for ll in linpla_iter]
    figpar = sess_plot_util.fig_init_linpla(figpar)

    # correct p-value (Bonferroni)
    if plot == "grped":
        p_val_thr = permpar["p_val"]
        if diff_info["tot_n_comps"] != 0:
            p_val_thr_corr = p_val_thr/diff_info["tot_n_comps"]
        else:
            p_val_thr_corr = p_val_thr
        sig_comps = [[] for _ in range(len(linpla_iter))]
        maxes = np.full([len(linpla_iter), len(sess_ns)], np.nan)

    subtitle = "Surp - reg activity"
    if lock:
        prepost_str = "{}s pre v post".format(stimpar["post"])
        if lock == "surplock":
            subtitle += " locked to surprise onset"
        elif lock == "reglock":
            subtitle = "Reg - surp activity locked to regular onset"
        elif lock == "stim_onset":
            subtitle = "Stimulus - grayscreen activity locked to stimulus onset"
        elif lock == "stim_offset":
            subtitle = ("Grayscreen - stimulus activity locked to stimulus "
                "offset")
        else:
            raise ValueError("If lock is not False, it must be 'reglock' or \
                'surplock'.")
    else:
        prepost_str = sess_str_util.prepost_par_str(
            stimpar["pre"], stimpar["post"], str_type="print")
    title = (f"{subtitle}\n({prepost_str} seqs) for {stimstr_pr}\n{statstr_pr}"
        f"{dimstr}{grp_str_pr} (sess {sess_ns_str}{dendstr_pr})")

    fig, ax = plot_util.init_fig(n_plots, **figpar["init"])
    fig.suptitle(title, y=1.03, weight="bold")

    for i, (d, [line, pla]) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        l_idx = get_linpla_idx(
            diff_info["linpla_ord"], line, pla, verbose=(d==0), newline=(i==0))
        if l_idx is None:
            continue

        ypos = plot_area_diff_per_linpla(ax[pl, li], sess_ns, 
            diff_info[f"all_diff_stats{grp_str}"][l_idx], sess_info[l_idx], 
            diff_info["mouse_diff_stats"][l_idx], 
            diff_info[f"CI_vals{grp_str}"][l_idx],
            diff_info[f"sign_sess{grp_str}"][l_idx], 
            plot=plot, d=d, col=pla_cols[pl], zero_line=False)
 
        # check p_val signif
        if d == "data" and plot == "grped":
            maxes[i] = ypos
            all_p_vals = diff_info["p_vals_grped"][l_idx]
            for p, p_val in enumerate(all_p_vals):
                if not np.isnan(p_val) and p_val < p_val_thr_corr:
                    sig_comps[i].append(p)

    if plot == "grped":
        plot_data_signif(ax, sess_ns, sig_comps, diff_info["lin_p_vals"], 
            maxes, p_val_thr=permpar["p_val"], n_comps=diff_info["tot_n_comps"])

    # Add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(ax, fluor=analyspar["fluor"], 
        area=True, datatype=datatype, lines=lines, planes=planes, 
        xticks=sess_ns, xlab="Sessions", kind="reg")

    return fig
   

#############################################
def plot_surp_area_diff(analyspar, sesspar, stimpar, basepar, permpar, extrapar, 
                        sess_info, diff_info, figpar=None, savedir=None):
    """
    plot_surp_area_diff(analyspar, sesspar, stimpar, basepar, permpar, extrapar, 
                        sess_info, diff_stats)

    From dictionaries, plots statistics across ROIs or mice of difference 
    between regular and surprise averaged across sequences.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - basepar (dict)  : dictionary with keys of BasePar namedtuple
        - permpar (dict)  : dictionary with keys of PermPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "t")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - sess_info (list): list of dictionaries for each mouse containing 
                              information from each session 
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - diff_info (dict): dictionary containing difference information
            ["all_diff_stats"] (list)  : difference stats across mice, 
                                         structured as plane/line x session 
                                                                  x stats
            ["mouse_diff_stats"] (list): difference statistics across ROIs or 
                                         seqs, structured as 
                                             plane/line x mouse x session 
                                                        x stats
            ["CI_vals"] (list)         : CIs values, structured as
                                             plane/line x session 
                                                        x perc (med, lo, high)
            ["sign_sess"] (list)       : significant session indices, 
                                         structured as plane/line (x tails)
            ["p_vals_sess"] (list)     : p values for each session, structured 
                                         as plane/line x session
            ["linpla_ord"] (list)      : order list of planes/lines
            if extrapar["datatype"] == "roi":
                ["all_diff_st_grped"] (list): difference stats across ROIs 
                                              (grouped across mice), 
                                              structured as 
                                                plane/line x session x stats
                ["CI_vals_grped"] (list)    : CIs values across ROIs, 
                                              structured as 
                                                plane/line x session 
                                                    x perc (med, lo, high)
                ["lin_p_vals"] (list)       : p-values for each line comparison, 
                                              structured as line x session 
                                              (np.nan for sessions  missing in 
                                              either plane)
                ["max_comps_per"] (int)     : total number of comparisons
                ["p_vals_grped"] (list)     : p values for each comparison, 
                                              organized by session pairs (where 
                                              the second session is cycled in 
                                              the inner loop, e.g., 0-1, 0-2, 
                                              1-2, including empty groups)
                ["sign_sess_grped"] (list)  : significant session indices, 
                                              structured as plane/line (x tails)
                ["tot_n_comps"] (int)       : total number of comparisons

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
    
    Returns:
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): name under which the figure is saved
    """
    
    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["bri_dir"], stimpar["bri_size"], stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    datatype = extrapar["datatype"]

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    base_str = sess_str_util.base_par_str(basepar["baseline"])[1:]

    if savedir is None:
        savedir = os.path.join(
            figpar["dirs"][datatype], 
            figpar["dirs"]["acr_sess"], 
            sess_str_util.get_stimdir(stimpar["stimtype"], stimpar["gabfr"]), 
            base_str)

    gen_savename = f"{datatype}_surp_diff_{sessstr}{dendstr}"
    part = "surp_diff"
    add_idx = gen_savename.find(part) + len(part)

    for p, plot in enumerate(["sep", "tog", "grped"]):
        if plot == "grped" and datatype == "run":
            continue
        fig = plot_area_diff_acr_sess(
            analyspar, sesspar, stimpar, permpar, extrapar, sess_info, 
            diff_info, figpar=figpar, lock=False, plot=plot)


        savename = f"{gen_savename[:add_idx]}_{plot}{gen_savename[add_idx:]}"
        fulldir = plot_util.savefig(
            fig, savename, savedir, log_dir=(p==0), **figpar["save"])

    return fulldir, gen_savename


#############################################
def plot_lock_area_diff(analyspar, sesspar, stimpar, basepar, permpar, extrapar, 
                        sess_info, diff_info, figpar=None, savedir=None):
    """
    plot_lock_area_diff(analyspar, sesspar, stimpar, basepar, permpar, extrapar, 
                        sess_info, diff_info)

    From dictionaries, plots statistics across ROIs or mice of difference 
    between regular and surprise averaged across sequences, locked to 
    transitions from regular to surprise, and vice versa.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - basepar (dict)  : dictionary with keys of BasePar namedtuple
        - permpar (dict)  : dictionary with keys of PermPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "l")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - sess_info (list): list of dictionaries containing information from 
                            each session, structured as 
                            [surp-locked, reg-locked]
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - diff_info (list): list of dictionaries containing difference 
                            information, structured as [surp-locked, reg-locked]
            ["all_diff_stats"] (list)  : difference stats across mice, 
                                         structured as plane/line x session 
                                                                  x stats
            ["mouse_diff_stats"] (list): difference statistics across ROIs or 
                                         seqs, structured as 
                                             plane/line x mouse x session 
                                                        x stats
            ["CI_vals"] (list)         : CIs values, structured as
                                             plane/line x session 
                                                        x perc (med, lo, high)
            ["sign_sess"] (list)       : significant session indices, 
                                         structured as plane/line (x tails)
            ["p_vals_sess"] (list)     : p values for each session, structured 
                                         as plane/line x session
            ["linpla_ord"] (list)      : order list of planes/lines
            if extrapar["datatype"] == "roi":
                ["all_diff_st_grped"] (list): difference stats across ROIs 
                                              (grouped across mice), 
                                              structured as 
                                                plane/line x session x stats
                ["CI_vals_grped"] (list)    : CIs values across ROIs, 
                                              structured as 
                                                plane/line x session 
                                                    x perc (med, lo, high)
                ["lin_p_vals"] (list)       : p-values for each line comparison, 
                                              structured as line x session 
                                              (np.nan for sessions  missing in 
                                              either plane)
                ["max_comps_per"] (int)     : total number of comparisons
                ["p_vals_grped"] (list)     : p values for each comparison, 
                                              organized by session pairs (where 
                                              the second session is cycled in 
                                              the inner loop, e.g., 0-1, 0-2, 
                                              1-2, including empty groups)
                ["sign_sess_grped"] (list)  : significant session indices, 
                                              structured as plane/line (x tails)
                ["tot_n_comps"] (int)       : total number of comparisons

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
    
    Returns:
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): name under which the figure is saved
    """
 
    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["bri_dir"], stimpar["bri_size"], stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    datatype = extrapar["datatype"]

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    base_str = sess_str_util.base_par_str(basepar["baseline"])[1:]

    if savedir is None:
        savedir = os.path.join(
            figpar["dirs"][datatype], 
            figpar["dirs"]["acr_sess"], 
            sess_str_util.get_stimdir(stimpar["stimtype"], stimpar["gabfr"]), 
            base_str)

    gen_savename = f"{datatype}_lock_diff_{sessstr}{dendstr}"

    for l, lock in enumerate(["surplock", "reglock"]):
        part = "lock_diff"
        lock_savename = gen_savename.replace("lock", lock)
        add_idx = lock_savename.find(part) + len(part)

        for p, plot in enumerate(["sep", "tog", "grped"]):
            if plot == "grped" and datatype == "run":
                continue
            fig = plot_area_diff_acr_sess(
                analyspar, sesspar, stimpar, permpar, extrapar, sess_info[l], 
                diff_info[l], figpar=figpar, lock=lock, plot=plot)

            savename = (f"{lock_savename[:add_idx]}_{plot}"
                f"{lock_savename[add_idx:]}")
            fulldir = plot_util.savefig(
                fig, savename, savedir, log_dir=(p==0), **figpar["save"])

    return fulldir, gen_savename


#############################################
def plot_stim_grayscr_diff(analyspar, sesspar, stimpar, basepar, permpar, 
                           extrapar, sess_info, diff_info, figpar=None, 
                           savedir=None):
    """
    plot_stim_grayscr_diff(analyspar, sesspar, stimpar, basepar, permpar, 
                           extrapar, sess_info, diff_info)

    From dictionaries, plots statistics across ROIs or mice of difference 
    between grayscreen and stimulus averaged across sequences, locked to 
    onset and offset transitions.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - basepar (dict)  : dictionary with keys of BasePar namedtuple
        - permpar (dict)  : dictionary with keys of PermPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "b")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - sess_info (list): list of dictionaries containing information from 
                            each session, structured as 
                            [stim-onset, stim-offset]
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - diff_info (list): list of dictionaries containing difference 
                            information, structured as [stim-onset, stim-offset]
            ["all_diff_stats"] (list)  : difference stats across mice, 
                                         structured as plane/line x session 
                                                                  x stats
            ["mouse_diff_stats"] (list): difference statistics across ROIs or 
                                         seqs, structured as 
                                             plane/line x mouse x session 
                                                        x stats
            ["CI_vals"] (list)         : CIs values, structured as
                                             plane/line x session 
                                                        x perc (med, lo, high)
            ["sign_sess"] (list)       : significant session indices, 
                                         structured as plane/line (x tails)
            ["p_vals_sess"] (list)     : p values for each session, structured 
                                         as plane/line x session
            ["linpla_ord"] (list)      : order list of planes/lines
            if extrapar["datatype"] == "roi":
                ["all_diff_st_grped"] (list): difference stats across ROIs 
                                              (grouped across mice), 
                                              structured as 
                                                plane/line x session x stats
                ["CI_vals_grped"] (list)    : CIs values across ROIs, 
                                              structured as 
                                                plane/line x session 
                                                    x perc (med, lo, high)
                ["lin_p_vals"] (list)       : p-values for each line comparison, 
                                              structured as line x session 
                                              (np.nan for sessions  missing in 
                                              either plane)
                ["max_comps_per"] (int)     : total number of comparisons
                ["p_vals_grped"] (list)     : p values for each comparison, 
                                              organized by session pairs (where 
                                              the second session is cycled in 
                                              the inner loop, e.g., 0-1, 0-2, 
                                              1-2, including empty groups)
                ["sign_sess_grped"] (list)  : significant session indices, 
                                              structured as plane/line (x tails)
                ["tot_n_comps"] (int)       : total number of comparisons

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
    
    Returns:
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): name under which the figure is saved
    """
 
    if stimpar["stimtype"] != "both":
        raise ValueError("Stimulus grayscreen analysis must include both "
            "stimulus types.")

    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["bri_dir"], stimpar["bri_size"], stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    datatype = extrapar["datatype"]

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    base_str = sess_str_util.base_par_str(basepar["baseline"])[1:]

    if savedir is None:
        savedir = os.path.join(
            figpar["dirs"][datatype], 
            figpar["dirs"]["acr_sess"], 
            sess_str_util.get_stimdir(stimpar["stimtype"], stimpar["gabfr"]), 
            base_str)

    gen_savename = f"{datatype}_stim_diff_{sessstr}{dendstr}"

    for l, lock in enumerate(["stim_onset", "stim_offset"]):
        part = f"{lock}_diff"
        lock_savename = gen_savename.replace("stim", lock)
        add_idx = lock_savename.find(part) + len(part)

        for p, plot in enumerate(["sep", "tog", "grped"]):
            if plot == "grped" and datatype == "run":
                continue
            fig = plot_area_diff_acr_sess(
                analyspar, sesspar, stimpar, permpar, extrapar, sess_info[l], 
                diff_info[l], figpar=figpar, lock=lock, plot=plot)

            savename = (f"{lock_savename[:add_idx]}_{plot}"
                f"{lock_savename[add_idx:]}")
            fulldir = plot_util.savefig(
                fig, savename, savedir, log_dir=(p==0), **figpar["save"])

    return fulldir, gen_savename


#############################################
def plot_traces(sub_ax, xran, trace_st, lock=False, col="k", lab=True, 
                stimtype="gabors", gabfr=0, ls=None, reg_col="gray"):
    """
    plot_traces(sub_ax, xran, trace_st)

    Plot data trace splits (single set).

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - xran (array-like)        : second values for each frame
        - trace_st (4D array)      : trace statistics, structured as
                                         split x frame x stats

    Optional args:
        - lock (bool)   : if True, plotted data is locked 
                          default: False
        - col  (str)    : colour for non-regular/non-locked data
                          default: "k"
        - lab (bool)    : if True, data label is included for legend
                          default: True
        - stimtype (str): stimtype ("gabors" or "bricks")
                          default: "gabors"
        - gabfr (int)   : gabor start frame number
                          default: 0
        - ls (float)    : trace line style
                          default: None
        - reg_col (str) : color for regular data
                          default: "gray"
    """

    if lock:
        cols = [col, col]
    else:
        cols = [reg_col, col]
    
    names = ["reg", "surp"]
    if lock and "stim" in lock:
        names = ["off", "on"]

    xran = np.asarray(xran)

    # horizontal and vertical 0 line
    all_rows = True
    if not sub_ax.is_last_row() or all_rows:
        alpha = 0.5
    else:
        # ensures that if no data is plotted in these subplots, there is at 
        # least a vertically finite horizontal object to prevent indefinite 
        # axis limit expansion bug.
        alpha = 0
    sub_ax.axhline(y=0, ls=HDASH, c="k", lw=3.0, alpha=alpha, zorder=-13)
    sub_ax.axvline(x=0, ls=VDASH, c="k", lw=3.0, alpha=0.5, zorder=-13)
        
    xticks = None
    if lock:
        xticks = np.linspace(-np.max(xran), np.max(xran), 5)

    if lock:
        rev_xran = xran[::-1] * -1
        full_xran = np.concatenate([rev_xran, xran])
    else:
        full_xran = xran

    if stimtype == "gabors":
        shade_col = DARKRED
        shade_lim = "pos"
        if lock in ["reglock", "stim_offset"]:
            shade_lim = "neg"
        sess_plot_util.plot_gabfr_pattern(sub_ax, full_xran, offset=gabfr, 
            bars_omit=[0], shade_col=shade_col, alpha=0.2, shade_lim=shade_lim)

    trace_st = np.asarray(trace_st)
    for i, (col, name) in enumerate(zip(cols, names)):
        label = name if lab and not lock else None
        if lock and name not in lock:
            xran_use = rev_xran
        else:
            xran_use = xran
        if lock in ["reglock", "stim_offset"]:
            i = 1 - i # data ordered as [non-reg, reg] instead of vv
        plot_util.plot_traces(sub_ax, xran_use, trace_st[i, :, 0], 
            trace_st[i, :, 1:], label=label, alpha_line=0.8, color=col, 
            xticks=xticks, ls=ls)


#############################################
def plot_single_prog(sub_ax, prog_st, col="k", label=None, alpha_line=0.8):
    """
    plot_single_prog(sub_ax, prog_st)

    Plot surprise or regular progression data (single set).

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - prog_st (2D array)       : progression statistics, structured as
                                         seqs x stats

    Optional args:
        - col  (str)      : colour for data
                            default: "k"
        - alpha_line (num): plt alpha variable controlling line transparency
                            default: 0.5
        - label (str)     : label for legend
                            default: None

    Returns:
        - n (int): number of sequences plotted

    """

    prog_st = np.asarray(prog_st)
    xran = np.arange(1, prog_st.shape[0] + 1)

    # ensures that if no data is plotted in these subplots, there is at 
    # least a vertically finite horizontal object to prevent indefinite 
    # axis expansion. (Occurs when, for some axes, there are only 
    # indefinite objects, e.g. shading or h/vline.) 
    sub_ax.axvline(x=0, ls=VDASH, c="k", lw=3.0, alpha=0, zorder=-13)
     
    sub_ax.axhline(y=0, ls=HDASH, c="k", lw=3.0, alpha=0.5, zorder=-13)
        
    plot_util.plot_traces(sub_ax, xran, prog_st[:, 0], prog_st[:, 1:], 
        alpha_line=alpha_line, color=col, label=label)

    # sets x ticks based on full dataset
    n = len(prog_st)
    
    return n


#############################################
def plot_traces_acr_sess(analyspar, sesspar, stimpar, extrapar, sess_info, 
                         trace_info, figpar=None, lock=False, grp="mice"):
    """
    plot_traces_acr_sess(analyspar, sesspar, stimpar, extrapar, sess_info, 
                         trace_info)

    From dictionaries, plots traces across ROIs or mice for split sequences.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "t")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/plane x mouse containing information 
                                   from each session, with None for missing 
                                   sessions
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - trace_info (dict)      : dictionary with difference info
            ["linpla_ord"] (list) : order list of planes/lines
            ["trace_stats"] (list): trace statistics, structured as
                                    plane/line x session x reg/surp x frame 
                                               x stats
            ["xran"] (list)       : second values for each frame
            if datatype == "roi":
                ["trace_st_acr_rois"] (list): trace statistics across ROIs, 
                                              grouped across mice, structured 
                                              as session x reg/surp 
                                                         x frame x stats
                                              (or surp/reg if surp == "reglock")

    Optional args:
        - figpar (dict)     : dictionary containing the following figure 
                              parameter dictionaries
                              default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - lock (bool or str): if "surplock" or "reglock", differences being
                              plotted are surp or reg-locked, correspondingly. 
                              If "stim_onset" or "stim_offset", differences 
                              plotted are from onset or offset.
                              default: False     
        - grp (str)         : if "mice", data is grouped per mouse, if "roi", 
                              data is grouped across mice
                              default: "mice"
    Returns:
        - fig (plt Figure) : pyplot figure
    """
 
    datatype = extrapar["datatype"]
    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["bri_dir"], stimpar["bri_size"],
        stimpar["gabk"], "print")
    statstr_pr = sess_str_util.stat_par_str(
        analyspar["stats"], analyspar["error"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")
    
    sess_ns = np.asarray(sesspar["sess_n"])
    if sesspar["sess_n"] in ["any", "all"]:
        sess_ns = np.asarray(range(len(sess_info[0][0]["sess_ns"]))) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.reshape(-1).tolist())
    n_sess = len(sess_ns)

    [lines, planes, linpla_iter, pla_cols, _, n_plots] = \
        sess_plot_util.fig_linpla_pars(
            n_sess=n_sess, n_grps=len(trace_info["linpla_ord"]))
    figpar = sess_plot_util.fig_init_linpla(figpar, kind="traces", n_sub=n_sess)

    subtitle = "Surp v reg activity"
    if lock:
        prepost_str = "{}s pre v post".format(stimpar["post"])
        if lock == "surplock":
            subtitle += " locked to surprise onset"
        elif lock == "reglock":
            subtitle = "Reg v surp activity locked to regular onset"
        elif lock == "stim_onset":
            subtitle = "Stimulus vs grayscreen activity locked to onset"
        elif lock == "stim_offset":
            subtitle = "Grayscreen vs stimulus activity locked to offset"
        else:
            raise ValueError("If lock is not False, it must be 'reglock', "
                "'surplock', 'stim_onset' or 'stim_offset'.")
    else:
        prepost_str = sess_str_util.prepost_par_str(
            stimpar["pre"], stimpar["post"], str_type="print")
    
    dim_str = "mice"
    grp_str = ""
    if grp == "rois":
        if datatype == "run":
            raise ValueError(f"Grouping ('grp') only across ROIs for ROI "
                "datatype.")
        dim_str = "ROIs (grouped)"
        grp_str = "_grped"
    title = (f"{subtitle}\n({prepost_str} seqs) for {stimstr_pr}\n"
        f"{statstr_pr} across {dim_str} (sess {sess_ns_str}{dendstr_pr})")

    fig, ax = plot_util.init_fig(n_plots, **figpar["init"])
    fig.suptitle(title, y=1.0, weight="bold")

    xran = trace_info["xran"]
    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        l_idx = get_linpla_idx(
            trace_info["linpla_ord"], line, pla, verbose=True, newline=(i==0))
        if l_idx is None:
            continue

        # plot the mouse traces
        for s in range(n_sess):
            sub_ax = ax[s + pl * n_sess, li]
            lab = (li == 0 and s == 0)
            plot_traces(
                sub_ax, xran, trace_info[f"trace_stats{grp_str}"][l_idx][s], 
                lock, pla_cols[pl], lab, stimpar["stimtype"], stimpar["gabfr"])

    # Add plane, line info to plots
    if stimpar["stimtype"] == "gabors":
        xticks = sess_plot_util.get_gab_time_xticks(xran, lock=lock)

    else:
        xticks = sub_ax.get_xticks()
        if len(xticks) > 1:
            diff = np.min(np.diff(xticks))
            n_dig = - np.floor(np.log10(np.absolute(diff))).astype(int) + 1
            xticks = [np.around(v, n_dig) for v in xticks]
    
    sess_plot_util.format_linpla_subaxes(ax, fluor=analyspar["fluor"], 
        area=False, datatype=datatype, lines=lines, planes=planes, 
        sess_ns=sess_ns, xticks=xticks, kind="traces")

    return fig
   

#############################################
def plot_surp_traces(analyspar, sesspar, stimpar, basepar, extrapar, sess_info, 
                     trace_info, figpar=None, savedir=None):
    """
    plot_surp_traces(analyspar, sesspar, stimpar, basepar, extrapar, sess_info, 
                     trace_info)

    From dictionaries, plots traces across ROIs or mice of difference 
    for regular and surprise averaged across sequences.
    
    Returns general figure name and save directory path.
    
    Required args:
        - analyspar (dict)       : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)         : dictionary with keys of SessPar namedtuple
        - stimpar (dict)         : dictionary with keys of StimPar namedtuple
        - basepar (dict)         : dictionary with keys of BasePar namedtuple
        - extrapar (dict)        : dictionary containing additional analysis 
                                   parameters
            ["analysis"] (str): analysis type (e.g., "t")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/plane x mouse containing information 
                                   from each session, with None for missing 
                                   sessions
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - trace_info (dict)      : dictionary with difference info
            ["linpla_ord"] (list) : order list of planes/lines            
            ["trace_stats"] (list): trace statistics, structured as
                                    plane/line x session x reg/surp x frame 
                                               x stats
            if datatype == "roi":
                ["trace_st_acr_rois"] (list): trace statistics across ROIs, 
                                              grouped across mice, structured 
                                              as session x reg/surp 
                                                         x frame x stats
                                              (or surp/reg if surp == "reglock")
            ["xran"] (list)       : second values for each frame

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
    
    Returns:
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): general name under which the figures are saved
    """
    
    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["bri_dir"], stimpar["bri_size"], stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    datatype = extrapar["datatype"]

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    base_str = sess_str_util.base_par_str(basepar["baseline"])[1:]
    if savedir is None:
        savedir = os.path.join(
            figpar["dirs"][datatype], 
            figpar["dirs"]["acr_sess"], 
            sess_str_util.get_stimdir(stimpar["stimtype"], stimpar["gabfr"]), 
            base_str)

    gen_savename = f"{datatype}_surp_tr_{sessstr}{dendstr}"

    for grp in ["mice", "rois"]:
        if grp == "rois" and datatype == "run":
            continue
        fig = plot_traces_acr_sess(
            analyspar, sesspar, stimpar, extrapar, sess_info, trace_info, 
            figpar=figpar, lock=False, grp=grp)
        
        savename = gen_savename
        if grp == "rois":
            savename = f"{gen_savename}_grped"
    
        fulldir = plot_util.savefig(fig, savename, savedir, **figpar["save"])

    return fulldir, gen_savename


#############################################
def plot_lock_traces(analyspar, sesspar, stimpar, basepar, extrapar, sess_info, 
                     trace_info, figpar=None, savedir=None):
    """
    plot_lock_traces(analyspar, sesspar, stimpar, extrapar, sess_info,
                     trace_info)

    From dictionaries, plots traces across ROIs or mice for regular and 
    surprise sequences, locked to transitions from regular to surprise, then
    from surprise to regular.
    
    Returns general figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - basepar (dict)  : dictionary with keys of BasePar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str)        : analysis type (e.g., "t")
            ["datatype"] (str)        : datatype (e.g., "run", "roi")
        - sess_info (list): list of dictionaries containing information from 
                            each session, structured as 
                            [surp-locked, reg-locked]
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - trace_info (list): list of dictionaries containing trace 
                             information, structured as 
                             [surp-locked, reg-locked]
            ["linpla_ord"] (list) : order list of planes/lines            
            ["trace_stats"] (list): trace statistics, structured as
                                    plane/line x session x reg/surp x frame 
                                               x stats
            if datatype == "roi":
                ["trace_st_grped"] (list): trace statistics across ROIs, 
                                           grouped across mice, structured 
                                           as session x reg/surp 
                                                      x frame x stats
                                           (or surp/reg if surp == "reglock")
            ["xran"] (list)       : second values for each frame

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
    
    Returns:
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): general name under which the figures are saved
    """
 
    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["bri_dir"], stimpar["bri_size"], stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    datatype = extrapar["datatype"]

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    gen_savename = f"{datatype}_lock_tr_{sessstr}{dendstr}"

    stimdir = sess_str_util.get_stimdir(stimpar["stimtype"], stimpar["gabfr"])

    for l, lock in enumerate(["surplock", "reglock"]):
        for grp in ["mice", "rois"]:
            if grp == "rois" and datatype == "run":
                continue
            fig = plot_traces_acr_sess(analyspar, sesspar, stimpar, extrapar, 
                sess_info[l], trace_info[l], figpar=figpar, lock=lock, grp=grp)

            base_str = sess_str_util.base_par_str(basepar["baseline"])[1:]
            if savedir is None:
                savedir = os.path.join(
                    figpar["dirs"][datatype], 
                    figpar["dirs"]["acr_sess"], 
                    stimdir, 
                    base_str)

            savename = gen_savename.replace("lock", lock)
            if grp == "rois":
                savename = f"{savename}_grped"
        
            fulldir = plot_util.savefig(
                fig, savename, savedir, **figpar["save"])

    return fulldir, gen_savename


#############################################
def plot_stim_grayscr_traces(analyspar, sesspar, stimpar, basepar, extrapar, 
                             sess_info, trace_info, figpar=None, savedir=None):
    """
    plot_stim_grayscr_traces(analyspar, sesspar, stimpar, basepar, extrapar, 
                             sess_info, trace_info)

    From dictionaries, plots traces across ROIs or mice for stimulus onset and 
    offset, locked to transitions.
    
    Returns general figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - basepar (dict)  : dictionary with keys of BasePar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str)        : analysis type (e.g., "a")
            ["datatype"] (str)        : datatype (e.g., "run", "roi")
        - sess_info (list): list of dictionaries containing information from 
                            each session, structured as 
                            [stim-onset, stim_offset]
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - trace_info (list): list of dictionaries containing trace 
                             information, structured as 
                             [stim-onset, stim_offset]
            ["linpla_ord"] (list) : order list of planes/lines            
            ["trace_stats"] (list): trace statistics, structured as
                                    plane/line x session x off/on x frame 
                                               x stats
            if datatype == "roi":
                ["trace_st_grped"] (list): trace statistics across ROIs, 
                                           grouped across mice, structured 
                                           as session x off/on 
                                                      x frame x stats
                                           (or on/off for stim_offset)
            ["xran"] (list)       : second values for each frame

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
    
    Returns:
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): general name under which the figures are saved
    """
 
    if stimpar["stimtype"] != "both":
        raise ValueError("Stimulus grayscreen analysis must include both "
            "stimulus types.")

    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["bri_dir"], stimpar["bri_size"], stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    datatype = extrapar["datatype"]

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    gen_savename = f"{datatype}_stim_{sessstr}{dendstr}"

    stimdir = sess_str_util.get_stimdir(stimpar["stimtype"], stimpar["gabfr"])

    for l, lock in enumerate(["stim_onset", "stim_offset"]):
        for grp in ["mice", "rois"]:
            if grp == "rois" and datatype == "run":
                continue
            fig = plot_traces_acr_sess(analyspar, sesspar, stimpar, extrapar, 
                sess_info[l], trace_info[l], figpar=figpar, lock=lock, grp=grp)

            base_str = sess_str_util.base_par_str(basepar["baseline"])[1:]
            if savedir is None:
                savedir = os.path.join(
                    figpar["dirs"][datatype], 
                    figpar["dirs"]["acr_sess"], 
                    stimdir, 
                    base_str)

            savename = gen_savename.replace("stim", lock)
            if grp == "rois":
                savename = f"{savename}_grped"
        
            fulldir = plot_util.savefig(
                fig, savename, savedir, **figpar["save"])

    return fulldir, gen_savename


#############################################
def plot_prog_per_mouse(ax, mouse_st, sess_info, sess_ns=None, cols=None, 
                        use_lab=True, prev_n=0, diff=True):
    """
    plot_prog_per_mouse(ax, mouse_st, sess_info)

    Plots surprise/regular sequence progression statistics for each mouse.

    Required args:
        - ax (1D array of subplots): array of subplots
        - mouse_st (3D array)      : statistics across ROIs or seqs, 
                                     structured as mouse x session x surps 
                                        x seq x stats
        - sess_info (list)         : list of dictionaries for each mouse 
                                     containing information from each session, 
                                     with None for missing sessions
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")

    Optional args:
        - sess_ns (array-like): session numbers for each session
                                default: None
        - cols (list)         : plotting colors (1 if diff, 2 otherwise)
                                default: None
        - use_lab (bool)      : if True, label with mouse numbers is added
                                default: True
        - diff (bool)         : if True, difference between surp/reg is taken
                                default: True
    
    Returns:
        - n (int): number of sequences plotted
    """

    mouse_st = np.asarray(mouse_st)
    n = mouse_st.shape[2]

    if sess_ns is None:
        sess_ns = np.asarray(range(len(mouse_st[0])))
    elif len(sess_ns) != len(mouse_st[0]):
        raise ValueError("Not as many session numbers as sessions.")

    labs = [None for _ in sess_ns]
    labs_used = [False for _ in sess_ns]
    for i in range(len(sess_ns)):
        mouse_ns = []
        for m, m_info in enumerate(sess_info):
            if np.isfinite(np.asarray(mouse_st[m][i])).any():
                mouse_ns.append(str(m_info["mouse_ns"][i]))
        labs[i] = "M{}".format(", ".join(mouse_ns))

    n_dims = 2 - diff

    cols = gen_util.list_if_not(cols) if diff else cols
    if len(cols) != n_dims:
        raise ValueError("Must provide 1 cols if diff, else 2.")

    for m, m_info in enumerate(sess_info):
        for i in range(len(sess_ns)):
            stats = np.asarray(mouse_st[m][i])

            for d in range(n_dims):
                if diff:
                    alpha = 0.6
                    stats_sp = stats
                else:
                    alpha = 0.4
                    stats_sp = stats[d]

                if not np.isfinite(stats_sp).any():
                    continue
                lab = None
                if d == 0 and not labs_used[i]:
                    lab = labs[i]
                    labs_used[i] = True
                # plot
                _ = plot_single_prog(
                    ax[i], stats_sp, col=cols[d], label=lab, alpha_line=alpha)

    return n


#############################################
def plot_prog_acr_sess(analyspar, sesspar, stimpar, extrapar, sess_info, 
                       prog_info, figpar=None, prog="progsurp", 
                       plot="tog", diff=True):
    """
    plot_prog_acr_sess(analyspar, sesspar, stimpar, extrapar, sess_info, 
                       prog_info)

    From dictionaries, plots progression of surprise or regular sequences 
    within sessions across ROIs or mice.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "t")
            ["datatype"] (str): datatype (e.g., "run", "roi")
            ["position"] (str): position plotted (e.g., "first", "second", etc.)
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/plane x mouse containing information 
                                   from each session, with None for missing 
                                   sessions
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - prog_info (dict)       : dictionary with progression info
            ["linpla_ord"] (list)       : order list of planes/lines            
            ["prog_stats"] (list)  : surprise progression stats across mice, 
                                          structured as plane/line x session 
                                            x surps x seq x stats
            ["mouse_prog_stats"] (list): surprise progression stats across 
                                          ROIs, structured as 
                                             plane/line x mouse x session 
                                                x surps x seq x stats
            if extrapar["datatype"] == "roi":
                ["prog_stats_grped"] (list): surprise progression stats across 
                                              ROIs (grouped across mice), 
                                              structured as 
                                                 plane/line x session x surps 
                                                     x seqs x stats

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - prog (str)   : if "progsurp" or "progreg", differences between
                         surprise and previous regular are plotted. If 
                         "progreg", v.v. 
                         default: "progsurp"     
        - plot (str)   : if "tog", data is grouped across mouse, if "grped", 
                         data is grouped across ROIs, if "sep", data is 
                         separated by mouse
                         default: "tog"
        - diff (bool)  : if True, difference is plotted instead of surp, reg 
                         separately
                         default: True

    Returns:
        - fig (plt Figure) : pyplot figure
    """
 
    datatype = extrapar["datatype"]
    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["bri_dir"], stimpar["bri_size"],
        stimpar["gabk"], "print")
    statstr_pr = sess_str_util.stat_par_str(
        analyspar["stats"], analyspar["error"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")
    pos = extrapar["position"]

    sess_ns = np.asarray(sesspar["sess_n"])
    if sesspar["sess_n"] in ["any", "all"]:
        sess_ns = np.asarray(range(len(sess_info[0][0]["sess_ns"]))) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.reshape(-1).tolist())
    n_sess = len(sess_ns)

    [lines, planes, linpla_iter, pla_cols, _, n_plots] = \
        sess_plot_util.fig_linpla_pars(n_sess=n_sess, 
            n_grps=len(prog_info["linpla_ord"]))
    figpar = sess_plot_util.fig_init_linpla(figpar, kind="prog", n_sub=n_sess)

    prepost_str = sess_str_util.prepost_par_str(
            stimpar["pre"], stimpar["post"], str_type="print")

    if prog == "progsurp":
        substrs = ["surprise", "regular"]
    elif prog == "progreg":
        substrs = ["regular", "surprise"]
    else:
        raise ValueError("If prog is not False, it must be 'progreg' or \
            'progsurp'.")
    subtitle = "Progression of each {} {} vs preceeding {} sequence".format(
        pos, *substrs)

    diff_str = "_diff" if diff else ""
    grp_str_pr = ""
    if plot == "tog":
        key_str = f"prog{diff_str}_stats"
        dim_str = " across mice"
    elif plot in ["sep", "grped"]:
        dim_str = f" across {sess_str_util.datatype_dim_str(datatype)}"
        key_str = f"mouse_prog{diff_str}_stats"
        if plot == "grped":
            if datatype == "run":
                raise ValueError(f"Grouping ('grp') only across ROIs for ROI "
                    "datatype.")
            key_str = f"prog{diff_str}_stats_grped"
            grp_str_pr = " (grouped)"
    else:
        gen_util.accepted_values_error("plot", plot, ["tog", "sep", "grped"])
    
    title = (f"{subtitle}\n({prepost_str} seqs) for {stimstr_pr} "
        f"{statstr_pr}{dim_str}{grp_str_pr} (sess {sess_ns_str}{dendstr_pr})")

    fig, ax = plot_util.init_fig(n_plots, **figpar["init"])
    fig.suptitle(title, y=1.03, weight="bold")

    max_n = 0
    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        l_idx = get_linpla_idx(
            prog_info["linpla_ord"], line, pla, verbose=True, newline=(i==0))
        if l_idx is None:
            continue

        cols = [pla_cols[pl]]
        if not diff:
            if prog == "progreg":
                cols = cols + ["gray"]
            elif prog == "progsurp":
                cols = ["gray"] + cols

        if plot == "sep":
            n = plot_prog_per_mouse(
                ax[pl, li * n_sess:(li + 1) * n_sess], 
                prog_info[key_str][l_idx], sess_info[l_idx], sess_ns, 
                cols, use_lab=True, diff=diff)
            max_n = np.max([n, max_n]) # max number of sequences
        else:   
            for s in range(n_sess):
                for d in range(2 - diff):
                    data = np.asarray(prog_info[key_str][l_idx][s])
                    if diff:
                        alpha = 0.8
                        data_sp = data
                    else:
                        alpha = 0.4
                        data_sp = data[d]
                    col = cols[d]
                    sub_ax = ax[pl, s + li * n_sess]
                    n = plot_single_prog(sub_ax, data_sp, col, alpha_line=alpha)
                    max_n = np.max([n, max_n]) # max number of sequences

    if plot == "sep":
        sub_ax = ax.reshape(-1)[-1]

    # choose axis ticks and limits to fit all progs
    if max_n == 0:
        xticks = None
    elif max_n < 5:
        xticks = range(max_n + 2)
    else:
        base_itv = 5
        max_nticks = 6
        interv = base_itv * np.ceil(((max_n + 1)//base_itv)/(max_nticks - 1))
        xticks = range(
            0, int(np.ceil((max_n + 1)/interv) * interv + 1), int(interv))

    # Add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(ax, fluor=analyspar["fluor"], 
        area=False, datatype=datatype, lines=lines, planes=planes, 
        sess_ns=sess_ns, xticks=xticks, kind="prog")

    # reset x_lims
    sub_ax.set_xlim([0, max_n + 1])

    return fig


#############################################
def plot_prog(analyspar, sesspar, stimpar, extrapar, sess_info, 
              prog_info, figpar=None, savedir=None):
    """
    plot_prog(analyspar, sesspar, stimpar, extrapar, sess_info, 
              prog_info)

    From dictionaries, plots progression of difference between each surprise 
    and the preceeding regular sequence within sessions. 
    
    Returns general figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - stimpar (dict)  : dictionary with keys of BasePar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "g")
            ["datatype"] (str): datatype (e.g., "run", "roi")
            ["position"] (str): position plotted (e.g., "first", "second", etc.)
        - sess_info (list): list of dictionaries containing information from 
                            each session, structured as 
                            [progsurp, progreg]
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - prog_info (list): list of dictionaries containing prog surprise or 
                             regular sequence information, structured as 
                             [progsurp, progreg]
            ["prog_stats"] (list)           : surprise progression stats across 
                                              mice, structured as 
                                                  plane/line x session x surps 
                                                  x seq x stats
            ["prog_diff_stats"] (list)      : surprise difference progression 
                                              stats across mice, structured as 
                                                  plane/line x session x seq 
                                                  x stats
            ["mouse_prog_stats"] (list)     : surprise progression stats across 
                                              ROIs, structured as 
                                                 plane/line x mouse x session x
                                                 surps x seq x stats
            ["mouse_prog_diff_stats"] (list): surprise difference progression 
                                              stats across ROIs, structured as 
                                                 plane/line x mouse x session x
                                                 seq x stats

            ["linpla_ord"] (list)           : order list of planes/lines
        if datatype == "roi":
            ["prog_stats_grped"] (list)     : surprise progression stats across 
                                              ROIs (grouped across mice), 
                                              structured as 
                                                  plane/line x session x surps 
                                                  x seqs x stats
            ["prog_diff_stats_grped"] (list): surprise difference progression 
                                              stats across ROIs (grouped across 
                                              mice), structured as 
                                                  plane/line x session 
                                                  x seqs x stats

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
    
    Returns:
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): general name under which the figures are saved
    """

    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["bri_dir"], stimpar["bri_size"], stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    datatype = extrapar["datatype"]
    pos = extrapar["position"]

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    gen_savename = f"{datatype}_prog_{pos}_{sessstr}{dendstr}"
    part = f"prog_{pos}"
    add_idx = gen_savename.find(part) + len(part)

    stimdir = sess_str_util.get_stimdir(stimpar["stimtype"], stimpar["gabfr"])

    for diff in [True, False]:
        diff_str = "_diff" if diff else ""
        for l, prog in enumerate(["progsurp", "progreg"]):
            for plot in ["sep", "tog", "grped"]:
                if plot == "grped" and datatype == "run":
                    continue
                fig = plot_prog_acr_sess(analyspar, sesspar, stimpar, extrapar, 
                    sess_info[l], prog_info[l], figpar=figpar, prog=prog, 
                    plot=plot, diff=diff)

                if savedir is None:
                    savedir = os.path.join(
                        figpar["dirs"][datatype], 
                        figpar["dirs"]["acr_sess"], 
                        stimdir,
                        figpar["dirs"]["prog"])

                savename = (f"{gen_savename[:add_idx]}_{plot}{diff_str}"
                    f"{gen_savename[add_idx:]}")
                savename = savename.replace("prog", prog)

                fulldir = plot_util.savefig(
                    fig, savename, savedir, **figpar["save"])

    return fulldir, gen_savename


#############################################
def plot_position_acr_sess(analyspar, sesspar, stimpar, extrapar, sess_info, 
                           pos_info, figpar=None, prog="prog_surp", plot="tog"):
    """
    plot_position_acr_sess(analyspar, sesspar, stimpar, extrapar, sess_info, 
                            diff_info)

    From dictionaries, plots statistics across ROIs or mice of difference 
    between each surprise and the preceeding regular sequence across sessions. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                              parameters
            ["analysis"] (str): analysis type (e.g., "t")
            ["datatype"] (str): datatype (e.g., "run", "roi")
            ["position"] (str): position plotted (e.g., "first", "second", etc.)
        - pos_info (dict)       : dictionary with surprise position info
            ["pos_stats"] (list)      : surprise position stats across 
                                        mice, structured as 
                                           plane/line x session x stats
            ["mouse_pos_stats"] (list): surprise position stats across 
                                        sequences, structured as 
                                            plane/line x mouse x session 
                                               x stats
            ["linpla_ord"] (list)      : order list of planes/lines
        if extrapar["datatype"] == "roi":
            ["pos_stats_grped"] (list): surprise position stats across 
                                        sequences (ROIs grouped across mice), 
                                        structured as 
                                            plane/line x session x stats

        - sess_info (nested list): nested list of dictionaries for each 
                                   line/plane x mouse containing information 
                                   from each session, with None for missing 
                                   sessions
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
                
    Optional args:
        - figpar (dict)     : dictionary containing the following figure 
                              parameter dictionaries
                              default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - prog (str)        : if "progsurp" or "progreg", differences between
                              surprise and previous regular are plotted. If 
                              "progreg", v.v. 
                              default: "progsurp" 
        - plot (str)        : if "tog", average is taken across mice, otherwise, 
                              if "sep", each mouse is plotted separately
                              default: "tog"

    Returns:
        - fig (plt Figure) : pyplot figure
    """
 
    datatype = extrapar["datatype"]
    pos = extrapar["position"]
    error = analyspar["error"]
    if datatype == "run":
        if plot == "sep":
            error = "None"
        elif plot == "grped":
            raise ValueError("grped plot types only supported for ROI datatype.")
    
    dim_str = ""
    if plot == "tog":
        dim_str = "across mice"
    elif datatype == "roi" and plot in ["sep", "grped"]:
        dim_str = "across sequences"
    
    grp_str, grp_str_pr = "", ""
    if plot == "grped":
        grp_str = "_grped"
        grp_str_pr = " (ROIs grouped)"
        
    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["bri_dir"], stimpar["bri_size"],
        stimpar["gabk"], "print")
    statstr_pr = sess_str_util.stat_par_str(analyspar["stats"], error, "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")
    
    sess_ns = np.asarray(sesspar["sess_n"])
    if sesspar["sess_n"] in ["any", "all"]:
        sess_ns = np.arange(len(sess_info[0][0]["sess_ns"])) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.reshape(-1).tolist())

    [lines, planes, linpla_iter, 
     pla_cols, _, n_plots] = sess_plot_util.fig_linpla_pars( 
        n_grps=len(pos_info["linpla_ord"]))
    figpar = sess_plot_util.fig_init_linpla(figpar)

    prepost_str = sess_str_util.prepost_par_str(
            stimpar["pre"], stimpar["post"], str_type="print")

    if prog == "progsurp":
        substrs = ["surprise", "regular"]
    elif prog == "progreg":
        substrs = ["regular", "surprise"]
    else:
        raise ValueError("If prog is not False, it must be 'progreg' or \
            'progsurp'.")
    subtitle = "{} {} - preceeding {} sequences".format(
        pos, *substrs).capitalize()
    
    title = (f"{subtitle}\n({prepost_str} seqs) for {stimstr_pr} "
        f"{statstr_pr}\n{dim_str}{grp_str_pr} (sess {sess_ns_str}{dendstr_pr})")

    fig, ax = plot_util.init_fig(n_plots, **figpar["init"])
    fig.suptitle(title, y=1.03, weight="bold")

    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        l_idx = get_linpla_idx(
            pos_info["linpla_ord"], line, pla, verbose=True, newline=(i==0))
        if l_idx is None:
            continue

        _ = plot_area_diff_per_linpla(ax[pl, li], sess_ns, 
            pos_info[f"pos_stats{grp_str}"][l_idx], sess_info[l_idx], 
            pos_info["mouse_pos_stats"][l_idx], plot=plot, d="data", 
            col=pla_cols[pl], zero_line=True)


    # Add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(ax, fluor=analyspar["fluor"], 
        area=True, datatype=datatype, lines=lines, planes=planes, 
        xticks=sess_ns, xlab="Sessions", kind="reg")

    return fig


#############################################
def plot_position(analyspar, sesspar, stimpar, extrapar, sess_info, 
                  pos_info, figpar=None, savedir=None):
    """
    plot_position(analyspar, sesspar, stimpar, extrapar, sess_info, 
                  pos_info)

    From dictionaries, plots statistics across ROIs or mice of difference 
    between each surprise and the preceeding regular sequence across sessions.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "l")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - sess_info (list): list of dictionaries containing information from 
                            each session, structured as 
                            [surp-locked, reg-locked]
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - pos_info (list)       : list of dictionaries containing prog surprise 
                                  or regular sequence information, structured as 
                                  [progsurp, progreg]:
            ["pos_stats"] (list)      : surprise position stats across 
                                        mice, structured as 
                                           plane/line x session x stats
            ["mouse_pos_stats"] (list): surprise position stats across 
                                        sequences, structured as 
                                            plane/line x mouse x session 
                                               x stats
            ["linpla_ord"] (list)      : order list of planes/lines
        if extrapar["datatype"] == "roi":
            ["pos_stats_grped"] (list): surprise position stats across 
                                        sequences (ROIs grouped across mice), 
                                        structured as 
                                            plane/line x session x stats

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
    
    Returns:
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): name under which the figure is saved
    """
 
    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["bri_dir"], stimpar["bri_size"], stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    datatype = extrapar["datatype"]
    pos = extrapar["position"]

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    gen_savename = f"{datatype}_pos_{pos}_{sessstr}{dendstr}"
    part = f"pos_{pos}"
    add_idx = gen_savename.find(part) + len(part)

    stimdir = sess_str_util.get_stimdir(stimpar["stimtype"], stimpar["gabfr"])

    for l, prog in enumerate(["progsurp", "progreg"]):
        for plot in ["sep", "tog", "grped"]:
            if plot == "grped" and datatype == "run":
                continue
            fig = plot_position_acr_sess(analyspar, sesspar, stimpar, extrapar, 
                sess_info[l], pos_info[l], figpar=figpar, prog=prog, 
                plot=plot)

            if savedir is None:
                savedir = os.path.join(
                    figpar["dirs"][datatype], 
                    figpar["dirs"]["acr_sess"], 
                    stimdir,
                    figpar["dirs"]["prog"])

            savename = \
                f"{gen_savename[:add_idx]}_{plot}{gen_savename[add_idx:]}"
            savename = savename.replace("pos", prog.replace("prog", "pos"))

            fulldir = plot_util.savefig(
                fig, savename, savedir, **figpar["save"])

    return fulldir, gen_savename


#############################################
def plot_stim_idx_hist(sub_ax, percs, sess_info, n_bins=None, data=None, 
                       rand_data=None, orig_edges=None, p_val=0.05, tails="2", 
                       plot="items", col="r", density=False):
    """
    plot_stim_idx_hist(sub_ax, percs, sess_info)

    Plots histograms of stimulus indices.

    Required args:
        - sub_ax (plt subplot): subplot (can be None, if plot is "skip")
        - percs (array-like)  : item percentile bin counts (bins assumed to be 
                                equal size from 0-100)
        - sess_info (dict)    : information for each sessions session, with 
                                None for missing sessions
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")

    Optional args:
        - n_bins (int)          : number of histogram bins to plot (must be a 
                                  divisor of original number of bins and 
                                  compatible with separately binning the data 
                                  outside the CI). If None, the minimum accepted 
                                  binning value is selected.
                                  default: None
        - data (array-like)     : item index bin counts, required if 
                                  plot == "items"
                                  default: None
        - rand_data (array_like): randomly calculated item index bin counts, 
                                  required if plot == "items"
                                  default: None 
        - orig_edges (None)     : origin edges used to generate counts 
                                  [min, max], required if plot == "items"
                                  default: None
        - p_val (float)         : p-value for confidence interval marking
                                  default: 0.05
        - tails (str)           : tails for confidence interval marking
                                  default: "2"
        - plot (str)            : type of plot 
                                  ("items" for ROIs/running value indices,
                                  "percs" for index percentiles, 
                                  "skip" to skip plotting)
                                  default: "items"
        - col (str)             : color for indices (only significant indices, 
                                   if plot == "data")
                                  default: "r"
        - density (bool)        : if True, densities are plotted instead of 
                                  frequencies.
                                  default: False
    
    Returns:
        - perc_signif (float)   : percentage of significant items
        - perc_signif_lo (float): percentage of significant items (low)
        - perc_signif_hi (float): percentage of significant items (high)
    """

    if len(percs) == 0:
        return np.nan, np.nan, np.nan

    CI_edges = math_util.get_percentiles(CI=(1 - p_val), tails=tails)[0]

    # check possible n_bin values
    CI_wid = np.max([CI_edges[0], 100 - CI_edges[1]])

    min_n_bins = int(np.around(100/CI_wid))
    orig_n_bins = len(percs)
    poss_b_bins = list(range(min_n_bins, orig_n_bins + 1, min_n_bins))

    if len(poss_b_bins) == 0:
        raise ValueError("Original binning of the data is incompatible with "
            f"the p-value of {p_val} ({tails} tails). ")
    elif n_bins is None:
        n_bins = poss_b_bins[0]
    elif n_bins not in poss_b_bins:
        raise ValueError("Target binning value is incompatible with other "
            "parameters. Must be among {}.".join(
                [str(val) for val in poss_b_bins]))
    
    join_bins = int(orig_n_bins/n_bins)

    rebinned_percs = np.sum(np.asarray(percs).reshape(-1, join_bins), axis=1)
    perc_bin_edges = np.linspace(0, 100, n_bins + 1).tolist()

    # count signif ROIs
    n_items = int(np.sum(rebinned_percs))
    CI_bin_edges = [perc_bin_edges.index(edge) for edge in CI_edges]
    n_sig_items_lo = int(np.sum(rebinned_percs[: CI_bin_edges[0]]))
    n_sig_items_hi = int(np.sum(rebinned_percs[CI_bin_edges[1] :]))

    if n_items == 0:
        perc_signif_lo, perc_signif_hi = np.nan, np.nan
        perc_signif = np.sum([perc_signif_lo, perc_signif_hi])
        perc_signif_pr = "NaN"
    else:
        perc_signif_lo = n_sig_items_lo * 100.0 / n_items
        perc_signif_hi = n_sig_items_hi * 100.0 / n_items
        perc_signif = np.sum([perc_signif_lo, perc_signif_hi])
        perc_signif_pr = np.around(perc_signif, 2)

    if plot != "skip":
        perc_label = (f"{perc_signif_pr}% sig\n"
            f"({n_sig_items_lo}-/{n_sig_items_hi}+ of {n_items})")
        sub_ax.plot([], label=perc_label, color=col)
        sub_ax.legend(fontsize="large")

    if plot == "items":
        if data is None:
            raise ValueError("Must provide 'data' if plot is 'items'.")
        if len(percs) != len(data):
            raise ValueError("Expected as many bins for percentiles and data.")
        if orig_edges is None:
            raise ValueError("Must provide 'orig_edges' if plot is 'items'.")
        if len(data) != len(percs):
            raise ValueError("'data' must be same length as 'percs'.")
        if len(orig_edges) != 2:
            raise ValueError("'orig_edges' must be of length 2 [min, max].")

        rebinned_data = np.sum(np.asarray(data).reshape(-1, join_bins), axis=1)
        bin_edges = np.linspace(orig_edges[0], orig_edges[1], n_bins + 1)

        colors = [col]
        plot_data = [rebinned_data]
        alphas = [0.6]

        # add random data
        if rand_data is not None:
            if len(rand_data) != len(percs):
                raise ValueError("If provided, 'rand_data' must be same "
                    "length as 'percs'.")
            # rebin and scale random data
            rebinned_rand_data = \
                np.sum(np.asarray(rand_data).reshape(-1, join_bins), axis=1)
            plot_data.insert(0, rebinned_rand_data)
            colors.insert(0, "gray")
            alphas.insert(0, 0.45)            
        
        for sub, c, a in zip(plot_data, colors, alphas):
            sub_ax.hist(
                bin_edges[:-1], bin_edges, weights=sub, color=c, alpha=a, 
                density=density)
    
    elif plot == "percs":
        for c, CI_edge in enumerate(CI_edges):
            # sub_ax.axvline(
            #     x=CI_edge, ls=VDASH, c=DARKRED, lw=80./n_bins, alpha=0.6)
            if c == 0:
                sub_ax.axvspan(0, CI_edge, color=DARKRED, alpha=0.15, 
                    lw=80./n_bins)
            elif c == 1:
                sub_ax.axvspan(CI_edge, 100, color=DARKRED, alpha=0.15, 
                    lw=80./n_bins)

        sub_ax.hist(perc_bin_edges[:-1], perc_bin_edges, 
            weights=rebinned_percs, color=col, alpha=0.7, density=density)
    
    elif plot != "skip":
        gen_util.accepted_values_error("plot", plot, ["items", "percs", "skip"])

    return perc_signif, perc_signif_lo, perc_signif_hi


#############################################
def plot_stim_idx_acr_sess(analyspar, sesspar, stimpar, permpar, extrapar, 
                           idx_info, sess_info, figpar=None, plot="tog", 
                           feature="surp", match_oris=False):
    """
    plot_stim_idx_acr_sess(analyspar, sesspar, stimpar, extrapar, sess_info, 
                           idx_info, sess_info)

    From dictionaries, stimulus feature indices for each ROI or single running 
    value, across sessions. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - permpar (dict)  : dictionary with keys of PermPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                              parameters
            ["analysis"] (str): analysis type (e.g., "t")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - idx_info (dict) : feature index information:
            ["item_idxs"] (list) : feature index bin counts for each 
                                   ROI or running value, grouped across mice, 
                                   structured as 
                                       plane/line x session x bin
            ["item_percs"] (list): feature percentile bin counts for each 
                                   ROI or running value, grouped across mice, 
                                   structured as 
                                       plane/line x session x bin
            ["rand_idxs"] (list) : random feature index bin counts for each 
                                   ROI or running value, grouped across mice, 
                                   structured as 
                                       plane/line x session x bin
            ["perc_pos"] (list)  : for each session number, percent ROIs with 
                                   positive indices, grouped across mice, 
                                   structured as 
                                       plane/line x session
            ["bin_edges"] (list) : bin edges for indices, structured as 
                                       plane/line x session x [min, max]
            ["linpla_ord"] (list): order list of planes/lines
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/plane x session containing information 
                                   from each mouse, with None for missing 
                                   sessions
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
                
    Optional args:
        - figpar (dict)     : dictionary containing the following figure 
                              parameter dictionaries
                              default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - plot (str)        : type of plot 
                              ("items" for ROIs/running value indices
                              "percs" for index percentiles)
                              default: "items"
        - feature (str)     : feature on which index is based, 
                              e.g. "surp" or "dir"
                              default: "surp"
        - match_oris (bool) : if True, only Gabor stimulus orientations 
                              shared by D and U frames are included 
                              ("surp" split only)
                              default: False

    Returns:
        - fig (plt Figure)    : pyplot figure
        - perc_sig_info (dict): dictionary containing information on percentage 
                                of significant items
            ["linpla_ord"] (list)  : order list of planes/lines
            ["perc_sig"] (3D array): percentage significant items, 
                                     structured as 
                                        plane/line x session x 
                                        [overall, low, high]
    """
 
    datatype = extrapar["datatype"]
    
    dim_str = sess_str_util.datatype_dim_str(datatype)
    if plot == "items":
        plot_str = "indices"
        n_bins = 40
    elif plot == "percs":
        plot_str = "index percentiles"
        n_bins = 40
    else:
        gen_util.accepted_values_error("plot", plot, ["items", "percs"])

    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["bri_dir"], stimpar["bri_size"],
        stimpar["gabk"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")
    
    sess_ns = np.asarray(sesspar["sess_n"])
    if sesspar["sess_n"] in ["any", "all"]:
        sess_ns = np.asarray(range(len(sess_info[0]))) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.reshape(-1).tolist())
    n_sess = len(sess_ns)

    [lines, planes, linpla_iter, 
     pla_cols, _, n_plots] = sess_plot_util.fig_linpla_pars(
        n_sess = len(sess_ns), n_grps=len(idx_info["linpla_ord"]))
    figpar = sess_plot_util.fig_init_linpla(figpar, kind="idx", n_sub=n_sess, 
        sharex=(plot == "percs"))

    prepost_str = sess_str_util.prepost_par_str(
            stimpar["pre"], stimpar["post"], str_type="print")

    if "dir" in feature:
        if feature == "dir_reg":
            feat_sub = "regular "
        elif feature == "dir_surp":
            feat_sub = "surprise "
        elif feature != "dir":
            raise ValueError("If 'dir' in 'feature', must be "
                "among 'dir_reg', 'dir_surp' or 'dir'.")
        feature_str = f"{feat_sub}direction (left v right)\n"
    else:
        feature_str = "surprise"
    
    match_str = ""
    if match_oris:
        if stimpar["stimtype"] != "gabors":
            raise ValueError("'match_oris' only applies to Gabors data.")
        match_str = " (matched oris)"

    subtitle = (f"{dim_str} {feature_str}{match_str} {plot_str}")
    
    title = (f"{subtitle} ({prepost_str} seqs)\nfor {stimstr_pr} "
        f"(sess {sess_ns_str}{dendstr_pr})")

    fig, ax = plot_util.init_fig(n_plots, **figpar["init"])
    fig.suptitle(title, y=1, weight="bold")

    perc_sig_info = {
        "linpla_ord": idx_info["linpla_ord"],
        "perc_sig"  : np.empty([len(linpla_iter), n_sess, 3]) * np.nan
    }
    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        l_idx = get_linpla_idx(
            idx_info["linpla_ord"], line, pla, verbose=True, newline=(i==0))
        if l_idx is None:
            continue
        for s in range(n_sess):
            sub_ax = ax[s + pl * n_sess, li]
            perc_sig_info["perc_sig"][l_idx, s] = plot_stim_idx_hist(sub_ax, 
                idx_info["item_percs"][l_idx][s], sess_info[l_idx][s], 
                data=idx_info["item_idxs"][l_idx][s], 
                rand_data=idx_info["rand_idxs"][l_idx][s],
                orig_edges=idx_info["bin_edges"][l_idx][s], 
                p_val=permpar["p_val"], tails=permpar["tails"], 
                col=pla_cols[pl], n_bins=n_bins, plot=plot)

    if plot == "percs":
        sub_ax = ax[-1, 0]
        nticks = 5
        xticks = [int(np.around(x, 0)) for x in np.linspace(0, 100, nticks)]
        for sub_ax in ax[-1]:
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks, weight="bold")
    
    elif plot == "items":
        nticks = 3
        plot_util.set_interm_ticks(
            ax, nticks, dim="x", weight="bold", share=False)
        for sub_ax in ax.reshape(-1):
            xticks = sub_ax.get_xticks()
            diff = np.diff(xticks)[0]
            if diff != 0:
                n_dig = - np.floor(np.log10(np.absolute(diff))).astype(int) + 1
                xticks = [np.around(v, n_dig) for v in xticks]
                if n_dig <= 0:
                    xticks = [int(v) for v in xticks]
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks, weight="bold")

    # Add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(ax, datatype=datatype, lines=lines, 
        planes=planes, ylab=f"N {dim_str}", xticks=None, sess_ns=sess_ns, 
        kind="idx")

    return fig, perc_sig_info


#############################################
def plot_perc_sig_acr_sess(analyspar, sesspar, stimpar, permpar, extrapar, 
                           perc_sig_info, sess_info, figpar=None, plot="items", 
                           feature="surp", match_oris=False):
    """
    plot_perc_sig_acr_sess(analyspar, sesspar, stimpar, permpar, extrapar, 
                           perc_sig_info, sess_info)

    From dictionaries, percentage of significant stimulus feature indices 
    across sessions. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict)    : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)      : dictionary with keys of SessPar namedtuple
        - stimpar (dict)      : dictionary with keys of StimPar namedtuple
        - permpar (dict)      : dictionary with keys of PermPar namedtuple
        - extrapar (dict)     : dictionary containing additional analysis 
                              parameters
            ["analysis"] (str): analysis type (e.g., "t")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - perc_sig_info (dict): dictionary containing information on percentage 
                                of significant items
            ["linpla_ord"] (list)  : order list of planes/lines
            ["perc_sig"] (3D array): percentage significant items, 
                                     structured as 
                                        plane/line x session x 
                                        [overall, low, high]
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/plane x session containing information 
                                   from each mouse, with None for missing 
                                   sessions
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
                
    Optional args:
        - figpar (dict)     : dictionary containing the following figure 
                              parameter dictionaries
                              default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - feature (str)     : feature on which index is based, 
                              e.g. "surp" or "dir"
                              default: "surp"
        - match_oris (bool) : if True, only Gabor stimulus orientations 
                              shared by D and U frames are included 
                              ("surp" split only)
                              default: False

    Returns:
        - fig (plt Figure)   : pyplot figure
        - perc_sig (3D array): percentage significant items, structured as 
                               plane/line x session x [overall, low, high]
    """
 
    datatype = extrapar["datatype"]
    
    dim_str = sess_str_util.datatype_dim_str(datatype)

    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["bri_dir"], stimpar["bri_size"],
        stimpar["gabk"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")
    
    sess_ns = np.asarray(sesspar["sess_n"])
    if sesspar["sess_n"] in ["any", "all"]:
        sess_ns = np.asarray(range(len(sess_info[0]))) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.reshape(-1).tolist())

    [lines, planes, linpla_iter, _, 
        pla_col_names, n_plots] = sess_plot_util.fig_linpla_pars(
            n_grps=len(perc_sig_info["linpla_ord"]))
    figpar = sess_plot_util.fig_init_linpla(
        figpar, kind="reg", sharex=True, sharey=True)

    prepost_str = sess_str_util.prepost_par_str(
            stimpar["pre"], stimpar["post"], str_type="print")

    if "dir" in feature:
        if feature == "dir_reg":
            feat_sub = "regular "
        elif feature == "dir_surp":
            feat_sub = "surprise "
        elif feature != "dir":
            raise ValueError("If 'dir' in 'feature', must be "
                "among 'dir_reg', 'dir_surp' or 'dir'.")
        feature_str = f"{feat_sub}direction (left v right)"
    else:
        feature_str = "surprise"

    match_str = ""
    if match_oris:
        if stimpar["stimtype"] != "gabors":
            raise ValueError("'match_oris' only applies to Gabors data.")
        match_str = " (matched oris)"

    subtitle = (f"Percentage of significant {dim_str}\nfor {feature_str}"
        f"{match_str}")
    
    title = (f"{subtitle} ({prepost_str} seqs)\nfor {stimstr_pr} "
        f"(sess {sess_ns_str}{dendstr_pr})")

    fig, ax = plot_util.init_fig(n_plots, **figpar["init"])
    fig.suptitle(title, y=1.05, weight="bold")

    # perc_sig = np.empty([len(linpla_iter), n_sess, 3]) * np.nan
    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        l_idx = get_linpla_idx(
            perc_sig_info["linpla_ord"], line, pla, verbose=True, 
            newline=(i==0))
        if l_idx is None:
            continue
        cols = plot_util.get_color_range(3, pla_col_names[pl])
        perc_sigs = ["all", "low", "high"]
        cols = [cols[i] for i in [1, 0, 2]] # reorder
        alphas = [0.3, 0.6, 0.6]
        for p, lab in enumerate(perc_sigs):
            sub_ax = ax[pl, li]
            keep_idx = np.where(np.isfinite(
                perc_sig_info["perc_sig"][l_idx, :, p]))[0]
            if len(keep_idx) == 0:
                continue
            plot_util.plot_errorbars(
                sub_ax, perc_sig_info["perc_sig"][l_idx, keep_idx, p], 
                x=sess_ns[keep_idx], color=cols[p], label=lab, alpha=alphas[p])

    # Add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(ax, datatype=datatype, lines=lines, 
        planes=planes, ylab=f"% signif.", xticks=sess_ns, sess_ns=sess_ns, 
        kind="reg")

    # max percent, shared
    ymax = np.max([int(np.ceil(ax[1, 0].get_ylim()[1]/10.) * 10), 10])
    yticks = np.linspace(0, ymax, 5)
    plot_util.set_ticks(ax[1, 0], "y", 0, ymax, 5, pad_p=0.1)
    ax[1, 0].set_yticklabels(
        [int(v) if i % 2 == 0 else "" for i, v in enumerate(yticks)], 
        fontdict={"weight": "bold"})

    return fig


#############################################
def plot_surp_idx(analyspar, sesspar, stimpar, basepar, permpar, idxpar, 
                  extrapar, sess_info, surpidx_info, figpar=None, savedir=None):
    """
    plot_surp_idx(analyspar, sesspar, stimpar, basepar, permpar, idxpar, 
                  extrapar, sess_info, surpidx_info)

    From dictionaries, plots surprise indices for all ROIs or running values.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - basepar (dict)  : dictionary with keys of BasePar namedtuple
        - permpar (dict)  : dictionary with keys of PermPar namedtuple
        - idxpar (dict)   : dictionary with keys of IdxPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "l")
            ["datatype"] (str): datatype (e.g., "run", "roi")
            ["seed"] (int)    : seed
        - sess_info (dict): dictionaries containing information from each 
                            session:
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - surpidx_info (dict): containing surprise index information:
            ["item_idxs"] (list) : surprise index bin counts for each 
                                   ROI or running value, grouped across mice, 
                                   structured as 
                                       plane/line x session x bin
            ["item_percs"] (list): surprise percentile bin counts for each 
                                   ROI or running value, grouped across mice, 
                                   structured as 
                                       plane/line x session x bin
            ["rand_idxs"] (list) : random surprise index bin counts for each 
                                   ROI or running value, grouped across mice, 
                                   structured as 
                                       plane/line x session x bin
            ["perc_pos"] (list)  : for each session number, percent ROIs with 
                                   positive indices, grouped across mice, 
                                   structured as 
                                       plane/line x session
            ["bin_edges"] (list) : bin edges for indices, structured as 
                                       plane/line x session x edge
            ["linpla_ord"] (list): order list of planes/lines

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
    
    Returns:
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): name under which the figure is saved
    """
 
    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["bri_dir"], stimpar["bri_size"], stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    datatype = extrapar["datatype"]

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    gen_savename = f"{datatype}_surpidx_{sessstr}{dendstr}"

    stimdir = sess_str_util.get_stimdir(stimpar["stimtype"], stimpar["gabfr"])

    for plot in ["items", "percs", "perc_sig"]:
        if plot != "perc_sig":
            fig, perc_sig_info = plot_stim_idx_acr_sess(
                analyspar, sesspar, stimpar, permpar, extrapar, surpidx_info, 
                sess_info, figpar=figpar, plot=plot, feature="surp")
        else:
            fig = plot_perc_sig_acr_sess(
                analyspar, sesspar, stimpar, permpar, extrapar, 
                perc_sig_info, sess_info, figpar=figpar, feature="surp")

        if savedir is None:
            base_str = sess_str_util.base_par_str(basepar["baseline"])[1:]
            savedir = os.path.join(
                figpar["dirs"][datatype], 
                figpar["dirs"]["acr_sess"], 
                stimdir,
                base_str,
                figpar["dirs"]["surp_idx"])
            if idxpar["feature"] != "bysurp":
                savedir = os.path.join(savedir, idxpar["feature"])

        savename = gen_savename
        if plot == "percs":
            savename = gen_savename.replace("idx", "idx_percs")
        elif plot == "perc_sig":
            savename = gen_savename.replace("idx", "_perc_sig")

        fulldir = plot_util.savefig(
            fig, savename, savedir, **figpar["save"])

    return fulldir, gen_savename


#############################################
def plot_surp_idx_match_oris(analyspar, sesspar, stimpar, basepar, permpar, 
                             idxpar, extrapar, sess_info, surpidx_info, 
                             figpar=None, savedir=None):
    """
    plot_surp_idx_match_oris(analyspar, sesspar, stimpar, basepar, permpar, 
                             idxpar, extrapar, sess_info, surpidx_info)

    From dictionaries, plots surprise indices with matching orientations for 
    all ROIs or running values.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - basepar (dict)  : dictionary with keys of BasePar namedtuple
        - permpar (dict)  : dictionary with keys of PermPar namedtuple
        - idxpar (dict)   : dictionary with keys of IdxPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "l")
            ["datatype"] (str): datatype (e.g., "run", "roi")
            ["seed"] (int)    : seed
        - sess_info (dict): dictionaries containing information from each 
                            session:
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - surpidx_info (dict): containing surprise index information:
            ["item_idxs"] (list) : surprise index bin counts for each 
                                   ROI or running value, grouped across mice, 
                                   structured as 
                                       plane/line x session x bin
            ["item_percs"] (list): surprise percentile bin counts for each 
                                   ROI or running value, grouped across mice, 
                                   structured as 
                                       plane/line x session x bin
            ["rand_idxs"] (list) : random surprise index bin counts for each 
                                   ROI or running value, grouped across mice, 
                                   structured as 
                                       plane/line x session x bin
            ["perc_pos"] (list)  : for each session number, percent ROIs with 
                                   positive indices, grouped across mice, 
                                   structured as 
                                       plane/line x session
            ["bin_edges"] (list) : bin edges for indices, structured as 
                                       plane/line x session x edge
            ["linpla_ord"] (list): order list of planes/lines

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
    
    Returns:
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): name under which the figure is saved
    """
 
    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["bri_dir"], stimpar["bri_size"], stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    datatype = extrapar["datatype"]

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    gen_savename = f"{datatype}_surpidx_match_oris_{sessstr}{dendstr}"

    stimdir = sess_str_util.get_stimdir(stimpar["stimtype"], stimpar["gabfr"])

    for plot in ["items", "percs", "perc_sig"]:
        if plot != "perc_sig":
            fig, perc_sig_info = plot_stim_idx_acr_sess(
                analyspar, sesspar, stimpar, permpar, extrapar, surpidx_info, 
                sess_info, figpar=figpar, plot=plot, feature="surp", 
                match_oris=True)
        else:
            fig = plot_perc_sig_acr_sess(
                analyspar, sesspar, stimpar, permpar, extrapar, 
                perc_sig_info, sess_info, figpar=figpar, feature="surp", 
                match_oris=True)

        if savedir is None:
            base_str = sess_str_util.base_par_str(basepar["baseline"])[1:]
            savedir = os.path.join(
                figpar["dirs"][datatype], 
                figpar["dirs"]["acr_sess"], 
                stimdir,
                base_str,
                figpar["dirs"]["surp_idx"])
            if idxpar["feature"] != "bysurp":
                savedir = os.path.join(savedir, idxpar["feature"])

        savename = gen_savename
        if plot == "percs":
            savename = gen_savename.replace("idx", "idx_percs")
        elif plot == "perc_sig":
            savename = gen_savename.replace("idx", "_perc_sig")

        fulldir = plot_util.savefig(
            fig, savename, savedir, **figpar["save"])

    return fulldir, gen_savename


#############################################
def plot_direction_idx(analyspar, sesspar, stimpar, basepar, permpar, idxpar, 
                       extrapar, sess_info, diridx_info, figpar=None, 
                       savedir=None):
    """
    plot_direction_idx(analyspar, sesspar, stimpar, basepar, permpar, idxpar, 
                       extrapar, sess_info, diridx_info)

    From dictionaries, plots direction indices for all ROIs or running values.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - basepar (dict)  : dictionary with keys of BasePar namedtuple
        - permpar (dict)  : dictionary with keys of PermPar namedtuple
        - idxpar (dict)   : dictionary with keys of IdxPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "l")
            ["datatype"] (str): datatype (e.g., "run", "roi")
            ["seed"] (int)    : seed
        - sess_info (dict): dictionaries containing information from each 
                            session:
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - diridx_info (dict) : list of dictionaries containing direction index 
                               information, ordered as ["dir_reg", "dir_surp"]:
            ["item_idxs"] (list) : direction index bin counts for each 
                                   ROI or running value, grouped across mice, 
                                   structured as 
                                       plane/line x session x bin
            ["item_percs"] (list): direction percentile bin counts for each 
                                   ROI or running value, grouped across mice, 
                                   structured as 
                                       plane/line x session x bin
            ["rand_idxs"] (list) : random direction index bin counts for each 
                                   ROI or running value, grouped across mice, 
                                   structured as 
                                       plane/line x session x bin
            ["perc_pos"] (list)  : for each session number, percent ROIs with 
                                   positive indices, grouped across mice, 
                                   structured as 
                                       plane/line x session
            ["bin_edges"] (list) : bin edges for indices, structured as 
                                       plane/line x session x edge
            ["linpla_ord"] (list): order list of planes/lines

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
    
    Returns:
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): name under which the figure is saved
    """
 
    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["bri_dir"], stimpar["bri_size"], stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    datatype = extrapar["datatype"]

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    gen_savename = f"{datatype}_diridx_{sessstr}{dendstr}"

    stimdir = sess_str_util.get_stimdir(stimpar["stimtype"], stimpar["gabfr"])

    for d, direc in enumerate(["dir_reg", "dir_surp"]):
        for plot in ["items", "percs", "perc_sig"]:
            if plot != "perc_sig":
                fig, perc_sig_info = plot_stim_idx_acr_sess(
                    analyspar, sesspar, stimpar, permpar, extrapar, 
                    diridx_info[d], sess_info, figpar=figpar, plot=plot, 
                    feature=direc)
            else:
                fig = plot_perc_sig_acr_sess(
                    analyspar, sesspar, stimpar, permpar, extrapar, 
                    perc_sig_info, sess_info, figpar=figpar, feature=direc)

            if savedir is None:
                base_str = sess_str_util.base_par_str(basepar["baseline"])[1:]
                savedir = os.path.join(
                    figpar["dirs"][datatype], 
                    figpar["dirs"]["acr_sess"], 
                    stimdir,
                    base_str,
                    figpar["dirs"]["dir_idx"])

            savename = gen_savename
            if plot == "percs":
                savename = gen_savename.replace("idx", "idx_percs")
            elif plot == "perc_sig":
                savename = gen_savename.replace("idx", "_perc_sig")
            savename = savename.replace("dir", direc)

            fulldir = plot_util.savefig(
                fig, savename, savedir, **figpar["save"])

    return fulldir, gen_savename


#############################################
def plot_surp_idx_cm(sub_ax, item_idxs, col="blue"):
    """
    plot_surp_idx_cm(sub_ax, item_idxs)

    Plot colormap for a set of stimulus parameters

    Required args:
        - sub_ax (plt subplot)  : subplot
        - item_idxs (array-like): item indices for each stimulus parameter

    Optional args:
        - col (str): color for indices
                     default: "blue"
    """

    cmap = cm.coolwarm

    # sort ROIs together
    item_idxs = np.asarray(copy.deepcopy(item_idxs))
    nstims, nrois = item_idxs.shape
    order = None
    for s, stimpar_idxs in enumerate(item_idxs):
        if order is None and np.isfinite(item_idxs).any():
           order = np.argsort(stimpar_idxs)
        if order is not None:
            item_idxs[s] = item_idxs[s][order]

    im = plot_util.plot_colormap(
        sub_ax, item_idxs, cmap=cmap, n_xticks=nstims, yticks_ev=nrois//10)

    return im


#############################################
def plot_surp_idx_cms_acr_sess(analyspar, sesspar, stimpar, extrapar, 
                               surpidx_info, sess_info, figpar=None):
    """
    plot_surp_idx_cms_acr_sess(analyspar, sesspar, stimpar, extrapar, 
                               surpidx_info, sess_info)

    From dictionaries, surprise indices for each ROI or single running 
    value, for each stimulus parameter, across sessions. 
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - permpar (dict)  : dictionary with keys of PermPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                              parameters
            ["analysis"] (str): analysis type (e.g., "t")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - idx_info (dict) : feature index information:
            ["item_idxs"] (list) : feature index bin counts for each 
                                   ROI or running value, grouped across mice, 
                                   structured as 
                                       plane/line x session x 
                                        stimulus parameter x item
            ["linpla_ord"] (list): order list of planes/lines
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/plane x session containing information 
                                   from each mouse, with None for missing 
                                   sessions
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
                
    Optional args:
        - figpar (dict)     : dictionary containing the following figure 
                              parameter dictionaries
                              default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters

    Returns:
        - fig (plt Figure)    : pyplot figure
    """
 
    datatype = extrapar["datatype"]
    
    dim_str = sess_str_util.datatype_dim_str(datatype)

    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["bri_dir"], stimpar["bri_size"],
        stimpar["gabk"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")
    
    sess_ns = np.asarray(sesspar["sess_n"])
    if sesspar["sess_n"] in ["any", "all"]:
        sess_ns = np.asarray(range(len(sess_info[0]))) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.reshape(-1).tolist())
    n_sess = len(sess_ns)

    if stimpar["stimtype"] == "gabors":
        if (stimpar["gab_ori"] not in ["all", "any"] and 
            len(stimpar["gab_ori"]) < 2):
            raise ValueError("Must include at least 2 gabor orientations "
                "for colormap surprise index analysis.")
        stim_vals = [0, 45, 90, 135]
        stim_name = "Gabor oris"
    elif stimpar["stimtype"] == "bricks":
        if (stimpar["bri_dir"] not in ["all", "any", "both"] and 
            len(stimpar["bri_dir"]) < 2):
            raise ValueError("Must include at least 2 brick directions "
                "for colormap surprise index analysis.")
        stim_vals = [sess_gen_util.get_bri_screen_mouse_direc(val) 
            for val in ["left", "right"]]
        stim_name = "Brick dirs"

    [lines, planes, linpla_iter, 
     pla_cols, _, n_plots] = sess_plot_util.fig_linpla_pars(
        n_sess = len(sess_ns), n_grps=len(surpidx_info["linpla_ord"]))
    figpar = sess_plot_util.fig_init_linpla(figpar, kind="idx", n_sub=n_sess, 
        sharex=False)

    prepost_str = sess_str_util.prepost_par_str(
            stimpar["pre"], stimpar["post"], str_type="print")

    feature_str = "surprise"

    subtitle = (f"{dim_str} {feature_str}")
    
    title = (f"{subtitle} ({prepost_str} seqs)\nfor {stimstr_pr} "
        f"(sess {sess_ns_str}{dendstr_pr})")

    fig, ax = plot_util.init_fig(n_plots, **figpar["init"])
    fig.suptitle(title, y=1, weight="bold")

    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        l_idx = get_linpla_idx(
            surpidx_info["linpla_ord"], line, pla, verbose=True, newline=(i==0))
        if l_idx is None:
            continue
        for s in range(n_sess):
            sub_ax = ax[s + pl * n_sess, li]
            im = plot_surp_idx_cm(
                sub_ax, surpidx_info["item_idxs"][l_idx][s], col=pla_cols[pl])
    
    xticks = np.linspace(1, len(stim_vals), len(stim_vals))
    for sub_ax in ax.reshape(-1):
        sub_ax.set_xticks(xticks)
        sub_ax.tick_params(axis="x", which="both", bottom=False) 
        sub_ax.set_xticklabels(stim_vals, weight="bold")

    # Add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(ax, datatype=datatype, lines=lines, 
        planes=planes, ylab=None, xticks=None, sess_ns=sess_ns, 
        kind="idx")


    return fig


#############################################
def plot_surp_idx_cms(analyspar, sesspar, stimpar, basepar, idxpar, extrapar, 
                      sess_info, surpidx_info, figpar=None, savedir=None):
    """
    plot_surp_idx_cms(analyspar, sesspar, stimpar, idxpar, extrapar, 
                      sess_info, surpidx_info)

    From dictionaries, plots surprise index colormaps for all ROIs or 
    running values.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - basepar (dict)  : dictionary with keys of BasePar namedtuple
        - idxpar (dict)   : dictionary with keys of IdxPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "l")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - sess_info (dict): dictionaries containing information from each 
                            session:
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - surpidx_info (dict): containing surprise index information:
            ["item_idxs"] (list) : surprise index for each ROI or running 
                                   value, grouped across mice, 
                                   structured as 
                                        plane/line x session x 
                                        stimulus parameter x item
            ["linpla_ord"] (list): order list of planes/lines

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
    
    Returns:
        - fulldir (str) : final name of the directory in which the figure 
                          is saved (may differ from input savedir, if 
                          datetime subfolder is added.)
        - savename (str): name under which the figure is saved
    """
 
    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["bri_dir"], stimpar["bri_size"], stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    datatype = extrapar["datatype"]

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    savename = f"{datatype}_surpidx_cm_{sessstr}{dendstr}"

    stimdir = sess_str_util.get_stimdir(stimpar["stimtype"], stimpar["gabfr"])

    fig = plot_surp_idx_cms_acr_sess(
            analyspar, sesspar, stimpar, extrapar, surpidx_info, 
            sess_info, figpar=figpar)

    if savedir is None:
        base_str = sess_str_util.base_par_str(basepar["baseline"])[1:]
        savedir = os.path.join(
            figpar["dirs"][datatype], 
            figpar["dirs"]["acr_sess"], 
            stimdir,
            base_str,
            figpar["dirs"]["surp_idx"],
            figpar["dirs"]["colormaps"])
        if idxpar["feature"] != "bysurp":
            savedir = os.path.join(savedir, idxpar["feature"])

    fulldir = plot_util.savefig(
        fig, savename, savedir, **figpar["save"])

    return fulldir, savename


#############################################
def plot_lat_clouds(sub_ax, sess_ns, lat_data, sess_info, datatype="roi", 
                    col="blue", alpha=0.2):
    """
    plot_lat_clouds(sub_ax, sess_ns, lat_data)

    Plots clouds of latency data in different shades for each mouse.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - sess_ns (array-like)     : session numbers
        - lat_data (2D array)      : latency values for each ROI, structured as
                                         session x mouse
        - sess_info (list)         : list of dictionaries for each mouse 
                                     containing information from each session, 
                                     with None for missing sessions
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")

    Optional args:
        - prev_maxes (1D array): array of previous max values for each session
        - datatype (str): datatype (e.g., "run", "roi")
                          default: "roi"
        - col (str)     : name of colour to base shades on
                          default: "blue"
        - alpha (float) : pyplot transparency parameter
                          default: 0.3
    
    Returns:
        - maxes (1D array): max values from data clouds for each session
    """

    n_mice = [len(s_vals) for s_vals in lat_data if s_vals is not None]

    if len(set(n_mice)) != 1:
        raise ValueError("Should be the same number of mice.")
    n_mice = n_mice[0]
    m_cols = plot_util.get_color_range(n_mice, col=col)

    labels = ["" for _ in range(n_mice)] # to collect labels
    clouds = [None for _ in range(n_mice)] # to collect artists
    maxes = np.full(len(sess_ns), np.nan)
    for s, sess_n in enumerate(sess_ns):
        for m in range(n_mice):
            if lat_data[s][m] is None:
                nrois = 0
            else:
                nrois = len(lat_data[s][m])
            if len(labels[m]) == 0:
                add = "M#"
                if datatype == "roi":
                    add = f"{add} ({nrois}"
            else:
                if datatype == "roi":
                    add = f"/{nrois}"
                else:
                    add = ""
            labels[m] = f"{labels[m]}{add}"
            if nrois == 0:
                continue
            elif "#" in labels[m]:
                labels[m] = labels[m].replace(
                    "#", str(sess_info[s]["mouse_ns"][m]))
            div_fact = np.max([1, len(lat_data[s][m])//50])
            alpha_spec = alpha/div_fact
            clouds[m] = plot_util.plot_data_cloud(
                sub_ax, sess_n, lat_data[s][m], 0.15, label=None, 
                color=m_cols[m], alpha=alpha_spec, zorder=-11)
            maxes[s] = np.nanmax([maxes[s], np.nanmax(lat_data[s][m])])
    if datatype == "roi":
        labels = [f"{label} ROIs)" for label in labels]

    # remove Nones
    clouds_filtered, labels_filtered = [], []
    for cl, lab in zip(clouds, labels):
        if cl is not None:
            clouds_filtered.append(cl)
            labels_filtered.append(lab)
            
    if len(clouds_filtered) != 0:
        sub_ax.legend(clouds_filtered, labels_filtered, fontsize="small")

    return maxes


#############################################
def plot_surp_latency(analyspar, sesspar, stimpar, permpar, latpar, extrapar, 
                      sess_info, lat_data, figpar=None, savedir=None):
    """
    plot_surp_latency(analyspar, sesspar, stimpar, permpar, latpar, extrapar, 
                      sess_info, lat_data)

    From dictionaries, plots surprise latency across mice, as well as for all 
    ROIs.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict)       : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)         : dictionary with keys of SessPar namedtuple
        - stimpar (dict)         : dictionary with keys of StimPar namedtuple
        - permpar (dict)         : dictionary with keys of PermPar namedtuple
        - latpar (LatPar)        : dictionary with keys of LatPar namedtuple
        - extrapar (dict)        : dictionary containing additional analysis 
                                   parameters
            ["analysis"] (str): analysis type (e.g., "t")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/plane x session containing information 
                                   from each mouse, with None for missing 
                                   sessions
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - lat_data (dict)        : dictionary with latency info
            ["linpla_ord"] (list)  : ordered list of planes/lines            
            ["lat_stats"] (list)   : latency statistics, structured as
                                         plane/line x session x stats
            ["lat_vals"] (list)    : latency values for each ROI, structured as
                                         plane/line x session x mouse
            ["lat_p_vals"] (list)  : p-values for each latency comparison within 
                                     session pairs, (where the second session is 
                                     cycled in the inner loop, e.g., 0-1, 0-2, 
                                     1-2, including None sessions)
                                     structured as plane/line x comp
            ["lin_p_vals"] (list)  : p-values for each line comparison, 
                                     structured as line x session (np.nan for 
                                     sessions  missing in either plane)
            ["max_comps_per"] (int): total number of comparisons
            ["n_sign_rois] (list)  : number of significant ROIs, structured as 
                                     plane/line x session
            ["tot_n_comps"] (int)  : total number of comparisons
            
    Optional args:
        - figpar (dict)    : dictionary containing the following figure parameter 
                             dictionaries
                             default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str)    : path of directory in which to save plots.
                             default: None    
    
    Returns:
        - fulldir (str) : final name of the directory in which the figure 
                          is saved (may differ from input savedir, if 
                          datetime subfolder is added.)
        - savename (str): name under which the figure is saved
    """
 
    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["bri_dir"], stimpar["bri_size"], stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    latstr = sess_str_util.lat_par_str(
        latpar["method"], latpar["p_val_thr"], latpar["rel_std"])

    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["bri_dir"], stimpar["bri_size"],
        stimpar["gabk"], "print")
    statstr_pr = sess_str_util.stat_par_str(
        analyspar["stats"], analyspar["error"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")
    latstr_pr = sess_str_util.lat_par_str(
        latpar["method"], latpar["p_val_thr"], latpar["rel_std"], "print")
    surp_resp_str = ""
    if latpar["surp_resp"]:
        surp_resp_str = "_surp_resp"
    
    datatype = extrapar["datatype"]

    sess_ns = np.asarray(sesspar["sess_n"])
    if sesspar["sess_n"] in ["any", "all"]:
        sess_ns = np.asarray(range(len(sess_info[0]))) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.reshape(-1).tolist())
    n_sess = len(sess_ns)

    [lines, planes, linpla_iter, 
    pla_cols, pla_col_names, n_plots] = sess_plot_util.fig_linpla_pars( 
        n_grps=len(lat_data["linpla_ord"]))

    # correct p-value (Bonferroni)
    p_val_thr = permpar["p_val"]
    if lat_data["tot_n_comps"] != 0:
        p_val_thr_corr = p_val_thr/lat_data["tot_n_comps"]
    else:
        p_val_thr_corr = p_val_thr
    sig_comps = [[] for _ in range(len(linpla_iter))]
    maxes = np.full([len(linpla_iter), n_sess], np.nan)

    figpar = sess_plot_util.fig_init_linpla(figpar, sharey=True)
    
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    prepost_str = sess_str_util.prepost_par_str(
        stimpar["pre"], stimpar["post"], str_type="print")
    title = (f"Surprise latencies ({prepost_str} seqs, {latstr_pr})\nfor "
        f"{stimstr_pr} - {statstr_pr} pooled across "
        f"\nROIs (grouped) (sess {sess_ns_str}{dendstr_pr})")

    fig, ax = plot_util.init_fig(n_plots, **figpar["init"])
    fig.suptitle(title, y=1.03, weight="bold")

    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        sub_ax = ax[pl, li]
        l_idx = get_linpla_idx(
            lat_data["linpla_ord"], line, pla, verbose=True, newline=(i==0))
        if l_idx is None:
            continue
        lat_st = np.asarray(lat_data["lat_stats"][l_idx])
        plot_util.plot_errorbars(
            sub_ax, lat_st[0], lat_st[1:], sess_ns, color=pla_cols[pl])
        # plot ROI cloud
        if datatype == "roi":
            maxes[i] = plot_lat_clouds(
                sub_ax, sess_ns, lat_data["lat_vals"][l_idx], sess_info[l_idx], 
                datatype=datatype, col=pla_col_names[pl])
        else:
            maxes[i] = np.max(lat_st[0])

        # check p_val signif
        all_p_vals = lat_data["lat_p_vals"][l_idx]
        for p, p_val in enumerate(all_p_vals):
            if not np.isnan(p_val) and p_val < p_val_thr_corr:
                sig_comps[i].append(p)
    
    plot_data_signif(
        ax, sess_ns, sig_comps, lat_data["lin_p_vals"], maxes, 
        p_val_thr=permpar["p_val"], n_comps=lat_data["tot_n_comps"])
    # Add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(
        ax, fluor=analyspar["fluor"], datatype=datatype, lines=lines, 
        planes=planes, xticks=sess_ns, ylab="Latency (s)", xlab="Sessions", 
        kind="reg")

    if savedir is None:
        savedir = os.path.join(
            figpar["dirs"][datatype], 
            figpar["dirs"]["acr_sess"], 
            sess_str_util.get_stimdir(stimpar["stimtype"], stimpar["gabfr"]), 
            figpar["dirs"]["lat"], 
            latpar["method"])

    savename = f"{datatype}_surp_lat_{sessstr}{dendstr}_{latstr}{surp_resp_str}"
    
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar["save"])

    return fulldir, savename


#############################################
def plot_resp_prop(analyspar, sesspar, stimpar, latpar, extrapar, sess_info, 
                   prop_data, permpar=None, figpar=None, savedir=None):
    """
    plot_resp_prop(analyspar, sesspar, stimpar, extrapar, sess_info,
                   prop_data)

    From dictionaries, plots surprise latency across mice, as well as for all 
    ROIs.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict)       : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)         : dictionary with keys of SessPar namedtuple
        - stimpar (dict)         : dictionary with keys of StimPar namedtuple
        - latpar (LatPar)        : dictionary with keys of LatPar parameters
        - extrapar (dict)        : dictionary containing additional analysis 
                                   parameters
            ["analysis"] (str): analysis type (e.g., "t")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/plane x session containing information 
                                   from each mouse, with None for missing 
                                   sessions
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ("raw", "dff")
        - prop_data (dict)        : dictionary with responsive proportion info
            ["linpla_ord"] (list): ordered list of planes/lines            
            ["prop_stats"] (list): proportion statistics, structured as
                                       plane/line x session x comb x stats
            ["comb_names"] (int) : names of combinations for with proportions 
                                   were calculated
            if latpar["surp_resp"]:
            ["n_sign_rois] (list): number of significant ROIs, structured as 
                                   plane/line x session

    Optional args:
        - permpar (PermPar): dictionary with keys of PermPar namedtuple
                             default: None
        - figpar (dict)    : dictionary containing the following figure 
                             parameter dictionaries
                             default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str)    : path of directory in which to save plots.
                             default: None    
    
    Returns:
        - fulldir (str) : final name of the directory in which the figure 
                          is saved (may differ from input savedir, if 
                          datetime subfolder is added.)
        - savename (str): name under which the figure is saved
    """
 
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    latstr = sess_str_util.lat_par_str(
        latpar["method"], latpar["p_val_thr"], latpar["rel_std"])

    statstr_pr = sess_str_util.stat_par_str(
        analyspar["stats"], analyspar["error"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")
    latstr_pr = sess_str_util.lat_par_str(
        latpar["method"], latpar["p_val_thr"], latpar["rel_std"], "print")
    surp_resp_str = ""
    if latpar["surp_resp"]:
        surp_resp_str = "_surp_resp"

    datatype = extrapar["datatype"]

    sess_ns = np.asarray(sesspar["sess_n"])
    if sesspar["sess_n"] in ["any", "all"]:
        sess_ns = np.asarray(range(len(sess_info[0]))) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.reshape(-1).tolist())

    # combinations: "gabfrs", "surps"
    ctrl_idx, surp_idx = [prop_data["comb_names"].index(comb) 
        for comb in ["gabfrs", "surps"]]

    [lines, planes, linpla_iter, pla_cols, _, n_plots] = \
        sess_plot_util.fig_linpla_pars(n_grps=len(prop_data["linpla_ord"]))
    figpar = sess_plot_util.fig_init_linpla(figpar)
    figpar["init"]["sharey"] = True

    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    prepost_str = sess_str_util.prepost_par_str(
        stimpar["pre"], stimpar["post"], str_type="print")

    title = (f"Proportion surprise responsive ROIs\n({prepost_str} seqs, "
        f"{latstr_pr})\n{statstr_pr} across mice (sess "
        f"{sess_ns_str}{dendstr_pr})")

    fig, ax = plot_util.init_fig(n_plots, **figpar["init"])
    fig.suptitle(title, y=1.03, weight="bold")

    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        sub_ax = ax[pl, li]

        l_idx = get_linpla_idx(
            prop_data["linpla_ord"], line, pla, verbose=True, newline=(i==0))
        if l_idx is None:
            continue
        
        for idx, col in zip([ctrl_idx, surp_idx], ["gray", pla_cols[pl]]):
            # retrieve proportion (* 100)
            prop_st = np.asarray([sess_vals[idx] for sess_vals 
                in prop_data["prop_stats"][l_idx]]) * 100
            plot_util.plot_errorbars(
                sub_ax, prop_st[:, 0], prop_st[:, 1:], sess_ns, color=col)
    
    # Add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(ax, fluor=analyspar["fluor"], 
        datatype=datatype, lines=lines, planes=planes, 
        xticks=sess_ns, ylab="Prop (%)", xlab="Sessions", kind="reg")

    for sub_ax in ax.reshape(-1):
        plot_util.set_ticks(
            sub_ax, axis="y", min_tick=0, max_tick=100, n=5, pad_p=0.1)
        # convert to int
        yticks = [int(v) for v in sub_ax.get_yticks()]
        sub_ax.set_yticklabels(yticks, weight="bold")

    if savedir is None:
        savedir = os.path.join(
            figpar["dirs"][datatype], 
            figpar["dirs"]["acr_sess"], 
            sess_str_util.get_stimdir(stimpar["stimtype"], stimpar["gabfr"]),
            figpar["dirs"]["prop"], 
            latpar["method"])

    savename = (f"{datatype}_prop_resp_sess{sess_ns_str}{dendstr}_{latstr}"
        f"{surp_resp_str}")
    
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar["save"])

    return fulldir, savename

