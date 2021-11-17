"""
modif_analysis_plots.py

This script is used to modify plot designs on the fly for ROI and running 
analyses from dictionaries, e.g. for presentations or papers.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import copy
import logging
import warnings
from pathlib import Path

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np

from util import file_util, gen_util, logger_util, plot_util
from sess_util import sess_plot_util, sess_str_util
from extra_plot_fcts import roi_analysis_plots as roi_plots

logger = logging.getLogger(__name__)


TAB = "    "


#############################################
def plot_from_dict(dict_path, plt_bkend=None, fontdir=None, plot_tc=True, 
                   parallel=False, datetime=True, overwrite=False):
    """
    plot_from_dict(dict_path)

    Plots data from dictionaries containing analysis parameters and results.

    Required args:
        - dict_path (Path): path to dictionary to plot data from
    
    Optional_args:
        - plt_bkend (str) : mpl backend to use for plotting (e.g., "agg")
                            default: None
        - fontdir (Path)  : path to directory where additional fonts are stored
                            default: None
        - plot_tc (bool)  : if True, tuning curves are plotted for each ROI 
                            (dummy argument)
                            default: True
        - parallel (bool) : if True, some of the analysis is parallelized 
                            across CPU cores
                            default: False
        - datetime (bool) : figpar["save"] datatime parameter (whether to 
                            place figures in a datetime folder)
                            default: True
        - overwrite (bool): figpar["save"] overwrite parameter (whether to 
                            overwrite figures)
                            default: False
    """

    logger.info(f"Plotting from dictionary: {dict_path}", 
        extra={"spacing": "\n"})
    
    figpar = sess_plot_util.init_figpar(
        plt_bkend=plt_bkend, fontdir=fontdir, datetime=datetime,
        overwrite=overwrite
        )
    plot_util.manage_mpl(cmap=False, **figpar["mng"])

    plt.rcParams["figure.titlesize"] = "xx-large"
    plt.rcParams["axes.titlesize"] = "xx-large"

    dict_path = Path(dict_path)

    info = file_util.loadfile(dict_path)
    savedir = dict_path.parent

    analysis = info["extrapar"]["analysis"]

    # 0. Plots the full traces for each session
    if analysis == "f": # full traces
        plot_full_traces(figpar=figpar, savedir=savedir, **info)

    # 1. Plot average traces by quantile x unexpected for each session 
    elif analysis == "t": # traces
        plot_traces_by_qu_unexp_sess(figpar=figpar, savedir=savedir, **info)

    # 2. Plot average traces by quantile, locked to unexpected for each session 
    elif analysis == "l": # unexpected locked traces
        plot_traces_by_qu_lock_sess(figpar=figpar, savedir=savedir, **info)

    # 4. Plot autocorrelations
    elif analysis == "a": # autocorr
        plot_autocorr(figpar=figpar, savedir=savedir, **info)

    # 6. Plot colormaps and traces for orientations/directions
    elif analysis == "o": # colormaps
        plot_oridirs(figpar=figpar, savedir=savedir, parallel=parallel, **info)

    else:
        warnings.warn(f"No modified plotting function for analysis {analysis}", 
            category=UserWarning, stacklevel=1)

    plt.close("all")

#############################################
def plot_full_traces(analyspar, sesspar, extrapar, sess_info, trace_info, 
                     figpar=None, savedir=None):
    """
    plot_full_traces(analyspar, sesspar, extrapar, sess_info, trace_info)

    From dictionaries, plots full traces for each session in a separate subplot.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict)  : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)    : dictionary with keys of SessPar namedtuple
        - extrapar (dict)   : dictionary containing additional analysis 
                              parameters
            ["analysis"] (str): analysis type (e.g., "t")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - sess_info (dict)  : dictionary containing information from each
                              session 
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session

        - trace_info (dict): dictionary containing trace information
            ["all_tr"] (nested list): trace values structured as
                                          sess x 
                                          ([each_roi (scaled), averaged] 
                                            if datatype is "roi" x)
                                          (ROI if each_roi x)
                                          (me/err if averaged x)
                                          frames
            ["all_edges"] (list)    : edge values for each parameter, 
                                      structured as sess x block x 
                                                    edges ([start, end])
            ["all_pars"] (list)     : stimulus parameter strings structured as 
                                                    sess x block
                
    Optional args:
        - figpar (dict) : dictionary containing the following figure parameter 
                          dictionaries
                          default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (Path): path of directory in which to save plots.
                          default: None    
    
    Returns:
        - fulldir (Path): final path of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
        - savename (str): name under which the figure is saved
    """
 
    statstr_pr = sess_str_util.stat_par_str(
        analyspar["stats"], analyspar["error"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")
    
    sessstr = f"sess{sesspar['sess_n']}_{sesspar['plane']}"
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    
    datatype = extrapar["datatype"]

    # extract some info from sess_info
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_ns, sess_ns, lines, planes] = [sess_info[key] for key in keys]
    
    n_sess = len(mouse_ns)
    nroi_strs = sess_str_util.get_nroi_strs(sess_info, empty=(datatype!="roi")) 

    n_rows = 1
    if datatype == "roi":
        n_rows = 2
    
    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)
    figpar["init"]["subplot_wid"] *= 4
    figpar["init"]["subplot_hei"] *= 4
    figpar["init"]["sharex"] = True
    figpar["init"]["sharey"] = False
    
    fig, ax = plot_util.init_fig(n_sess*n_rows, n_sess, **figpar["init"])
    for i in range(n_sess):
        title = (f"Mouse {mouse_ns[i]} (sess {sess_ns[i]}, {lines[i]} "
            f"{planes[i]}{dendstr_pr}{nroi_strs[i]})")

        sub_axs = ax[:, i]
        sub_axs[0].set_title(title, y=1.02)
        if datatype == "roi":
            # average trace
            av_tr = np.asarray(trace_info["all_tr"][i][1])
            xran  = range(roi_tr_sep.shape[1])
            subtitle = "u{} across ROIs".format(statstr_pr)
            plot_util.plot_traces(
                sub_axs[1], xran, av_tr[0], av_tr[1:], title=subtitle, 
                xticks="auto")
            
            # each ROI trace
            roi_tr = np.asarray(trace_info["all_tr"][i][0])
            # values to add to each ROI to split them apart
            add = np.linspace(
                0, roi_tr.shape[0] * 1.5 + 1, roi_tr.shape[0])[:, np.newaxis]
            roi_tr_sep = roi_tr + add
            sub_axs[0].plot(roi_tr_sep.T)
        else:
            run_tr = np.asarray(trace_info["all_tr"][i])
            sub_axs[0].plot(run_tr)

        for b, block in enumerate(trace_info["all_edges"]):
            # all block labels to the lower plot
            plot_util.add_labels(
                sub_axs[-1], trace_info["all_pars"][b], np.mean(block), 0.85, 
                color="k")
            # add lines to both plots
            for r in range(n_rows):
                plot_util.add_bars(sub_axs[r], hbars=block)
                sess_plot_util.add_axislabels(
                    sub_axs[r], fluor=analyspar["fluor"], datatype=datatype, 
                    x_ax="")
                
    if savedir is None:
        savedir = Path(
            figpar["dirs"][datatype])

    savename = f"{datatype}_tr_{sessstr}{dendstr}"
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar["save"])

    return fulldir, savename


#############################################
def plot_traces_by_qu_unexp_sess(analyspar, sesspar, stimpar, extrapar, 
                                quantpar, sess_info, trace_stats, figpar=None, 
                                savedir=None):
    """
    plot_traces_by_qu_unexp_sess(analyspar, sesspar, stimpar, extrapar, 
                                quantpar, sess_info, trace_stats)

    From dictionaries, plots traces by quantile/unexpected with each session in a 
    separate subplot.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict)  : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)    : dictionary with keys of SessPar namedtuple
        - stimpar (dict)    : dictionary with keys of StimPar namedtuple
        - extrapar (dict)   : dictionary containing additional analysis 
                              parameters
            ["analysis"] (str): analysis type (e.g., "t")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - quantpar (dict)   : dictionary with keys of QuantPar namedtuple
        - sess_info (dict)  : dictionary containing information from each
                              session 
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session

        - trace_stats (dict): dictionary containing trace stats information
            ["xrans"] (list)           : time values for the 2p frames, for 
                                         each session
            ["all_stats"] (list)       : list of 4D arrays or lists of trace 
                                         data statistics across ROIs, 
                                         structured as:
                                            unexp x quantiles x
                                            stats (me, err) x frames
            ["all_counts"] (array-like): number of sequences, structured as:
                                                sess x unexp x quantiles
                
    Optional args:
        - figpar (dict) : dictionary containing the following figure parameter 
                          dictionaries
                          default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (Path): path of directory in which to save plots.
                          default: None    
    
    Returns:
        - fulldir (Path): final path of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
        - savename (str): name under which the figure is saved
    """
 
    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["visflow_dir"], stimpar["visflow_size"],
        stimpar["gabk"], "print")
    statstr_pr = sess_str_util.stat_par_str(
        analyspar["stats"], analyspar["error"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")
    
    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["visflow_dir"], stimpar["visflow_size"], stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    
    datatype = extrapar["datatype"]
    dimstr = sess_str_util.datatype_dim_str(datatype)

    # extract some info from sess_info
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_ns, sess_ns, lines, planes] = [sess_info[key] for key in keys]
    
    n_sess = len(mouse_ns)
    nroi_strs = sess_str_util.get_nroi_strs(
        sess_info, empty=(datatype!="roi"), style="par"
        )

    xrans      = [np.asarray(xran) for xran in trace_stats["xrans"]]
    all_stats  = [np.asarray(sessst) for sessst in trace_stats["all_stats"]]
    all_counts = trace_stats["all_counts"]

    cols, lab_cols = sess_plot_util.get_quant_cols(quantpar["n_quants"])
    alpha = np.min([0.4, 0.8/quantpar["n_quants"]])

    unexps = ["exp", "unexp"]
    n = 6
    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    fig, ax = plot_util.init_fig(n_sess, **figpar["init"])
    for i in range(n_sess):
        sub_ax = plot_util.get_subax(ax, i)
        for s, [col, leg_ext] in enumerate(zip(cols, unexps)):
            for q, qu_lab in enumerate(quantpar["qu_lab"]):
                if qu_lab != "":
                    qu_lab = f"{qu_lab.capitalize()} "
                line, plane = "5", "dendrites"
                if "23" in lines[i]:
                    line = "2/3"
                if "soma" in planes[i]:
                    plane = "somata"
                title = (f"M{mouse_ns[i]} - layer {line} {plane}{dendstr_pr}")
                # title=(f"Mouse {mouse_ns[i]} - {stimstr_pr} " + 
                #     u"{} ".format(statstr_pr) + f"across {dimstr}\n"
                #     f"(sess {sess_ns[i]}, {lines[i]} {planes[i]},"
                #     f"{nroi_strs[i]}")
                leg = None
                y_ax = ""
                if i == 0:
                    leg = f"{qu_lab}{leg_ext}"
                    y_ax = None
                # leg = f"{qu_lab}{leg_ext} ({all_counts[i][s][q]})"
                plot_util.plot_traces(
                    sub_ax, xrans[i], all_stats[i][s, q, 0], 
                    all_stats[i][s, q, 1:], title, color=col[q], alpha=alpha, 
                    label=leg, n_xticks=n, xticks="auto")
                sess_plot_util.add_axislabels(
                    sub_ax, fluor=analyspar["fluor"], datatype=datatype, 
                    y_ax=y_ax)

    plot_util.turn_off_extra(ax, n_sess)

    if stimpar["stimtype"] == "gabors": 
        sess_plot_util.plot_labels(
            ax, stimpar["gabfr"], "both", pre=stimpar["pre"], 
            post=stimpar["post"], cols=lab_cols, 
            sharey=figpar["init"]["sharey"])
    
    if savedir is None:
        savedir = Path(
            figpar["dirs"][datatype], 
            figpar["dirs"]["unexp_qu"])

    qu_str = f"_{quantpar['n_quants']}q"
    if quantpar["n_quants"] == 1:
        qu_str = ""

    savename = f"{datatype}_av_{sessstr}{dendstr}{qu_str}"
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar["save"])

    return fulldir, savename


#############################################
def plot_traces_by_qu_lock_sess(analyspar, sesspar, stimpar, extrapar, 
                                quantpar, sess_info, trace_stats, 
                                figpar=None, savedir=None):
    """
    plot_traces_by_qu_lock_sess(analyspar, sesspar, stimpar, extrapar, 
                                quantpar, sess_info, trace_stats)

    From dictionaries, plots traces by quantile, locked to transitions from 
    unexpected to expected or v.v. with each session in a separate subplot.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict)  : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)    : dictionary with keys of SessPar namedtuple
        - stimpar (dict)    : dictionary with keys of StimPar namedtuple
        - extrapar (dict)   : dictionary containing additional analysis 
                              parameters
            ["analysis"] (str): analysis type (e.g., "l")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - quantpar (dict)   : dictionary with keys of QuantPar namedtuple
        - sess_info (dict)  : dictionary containing information from each
                              session 
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session

        - trace_stats (dict): dictionary containing trace stats information
            ["xrans"] (list)           : time values for the 2p frames, for 
                                         each session
            ["all_stats"] (list)       : list of 4D arrays or lists of trace 
                                         data statistics across ROIs, 
                                         structured as:
                                            (unexp_len x) quantiles x
                                            stats (me, err) x frames
            ["all_counts"] (array-like): number of sequences, structured as:
                                                sess x (unexp_len x) quantiles
            ["lock"] (str)             : value to which segments are locked:
                                         "unexp", "exp" or "unexp_split"
            ["baseline"] (num)         : number of seconds used for baseline
            ["exp_stats"] (list)       : list of 3D arrays or lists of trace 
                                         data statistics across ROIs for
                                         expected sampled sequences, 
                                         structured as:
                                            quantiles (1) x stats (me, err) 
                                            x frames
            ["exp_counts"] (array-like): number of sequences corresponding to
                                         exp_stats, structured as:
                                            sess x quantiles (1)
            
            if data is by unexp_len:
            ["unexp_lens"] (list)       : number of consecutive segments for
                                         each unexp_len, structured by session
                
    Optional args:
        - figpar (dict) : dictionary containing the following figure parameter 
                          dictionaries
                          default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (Path): path of directory in which to save plots.
                          default: None    
    
    Returns:
        - fulldir (Path): final path of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
        - savename (str): name under which the figure is saved
    """

    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["visflow_dir"], stimpar["visflow_size"],
        stimpar["gabk"], "print")
    statstr_pr = sess_str_util.stat_par_str(
        analyspar["stats"], analyspar["error"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")

    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["visflow_dir"], stimpar["visflow_size"], stimpar["gabk"])
    basestr = sess_str_util.base_par_str(trace_stats["baseline"])
    basestr_pr = sess_str_util.base_par_str(trace_stats["baseline"], "print")

    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
    
    datatype = extrapar["datatype"]
    dimstr = sess_str_util.datatype_dim_str(datatype)

    # extract some info from sess_info
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_ns, sess_ns, lines, planes] = [sess_info[key] for key in keys]
    
    n_sess = len(mouse_ns)
    nroi_strs = sess_str_util.get_nroi_strs(
        sess_info, empty=(datatype!="roi"), style="par"
        )

    xrans      = [np.asarray(xran) for xran in trace_stats["xrans"]]
    all_stats  = [np.asarray(sessst) for sessst in trace_stats["all_stats"]]
    exp_stats  = [np.asarray(expst) for expst in trace_stats["exp_stats"]]
    all_counts = trace_stats["all_counts"]
    exp_counts = trace_stats["exp_counts"]

    lock  = trace_stats["lock"]
    col_idx = 0
    if "unexp" in lock:
        lock = "unexp"
        col_idx = 1
    
    unexp_lab, len_ext = "", ""
    unexp_lens = [[None]] * n_sess
    offset = 0
    unexp_len_default = True # plot default unexpected lengths
    if (stimpar["stimtype"] == "gabors" and 
        stimpar["gabfr"] not in ["any", "all"]):
        offset = stimpar["gabfr"]
    if "unexp_lens" in trace_stats.keys():
        unexp_lens = trace_stats["unexp_lens"]
        len_ext = "_bylen"
        if stimpar["stimtype"] == "gabors":
            offset = stimpar["gabfr"]
            unexp_lens = [[sl * 1.5/4 - 0.3 * offset for sl in sls] 
                for sls in unexp_lens]
        unexp_len_default = False

    # plot unexp_lens default values and RANGE TO PLOT
    if stimpar["stimtype"] == "gabors":
        DEFAULT_UNEXP_LENS = [3.0, 4.5, 6.0]
        end_val = 8.0
    else:
        DEFAULT_UNEXP_LENS = [2.0, 3.0, 4.0]
        end_val = 6.0
    if lock == "unexp":
        st_val = -2.0
        inv = 1
    else:
        st_val = -end_val + 2
        end_val = end_val - 2
        inv = -1
    n_ticks = int((end_val - st_val)//2 + 1)
    
    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)
    figpar["init"]["subplot_wid"] = 6.5

    fig, ax = plot_util.init_fig(n_sess, **figpar["init"])
    for i, (stats, counts) in enumerate(zip(all_stats, all_counts)):
        xran = xrans[i]
        st, end = 0, len(xran)
        st_vals = list(filter(lambda i: xran[i] <= st_val, range(len(xran))))
        if len(st_vals) != 0:
            st = st_vals[-1]
        end_vals = list(filter(lambda i: xran[i] >= end_val, range(len(xran))))
        if len(end_vals) != 0:
            end = end_vals[0] + 1

        sub_ax = plot_util.get_subax(ax, i)

        line, plane = "5", "dendrites"
        if "23" in lines[i]:
            line = "2/3"
        if "soma" in planes[i]:
            plane = "somata"

        title = f"M{mouse_ns[i]} - layer {line} {plane}{dendstr_pr}"
        # title=(f"Mouse {mouse_ns[i]} - {stimstr_pr} " + 
        #     u"{statstr_pr} " + f"{lock} locked across {dimstr}{basestr_pr}"
        #     f"\n(sess {sess_ns[i]}, {lines[i]} {planes[i]}, {nroi_strs[i]}")
        y_ax = ""
        if i == 0:
            y_ax = None

        sess_plot_util.add_axislabels(
            sub_ax, fluor=analyspar["fluor"], datatype=datatype, y_ax=y_ax)
        plot_util.add_bars(sub_ax, hbars=0)
        n_lines = quantpar["n_quants"] * len(unexp_lens[i])
        if col_idx < n_lines:
            cols = sess_plot_util.get_quant_cols(n_lines)[0][col_idx]
        else:
            cols = [None] * n_lines
        alpha = np.min([0.4, 0.8/n_lines])
        if stimpar["stimtype"] == "gabors":
            sess_plot_util.plot_gabfr_pattern(sub_ax, xran[st:end], 
                offset=offset, bars_omit=[0] + unexp_lens[i])
        # plot expected data
        if exp_stats[i].shape[0] != 1:
            raise RuntimeError("Expected only one quantile for exp_stats.")
        
        leg = None
        if i == 0:
            leg = "exp"
        if unexp_len_default:
            for leng in DEFAULT_UNEXP_LENS:
                edge = leng * inv
                if edge < 0:
                    edge = np.max([xran[st], edge])
                elif edge > 0:
                    edge = np.min([xran[end - 1], edge])
                plot_util.add_vshade(
                    sub_ax, 0, edge, color=cols[-1], alpha=0.1)
        # leg = f"exp (no lock) ({exp_counts[i][0]})"
        plot_util.plot_traces(
            sub_ax, xran[st:end], exp_stats[i][0][0, st:end], 
            exp_stats[i][0][1:, st:end], alpha=alpha, label=leg, 
            alpha_line=0.8, color="darkgray", xticks="auto")
        n = 0 # count lines plotted
        for s, unexp_len in enumerate(unexp_lens[i]):
            if unexp_len is not None:
                edge = unexp_len * inv
                if edge < 0:
                    edge = np.max([xran[st], edge])
                elif edge > 0:
                    edge = np.min([xran[end - 1], edge])
                plot_util.add_vshade(
                    sub_ax, 0, edge, color=cols[n], alpha=0.1)
                counts, stats = all_counts[i][s], all_stats[i][s]                
                unexp_lab = f"unexp len {unexp_len}"
            else:
                # unexp_lab = "unexp lock"
                unexp_lab = "unexp"
            for q, qu_lab in enumerate(quantpar["qu_lab"]):
                if qu_lab != "":
                    qu_lab = f"{qu_lab.capitalize()} "
                lab = f"{qu_lab}{unexp_lab}"
                if n == 2 and cols[n] is None:
                    sub_ax.plot([], []) # to advance the color cycle (past gray)
                #leg = f"{lab} ({counts[q]})"
                leg = None
                if i == 0:
                    leg = lab
                plot_util.plot_traces(
                    sub_ax, xran[st:end], stats[q][0, st:end], 
                    stats[q][1:, st:end], title, alpha=alpha, label=leg, 
                    n_xticks=n_ticks, alpha_line=0.8, color=cols[n], 
                    xticks="auto")
                n += 1
            if unexp_len is not None:
                plot_util.add_bars(sub_ax, hbars=unexp_len, 
                                   color=sub_ax.lines[-1].get_color(), alpha=1)
    
    plot_util.turn_off_extra(ax, n_sess)

    if savedir is None:
        savedir = Path(
            figpar["dirs"][datatype], 
            figpar["dirs"]["unexp_qu"], 
            f"{lock}_lock", basestr.replace("_", ""))

    qu_str = f"_{quantpar['n_quants']}q"
    if quantpar["n_quants"] == 1:
        qu_str = ""
 
    savename = (f"{datatype}_av_{lock}lock{len_ext}{basestr}_{sessstr}"
        f"{dendstr}{qu_str}")
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar["save"])

    return fulldir, savename


#############################################
def plot_autocorr(analyspar, sesspar, stimpar, extrapar, autocorrpar, 
                  sess_info, autocorr_data, figpar=None, savedir=None):
    """
    plot_autocorr(analyspar, sesspar, stimpar, extrapar, autocorrpar, 
                  sess_info, autocorr_data)

    From dictionaries, plots autocorrelation during stimulus blocks.

    Required args:
        - analyspar (dict)    : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)      : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)      : dictionary with keys of StimPar namedtuple
        - extrapar (dict)     : dictionary containing additional analysis 
                                parameters
            ["analysis"] (str): analysis type (e.g., "a")
            ["datatype"] (str): datatype (e.g., "run", "roi")
        - autocorrpar (dict)  : dictionary with keys of AutocorrPar namedtuple
        - sess_info (dict)    : dictionary containing information from each
                                session 
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session

        - autocorr_data (dict): dictionary containing data to plot:
            ["xrans"] (list): list of lag values in seconds for each session
            ["stats"] (list): list of 3D arrays (or nested lists) of
                              autocorrelation statistics, structured as:
                                     sessions stats (me, err) 
                                     x ROI or 1x and 10x lag 
                                     x lag
    
    Optional args:
        - figpar (dict) : dictionary containing the following figure parameter 
                          dictionaries
                          default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (Path): path of directory in which to save plots.
                          default: None
    
    Returns:
        - fulldir (Path): final path of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
        - savename (str): name under which the figure is saved
    """


    statstr_pr = sess_str_util.stat_par_str(
        analyspar["stats"], analyspar["error"], "print")
    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["visflow_dir"], stimpar["visflow_size"], 
        stimpar["gabk"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")

    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["visflow_dir"],stimpar["visflow_size"], stimpar["gabk"]) 
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])


    datatype = extrapar["datatype"]
    if datatype == "roi":
        fluorstr_pr = sess_str_util.fluor_par_str(
            analyspar["fluor"], str_type="print")
        title_str = u"{} autocorrelation".format(fluorstr_pr)
        if not autocorrpar["byitem"]:
            title_str = f"{title_str} across ROIs" 
    elif datatype == "run":
        datastr = sess_str_util.datatype_par_str(datatype)
        title_str = f"{datastr} autocorrelation"

    if stimpar["stimtype"] == "gabors":
        seq_bars = [-1.5, 1.5] # light lines
    else:
        seq_bars = [-1.0, 1.0] # light lines

    # extract some info from sess_info
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_ns, sess_ns, lines, planes] = [sess_info[key] for key in keys]
    
    n_sess = len(mouse_ns)
    nroi_strs = sess_str_util.get_nroi_strs(
        sess_info, empty=(datatype!="roi"), style="par"
        )

    xrans = autocorr_data["xrans"]
    stats = [np.asarray(stat) for stat in autocorr_data["stats"]]

    lag_s = autocorrpar["lag_s"]
    xticks = np.linspace(-lag_s, lag_s, lag_s*2+1)
    yticks = np.linspace(0, 1, 6)

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    byitemstr = ""
    if autocorrpar["byitem"]:
        byitemstr = "_byroi"

    fig, ax = plot_util.init_fig(n_sess, **figpar["init"])
    for i in range(n_sess):
        sub_ax = plot_util.get_subax(ax, i)
        line, plane = "5", "dendrites"
        if "23" in lines[i]:
            line = "2/3"
        if "soma" in planes[i]:
            plane = "somata"
        title=f"M{mouse_ns[i]} - layer {line} {plane}{dendstr_pr}"
        # title = (f"Mouse {mouse_ns[i]} - {stimstr_pr}" + 
        #     u"{}".format(statstr_pr) + 
        #     f"{title_str}\n(sess {sess_ns[i]}, {lines[i]} {planes[i]},"
        #     f"{nroi_strs[i]})")
        # transpose to ROI/lag x stats x series
        sess_stats = stats[i].transpose(1, 0, 2) 
        for s, sub_stats in enumerate(sess_stats):
            lab = None
            if i == 0:
                if not autocorrpar["byitem"]:
                    lab = ["actual lag", "10x lag"][s]
            plot_util.plot_traces(
                sub_ax, xrans[i], sub_stats[0], sub_stats[1:], xticks=xticks, 
                yticks=yticks, alpha=0.2, label=lab)
        plot_util.add_bars(sub_ax, hbars=seq_bars)
        sub_ax.set_ylim([0, 1])
        sub_ax.set_title(title, y=1.02)
        sub_ax.set_xlabel("Lag (s)")

    plot_util.turn_off_extra(ax, n_sess)

    if savedir is None:
        savedir = Path(
            figpar["dirs"][datatype], 
            figpar["dirs"]["autocorr"])

    savename = f"{datatype}_autocorr{byitemstr}_{sessstr}{dendstr}"

    fulldir = plot_util.savefig(fig, savename, savedir, **figpar["save"])

    return fulldir, savename


#############################################
def plot_oridir_traces(analyspar, sesspar, stimpar, extrapar, quantpar, 
                        tr_data, sess_info, figpar=None, savedir=None):
    """
    plot_oridir_traces(analyspar, sesspar, stimpar, extrapar, quantpar, 
                       tr_data, sess_info)

    From dictionaries, plots average activity across gabor orientations or 
    visual flow directions per ROI as colormaps for a single session and 
    optionally a single quantile. 

    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "o")
            ["datatype"] (str): datatype (e.g., "roi")
        - quantpar (dict) : dictionary with keys of QuantPar namedtuple
        - sess_info (dict): dictionary containing information from each
                            session (only first session used)
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session

        - tr_data (dict)   : dictionary containing information to plot colormap.
                             unexpected x ori/dir keys are formatted as 
                             [{s}_{od}] for unexp in ["exp", "unexp"]
                                        and od in [0, 45, 90, 135] or 
                                                  ["right", "left"]
            ["n_seqs"] (dict): dictionary containing number of seqs for each
                               unexpected x ori/dir combination under a 
                               separate key
            ["stats"] (dict) : dictionary containing trace mean/medians across
                               ROIs in 2D arrays or nested lists, 
                               structured as:
                                   stats (me, err) x frames
                               with each unexpected x ori/dir combination under a 
                               separate key
                               (NaN arrays for combinations with 0 seqs.)
            ["xran"] (list)  : time values for the 2p frames

    Optional args:
        - figpar (dict) : dictionary containing the following figure parameter 
                          dictionaries
                          default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (Path): path of directory in which to save plots.
                          default: None
    
    Returns:
        - fulldir (Path): final path of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
        - savename (str): name under which the figure is saved
    """

    statstr_pr = sess_str_util.stat_par_str(
        analyspar["stats"], analyspar["error"], "print")
    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["visflow_dir"], stimpar["visflow_size"], 
        stimpar["gabk"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")

    stimstr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["visflow_dir"], stimpar["visflow_size"], 
        stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])

    datatype = extrapar["datatype"]
    if datatype != "roi":
        raise NotImplementedError("Function only implemented for roi datatype.")

    if savedir is None:
        savedir = Path(
            figpar["dirs"][datatype], 
            figpar["dirs"]["oridir"])

    # extract some info from dictionaries
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_n, sess_n, line, plane] = [sess_info[key][0] for key in keys]

    nroi_str = sess_str_util.get_nroi_strs(sess_info, style="par")[0] 

    xran = tr_data["xran"]

    unexps = ["exp", "unexp"]
    if stimpar["stimtype"] == "gabors":
        unexp_labs = unexps
        deg = u"\u00B0"
        oridirs = stimpar["gab_ori"]
        n = 6
    elif stimpar["stimtype"] == "visflow":
        unexp_labs = [f"{unexps[i]} -> {unexps[1-i]}" for i in range(len(unexps))]
        deg  = ""
        oridirs = stimpar["visflow_dir"]
        n = 7

    qu_str, qu_str_pr = quantpar["qu_lab"][0], quantpar["qu_lab_pr"][0]
    if qu_str != "":
        qu_str    = f"_{qu_str}"      
    if qu_str_pr != "":
        qu_str_pr = f" - {qu_str_pr.capitalize()}"

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    figpar["init"]["ncols"] = len(oridirs) 
    
    # suptitle = (f"Mouse {mouse_n} - {stimstr_pr}" + 
    #     u"{} ".format(statstr_pr) + f"across {dimstr}{qu_str_pr}\n"
    #     f"(sess {sess_n}, {line} {plane},{nroi_str}")
    line_str, plane_str = "5", "dendrites"
    if "23" in line:
        line_str = "2/3"
    if "soma" in plane:
        plane_str = "somata"
    suptitle = f"M{mouse_n} - layer {line_str} {plane_str}{dendstr_pr}"

    savename = (f"{datatype}_tr_m{mouse_n}_sess{sess_n}{qu_str}_{stimstr}_"
        f"{plane}{dendstr}")
    
    fig, ax = plot_util.init_fig(len(oridirs), **figpar["init"])
    for o, od in enumerate(oridirs):
        cols = []
        for unexp, unexp_lab in zip(unexps, unexp_labs): 
            sub_ax = plot_util.get_subax(ax, o)
            key = f"{unexp}_{od}"
            stimtype_str_pr = stimpar["stimtype"][:-1].capitalize()
            title_tr = u"{} traces ({}{})".format(stimtype_str_pr, od, deg)
            lab = f"{unexp_lab} (n={tr_data['n_seqs'][key]})"
            sess_plot_util.add_axislabels(sub_ax, datatype=datatype)
            me  = np.asarray(tr_data["stats"][key][0])
            err = np.asarray(tr_data["stats"][key][1:])
            plot_util.plot_traces(
                sub_ax, xran, me, err, title_tr, label=lab, n_xticks=n, 
                xticks="auto")
            cols.append(sub_ax.lines[-1].get_color())
    
    
    if stimpar["stimtype"] == "gabors":
        sess_plot_util.plot_labels(
            ax, stimpar["gabfr"], cols=cols, pre=stimpar["pre"], 
            post=stimpar["post"], sharey=figpar["init"]["sharey"])

    fig.suptitle(suptitle, y=1.08)
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar["save"])

    return fulldir


#############################################
def plot_oridir_colormap(fig_type, analyspar, stimpar, quantpar, tr_data, 
                         sess_info, figpar=None, savedir=None, log_dir=True):
    """
    plot_oridir_colormap(fig_type, analyspar, stimpar, quantpar, tr_data, 
                         sess_info)

    From dictionaries, plots average activity across gabor orientations or 
    visual flow directions per ROI for a single session and optionally a single 
    quantile. (Single figure type) 

    Required args:
        - fig_type (str)  : type of figure to plot, i.e., "byplot", "byexp", 
                            "byfir" or "by{}{}" (ori/dir, deg)
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - quantpar (dict) : dictionary with keys of QuantPar namedtuple
        - sess_info (dict): dictionary containing information from each
                            session (only first session used)
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session

        - tr_data (dict)   : dictionary containing information to plot colormap.
                             unexpected x ori/dir keys are formatted as 
                             [{s}_{od}] for unexp in ["exp", "unexp"]
                                        and od in [0, 45, 90, 135] or 
                                                  ["right", "left"]
            ["n_seqs"] (dict)    : dictionary containing number of seqs for 
                                   each unexpected x ori/dir combination under a 
                                   separate key
            ["scale_vals"] (dict): dictionary containing 1D array or list of 
                                   scaling values for each unexpected x ori/dir 
                                   combination under a separate key.
                                   (NaN arrays for combinations with 0 seqs.)
                ["{}_min"] (num): minimum value from corresponding tr_stats 
                                  mean/medians
                ["{}_max"] (num): maximum value from corresponding tr_stats 
                                  mean/medians
            ["roi_sort"] (dict)  : dictionary containing 1D arrays or list 
                                   of peak sorting order for each 
                                   unexpected x ori/dir combination under a 
                                   separate key.
                                   (NaN arrays for combinations with 0 seqs.)
            ["roi_me"] (dict)    : dictionary containing trace mean/medians for 
                                   each ROI as 2D arrays or nested lists, 
                                   structured as:
                                       ROIs x frames, 
                                   with each unexpected x ori/dir combination 
                                   under a separate key
                                   (NaN arrays for combinations with 0 seqs.)
            ["xran"] (list)      : time values for the 2p frames
    
    Optional args:
        - figpar (dict) : dictionary containing the following figure parameter 
                          dictionaries
                          default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
            ["mng"]  (dict): dictionary with parameters to manage matplotlib
        - savedir (Path): path of directory in which to save plots.
                          default: None
        - log_dir (bool): if True, the figure saving directory is logged.
                          default: True
    
    Returns:
        - fulldir (Path): final path of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
    """

    statstr_pr = sess_str_util.stat_par_str(
        analyspar["stats"], analyspar["error"], "print")
    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["visflow_dir"], stimpar["visflow_size"], 
        stimpar["gabk"], "print")

    stimstr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["visflow_dir"], stimpar["visflow_size"], 
        stimpar["gabk"])

    if savedir is None:
        savedir = Path(
            figpar["dirs"]["roi"], 
            figpar["dirs"]["oridir"])

    cmap = plot_util.manage_mpl(cmap=True, nbins=100, **figpar["mng"])

    # extract some info from sess_info (only one session)
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_n, sess_n, line, plane] = [sess_info[key][0] for key in keys]

    dendstr = sess_str_util.dend_par_str(analyspar["dend"], plane, "roi")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], plane, "roi", "print")

    unexps = ["exp", "unexp"]
    if stimpar["stimtype"] == "gabors":
        unexp_labs = unexps
        var_name = "orientation"
        deg  = "deg"
        deg_pr = u"\u00B0"
        oridirs = stimpar["gab_ori"]
        n = 6
    elif stimpar["stimtype"] == "visflow":
        unexp_labs = [f"{unexps[i]} -> {unexps[1-i]}" for i in range(len(unexps))]
        var_name = "direction"
        deg  = ""
        deg_pr = ""
        oridirs = stimpar["visflow_dir"]
        n = 7
    
    qu_str, qu_str_pr = quantpar["qu_lab"][0], quantpar["qu_lab_pr"][0]
    if qu_str != "":
        qu_str    = f"_{qu_str}"      
    if qu_str_pr != "":
        qu_str_pr = f" - {qu_str_pr.capitalize()}"

    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)
    figpar["init"]["ncols"] = len(oridirs)
    figpar["init"]["sharex"] = True
    
    # plot colormaps
    # gentitle = (f"Mouse {mouse_n} - {stimstr_pr} " + 
    #     u"{} ".format(statstr_pr) + "across seqs colormaps"
    #     f"{qu_str_pr} \n(sess {sess_n}, {line} {plane})")
    line_str, plane_str = "5", "dendrites"
    if "23" in line:
        line_str = "2/3"
    if "soma" in plane:
        plane_str = "somata"
    gentitle = f"Mouse {mouse_n} - layer {line_str} {plane_str}{dendstr_pr}"
    
    gen_savename = (f"roi_cm_m{mouse_n}_sess{sess_n}{qu_str}_{stimstr}_"
        f"{plane}{dendstr}")

    gen_savename = f"colormap_m{mouse_n}s{sess_n}_{stimstr}_{plane}"

    if fig_type != "byfir":
        return ""

    if fig_type == "byplot":
        scale_type = "per plot"
        peak_sort  = ""
        figpar["init"]["sharey"] = False
    elif fig_type == "byexp":
        scale_type = f"within {var_name}"
        peak_sort  = f" of {unexps[0]}"
        figpar["init"]["sharey"] = False
    elif fig_type == f"by{oridirs[0]}{deg}":
        scale_type = "within unexp/exp"
        peak_sort  = f" of first {var_name}"
        figpar["init"]["sharey"] = True
    elif fig_type == "byfir":
        scale_type = "across plots"
        peak_sort  = " of first plot"
        figpar["init"]["sharey"] = True
    else:
        gen_util.accepted_values_error("fig_type", fig_type, 
            ["byplot", "byexp", f"by{oridirs[0]}{deg}", "byfir"])

    subtitle = (f"ROIs sorted by peak activity{peak_sort} and scaled "
                f"{scale_type}")
    logger.info(f"- {subtitle}", extra={"spacing": TAB})
    # suptitle = f"{gentitle}\n({subtitle})"
    suptitle = gentitle
    
    # get scaled and sorted ROI mean/medians (ROI x frame)
    scaled_sort_me = roi_plots.scale_sort_trace_data(
        tr_data, fig_type, unexps, oridirs)
    fig, ax = plot_util.init_fig(len(oridirs) * len(unexps), **figpar["init"])
    
    xran_edges = [np.min(tr_data["xran"]), np.max(tr_data["xran"])]

    nrois = scaled_sort_me[f"{unexps[0]}_{oridirs[0]}"].shape[1]
    yticks_ev = int(10 * np.max([1, np.ceil(nrois/100)])) # avoid > 10 ticks
    for o, od in enumerate(oridirs):
        for s, (unexp, unexp_lab) in enumerate(zip(unexps, unexp_labs)):    
            sub_ax = ax[s][o]
            key = f"{unexp}_{od}"
            title = u"{} seqs ({}{}) (n={})".format(
                unexp_lab.capitalize(), od, deg_pr, tr_data["n_seqs"][key])
            if s == 0:
                od_pr = od
                if stimpar["stimtype"] == "visflow":
                    od_pr = od_pr.capitalize()
                title = u"{}{}".format(od_pr, deg_pr)
            else:
                title = None
            x_ax = None
            y_ax = "ROIs"
            if s != 1 or o != 0:
                y_ax = ""
            if stimpar["stimtype"] == "gabors":
                x_ax = ""
            sess_plot_util.add_axislabels(
                sub_ax, fluor=analyspar["fluor"], x_ax=x_ax, y_ax=y_ax, 
                datatype="roi")
            im = plot_util.plot_colormap(
                sub_ax, scaled_sort_me[key], title=title, cmap=cmap, n_xticks=n,
                yticks_ev=yticks_ev, xran=xran_edges, xticks="auto")
            
            if stimpar["stimtype"] == "visflow":
                plot_util.add_bars(sub_ax, 0)
            else:
                sub_ax.set_xticks([])

    for s, unexp in enumerate(unexps):
        sub_ax = ax[s:s+1]
        if stimpar["stimtype"] == "gabors":
            sess_plot_util.plot_labels(
                sub_ax, stimpar["gabfr"], unexp, pre=stimpar["pre"], 
                post=stimpar["post"], sharey=figpar["init"]["sharey"], 
                t_heis=-0.05)
    
    plot_util.add_colorbar(fig, im, len(oridirs), cm_prop=0.06)
    fig.suptitle(suptitle, fontsize="xx-large", y=1.08)
    savename = f"{gen_savename}_{fig_type}"
    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=log_dir, **figpar["save"])
    
    plt.close(fig)
    
    return fulldir


#############################################
def plot_oridir_colormaps(analyspar, sesspar, stimpar, extrapar, quantpar, 
                          tr_data, sess_info, figpar=None, savedir=None, 
                          parallel=False):
    """
    plot_oridir_colormaps(analyspar, sesspar, stimpar, extrapar, quantpar, 
                          tr_data, sess_info)

    From dictionaries, plots average activity across gabor orientations or 
    visual flow directions per ROI as colormaps for a single session and 
    optionally a single quantile. 

    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "o")
            ["datatype"] (str): datatype (e.g., "roi")
        - quantpar (dict) : dictionary with keys of QuantPar namedtuple
        - sess_info (dict): dictionary containing information from each
                            session (only first session used)
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session

        - tr_data (dict)   : dictionary containing information to plot colormap.
                             unexpected x ori/dir keys are formatted as 
                             [{s}_{od}] for unexp in ["exp", "unexp"]
                                        and od in [0, 45, 90, 135] or 
                                                  ["right", "left"]
            ["n_seqs"] (dict)    : dictionary containing number of seqs for 
                                   each unexpected x ori/dir combination under a 
                                   separate key
            ["scale_vals"] (dict): dictionary containing 1D array or list of 
                                   scaling values for each unexpected x ori/dir 
                                   combination under a separate key.
                                   (NaN arrays for combinations with 0 seqs.)
                ["{}_min"] (num): minimum value from corresponding tr_stats 
                                  mean/medians
                ["{}_max"] (num): maximum value from corresponding tr_stats 
                                  mean/medians
            ["roi_sort"] (dict)  : dictionary containing 1D arrays or list 
                                   of peak sorting order for each 
                                   unexpected x ori/dir combination under a 
                                   separate key.
                                   (NaN arrays for combinations with 0 seqs.)
            ["roi_me"] (dict)    : dictionary containing trace mean/medians for 
                                   each ROI as 2D arrays or nested lists, 
                                   structured as:
                                       ROIs x frames, 
                                   with each unexpected x ori/dir combination 
                                   under a separate key
                                   (NaN arrays for combinations with 0 seqs.)
    
    Optional args:
        - figpar (dict)  : dictionary containing the following figure parameter 
                           dictionaries
                           default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
            ["mng"]  (dict): dictionary with parameters to manage matplotlib
        - savedir (Path) : path of directory in which to save plots.
                           default: None
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
                           default: False
    
    Returns:
        - fulldir (Path): final path of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
    """

    datatype = extrapar["datatype"]
    if datatype != "roi":
        raise NotImplementedError("Function only implemented for roi datatype.")

    if stimpar["stimtype"] == "gabors":
        oridirs = stimpar["gab_ori"]
        deg  = "deg"
    elif stimpar["stimtype"] == "visflow":
        oridirs = stimpar["visflow_dir"]
        deg = ""
    
    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()
    figpar["save"]["fig_ext"] = "svg" # svg too big

    fig_types  = ["byplot", "byexp", f"by{oridirs[0]}{deg}", "byfir"]
    fig_types = ["byfir"]
    fig_last = len(fig_types) - 1
    
    # optionally runs in parallel
    if parallel:
        n_jobs = gen_util.get_n_jobs(len(fig_types))
        fulldirs = Parallel(n_jobs=n_jobs)(delayed(plot_oridir_colormap)
            (fig_type, analyspar, stimpar, quantpar, tr_data, sess_info, 
            figpar, savedir, (f == fig_last)) 
            for f, fig_type in enumerate(fig_types)) 
        fulldir = fulldirs[-1]
    else:
        for f, fig_type in enumerate(fig_types):
            log_dir = (f == fig_last)
            fulldir = plot_oridir_colormap(
                fig_type, analyspar, stimpar, quantpar, tr_data, sess_info, 
                figpar, savedir, log_dir)

    return fulldir


#############################################
def plot_oridirs(analyspar, sesspar, stimpar, extrapar, quantpar, 
                 tr_data, sess_info, figpar=None, savedir=None, 
                 parallel=False):
    """
    plot_oridirs(analyspar, sesspar, stimpar, extrapar, quantpar, 
                 tr_data, sess_info)

    From dictionaries, plots average activity across gabor orientations or 
    visual flow directions per ROI as colormaps, as well as traces across ROIs 
    for a single session and optionally a single quantile. 

    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "o")
            ["datatype"] (str): datatype (e.g., "roi")
        - quantpar (dict) : dictionary with keys of QuantPar namedtuple
        - sess_info (dict): dictionary containing information from each
                            session (one first session used) 
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session

        - tr_data (dict)   : dictionary containing information to plot colormap.
                             unexpected x ori/dir keys are formatted as 
                             [{s}_{od}] for unexp in ["exp", "unexp"]
                                        and od in [0, 45, 90, 135] or 
                                                  ["right", "left"]
            ["n_seqs"] (dict)    : dictionary containing number of seqs for each
                                   unexpected x ori/dir combination under a 
                                   separate key
            ["scale_vals"] (dict): dictionary containing 1D array or list of 
                                   scaling values for each unexpected x ori/dir 
                                   combination under a separate key.
                                   (NaN arrays for combinations with 0 seqs.)
                ["{}_min"] (num): minimum value from corresponding tr_stats 
                                  mean/medians
                ["{}_max"] (num): maximum value from corresponding tr_stats 
                                  mean/medians
            ["roi_sort"] (dict) : dictionary containing 1D arrays or list of 
                                  peak sorting order for each 
                                  unexpected x ori/dir combination under a 
                                  separate key.
                                  (NaN arrays for combinations with 0 seqs.)
            ["roi_me"] (dict)   : dictionary containing trace mean/medians
                                  for each ROI as 2D arrays or nested lists, 
                                  structured as:
                                      ROIs x frames, 
                                  with each unexpected x ori/dir combination 
                                  under a separate key.
                                  (NaN arrays for combinations with 0 seqs.)
            ["stats"] (dict)    : dictionary containing trace mean/medians 
                                  across ROIs in 2D arrays or nested lists, 
                                  structured as: 
                                      stats (me, err) x frames
                                  with each unexpected x ori/dir combination 
                                  under a separate key
                                  (NaN arrays for combinations with 0 seqs.)

    Optional args:
        - figpar (dict)  : dictionary containing the following figure parameter 
                           dictionaries
                           default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
            ["mng"]  (dict): dictionary with parameters to manage matplotlib
        - savedir (Path) : path of directory in which to save plots.
                           default: None
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
                           default: False
    """

    datatype = extrapar["datatype"]
    if datatype != "roi":
        raise NotImplementedError("Function only implemented for roi datatype.")

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    comm_info = {"analyspar": analyspar,
                 "sesspar"  : sesspar,
                 "stimpar"  : stimpar,
                 "extrapar" : extrapar,
                 "quantpar" : quantpar,
                 "sess_info": sess_info,
                 "tr_data" : tr_data,
                 "figpar"   : figpar,
                 }

    if "roi_me" in tr_data.keys():
        plot_oridir_colormaps(
            savedir=savedir, parallel=parallel, **comm_info)

    # if "stats" in tr_data.keys():
    #     plot_oridir_traces(savedir=savedir, **comm_info)

