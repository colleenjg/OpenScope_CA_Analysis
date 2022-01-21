"""
gen_analysis_plots.py

This script contains functions to plot results of ROI and running analyses on 
specific sessions (gen_analys.py) from dictionaries.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import copy
import logging
import warnings
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from util import file_util, gen_util, logger_util, plot_util
from sess_util import sess_gen_util, sess_plot_util, sess_str_util

logger = logging.getLogger(__name__)

# skip tight layout warning
warnings.filterwarnings("ignore", message="This figure includes*")


#############################################
def plot_from_dict(dict_path, plt_bkend=None, fontdir=None, parallel=False, 
                   datetime=True, overwrite=False):
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
        - parallel (bool) : if True, some of the plotting is parallelized 
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

    # 3. Plot magnitude of change in dF/F area from first to last quantile of 
    # unexpected vs no unexpected sequences, for each session
    elif analysis == "m": # mag
        plot_mag_change(figpar=figpar, savedir=savedir, **info)

    # 4. Plot autocorrelations
    elif analysis == "a": # autocorr
        plot_autocorr(figpar=figpar, savedir=savedir, **info)

    else:
        warnings.warn(f"No plotting function for analysis {analysis}", 
            category=UserWarning, stacklevel=1)

    plot_util.cond_close_figs()
    

#############################################
def plot_full_traces(analyspar, sesspar, extrapar, sess_info, trace_info, 
                     roi_tr=None, figpar=None, savedir=None):
    """
    plot_full_traces(analyspar, sesspar, extrapar, sess_info, trace_info)

    From dictionaries, plots full traces for each session in a separate subplot.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict)  : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)    : dictionary with keys of SessPar namedtuple
        - extrapar (dict)   : dictionary containing additional analysis 
                              parameters
            ["analysis"] (str): analysis type (e.g., "f")
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
                                          (me/err if datatype is "roi" x)
                                          frames
            ["all_edges"] (list)    : frame edge values for each parameter, 
                                      structured as sess x block x 
                                                    edges ([start, end])
            ["all_pars"] (list)     : stimulus parameter strings structured as 
                                                    sess x block
                
    Optional args:
        - roi_tr (list): trace values for each ROI, structured as 
                         sess x ROI x frames
                         default: None
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
    
    Returns:
        - fulldir (str) : final path of the directory in which the figure is 
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
    nroi_strs = sess_str_util.get_nroi_strs(sess_info, empty=(datatype!="roi")) 
    n_sess = len(mouse_ns)

    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)
    figpar["init"]["subplot_wid"] = 10
    figpar["init"]["ncols"] = n_sess
    if datatype == "roi":
        figpar["save"]["fig_ext"] = "jpg"
        figpar["init"]["sharex"] = False
        figpar["init"]["sharey"] = False
        # set subplot ratios and removes extra space between plots vertically
        gs = {"height_ratios": [5, 1], "hspace": 0.1} 
        n_rows = 2
        if roi_tr is None:
            raise ValueError("Cannot plot data as ROI traces are missing "
                "(not recorded in dictionary, likely due to size).")
    else:
        gs = None
        n_rows = 1

    if datatype == "roi" and not figpar["save"]["save_fig"]:
        warnings.warn("Figure plotting is being skipped. Since full ROI traces "
            "are not saved to dictionary, to actually plot traces, analysis "
            "will have to be rerun with 'save_fig' set to True.", 
            stacklevel=1)

    fig, ax = plot_util.init_fig(n_sess*n_rows, gs=gs, **figpar["init"])

    label_height = 0.8
    if datatype == "roi":
        fig.subplots_adjust(top=0.92) # remove extra white space at top
        label_height = 0.55
    for i in range(n_sess):
        title = (f"Mouse {mouse_ns[i]} (sess {sess_ns[i]}, {lines[i]} "
            f"{planes[i]}{dendstr_pr}{nroi_strs[i]})")
        sub_axs = ax[:, i]
        sub_axs[0].set_title(title, y=1.02)
        if datatype == "roi":
            xran = range(len(trace_info["all_tr"][i][1]))   
            # each ROI (top subplot)
            plot_util.plot_sep_data(sub_axs[0], np.asarray(roi_tr[i]), 0.1)
            sess_plot_util.add_axislabels(
                sub_axs[0], fluor=analyspar["fluor"], scale=True, 
                datatype=datatype, x_ax="")
            
            # average across ROIs (bottom subplot)
            av_tr = np.asarray(trace_info["all_tr"][i])
            subtitle = u"{} across ROIs".format(statstr_pr)
            plot_util.plot_traces(
                sub_axs[1], xran, av_tr[0], av_tr[1:], lw=0.2, xticks="auto",
                title=subtitle
                )
        else:
            xran = range(len(trace_info["all_tr"][i]))
            run_tr = np.asarray(trace_info["all_tr"][i])
            sub_axs[0].plot(run_tr, lw=0.2)
        for b, block in enumerate(trace_info["all_edges"][i]):
            # all block labels to the lower plot
            plot_util.add_labels(
                sub_axs[-1], trace_info["all_pars"][i][b], np.mean(block), 
                label_height, color="k")
            sess_plot_util.add_axislabels(
                sub_axs[-1], fluor=analyspar["fluor"], datatype=datatype, 
                x_ax="")
            plot_util.remove_ticks(sub_axs[-1], True, False)
            plot_util.remove_graph_bars(sub_axs[-1], bars="horiz")
            # add lines to both plots
            for r in range(n_rows):
                plot_util.add_bars(sub_axs[r], bars=block)
                
    if savedir is None:
        savedir = Path(
            figpar["dirs"][datatype], 
            figpar["dirs"]["full"])

    y = 1 if datatype == "run" else 0.98
    fig.suptitle("Full traces across sessions", fontsize="xx-large", y=y)

    savename = f"{datatype}_tr_{sessstr}{dendstr}"
    fulldir = plot_util.savefig(
        fig, savename, savedir, dpi=400, **figpar["save"])


    return fulldir, savename


#############################################
def plot_traces_by_qu_unexp_sess(analyspar, sesspar, stimpar, extrapar, 
                                quantpar, sess_info, trace_stats, figpar=None, 
                                savedir=None, modif=False):
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
            if extrapar["datatype"] == "roi":
            ["nrois"] (list)      : number of ROIs in session

        - trace_stats (dict): dictionary containing trace stats information
            ["xrans"] (list)           : time values for the frames, for each 
                                         session
            ["all_stats"] (list)       : list of 4D arrays or lists of trace 
                                         data statistics across ROIs for each
                                         session, structured as:
                                            sess x unexp x quantiles x
                                            stats (me, err) x frames
            ["all_counts"] (array-like): number of sequences, structured as:
                                                sess x unexp x quantiles
                
    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
        - modif (bool) : if True, modified (slimmed-down) plots are created
                         instead
                         default: False
    
    Returns:
        - fulldir (str) : final path of the directory in which the figure is 
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
    nroi_strs = sess_str_util.get_nroi_strs(sess_info, empty=(datatype!="roi")) 

    n_sess = len(mouse_ns)

    xrans      = [np.asarray(xran) for xran in trace_stats["xrans"]]
    all_stats  = [np.asarray(sessst) for sessst in trace_stats["all_stats"]]
    all_counts = trace_stats["all_counts"]

    cols, lab_cols = sess_plot_util.get_quant_cols(quantpar["n_quants"])
    alpha = np.min([0.4, 0.8 / quantpar["n_quants"]])

    unexps = ["exp", "unexp"]
    n = 6
    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    
    fig, ax = plot_util.init_fig(n_sess, **figpar["init"])
    for i in range(n_sess):
        sub_ax = plot_util.get_subax(ax, i)
        for s, [col, leg_ext] in enumerate(zip(cols, unexps)):
            for q, qu_idx in enumerate(quantpar["qu_idx"]):
                qu_lab = ""
                if quantpar["n_quants"] > 1:
                    qu_lab = "{} ".format(sess_str_util.quantile_str(
                        qu_idx, quantpar["n_quants"], str_type="print"
                        ))
                if modif:
                    line = "2/3" if "23" in lines[i] else "5"
                    plane = "somata" if "soma" in planes[i] else "dendrites"
                    title = f"M{mouse_ns[i]} - layer {line} {plane}{dendstr_pr}"
                    leg = f"{qu_lab}{leg_ext}" if i == 0 else None
                    y_ax = None if i == 0 else ""

                else:
                    title=(f"Mouse {mouse_ns[i]} - {stimstr_pr}, " 
                        u"{}\n".format(statstr_pr) + f"across {dimstr} (sess "
                        f"{sess_ns[i]}, {lines[i]} {planes[i]}{dendstr_pr}"
                        f"{nroi_strs[i]})")
                    leg = f"{qu_lab}{leg_ext} ({all_counts[i][s][q]})"
                    y_ax = None

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
                                figpar=None, savedir=None, modif=False):
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
            if datatype == 
            ["nrois"] (list)      : number of ROIs in session

        - trace_stats (dict): dictionary containing trace stats information
            ["xrans"] (list)           : time values for the 2p frames for each 
                                         session
            ["all_stats"] (list)       : list of 4D arrays or lists of trace 
                                         data statistics across ROIs for each 
                                         session, structured as:
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
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None   
        - modif (bool) : if True, modified (slimmed-down) plots are created
                         instead
                         default: False
    
    Returns:
        - fulldir (str) : final path of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
        - savename (str): name under which the figure is saved
    """
    analyspar["dend"] = None
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
     
    basestr = sess_str_util.base_par_str(trace_stats["baseline"])
    basestr_pr = sess_str_util.base_par_str(trace_stats["baseline"], "print")

    datatype = extrapar["datatype"]
    dimstr = sess_str_util.datatype_dim_str(datatype)

    # extract some info from sess_info
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_ns, sess_ns, lines, planes] = [sess_info[key] for key in keys]
    nroi_strs = sess_str_util.get_nroi_strs(sess_info, empty=(datatype!="roi")) 

    n_sess = len(mouse_ns)

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
    
    # plot unexp_lens default values
    if stimpar["stimtype"] == "gabors":
        DEFAULT_UNEXP_LEN = [3.0, 4.5, 6.0]
        if stimpar["gabfr"] not in ["any", "all"]:
            offset = sess_str_util.gabfr_nbrs(stimpar["gabfr"])
    else:
        DEFAULT_UNEXP_LEN = [2.0, 3.0, 4.0]
    
    offset = 0
    unexp_lab, len_ext = "", ""
    unexp_lens = [[None]] * n_sess
    unexp_len_default = True
    if "unexp_lens" in trace_stats.keys():
        unexp_len_default = False
        unexp_lens = trace_stats["unexp_lens"]
        len_ext = "_bylen"
        
        if stimpar["stimtype"] == "gabors":
            unexp_lens = [
                [sl * 1.5/5 - 0.3 * offset for sl in sls] for sls in unexp_lens
                ]

    inv = 1 if lock == "unexp" else -1
    # RANGE TO PLOT
    if modif:
        st_val = -2.0
        end_val  = 6.0
        n_ticks = int((end_val - st_val) // 2 + 1)
    else:
        n_ticks = 21

    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)
    if modif:
        figpar["init"]["subplot_wid"] = 6.5
    else:
        figpar["init"]["subplot_wid"] *= 2

    fig, ax = plot_util.init_fig(n_sess, **figpar["init"])
    exp_min, exp_max = np.inf, -np.inf
    for i, (stats, counts) in enumerate(zip(all_stats, all_counts)):
        sub_ax = plot_util.get_subax(ax, i)

        # plot expected data
        if exp_stats[i].shape[0] != 1:
            raise ValueError("Expected only one quantile for exp_stats.")

        n_lines = quantpar["n_quants"] * len(unexp_lens[i])
        cols = sess_plot_util.get_quant_cols(n_lines)[0][col_idx]
        if len(cols) < n_lines:
            cols = [None] * n_lines

        if modif:
            line = "2/3" if "23" in lines[i] else "5"
            plane = "somata" if "soma" in planes[i] else "dendrites"
            title = f"M{mouse_ns[i]} - layer {line} {plane}{dendstr_pr}"
            lab = "exp" if i == 0 else None
            y_ax = None if i == 0 else ""

            st, end = 0, len(xrans[i])
            st_vals = list(filter(
                lambda j: xrans[i][j] <= st_val, range(len(xrans[i]))
                ))
            end_vals = list(filter(
                lambda j: xrans[i][j] >= end_val, range(len(xrans[i]))
                ))
            if len(st_vals) != 0:
                st = st_vals[-1]
            if len(end_vals) != 0:
                end = end_vals[0] + 1
            time_slice = slice(st, end)

        else:
            title = (f"Mouse {mouse_ns[i]} - {stimstr_pr}, "
                u"{} ".format(statstr_pr) + f"{lock} locked across {dimstr}"
                f"{basestr_pr}\n(sess {sess_ns[i]}, {lines[i]} {planes[i]}"
                f"{dendstr_pr}{nroi_strs[i]})")
            lab = f"exp (no lock) ({exp_counts[i][0]})"
            y_ax = None
            st = 0
            end = len(xrans[i])
            time_slice = slice(None) # use all

        # add length markers
        use_unexp_lens = unexp_lens[i]
        if unexp_len_default:
            use_unexp_lens = DEFAULT_UNEXP_LEN
        leng_col = sess_plot_util.get_quant_cols(1)[0][col_idx][0]
        for leng in use_unexp_lens:
            if leng is None:
                continue
            edge = leng * inv
            if edge < 0:
                edge = np.max([xrans[i][st], edge])
            elif edge > 0:
                edge = np.min([xrans[i][end - 1], edge])
            plot_util.add_vshade(
                sub_ax, 0, edge, color=leng_col, alpha=0.1)

        sess_plot_util.add_axislabels(
            sub_ax, fluor=analyspar["fluor"], datatype=datatype, y_ax=y_ax
            )
        plot_util.add_bars(sub_ax, hbars=0)
        alpha = np.min([0.4, 0.8 / n_lines])

        if stimpar["stimtype"] == "gabors":
            sess_plot_util.plot_gabfr_pattern(
                sub_ax, xrans[i], offset=offset, bars_omit=[0] + unexp_lens[i]
                )

        plot_util.plot_traces(
            sub_ax, xrans[i][time_slice], exp_stats[i][0][0, time_slice], 
            exp_stats[i][0][1:, time_slice], n_xticks=n_ticks,
            alpha=alpha, label=lab, alpha_line=0.8, color="darkgray", 
            xticks="auto")

        # get expected data range to adjust y lims
        exp_min = np.min([exp_min, np.nanmin(exp_stats[i][0][0])])
        exp_max = np.max([exp_max, np.nanmax(exp_stats[i][0][0])])

        n = 0 # count lines plotted
        for s, unexp_len in enumerate(unexp_lens[i]):
            if unexp_len is not None:
                counts, stats = all_counts[i][s], all_stats[i][s]       
                # remove offset   
                unexp_lab = f"unexp len {unexp_len + 0.3 * offset}"
            else:
                unexp_lab = "unexp" if modif else f"{lock} lock"
            for q, qu_idx in enumerate(quantpar["qu_idx"]):
                qu_lab = ""
                if quantpar["n_quants"] > 1:
                    qu_lab = "{} ".format(sess_str_util.quantile_str(
                        qu_idx, quantpar["n_quants"], str_type="print"
                        ))
                lab = f"{qu_lab}{unexp_lab}"
                if modif:
                    lab = lab if i == 0 else None
                else:
                    lab = f"{lab} ({counts[q]})"
                if n == 2 and cols[n] is None:
                    sub_ax.plot([], []) # to advance the color cycle (past gray)
                plot_util.plot_traces(sub_ax, xrans[i][time_slice], 
                    stats[q][0, time_slice], stats[q][1:, time_slice], title, 
                    alpha=alpha, label=lab, n_xticks=n_ticks, alpha_line=0.8, 
                    color=cols[n], xticks="auto")
                n += 1
            if unexp_len is not None:
                plot_util.add_bars(
                    sub_ax, hbars=unexp_len, color=sub_ax.lines[-1].get_color(), 
                    alpha=1)
    
    plot_util.turn_off_extra(ax, n_sess)

    if savedir is None:
        savedir = Path(
            figpar["dirs"][datatype], 
            figpar["dirs"]["unexp_qu"], 
            f"{lock}_lock", basestr.replace("_", ""))

    if not modif:
        if stimpar["stimtype"] == "visflow":
            plot_util.rel_confine_ylims(sub_ax, [exp_min, exp_max], 5)

    qu_str = f"_{quantpar['n_quants']}q"
    if quantpar["n_quants"] == 1:
        qu_str = ""
 
    savename = (f"{datatype}_av_{lock}_lock{len_ext}{basestr}_{sessstr}"
        f"{dendstr}{qu_str}")
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar["save"])

    return fulldir, savename


############################################
def plot_mag_change(analyspar, sesspar, stimpar, extrapar, permpar, quantpar, 
                    sess_info, mags, figpar=None, savedir=None):
    """
    plot_mag_change(analyspar, sesspar, stimpar, extrapar, permpar, quantpar, 
                    sess_info, mags) 

    From dictionaries, plots magnitude of change in unexpected and expected
    responses across quantiles.

    Returns figure name and save directory path.

    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "m")
            ["datatype"] (str): datatype (e.g., "run", "roi")
            ["seed"]     (int): seed value used
        - permpar (dict)  : dictionary with keys of PermPar namedtuple 
        - quantpar (dict) : dictionary with keys of QuantPar namedtuple
        - roigrppar (dict): dictionary with keys of RoiGrpPar namedtuple
        - sess_info (dict): dictionary containing information from each
                            session 
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session

        - mags (dict)     : dictionary containing magnitude data to plot
            ["L2"] (array-like)    : nested list containing L2 norms, 
                                     structured as: 
                                         sess x scaling x unexp
            ["L2_sig"] (list)      : L2 significance results for each session 
                                         ("hi", "lo" or "no")
            ["mag_sig"] (list)     : magnitude significance results for each 
                                     session 
                                         ("hi", "lo" or "no")
            ["mag_st"] (array-like): array or nested list containing magnitude 
                                     stats across ROIs, structured as: 
                                         sess x scaling x unexp x stats

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
        - fulldir (str) : final path of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
        - savename (str): name under which the figure is saved
    """
    
    sessstr_pr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["visflow_dir"], stimpar["visflow_size"], stimpar["gabk"], 
        "print")
    statstr_pr = sess_str_util.stat_par_str(
        analyspar["stats"], analyspar["error"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")
        
    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["visflow_dir"],stimpar["visflow_size"], stimpar["gabk"]) 
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])
     
    datatype = extrapar["datatype"]
    dimstr = sess_str_util.datatype_dim_str(datatype)
    
    # extract some info from sess_info
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_ns, sess_ns, lines, planes] = [sess_info[key] for key in keys]
    nroi_strs = sess_str_util.get_nroi_strs(
        sess_info, empty=(datatype!="roi"), style="par"
        )    

    n_sess = len(mouse_ns)

    qu_ns = [gen_util.pos_idx(q, quantpar["n_quants"]) + 1 
        for q in quantpar["qu_idx"]]
    if len(qu_ns) != 2:
        raise ValueError(f"Expected 2 quantiles, not {len(qu_ns)}.")
    
    mag_st = np.asarray(mags["mag_st"])

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()
 
    figpar["init"]["subplot_wid"] *= n_sess/2.0
    
    scales = [False, True]

    # get plot elements
    barw = 0.75
    # scaling strings for printing and filenames
    leg = ["exp", "unexp"]    
    cent, bar_pos, xlims = plot_util.get_barplot_xpos(n_sess, len(leg), barw)   
    title = (u"Magnitude ({}) of difference in activity".format(statstr_pr) +
        f"\nbetween Q{qu_ns[0]} and {qu_ns[1]} across {dimstr} "
        f"\n({sessstr_pr})")
    labels = [f"Mouse {mouse_ns[i]} sess {sess_ns[i]},\n {lines[i]} {planes[i]}"
        f"{dendstr_pr}{nroi_strs[i]}" for i in range(n_sess)]

    figs, axs = [], []
    for sc, scale in enumerate(scales):
        scalestr_pr = sess_str_util.scale_par_str(scale, "print")
        fig, ax = plot_util.init_fig(1, **figpar["init"])
        figs.append(fig)
        axs.append(ax)
        sub_ax = ax[0, 0]
        # always set ticks (even again) before setting labels
        sub_ax.set_xticks(cent)
        sub_ax.set_xticklabels(labels)
        title_scale = u"{}{}".format(title, scalestr_pr)
        sess_plot_util.add_axislabels(
            sub_ax, fluor=analyspar["fluor"], area=True, scale=scale, x_ax="", 
            datatype=datatype)
        for s, lab in enumerate(leg):
            xpos = list(zip(*bar_pos))[s]
            plot_util.plot_bars(
                sub_ax, xpos, mag_st[:, sc, s, 0], err=mag_st[:, sc, s, 1:], 
                width=barw, xlims=xlims, xticks="None", label=lab, capsize=4,
                title=title_scale, hline=0)
    
    # add significance markers
    for i in range(n_sess):
        signif = mags["mag_sig"][i]
        if signif in ["hi", "lo"]:
            xpos = bar_pos[i]
            for sc, (ax, scale) in enumerate(zip(axs, scales)):
                yval = mag_st[i, sc, :, 0]
                yerr = mag_st[i, sc, :, 1:]
                plot_util.plot_barplot_signif(ax[0, 0], xpos, yval, yerr)
    
    plot_util.turn_off_extra(ax, n_sess)

   # figure directory
    if savedir is None:
        savedir = Path(
            figpar["dirs"][datatype], 
            figpar["dirs"]["unexp_qu"], 
            figpar["dirs"]["mags"])
    
    log_dir = False
    for i, (fig, scale) in enumerate(zip(figs, scales)):
        if i == len(figs) - 1:
            log_dir = True
        scalestr = sess_str_util.scale_par_str(scale)
        savename = f"{datatype}_mag_diff_{sessstr}{dendstr}"
        savename_full = f"{savename}{scalestr}"
        fulldir = plot_util.savefig(
            fig, savename_full, savedir, log_dir=log_dir, ** figpar["save"])

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
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None
    
    Returns:
        - fulldir (str) : final path of the directory in which the figure is 
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
        if autocorrpar["byitem"]:
            title_str = u"{}\nautocorrelation".format(fluorstr_pr)
        else:
            title_str = "\nautocorr. acr. ROIs" .format(fluorstr_pr)

    elif datatype == "run":
        datastr = sess_str_util.datatype_par_str(datatype)
        title_str = u"\n{} autocorrelation".format(datastr)

    if stimpar["stimtype"] == "gabors":
        seq_bars = [-1.5, 1.5] # light lines
    else:
        seq_bars = [-1.0, 1.0] # light lines

    # extract some info from sess_info
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_ns, sess_ns, lines, planes] = [sess_info[key] for key in keys]
    nroi_strs = sess_str_util.get_nroi_strs(sess_info, empty=(datatype!="roi")) 

    n_sess = len(mouse_ns)

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
        title = (f"Mouse {mouse_ns[i]} - {stimstr_pr}, "
            u"{} ".format(statstr_pr) + f"{title_str} (sess "
            f"{sess_ns[i]}, {lines[i]} {planes[i]}{dendstr_pr}{nroi_strs[i]})")
        # transpose to ROI/lag x stats x series
        sess_stats = stats[i].transpose(1, 0, 2) 
        for s, sub_stats in enumerate(sess_stats):
            lab = None
            if not autocorrpar["byitem"]:
                lab = ["actual lag", "10x lag"][s]

            plot_util.plot_traces(
                sub_ax, xrans[i], sub_stats[0], sub_stats[1:], xticks=xticks, 
                yticks=yticks, alpha=0.2, label=lab)

        plot_util.add_bars(sub_ax, hbars=seq_bars)
        sub_ax.set_ylim([0, 1])
        sub_ax.set_title(title, y=1.02)
        if sub_ax.is_last_row():
            sub_ax.set_xlabel("Lag (s)")

    plot_util.turn_off_extra(ax, n_sess)

    if savedir is None:
        savedir = Path(
            figpar["dirs"][datatype], 
            figpar["dirs"]["autocorr"])

    savename = (f"{datatype}_autocorr{byitemstr}_{sessstr}{dendstr}")

    fulldir = plot_util.savefig(fig, savename, savedir, **figpar["save"])

    return fulldir, savename
    
