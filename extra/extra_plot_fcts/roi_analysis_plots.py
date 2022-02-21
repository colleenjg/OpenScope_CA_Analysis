"""
roi_analysis_plots.py

This script contains functions to plot results of ROI analyses on specific
sessions (roi_analys.py) from dictionaries.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import copy
import warnings
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import scipy.stats as st

from util import file_util, gen_util, logger_util, plot_util
from sess_util import sess_plot_util, sess_str_util
from extra_plot_fcts import gen_analysis_plots as gen_plots


TAB = "    "

# skip tight layout warning
warnings.filterwarnings("ignore", message="This figure includes*")


logger = logger_util.get_module_logger(name=__name__)


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
    
    dict_path = Path(dict_path)

    info = file_util.loadfile(dict_path)
    savedir = dict_path.parent

    analysis = info["extrapar"]["analysis"]

    # 0. Plots the full traces for each session
    if analysis == "f": # full traces
        gen_plots.plot_full_traces(figpar=figpar, savedir=savedir, **info)

    # 1. Plot average traces by quantile x unexpected for each session 
    if analysis == "t": # traces
        gen_plots.plot_traces_by_qu_unexp_sess(
            figpar=figpar, savedir=savedir, **info)

    # 2. Plot average traces by quantile, locked to unexpected for each session 
    elif analysis == "l": # unexpected locked traces
        gen_plots.plot_traces_by_qu_lock_sess(
            figpar=figpar, savedir=savedir, **info)

    # 3. Plot magnitude of change in dF/F area from first to last quantile of 
    # unexpected vs expected sequences, for each session
    elif analysis == "m": # mag
        gen_plots.plot_mag_change(figpar=figpar, savedir=savedir, **info)

    # 4. Plot autocorrelations
    elif analysis == "a": # autocorr
        gen_plots.plot_autocorr(figpar=figpar, savedir=savedir, **info)
    
    # 5. Plots: a) trace areas by quantile, b) average traces, c) trace areas 
    # by suprise for first vs last quantile, for each ROI group, for each 
    # session
    elif analysis == "g": # roi_grps_ch
        plot_rois_by_grp(figpar=figpar, savedir=savedir, **info)

    # 6. Plot colormaps and traces for orientations/directions
    elif analysis == "o": # colormaps
        plot_oridirs(figpar=figpar, savedir=savedir, parallel=parallel, **info)

    # 7. Plot orientation tuning curves for ROIs
    elif analysis == "c": # tuning curves
        plot_tune_curves(
            figpar=figpar, savedir=savedir, parallel=parallel, plot_tc=plot_tc, 
            **info)
    
    # 8. Plots ROI responses for location and mean gabor orientations
    elif analysis == "p": # location orientation resp
        plot_loc_ori_resp(figpar=figpar, savedir=savedir, **info)

    # 9. Plots trial trajectories in 2 principal components
    elif analysis == "v": # PCs
        plot_trial_pc_traj(figpar=figpar, savedir=savedir, **info)

    else:
        warnings.warn(f"No plotting function for analysis {analysis}", 
            category=UserWarning, stacklevel=1)

    plot_util.cond_close_figs()


#############################################
def plot_roi_areas_by_grp_qu(analyspar, sesspar, stimpar, extrapar, permpar, 
                             quantpar, roigrppar, sess_info, roi_grps, 
                             figpar=None, savedir=None):
    """
    plot_roi_areas_by_grp_qu(analyspar, sesspar, stimpar, extrapar, permpar, 
                             quantpar, roigrppar, sess_info, roi_grps)

    From dictionaries, plots average integrated unexpected, expected or 
    difference between unexpected and expected activity per group of ROIs 
    showing significant unexpected response in first and/or last quantile. Each 
    session is in a different plot.

    Returns figure name and save directory path.

    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "g")
            ["datatype"] (str): datatype (e.g., "roi")
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

        - roi_grps (dict) : dictionary containing ROI grps information
            ["all_roi_grps"] (list): nested lists containing ROI numbers 
                                     included in each group, structured as 
                                     follows:
                                         if sets of groups are passed: 
                                             session x set x roi_grp
                                         if one group is passed: 
                                             session x roi_grp
            ["grp_names"] (list)   : list of names of the ROI groups in roi grp 
                                     lists (order preserved)
            ["grp_st"] (array-like): nested list or array of group stats 
                                     (mean/median, error) across ROIs, 
                                     structured as:
                                         session x quantile x grp x stat
            ["grp_ns"] (array-like): nested list of group ns, structured as: 
                                         session x grp

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


    opstr_pr = sess_str_util.op_par_str(
        roigrppar["plot_vals"], roigrppar["op"], str_type="print")
    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["visflow_dir"], stimpar["visflow_size"], 
        stimpar["gabk"], "print")
    statstr_pr = sess_str_util.stat_par_str(
        analyspar["stats"], analyspar["error"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")

    opstr = sess_str_util.op_par_str(roigrppar["plot_vals"], roigrppar["op"])
    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["visflow_dir"],stimpar["visflow_size"], stimpar["gabk"]) 
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])

    datatype = extrapar["datatype"]
    if datatype != "roi":
        raise NotImplementedError("Function only implemented for roi datatype.")
    dimstr = sess_str_util.datatype_dim_str(datatype)

    # extract some info from sess_info
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_ns, sess_ns, lines, planes] = [sess_info[key] for key in keys]
    nroi_strs = sess_str_util.get_nroi_strs(sess_info) 

    n_sess = len(mouse_ns)

    grp_st = np.asarray(roi_grps["grp_st"])
    grp_ns = np.asarray(roi_grps["grp_ns"])

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    fig, ax = plot_util.init_fig(n_sess, **figpar["init"])
    for i, sess_st in enumerate(grp_st):
        sub_ax = plot_util.get_subax(ax, i)
        for g, g_n in enumerate(grp_ns[i]):
            grp_name = roi_grps["grp_names"][g].replace("_", " ")
            if not grp_name.endswith("change"):
                grp_name = f"{grp_name}." # for abbreviations
            leg = f"{grp_name} ({g_n})"
            plot_util.plot_errorbars(
                sub_ax, y=sess_st[:, g, 0], err=sess_st[:, g, 1:], label=leg, 
                xticks="auto")

        title=(f"Mouse {mouse_ns[i]} - {stimstr_pr}, "
            u"{} ".format(statstr_pr) + f"\nacross {dimstr} {opstr_pr} "
            f"seqs \n(sess {sess_ns[i]}, {lines[i]} {planes[i]}{dendstr_pr}, "
            f"{permpar['tails']} tail{nroi_strs[i]})")
        
        sess_plot_util.add_axislabels(
            sub_ax, fluor=analyspar["fluor"], area=True, x_ax="Quantiles", 
            datatype=datatype)
        sub_ax.set_title(title)

    plot_util.turn_off_extra(ax, n_sess)

    if savedir is None:
        savedir = Path(
            figpar["dirs"][datatype], 
            figpar["dirs"]["unexp_qu"], 
            figpar["dirs"]["grped"])
    
    savename = (f"{datatype}_{sessstr}{dendstr}_grps_{opstr}_" + 
        f"{quantpar['n_quants']}q_{permpar['tails']}tail")

    fulldir = plot_util.savefig(fig, savename, savedir, **figpar["save"])

    return fulldir, savename


#############################################
def plot_roi_traces_by_grp(analyspar, sesspar, stimpar, extrapar, permpar, 
                           quantpar, roigrppar, sess_info, roi_grps, 
                           figpar=None, savedir=None):
    """
    plot_roi_traces_by_grp(analyspar, sesspar, stimpar, extrapar, permpar, 
                           quantpar, roigrppar, sess_info, roi_grps)

    From dictionaries, plots ROI traces by group across unexpected, expected or 
    difference between unexpected and expected activity per quantile 
    (first/last) with each group in a separate subplot and each session in a 
    different figure.

    Returns save directory path.

    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "g")
            ["datatype"] (str): datatype (e.g., "roi")
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

        - roi_grps (dict) : dictionary containing ROI groups:
            ["all_roi_grps"] (list): nested lists containing ROI numbers 
                                     included in each group, structured 
                                     as follows:
                                         if sets of groups are passed: 
                                             session x set x roi_grp
                                         if one group is passed: 
                                             session x roi_grp
            ["grp_names"] (list)   : list of names of the ROI groups in 
                                     ROI grp lists (order preserved)
            ["xrans"] (list)       : time values for the traces, for each 
                                     session
            ["trace_stats"] (list) : nested lists or array of statistics 
                                          across ROIs for ROI groups 
                                          structured as:
                                              sess x qu x ROI grp x 
                                              stats x frame
    
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

    opstr_pr = sess_str_util.op_par_str(
        roigrppar["plot_vals"], roigrppar["op"], "print")
    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["visflow_dir"], stimpar["visflow_size"], 
        stimpar["gabk"], "print")
    statstr_pr = sess_str_util.stat_par_str(
        analyspar["stats"], analyspar["error"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")

    opstr = sess_str_util.op_par_str(roigrppar["plot_vals"], roigrppar["op"])
    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["visflow_dir"],stimpar["visflow_size"], stimpar["gabk"]) 
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])

    datatype = extrapar["datatype"]
    if datatype != "roi":
        raise NotImplementedError("Function only implemented for roi datatype.")
    dimstr = sess_str_util.datatype_dim_str(datatype)

    # extract some info from sess_info
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_ns, sess_ns, lines, planes] = [sess_info[key] for key in keys]
    nroi_strs = sess_str_util.get_nroi_strs(sess_info) 

    n_sess = len(mouse_ns)

    xrans = [np.asarray(xran) for xran in roi_grps["xrans"]]
    
    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    # figure directories
    if savedir is None:
        savedir = Path(
            figpar["dirs"][datatype], 
            figpar["dirs"]["unexp_qu"], 
            figpar["dirs"]["grped"])

    if figpar["save"]["use_dt"] is None:
        figpar = copy.deepcopy(figpar)
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    log_dir = False
    for i in range(n_sess):
        sess_traces = np.asarray(roi_grps["trace_stats"][i])
        if i == n_sess - 1:
            log_dir = True
        n_grps = len(roi_grps["all_roi_grps"][i])
        fig, ax = plot_util.init_fig(n_grps, **figpar["init"])
        for g, [grp_name, grp_rois] in enumerate(
            zip(roi_grps["grp_names"], roi_grps["all_roi_grps"][i])):
            grp_name = grp_name.replace("_", " ")
            if not grp_name.endswith("change"):
                grp_name = f"{grp_name}." # for abbreviations
            title = f"{grp_name} group (n={len(grp_rois)})"
            sub_ax = plot_util.get_subax(ax, g)
            sess_plot_util.add_axislabels(
                sub_ax, fluor=analyspar["fluor"], datatype=datatype)
            for q, qu_idx in enumerate(quantpar["qu_idx"]):
                qu_lab = sess_str_util.quantile_str(
                    qu_idx, quantpar["n_quants"], str_type="print"
                    )
                plot_util.plot_traces(
                    sub_ax, xrans[i], sess_traces[q, g, 0], 
                    sess_traces[q, g, 1:], title=title, 
                    alpha=0.8/len(quantpar["qu_idx"]), 
                    label=qu_lab, xticks="auto")

        plot_util.turn_off_extra(ax, n_grps)

        if stimpar["stimtype"] == "gabors": 
            sess_plot_util.plot_labels(
                ax, stimpar["gabfr"], roigrppar["plot_vals"], 
                pre=stimpar["pre"], post=stimpar["post"], 
                sharey=figpar["init"]["sharey"])

        suptitle = (f"Mouse {mouse_ns[i]} - {stimstr_pr}, "
            u"{} ".format(statstr_pr) + f"across {dimstr} {opstr_pr} seqs "
            f"\n(sess {sess_ns[i]}, {lines[i]} {planes[i]}{dendstr_pr}, "
            f"{permpar['tails']} tail{nroi_strs[i]})")

        fig.suptitle(suptitle, y=1)

        savename = (f"{datatype}_tr_m{mouse_ns[i]}_{sessstr}{dendstr}_"
            f"grps_{opstr}_{quantpar['n_quants']}q_{permpar['tails']}tail")
        
        fulldir = plot_util.savefig(
            fig, savename, savedir, log_dir=log_dir, **figpar["save"])

    return fulldir


#############################################
def plot_roi_areas_by_grp(analyspar, sesspar, stimpar, extrapar, permpar, 
                          quantpar, roigrppar, sess_info, roi_grps, 
                          figpar=None, savedir=None):
    """
    plot_roi_areas_by_grp(analyspar, sesspar, stimpar, extrapar, permpar, 
                          quantpar, roigrppar, sess_info, roi_grps)

    From dictionaries, plots ROI traces by group across unexpected, expected or 
    difference between unexpected and expected activity per quantile 
    (first/last) with each group in a separate subplot and each session in a 
    different figure.

    Returns save directory path.

    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "g")
            ["datatype"] (str): datatype (e.g., "roi")
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

        - roi_grps (dict)  : dictionary containing ROI groups:
            ["all_roi_grps"] (list)           : nested lists containing ROI 
                                                numbers included in each group, 
                                                structured as follows:
                                                  if sets of groups are passed: 
                                                      session x set x roi_grp
                                                  if one group is passed: 
                                                      session x roi_grp
            ["grp_names"] (list)              : list of names of the ROI groups  
                                                in ROI grp lists (order 
                                                preserved)
            ["area_stats"] (array-like)       : ROI group stats (mean/median,  
                                                error) across ROIs, structured
                                                as:
                                                  session x quantile x 
                                                  grp x stat
            ["area_stats_scaled"] (array-like): same as "area_stats", but 
                                                with last quantile scaled 
                                                relative to first

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
   
    opstr_pr = sess_str_util.op_par_str(
        roigrppar["plot_vals"], roigrppar["op"], "print")
    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["visflow_dir"], stimpar["visflow_size"], 
        stimpar["gabk"], "print")
    statstr_pr = sess_str_util.stat_par_str(
        analyspar["stats"], analyspar["error"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"], "print")

    opstr = sess_str_util.op_par_str(roigrppar["plot_vals"], roigrppar["op"])
    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["visflow_dir"], stimpar["visflow_size"], stimpar["gabk"]) 
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])

    datatype = extrapar["datatype"]
    if datatype != "roi":
        raise NotImplementedError("Function only implemented for roi datatype.")
    dimstr = sess_str_util.datatype_dim_str(datatype)

    # extract some info from sess_info
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_ns, sess_ns, lines, planes] = [sess_info[key] for key in keys]
    nroi_strs = sess_str_util.get_nroi_strs(sess_info, style="comma")

    n_sess = len(mouse_ns)

    # scaling strings for printing and filenames
    scales = [False, True]
    
    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    figpar["init"]["subplot_wid"] /= 2.0
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    # for spacing the bars on the graph
    barw = 0.75
    _, bar_pos, xlims = plot_util.get_barplot_xpos(
        1, len(quantpar["qu_idx"]), barw, btw_grps=1.5)
    bar_pos = bar_pos[0] # only one grp

    log_dir = False
    for i in range(n_sess):
        if i == n_sess - 1:
            log_dir = True
        n_grps = len(roi_grps["all_roi_grps"][i])
        figs = []
        for scale in scales:
            sc_str    = sess_str_util.scale_par_str(scale)
            fig, ax = plot_util.init_fig(n_grps, **figpar["init"])
            figs.append(fig)
            for g, [grp_name, grp_rois] in enumerate(
                zip(roi_grps["grp_names"], roi_grps["all_roi_grps"][i])):
                grp_name = grp_name.replace("_", " ")
                if not grp_name.endswith("change"):
                    grp_name = f"{grp_name}." # for abbreviations
                title = f"{grp_name} group (n={len(grp_rois)})"
                sub_ax = plot_util.get_subax(ax, g)
                sub_ax.tick_params(labelbottom=False)
                sub_ax.spines["bottom"].set_visible(False)
                sess_plot_util.add_axislabels(
                    sub_ax, area=True, scale=scale, x_ax="", datatype=datatype)
                for q, qu_idx in enumerate(quantpar["qu_idx"]):
                    qu_lab = sess_str_util.quantile_str(
                        qu_idx, quantpar["n_quants"], str_type="print"
                    )
                    vals = roi_grps[f"area_stats{sc_str}"]
                    vals = np.asarray(vals)[i, q, g]
                    plot_util.plot_bars(
                        sub_ax, bar_pos[q], vals[0], vals[1:], title, 
                        alpha=0.5, xticks="None", xlims=xlims, 
                        label=qu_lab, hline=0, width=barw)

            plot_util.turn_off_extra(ax, n_grps)

        suptitle = (f"Mouse {mouse_ns[i]} - {stimstr_pr}, " + 
            u"{}".format(statstr_pr) + f" across {dimstr} " +
            f"{opstr_pr} seqs\n(sess {sess_ns[i]}, {lines[i]} " +
            f"{planes[i]}{dendstr_pr}, {permpar['tails']} tail{nroi_strs[i]})")

        savename = (f"{datatype}_area_m{mouse_ns[i]}_{sessstr}{dendstr}_grps_"
            f"{opstr}_{quantpar['n_quants']}q_{permpar['tails']}tail")
        
        # figure directories
        if savedir is None:
            savedir = Path(
                figpar["dirs"][datatype], 
                figpar["dirs"]["unexp_qu"], 
                figpar["dirs"]["grped"])

        for i, (fig, scale) in enumerate(zip(figs, scales)):
            scale_str    = sess_str_util.scale_par_str(scale)
            scale_str_pr = sess_str_util.scale_par_str(scale, "print")
            fig.suptitle(u"{}{}".format(suptitle, scale_str_pr), y=1)
            full_savename = f"{savename}{scale_str}"
            fulldir = plot_util.savefig(
                fig, full_savename, savedir, log_dir=log_dir, 
                **figpar["save"])

    return fulldir


#############################################
def plot_rois_by_grp(analyspar, sesspar, stimpar, extrapar, permpar, quantpar, 
                     roigrppar, sess_info, roi_grps, figpar=None, savedir=None):
    """
    plot_rois_by_grp(analyspar, sesspar, stimpar, extrapar, permpar, quantpar, 
                     roigrppar, sess_info, roi_grps)

    From dictionaries, plots ROI data by group across unexpected, expected or 
    difference between unexpected and expected activity per quantile 
    (first/last) with each group in a separate subplot and each session in a 
    different figure.

    Two types of ROI data are plotted:
        1. ROI traces, if "trace_stats" is in roi_grps
        2. ROI areas, if "area_stats" is in roi_grps 

    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ["analysis"] (str): analysis type (e.g., "g")
            ["datatype"] (str): datatype (e.g., "roi")
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

        - roi_grps (dict) : dictionary containing ROI groups:
            ["all_roi_grps"] (list)           : nested lists containing ROI  
                                                numbers included in each group, 
                                                structured as follows:
                                                  if sets of groups are passed: 
                                                      session x set x roi_grp
                                                  if one group is passed: 
                                                      session x roi_grp
            ["grp_names"] (list)              : list of names of the ROI groups 
                                                in ROI grp lists (order 
                                                preserved)
            ["grp_st"] (array-like)           : nested list or array of group 
                                                stats (mean/median, error) 
                                                across ROIs, structured as:
                                                    session x quantile x grp x 
                                                    stat
            ["grp_ns"] (array-like)           : nested list of group ns, 
                                                structured as: 
                                                    session x grp
            ["xrans"] (list)                  : time values for the frame 
                                                chunks, for each session
            ["trace_stats"] (array-like)      : array or nested list of 
                                                statistics across ROIs, for ROI 
                                                groups structured as:
                                                    sess x qu x ROI grp x stats 
                                                    x frame
            ["area_stats"] (array-like)       : ROI group stats (mean/median, 
                                                error) across ROIs, structured 
                                                as:
                                                    session x quantile x grp x 
                                                    stat
            ["area_stats_scaled"] (array-like): same as "area_stats", but 
                                                with last quantile scaled
                                                relative to first
    
    Optional args:
        - figpar (dict) : dictionary containing the following figure parameter 
                          dictionaries
                          default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (Path): path of directory in which to save plots.
                          default: None
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
                 "permpar"  : permpar,
                 "quantpar" : quantpar,
                 "roigrppar": roigrppar,
                 "sess_info": sess_info,
                 "roi_grps" : roi_grps,
                 "figpar"   : figpar,
                 }

    if "grp_st" in roi_grps.keys():
        plot_roi_areas_by_grp_qu(savedir=savedir, **comm_info)

    if "trace_stats" in roi_grps.keys():
        plot_roi_traces_by_grp(savedir=savedir, **comm_info)

    if "area_stats" in roi_grps.keys():
        plot_roi_areas_by_grp(savedir=savedir, **comm_info)


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
            ["n_seqs"] (dict): dictionary containing number of segs for each
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
    dimstr = sess_str_util.datatype_dim_str(datatype)

    if savedir is None:
        savedir = Path(
            figpar["dirs"][datatype], 
            figpar["dirs"]["oridir"])

    stimstr_pr = stimstr_pr[:-1] if stimstr_pr[-1] == "s" else stimstr_pr

    # extract some info from dictionaries
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_n, sess_n, line, plane] = [sess_info[key][0] for key in keys]
    nroi_str = sess_str_util.get_nroi_strs(sess_info)[0]

    xran = tr_data["xran"]

    unexps = ["exp", "unexp"]
    if stimpar["stimtype"] == "gabors":
        unexp_labs = unexps
        deg = u"\u00B0"
        oridirs = stimpar["gab_ori"]
        n = 6
    elif stimpar["stimtype"] == "visflow":
        unexp_labs = \
            [f"{unexps[i]} -> {unexps[1-i]}" for i in range(len(unexps))]
        deg = ""
        oridirs = stimpar["visflow_dir"]
        n = 7

    qu_str, qu_str_pr = "", ""
    if quantpar["n_quants"] > 1:
        qu_str_pr = sess_str_util.quantile_str(
            quantpar["qu_idx"], n_quants=quantpar["n_quants"], out_of=True, 
            str_type="print"
            )
        qu_str_pr = f" - {qu_str_pr}"

        qu_str = sess_str_util.quantile_str(
            quantpar["qu_idx"], n_quants=quantpar["n_quants"], str_type="file"
            )
        qu_str = f"_{qu_str}"

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    figpar["init"]["ncols"] = len(oridirs) 
    
    suptitle = (f"Mouse {mouse_n} - {stimstr_pr}, " + 
        u"{} ".format(statstr_pr) + 
        f"across {dimstr}{qu_str_pr}\n(sess {sess_n}, {line} " +
        f"{plane}{dendstr_pr}{nroi_str})")
    savename = (f"{datatype}_tr_m{mouse_n}_sess{sess_n}{qu_str}_{stimstr}_" + 
        f"{plane}{dendstr}")
    
    fig, ax = plot_util.init_fig(len(oridirs), **figpar["init"])
    for o, od in enumerate(oridirs):
        cols = []
        od_str = od
        if stimpar["stimtype"] == "visflow":
            od_str = sess_str_util.dir_par_str(od, str_type="print")
            od_str = od_str.replace("vis. flow ", "")[1:-1] # remove parentheses
        for unexp, unexp_lab in zip(unexps, unexp_labs): 
            sub_ax = plot_util.get_subax(ax, o)
            key = f"{unexp}_{od}"
            stimstr_pr = stimstr_pr.capitalize()
            title_tr = u"{} traces ({}{})".format(stimstr_pr, od_str, deg)
            lab = f"{unexp_lab} (n={tr_data['n_seqs'][key]})"
            sess_plot_util.add_axislabels(sub_ax, datatype=datatype)
            me  = np.asarray(tr_data["stats"][key][0])
            err = np.asarray(tr_data["stats"][key][1:])
            plot_util.plot_traces(
                sub_ax, xran, me, err, title_tr, n_xticks=n, label=lab, 
                xticks="auto")
            cols.append(sub_ax.lines[-1].get_color())
            if stimpar["stimtype"] == "visflow":
                plot_util.add_bars(sub_ax, 0)

    if stimpar["stimtype"] == "gabors":
        sess_plot_util.plot_labels(
            ax, stimpar["gabfr"], cols=cols, pre=stimpar["pre"], 
            post=stimpar["post"], sharey=figpar["init"]["sharey"])

    fig.suptitle(suptitle, y=1)
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar["save"])

    return fulldir


#############################################
def scale_sort_trace_data(tr_data, fig_type="byplot", unexps=["exp", "unexp"], 
                          oridirs=[0, 45, 90, 135]):
    """
    scale_sort_trace_data(tr_data)

    Returns a dictionary containing ROI traces scaled and sorted as 
    specified.

    Required args:        
        - tr_data (dict): dictionary containing information to plot colormap.
                            unexpected x ori/dir keys are formatted as 
                            [{s}_{od}] for unexp in ["exp", "unexp"]
                                           and od in [0, 45, 90, 135] or 
                                                     ["right", "left"]
            ["n_seqs"] (dict)    : dictionary containing number of segs for 
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
                                   unexpected x ori/dir combination under
                                   a separate key.
                                   (NaN arrays for combinations with 0 seqs.)
            ["roi_me"] (dict)    : dictionary containing trace mean/medians
                                   for each ROI as 2D arrays or nested lists, 
                                   structured as:
                                       ROIs x frames, 
                                   with each unexpected x ori/dir combination 
                                   under a separate key. 
                                   (NaN arrays for combinations with 0 seqs.)
            ["xran"] (list)      : time values for the 2p frames

    Optional args:
        - fig_type (str) : how to scale and sort ROIs, 
                               i.e. each plot separately ("byplot"), 
                                   each orientation/direction by its expected 
                                       plot ("byexp"), 
                                   each exp/unexp by the first 
                                       orientation/direction 
                                       ("by0deg" or "byright")
                                   each plot by the first plot ("byfirst")
                           default: "byplot"
        - unexps (list)   : unexpected value names used in keys, ordered
                           default: ["exp", "unexp"]
        - oridirs (list) : orientation/direction value names used in keys,
                           ordered
                           default: [0, 45, 90, 135]
    
    Returns:
        - scaled_sort_data_me (dict): dictionary containing scaled and 
                                      sorted trace mean/medians for each ROI as 
                                      2D arrays, structured as:
                                          ROIs x frames, 
                                      with each unexpected x ori/dir combination 
                                      under a separate key, as above
    """

    scaled_sort_data_me = dict()
    scale_vals  = tr_data["scale_vals"]
    roi_sort   = tr_data["roi_sort"]
    for od in oridirs:
        for s in unexps:
            key = f"{s}_{od}"
            me = np.asarray(tr_data["roi_me"][key])
            # mean/median organized as ROI x fr
            if tr_data["n_seqs"][key] == 0: # no data under these criteria
                scaled_sort_data_me[key] = me.T
                continue
            if fig_type == "byplot":
                min_v = np.asarray(scale_vals[f"{key}_min"])
                max_v = np.asarray(scale_vals[f"{key}_max"])
                sort_arg = roi_sort[key]

            elif fig_type == "byexp":
                mins = [scale_vals[f"{sv}_{od}_min"] for sv in unexps]
                maxs = [scale_vals[f"{sv}_{od}_max"] for sv in unexps]
                min_v = np.nanmin(np.asarray(mins), axis=0)
                max_v = np.nanmax(np.asarray(maxs), axis=0)
                idx = 0
                # find first exp/unexp plot with data
                while tr_data["n_seqs"][f"{unexps[idx]}_{od}"] == 0:
                    idx += 1
                sort_arg = roi_sort[f"{unexps[idx]}_{od}"]
            
            elif fig_type in ["by0deg", "byright"]:
                mins = [scale_vals[f"{s}_{odv}_min"] for odv in oridirs]
                maxs = [scale_vals[f"{s}_{odv}_max"] for odv in oridirs]
                min_v = np.nanmin(np.asarray(mins), axis=0)
                max_v = np.nanmax(np.asarray(maxs), axis=0)
                idx = 0
                # find first oridir plot with data
                while tr_data["n_seqs"][f"{s}_{oridirs[idx]}"] == 0:
                    idx += 1
                sort_arg = roi_sort[f"{s}_{oridirs[idx]}"]
                
            elif fig_type == "byfir":

                mins = [scale_vals[f"{sv}_{odv}_min"] for sv in unexps 
                    for odv in oridirs]
                maxs = [scale_vals[f"{sv}_{odv}_max"] for sv in unexps 
                    for odv in oridirs]
                min_v = np.nanmin(np.asarray(mins), axis=0)
                max_v = np.nanmax(np.asarray(maxs), axis=0)
                idx_s, idx_od, count = 0, 0, 0
                # find first plot with data (by oridirs, then unexps)
                while tr_data["n_seqs"][
                    f"{unexps[idx_s]}_{oridirs[idx_od]}"] == 0:
                    count += 1
                    idx_od = count % len(oridirs)
                    idx_s  = count // len(oridirs) % len(unexps)
                sort_arg = roi_sort[f"{unexps[idx_s]}_{oridirs[idx_od]}"]

            me_scaled = ((me.T - min_v)/(max_v - min_v))
            scaled_sort_data_me[key] = me_scaled[:, sort_arg]
    
    return scaled_sort_data_me


#############################################
def plot_oridir_colormap(fig_type, analyspar, sesspar, stimpar, quantpar, 
                         tr_data, sess_info, figpar=None, savedir=None, 
                         log_dir=True):
    """
    plot_oridir_colormap(fig_type, analyspar, sesspar, stimpar, quantpar,  
                         tr_data, sess_info)

    From dictionaries, plots average activity across gabor orientations or 
    visual flow directions per ROI for a single session and optionally a single 
    quantile. (Single figure type) 

    Required args:
        - fig_type (str)  : type of figure to plot, i.e., "byplot", "byexp", 
                            "byfir" or "by{}{}" (ori/dir, deg)
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
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
            ["n_seqs"] (dict)    : dictionary containing number of segs for 
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
        - log_dir (bool): if True, figure saving directory is logged
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
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], "roi", "print")

    stimstr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["visflow_dir"], stimpar["visflow_size"], 
        stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], "roi")

    stimstr_pr = stimstr_pr[:-1] if stimstr_pr[-1] == "s" else stimstr_pr

    if savedir is None:
        savedir = Path(
            figpar["dirs"]["roi"], 
            figpar["dirs"]["oridir"])

    # must update matplotlib settings within parallel
    plot_util.manage_mpl(**figpar["mng"])
    cmap = "viridis"

    # extract some info from sess_info (only one session)
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_n, sess_n, line, plane] = [sess_info[key][0] for key in keys]

    unexps = ["exp", "unexp"]
    if stimpar["stimtype"] == "gabors":
        unexp_labs = unexps
        var_name = "orientation"
        deg  = "deg"
        deg_pr = u"\u00B0"
        oridirs = stimpar["gab_ori"]
        n = 6
    elif stimpar["stimtype"] == "visflow":
        unexp_labs = [
            f"{unexps[i]} -> {unexps[1-i]}" for i in range(len(unexps))
            ]
        var_name = "direction"
        deg  = ""
        deg_pr = ""
        oridirs = stimpar["visflow_dir"]
        n = 7
    
    qu_str, qu_str_pr = "", ""
    if quantpar["n_quants"] > 1:
        qu_str_pr = sess_str_util.quantile_str(
            quantpar["qu_idx"], n_quants=quantpar["n_quants"], out_of=True, 
            str_type="print"
            )
        qu_str_pr = f" - {qu_str_pr}"

        qu_str = sess_str_util.quantile_str(
            quantpar["qu_idx"], n_quants=quantpar["n_quants"], str_type="file"
            )
        qu_str = f"_{qu_str}"

    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)
    figpar["init"]["ncols"] = len(oridirs)
    figpar["init"]["sharex"] = True
    
    # plot colormaps
    gentitle = (f"Mouse {mouse_n} - {stimstr_pr}, " 
        u"{} ".format(statstr_pr) + 
        f"across seqs colormaps{qu_str_pr}\n(sess {sess_n}, {line} "
        f"{plane}{dendstr_pr})")
    gen_savename = (f"roi_cm_m{mouse_n}_sess{sess_n}{qu_str}_{stimstr}_"
        f"{plane}{dendstr}")

    if fig_type == "byplot":
        scale_type = "per plot"
        peak_sort  = ""
        figpar["init"]["sharey"] = False
    elif fig_type == "byexp":
        scale_type = f"within {var_name}"
        peak_sort  = f" of {unexps[0]}."
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
    suptitle = f"{gentitle}\n({subtitle})"
    
    # get scaled and sorted ROI mean/medians (ROI x frame)
    scaled_sort_me = scale_sort_trace_data(tr_data, fig_type, unexps, oridirs)
    fig, ax = plot_util.init_fig(len(oridirs) * len(unexps), **figpar["init"])

    xran_edges = [np.min(tr_data["xran"]), np.max(tr_data["xran"])]
    alpha=0.8

    nrois = scaled_sort_me[f"{unexps[0]}_{oridirs[0]}"].shape[1]
    yticks_ev = int(10 * np.max([1, np.ceil(nrois/100)])) # avoid > 10 ticks
    for o, od in enumerate(oridirs):
        od_str = od
        if stimpar["stimtype"] == "visflow":
            od_str = sess_str_util.dir_par_str(od, str_type="print")
            od_str = od_str.replace("vis. flow ", "")[1:-1] # remove parentheses
        for s, (unexp, unexp_lab) in enumerate(zip(unexps, unexp_labs)):    
            sub_ax = ax[s][o]
            key = f"{unexp}_{od}"
            title = u"{} seqs ({}{}) (n={})".format(
                unexp_lab.capitalize(), od_str, deg_pr, tr_data["n_seqs"][key])
            sess_plot_util.add_axislabels(
                sub_ax, fluor=analyspar["fluor"], y_ax="ROIs", datatype="roi")
            im = plot_util.plot_colormap(
                sub_ax, scaled_sort_me[key], title=title, cmap=cmap,
                xran=xran_edges, n_xticks=n, yticks_ev=yticks_ev, 
                xticks="auto", origin="upper")
            if stimpar["stimtype"] == "visflow":
                plot_util.add_bars(sub_ax, 0, lw=3.5, alpha=alpha)

    for s, unexp in enumerate(unexps):
        sub_ax = ax[s:s+1]
        if stimpar["stimtype"] == "gabors":
            sess_plot_util.plot_labels(
                sub_ax, stimpar["gabfr"], unexp, pre=stimpar["pre"], 
                post=stimpar["post"], sharey=figpar["init"]["sharey"], 
                t_heis=-0.05, omit_empty=False, lw=[5, 3], alpha=alpha)
        
    plot_util.add_colorbar(fig, im, len(oridirs))
    fig.suptitle(suptitle, y=1)

    savename = f"{gen_savename}_{fig_type}"
    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=log_dir, **figpar["save"])
    
    plot_util.cond_close_figs()
    
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
            ["xran"] (list)      : time values for the 2p frames

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
    figpar["save"]["fig_ext"] = "png" # svg too big

    fig_types  = ["byplot", "byexp", f"by{oridirs[0]}{deg}", "byfir"]
    fig_last = len(fig_types) - 1
    
    # optionally runs in parallel
    if parallel:
        n_jobs = gen_util.get_n_jobs(len(fig_types))
        with gen_util.ParallelLogging():
            fulldirs = Parallel(n_jobs=n_jobs)(
                delayed(plot_oridir_colormap)
                (fig_type, analyspar, sesspar, stimpar, quantpar, 
                tr_data=tr_data, sess_info=sess_info, figpar=figpar, 
                savedir=savedir, log_dir=(f == fig_last)) 
                for f, fig_type in enumerate(fig_types)) 
        fulldir = fulldirs[-1]
    else:
        for f, fig_type in enumerate(fig_types):
            log_dir = (f == fig_last)
            fulldir = plot_oridir_colormap(
                fig_type, analyspar, sesspar, stimpar, quantpar, 
                tr_data=tr_data, sess_info=sess_info, figpar=figpar, 
                savedir=savedir, log_dir=log_dir)

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
            ["n_seqs"] (dict)    : dictionary containing number of segs for each
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
            ["xran"] (list)     : time values for the 2p frames

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
                 "tr_data"  : tr_data,
                 "figpar"   : figpar,
                 }

    if "roi_me" in tr_data.keys():
        plot_oridir_colormaps(
            savedir=savedir, parallel=parallel, **comm_info)

    if "stats" in tr_data.keys():
        plot_oridir_traces(savedir=savedir, **comm_info)


#############################################
def plot_vm_estim_analysis(subax_col, xran, gab_oris, gab_data, gab_vm_pars, 
                           gab_hist_pars, title_str="", fluor="dff"):
    """
    plot_vm_estim_analysis(subax_col, xran, gab_oris, gab_data, gab_vm_pars, 
                           gab_hist_pars)

    Plots orientation fluorescence data for a specific ROI, as well as ROI 
    orientation tuning curves, estimated by fitting a von Mises distribution.

    Required args:
        - subax_col (plt Axis)    : axis column
        - xran (array-like)       : x values
        - gab_oris (list)         : list of orientation values corresponding to 
                                    the gab_data
        - gab_data (list)         : list of mean integrated fluorescence data 
                                    per orientation
        - gab_vm_pars (array-like): array of Von Mises parameters 
                                    (kappa, mean, scale)
        - gab_hist_pars (list)  : parameters used to convert gab_data to 
                                  histogram values (sub, mult) used in Von 
                                  Mises parameter estimation (sub, mult)
    
    Optional args:
        - title_str (str)   : main title
                              default: ""
        - fluor (str)       : fluorescence type
                              default: "dff"
    """

    deg = u"\u00B0"

    # Top: Von Mises fits
    vm_fit = st.vonmises.pdf(np.radians(xran), *gab_vm_pars)
    subax_col[0].plot(xran, vm_fit)
    subax_col[0].set_title(title_str, y=1.02)
    subax_col[0].set_ylabel("Probability density")
    col = subax_col[0].lines[-1].get_color()
    
    # Mid: actual data
    subax_col[1].plot(gab_oris, gab_data, marker=".", lw=0, alpha=0.3, 
                       color=col)
    subax_col[1].set_title("Mean AUC per orientation", y=1.02)
    y_str = sess_str_util.fluor_par_str(fluor, str_type="print")
    subax_col[1].set_ylabel(u"{} area (mean)".format(y_str))

    # Bottom: data as histogram for fitting
    counts = np.around(
        (np.asarray(gab_data) - gab_hist_pars[0]) * gab_hist_pars[1]
        ).astype(int)
    freq_data = np.repeat(np.asarray(gab_oris), counts)                
    subax_col[2].hist(freq_data, 360, color=col)
    subax_col[2].set_title("Orientation histogram", y=1.02)
    subax_col[2].set_xlabel(u"Orientations ({})".format(deg))
    subax_col[2].set_ylabel("Artificial counts")
    plot_util.set_ticks(subax_col[2], "x", np.min(xran), np.max(xran), 10)


#############################################
def plot_roi_tune_curves(tc_oris, roi_data, n, nrois, seq_info, 
                         roi_vm_pars=None, roi_hist_pars=None, 
                         fluor="dff", comb_gabs=False, gentitle="", 
                         gen_savename="", figpar=None, savedir=None):
    """
    plot_roi_tune_curves(tc_oris, roi_data, n, nrois, seq_info)
    
    Plots orientation fluorescence data for a specific ROI, as well as ROI 
    orientation tuning curves, if provided. 


    Required args:
        - tc_oris (list)        : list of orientation values corresponding to 
                                  the tc_data
                                    unexp x gabor (1 if comb_gabs) x oris
        - roi_data (list)       : list of mean integrated fluorescence data per 
                                  orientation, structured as 
                                    unexp (x gabor (1 if comb_gabs)) x oris
        - n (int)               : ROI number
        - nrois (int)           : total number of ROIs
        - seq_info (list)       : list of strings with info on each group
                                  plotted (2)
    
    Optional args:
        - roi_vm_pars (3D array): array of Von Mises parameters: 
                                    unexp x gabor (1 if comb_gabs) 
                                        x par (kappa, mean, scale)
                                  default: None
        - roi_hist_pars (list)  : parameters used to convert tc_data to 
                                  histogram values (sub, mult) used in Von 
                                  Mises parameter estimation, structured as:
                                    unexp x gabor (1 if comb_gabs) x 
                                    param (sub, mult)
                                  default: None
        - fluor (str)           : fluorescence type
                                  default: "dff"
        - comb_gabs (bool)      : if True, data from all gabors was combined
                                  default: False
        - gentitle (str)        : general title for the plot
                                  default: ""
        - gen_savename (str)    : general title for the plot
                                  default: ""
        - figpar (dict)         : dictionary containing the following figure 
                                  parameter dictionaries
                                  default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
            ["mng"]  (dict): dictionary with parameters to manage matplotlib
        - savedir (Path)        : path of directory in which to save plots.
                                  default: None
    Returns:
        - fulldir (Path)    : final path of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
    """

    flat_oris = [o for gos in tc_oris for oris in gos for o in oris]

    if np.min(flat_oris) < 0:
        raise ValueError("Orientations expected to be positive.")  
    max_val = 180
    if np.max(flat_oris) > max_val:
        max_val = 360
        if np.max(flat_oris) > max_val:
            raise ValueError(
                "Orientations expected to be at most between 0 and 360."
                )
    xran = np.linspace(0, max_val, 360)

    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()
    figpar["init"]["ncols"] = 2
    figpar["init"]["sharex"] = True
    figpar["init"]["sharey"] = False
    figpar["save"]["fig_ext"] = "png" # svg too big

    # must update matplotlib settings within parallel
    plot_util.manage_mpl(**figpar["mng"])

    if savedir is None:
        savedir = Path(
            figpar["dirs"]["roi"], 
            figpar["dirs"]["tune_curv"])
        if roi_vm_pars is not None:
            savedir = savedir.joinpath("vm_estim")

    log_dir = False
    if n == 0:
        log_dir = True
    if (n + 1) % 15 == 0 or (n + 1) == nrois:
        logger.info(f"ROI {n}/{nrois}")

    n_subplots = figpar["init"]["ncols"]

    if roi_vm_pars is not None:
        n_subplots *= 3
        gentitle = gentitle.replace("tuning", "tuning (Von Mises fits)")

    fig, ax = plot_util.init_fig(n_subplots, **figpar["init"])
    fig.suptitle(f"{gentitle} - ROI {n} ({nrois} total)", y=1)
    
    deg = u"\u00B0"
    for s, unexp_oris in enumerate(tc_oris):
        if comb_gabs:
            gab_str = "(gabors combined)"
            for subax in ax[0, s+1:]: # advance color cycle past gray
                subax.plot([], [])
        else:
            gab_str = f"({len(unexp_oris)} gabors)"
        seq_str = f"{seq_info[s][0].upper()}{seq_info[s][1:]}"
        title_str = f"{seq_str} {gab_str}"
        for g, gab_oris in enumerate(unexp_oris):
            if roi_vm_pars is not None:
                plot_vm_estim_analysis(
                    ax[:, s], xran, gab_oris, roi_data[s][g], 
                    roi_vm_pars[s][g], roi_hist_pars[s][g], title_str, fluor)
            else:
                # Just plot activations by orientation
                ax[0, s].plot(
                    gab_oris, roi_data[s], marker=".", lw=0, alpha=0.3)
                ax[0, s].set_title(title_str, y=1.02)
                xlab = u"Orientations ({})".format(deg)
                sess_plot_util.add_axislabels(
                    ax[0, s], fluor=fluor, area=True, x_ax=xlab, datatype="roi")
                plot_util.set_ticks(ax[0, s], "x", 0, max_val, 5)

    # share y axis ranges within rows
    plot_util.share_lims(ax, "row")
    savename = f"{gen_savename}_roi{n}"
    fulldir = plot_util.savefig(
        fig, savename, savedir, log_dir=log_dir, **figpar["save"])

    plot_util.cond_close_figs(fig)

    return fulldir


#############################################
def plot_tune_curve_regr(vm_means, vm_regr, seq_info, gentitle="", 
                         gen_savename="", figpar=None, savedir=None):
    """
    plot_tune_curve_regr(vm_means, vm_regr, seq_info)
    
    Plots correlation for expected vs unexpected orientation preferences. 

    Required args:
        - vm_means (3D array): array of mean Von Mises means for each ROI, not 
                               weighted by kappa value or weighted (in rad): 
                               ROI x unexp x kappa weighted (False, (True)
        - vm_regr (2D array): array of regression results correlating unexpected 
                              and non unexpected means across ROIs, not weighted 
                              by kappa value or weighted (in rad): 
                                  regr_val (score, slope, intercept)
                                     x kappa weighted (False, (True)
        - seq_info (list)   : list of strings with info on each group
                              plotted (2)
    
    Optional args:
        - gentitle (str)        : general title for the plot
                                  default: ""
        - gen_savename (str)    : general title for the plot
                                  default: ""
        - figpar (dict)         : dictionary containing the following figure 
                                  parameter dictionaries
                                  default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
            ["mng"]  (dict): dictionary with parameters to manage matplotlib
        - savedir (Path)        : path of directory in which to save plots.
                                  default: None
    
    Returns:
        - fulldir (Path)    : final path of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
    """
    
    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)

    figpar["init"]["ncols"] = 1

    if savedir is None:
        savedir = Path(
            figpar["dirs"]["roi"], 
            figpar["dirs"]["tune_curv"],
            "vm_estim")

    vm_means = np.asarray(vm_means)
    vm_regr = np.asarray(vm_regr)

    if np.min(vm_means) < 0:
        raise ValueError("Orientations expected to be positive.")  
    max_val = 180
    if np.max(vm_means) > np.deg2rad(max_val):
        max_val = 360
        if np.max(vm_means) > np.deg2rad(max_val):
            raise ValueError(
                "Orientations expected to be at most between 0 and 2 * pi."
                )

    xvals = [0, max_val]
    
    deg = u"\u00B0"
    
    kapw = [0, 1]
    if vm_regr.shape[1] == 1:
        kapw = [0]

    for i in kapw:
        if i == 0:
            kap_str, kap_str_pr = "", ""
        if i == 1:
            kap_str = "_kapw"
            kap_str_pr = " (kappa weighted)"
        data = np.rad2deg(vm_means[:, :, i])
        r_sqr, slope, interc = vm_regr[:, i]
        interc = np.rad2deg(interc)
        figpar["init"]["ncols"] = 1
        fig, ax = plot_util.init_fig(1, **figpar["init"])
        ax = ax[0, 0]
        ax.plot(data[:, 0], data[:, 1], marker=".", lw=0, alpha=0.4)

        # plot trend line
        col = ax.lines[-1].get_color()
        yvals = [x * slope + interc for x in xvals]
        lab = u"R{} = {:.4f}".format(u"\u00b2", r_sqr) # R2 = ##
        ax.plot(xvals, yvals, marker="", label=lab, color=col, lw=4)
        for ax_let in ["x", "y"]:
            plot_util.set_ticks(ax, ax_let, xvals[0], xvals[1], 5)
        ax.set_xlabel(u"Mean orientation preference\nfrom {} "
            u"({})".format(seq_info[0], deg), fontsize="x-large")
        ax.set_ylabel(u"Mean orientation preference\nfrom {} "
            u"({})".format(seq_info[1], deg), fontsize="x-large")
        ax.set_title(f"{gentitle}\nCorrelation btw orientation "
            f"prefs{kap_str_pr}", y=1.1)
        ax.legend()
        savename = f"{gen_savename}_regr{kap_str}"
        fulldir = plot_util.savefig(
            fig, savename, savedir, log_dir=False, **figpar["save"])
            
        plot_util.cond_close_figs(fig)

    return fulldir


#############################################
def plot_tune_curves(analyspar, sesspar, stimpar, extrapar, tcurvpar, 
                     tcurv_data, sess_info, figpar=None, savedir=None, 
                     parallel=False, plot_tc=True):
    """
    plot_tune_curves(analyspar, sesspar, stimpar, extrapar, tcurv_data, 
                     sess_info)

    Plots orientation fluorescence data, as well as ROI orientation tuning 
    curves, and a correlation plot for expected vs unexpected orientation 
    preferences, if provided. 

    Required args:
        - analyspar (dict) : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)   : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)   : dictionary with keys of StimPar namedtuple
        - extrapar (dict)  : dictionary containing additional analysis 
                             parameters
            ["analysis"] (str)  : analysis type (e.g., "o")
            ["comb_gabs"] (bool): if True, gabors were combined
            ["datatype"] (str): datatype (e.g., "roi")
            ["seed"] (int)      : seed
        - tcurvpar (dict)  : dictionary with keys of TCurvPar namedtuple
        - sess_info (dict) : dictionary containing information from each
                             session (one first session used) 
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session

        - tcurv_data (dict): tuning curve data 
            ["oris"] (list)         : list of orientation values 
                                      corresponding to the tc_data
                                      unexp x gabor (1 if comb_gabs) x oris
            ["data"] (list)         : list of mean integrated fluorescence 
                                      data per orientation, for each ROI, 
                                      structured as 
                                      ROI x unexp x gabor (1 if comb_gabs) 
                                          x oris
            ["nseqs"] (list)        : number of sequences per unexp

            if tcurvpar["vm_estim"]:
            ["vm_pars"] (4D array)  : array of Von Mises parameters for each
                                      ROI: 
                                      ROI x unexp x gabor (1 if comb_gabs) 
                                          x par
            ["vm_mean"] (3D array)  : array of mean Von Mises means for each
                                      ROI, not weighted by kappa value or 
                                      weighted (if not comb_gabs) (in rad): 
                                      ROI x unexp 
                                          x kappa weighted (False, (True))
            ["vm_regr"] (2D array)  : array of regression results 
                                      correlating unexpected and expected
                                      means across ROIs, not weighted by 
                                      kappa value or weighted (if not 
                                      comb_gabs) (in rad): 
                                      regr_val (score, slope, intercept)
                                          x kappa weighted (False, (True)) 
            ["hist_pars"] (list)    : parameters used to convert tc_data to 
                                      histogram values (sub, mult) used
                                      in Von Mises parameter estimation, 
                                      structured as:
                                      ROI x unexp x gabor (1 if comb_gabs) x 
                                         param (sub, mult)

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
        - plot_tc (bool) : if True, tuning curves are plotted for each ROI 

    Returns:
        - fulldir (Path)    : final path of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): general name under which the figures are saved
    """
    
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

    gabfrs = gen_util.list_if_not(stimpar["gabfr"])
    if len(gabfrs) == 1:
        gabfrs = gabfrs * 2
    
    stimstr_pr = stimstr_pr[:-1] if stimstr_pr[-1] == "s" else stimstr_pr
    stimstr_pr = stimstr_pr.capitalize()

    rand_str, rand_str_pr = "", ["", ""]
    if tcurvpar["grp2"] == "unexp":
        unexps = [0, 1]
        seq_types = ["exp.", "unexp."]
    elif tcurvpar["grp2"] in ["exp", "rand"]:
        unexps = [0, 0]
        seq_types = ["exp.", "exp."]
        if tcurvpar["grp2"] == "rand":
            rand_str = "_rand"
            rand_str_pr = ["", "sampled "]
    gabfr_letts = sess_str_util.gabfr_letters(gabfrs, unexps)

    if extrapar["comb_gabs"]:
        comb_gab_str = "_comb"
    else:
        comb_gab_str = ""

    # extract some info from sess_info (only one session)
    keys = ["mouse_ns", "sess_ns", "lines", "planes", "nrois"]
    [mouse_n, sess_n, line, plane, nrois] = [sess_info[key][0] for key in keys]
    
    seq_info = [f"{ra}{sqt} {gf} seqs ({nseqs})" for ra, sqt, gf, nseqs in 
        zip(rand_str_pr, seq_types, gabfr_letts, tcurv_data["nseqs"])]
    
    gentitle = (f"Mouse {mouse_n} - {stimstr_pr} orientation tuning\n"
        f"(sess {sess_n}, {line} {plane}{dendstr_pr})")
    gen_savename = (f"{datatype}_tc_m{mouse_n}_sess{sess_n}_{stimstr}_{plane}"
        f"{dendstr}{comb_gab_str}")
    if savedir is None:
        savedir = Path(
            figpar["dirs"][datatype], 
            figpar["dirs"]["tune_curv"], 
            f"{gabfr_letts[0]}v{rand_str}{gabfr_letts[1]}_seqs")
        if tcurvpar["vm_estim"]:
            savedir = savedir.joinpath("vm_estim")

    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    if tcurvpar["vm_estim"]: # plot regression
        fulldir = plot_tune_curve_regr(tcurv_data["vm_mean"], 
            tcurv_data["vm_regr"], seq_info, gentitle, gen_savename, figpar, 
            savedir)
    else:
        fulldir = plot_util.savefig(
            None, None, fulldir=savedir, log_dir=False, **figpar["save"])

    if plot_tc:
        # ROI numbers in plots may not exactly match their indices if bad ROIs 
        # have been removed
        if not tcurvpar["test"]:
            logger.warning("Plotting tuning curves for each ROI may take a "
                "while. To skip plotting, set plot_tc to False.", 
                extra={"spacing": TAB})

        if not tcurvpar["vm_estim"]: # send None values if not using von Mises analysis
            tcurv_data = copy.deepcopy(tcurv_data)
            tcurv_data ["vm_pars"] = [None] * len(tcurv_data["data"])
            tcurv_data ["hist_pars"] = [None] * len(tcurv_data["data"])
        # optionally runs in parallel
        if parallel:
            n_jobs = gen_util.get_n_jobs(len(tcurv_data["data"]))
            with gen_util.ParallelLogging():
                fulldirs = Parallel(n_jobs=n_jobs)(
                    delayed(plot_roi_tune_curves)
                    (tcurv_data["oris"], roi_data, n, nrois, seq_info, 
                    tcurv_data["vm_pars"][n], tcurv_data["hist_pars"][n], 
                    analyspar["fluor"], comb_gabs=extrapar["comb_gabs"], 
                    gentitle=gentitle, gen_savename=gen_savename, 
                    figpar=figpar, savedir=savedir)
                    for n, roi_data in enumerate(tcurv_data["data"]))
            fulldir = fulldirs[-1]
        else:
            for n, roi_data in enumerate(tcurv_data["data"]):
                fulldir = plot_roi_tune_curves(tcurv_data["oris"], roi_data, 
                    n, nrois, seq_info, tcurv_data["vm_pars"][n], 
                    tcurv_data["hist_pars"][n], analyspar["fluor"], 
                    comb_gabs=extrapar["comb_gabs"], gentitle=gentitle, 
                    gen_savename=gen_savename, figpar=figpar, savedir=savedir)
    else:
        logger.info("Not plotting tuning curves.")

    return fulldir, gen_savename


#############################################
def plot_loc_ori_resp(analyspar, sesspar, stimpar, extrapar, sess_info, 
                      loc_ori_data, figpar=None, savedir=None):
    """
    plot_loc_ori_resp(analyspar, sesspar, stimpar, extrapar, sess_info, 
                      loc_ori_data)

    Plots responses according to location and orientation for a specific 
    session.

    Required args:
        - analyspar (dict)  : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)    : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)    : dictionary with keys of StimPar namedtuple
        - extrapar (dict)   : dictionary containing additional analysis 
                              parameters
            ["analysis"] (str): analysis type (e.g., "p")
            ["datatype"] (str): datatype (e.g., "roi")
        - sess_info (dict) : dictionary containing information from each
                             session (one first session used) 
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session

        - loc_ori_data (dict): location and orientation responses
            ["oris"] (list)     : stimulus mean orientations
            ["roi_stats"] (list): ROI statistics, structured as 
                                    mean orientation x gaborframe x stats x ROI
            ["nseqs"] (list)    : number of sequences structured as 
                                    mean orientation x gaborframe

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
    
    stimstr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["visflow_dir"], stimpar["visflow_size"], 
        stimpar["gabk"])
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], extrapar["datatype"])

    datatype = extrapar["datatype"]
    if datatype != "roi":
        raise NotImplementedError("Function only implemented for roi datatype.")

    # extract some info from sess_info (only one session)
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_n, sess_n, line, plane] = [sess_info[key][0] for key in keys]
    
    title = (f"Mouse {mouse_n} - {stimstr_pr}, " + u"{} ".format(statstr_pr) + 
        "location and orientation responses across seqs\n"
        f"(sess {sess_n}, {line} {plane}{dendstr_pr})")
    savename = (f"{datatype}_loc_ori_m{mouse_n}_sess{sess_n}_{stimstr}_" 
        f"{plane}{dendstr}")
    if savedir is None:
        savedir = Path(
            figpar["dirs"][datatype], 
            figpar["dirs"]["loc_ori"])

    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()
    figpar["init"]["n_subplots"] = 1
    figpar["init"]["subplot_wid"] *= 2 

    fig, ax = plot_util.init_fig(**figpar["init"])
    sub_ax = ax[0, 0]

    n_oris = len(loc_ori_data["roi_stats"])
    n_gabfr = len(loc_ori_data["roi_stats"][0])
    n_rois = len(loc_ori_data["roi_stats"][0][0][0])

    if n_gabfr != 5:
        raise ValueError("Expected data for 5 gabor frames.")

    center_pos, dot_pos, _ = plot_util.get_barplot_xpos(
        n_oris, n_gabfr, in_grp=2.5, barw=0.3)

    all_labs = []
    tick_pos = []
    roi_cols = [None] * n_rois
    for o, ori_data in enumerate(loc_ori_data["roi_stats"]):
        # ROI x stats x gab
        ori_data = np.transpose(ori_data, (2, 1, 0))
        xpos = dot_pos[o] 
        for r, roi_data in enumerate(ori_data):
            plot_util.plot_errorbars(
                sub_ax, roi_data[0], roi_data[1:], x=xpos, title=title, 
                color=roi_cols[r], xticks="None")
            roi_cols[r] = sub_ax.lines[-1].get_color()
        labels = ["A", "B", "C", "D", "U"]
        labels = [f"{lab}\n({nseqs})" for lab, nseqs 
            in zip(labels, loc_ori_data["nseqs"][o])]
        all_labs.extend(labels)
        tick_pos.extend(xpos)

    deg = u"\u00B0"
    oris = [u"{}{}".format(ori, deg) for ori in loc_ori_data["oris"]]
    plot_util.add_labels(sub_ax, oris, center_pos, t_hei=-0.18, color="k")

    # always set ticks (even again) before setting labels
    sub_ax.set_xticks(tick_pos)
    sub_ax.set_xticklabels(all_labs, fontsize=16)

    sess_plot_util.add_axislabels(
        sub_ax, fluor=analyspar["fluor"], area=True, x_ax="", datatype=datatype)

    fulldir = plot_util.savefig(fig, savename, savedir, **figpar["save"])

    return fulldir, savename


#############################################
def plot_trial_pc_traj(analyspar, sesspar, stimpar, extrapar, sess_info, 
                       pc_traj_data, figpar=None, savedir=None):
    """
    plot_trial_pc_traj(analyspar, sesspar, stimpar, extrapar, sess_info, 
                       pc_traj_data)

    Plots responses according to location and orientation for a specific 
    session.

    Required args:
        - analyspar (dict)  : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)    : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)    : dictionary with keys of StimPar namedtuple
        - extrapar (dict)   : dictionary containing additional analysis 
                              parameters
            ["analysis"] (str): analysis type (e.g., "v")
            ["datatype"] (str): datatype (e.g., "roi")
        - sess_info (dict) : dictionary containing information from each
                             session (one first session used) 
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session

        - pc_traj_data (dict): trial PC trajectories





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

    Returns:
        - fulldir (Path): final path of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
        - savename (str): name under which the figure is saved
    """

    return
