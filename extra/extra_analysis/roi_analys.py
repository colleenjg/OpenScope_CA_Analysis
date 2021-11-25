"""
roi_analys.py

This script runs ROI trace analyses using a Session object with data generated 
by the Allen Institute OpenScope experiments for the Credit Assignment Project.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import copy
import logging
import warnings

from joblib import Parallel, delayed
import numpy as np

from util import file_util, gen_util, logger_util, math_util, rand_util
from sess_util import sess_gen_util, sess_ntuple_util, sess_str_util
from extra_analysis import ori_analys, quant_analys, signif_grps
from extra_plot_fcts import roi_analysis_plots as roi_plots

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def run_roi_areas_by_grp_qu(sessions, analyspar, sesspar, stimpar, extrapar,
                            permpar, quantpar, roigrppar, roi_grps, figpar, 
                            savedict=True):

    """
    run_roi_areas_by_grp_qu(sessions, analysis, analyspar, sesspar, stimpar, 
                            permpar, quantpar, roigrppar, roi_grps, figpar)

    Plots average integrated unexpected, expected or difference between 
    unexpected and expected activity across ROIs per group for each quantiles
    with each session in a separate subplot.

    Returns save directory path and results in roi_grps dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - extrapar (dict)      : dictionary containing additional analysis 
                                 parameters
            ["datatype"] (str): datatype (e.g., "roi")
        - permpar (PermPar)    : named tuple containing permutation parameters
        - quantpar (QuantPar)  : named tuple containing quantile analysis 
                                 parameters
        - roigrppar (RoiGrpPar): named tuple containing ROI grouping parameters
        - roi_grps (dict)      : dictionary containing ROI grps information:
            ["grp_names"] (list)   : list of names of the ROI groups in roi grp 
                                     lists (order preserved)
            ["all_roi_grps"] (list): nested lists containing ROI numbers 
                                     included in each group, structured as 
                                     follows:
                                         if sets of groups are passed: 
                                             session x set x roi_grp
                                         if one group is passed: 
                                             session x roi_grp
        - figpar (dict)        : dictionary containing figure parameters
    
    Optional args:
        - savedict (bool): if True, dictionaries containing parameters used
                           for analysis are saved

    Returns:
        - fulldir (str)  : final name of the directory in which the figures are 
                           saved 
        - roi_grps (dict): dictionary containing ROI grps information:
            ["grp_names"] (list)   : see above
            ["all_roi_grps"] (list): see above
            ["grp_st"] (array-like): nested list or array of group stats 
                                     (mean/median, error) across ROIs, 
                                     structured as:
                                         session x quantile x grp x stat
            ["grp_ns"] (array-like): nested list of group ns, structured as: 
                                         session x grp
    """

    opstr_pr = sess_str_util.op_par_str(
        roigrppar.plot_vals, roigrppar.op, str_type="print")
    sessstr_pr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.visflow_dir, 
        stimpar.visflow_size, stimpar.gabk, "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, extrapar["datatype"], "print")
    datastr = sess_str_util.datatype_par_str(extrapar["datatype"])
    
    if extrapar["datatype"] != "roi":
        raise NotImplementedError("Analysis only implemented for roi datatype.")

    logger.info(f"Analysing and plotting {opstr_pr} {datastr} average "
        f"response by quantile ({quantpar.n_quants}). \n{sessstr_pr}"
        f"{dendstr_pr}.", extra={"spacing": "\n"})
    
    # get full data for qu of interest: session x unexp x [seq x ROI]
    integ_info = quant_analys.trace_stats_by_qu_sess(
        sessions, analyspar, stimpar, quantpar.n_quants, "all", by_exp=True, 
        integ=True)

    # retrieve only mean/medians per ROI
    all_me = [sess_stats[:, :, 0] for sess_stats in integ_info[1]]

    # get statistics per group and number of ROIs per group
    grp_st, grp_ns = signif_grps.grp_stats(
        all_me, roi_grps["all_roi_grps"], roigrppar.plot_vals, roigrppar.op, 
        analyspar.stats, analyspar.error)

    roi_grps = copy.deepcopy(roi_grps)
    roi_grps["grp_st"] = grp_st.tolist()
    roi_grps["grp_ns"] = grp_ns.tolist()

    sess_info = sess_gen_util.get_sess_info(
        sessions, analyspar.fluor, rem_bad=analyspar.rem_bad
        )
    
    info = {
        "analyspar": analyspar._asdict(),
        "sesspar"  : sesspar._asdict(),
        "stimpar"  : stimpar._asdict(),
        "extrapar" : extrapar,
        "quantpar" : quantpar._asdict(),
        "permpar"  : permpar._asdict(),
        "roigrppar": roigrppar._asdict(),
        "sess_info": sess_info,
        "roi_grps" : roi_grps
        }
    
    # plot
    fulldir, savename = roi_plots.plot_roi_areas_by_grp_qu(
        figpar=figpar, **info)
    
    if savedict:
        file_util.saveinfo(info, savename, fulldir, "json")

    return fulldir, roi_grps


#############################################
def run_roi_traces_by_grp(sessions, analyspar, sesspar, stimpar, extrapar, 
                          permpar, quantpar, roigrppar, roi_grps, figpar, 
                          savedict=True):
                           
    """
    run_roi_traces_by_grp(sessions, analysis, sesspar, stimpar, extrapar, 
                          permpar, quantpar, roigrppar, roi_grps, figpar)

    Calculates and plots ROI traces across ROIs by group for unexpected, 
    expected or difference between unexpected and expected activity per 
    quantile (first/last) with each group in a separate subplot and each 
    session in a different figure.

    Optionally saves results and parameters relevant to analysis in a 
    dictionary.

    Returns save directory path and results in roi_grps dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - extrapar (dict)      : dictionary containing additional analysis 
                                 parameters
            ["datatype"] (str): datatype (e.g., "roi")
        - permpar (PermPar)    : named tuple containing permutation parameters
        - quantpar (QuantPar)  : named tuple containing quantile analysis 
                                parameters
        - roigrppar (RoiGrpPar): named tuple containing ROI grouping parameters
        - roi_grps (dict) : dictionary containing ROI grps information:
            ["grp_names"] (list)   : list of names of the ROI groups in roi grp 
                                     lists (order preserved)
            ["all_roi_grps"] (list): nested lists containing ROI numbers 
                                     included in each group, structured as 
                                     follows:
                                         if sets of groups are passed: 
                                             session x set x roi_grp
                                         if one group is passed: 
                                             session x roi_grp
        - figpar (dict)        : dictionary containing figure parameters
        
    Optional args:
        - savedict (bool): if True, dictionaries containing parameters used
                           for analysis are saved

    Returns:
        - fulldir (str)  : final name of the directory in which the figures are 
                           saved 
        - roi_grps (dict): dictionary containing ROI grps information:
            ["grp_names"] (list)        : see above
            ["all_roi_grps"] (list)     : see above
            ["xrans"] (list)            : list of time values for the
                                          frame chunks, for each session
            ["trace_stats"] (array-like): array or nested list of statistics of
                                          ROI groups for quantiles of interest
                                          structured as:
                                              sess x qu x ROI grp x stats 
                                              x frame
    """

    opstr_pr = sess_str_util.op_par_str(
        roigrppar.plot_vals, roigrppar.op, str_type="print")
    sessstr_pr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.visflow_dir,
        stimpar.visflow_size, stimpar.gabk, "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, extrapar["datatype"], "print")
    opstr = sess_str_util.op_par_str(roigrppar.plot_vals, roigrppar.op)
    sessstr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.visflow_dir,
        stimpar.visflow_size, stimpar.gabk, "file")
    dendstr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, extrapar["datatype"])
     
    datastr = sess_str_util.datatype_par_str(extrapar["datatype"])
    if extrapar["datatype"] != "roi":
        raise NotImplementedError("Analysis only implemented for roi datatype.")

    logger.info(f"Analysing and plotting {opstr_pr} {datastr} unexp vs exp "
        f"traces by quantile ({quantpar.n_quants}). \n{sessstr_pr}{dendstr_pr}.", 
        extra={"spacing": "\n"})

    # get sess x unexp x quant x stats x ROIs x frames
    trace_info = quant_analys.trace_stats_by_qu_sess(
        sessions, analyspar, stimpar, n_quants=quantpar.n_quants, 
        qu_idx=quantpar.qu_idx, byroi=True, by_exp=True)
    xrans = [xran.tolist() for xran in trace_info[0]]

    # retain mean/median from trace stats
    trace_me = [sessst[:, :, 0] for sessst in trace_info[1]]

    grp_stats = signif_grps.grp_traces_by_qu_unexp_sess(
        trace_me, analyspar, roigrppar, roi_grps["all_roi_grps"])

    roi_grps = copy.deepcopy(roi_grps)
    roi_grps["xrans"] = xrans
    roi_grps["trace_stats"] = grp_stats

    sess_info = sess_gen_util.get_sess_info(
        sessions, analyspar.fluor, rem_bad=analyspar.rem_bad
        )

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "stimpar"    : stimpar._asdict(),
            "extrapar"   : extrapar,
            "permpar"    : permpar._asdict(),
            "quantpar"   : quantpar._asdict(),
            "roigrppar"  : roigrppar._asdict(),
            "sess_info"  : sess_info,
            "roi_grps"   : roi_grps
            }

    fulldir = roi_plots.plot_roi_traces_by_grp(figpar=figpar, **info)

    if savedict:
        infoname = (f"roi_tr_{sessstr}{dendstr}_grps_{opstr}_"
            f"{quantpar.n_quants}quant_{permpar.tails}tail")

        file_util.saveinfo(info, infoname, fulldir, "json")

    return fulldir, roi_grps


#############################################
def run_roi_areas_by_grp(sessions, analyspar, sesspar, stimpar, extrapar,  
                         permpar, quantpar, roigrppar, roi_grps, figpar, 
                         savedict=False):
    """
    run_roi_areas_by_grp(sessions, analyspar, sesspar, stimpar, extrapar, 
                         permpar, quantpar, roigrppar, roi_grps, fig_par)

    Calculates and plots ROI traces across ROIs by group for unexpected, 
    expected or difference between unexpected and expected activity per 
    quantile (first/last) with each group in a separate subplot and each 
    session in a different figure. 

    Optionally saves results and parameters relevant to analysis in a 
    dictionary.

    Returns save directory path and results in roi_grps dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - extrapar (dict)      : dictionary containing additional analysis 
                                 parameters
            ["datatype"] (str): datatype (e.g., "roi")
        - permpar (PermPar)    : named tuple containing permutation parameters
        - quantpar (QuantPar)  : named tuple containing quantile analysis 
                                 parameters
        - roigrppar (RoiGrpPar): named tuple containing ROI grouping parameters
        - roi_grps (dict)      : dictionary containing ROI grps information:
            ["all_roi_grps"] (list): nested lists containing ROI numbers 
                                     included in each group, structured as 
                                     follows:
                                         if sets of groups are passed: 
                                             session x set x roi_grp
                                         if one group is passed: 
                                             session x roi_grp
            ["grp_names"] (list)   : list of names of the ROI groups in ROI 
                                     grp lists (order preserved)
        - figpar (dict)        : dictionary containing figure parameters
        
    Optional args:
        - savedict (bool): if True, dictionaries containing parameters used
                           for analysis are saved

    Returns:
        - fulldir (str)  : final name of the directory in which the figures are 
                           saved 
        - roi_grps (dict): dictionary containing ROI groups:
            ["all_roi_grps"] (list)          : see above
            ["grp_names"] (list)             : see above
            ["area_stats"] (array-like)      : ROI group stats (mean/median,      
                                               error) for quantiles of interest,
                                               structured as:
                                                 session x quantile x grp x 
                                                 stat
            ["area_stats_scale"] (array-like): same as "area_stats", but with 
                                              last quantile scaled relative to 
                                              first
    """
    
    opstr_pr = sess_str_util.op_par_str(
        roigrppar.plot_vals, roigrppar.op, str_type="print")
    sessstr_pr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.visflow_dir,
        stimpar.visflow_size, stimpar.gabk, "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, extrapar["datatype"], "print")
   
    opstr = sess_str_util.op_par_str(roigrppar.plot_vals, roigrppar.op)
    sessstr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.visflow_dir,
        stimpar.visflow_size, stimpar.gabk)
    dendstr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, extrapar["datatype"])
     
    datastr = sess_str_util.datatype_par_str(extrapar["datatype"])
    if extrapar["datatype"] != "roi":
        raise NotImplementedError("Analysis only implemented for roi datatype.")

    logger.info(f"Analysing and plotting {opstr_pr} {datastr} unexp vs exp "
        f"average responses by quantile ({quantpar.n_quants}). \n{sessstr_pr}"
        f"{dendstr_pr}.", extra={"spacing": "\n"})

    # get full data for qu of interest: session x unexp x [seq x ROI]
    integ_info = quant_analys.trace_stats_by_qu_sess(
        sessions, analyspar, stimpar, quantpar.n_quants, quantpar.qu_idx, 
        by_exp=True, integ=True)     

    # retrieve only mean/medians per ROI
    all_me = [sess_stats[:, :, 0] for sess_stats in integ_info[1]]

    roi_grps = copy.deepcopy(roi_grps)
    # get statistics per group and number of ROIs per group
    for scale in [False, True]:
        scale_str = sess_str_util.scale_par_str(scale)
        # sess x quant x grp x stat
        grp_st, _ = signif_grps.grp_stats(
            all_me, roi_grps["all_roi_grps"], roigrppar.plot_vals, roigrppar.op, 
            analyspar.stats, analyspar.error, scale)
        roi_grps[f"area_stats{scale_str}"] = grp_st.tolist()

    sess_info = sess_gen_util.get_sess_info(
        sessions, analyspar.fluor, rem_bad=analyspar.rem_bad
        )

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "extrapar" : extrapar,
            "permpar"  : permpar._asdict(),
            "quantpar" : quantpar._asdict(),
            "roigrppar": roigrppar._asdict(),
            "sess_info": sess_info,
            "roi_grps" : roi_grps
            }
        
    fulldir = roi_plots.plot_roi_areas_by_grp(figpar=figpar, **info)

    if savedict:
        infoname = (f"roi_area_{sessstr}{dendstr}_grps_{opstr}_"
            f"{quantpar.n_quants}quant_{permpar.tails}tail")
        file_util.saveinfo(info, infoname, fulldir, "json")
    
    return fulldir, roi_grps


#############################################
def run_rois_by_grp(sessions, analysis, seed, analyspar, sesspar, stimpar, 
                    permpar, quantpar, roigrppar, figpar):
    """
    run_rois_by_grp(sessions, analysis, seed, analyspar, sesspar, stimpar, 
                    permpar, quantpar, roigrppar, figpar)

    Identifies ROIs showing significant unexpected in first and/or last quantile,
    group accordingly and plots traces and areas across ROIs for unexpected, 
    expected or difference between unexpected and expected activity per 
    quantile (first/last) with each group in a separate subplot and each 
    session in a different figure. 
    
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., "g")
        - seed (int)           : seed value to use. (-1 treated as None) 
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - permpar (PermPar)    : named tuple containing permutation parameters
        - quantpar (QuantPar)  : named tuple containing quantile analysis 
                                 parameters
        - roigrppar (RoiGrpPar): named tuple containing ROI grouping parameters
        - figpar (dict)        : dictionary containing figure parameters
    """

    datatype = "roi"

    opstr = sess_str_util.op_par_str(roigrppar.plot_vals, roigrppar.op)
    sessstr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.visflow_dir,
        stimpar.visflow_size, stimpar.gabk)
    dendstr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, datatype)
    
    sessids = [sess.sessid for sess in sessions]
    
    # get full data for qu of interest: session x unexp x [seq x ROI]
    integ_info = quant_analys.trace_stats_by_qu_sess(
        sessions, analyspar, stimpar, quantpar.n_quants, quantpar.qu_idx, 
        by_exp=True, integ=True, ret_arr=True)

    _, _, _, qu_data = integ_info     

    if analyspar.rem_bad:
        nanpol = None
    else:
        nanpol = "omit"

    seed = rand_util.seed_all(seed, "cpu", log_seed=False)

    qu_labs = [
        sess_str_util.quantile_str(q, quantpar.n_quants, str_type="print")
        for q in quantpar.qu_idx 
    ]        
        
    # identify significant ROIs 
    # (returns all_roi_grps, grp_names)
    all_roi_grps, grp_names = signif_grps.signif_rois_by_grp_sess(
        sessids, qu_data, permpar, roigrppar, qu_labs, 
        stats=analyspar.stats, nanpol=nanpol)

    roi_grps = {"all_roi_grps": all_roi_grps,
                "grp_names"   : grp_names,
               }
    
    extrapar  = {"analysis": analysis,
                 "datatype": datatype,
                 "seed"    : seed
                 }

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    _, roi_grps_q = run_roi_areas_by_grp_qu(sessions, analyspar, sesspar, 
        stimpar, extrapar, permpar, quantpar, roigrppar, roi_grps, 
        figpar, savedict=False)    

    _, roi_grps_t = run_roi_traces_by_grp(sessions, analyspar, sesspar, 
        stimpar, extrapar, permpar, quantpar, roigrppar, roi_grps, figpar, 
        savedict=False)

    fulldir, roi_grps_a = run_roi_areas_by_grp(sessions, analyspar, sesspar, 
        stimpar, extrapar, permpar, quantpar, roigrppar, roi_grps,
        figpar, savedict=False)

    # add roi_grps_t and roi_grps_a keys to roi_grps dictionary
    for roi_grps_dict in [roi_grps_q, roi_grps_t, roi_grps_a]:
        for key in roi_grps_dict.keys():
            if key not in roi_grps:
                roi_grps[key] = roi_grps_dict[key]

    sess_info = sess_gen_util.get_sess_info(
        sessions, analyspar.fluor, rem_bad=analyspar.rem_bad
        )

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "extrapar" : extrapar,
            "permpar"  : permpar._asdict(),
            "quantpar" : quantpar._asdict(),
            "roigrppar": roigrppar._asdict(),
            "sess_info": sess_info,
            "roi_grps" : roi_grps
            }
     
    infoname = (f"roi_{sessstr}{dendstr}_grps_{opstr}_{quantpar.n_quants}q_"
        f"{permpar.tails}tail")

    file_util.saveinfo(info, infoname, fulldir, "json")


#############################################
def run_oridirs_by_qu_sess(sess, oridirs, unexps, xran, mes, counts, 
                           analyspar, sesspar, stimpar, extrapar, quantpar, 
                           figpar, parallel=False):
    """

    run_oridirs_by_qu_sess(sess, oridirs, unexps, xrans, mes, counts, 
                           analyspar, sesspar, stimpar, extrapar, quantpar, 
                           figpar)

    Plots average activity across gabor orientations or across visual flow 
    directions, locked to unexpected/expected transition per ROI as colormaps, 
    and across ROIs as traces for a single session and specified quantile.
    Saves results and parameters relevant to analysis in a dictionary. 

    Required args:
        - sess (Session)       : Session object
        - oridirs (list)       : list of orientations/directions
        - unexps (list)         : list of unexpected values
        - xran (list)          : list of time values for the 2p frames
        - mes (nested list)    : ROI mean/median data, structured as:
                                    oridirs x unexp x ROI x frames
        - counts (nested list) : number of sequences, structured as:
                                    oridirs x unexp
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - extrapar (dict)      : dictionary containing additional analysis 
                                 parameters
            ["analysis"] (str): analysis type (e.g., "o")
            ["datatype"] (str): datatype (e.g., "roi")
        - quantpar (QuantPar)  : named tuple containing quantile analysis 
                                 parameters
        - figpar (dict)        : dictionary containing figure parameters

    Optional args:
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
                           default: False
    """

    stimstr = sess_str_util.stim_par_str(
        stimpar.stimtype, stimpar.visflow_dir, stimpar.visflow_size, stimpar.gabk)
    dendstr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, extrapar["datatype"])
       
    if extrapar["datatype"] != "roi":
        raise NotImplementedError("Analysis only implemented for roi datatype.")

    qu_str, qu_str_pr = "", ""
    if quantpar.n_quants > 1:
        qu_str_pr = sess_str_util.quantile_str(
            quantpar.qu_idx, n_quants=quantpar.n_quants, out_of=True, 
            str_type="print"
            )
        qu_str_pr = f", {qu_str_pr}"

        qu_str = sess_str_util.quantile_str(
            quantpar.qu_idx, n_quants=quantpar.n_quants, str_type="file"
            )
        qu_str = f"_{qu_str}"

    if not analyspar.rem_bad:
        nanpol = "omit"
    else:
        nanpol = None

    logger.info(f"Mouse {sess.mouse_n}, sess {sess.sess_n}, {sess.line}, "
        f"{sess.plane}{qu_str_pr}", extra={"spacing": TAB})
    [n_seqs, roi_me, stats, 
        scale_vals, roi_sort] = [dict(), dict(), dict(), dict(), dict()]
    for o, od in enumerate(oridirs):
        for s, unexp in enumerate(unexps):
            key = f"{unexp}_{od}"
            me = mes[o][s] # me per ROI
            n_seqs[key] = int(counts[o][s])
            # sorting idx
            roi_sort[key] = np.argsort(np.argmax(me, axis=1)).tolist()
            scale_vals[f"{key}_max"] = np.max(me, axis=1).tolist()
            scale_vals[f"{key}_min"] = np.min(me, axis=1).tolist()
            roi_me[key] = me.tolist()
            # stats across ROIs
            stats[key]  = math_util.get_stats(
                me, analyspar.stats, analyspar.error, 0, nanpol).tolist()

    tr_data = {"xran"      : xran.tolist(),
               "n_seqs"    : n_seqs,
               "roi_me"    : roi_me,
               "stats"     : stats,
               "scale_vals": scale_vals,
               "roi_sort"  : roi_sort
               }

    sess_info = sess_gen_util.get_sess_info(
        sess, analyspar.fluor, rem_bad=analyspar.rem_bad
        )

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "extrapar" : extrapar,
            "quantpar" : quantpar._asdict(),
            "tr_data"  : tr_data,
            "sess_info": sess_info
            }
    
    roi_plots.plot_oridir_colormaps(figpar=figpar, parallel=parallel, **info)

    fulldir = roi_plots.plot_oridir_traces(figpar=figpar, **info)

    savename = (f"roi_cm_tr_m{sess.mouse_n}_sess{sess.sess_n}{qu_str}_"
        f"{stimstr}_{sess.plane}{dendstr}")

    file_util.saveinfo(info, savename, fulldir, "json")


#############################################
def run_oridirs_by_qu(sessions, oridirs, unexps, analyspar, sesspar, stimpar, 
                      extrapar, quantpar, figpar, parallel=False):
    """
    run_oridirs_by_qu(sessions, oridirs, unexps, analyspar, sesspar, stimpar,
                      extrapar, quantpar, figpar)

    Plots average activity across gabor orientations or across visual flow 
    directions, locked to unexpected/expected transition per ROI as colormaps, 
    and across ROIs as traces for a specified quantile.
    Saves results and parameters relevant to analysis in a dictionary. 

    Required args:
        - sessions (list)      : list of Session objects
        - oridirs (list)       : list of orientations/directions
        - unexps (list)         : list of unexpected values
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - extrapar (dict)      : dictionary containing additional analysis 
                                 parameters
            ["analysis"] (str): analysis type (e.g., "o")
            ["datatype"] (str): datatype (e.g., "roi")
        - quantpar (QuantPar)  : named tuple containing quantile analysis 
                                 parameters
        - figpar (dict)        : dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
    """

    # for each orientation/direction
    mes, counts = [], []
    for od in oridirs:
        # create a specific stimpar for each direction or orientation
        if stimpar.stimtype == "visflow":
            key = "visflow_dir"
            lock = "both"
        elif stimpar.stimtype == "gabors":
            key = "gab_ori"
            lock = "no"
        stimpar_od = sess_ntuple_util.get_modif_ntuple(stimpar, key, od)
        # NaN stats if no segments fit criteria
        nan_empty = True
        trace_info = quant_analys.trace_stats_by_qu_sess(
            sessions, analyspar, stimpar_od, quantpar.n_quants, 
            quantpar.qu_idx, byroi=True, by_exp=True, lock=lock, 
            nan_empty=nan_empty)
        xrans = trace_info[0]
        # retrieve mean/medians and single quantile data:
        # sess x [unexp x ROIs x frames]
        mes.append([sess_stats[:, 0, 0] for sess_stats in trace_info[1]]) 
        # retrieve single quantile counts: sess x unexp
        counts.append(
            [[unexp_c[0] for unexp_c in sess_c] for sess_c in trace_info[2]])

    mes = [np.asarray(vals) for vals in zip(*mes)]
    counts = [np.asarray(vals) for vals in zip(*counts)]

    # optionally runs in parallel
    if parallel and len(sessions) > 1:
        n_jobs = gen_util.get_n_jobs(len(sessions))
        Parallel(n_jobs=n_jobs)(delayed(run_oridirs_by_qu_sess)
            (sess, oridirs, unexps, xrans[se], mes[se], counts[se], analyspar, 
            sesspar, stimpar, extrapar, quantpar, figpar, False) 
            for se, sess in enumerate(sessions))
    else:
        for se, sess in enumerate(sessions):
            run_oridirs_by_qu_sess(
                sess, oridirs, unexps, xrans[se], mes[se], counts[se], 
                analyspar, sesspar, stimpar, extrapar, quantpar, figpar, 
                parallel)


#############################################
def run_oridirs(sessions, analysis, analyspar, sesspar, stimpar, quantpar, 
                figpar, parallel=False):
    """
    run_oridirs(sessions, analysis, analyspar, sesspar, stimpar, quantpar, 
                figpar)

    Plots average activity across gabor orientations or visual flow directions 
    per ROI as colormaps, and across ROIs as traces. 
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., "o")
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - quantpar (QuantPar)  : named tuple containing quantile analysis 
                                 parameters
        - figpar (dict)        : dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
    """
   
    datatype = "roi"

    # update stim parameters parameters
    if stimpar.stimtype == "visflow":
        # update stimpar with both visual flow directions
        oridirs = ["right", "left"]
        keys = ["visflow_dir", "pre", "post"]
        vals = [oridirs, 2.0, 4.0]
    elif stimpar.stimtype == "gabors":
        # update stimpar with gab_fr = 0 and all gabor orientations
        oridirs = sess_gen_util.filter_gab_oris("D", stimpar.gab_ori)
        keys = ["gabfr", "gab_ori"]
        vals = [0, oridirs]
    stimpar = sess_ntuple_util.get_modif_ntuple(stimpar, keys, vals)            

    sessstr_pr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.visflow_dir, 
        stimpar.visflow_size, stimpar.gabk, "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, datatype, "print")
  
    # split quantiles apart and add a quant=1
    quantpar_one  = sess_ntuple_util.init_quantpar(1, 0)
    quantpars = [quantpar_one]
    for qu_idx in quantpar.qu_idx:
        qp = sess_ntuple_util.init_quantpar(quantpar.n_quants, qu_idx)
        quantpars.append(qp)

    logger.info("Analysing and plotting colormaps and traces "
          f"({sessstr_pr}{dendstr_pr}).", extra={"spacing": "\n"})

    extrapar = {"analysis": analysis,
                "datatype": datatype,
                }

    unexps = ["exp", "unexp"]  
    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    # optionally runs in parallel
    if parallel and (len(quantpars) > np.max([1, len(sessions)])):
        n_jobs = gen_util.get_n_jobs(len(quantpars))
        Parallel(n_jobs=n_jobs)(delayed(run_oridirs_by_qu)
            (sessions, oridirs, unexps, analyspar, sesspar, stimpar, 
            extrapar, quantpar, figpar, False) 
            for quantpar in quantpars)
    else:
        for quantpar in quantpars:
            run_oridirs_by_qu(
                sessions, oridirs, unexps, analyspar, sesspar, stimpar, 
                extrapar, quantpar, figpar, parallel)


#############################################
def run_tune_curves(sessions, analysis, seed, analyspar, sesspar, stimpar, 
                    tcurvpar, figpar, parallel=False, plot_tc=True):
    """
    run_tune_curves(sessions, analysis, seed, analyspar, sesspar, stimpar, 
                    tcurvpar, figpar)

    Calculates and plots estimated ROI orientation tuning curves, as well as a 
    correlation plot for expected vs unexpected orientation preferences. 

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., "c")
        - seed (int)           : seed to use
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - tcurvpar (TCurvPar)  : named tuple containing tuning curve 
                                 parameters
        - figpar (dict)        : dictionary containing figure parameters

    Optional args:
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores 
        - plot_tc (bool) : if True, tuning curves are plotted for each ROI 
                           (causes errors on the clusters...)  
    """

    datatype = "roi"

    if stimpar.stimtype == "visflow":
        warnings.warn("Tuning curve analysis not implemented for visual flow.", 
            category=UserWarning, stacklevel=1)
        return
    
    sessstr_pr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.visflow_dir, 
        stimpar.visflow_size, stimpar.gabk, "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, datatype, "print")
  
    logger.info("Analysing and plotting ROI tuning curves for orientations "
        f"({sessstr_pr}{dendstr_pr}).", extra={"spacing": "\n"})

    # small values for testing
    if tcurvpar.test:
        nrois = 8
        ngabs = "all"
        comb_gabs_all = [True]
    else:
        nrois = "all"
        ngabs = "all"
        comb_gabs_all = [True, False]

        logger.warning("This analysis may take a long time, as each ROI is "
            "analysed separately. To run on only a few ROIs, "
            "set tcurvpar.test to True.", extra={"spacing": TAB})
    
    logger.info(f"Number ROIs: {nrois}\nNumber of gabors: {ngabs}")

    # modify parameters
    stimpar_tc = sess_ntuple_util.get_modif_ntuple(
        stimpar, ["gabfr", "pre", "post"], 
        [tcurvpar.gabfr, tcurvpar.pre, tcurvpar.post])

    seed = rand_util.seed_all(seed, "cpu", log_seed=False)

    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    for comb_gabs in comb_gabs_all:
        for sess in sessions:
            returns = ori_analys.calc_tune_curvs(
                sess, analyspar, stimpar_tc, nrois, ngabs, tcurvpar.grp2, 
                comb_gabs, tcurvpar.vm_estim, collapse=True, parallel=parallel)
            if tcurvpar.vm_estim:
                [tc_oris, tc_data, tc_nseqs, tc_vm_pars, 
                    tc_vm_mean, tc_hist_pars] = returns
            else:
                tc_oris, tc_data, tc_nseqs = returns

            tcurv_data = {"oris" : tc_oris,
                          "data" : [list(data) for data in zip(*tc_data)],
                          "nseqs": tc_nseqs,
                          }

            # estimate tuning curves by fitting a Von Mises distribution
            if tcurvpar.vm_estim:
                tcurv_data["vm_pars"] = np.transpose(
                    np.asarray(tc_vm_pars), [1, 0, 2, 3]).tolist()
                tcurv_data["vm_mean"] = np.transpose(
                    np.asarray(tc_vm_mean), [1, 0, 2]).tolist()
                tcurv_data["hist_pars"] = np.transpose(
                    np.asarray(tc_hist_pars), [1, 0, 2, 3]).tolist()
                tcurv_data["vm_regr"] = ori_analys.ori_pref_regr(
                    tcurv_data["vm_mean"]).tolist()

            extrapar = {"analysis" : analysis,
                        "datatype" : datatype,
                        "seed"     : seed,
                        "comb_gabs": comb_gabs,
                        }

            sess_info = sess_gen_util.get_sess_info(
                sess, analyspar.fluor, rem_bad=analyspar.rem_bad
            )

            info = {"analyspar" : analyspar._asdict(),
                    "sesspar"   : sesspar._asdict(),
                    "stimpar"   : stimpar_tc._asdict(),
                    "extrapar"  : extrapar,
                    "tcurvpar"  : tcurvpar._asdict(),
                    "tcurv_data": tcurv_data,
                    "sess_info" : sess_info
                    }

            fulldir, savename = roi_plots.plot_tune_curves(
                figpar=figpar, parallel=parallel, plot_tc=plot_tc, **info)

            file_util.saveinfo(info, savename, fulldir, "json")


#############################################
def loc_ori_resp(sess, analyspar, stimpar, nrois="all"):
    """
    loc_ori_resp(sess, analyspar, stimpar)

    Calculates integrated fluorescence levels for ROI locations and 
    orientations.

    Required args:
        - sess (Session): Session object
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters

    Optional args:
        - nrois (int): number of ROIs to include in analysis
                       default: "all"

    Returns:
        - oris (list)     : stimulus mean orientations
        - roi_stats (list): ROI statistics, structured as 
                                mean orientation x gaborframe x stats x ROI
        - nseqs (list)    : number of sequences structured as 
                                mean orientation x gaborframe
    """

    if sess.only_tracked_rois != analyspar.tracked:
        raise RuntimeError(
            "sess.only_tracked_rois should match analyspar.tracked."
            )

    stim = sess.get_stim(stimpar.stimtype)
    oris = stim.exp_gabfr_mean_oris
    nrois_tot = sess.get_nrois(analyspar.rem_bad, analyspar.fluor)
    if nrois == "all":
        sess_nrois = nrois_tot
    else:
        sess_nrois = np.min([nrois_tot, nrois])

    roi_stats = []
    nseqs = []
    
    for ori in oris:
        ori_stats = []
        ori_nseqs = []
        for gf in range(5):
            if gf == 3: # D
                s = 0
            elif gf == 4: # U
                s = 1
                gf = 3
                ori = sess_gen_util.get_unexp_gab_ori(ori)
            else:
                s = "any"
            # get segments
            segs = stim.get_segs_by_criteria(
                gabfr=gf, visflow_dir=stimpar.visflow_dir, 
                visflow_size=stimpar.visflow_size, 
                gab_ori=ori, gabk=stimpar.gabk, unexp=s, by="seg")
            ori_nseqs.append(len(segs))
            twopfr = stim.get_fr_by_seg(
                segs, start=True, fr_type="twop"
                )["start_frame_twop"]
            # stats x ROI
            gf_stats = gen_util.reshape_df_data(stim.get_roi_stats_df(twopfr, 
                stimpar.pre, stimpar.post, byroi=True, fluor=analyspar.fluor, 
                integ=True, rem_bad=analyspar.rem_bad, stats=analyspar.stats, 
                error=analyspar.error, scale=analyspar.scale).loc["stats"], 
                squeeze_cols=True).T
            ori_stats.append(gf_stats[:, :sess_nrois].tolist())
        roi_stats.append(ori_stats)
        nseqs.append(ori_nseqs)
    
    return oris, roi_stats, nseqs 


#############################################
def run_loc_ori_resp(sessions, analysis, analyspar, sesspar, stimpar, figpar, 
                     parallel=False):
    """
    run_loc_ori_resp(sessions, analysis, analyspar, sesspar, stimpar, figpar)

    Calculates and plots integrated fluorescence levels for ROI locations and 
    mean orientations.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., "p")
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - figpar (dict)        : dictionary containing figure parameters

    Optional args:
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores 
    """

    datatype = "roi"
    
    if stimpar.stimtype == "visflow":
        warnings.warn(
            "Location preference analysis not implemented for visual flow.", 
            category=UserWarning, stacklevel=1
            )
        return
    
    sessstr_pr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.visflow_dir, 
        stimpar.visflow_size, stimpar.gabk, "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, datatype, "print")
  
    datastr = sess_str_util.datatype_par_str(datatype)
    
    logger.info(f"Analysing and plotting {datastr} location preferences "
        f"({sessstr_pr}{dendstr_pr}).", extra={"spacing": "\n"})

    nrois = "all"
    nrois = 8
    
    logger.info(f"Number ROIs: {nrois}")

    # modify parameters
    stimpar_loc = sess_ntuple_util.get_modif_ntuple(
        stimpar, ["pre", "post"], [0, 0.45])

    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    for sess in sessions:
        oris, roi_stats, nseqs = loc_ori_resp(
            sess, analyspar, stimpar_loc, nrois)
        loc_ori_data = {"oris"     : oris,
                        "roi_stats": roi_stats,
                        "nseqs"    : nseqs
                      }

        extrapar = {"analysis": analysis,
                   "datatype" : datatype,
                   }

        sess_info = sess_gen_util.get_sess_info(
            sess, analyspar.fluor, rem_bad=analyspar.rem_bad
            )

        info = {"analyspar"   : analyspar._asdict(),
                "sesspar"     : sesspar._asdict(),
                "stimpar"     : stimpar_loc._asdict(),
                "extrapar"    : extrapar,
                "loc_ori_data": loc_ori_data,
                "sess_info"   : sess_info
                }

        fulldir, savename = roi_plots.plot_loc_ori_resp(figpar=figpar, **info)

        file_util.saveinfo(info, savename, fulldir, "json")


#############################################
def run_trial_pc_traj(sessions, analysis, analyspar, sesspar, stimpar, figpar, 
                      parallel=False):
    """
    run_trial_pc_traj(sessions, analysis, analyspar, sesspar, stimpar, figpar)

    Calculates and plots trial trajectories in the first 2 PCs.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., "v")
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - figpar (dict)        : dictionary containing figure parameters

    Optional args:
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores 
    """

    datatype = "roi"

    sessstr_pr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.visflow_dir, 
        stimpar.visflow_size, stimpar.gabk, "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, datatype, "print")
       
    datastr = sess_str_util.datatype_par_str(datatype)

    logger.info(f"Analysing and plotting trial trajectories in 2 principal "
        f"components \n({sessstr_pr}{dendstr_pr}) - INCOMPLETE IMPLEMENTATION.", 
        extra={"spacing": "\n"})

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()
    
    logger.info("Updating stimpar to appropriate values.")
    if stimpar.stimtype == "gabors":
        stimpar = sess_ntuple_util.get_modif_ntuple(
            stimpar, ["gabfr", "pre", "post"], [0, 0, 1.5])
        unexps = [0, 1]
    elif stimpar.stimtype == "visflow":
        stimpar = sess_ntuple_util.get_modif_ntuple(
            stimpar, ["pre", "post"], [4, 8])
        unexps = [1]
    else:
        gen_util.accepted_values_error(
            "stimpar.stimtype", stimpar.stimtype, ["gabors", "visflow"])

    for sess in sessions:
        stim = sess.get_stim(stimpar.stimtype)
        if sess.only_tracked_rois != analyspar.tracked:
            raise RuntimeError(
                "sess.only_tracked_rois should match analyspar.tracked."
                )
        all_traces = []
        for unexp in unexps:        
            all_segs = stim.get_segs_by_criteria(gabfr=stimpar.gabfr,
                gabk=stimpar.gabk, gab_ori=stimpar.gab_ori,
                visflow_dir=stimpar.visflow_dir, 
                visflow_size=stimpar.visflow_size,
                unexp=unexp, by="seg")
            if stimpar.stimtype == "visflow":
                all_segs, n_consec = gen_util.consec(all_segs)
            twop_fr = stim.get_fr_by_seg(
                all_segs, start=True, fr_type="twop")["start_frame_twop"]
            # ROI x sequences (x frames)
            traces_df = stim.get_roi_data(
                twop_fr, stimpar.pre, stimpar.post, fluor=analyspar.fluor, 
                rem_bad=analyspar.rem_bad, scale=analyspar.scale)
            all_traces.append(gen_util.reshape_df_data(
                traces_df, squeeze_cols=True))
        if stimpar.stimtype == "gabors":
            n_exp = len(all_traces[0])
            all_traces = np.concatenate(all_traces, axis=1)
        if stimpar.stimtype == "visflow":
            all_traces = all_traces[0]

        ################
        # Incomplete - obtain PCs


        # extrapar = {"analysis": analysis,
        #             "datatype": datatype,
        #             }

        # all_stats = [sessst.tolist() for sessst in trace_info[1]]
        # trace_stats = {"x_ran"     : trace_info[0].tolist(),
        #                "all_stats" : all_stats,
        #                "all_counts": trace_info[2]
        #               }

        # sess_info = sess_gen_util.get_sess_info(
        #     sessions, analyspar.fluor, incl_roi=(datatype=="roi"), 
        #     rem_bad=analyspar.rem_bad)

        # info = {"analyspar"  : analyspar._asdict(),
        #         "sesspar"    : sesspar._asdict(),
        #         "stimpar"    : stimpar._asdict(),
        #         "quantpar"   : quantpar._asdict(),
        #         "extrapar"   : extrapar,
        #         "sess_info"  : sess_info,
        #         "trace_stats": trace_stats
        #         }

        # fulldir, savename = gen_plots.plot_traces_by_qu_unexp_sess(figpar=figpar, 
        #                                                           **info)
        # file_util.saveinfo(info, savename, fulldir, "json")

      


