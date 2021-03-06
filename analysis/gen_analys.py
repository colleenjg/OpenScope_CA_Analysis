"""
gen_analys.py

This script runs ROI and running trace analyses using a Session object with 
data generated by the Allen Institute OpenScope experiments for the Credit 
Assignment Project.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import copy
import glob
import logging
import os
import warnings

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy.stats as st

from util import file_util, gen_util, logger_util, math_util
from sess_util import sess_gen_util, sess_ntuple_util, sess_str_util
from analysis import pup_analys, ori_analys, quint_analys, signif_grps
from plot_fcts import gen_analysis_plots as gen_plots

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def run_full_traces(sessions, analysis, analyspar, sesspar, figpar, 
                    datatype="roi"):
    """
    run_full_traces(sessions, analysis, analyspar, sesspar, figpar)

    Plots full traces across an entire session. If ROI traces are plotted,
    each ROI is scaled and plotted separately and an average is plotted.
    
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., "t")
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - figpar (dict)        : dictionary containing figure parameters
    
    Optional args:
        - datatype (str): type of data (e.g., "roi", "run")
    """

    dendstr_pr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, datatype, "print")
    
    sessstr_pr = (f"session: {sesspar.sess_n}, "
        f"plane: {sesspar.plane}{dendstr_pr}")

    datastr = sess_str_util.datatype_par_str(datatype)

    logger.info(f"Plotting {datastr} traces across an entire "
        f"session\n({sessstr_pr}).", extra={"spacing": "\n"})

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()
    
    param_names = ["kappa", "direction", "size", "number"]
    all_tr, roi_tr, all_edges, all_pars = [], [], [], []
    for sess in sessions:
        # get the block edges and parameters
        edge_fr, par_descrs = [], []
        for stim in sess.stims:
            if datatype == "roi":
                fr_type = "twop_fr"
            elif datatype == "run":
                fr_type = "stim_fr"
            else:
                gen_util.accepted_values_error(
                    "datatype", datatype, ["roi", "run"])
            for b in stim.block_params.index.unique("block_n"):
                row = stim.block_params.loc[pd.IndexSlice[:, b], ]
                edge_fr.append([row[f"start_{fr_type}", ].tolist()[0], 
                    row[f"end_{fr_type}", ].tolist()[0]])
                params = filter(lambda par: 
                    par in stim.block_params.columns.unique("parameters"), 
                    param_names)
                par_vals = [row[param, ].values[0] for param in params]
                pars_str = "\n".join([str(par) for par in par_vals][0:2])
                par_descrs.append(sess_str_util.pars_to_descr(
                    f"{stim.stimtype.capitalize()}\n{pars_str}"))
            
        if datatype == "roi":
            nanpol = None
            if not analyspar.remnans:
                nanpol = "omit"
            all_rois = gen_util.reshape_df_data(
                sess.get_roi_traces(
                    None, analyspar.fluor, analyspar.remnans, analyspar.scale
                )["roi_traces"], squeeze_cols=True)
            full_tr = math_util.get_stats(
                all_rois, analyspar.stats, analyspar.error, axes=0, 
                nanpol=nanpol).tolist()
            roi_tr.append(all_rois.tolist())
        elif datatype == "run":
            full_tr = sess.get_run_velocity(
                remnans=analyspar.remnans, scale=analyspar.scale
                ).to_numpy().squeeze().tolist()
            roi_tr = None
        all_tr.append(full_tr)
        all_edges.append(edge_fr)
        all_pars.append(par_descrs)

    extrapar = {"analysis": analysis,
                "datatype": datatype,
                }

    trace_info = {"all_tr"   : all_tr,
                  "all_edges": all_edges,
                  "all_pars" : all_pars
                  }

    sess_info = sess_gen_util.get_sess_info(
        sessions, analyspar.fluor, incl_roi=(datatype=="roi"))

    info = {"analyspar" : analyspar._asdict(),
            "sesspar"   : sesspar._asdict(),
            "extrapar"  : extrapar,
            "sess_info" : sess_info,
            "trace_info": trace_info
            }

    fulldir, savename = gen_plots.plot_full_traces(
        roi_tr=roi_tr, figpar=figpar, **info)
    file_util.saveinfo(info, savename, fulldir, "json")
    

#############################################
def run_traces_by_qu_surp_sess(sessions, analysis, analyspar, sesspar, 
                               stimpar, quintpar, figpar, datatype="roi"):
    """
    run_traces_by_qu_surp_sess(sessions, analysis, analyspar, sesspar, 
                               stimpar, quintpar, figpar)

    Retrieves trace statistics by session x surp val x quintile and
    plots traces across ROIs by quintile/surprise with each session in a 
    separate subplot.
    
    Also runs analysis for one quintile (full data).
    
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., "t")
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - quintpar (QuintPar)  : named tuple containing quintile analysis 
                                 parameters
        - figpar (dict)        : dictionary containing figure parameters
    
    Optional args:
        - datatype (str): type of data (e.g., "roi", "run")
    """

    sessstr_pr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.bri_dir, 
        stimpar.bri_size, stimpar.gabk, "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, datatype, "print")
       
    datastr = sess_str_util.datatype_par_str(datatype)

    logger.info(f"Analysing and plotting surprise vs non surprise {datastr} "
        f"traces by quintile ({quintpar.n_quints}) \n({sessstr_pr}"
        f"{dendstr_pr}).", extra={"spacing": "\n"})
    
    # modify quintpar to retain all quintiles
    quintpar_one  = sess_ntuple_util.init_quintpar(1, 0, "", "")
    n_quints      = quintpar.n_quints
    quintpar_mult = sess_ntuple_util.init_quintpar(n_quints, "all")

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()
        
    for quintpar in [quintpar_one, quintpar_mult]:
        logger.info(f"{quintpar.n_quints} quint", extra={"spacing": "\n"})
        # get the stats (all) separating by session, surprise and quintiles    
        trace_info = quint_analys.trace_stats_by_qu_sess(sessions, analyspar, 
            stimpar, quintpar.n_quints, quintpar.qu_idx, byroi=False, 
            bysurp=True, datatype=datatype)
        
        extrapar = {"analysis": analysis,
                    "datatype": datatype,
                    }

        xrans = [xran.tolist() for xran in trace_info[0]]
        all_stats = [sessst.tolist() for sessst in trace_info[1]]
        trace_stats = {"xrans"     : xrans,
                       "all_stats" : all_stats,
                       "all_counts": trace_info[2]
                      }

        sess_info = sess_gen_util.get_sess_info(
            sessions, analyspar.fluor, incl_roi=(datatype=="roi"))

        info = {"analyspar"  : analyspar._asdict(),
                "sesspar"    : sesspar._asdict(),
                "stimpar"    : stimpar._asdict(),
                "quintpar"   : quintpar._asdict(),
                "extrapar"   : extrapar,
                "sess_info"  : sess_info,
                "trace_stats": trace_stats
                }

        fulldir, savename = gen_plots.plot_traces_by_qu_surp_sess(
            figpar=figpar, **info)
        file_util.saveinfo(info, savename, fulldir, "json")

      
#############################################
def run_traces_by_qu_lock_sess(sessions, analysis, seed, analyspar, sesspar, 
                               stimpar, quintpar, figpar, datatype="roi"):
    """
    run_traces_by_qu_lock_sess(sessions, analysis, analyspar, sesspar, 
                               stimpar, quintpar, figpar)

    Retrieves trace statistics by session x quintile at the transition of
    regular to surprise sequences (or v.v.) and plots traces across ROIs by 
    quintile with each session in a separate subplot.
    
    Also runs analysis for one quintile (full data) with different surprise 
    lengths grouped separated 
    
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., "l")
        - seed (int)           : seed value to use. (-1 treated as None)
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - quintpar (QuintPar)  : named tuple containing quintile analysis 
                                 parameters
        - figpar (dict)        : dictionary containing figure parameters
    
    Optional args:
        - datatype (str): type of data (e.g., "roi", "run")

    """

    sessstr_pr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.bri_dir, 
        stimpar.bri_size, stimpar.gabk, "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, datatype, "print")
       
    datastr = sess_str_util.datatype_par_str(datatype)

    logger.info(f"Analysing and plotting surprise vs non surprise {datastr} "
        f"traces locked to surprise onset by quintile ({quintpar.n_quints}) "
        f"\n({sessstr_pr}{dendstr_pr}).", extra={"spacing": "\n"})

    seed = gen_util.seed_all(seed, "cpu", log_seed=False)

    # modify quintpar to retain all quintiles
    quintpar_one  = sess_ntuple_util.init_quintpar(1, 0, "", "")
    n_quints      = quintpar.n_quints
    quintpar_mult = sess_ntuple_util.init_quintpar(n_quints, "all")

    if stimpar.stimtype == "bricks":
        pre_post = [2.0, 6.0]
    elif stimpar.stimtype == "gabors":
        pre_post = [2.0, 8.0]
    else:
        gen_util.accepted_values_error(
            "stimpar.stimtype", stimpar.stimtype, ["bricks", "gabors"])
    logger.warning("Setting pre to {}s and post to {}s.".format(*pre_post))
    
    stimpar = sess_ntuple_util.get_modif_ntuple(
        stimpar, ["pre", "post"], pre_post)

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()
        
    for baseline in [None, stimpar.pre]:
        basestr_pr = sess_str_util.base_par_str(baseline, "print")
        for quintpar in [quintpar_one, quintpar_mult]:
            locks = ["surp", "reg"]
            if quintpar.n_quints == 1:
                locks.append("surp_split")
            # get the stats (all) separating by session and quintiles
            for lock in locks:
                logger.info(
                    f"{quintpar.n_quints} quint, {lock} lock{basestr_pr}", 
                    extra={"spacing": "\n"})
                if lock == "surp_split":
                    trace_info = quint_analys.trace_stats_by_surp_len_sess(
                        sessions, analyspar, stimpar, quintpar.n_quints, 
                        quintpar.qu_idx, byroi=False, nan_empty=True, 
                        baseline=baseline, datatype=datatype)
                else:
                    trace_info = quint_analys.trace_stats_by_qu_sess(
                        sessions, analyspar, stimpar, quintpar.n_quints, 
                        quintpar.qu_idx, byroi=False, lock=lock, nan_empty=True, 
                        baseline=baseline, datatype=datatype)

                # for comparison, locking to middle of regular sample (1 quint)
                reg_samp = quint_analys.trace_stats_by_qu_sess(
                    sessions, analyspar, stimpar, quintpar_one.n_quints, 
                    quintpar_one.qu_idx, byroi=False, lock="regsamp", 
                    nan_empty=True, baseline=baseline, datatype=datatype)

                extrapar = {"analysis": analysis,
                            "datatype": datatype,
                            "seed"    : seed,
                            }

                xrans = [xran.tolist() for xran in trace_info[0]]
                all_stats = [sessst.tolist() for sessst in trace_info[1]]
                reg_stats = [regst.tolist() for regst in reg_samp[1]]
                trace_stats = {"xrans"     : xrans,
                               "all_stats" : all_stats,
                               "all_counts": trace_info[2],
                               "lock"      : lock,
                               "baseline"  : baseline,
                               "reg_stats" : reg_stats,
                               "reg_counts": reg_samp[2]
                               }

                if lock == "surp_split":
                    trace_stats["surp_lens"] = trace_info[3]

                sess_info = sess_gen_util.get_sess_info(
                    sessions, analyspar.fluor, incl_roi=(datatype=="roi"))

                info = {"analyspar"  : analyspar._asdict(),
                        "sesspar"    : sesspar._asdict(),
                        "stimpar"    : stimpar._asdict(),
                        "quintpar"   : quintpar._asdict(),
                        "extrapar"   : extrapar,
                        "sess_info"  : sess_info,
                        "trace_stats": trace_stats
                        }

                fulldir, savename = gen_plots.plot_traces_by_qu_lock_sess(
                    figpar=figpar, **info)
                file_util.saveinfo(info, savename, fulldir, "json")

      
#############################################
def run_mag_change(sessions, analysis, seed, analyspar, sesspar, stimpar, 
                   permpar, quintpar, figpar, datatype="roi"):
    """
    run_mag_change(sessions, analysis, seed, analyspar, sesspar, stimpar, 
                   permpar, quintpar, figpar)

    Calculates and plots the magnitude of change in activity of ROIs between 
    the first and last quintile for non surprise vs surprise sequences.
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., "m")
        - seed (int)           : seed value to use. (-1 treated as None) 
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - permpar (PermPar)    : named tuple containing permutation parameters
        - quintpar (QuintPar)  : named tuple containing quintile analysis 
                                 parameters
        - figpar (dict)        : dictionary containing figure parameters   

    Optional args:
        - datatype (str): type of data (e.g., "roi", "run") 
    """

    sessstr_pr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.bri_dir,
        stimpar.bri_size, stimpar.gabk, "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, datatype, "print")
  
    datastr = sess_str_util.datatype_par_str(datatype)

    logger.info(f"Calculating and plotting the magnitude changes in {datastr} "
        f"activity across quintiles \n({sessstr_pr}{dendstr_pr})", 
        extra={"spacing": "\n"})

    
    if permpar.multcomp:
        warnings.warn("Multiple comparisons not implemented for magnitude "
            "analysis. Setting to False.")
        permpar = sess_ntuple_util.get_modif_ntuple(permpar, "multcomp", False)

    # get full data: session x surp x quints of interest x [ROI x seq]
    integ_info = quint_analys.trace_stats_by_qu_sess(
        sessions, analyspar, stimpar, quintpar.n_quints, quintpar.qu_idx, 
        bysurp=True, integ=True, ret_arr=True, datatype=datatype)
    all_counts = integ_info[-2]
    qu_data = integ_info[-1]

    # extract session info
    mouse_ns = [sess.mouse_n for sess in sessions]
    lines    = [sess.line for sess in sessions]

    if analyspar.remnans:
        nanpol = None
    else:
        nanpol = "omit"

    seed = gen_util.seed_all(seed, "cpu", log_seed=False)

    mags = quint_analys.qu_mags(
        qu_data, permpar, mouse_ns, lines, analyspar.stats, analyspar.error, 
        nanpol=nanpol, op_qu="diff", op_surp="diff")

    # convert mags items to list
    mags = copy.deepcopy(mags)
    mags["all_counts"] = all_counts
    for key in ["mag_st", "L2", "mag_rel_th", "L2_rel_th"]:
        mags[key] = mags[key].tolist()

    sess_info = sess_gen_util.get_sess_info(
        sessions, analyspar.fluor, incl_roi=(datatype=="roi"))
    extrapar  = {"analysis": analysis,
                 "datatype": datatype,
                 "seed"    : seed
                 }

    info = {"analyspar": analyspar._asdict(),
            "sesspar": sesspar._asdict(),
            "stimpar": stimpar._asdict(),
            "extrapar": extrapar,
            "permpar": permpar._asdict(),
            "quintpar": quintpar._asdict(),
            "mags": mags,
            "sess_info": sess_info
            }
    
    fulldir, savename = gen_plots.plot_mag_change(figpar=figpar, **info)

    file_util.saveinfo(info, savename, fulldir, "json")


#############################################
def run_autocorr(sessions, analysis, analyspar, sesspar, stimpar, autocorrpar, 
                 figpar, datatype="roi"):
    """
    run_autocorr(sessions, analysis, analyspar, sesspar, stimpar, autocorrpar, 
                 figpar)

    Calculates and plots autocorrelation during stimulus blocks.
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)          : list of Session objects
        - analysis (str)           : analysis type (e.g., "a")
        - analyspar (AnalysPar)    : named tuple containing analysis parameters
        - sesspar (SessPar)        : named tuple containing session parameters
        - stimpar (StimPar)        : named tuple containing stimulus parameters
        - autocorrpar (AutocorrPar): named tuple containing autocorrelation 
                                     analysis parameters
        - figpar (dict)            : dictionary containing figure parameters

    Optional args:
        - datatype (str): type of data (e.g., "roi", "run")
    """

    sessstr_pr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.bri_dir,
        stimpar.bri_size, stimpar.gabk, "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, datatype, "print")
  
    datastr = sess_str_util.datatype_par_str(datatype)

    logger.info(f"Analysing and plotting {datastr} autocorrelations " 
        f"({sessstr_pr}{dendstr_pr}).", extra={"spacing": "\n"})

    xrans = []
    stats = []
    for sess in sessions:
        stim = sess.get_stim(stimpar.stimtype)
        all_segs = stim.get_segs_by_criteria(
            bri_dir=stimpar.bri_dir, bri_size=stimpar.bri_size, 
            gabk=stimpar.gabk, by="block")
        sess_traces = []
        for segs in all_segs:
            if len(segs) == 0:
                continue
            segs = sorted(segs)
            # check that segs are contiguous
            if max(np.diff(segs)) > 1:
                raise NotImplementedError("Segments used for autocorrelation "
                    "must be contiguous within blocks.")
            if datatype == "roi":
                frame_edges = stim.get_twop_fr_by_seg([min(segs), max(segs)])
                fr = list(range(min(frame_edges[0]), max(frame_edges[1])+1))
                traces = gen_util.reshape_df_data(
                    sess.get_roi_traces(fr, fluor=analyspar.fluor, 
                        remnans=analyspar.remnans, scale=analyspar.scale), 
                    squeeze_cols=True)

            elif datatype == "run":
                if autocorrpar.byitem != False:
                    raise ValueError("autocorrpar.byitem must be False for "
                        "running data.")
                frame_edges = stim.get_stim_fr_by_seg([min(segs), max(segs)])
                fr = list(range(min(frame_edges[0]), max(frame_edges[1])+1))
                
                traces = sess.get_run_velocity_by_fr(fr, fr_type="stim", 
                    remnans=analyspar.remnans, scale=analyspar.scale
                    ).to_numpy().reshape(1, -1)
                
            sess_traces.append(traces)
        xran, ac_st = math_util.autocorr_stats(
            sess_traces, autocorrpar.lag_s, sess.twop_fps, 
            byitem=autocorrpar.byitem, stats=analyspar.stats, 
            error=analyspar.error)
        if not autocorrpar.byitem: # also add a 10x lag
            lag_fr = 10 * int(autocorrpar.lag_s * sess.twop_fps)
            _, ac_st_10x = math_util.autocorr_stats(
                sess_traces, lag_fr, byitem=autocorrpar.byitem, 
                stats=analyspar.stats, error=analyspar.error)
            downsamp = range(0, ac_st_10x.shape[-1], 10)
            if len(downsamp) != ac_st.shape[-1]:
                raise ValueError("Failed to downsample correctly. "
                    "Check implementation.")
            ac_st = np.stack([ac_st, ac_st_10x[:, downsamp]], axis=1)
        xrans.append(xran)
        stats.append(ac_st)

    autocorr_data = {"xrans": [xran.tolist() for xran in xrans],
                     "stats": [stat.tolist() for stat in stats]
                     }

    sess_info = sess_gen_util.get_sess_info(
        sessions, analyspar.fluor, incl_roi=(datatype=="roi"))
    extrapar  = {"analysis": analysis,
                 "datatype": datatype,
                 }

    info = {"analyspar"     : analyspar._asdict(),
            "sesspar"       : sesspar._asdict(),
            "stimpar"       : stimpar._asdict(),
            "extrapar"      : extrapar,
            "autocorrpar"   : autocorrpar._asdict(),
            "autocorr_data" : autocorr_data,
            "sess_info"     : sess_info
            }

    fulldir, savename = gen_plots.plot_autocorr(figpar=figpar, **info)

    file_util.saveinfo(info, savename, fulldir, "json")


#############################################
def run_trace_corr_acr_sess(sessions, analysis, analyspar, sesspar, 
                            stimpar, figpar, datatype="roi"):
    """
    run_trace_corr_acr_sess(sessions, analysis, analyspar, sesspar, 
                             stimpar, quintpar, figpar)

    Retrieves trace statistics by session x surp val and calculates 
    correlations across sessions per surp val.
    
    Currently only logs results to the console. Does NOT save results and 
    parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., "r")
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - figpar (dict)        : dictionary containing figure parameters
    
    Optional args:
        - datatype (str): type of data (e.g., "roi", "run")
    """

    sessstr_pr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.bri_dir, 
        stimpar.bri_size, stimpar.gabk,"print")
    # dendstr_pr = sess_str_util.dend_par_str(
    # analyspar.dend, sesspar.plane, datatype, "print")
       
    datastr = sess_str_util.datatype_par_str(datatype)

    if sesspar.plane in ["any", "all"] and sesspar.runtype == "pilot":
        logger.warning("Planes may not match between sessions for a mouse!")

    logger.info("Analysing and plotting correlations between surprise vs non "
        f"surprise {datastr} traces between sessions ({sessstr_pr}).", 
        extra={"spacing": "\n"})

    figpar = copy.deepcopy(figpar)
    if figpar["save"]["use_dt"] is None:
        figpar["save"]["use_dt"] = gen_util.create_time_str()

    prev_level = logger.level
    if prev_level > logging.INFO:
        logger.setLevel(logging.INFO)
        logger.warning("Temporarily lowered log level for correlation "
            "analysis results.")

    # correlate average traces between sessions for each mouse and each surprise
    # value   
    all_counts = []
    all_me_tr = []
    all_corrs = []
    logger.info("Intramouse correlations", extra={"spacing": "\n"})
    for sess_grp in sessions:
        logger.info(f"Mouse {sess_grp[0].mouse_n}, sess {sess_grp[0].sess_n} "
            f"vs {sess_grp[1].sess_n} corr:")
        trace_info = quint_analys.trace_stats_by_qu_sess(sess_grp, analyspar, 
            stimpar, 1, [0], byroi=False, bysurp=True, datatype=datatype)
        # remove quint dim
        grp_stats = np.asarray(trace_info[1]).squeeze(2) 
        all_counts.append([[qu_c[0] for qu_c in c] for c in trace_info[2]])
        # get mean/median per grp (sess x surp_val x frame)
        grp_me = grp_stats[:, :, 0]
        grp_corrs = []
        for s, surp in enumerate(["reg", "surp"]):
            corr = st.pearsonr(grp_me[0, s], grp_me[1, s])
            logger.info(f"{surp}: {corr[0]:.4f} (p={corr[1]:.2f})", 
                extra={"spacing": TAB})
            corr = corr[0]
            grp_corrs.append(corr)
        all_corrs.append(grp_corrs)
        all_me_tr.append(grp_me)

    # mice x sess x surp x frame
    all_me_tr = np.asarray(all_me_tr)
    logger.info("Intermouse correlations", extra={"spacing": "\n"})
    all_mouse_corrs = []
    for n, m1_sess_mes in enumerate(all_me_tr):
        if n + 1 < len(all_me_tr):
            mouse_corrs = []
            for n_add, m2_sess_mes in enumerate(all_me_tr[n + 1 :]):
                sess_corrs = []
                logger.info(f"Mouse {sessions[n][0].mouse_n} vs "
                    f"{sessions[n + 1 + n_add][0].mouse_n} corr:")
                for se, m1_s1_me in enumerate(m1_sess_mes):
                    surp_corrs = []
                    logger.info(f"sess {sessions[n][se].sess_n}:", 
                        extra={"spacing": TAB})
                    for s, surp in enumerate(["reg", "surp"]):
                        corr = st.pearsonr(m1_s1_me[s], m2_sess_mes[se][s])
                        logger.info(
                            f"{surp}: {corr[0]:.4f} (p={corr[1]:.2f})", 
                            extra={"spacing": f"{TAB}{TAB}"})
                        corr = corr[0]
                        surp_corrs.append(corr)
                    sess_corrs.append(surp_corrs)
                mouse_corrs.append(sess_corrs)
            all_mouse_corrs.append(mouse_corrs)

    # reset logger level
    logger.setLevel(prev_level)

    # CURRENTLY RESULTS ARE ONLY PRINTED TO THE CONSOLE, NOT SAVED
    # extrapar = {"analysis": analysis,
    #             "datatype": datatype,
    #             }

    # corr_data = {"all_corrs"      : all_corrs,
    #              "all_mouse_corrs": all_mouse_corrs,
    #              "all_counts"     : all_counts
    #             }
    
    # sess_info = []
    # for sess_grp in sessions:
    #     sess_info.append(sess_gen_util.get_sess_info(sess_grp, analyspar.fluor, 
    #                                    incl_roi=(datatype=="roi")))

    # info = {"analyspar": analyspar._asdict(),
    #         "sesspar"  : sesspar._asdict(),
    #         "stimpar"  : stimpar._asdict(),
    #         "extrapar" : extrapar,
    #         "sess_info": sess_info,
    #         "corr_data": corr_data
    #         }

    # savename = "trace_corr" # should be modified to include session info
    # file_util.saveinfo(info, savename, fulldir, "json")


