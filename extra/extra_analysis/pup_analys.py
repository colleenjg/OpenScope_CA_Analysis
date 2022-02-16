"""
pup_analys.py

This module analyses pupil data generated by the Allen Institute OpenScope 
experiments for the Credit Assignment Project.

Authors: Jay Pina and Colleen Gillon

Date: July, 2019

Note: this code uses python 3.7.

"""

import copy
import warnings

from joblib import Parallel, delayed
import numpy as np

from util import file_util, gen_util, logger_util, math_util
from sess_util import sess_gen_util, sess_str_util, sess_ntuple_util
from extra_plot_fcts import pup_analysis_plots as pup_plots


logger = logger_util.get_module_logger(name=__name__)


#############################################
def get_ran_s(ran_s=None, datatype="both"):
    """
    get_ran_s()

    Ensures that ran_s is a dictionary and has the correct keys or initializes 
    it as a dictionary if needed, and returns it.

    Optional args:
        - ran_s (dict, list or num): number of frames to take before and after 
                                     unexpected events for each datatype 
                                     (ROI, run, pupil) (in sec). 
                                        If dictionary, expected keys are:
                                            "pup_pre", "pup_post", 
                                            ("roi_pre", "roi_post"), 
                                            ("run_pre", "run_post"), 
                                        If list, should be structured as 
                                        [pre, post] and the same values will be 
                                        used for all datatypes. 
                                        If num, the same value will be used 
                                            for all keys. 
                                        If None, all keys are initialized with 
                                            3.5.
        - datatype (str)           : if "roi", roi keys are included, if "run", 
                                     run keys are included. If "both", all keys
                                     are included.
                                     default: "both"
    Returns:                
        - ran_s (dict): dictionary specifying number of frames to take before 
                        and after unexpected events for each datatype 
                        (ROI, run, pupil), with keys: 
                        ("roi_pre", "roi_post"), ("run_pre", "run_post"), 
                        "pup_pre", "pup_post"
    """

    ran_s = copy.deepcopy(ran_s)

    keys = ["pup_pre", "pup_post"]
    if datatype in ["both", "roi"]:
        keys.extend(["roi_pre", "roi_post"])
    if datatype in ["both", "run"]:
        keys.extend(["run_pre", "run_post"])
    elif datatype not in ["both", "run", "roi"]:
        gen_util.accepted_values_error(
            "datatype", datatype, ["both", "roi", "run"])

    if isinstance(ran_s, dict):
        missing = [key for key in keys if key not in ran_s.keys()]
        if len(missing) > 0:
            raise KeyError(f"'ran_s' is missing keys: {', '.join(missing)}")
    else:
        if ran_s is None:
            vals = 3.5
        elif isinstance(ran_s, list):
            if len(ran_s) == 2:
                vals = ran_s[:]
            else:
                raise ValueError("If 'ran_s' is a list, must be of length 2.")
        else:
            vals = [ran_s, ran_s]
        ran_s = dict()
        for key in keys:
            if "pre" in key:
                ran_s[key] = vals[0]
            elif "post" in key:
                ran_s[key] = vals[1]

    return ran_s


#############################################
def peristim_data(sess, stimpar, ran_s=None, datatype="both",
                  returns="diff", fluor="dff", stats="mean", 
                  rem_bad=True, scale=False, first_unexp=True, trans_all=False):
    """
    peristim_data(sess, stimpar)

    Returns pupil, ROI and run data around unexpected onset, or the difference 
    between post and pre unexpected onset, or both.

    Required args:
        - sess (Session)   : session object
        - stimpar (StimPar): named tuple containing stimulus parameters

    Optional args:
        - ran_s (dict, list or num): number of frames to take before and after 
                                     unexpected for each datatype (ROI, run, 
                                     pupil) (in sec).  
                                         If dictionary, expected keys are:
                                            "pup_pre", "pup_post", 
                                            ("roi_pre", "roi_post"), 
                                            ("run_pre", "run_post"), 
                                        If list, should be structured as 
                                        [pre, post] and the same values will be 
                                        used for all datatypes. 
                                        If num, the same value will be used 
                                            for all keys. 
                                        If None, the values are taken from the
                                            stimpar pre and post attributes.
                                     default: None
        - datatype (str)           : type of data to include with pupil data, 
                                     "roi", "run" or "both"
                                     default: "roi" 
        - returns (str)            : type of data to return (data around 
                                     unexpected, difference between post and pre 
                                     unexpected)
                                     default: "diff"
        - fluor (str)              : if "dff", dF/F is used, if "raw", ROI 
                                     traces
                                     default: "dff"
        - stats (str)              : measure on which to take the pre and post
                                     unexpected difference: either mean ("mean") 
                                     or median ("median")
                                     default: "mean"
        - rem_bad (bool)           : if True, removes ROIs with NaN/Inf values 
                                     anywhere in session and running array with
                                     NaNs linearly interpolated is used. If 
                                     False, NaNs are ignored in calculating 
                                     statistics for the ROI and running data 
                                     (always ignored for pupil data)
                                     default: True
        - scale (bool)             : if True, data is scaled
                                     default: False
        - first_unexp (bool)        : if True, only the first of consecutive 
                                     unexpecteds are retained
                                     default: True
        - trans_all (bool)         : if True, only ROIs with transients are 
                                     retained
                                     default: False

    Returns:
        if datatype == "data" or "both":
        - datasets (list): list of 2-3D data arrays, structured as
                               datatype (pupil, (ROI), (running)) x 
                               [trial x frames (x ROI)]
        elif datatype == "diff" or "both":
        - diffs (list)   : list of 1-2D data difference arrays, structured as
                               datatype (pupil, (ROI), (running)) x 
                               [trial (x ROI)]    
    """

    stim = sess.get_stim(stimpar.stimtype)

    # initialize ran_s dictionary if needed
    if ran_s is None:
        ran_s = [stimpar.pre, stimpar.post]
    ran_s = get_ran_s(ran_s, datatype)

    if first_unexp:
        unexp_segs = stim.get_segs_by_criteria(
            visflow_dir=stimpar.visflow_dir, visflow_size=stimpar.visflow_size, 
            gabk=stimpar.gabk, unexp=1, remconsec=True, by="seg")
        if stimpar.stimtype == "gabors":
            unexp_segs = [seg + stimpar.gabfr for seg in unexp_segs]
    else:
        unexp_segs = stim.get_segs_by_criteria(
            visflow_dir=stimpar.visflow_dir, visflow_size=stimpar.visflow_size, 
            gabk=stimpar.gabk, gabfr=stimpar.gabfr, unexp=1, remconsec=False, 
            by="seg")
    
    unexp_twopfr = stim.get_fr_by_seg(
        unexp_segs, start=True, fr_type="twop")["start_frame_twop"]
    unexp_stimfr = stim.get_fr_by_seg(
        unexp_segs, start=True, fr_type="stim")["start_frame_stim"]
    # get data dataframes
    pup_data = gen_util.reshape_df_data(stim.get_pup_diam_data(
        unexp_twopfr, ran_s["pup_pre"], ran_s["pup_post"], 
        rem_bad=rem_bad, scale=scale)["pup_diam"], squeeze_cols=True)

    datasets = [pup_data]
    datanames = ["pup"]
    if datatype in ["roi", "both"]:
        # ROI x trial x fr
        roi_data = gen_util.reshape_df_data(stim.get_roi_data(
            unexp_twopfr, ran_s["roi_pre"], ran_s["roi_post"], fluor=fluor, 
            integ=False, rem_bad=rem_bad, scale=scale, 
            transients=trans_all)["roi_traces"], squeeze_cols=True) 
        datasets.append(roi_data.transpose([1, 2, 0])) # ROIs last
        datanames.append("roi")
    if datatype in ["run", "both"]:
        run_data = gen_util.reshape_df_data(stim.get_run_data(
            unexp_stimfr, ran_s["run_pre"], ran_s["run_post"], rem_bad=rem_bad, 
            scale=scale), squeeze_cols=True)
        datasets.append(run_data)
        datanames.append("run")

    if rem_bad:
        nanpolgen = None
    else:
        nanpolgen = "omit"

    if returns in ["diff", "both"]:
        for key in ran_s.keys():
            if "pre" in key and ran_s[key] == 0:
                raise ValueError(
                    "Cannot set pre to 0 if returns is 'diff' or 'both'."
                    )
        # get avg for first and second halves
        diffs = []
        for dataset, name in zip(datasets, datanames):
            if name == "pup":
                nanpol = "omit"
            else:
                nanpol = nanpolgen
            n_fr = dataset.shape[1]
            pre_s  = ran_s[f"{name}_pre"]
            post_s = ran_s[f"{name}_post"]
            split = int(np.round(pre_s/(pre_s + post_s) * n_fr)) # find 0 mark
            pre  = math_util.mean_med(dataset[:, :split], stats, 1, nanpol)
            post = math_util.mean_med(dataset[:, split:], stats, 1, nanpol)
            diffs.append(post - pre)

    if returns == "data":
        return datasets
    elif returns == "diff":
        return diffs
    elif returns == "both":
        return datasets, diffs
    else:
        gen_util.accepted_values_error(
            "returns", returns, ["data", "diff", "both"])


#############################################
def run_pupil_diff_corr(sessions, analysis, analyspar, sesspar, 
                        stimpar, figpar, datatype="roi"):
    """
    run_pupil_diff_corr(sessions, analysis, analyspar, sesspar, 
                        stimpar, figpar)
    
    Calculates and plots between pupil and ROI/running changes
    locked to each unexpected, as well as the correlation.

    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., "c")
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - figpar (dict)        : dictionary containing figure parameters
    
    Optional args:
        - datatype (str): type of data (e.g., "roi", "run")
    """

    sessstr_pr = sess_str_util.sess_par_str(
        sesspar.sess_n, stimpar.stimtype, sesspar.plane, stimpar.visflow_dir, 
        stimpar.visflow_size, stimpar.gabk, "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, datatype, "print")
       
    datastr = sess_str_util.datatype_par_str(datatype)

    logger.info("Analysing and plotting correlations between unexpected vs "
        f"expected {datastr} traces between sessions ({sessstr_pr}"
        f"{dendstr_pr}).", extra={"spacing": "\n"})

    sess_diffs = []
    sess_corr = []
    
    for sess in sessions:
        if datatype == "roi" and (sess.only_tracked_rois != analyspar.tracked):
            raise RuntimeError(
                "sess.only_tracked_rois should match analyspar.tracked."
                )
        diffs = peristim_data(
            sess, stimpar, datatype=datatype, returns="diff", 
            scale=analyspar.scale, first_unexp=True)
        [pup_diff, data_diff] = diffs 
        # trials (x ROIs)
        if datatype == "roi":
            if analyspar.rem_bad:
                nanpol = None
            else:
                nanpol = "omit"
            data_diff = math_util.mean_med(
                data_diff, analyspar.stats, axis=-1, nanpol=nanpol)
        elif datatype != "run":
            gen_util.accepted_values_error(
                "datatype", datatype, ["roi", "run"])
        sess_corr.append(np.corrcoef(pup_diff, data_diff)[0, 1])
        sess_diffs.append([diff.tolist() for diff in [pup_diff, data_diff]])
    
    extrapar = {"analysis": analysis,
                "datatype": datatype,
                }
    
    corr_data = {"corrs": sess_corr,
                 "diffs": sess_diffs
                 }

    sess_info = sess_gen_util.get_sess_info(
        sessions, analyspar.fluor, incl_roi=(datatype=="roi"), 
        rem_bad=analyspar.rem_bad)
    
    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "extrapar" : extrapar,
            "sess_info": sess_info,
            "corr_data": corr_data
            }

    fulldir, savename = pup_plots.plot_pup_diff_corr(figpar=figpar, **info)

    file_util.saveinfo(info, savename, fulldir, "json")


#############################################
def run_pup_roi_stim_corr(sessions, analysis, analyspar, sesspar, stimpar, 
                          figpar, datatype="roi", parallel=False):
    """
    run_pup_roi_stim_corr(sessions, analysis, analyspar, sesspar, stimpar, 
                          figpar)
    
    Calculates and plots correlation between pupil and ROI changes locked to
    unexpected for gabors vs visflow.
    
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., "r")
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - figpar (dict)        : dictionary containing figure parameters
    
    Optional args:
        - datatype (str) : type of data (e.g., "roi", "run")
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
                           default: False
    """

    if datatype != "roi":
        raise NotImplementedError("Analysis only implemented for roi datatype.")

    stimtypes = ["gabors", "visflow"]
    if stimpar.stimtype != "both":
        non_stimtype = stimtypes[1 - stimtypes.index(stimpar.stimtype)]
        warnings.warn("stimpar.stimtype will be set to 'both', but non default "
            f"{non_stimtype} parameters are lost.", 
            category=RuntimeWarning, stacklevel=1)
        stimpar_dict = stimpar._asdict()
        for key in list(stimpar_dict.keys()): # remove any "none"s
            if stimpar_dict[key] == "none":
                stimpar_dict.pop(key)

    sessstr_pr = f"session: {sesspar.sess_n}, plane: {sesspar.plane}"
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar.dend, sesspar.plane, datatype, "print")
    stimstr_pr = []
    stimpars = []
    for stimtype in stimtypes:
        stimpar_dict["stimtype"] = stimtype
        stimpar_dict["gabfr"] = 3
        stimpars.append(sess_ntuple_util.init_stimpar(**stimpar_dict))
        stimstr_pr.append(sess_str_util.stim_par_str(
            stimtype, stimpars[-1].visflow_dir, stimpars[-1].visflow_size, 
            stimpars[-1].gabk, "print"))
    stimpar_dict = stimpars[0]._asdict()
    stimpar_dict["stimtype"] = "both"

    logger.info("Analysing and plotting correlations between unexpected vs "
          f"expected ROI traces between sessions ({sessstr_pr}{dendstr_pr}).", 
          extra={"spacing": "\n"})
    sess_corrs = []
    sess_roi_corrs = []
    for sess in sessions:
        if datatype == "roi" and (sess.only_tracked_rois != analyspar.tracked):
            raise RuntimeError(
                "sess.only_tracked_rois should match analyspar.tracked."
                )
        stim_corrs = []
        for sub_stimpar in stimpars:
            diffs = peristim_data(
                sess, sub_stimpar, datatype="roi", returns="diff", 
                first_unexp=True, rem_bad=analyspar.rem_bad, 
                scale=analyspar.scale)
            [pup_diff, roi_diff] = diffs 
            nrois = roi_diff.shape[-1]
            # optionally runs in parallel
            if parallel and nrois > 1:
                n_jobs = gen_util.get_n_jobs(nrois)
                with gen_util.ParallelLogging():
                    corrs = Parallel(n_jobs=n_jobs)(
                        delayed(np.corrcoef)
                        (roi_diff[:, r], pup_diff) for r in range(nrois))
                corrs = np.asarray([corr[0, 1] for corr in corrs])
            else:
                corrs = np.empty(nrois)
                for r in range(nrois): # cycle through ROIs
                    corrs[r] = np.corrcoef(roi_diff[:, r], pup_diff)[0, 1]
            stim_corrs.append(corrs)
        sess_corrs.append(np.corrcoef(stim_corrs[0], stim_corrs[1])[0, 1])
        sess_roi_corrs.append([corrs.tolist() for corrs in stim_corrs])

    extrapar = {"analysis": analysis,
                "datatype": datatype,
                }
    
    corr_data = {"stim_order": stimtypes,
                 "roi_corrs" : sess_roi_corrs,
                 "corrs"     : sess_corrs
                 }

    sess_info = sess_gen_util.get_sess_info(
        sessions, analyspar.fluor, incl_roi=(datatype=="roi"), 
        rem_bad=analyspar.rem_bad)
    
    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar_dict,
            "extrapar" : extrapar,
            "sess_info": sess_info,
            "corr_data": corr_data
            }

    fulldir, savename = pup_plots.plot_pup_roi_stim_corr(figpar=figpar, **info)

    file_util.saveinfo(info, savename, fulldir, "json")

