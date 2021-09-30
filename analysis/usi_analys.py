"""
usi_analys.py

This script contains functions for USI (index) analysis.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import copy
import logging

import numpy as np
import pandas as pd
import scipy.stats as scist

from util import gen_util, math_util, logger_util
from analysis import basic_analys, misc_analys
from plot_fcts import plot_helper_fcts

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def sess_stim_idxs(sess, analyspar, stimpar, n_perms=1000, split="by_exp", 
                   op="d-prime", baseline=0.0, common_oris=False, seed=None, 
                   run_random=True):
    """
    sess_stim_idxs(sess, analyspar, stimpar)
    
    Returns session ROI indices, using the specified split. 

    If run_random, also return index percentiles and the random permutations 
    used to calculate them.

    Required args:
        - sess (Session): 
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - n_perms (int): 
            number of permutations for CI estimation
            default: 1000
        - split (str): 
            how to split data:
            "by_exp" (all exp, all unexp), 
            "unexp_lock" (unexp, preceeding exp), 
            "exp_lock" (exp, preceeding unexp),
            "stim_onset" (grayscr, stim on), 
            "stim_offset" (stim off, grayscr)
            default: "by_exp"
        - op (str): 
            operation to use in measuring indices 
            ("diff", "rel_diff", "d-prime")
            default: "d-prime"
        - baseline (bool or num): 
            if not False, number of second to use for baseline (not implemented)
            default: 0.0
        - common_oris (bool): 
            if True, only Gabor stimulus orientations common to D and U frames 
            are included ("by_exp" feature only)
            default: False
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None
        - run_random (bool): 
            if True, randomization is run and results are returns 
            (roi_percs, all_rand)
            default: True

    Returns:
        - roi_idxs (1D array): 
            ROI indices for the session
        
        if run_random:
        - roi_percs (1D array): 
            ROI index percentiles for the session, based on each ROI's random 
            permutations
        - all_rand (2D array): 
            ROI indices calculated through randomized permutation
            dims: item x n_perms
    """

    seed = gen_util.seed_all(seed, "cpu", log_seed=False)

    nanpol = None if analyspar.remnans else "omit"

    data_arr, _ = basic_analys.get_split_data_by_sess(
        sess, analyspar, stimpar, split=split, integ=True, baseline=baseline, 
        common_oris=common_oris
        )
    
    if op == "d-prime":
        idx_data = data_arr
        axis = -1
    else:
        # take statistic across sequences
        idx_data = np.stack([math_util.mean_med(
            arr, stats=analyspar.stats, axis=-1, nanpol=nanpol) 
            for arr in data_arr])
        axis = None

    # take relative difference (index)
    roi_idxs = math_util.calc_op(idx_data, op=op, nanpol=nanpol, axis=axis)

    if not run_random:
        return roi_idxs

    # randomized indices (items x perms)
    last_dim = np.sum(
        [np.asarray(split_data).shape[-1] for split_data in data_arr]
        )
    data_concat = np.concatenate(data_arr, axis=-1).reshape(-1, last_dim)

    div = np.asarray(data_arr[0]).shape[-1] # length of first data split
    all_rand = math_util.permute_diff_ratio(
        data_concat, div=div, n_perms=n_perms, stats=analyspar.stats, 
        nanpol=nanpol, op=op
        )

    # true index percentiles wrt randomized indices
    roi_percs = np.empty(len(roi_idxs))
    for r, (roi_idx, roi_rand) in enumerate(zip(roi_idxs, all_rand)):
        roi_percs[r] = scist.percentileofscore(
            roi_rand, roi_idx, kind="mean")
    
    return roi_idxs, roi_percs, all_rand
    
    
#############################################
def get_idx_info(sess, analyspar, stimpar, basepar, idxpar, permpar,  
                 seed=None):
    """
    get_idx_info(sess, analyspar, stimpar, basepar, idxpar, permpar)

    Returns ROI index values and ROI trace smoothness measure for a session's 
    ROIs.

    Required args:
        - sess (Session object): 
            session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - idxpar (IdxPar): 
            named tuple containing index parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters

    Optional args:
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None

    Returns:
        - roi_idxs (1D array) : 
            feature index for each ROI
        - roi_percs (1D array): 
            feature index percentile, relative to its null distribution, for 
            each ROI
        - roi_mses (1D array): 
            MSE between original and smoothed mean trace, for each ROI 
            (trace smoothness measure)
    """

    roi_idxs, roi_percs, _ = sess_stim_idxs(
        sess, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        n_perms=permpar.n_perms, 
        split=idxpar.feature, 
        op=idxpar.op, 
        baseline=basepar.baseline, 
        seed=seed
        )

    # split x ROIs x frames
    roi_trace_stats, _ = basic_analys.get_sess_roi_trace_stats(
        sess, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        split=idxpar.feature
        ) 

    stats_me = roi_trace_stats[..., 0] # retain mean/median only
    stats_me_smoothed = math_util.rolling_mean(stats_me, win=3)
    roi_mses = np.nanmean((stats_me - stats_me_smoothed) ** 2, axis=(0, 2))

    return roi_idxs, roi_percs, roi_mses


#############################################
def choose_roi(target_idx_val, target_idx_sig, roi_idxs, roi_percs,  
               roi_mses, nrois_per_sess, permpar):
    """
    choose_roi(target_idx_val, target_idx_sig, roi_idxs,  roi_percs,
               roi_mses, nrois_per_sess, permpar)

    Returns number of the chosen ROI.

    Required args:
        - target_idx_val (float): 
            target USI value
        - target_idx_sig (str): 
            target USI significance ("sig" or "not_sig")
        - roi_idxs (1D array): 
            feature index for ROIs concatenated across sessions
        - roi_percs (1D array): 
            feature index percentile, relative to its null distribution, for 
            ROIs concatenated across sessions
        - roi_mses  (1D array): 
            MSE between original and smoothed mean trace, for ROIs concatenated 
            across sessions (trace smoothness measure)
        - nrois_per_sess (list): 
            number of ROIs per session
        - permpar (PermPar): 
            named tuple containing permutation parameters

    Returns:
        - concat_n (int): 
            selected ROI number in roi_idxs array
        - roi_n (int): 
            selected ROI number within session
        - sess_n (int): 
            selected session number, base on nrois_per_sess
    """

    if np.isnan(roi_idxs).any() or np.isnan(roi_mses).any():
        raise NotImplementedError("Function not implemented if NaNs in data.")

    math_util.check_n_rand(permpar.n_perms, permpar.p_val)
    p_low, p_high = math_util.get_percentiles(
        CI=(1 - permpar.p_val), tails=permpar.tails
        )[0]

    sig_bool = (roi_percs < p_low) + (roi_percs > p_high)
    smooth_thr = np.percentile(roi_mses, q=8) # set MSE threshold
    smooth_bool = roi_mses < smooth_thr
    if target_idx_sig == "sig":
        keep_roi = np.where(sig_bool * smooth_bool)[0]
    elif target_idx_sig == "not_sig":
        keep_roi = np.where(~sig_bool * smooth_bool)[0]
    else:
        gen_util.accepted_values_error(
            "target_idx_sig", target_idx_sig, ["sig", "not_sig"]
            )

    if len(keep_roi) == 0:
        raise ValueError("No ROIs meet the thresholds.")

    # get closest to value (that meets significance criterion)
    keep_ns = np.argmin(np.absolute(roi_idxs[keep_roi] - target_idx_val))

    concat_n = keep_roi[keep_ns]
    for s in range(len(nrois_per_sess)):
        if concat_n < np.sum(nrois_per_sess[: s + 1]):
            roi_n = int(concat_n - np.sum(nrois_per_sess[: s]))
            sess_n = s
            break
    
    return concat_n, roi_n, sess_n


#############################################
def get_chosen_roi_df(sessions, analyspar, stimpar, basepar, idxpar, permpar, 
                      target_idx_vals=None, target_idx_sigs=None, seed=None, 
                      parallel=False):
    """
    get_chosen_roi_df(sessions, analyspar, stimpar, basepar, idxpar, permpar)

    Selects ROIs that meet the targets amongst sessions with the same session 
    number and line/plane, and returns their information.

    Required args:
        - sessions (list): 
            list of Session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - idxpar (IdxPar): 
            named tuple containing index parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters

    Optional args:
        - target_idx_vals (list): 
            target ROI index values. If None, default values are used.
            default: None
        - target_idx_sigs (list): 
            target ROI index significance. If None, default values are used.
            default: None
        - seed (int): seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): if True, some of the analysis is run in parallel 
            across CPU cores 
            default: False

    Returns:
        - chosen_rois_df (pd.DataFrame):
            dataframe with a row for each ROI retained, and the following 
            columns, in addition to the basic sess_df columns: 
            - target_idxs (str): index values and significance aimed for
            - roi_idxs (float): ROI index
            - roi_idx_percs (float): ROI index percentile
            - roi_ns (int): ROI number with in its session
    """
    
    if target_idx_vals is None and target_idx_sigs is None:
        target_idx_vals = [0.5, 0, -0.5]
        target_idx_sigs = ["sig", "not_sig", "sig"]

    elif target_idx_vals is None or target_idx_sigs is None:
        raise ValueError(
            ("If providing target_idx_vals or target_idx_sigs, must "
            "provide both.")
            )
    
    if len(target_idx_vals) != len(target_idx_sigs):
        raise ValueError(
            "target_idx_vals and target_idx_sigs must have the same length."
            )

    idx_df = misc_analys.get_check_sess_df(sessions, None, analyspar)
    sess_df_cols = idx_df.columns
    chosen_rois_df = pd.DataFrame(columns=sess_df_cols)

    # retrieve ROI index information
    args_dict = {
        "analyspar": analyspar, 
        "stimpar"  : stimpar, 
        "basepar"  : basepar, 
        "idxpar"   : idxpar, 
        "permpar"  : permpar, 
        "seed"     : seed
    }

    logger.info(
        f"Calculating ROI USIs for each session...", extra={"spacing": TAB}
        )
    returns = gen_util.parallel_wrap(
        get_idx_info, sessions, args_dict=args_dict, 
        parallel=parallel, zip_output=True
        )

    misc_analys.get_check_sess_df(sessions, idx_df)
    idx_columns = ["all_roi_idxs", "all_roi_percs", "all_roi_mses"]
    for c, column in enumerate(idx_columns):
        idx_df[column] = returns[c]

    # for each group, select an ROI for each target
    loop_args = list(zip(target_idx_vals, target_idx_sigs))
    group_columns = ["lines", "planes", "sess_ns"]
    for idx_df_vals, idx_df_grp in idx_df.groupby(group_columns):
        line, plane, sess_n = idx_df_vals
        lp_name = plot_helper_fcts.get_line_plane_name(line, plane)
        logger.info(
            f"Selecting ROIs for {lp_name}, session {sess_n}...", 
            extra={"spacing": TAB}
            )

        roi_idxs = np.concatenate(idx_df_grp["all_roi_idxs"].tolist())
        roi_percs = np.concatenate(idx_df_grp["all_roi_percs"].tolist())
        roi_mses = np.concatenate(idx_df_grp["all_roi_mses"].tolist())
        nrois_per_sess = [len(idxs) for idxs in idx_df_grp["all_roi_idxs"]]

        # select ROIs
        args_list = [roi_idxs, roi_percs, roi_mses, nrois_per_sess, permpar]
        _, roi_ns, sess_ns = gen_util.parallel_wrap(
            choose_roi, loop_args, args_list=args_list, 
            parallel=parallel, zip_output=True, mult_loop=True,
            )

        # Add selected ROIs to dataframe
        for (roi_n, sess_n, targ_idx_val, targ_idx_sig) in zip(
            roi_ns, sess_ns, target_idx_vals, target_idx_sigs
            ):
            row_n = len(chosen_rois_df)
            source_row_n = idx_df_grp.index[sess_n]

            # add in session columns
            for column in sess_df_cols: 
                chosen_rois_df.loc[row_n, column] = \
                   idx_df_grp.loc[source_row_n, column]
            
            # add in chosen ROI info
            target_idx = f"{targ_idx_val}_{targ_idx_sig}"
            chosen_rois_df.loc[row_n, "target_idxs"] = target_idx
            chosen_rois_df.loc[row_n, "roi_ns"] = roi_n
            chosen_rois_df.loc[row_n, "roi_idxs"] = \
                idx_df_grp.loc[source_row_n, "all_roi_idxs"][roi_n]
            chosen_rois_df.loc[row_n, "roi_idx_percs"] = \
                idx_df_grp.loc[source_row_n, "all_roi_percs"][roi_n]

    # preserve original dtypes
    for column in sess_df_cols: 
        chosen_rois_df[column] = chosen_rois_df[column].astype(
            idx_df[column].dtype)
    chosen_rois_df["roi_ns"] = chosen_rois_df["roi_ns"].astype(int)

    return chosen_rois_df


#############################################
def add_chosen_roi_traces(sessions, chosen_rois_df, analyspar, stimpar, basepar, 
                          split="by_exp", parallel=False):
    """
    add_chosen_roi_traces(sessions, chosen_rois_df, analyspar, stimpar, basepar)

    Adds ROI traces to chosen_rois_df.

    Required args:
        - sessions (list): 
            list of Session objects
        - chosen_rois_df (pd DataFrame):
            see get_chosen_roi_df() output.
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters


    Optional args:
        - split (str): 
            how to split data:
            "by_exp" (all exp, all unexp), 
            "unexp_lock" (unexp, preceeding exp), 
            "exp_lock" (exp, preceeding unexp),
            "stim_onset" (grayscr, stim on), 
            "stim_offset" (stim off, grayscr)
            default: "by_exp"
        - parallel (bool): if True, some of the analysis is run in parallel 
            across CPU cores 
            default: False

    Returns:
        - chosen_rois_df (pd.DataFrame):
            output of get_chosen_roi_df(), with added columns: 
            - roi_trace_stats (list): 
                ROI trace stats (split x frames x stat (me, err))
            - time_values (list):
                values for each frame, in seconds
                (only 0 to stimpar.post, unless split is "by_exp")
    """

    chosen_rois_df = copy.deepcopy(chosen_rois_df)

    # find the sessions for the dataframe
    sessids = [session.sessid for session in sessions]
    sessids_df = chosen_rois_df["sessids"].unique()
    
    sessions_to_use = []
    for sessid in sessids_df:
        if sessid not in sessids:
            raise ValueError("sessions must include all sessions in "
                f"'chosen_rois_dfs', but session {sessid} is missing.")
        sessions_to_use.append(sessions[sessids.index(sessid)])

    # retrieve ROI index information
    args_dict = {
        "analyspar": analyspar, 
        "stimpar"  : stimpar, 
        "basepar"  : basepar, 
        "split"    : split, 
    }

    # sess x split x ROIs x frames
    roi_trace_stats, all_timevalues = gen_util.parallel_wrap(
        basic_analys.get_sess_roi_trace_stats, sessions_to_use, 
        args_dict=args_dict, parallel=parallel, zip_output=True
        )

    # add traces to correct rows
    new_columns = ["roi_trace_stats", "time_values"]
    for column in new_columns:
        chosen_rois_df[column] = np.nan
        chosen_rois_df[column] = chosen_rois_df[column].astype(object)

    for s, sessid in enumerate(sessids_df):
        sessid_loc = (chosen_rois_df["sessids"] == sessid)
        for r in chosen_rois_df.loc[sessid_loc].index:
            roi_n = int(chosen_rois_df.loc[r, "roi_ns"])
            chosen_rois_df.at[r, "roi_trace_stats"] = \
                roi_trace_stats[s][:, roi_n].tolist()
            chosen_rois_df.at[r, "time_values"] = all_timevalues[s].tolist()

    return chosen_rois_df
    

#############################################
def bin_idxs(roi_idxs, roi_percs, rand_idxs, permpar, n_bins=40):
    """
    bin_idxs(roi_idxs, roi_percs, rand_idxs, permpar)

    Bins indices.

    Required args:
        - roi_idxs (1D array): 
            ROI indices for the session
        - roi_percs (1D array): 
            ROI index percentiles for the session, based on each ROI's 
            random permutations
        - all_rand (1 or 2D array): 
            ROI indices calculated through randomized permutation
            dims: (item x) n_perms
        - permpar (PermPar): 
            named tuple containing permutation parameters

    Optional args:
        - n_bins (int):
            number of bins
            default: 40

    Returns:
        - binned_idxs (dict):
            dictionary with the following keys:
            - ["bin_edges"] (list): first and last bin edge
            - ["CI_perc"] (list): confidence interval percentile limits (lo, hi)
            - ["CI_edges"] (list): confidence interval limit values (lo, hi)
            - ["n_pos"] (list): number of positive ROIs
            - ["n_signif_lo"] (num): number of significant ROIs (low)
            - ["n_signif_hi"] (num): number of significant ROIs (high)
            - ["roi_idx_binned"] (1D array): bin counts for the ROI indices
            - ["rand_idx_binned"] (1D array): bin counts for the random ROI 
                indices
            - ["perc_idx_binned"] (1D array): bin counts for the ROI index 
                percentiles
    """
    
    if np.isnan(roi_idxs).any() or np.isnan(rand_idxs).any():
        raise NotImplementedError("Function not implemented if NaNs in data.")

    # gather index histogram information
    math_util.check_n_rand(rand_idxs.shape[-1], permpar.p_val)
    CI_perc = math_util.get_percentiles(1 - permpar.p_val, permpar.tails)[0]
    # check bin size
    CI_wid = np.max([CI_perc[0], 100 - CI_perc[1]])
    bin_wid = 100 / n_bins
    if (CI_wid // bin_wid) != (CI_wid / bin_wid):
        raise ValueError(f"{n_bins} bins are not compatible with a "
            f"confidence interval bounds {CI_perc[0]} to {CI_perc[1]} "
            "as the outer areas cannot be divided neatly into bins.")

    n_signif_lo = np.sum(roi_percs < CI_perc[0])
    n_signif_hi = np.sum(roi_percs > CI_perc[1])
    n_pos = np.sum(roi_idxs > 0)

    lo_bound = np.min([rand_idxs.min(), roi_idxs.min()])
    hi_bound = np.max([rand_idxs.max(), roi_idxs.max()])
    bins = np.linspace(lo_bound, hi_bound, n_bins + 1)
    rand_idx_binned = np.histogram(rand_idxs, bins=bins)[0]
    rand_idx_binned = rand_idx_binned / np.sum(rand_idx_binned)

    roi_idx_binned = np.histogram(roi_idxs, bins=bins)[0]
    roi_idx_binned = roi_idx_binned / np.sum(roi_idx_binned)

    # gather percentile info
    perc_bins = np.linspace(0, 100, n_bins + 1)
    perc_idx_binned = np.histogram(roi_percs, bins=perc_bins)[0]
    perc_idx_binned = perc_idx_binned / np.sum(perc_idx_binned)
    
    # gather CI lims and number of significant ROI indices
    bin_edges = [lo_bound, hi_bound]
    CI_edges = [np.percentile(rand_idxs, q=p) for p in CI_perc]

    binned_idxs = {
        "CI_perc"        : CI_perc,
        "CI_edges"       : CI_edges,
        "bin_edges"      : bin_edges,
        "n_pos"          : n_pos,
        "n_signif_lo"    : n_signif_lo,
        "n_signif_hi"    : n_signif_hi,
        "perc_idx_binned": perc_idx_binned,
        "roi_idx_binned" : roi_idx_binned,
        "rand_idx_binned": rand_idx_binned,
    }

    return binned_idxs


#############################################
def get_ex_idx_df(sess, analyspar, stimpar, basepar, idxpar, permpar, 
                  seed=None, target_roi_perc=99.8):
    """
    get_ex_idx_df(sess, analyspar, stimpar, basepar, idxpar, permpar)

    Returns information for a selected significant ROI feature index.

    Required args:
        - sess (Session): 
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - idxpar (IdxPar): 
            named tuple containing index parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters
    
    Optional args:
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None
        - target_roi_perc (float): 
            target ROI feature index percentile
            default: 99.8
    
    Returns:
        - ex_idx_df (pd.DataFrame):
            dataframe with a row for the example ROI, and the following 
            columns, in addition to the basic sess_df columns:
            - roi_ns (int): ROI number in session
            - roi_idxs (float): ROI feature index
            - roi_idx_percs (float): ROI feature index percentile
            - rand_idx_binned (list): bin counts for the random ROI indices
            - bin_edges (list): first and last bin edge
            - CI_perc (list): confidence interval percentile limits (lo, hi)
            - CI_edges (list): confidence interval limit values (lo, hi)
    """

    roi_idxs, roi_percs, rand_idxs = sess_stim_idxs(
        sess, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        n_perms=permpar.n_perms, 
        split=idxpar.feature, 
        op=idxpar.op, 
        baseline=basepar.baseline, 
        seed=seed
        )

    ex_idx_df = misc_analys.get_check_sess_df(sess, analyspar=analyspar)

    if np.isnan(roi_idxs).any() or np.isnan(rand_idxs).any():
        raise NotImplementedError("Function not implemented if NaNs in data.")

    # select a significant ROI
    if permpar.tails == "lo":
        raise ValueError("Expected permpar.tails to be 2 or 'hi', not 'lo'.")
    math_util.check_n_rand(permpar.n_perms, permpar.p_val)
    CI_perc = math_util.get_percentiles(1 - permpar.p_val, permpar.tails)[0]
    possible = np.where((roi_percs > CI_perc[1]) * (roi_percs < 100))[0]
    preferred = np.where(
        np.asarray(
            [np.around(v, 1) for v in roi_percs[possible]]
            ) == target_roi_perc
        )[0]

    # if preference is not found, use first value
    chosen_idx = preferred[0] if len(preferred) else 0
    roi_n = int(possible[chosen_idx])
    
    # gather index histogram information
    binned_idxs = bin_idxs(
        roi_idxs[roi_n], roi_percs[roi_n], rand_idxs[roi_n], permpar, n_bins=40
        )

    ex_idx_df.loc[0, "roi_ns"]         = roi_n
    ex_idx_df.loc[0, "roi_idxs"]       = roi_idxs[roi_n]
    ex_idx_df.loc[0, "roi_idx_percs"]  = roi_percs[roi_n]

    for column in ["rand_idx_binned", "bin_edges", "CI_edges", "CI_perc"]:
        data = binned_idxs[column]
        if isinstance(data, np.ndarray):
            data = data.tolist()
        ex_idx_df[column] = np.nan
        ex_idx_df[column] = ex_idx_df[column].astype(object)
        ex_idx_df.at[0, column] = data
    
    ex_idx_df["roi_ns"] = ex_idx_df["roi_ns"].astype(int)
        
    return ex_idx_df


#############################################
def get_idx_only_df(sessions, analyspar, stimpar, basepar, idxpar, 
                    parallel=False):
    """
    get_idx_only_df(sessions, analyspar, stimpar, basepar)

    Returns indices for each session, without significance information

    Required args:
        - sessions (list): 
            Session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - idxpar (IdxPar): 
            named tuple containing index parameters
    
    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    
    Returns:
        - idx_only_df (pd.DataFrame):
            dataframe with one row per session, and the following columns, in 
            addition to the basic sess_df columns:
            - roi_idxs (list): index for each ROI
    """

    idx_only_df = misc_analys.get_check_sess_df(sessions, analyspar=analyspar)

    args_dict = {
        "analyspar"  : analyspar, 
        "stimpar"    : stimpar, 
        "split"      : idxpar.feature, 
        "op"         : idxpar.op, 
        "baseline"   : basepar.baseline, 
        "run_random" : False
    }    

    logger.info(
        ("Calculating ROI USIs for each session..."), extra={"spacing": TAB}
        )
    roi_idxs = gen_util.parallel_wrap(
        sess_stim_idxs, sessions, args_dict=args_dict, parallel=parallel
        )
    
    misc_analys.get_check_sess_df(sessions, idx_only_df)
    idx_only_df["roi_idxs"] = [
        sess_roi_idxs.tolist() for sess_roi_idxs in roi_idxs
        ]

    return idx_only_df
    
    
#############################################
def get_idx_df(sessions, analyspar, stimpar, basepar, idxpar, permpar, 
               n_bins=40, common_oris=False, by_mouse=False, seed=None, 
               parallel=False):
    """
    get_idx_df(sessions, analyspar, stimpar, basepar, idxpar, permpar)

    Returns index information for each line/plane.

    Required args:
        - sessions (list): 
            Session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - idxpar (IdxPar): 
            named tuple containing index parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters
    
    Optional args:
        - n_bins (int):
            number of bins
            default: 40
        - common_oris (bool): 
            if True, data is for common orientations
            default: False
        - by_mouse (bool): 
            if True, data is kept separated by mouse
            default: False
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    
    Returns:
        - idx_df (pd.DataFrame):
            dataframe with one row per (mouse/)session/line/plane, and the 
            following columns, in addition to the basic sess_df columns:
            - bin_edges (list): first and last bin edge
            - CI_perc (list): confidence interval percentile limits (lo, hi)
            - CI_edges (list): confidence interval limit values (lo, hi)
            - n_pos (int): number of positive ROIs
            - n_signif_lo (int): number of significant ROIs (low) 
            - n_signif_hi (int): number of significant ROIs (high)
            - roi_idx_binned (list): bin counts for the ROI indices
            - rand_idx_binned (list): bin counts for the random ROI indices
            - perc_idx_binned (list): bin counts for the ROI index percentiles
    """

    full_idx_df = misc_analys.get_check_sess_df(sessions, analyspar=analyspar)
    initial_columns = full_idx_df.columns

    args_dict = {
        "analyspar"  : analyspar, 
        "stimpar"    : stimpar, 
        "n_perms"    : permpar.n_perms, 
        "split"      : idxpar.feature, 
        "op"         : idxpar.op, 
        "baseline"   : basepar.baseline, 
        "common_oris": common_oris,
        "seed"       : seed,
    }    

    logger.info(
        ("Calculating ROI USIs and their null distributions for each "
        "session..."), extra={"spacing": TAB}
        )
    roi_idxs, roi_percs, rand_idxs = gen_util.parallel_wrap(
        sess_stim_idxs, sessions, args_dict=args_dict, parallel=parallel, 
        zip_output=True
        )
    
    misc_analys.get_check_sess_df(sessions, full_idx_df)
    full_idx_df["roi_idxs"] = roi_idxs
    full_idx_df["roi_percs"] = roi_percs
    full_idx_df["rand_idxs"] = rand_idxs

    idx_df = pd.DataFrame(columns=initial_columns)
    # join within line/plane
    group_columns = ["lines", "planes", "sess_ns"]
    if by_mouse:
        group_columns.append("mouse_ns")
    aggreg_cols = [col for col in initial_columns if col not in group_columns]
    for grp_vals, grp_df in full_idx_df.groupby(group_columns):
        row_idx = len(idx_df)
        for g, group_column in enumerate(group_columns):
            idx_df.loc[row_idx, group_column] = grp_vals[g]

        # add aggregated values for initial columns
        idx_df = misc_analys.aggreg_columns(
            grp_df, idx_df, aggreg_cols, row_idx=row_idx, in_place=True, 
            by_mouse=by_mouse
            )

        roi_idxs = np.concatenate(grp_df["roi_idxs"].tolist())
        roi_percs = np.concatenate(grp_df["roi_percs"].tolist())
        rand_idxs = np.concatenate(grp_df["rand_idxs"].tolist())

        binned_idxs = bin_idxs(
            roi_idxs, roi_percs, rand_idxs, permpar, n_bins=n_bins
            )
        
        for key, value in binned_idxs.items():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if (key not in idx_df.columns) and (isinstance(value, list)):
                idx_df[key] = np.nan
                idx_df[key] = idx_df[key].astype(object)
            idx_df.at[row_idx, key] = value
    
    for key in ["sess_ns", "n_pos", "n_signif_lo", "n_signif_hi"]:
        idx_df[key] = idx_df[key].astype(int)
    
    if by_mouse:
        idx_df["mouse_ns"] = idx_df["mouse_ns"].astype(int)

    return idx_df


#############################################
def get_perc_sig_df(idx_df, analyspar, permpar, seed=None):
    """
    get_perc_sig_df(idx_df, permpar)

    Returns dictionary with confidence interval over percentages of 
    significant or positive ROI feature indices.

    Required args:
        - idx_df (pd.DataFrame):
            dataframe with one row per (mouse/)session/line/plane, and the 
            following columns, in addition to the basic sess_df columns:
            - n_pos (int): number of positive ROIs
            - n_signif_lo (int): number of significant ROIs (low) 
            - n_signif_hi (int): number of significant ROIs (high)
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters

    Optional args:
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None

    Returns:
        - perc_sig_df (pd.DataFrame):
            dataframe with one row per (mouse/)session/line/plane, and the 
            following columns, in addition to the basic sess_df columns:
            - n_pos (int): number of positive ROIs
            - n_signif_lo (int): number of significant ROIs (low) 
            - n_signif_hi (int): number of significant ROIs (high)

            - perc_pos_idxs (num): percent positive ROIs (0-100)
            - perc_pos_idxs_stds (num): bootstrapped standard deviation 
                over percent significant ROIs
            - perc_pos_idxs_CIs (list): adjusted confidence interval for 
                percent pos. ROIs 
            - perc_pos_idxs_null_CIs (list): adjusted null CI for percent pos. 
                ROIs
            - perc_pos_idxs_raw_p_vals (num): uncorrected p-value for percent 
                pos. ROIs
            - perc_pos_idxs_p_vals (num): p-value for percent pos. ROIs, 
                corrected for multiple comparisons and tails

            for sig in ["lo", "hi"]: for low vs high ROI indices
            - perc_sig_{sig}_idxs (num): percent significant ROIs (0-100)
            - perc_sig_{sig}_idxs_stds (num): bootstrapped standard deviation 
                over percent significant ROIs
            - perc_sig_{sig}_idxs_CIs (list): adjusted CI for percent sig. ROIs 
            - perc_sig_{sig}_idxs_null_CIs (list): adjusted null CI for percent 
                sig. ROIs
            - perc_sig_{sig}_idxs_raw_p_vals (num): uncorrected p-value for 
                percent sig. ROIs
            - perc_sig_{sig}_idxs_p_vals (num): p-value for percent sig. 
                ROIs, corrected for multiple comparisons and tails
    """    

    seed = gen_util.seed_all(seed, "cpu", log_seed=False)

    if analyspar.stats != "mean" or analyspar.error != "std":
        raise NotImplementedError(
            "analyspar.stats must be set to 'mean', and "
            "analyspar.error must be set to 'std'."
            )

    perc_sig_df = idx_df.copy(deep=True)

    # trim
    for column in perc_sig_df.columns:
        if "bin" in column or "CI" in column:
            perc_sig_df = perc_sig_df.drop(columns=column)

    # add new columns
    for grp in ["pos", "sig_lo", "sig_hi"]:
        for data in ["CIs", "null_CIs"]:
            column_name = f"perc_{grp}_idxs_{data}"
            perc_sig_df[column_name] = np.nan
            perc_sig_df[column_name] = perc_sig_df[column_name].astype(object)

    nrois = np.asarray([np.sum(nrois) for nrois in idx_df["nrois"]])
    n_pos = np.asarray(idx_df["n_pos"])

    # get positive ROI index info
    null_perc = 50 # null percentage is 50%
    for g, (grp_n_pos, grp_nroi) in enumerate(zip(n_pos, nrois)):
        perc_pos = grp_n_pos / grp_nroi * 100
        CI, null_CI, p_val = math_util.binom_CI(
            permpar.p_val, perc_pos, grp_nroi, null_perc, permpar.tails, 
            permpar.multcomp
            )
        perc_std = 100 * math_util.bootstrapped_std(
            perc_pos / 100, n=grp_nroi, n_samples=misc_analys.N_BOOTSTRP, 
            proportion=True
            )

        perc_sig_df.loc[g, "perc_pos_idxs"] = perc_pos
        perc_sig_df.loc[g, "perc_pos_idxs_stds"] = perc_std
        perc_sig_df.at[g, "perc_pos_idxs_CIs"] = CI.tolist()
        perc_sig_df.at[g, "perc_pos_idxs_null_CIs"] = null_CI.tolist()
        perc_sig_df.loc[g, "perc_pos_idxs_p_vals"] = p_val
    
    # get significant ROI index info
    null_perc = 100 * permpar.p_val / 2 # null percentage is 100 * p_val / 2
    for sig in ["lo", "hi"]:
        n_signifs = np.asarray(idx_df[f"n_signif_{sig}"])
    
        for g, (grp_n_sig, grp_nroi) in enumerate(zip(n_signifs, nrois)):
            perc_sig = grp_n_sig / grp_nroi * 100
            CI, null_CI, p_val = math_util.binom_CI(
                permpar.p_val, perc_sig, grp_nroi, null_perc, permpar.tails, 
                permpar.multcomp
                )
            perc_std = 100 * math_util.bootstrapped_std(
                perc_sig / 100, n=grp_nroi, n_samples=misc_analys.N_BOOTSTRP, 
                proportion=True
                )

            perc_sig_df.loc[g, f"perc_sig_{sig}_idxs"] = perc_sig
            perc_sig_df.loc[g, f"perc_sig_{sig}_idxs_stds"] = perc_std
            perc_sig_df.at[g, f"perc_sig_{sig}_idxs_CIs"] = CI.tolist()
            perc_sig_df.at[g, f"perc_sig_{sig}_idxs_null_CIs"] = \
                null_CI.tolist()
            perc_sig_df.loc[g, f"perc_sig_{sig}_idxs_p_vals"] = p_val
    
    # add corrected p-values
    perc_sig_df = misc_analys.add_corr_p_vals(perc_sig_df, permpar)

    return perc_sig_df


#############################################
def get_idx_stats_df(sessions, analyspar, stimpar, basepar, idxpar, 
                     permpar=None, absolute=True, by_mouse=False, seed=None, 
                     parallel=False):
    """
    get_idx_stats_df(sessions, analyspar, stimpar, basepar)

    Returns statistics over ROI indices for each line/plane/session.

    Required args:
        - sessions (list): 
            Session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - idxpar (IdxPar): 
            named tuple containing index parameters
    
    Optional args:
        - permpar (PermPar): 
            named tuple containing permutation parameters. If None, no 
            significance testing is done.
            default: None
        - absolute (bool):
            if True, statistics are taken across 
        - by_mouse (bool): 
            if True, data is kept separated by mouse
            default: False
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
   
    Returns:
        - idx_stats_df (pd.DataFrame):
            dataframe with one row per (mouse/)session/line/plane, and the 
            following columns, in addition to the basic sess_df columns:
            - roi_idxs or abs_roi_idxs (list): 
                USI statistics or absolute USI statistics
            for session comparisons, e.g. 1v2 (if permpar is not None):
            - raw_p_vals_{}v{} (float): uncorrected p-value for differences
                between sessions 
            - p_vals_{}v{} (float): p-value for differences between sessions, 
                corrected for multiple comparisons and tails
    """

    nanpol = None if analyspar.remnans else "omit"

    if permpar is not None:
        seed = gen_util.seed_all(seed, "cpu", log_seed=False)

    if analyspar.tracked:
        misc_analys.check_sessions_complete(sessions, raise_err=True)

    initial_columns = misc_analys.get_sess_df_columns(sessions[0], analyspar)

    idx_only_df = get_idx_only_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        parallel=parallel
        )
    
    # remove roi_idxs
    data_col = "roi_idx_stats"
    if absolute: 
        data_col = f"abs_{data_col}"
    columns = initial_columns + [data_col]

    # initialize new dataframe
    idx_stats_df = pd.DataFrame(columns=columns)

    # join within line/plane
    group_columns = ["lines", "planes"]
    if by_mouse:
        group_columns.append("mouse_ns")
    aggreg_cols = [
        col for col in initial_columns if col not in group_columns + ["sess_ns"]
        ]
    for grp_vals, grp_df in idx_only_df.groupby(group_columns):
        sess_ns = sorted(grp_df["sess_ns"].unique())
        row_indices = []
        sess_roi_idxs = []
        for sess_n in sess_ns:
            sess_grp_df = grp_df.loc[grp_df["sess_ns"] == sess_n]
            row_idx = len(idx_stats_df)
            row_indices.append(row_idx)

            idx_stats_df.loc[row_idx, "sess_ns"] = sess_n
            for g, group_column in enumerate(group_columns):
                idx_stats_df.loc[row_idx, group_column] = grp_vals[g]

            # add aggregated values for initial columns
            idx_stats_df = misc_analys.aggreg_columns(
                sess_grp_df, idx_stats_df, aggreg_cols, row_idx=row_idx, 
                in_place=True, by_mouse=by_mouse
                )

            # collect indices
            roi_idxs = np.concatenate(sess_grp_df["roi_idxs"].tolist())
            if absolute:
                roi_idxs = np.absolute(roi_idxs)

            # take statistics
            roi_idx_stats = math_util.get_stats(
                roi_idxs, stats=analyspar.stats, error=analyspar.error, 
                nanpol=nanpol
                )
            idx_stats_df.at[row_idx, data_col] = roi_idx_stats.tolist()

            sess_roi_idxs.append(roi_idxs)

        # calculate p-values between sessions (0-1, 0-2, 1-2...)
        if permpar is not None:
            p_vals = math_util.comp_vals_acr_groups(
                sess_roi_idxs, n_perms=permpar.n_perms, stats=analyspar.stats, 
                paired=analyspar.tracked, nanpol=nanpol
                )
            p = 0
            for i, sess_n in enumerate(sess_ns):
                for j, sess_n2 in enumerate(sess_ns[i + 1:]):
                    key = f"p_vals_{int(sess_n)}v{int(sess_n2)}"
                    idx_stats_df.loc[row_indices[i], key] = p_vals[p]
                    idx_stats_df.loc[row_indices[j + 1], key] = p_vals[p]
                    p += 1

    idx_stats_df["sess_ns"] = idx_stats_df["sess_ns"].astype(int)
    if by_mouse:
        idx_stats_df["mouse_ns"] = idx_stats_df["mouse_ns"].astype(int)

    # correct p-values
    if permpar is not None:
        idx_stats_df = misc_analys.add_corr_p_vals(idx_stats_df, permpar)

    return idx_stats_df
    
