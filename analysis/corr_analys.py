"""
corr_analys.py

This script contains functions for correlation analysis.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

import numpy as np
import pandas as pd
import scipy.stats as scist

from util import logger_util, gen_util, math_util
from sess_util import sess_ntuple_util
from analysis import misc_analys, usi_analys

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def get_corr_pairs(sess_df, consec_only=True):
    """
    get_corr_pairs(sess_df)

    Returns correlation pairs.

    Required args:
        - sess_df (pd.DataFrame):
            dataframe containing session information, including the following 
            keys: "sess_ns", "lines", "planes"

    Optional args:
        - consec_only (bool):
            if True, only consecutive session numbers are correlated
            default: True

    Returns:
        - corr_ns (list):
            session number pairs, e.g. [[s1, s2], [s2, s3], ...]
    """

    # identify correlation pairs
    corr_ns = []
    for _, lp_df in sess_df.groupby(["lines", "planes"]):
        sessions = np.sort(lp_df["sess_ns"].unique())
        if len(sessions) == 1:
            continue
        for i, sess1 in enumerate(sessions):
            for sess2 in sessions[i + 1:]:
                if consec_only and (sess2 - sess1 != 1):
                    continue
                corr_pair = [sess1, sess2]
                if corr_pair not in corr_ns:
                    corr_ns.append(corr_pair)
    
    if len(corr_ns) == 0:
        raise RuntimeError("No session pairs found.")

    return corr_ns


#############################################
def set_multcomp(permpar, sessions, analyspar, consec_only=True):
    """
    set_multcomp(permpar, sessions, analyspar)

    Returns permpar updated with the number of comparisons computed from the 
    sessions.

    Required args:
        - permpar (PermPar or dict): 
            named tuple containing permutation parameters
        - sessions (list):
            Session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters

    Optional args:
        - consec_only (bool):
            if True, only consecutive session numbers are correlated
            default: True

    Returns:
        - permpar (PermPar):
            updated permutation parameter named tuple
    """
    
    sess_df = misc_analys.get_check_sess_df(sessions, analyspar=analyspar)

    n_comps = 0
    for _, lp_df in sess_df.groupby(["lines", "planes"]):
        corr_ns = get_corr_pairs(lp_df, consec_only=consec_only)
        n_comps += len(corr_ns)
    
    permpar = sess_ntuple_util.get_modif_ntuple(permpar, "multcomp", n_comps)

    return permpar


#############################################
def get_norm_corrs(corr_data, med=0):
    """
    get_norm_corrs(data, med=0)
    
    Returns normalized correlation values.

    Required args:
        - corr_data (1D array): 
            values to normalize
    
    Optional args:
        - med (float): 
            null distribution median for normalization
            default: 0

    Returns:
        - norm_corr_data (1D array): normalized correlations
    """

    corr_data = np.asarray(corr_data)

    # normalize all data
    if np.absolute(med) > 1:
        raise RuntimeError(
            "Absolute correlations should not be greater than 1."
            )
        
    lens_to_bound = np.asarray([np.absolute(med + 1), np.absolute(1 - med)])

    corr_sides = (corr_data > med).astype(int)
    norm_corr_data = (corr_data - med) / lens_to_bound[corr_sides]
    
    return norm_corr_data


#############################################
def corr_bootstrapped_std(data, n_samples=1000, randst=None, return_rand=False, 
                          nanpol=None, med=0):
    """
    corr_bootstrapped_std(data)
    
    Returns bootstrapped standard deviation for normalized Pearson correlations.

    Required args:
        - data (2D array): 
            values to correlate for each of 2 groups (2, n)
    
    Optional args:
        - n (int): 
            number of datapoints in dataset. Required if proportion is True.
            default: None
        - n_samples (int): 
            number of samplings to take for bootstrapping
            default: 1000
        - randst (int): 
            seed or random state to use when generating random values.
            default: None
        - return_rand (bool): if True, random correlations are returned
            default: False
        - nanpol (str): 
            policy for NaNs, "omit" or None
            default: None
        - med (float): 
            null distribution median for normalization
            default: 0

    Returns:
        - bootstrapped_std (float): 
            bootstrapped standard deviation of normalized correlations
        if return_rand:
        - rand_corrs (1D array): 
            randomly generated correlations
    """

    if randst is None:
        randst = np.random
    elif isinstance(randst, int):
        randst = np.random.RandomState(randst) 

    n_samples = int(n_samples)

    data = np.asarray(data)

    if len(data.shape) != 2 or data.shape[0] != 2:
        raise ValueError(
            "data must have 2 dimensions, with the first having length 2."
            )

    n = data.shape[1]

    # random choices
    choices = np.arange(n)

    # random corrs
    rand_corrs = math_util.np_pearson_r(
        *list(data[:, randst.choice(choices, (n, n_samples), replace=True)]), 
        nanpol=nanpol, axis=0
        )
    
    norm_rand_corrs = get_norm_corrs(rand_corrs, med=med)

    bootstrapped_std = math_util.error_stat(
        norm_rand_corrs, stats="mean", error="std", nanpol=nanpol
        )
    
    if return_rand:
        return bootstrapped_std, rand_corrs
    else:
        return bootstrapped_std


#############################################
def get_corr_data(sess_pair, data_df, analyspar, permpar, 
                  permute_tracking=False, norm=True, return_rand=False, 
                  seed=None, raise_no_pair=True):
    """
    get_corr_data(sess_pair, data_df, analyspar, permpar)

    Returns correlation data for a session pair.

    Required args:
        - sess_pair (list):
            sessions to correlate, e.g. [1, 2]
        - data_df (pd.DataFrame):
            dataframe with one row per line/plane/session, and the following 
            columns, in addition to the basic sess_df columns:
            - roi_idxs (list): index for each ROI
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters.

    Optional args:
        - permute_tracking (bool):
            if True, in permutation test, ROI tracked pairs are shuffled, 
            instead of session order being shuffled within tracked ROI pairs
            default: False
        - norm (bool):
            if True, normalized correlation data is returned
            default: True
        - return_rand (bool):
            if True, random normalized correlation values are returned
            default: False
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None
        - raise_no_pair (bool):
            if True, if sess_pair session numbers are not found, an error is 
            raised. Otherwise, None is returned.
            default: True

    Returns:
        - roi_corr (float):
             (normalized) correlation between sessions
        - roi_corr_std (float):
            bootstrapped standard deviation for the (normalized) correlation 
            between sessions
        - null_CI (list):
            adjusted, null CI for the (normalized) correlation between sessions
        - p_val (float):
            uncorrected p-value for the correlation between sessions
        if return_rand:
        - rand_vals (1D array):
            (normalized) random correlation between sessions
    """

    nanpol = None if analyspar.remnans else "omit"

    seed = gen_util.seed_all(seed, "cpu", log_seed=False)

    roi_idxs = []
    for sess_n in sess_pair:
        row = data_df.loc[data_df["sess_ns"] == sess_n]
        if len(row) < 1:
            continue
        elif len(row) > 1:
            raise RuntimeError("Expected at most one row.")

        data = np.asarray(row.loc[row.index[0], "roi_idxs"])
        roi_idxs.append(data)

    if len(roi_idxs) != 2:
        if raise_no_pair:
            raise RuntimeError("Session pairs not found.")
        else:
            return None

    if roi_idxs[0].shape != roi_idxs[1].shape:
        raise RuntimeError(
            "Sessions should have the same number of ROI indices."
            )

    # get real correlation
    first, sec = roi_idxs[0], roi_idxs[1]
    diffs = sec - first
    roi_corr = math_util.np_pearson_r(first, diffs, nanpol=nanpol)

    # determine type of randomization to use
    if permute_tracking:
        paired = "within"
    else:
        paired = True

    # get random correlation info
    returns = math_util.get_op_p_val(
        [first, sec], n_perms=permpar.n_perms, 
        stats=analyspar.stats, op="diff_corr", return_CIs=True, 
        p_thresh=permpar.p_val, tails=permpar.tails, 
        multcomp=permpar.multcomp, paired=paired, nanpol=nanpol, 
        return_rand=return_rand
        )
    
    if return_rand:
        p_val, null_CI, rand_vals = returns
    else:
        p_val, null_CI = returns

    if norm:
        # normalize all data
        med = null_CI[1]
        roi_corr = float(get_norm_corrs(roi_corr, med=med))
        null_CI = get_norm_corrs(null_CI, med=med)

    else:
        med = 0
    
    # get bootstrapped std over corr
    roi_corr_std = corr_bootstrapped_std(
        [first, diffs], n_samples=misc_analys.N_BOOTSTRP, 
        randst=seed, return_rand=False, nanpol=nanpol, med=med
        )

    returns = [roi_corr, roi_corr_std, null_CI, p_val]
   
    if return_rand:
        if norm:
            rand_vals = get_norm_corrs(rand_vals, med=med)
        returns = returns + [rand_vals]
    
    return returns


#############################################
def get_lp_idx_df(sessions, analyspar, stimpar, basepar, idxpar, 
                  parallel=False):
    """
    get_lp_idx_df(sessions, analyspar, stimpar, basepar, idxpar)

    Returns ROI index dataframe, grouped by line/plane/session.

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
        - lp_idx_df (pd.DataFrame):
            dataframe with one row per line/plane/session, and the following 
            columns, in addition to the basic sess_df columns:
            - roi_idxs (list): index for each ROI
    """

    if analyspar.tracked:
        misc_analys.check_sessions_complete(sessions, raise_err=True)

    initial_columns = misc_analys.get_sess_df_columns(sessions[0], analyspar)

    idx_only_df = usi_analys.get_idx_only_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        parallel=parallel
        )
    
    # aggregate by line/plane/session
    lp_idx_df = pd.DataFrame(columns=initial_columns + ["roi_idxs"])

    # aggregate within line/plane/sessions
    group_columns = ["lines", "planes", "sess_ns"]
    aggreg_cols = [col for col in initial_columns if col not in group_columns]
    for grp_vals, grp_df in idx_only_df.groupby(group_columns):
        row_idx = len(lp_idx_df)
        for g, group_column in enumerate(group_columns):
            lp_idx_df.loc[row_idx, group_column] = grp_vals[g]

        # add aggregated values for initial columns
        lp_idx_df = misc_analys.aggreg_columns(
            grp_df, lp_idx_df, aggreg_cols, row_idx=row_idx, in_place=True
            )

        lp_idx_df.at[row_idx, "roi_idxs"] = \
            np.concatenate(grp_df["roi_idxs"].tolist())
    
    lp_idx_df["sess_ns"] = lp_idx_df["sess_ns"].astype(int)

    return lp_idx_df


#############################################
def get_basic_idx_corr_df(lp_idx_df, consec_only=False):
    """

    DOCSTRING!!!!!!!!!!!!!!!!
    DOCSTRING!!!!!!!!!!!!!!!!
    DOCSTRING!!!!!!!!!!!!!!!!
    DOCSTRING!!!!!!!!!!!!!!!!


    """

    initial_columns = [col for col in lp_idx_df.columns if col != "roi_idxs"]

    # get correlation pairs
    corr_ns = get_corr_pairs(lp_idx_df, consec_only=consec_only)

    # aggregate by line/plane for correlation dataframe
    group_columns = ["lines", "planes"]
    
    CI_cols = [
        f"{corr_pair[0]}v{corr_pair[1]}_null_CIs" for corr_pair in corr_ns
        ]
    idx_corr_df = pd.DataFrame(columns=initial_columns + CI_cols)
    aggreg_cols = [
        col for col in initial_columns if col not in group_columns
        ]

    for grp_vals, grp_df in lp_idx_df.groupby(group_columns):
            row_idx = len(idx_corr_df)

            for g, group_column in enumerate(group_columns):
                idx_corr_df.loc[row_idx, group_column] = grp_vals[g]

            # add aggregated values for initial columns
            idx_corr_df = misc_analys.aggreg_columns(
                grp_df, idx_corr_df, aggreg_cols, row_idx=row_idx, 
                in_place=True, sort_by="sess_ns"
                )
            
            # amend mouse info
            for col in ["mouse_ns", "mouseids"]:
                vals = [tuple(ns) for ns in idx_corr_df.loc[row_idx, col]]
                if len(list(set(vals))) != 1:
                    raise RuntimeError(
                        "Aggregated sessions should share same mouse "
                        "information."
                        )
                idx_corr_df.at[row_idx, col] = list(vals[0])

    return idx_corr_df


#############################################
def get_ex_idx_corr_df(sessions, analyspar, stimpar, basepar, idxpar, permpar,
                       permute_tracking=False, seed=None, parallel=False):
    """

    DOCSTRING!!!!!!!!!!!!!!!!
    DOCSTRING!!!!!!!!!!!!!!!!
    DOCSTRING!!!!!!!!!!!!!!!!
    DOCSTRING!!!!!!!!!!!!!!!!



    """

    initial_columns = misc_analys.get_sess_df_columns(sessions[0], analyspar)
    
    lp_idx_df = get_lp_idx_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar,
        parallel=parallel
        )
    
    idx_corr_df = get_basic_idx_corr_df(lp_idx_df, consec_only=False)
    if len(idx_corr_df) != 1:
        raise ValueError("sessions should be from the same line/plane.")

    # get correlation pairs
    corr_ns = get_corr_pairs(lp_idx_df)

    if len(corr_ns) != 1:
        raise ValueError("Sessions should allow only one pair.")
    sess_pair = corr_ns[0]
    corr_name = f"{sess_pair[0]}v{sess_pair[1]}"

    drop_columns = [
        col for col in idx_corr_df.columns if col not in initial_columns
        ]
    idx_corr_df = idx_corr_df.drop(columns=drop_columns)

    logger.info(
        ("Calculating ROI USI correlations for a single session pair..."), 
        extra={"spacing": TAB}
        )

    returns = get_corr_data(
        sess_pair, 
        lp_idx_df=lp_idx_df, 
        analyspar=analyspar, 
        permpar=permpar, 
        permute_tracking=permute_tracking, 
        return_rand=True, 
        norm=False,
        seed=seed
        )

    roi_corr, _, null_CI, _, rand_vals = returns

    row_idx = idx_corr_df.index[0]

    idx_corr_df.loc[row_idx, f"{corr_name}_corrs"] = roi_corr
    
    # ADD COLUMN FIRST
    
    idx_corr_df.at[row_idx, f"{corr_name}_null_CIs"] = null_CI.tolist()

    # OBTAIN HISTOGRAM

    # OBTAIN PERMUTATION EXAMPLE AND RANDOM CORR


    return idx_corr_df
    

#############################################
def get_idx_corrs_df(sessions, analyspar, stimpar, basepar, idxpar, permpar, 
                     consec_only=True, permute_tracking=False, seed=None, 
                     parallel=False):
    """
    get_idx_corrs_df(sessions, analyspar, stimpar, basepar, idxpar, permpar)

    Returns ROI index correlation data for each line/plane/session.

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
            named tuple containing permutation parameters.
    
    Optional args:
        - consec_only (bool):
            if True, only consecutive session numbers are correlated
            default: True
        - permute_tracking (bool):
            if True, in permutation test, ROI tracked pairs are shuffled, 
            instead of session order being shuffled within tracked ROI pairs
            default: False
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
   
    Returns:
        - idx_corr_df (pd.DataFrame):
            dataframe with one row per line/plane, and the 
            following columns, in addition to the basic sess_df columns:

            for session comparisons, e.g. 1v2
            - {}v{}_norm_corrs (float): normalized intersession ROI index 
                correlations
            - {}v{}_norm_corr_stds (float): bootstrapped normalized 
                intersession ROI index correlation standard deviation
            - {}v{}_null_CIs (list): adjusted null CI for normalized 
                intersession ROI index correlations
            - {}v{}_raw_p_vals (float): p-value for normalized intersession 
                correlations
            - {}v{}_p_vals (float): p-value for normalized intersession 
                correlations, corrected for multiple comparisons and tails
    """
    
    lp_idx_df = get_lp_idx_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar,
        parallel=parallel
        )

    idx_corr_df = get_basic_idx_corr_df(lp_idx_df, consec_only=False)

    # get correlation pairs
    corr_ns = get_corr_pairs(lp_idx_df, consec_only=consec_only)

    logger.info(
        ("Calculating ROI USI correlations across sessions..."), 
        extra={"spacing": TAB}
        )
    group_columns = ["lines", "planes"]
    for grp_vals, grp_df in lp_idx_df.groupby(group_columns):
        line, plane = grp_vals
        row_idx = idx_corr_df.loc[
            (idx_corr_df["lines"] == line) &
            (idx_corr_df["planes"] == plane)
        ].index

        if len(row_idx) != 1:
            raise RuntimeError("Expected exactly one row to match.")
        row_idx = row_idx[0]

        # obtain correlation data
        args_dict = {
            "data_df"         : grp_df,
            "analyspar"       : analyspar,
            "permpar"         : permpar,
            "permute_tracking": permute_tracking,
            "raise_no_pair"   : raise_no_pair,
            "seed"            : seed,
        }
        all_corr_data = gen_util.parallel_wrap(
            get_corr_data, 
            corr_ns, 
            args_dict=args_dict, 
            parallel=parallel, 
            zip_output=False
        )

        # add to dataframe
        for sess_pair, corr_data in zip(corr_ns, all_corr_data):

            if corr_data is None:
                continue
            
            corr_name = f"{sess_pair[0]}v{sess_pair[1]}"
            norm_roi_corr, norm_roi_corr_std, norm_null_CI, p_val = corr_data

            idx_corr_df.loc[row_idx, f"{corr_name}_norm_corrs"] = norm_roi_corr
            idx_corr_df.loc[row_idx, f"{corr_name}_norm_corr_stds"] = \
                norm_roi_corr_std
            idx_corr_df.at[row_idx, f"{corr_name}_null_CIs"] = \
                norm_null_CI.tolist()
            idx_corr_df.loc[row_idx, f"{corr_name}_p_vals"] = p_val

    # corrected p-values
    idx_corr_df = misc_analys.add_corr_p_vals(idx_corr_df, permpar)

    return idx_corr_df

