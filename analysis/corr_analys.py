"""
corr_analys.py

This script contains functions for USI correlation analysis.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import copy
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.ndimage as scind

from util import logger_util, gen_util, math_util, rand_util
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
        sess_ns = np.sort(lp_df["sess_ns"].unique())
        if len(sess_ns) == 1:
            continue
        for i, sess1 in enumerate(sess_ns):
            for sess2 in sess_ns[i + 1:]:
                if consec_only and (sess2 - sess1 != 1):
                    continue
                corr_pair = [sess1, sess2]
                if corr_pair not in corr_ns:
                    corr_ns.append(corr_pair)
    
    if len(corr_ns) == 0:
        raise RuntimeError("No session pairs found.")

    return corr_ns


#############################################
def set_multcomp(permpar, sessions, analyspar, consec_only=True, factor=1):
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
        - factor (int):
            multiplicative factor
            default: 1

    Returns:
        - permpar (PermPar):
            updated permutation parameter named tuple
    """
    
    sess_df = misc_analys.get_check_sess_df(sessions, analyspar=analyspar)

    n_comps = 0
    for _, lp_df in sess_df.groupby(["lines", "planes"]):
        corr_ns = get_corr_pairs(lp_df, consec_only=consec_only)
        n_comps += len(corr_ns)
    
    n_comps = n_comps * factor

    permpar = sess_ntuple_util.get_modif_ntuple(permpar, "multcomp", n_comps)

    return permpar


#############################################
def get_corr_info(permpar, corr_type="corr", permute="sess", norm=True):
    """
    get_corr_info(permpar)

    Returns updated correlation parameters.

    Required args:
        - permpar (PermPar): 
            named tuple containing permutation parameters.

    Optional args:
        - corr_type (str):
            type of correlation to run, i.e. "corr" or "R_sqr"
            default: "corr"
        - permute (str):
            type of permutation to due ("tracking", "sess" or "all")
            default: "sess"
        - norm (bool):
            if True, normalized correlation data is returned, if corr_type if 
            "diff_corr"
            default: True

    Returns:
        - corr_type (str):
            type of correlation to run, i.e. "corr" or "R_sqr"
            default: "corr"
        - paired (bool):
            type of permutation pairing
            default: True
        - norm (bool):
            if True, normalized correlation data is returned, if corr_type if 
            "diff_corr"
            default: True
    """

    # determine type of randomization to use
    if permute == "sess":
        paired = True
    elif permute == "tracking":
        paired = "within"
    elif permute == "all":
        paired = False
    else:
        gen_util.accepted_values_error(
            "permute", permute, ["sess", "tracking", "all"]
            )

    # get permutation information
    if permute in ["sess", "all"] and "diff_" not in corr_type:
        corr_type = f"diff_{corr_type}"

    if corr_type == "diff_corr":
        norm = False # does not apply
    if "R_sqr" in corr_type and permpar.tails != "hi":
        raise NotImplementedError(
            "For R-squared analyses, permpar.tails should be set to 'hi'."
            )
    corr_types = ["corr", "diff_corr", "R_sqr", "diff_R_sqr"]
    if corr_type not in corr_types:
        gen_util.accepted_values_error("corr_type", corr_type, corr_types)

    return corr_type, paired, norm


#############################################
def get_norm_corrs(corr_data, med=0, corr_type="diff_corr"):
    """
    get_norm_corrs(corr_data)
    
    Returns normalized correlation values.

    Required args:
        - corr_data (1D array): 
            values to normalize
    
    Optional args:
        - med (float): 
            null distribution median for normalization
            default: 0
        - corr_type (str):
            type of correlation run (for checking), i.e. "diff_corr"
            default: "corr"

    Returns:
        - norm_corr_data (1D array): normalized correlations
    """

    if corr_type != "diff_corr":
        raise ValueError("Normalization should only be used with 'diff_corr'.")

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
def corr_bootstrapped_std(data, n_samples=1000, randst=None, corr_type="corr", 
                          return_rand=False, nanpol=None, med=0, norm=True):
    """
    corr_bootstrapped_std(data)
    
    Returns bootstrapped standard deviation for Pearson correlations.

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
        - randst (int or np.random.RandomState): 
            seed or random state to use when generating random values.
            default: None
        - return_rand (bool): if True, random correlations are returned
            default: False
        - nanpol (str): 
            policy for NaNs, "omit" or None
            default: None
        - med (float): 
            null distribution median for normalization, if norm is True
            default: 0
        - norm (bool):
            if True, normalized correlation data is returned
            default: True

    Returns:
        - bootstrapped_std (float): 
            bootstrapped standard deviation of correlations, 
            normalized if norm is True
        if return_rand:
        - rand_corrs (1D array): 
            randomly generated correlations, normalized if norm is True
    """

    randst = rand_util.get_np_rand_state(randst, set_none=True)

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
    rand_corrs = math_util.calc_op(
        list(data[:, randst.choice(choices, (n, n_samples), replace=True)]), 
        op=corr_type, nanpol=nanpol, axis=0,
        )
    
    if norm:
        rand_corrs = get_norm_corrs(rand_corrs, med=med, corr_type=corr_type)

    bootstrapped_std = math_util.error_stat(
        rand_corrs, stats="mean", error="std", nanpol=nanpol
        )
    
    if return_rand:
        return bootstrapped_std, rand_corrs
    else:
        return bootstrapped_std


#############################################
def get_corr_data(sess_pair, data_df, analyspar, permpar, 
                  corr_type="corr", permute="sess", absolute=False, norm=True, 
                  return_data=False, return_rand=False, n_rand_ex=1, 
                  randst=None, raise_no_pair=True):
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
        - corr_type (str):
            type of correlation to run, i.e. "corr" or "R_sqr"
            default: "corr"
        - permute (str):
            type of permutation to due ("tracking", "sess" or "all")
            default: "sess"
        - absolute (bool):
            if True, absolute USIs are used for correlation calculation instead 
            of signed USIs
            default: False
        - norm (bool):
            if True, normalized correlation data is returned, if corr_type if 
            "diff_corr"
            default: True
        - return_data (bool):
            if True, data to correlate is returned
            default: False
        - return_rand (bool):
            if True, random normalized correlation values are returned, along 
            with random data to correlate for one example permutation
            default: False
        - n_rand_ex (int):
            number of examples to return, if return_rand is True
            default: 1
        - randst (int or np.random.RandomState): 
            random state or seed value to use. (-1 treated as None)
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
        - null_CI (1D array):
            adjusted, null CI for the (normalized) correlation between sessions
        - p_val (float):
            uncorrected p-value for the correlation between sessions
        
        if return_data:
        - corr_data (2D array):
            data to correlate (grps (2) x datapoints)            
        
        if return_rand:
        - rand_corrs (1D array):
            (normalized) random correlation between sessions
        - rand_ex (3D array):
            example randomized data pairs to correlate 
            (grps (2) x datapoints x n_rand_ex)
        - rand_ex_corr (1D array):
            correlation for example randomized data pairs
    """

    nanpol = None if analyspar.remnans else "omit"

    if analyspar.stats != "mean" or analyspar.error != "std":
        raise NotImplementedError(
            "analyspar.stats must be set to 'mean', and "
            "analyspar.error must be set to 'std'."
            )

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

    # get updated correlation parameters
    corr_type, paired, norm = get_corr_info(
        permpar, corr_type=corr_type, permute=permute, norm=norm
        )

    # check correlation type and related parameters
    corr_data = np.vstack([roi_idxs[0], roi_idxs[1]]) # 2 x datapoints

    if absolute:
        corr_data = np.absolute(corr_data)

    # get actual correlation
    roi_corr = math_util.calc_op(corr_data, nanpol=nanpol, op=corr_type)

    # get first set of random values
    if return_rand:
        use_randst = copy.deepcopy(randst)
        if paired:
            perm_data = corr_data.T # groups x datapoints (2)
        else:
            perm_data = corr_data.reshape(1, -1) # 2 groups concatenate
        rand_exs = rand_util.run_permute(
            perm_data, n_perms=n_rand_ex, paired=paired, randst=use_randst
            )
        rand_exs = np.transpose(rand_exs, [1, 0, 2])
        if not paired:
            rand_exs = rand_exs.reshape(2, -1, n_rand_ex)
        rand_ex_corrs = math_util.calc_op(
            rand_exs, nanpol=nanpol, op=corr_type, axis=1
            )

    # get random correlation info
    returns = rand_util.get_op_p_val(
        corr_data, n_perms=permpar.n_perms, 
        stats=analyspar.stats, op=corr_type, return_CIs=True, 
        p_thresh=permpar.p_val, tails=permpar.tails, 
        multcomp=permpar.multcomp, paired=paired, nanpol=nanpol, 
        return_rand=return_rand, randst=randst
        )
    
    if return_rand:
        p_val, null_CI, rand_corrs = returns
    else:
        p_val, null_CI = returns

    med = null_CI[1]
    null_CI = np.asarray(null_CI)
    if norm:
        # normalize all data
        roi_corr = float(get_norm_corrs(roi_corr, med=med, corr_type=corr_type))
        null_CI = get_norm_corrs(null_CI, med=med, corr_type=corr_type)
    
    # get bootstrapped std over corr
    roi_corr_std = corr_bootstrapped_std(
        corr_data, n_samples=misc_analys.N_BOOTSTRP, randst=randst, 
        return_rand=False, nanpol=nanpol, norm=norm, med=med, 
        corr_type=corr_type
        )

    returns = [roi_corr, roi_corr_std, null_CI, p_val]
    
    if return_data:
        corr_data = np.vstack(corr_data)
        if "diff" in corr_type: # take diff
            corr_data[1] = corr_data[1] - corr_data[0]
        returns = returns + [corr_data]

    if return_rand:
        if norm:
            rand_corrs = get_norm_corrs(
                rand_corrs, med=med, corr_type=corr_type
                )
        if "diff" in corr_type: # take diff
            rand_exs[1] = rand_exs[1] - rand_exs[0]
        returns = returns + [rand_corrs, rand_exs, rand_ex_corrs]
    
    return returns


#############################################
def get_lp_idx_df(sessions, analyspar, stimpar, basepar, idxpar, permpar=None, 
                  sig_only=False, randst=None, parallel=False):
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
        - permpar (PermPar): 
            named tuple containing permutation parameters, required if 
            sig_only is True
            default: None
        - sig_only (bool):
            if True, ROIs with significant USIs are included 
            (only possible if analyspar.tracked is True)
            default: False
        - randst (int or np.random.RandomState): 
            random state or seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - lp_idx_df (pd.DataFrame):
            dataframe with one row per line/plane/session, and the following 
            columns, in addition to the basic sess_df columns:
            - roi_idxs (list): index for each ROI 
              (or each ROI that is significant in at least one session, 
              if sig_only)
    """

    if analyspar.tracked:
        misc_analys.check_sessions_complete(sessions, raise_err=True)
    
    if sig_only and permpar is None:
            raise ValueError("If sig_only is True, permpar cannot be None.")

    initial_columns = misc_analys.get_sess_df_columns(sessions[0], analyspar)

    args_dict = {
        "analyspar": analyspar,
        "stimpar"  : stimpar,
        "basepar"  : basepar,
        "idxpar"   : idxpar,
        "parallel" : parallel,
    } 

    if sig_only:
        idx_df = usi_analys.get_idx_sig_df(
            sessions, 
            permpar=permpar,
            randst=randst,
            aggreg_sess=True,
            **args_dict
            )
    else:
        idx_df = usi_analys.get_idx_only_df(sessions, **args_dict)
    
    # aggregate by line/plane/session
    lp_idx_df = pd.DataFrame(columns=initial_columns + ["roi_idxs"])

    # aggregate within line/plane/sessions
    group_columns = ["lines", "planes", "sess_ns"]
    aggreg_cols = [col for col in initial_columns if col not in group_columns]
    for grp_vals, grp_df in idx_df.groupby(group_columns):
        grp_df = grp_df.sort_values("mouse_ns")
        row_idx = len(lp_idx_df)
        for g, group_column in enumerate(group_columns):
            lp_idx_df.loc[row_idx, group_column] = grp_vals[g]

        # add aggregated values for initial columns
        lp_idx_df = misc_analys.aggreg_columns(
            grp_df, lp_idx_df, aggreg_cols, row_idx=row_idx, in_place=True
            )

        roi_idxs = grp_df["roi_idxs"].tolist()
        if sig_only:
            roi_idxs = [
                np.asarray(idx_vals)[np.asarray(sig_ns).astype(int)] 
                for idx_vals, sig_ns in zip(roi_idxs, grp_df["sig_idxs"])
                ]

        lp_idx_df.at[row_idx, "roi_idxs"] = np.concatenate(roi_idxs).tolist()
  
    lp_idx_df["sess_ns"] = lp_idx_df["sess_ns"].astype(int)

    return lp_idx_df


#############################################
def get_basic_idx_corr_df(lp_idx_df, consec_only=False, null_CI_cols=True):
    """
    get_basic_idx_corr_df(lp_idx_df)

    Returns index correlation dataframe for each line/plane, and optionally 
    columns added for null confidence intervals.

    Required args:
        - lp_idx_df (pd.DataFrame):
            dataframe with one row per line/plane/session, and the following 
            columns, in addition to the basic sess_df columns:
            - roi_idxs (list): index for each ROI

     Optional args:
        - consec_only (bool):
            if True, only consecutive session numbers are correlated
            default: True
        - null_CI_cols (bool):
            if True, null CI columns are included in the dataframe.

    Returns:
        - idx_corr_df (pd.DataFrame):
            dataframe with one row per line/plane, and the following 
            columns, in addition to the basic sess_df columns:
            - roi_idxs (list): index for each ROI

            if null_CI_cols:
            for session comparisons, e.g. 1v2
            - {}v{}_null_CIs (object): empty
    """

    initial_columns = [col for col in lp_idx_df.columns if col != "roi_idxs"]

    # get correlation pairs
    corr_ns = get_corr_pairs(lp_idx_df, consec_only=consec_only)

    # aggregate by line/plane for correlation dataframe
    group_columns = ["lines", "planes"]
    
    all_columns = initial_columns
    if null_CI_cols:
        CI_columns = [
            f"{corr_pair[0]}v{corr_pair[1]}_null_CIs" for corr_pair in corr_ns
            ]
        all_columns = initial_columns + CI_columns
    
    idx_corr_df = pd.DataFrame(columns=all_columns)
    aggreg_cols = [
        col for col in initial_columns if col not in group_columns
        ]

    for grp_vals, grp_df in lp_idx_df.groupby(group_columns):
        grp_df = grp_df.sort_values("sess_ns") # mice already aggregated
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
def get_ex_idx_corr_norm_df(sessions, analyspar, stimpar, basepar, idxpar, 
                            permpar, permute="sess", sig_only=False, n_bins=40, 
                            randst=None, parallel=False):
    """
    get_ex_idx_corr_norm_df(sessions, analyspar, stimpar, basepar, idxpar, 
                            permpar)

    Returns example correlation normalization data.

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
        - permute (bool):
            type of permutation to due ("tracking", "sess" or "all")
            default: "sess"
        - sig_only (bool):
            if True, ROIs with significant USIs are included 
            (only possible if analyspar.tracked is True)
            default: False
        - n_bins (int):
            number of bins
            default: 40
        - randst (int or np.random.RandomState): 
            seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - idx_corr_norm_df (pd.DataFrame):
            dataframe with one row for a line/plane, and the 
            following columns, in addition to the basic sess_df columns:

            for a specific session comparison, e.g. 1v2
            - {}v{}_corrs (float): unnormalized intersession ROI index 
                correlations
            - {}v{}_norm_corrs (float): normalized intersession ROI index 
                correlations
            - {}v{}_rand_ex_corrs (float): unnormalized intersession 
                ROI index correlations for an example of randomized data
            - {}v{}_rand_corr_meds (float): median of randomized correlations

            - {}v{}_corr_data (list): intersession values to correlate
            - {}v{}_rand_ex (list): intersession values for an example of 
                randomized data
            - {}v{}_rand_corrs_binned (list): binned random unnormalized 
                intersession ROI index correlations
            - {}v{}_rand_corrs_bin_edges (list): bins edges
    """

    nanpol = None if analyspar.remnans else "omit"

    initial_columns = misc_analys.get_sess_df_columns(sessions[0], analyspar)
    
    lp_idx_df = get_lp_idx_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar,
        permpar=permpar,
        sig_only=sig_only,
        randst=randst,
        parallel=parallel,
        )

    idx_corr_norm_df = get_basic_idx_corr_df(lp_idx_df, consec_only=False)
    if len(idx_corr_norm_df) != 1:
        raise ValueError("sessions should be from the same line/plane.")

    # get correlation pairs
    corr_ns = get_corr_pairs(lp_idx_df)

    if len(corr_ns) != 1:
        raise ValueError("Sessions should allow only one pair.")
    sess_pair = corr_ns[0]
    corr_name = f"{sess_pair[0]}v{sess_pair[1]}"

    drop_columns = [
        col for col in idx_corr_norm_df.columns if col not in initial_columns
        ]
    idx_corr_norm_df = idx_corr_norm_df.drop(columns=drop_columns)

    logger.info(
        ("Calculating ROI USI correlations for a single session pair..."), 
        extra={"spacing": TAB}
        )

    corr_type = "diff_corr"
    returns = get_corr_data(
        sess_pair, 
        data_df=lp_idx_df, 
        analyspar=analyspar, 
        permpar=permpar, 
        permute=permute, 
        corr_type=corr_type,
        absolute=False,
        norm=False,
        return_data=True,
        return_rand=True,
        n_rand_ex=1, 
        randst=randst
        )

    roi_corr, _, _, _, corr_data, rand_corrs, rand_exs, rand_ex_corrs = returns
    rand_ex = rand_exs[..., 0]
    rand_ex_corr = rand_ex_corrs[0]

    rand_corr_med = math_util.mean_med(
        rand_corrs, stats="median", nanpol=nanpol
        )
    norm_roi_corr = float(
        get_norm_corrs(roi_corr, med=rand_corr_med, corr_type=corr_type)
        )

    row_idx = idx_corr_norm_df.index[0]

    idx_corr_norm_df.loc[row_idx, f"{corr_name}_corrs"] = roi_corr
    idx_corr_norm_df.loc[row_idx, f"{corr_name}_rand_ex_corrs"] = rand_ex_corr
    idx_corr_norm_df.loc[row_idx, f"{corr_name}_rand_corr_meds"] = rand_corr_med
    idx_corr_norm_df.loc[row_idx, f"{corr_name}_norm_corrs"] = norm_roi_corr

    cols = [
        f"{corr_name}_{col_name}" 
        for col_name in 
        ["corr_data", "rand_ex", "rand_corrs_binned", "rand_corrs_bin_edges"]
        ]
    idx_corr_norm_df = gen_util.set_object_columns(
        idx_corr_norm_df, cols, in_place=True
        )

    idx_corr_norm_df.at[row_idx, f"{corr_name}_corr_data"] = corr_data.tolist()
    idx_corr_norm_df.at[row_idx, f"{corr_name}_rand_ex"] = rand_ex.tolist()

    fcts = [np.min, np.max] if nanpol is None else [np.nanmin, np.nanmax]
    bounds = [fct(rand_corrs) for fct in fcts]
    bins = np.linspace(*bounds, n_bins + 1)
    rand_corrs_binned = np.histogram(rand_corrs, bins=bins)[0]

    idx_corr_norm_df.at[row_idx, f"{corr_name}_rand_corrs_bin_edges"] = \
        [bounds[0], bounds[-1]]
    idx_corr_norm_df.at[row_idx, f"{corr_name}_rand_corrs_binned"] = \
        rand_corrs_binned.tolist()

    return idx_corr_norm_df
    

#############################################
def get_idx_corrs_df(sessions, analyspar, stimpar, basepar, idxpar, permpar, 
                     consec_only=True, permute="sess", corr_type="corr", 
                     sig_only=False, randst=None, parallel=False):
    """
    get_idx_corrs_df(sessions, analyspar, stimpar, basepar, idxpar, permpar)

    Returns ROI index correlation data for each line/plane/session comparison.

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
        - corr_type (str):
            type of correlation to run, i.e. "corr" or "R_sqr"
            default: "corr"
        - permute (bool):
            type of permutation to due ("tracking", "sess" or "all")
            default: "sess"
        - sig_only (bool):
            if True, ROIs with significant USIs are included 
            (only possible if analyspar.tracked is True)
            default: False
        - randst (int or np.random.RandomState): 
            seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
   
    Returns:
        - idx_corr_df (pd.DataFrame):
            dataframe with one row per line/plane, and the 
            following columns, in addition to the basic sess_df columns:

            for correlation data (normalized if corr_type is "diff_corr") for 
            session comparisons, e.g. 1v2
            - {}v{}{norm_str}_corrs (float): intersession ROI index correlations
            - {}v{}{norm_str}_corr_stds (float): bootstrapped intersession ROI 
                index correlation standard deviation
            - {}v{}_null_CIs (list): adjusted null CI for intersession ROI 
                index correlations
            - {}v{}_raw_p_vals (float): p-value for intersession correlations
            - {}v{}_p_vals (float): p-value for intersession correlations, 
                corrected for multiple comparisons and tails
    """
    
    lp_idx_df = get_lp_idx_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar,
        permpar=permpar,
        sig_only=sig_only,
        randst=randst,
        parallel=parallel,
        )

    idx_corr_df = get_basic_idx_corr_df(lp_idx_df, consec_only=consec_only)

    # get correlation pairs
    corr_ns = get_corr_pairs(lp_idx_df, consec_only=consec_only)

    # get norm information
    norm = False
    if permute in ["sess", "all"]:
        corr_type = f"diff_{corr_type}"
        if corr_type == "diff_corr":
            norm = True
    norm_str = "_norm" if norm else ""

    logger.info(
        ("Calculating ROI USI correlations across sessions..."), 
        extra={"spacing": TAB}
        )
    group_columns = ["lines", "planes"]
    for grp_vals, grp_df in lp_idx_df.groupby(group_columns):
        grp_df = grp_df.sort_values("sess_ns") # mice already aggregated
        line, plane = grp_vals
        row_idx = idx_corr_df.loc[
            (idx_corr_df["lines"] == line) &
            (idx_corr_df["planes"] == plane)
        ].index

        if len(row_idx) != 1:
            raise RuntimeError("Expected exactly one row to match.")
        row_idx = row_idx[0]
    
        use_randst = copy.deepcopy(randst) # reset each time

        # obtain correlation data
        args_dict = {
            "data_df"  : grp_df,
            "analyspar": analyspar,
            "permpar"  : permpar,
            "permute"  : permute,
            "corr_type": corr_type,
            "absolute" : False,
            "norm"     : norm,
            "randst"   : use_randst,
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
            roi_corr, roi_corr_std, null_CI, p_val = corr_data

            idx_corr_df.loc[row_idx, f"{corr_name}{norm_str}_corrs"] = roi_corr
            idx_corr_df.loc[row_idx, f"{corr_name}{norm_str}_corr_stds"] = \
                roi_corr_std
            idx_corr_df.at[row_idx, f"{corr_name}_null_CIs"] = null_CI.tolist()
            idx_corr_df.loc[row_idx, f"{corr_name}_p_vals"] = p_val

    # corrected p-values
    idx_corr_df = misc_analys.add_corr_p_vals(idx_corr_df, permpar)

    return idx_corr_df


#############################################
def corr_scatterplots(sessions, analyspar, stimpar, basepar, idxpar, permpar, 
                      permute="sess", sig_only=False, randst=None, n_bins=200, 
                      parallel=False):
    """
    corr_scatterplots(sessions, analyspar, stimpar, basepar, idxpar, permpar)

    Returns ROI index correlation scatterplot data for each line/plane/session 
    comparison.

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
        - permute (bool):
            type of permutation to due ("tracking", "sess" or "all")
            default: "sess"
        - sig_only (bool):
            if True, ROIs with significant USIs are included 
            (only possible if analyspar.tracked is True)
            default: False
        - randst (int or np.random.RandomState): 
            seed value to use. (-1 treated as None)
            default: None
        - n_bins (int): 
            number of bins for random data
            default: 200
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
   
    Returns:
        - idx_corr_df (pd.DataFrame):
            dataframe with one row per line/plane, and the 
            following columns, in addition to the basic sess_df columns:

            for correlation data (normalized if corr_type is "diff_corr") for 
            session comparisons (x, y), e.g. 1v2
            - binned_rand_stats (list): number of random correlation values per 
                bin (xs x ys)
            - corr_data_xs (list): USI values for x
            - corr_data_ys (list): USI values for y
            - corrs (float): correlation between session data (x and y)
            - p_vals (float): p-value for correlation, corrected for 
                multiple comparisons and tails
            - rand_corr_meds (float): median of the random correlations
            - raw_p_vals (float): p-value for intersession correlations
            - regr_coefs (float): regression correlation coefficient (slope)
            - regr_intercepts (float): regression correlation intercept
            - x_bin_mids (list): x mid point for each random correlation bin
            - y_bin_mids (list): y mid point for each random correlation bin
    """
    
    lp_idx_df = get_lp_idx_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar,
        permpar=permpar,
        sig_only=sig_only,
        randst=randst,
        parallel=parallel,
        )

    idx_corr_df = get_basic_idx_corr_df(
        lp_idx_df, consec_only=False, null_CI_cols=False
        )

    # get correlation pairs
    corr_ns = get_corr_pairs(lp_idx_df)
    if len(corr_ns) != 1:
        raise ValueError("Expected only 1 session correlation pair.")
    sess_pair = corr_ns[0]

    # get norm information
    norm = False
    corr_type = "corr"
    if permute in ["sess", "all"]:
        corr_type = "diff_corr"
        norm = True

    # add array columns
    columns = ["corr_data_xs", "corr_data_ys", "binned_rand_stats", 
        "x_bin_mids", "y_bin_mids"]
    idx_corr_df = gen_util.set_object_columns(idx_corr_df, columns)

    logger.info(
        ("Calculating ROI USI correlations across sessions..."), 
        extra={"spacing": TAB}
        )
    group_columns = ["lines", "planes"]
    for grp_vals, grp_df in lp_idx_df.groupby(group_columns):
        grp_df = grp_df.sort_values("sess_ns") # mice already aggregated
        line, plane = grp_vals
        row_idx = idx_corr_df.loc[
            (idx_corr_df["lines"] == line) &
            (idx_corr_df["planes"] == plane)
        ].index

        if len(row_idx) != 1:
            raise RuntimeError("Expected exactly one row to match.")
        row_idx = row_idx[0]

        if len(grp_df) > 2:
            raise RuntimeError("Expected no more than 2 rows to correlate.")
        if len(grp_df) < 2:
            continue # no pair
    
        use_randst = copy.deepcopy(randst) # reset each time

        # obtain correlation data
        args_dict = {
            "data_df"    : grp_df,
            "analyspar"  : analyspar,
            "permpar"    : permpar,
            "permute"    : permute,
            "corr_type"  : corr_type,
            "absolute"   : False,
            "norm"       : norm,
            "randst"     : use_randst,
            "return_data": True,
            "return_rand": True,
            "n_rand_ex"  : 1000,
        }

        all_corr_data = get_corr_data(sess_pair, **args_dict)
        [roi_corr, _, null_CI, p_val, corr_data, _, rand_exs, _] = all_corr_data

        regr = LinearRegression().fit(corr_data[0].reshape(-1, 1), corr_data[1])

        # bin data
        rand_stats, x_edge, y_edge = np.histogram2d(
            rand_exs[0].reshape(-1), rand_exs[1].reshape(-1), bins=n_bins, 
            density=False
            )
        x_mids = np.diff(x_edge) / 2 + x_edge[:-1]
        y_mids = np.diff(y_edge) / 2 + y_edge[:-1]

        rand_binned = scind.gaussian_filter(
            rand_stats, n_bins / 20, mode="constant"
            )

        idx_corr_df.loc[row_idx, "corrs"] = roi_corr
        idx_corr_df.loc[row_idx, "rand_corr_meds"] = null_CI[1]
        idx_corr_df.loc[row_idx, "regr_coefs"] = regr.coef_
        idx_corr_df.loc[row_idx, "regr_intercepts"] = regr.intercept_

        idx_corr_df.at[row_idx, "corr_data_xs"] = corr_data[0].tolist()
        idx_corr_df.at[row_idx, "corr_data_ys"] = corr_data[1].tolist()

        idx_corr_df.at[row_idx, "binned_rand_stats"] = rand_binned.tolist()
        idx_corr_df.at[row_idx, "x_bin_mids"] = x_mids.tolist()
        idx_corr_df.at[row_idx, "y_bin_mids"] = y_mids.tolist()

        idx_corr_df.loc[row_idx, "p_vals"] = p_val

    # corrected p-values
    idx_corr_df = misc_analys.add_corr_p_vals(idx_corr_df, permpar)

    return idx_corr_df

