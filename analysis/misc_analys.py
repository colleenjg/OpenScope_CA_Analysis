"""
misc_analys.py

This script contains miscellaneous analysis functions.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import warnings

import numpy as np
import pandas as pd
import scipy.stats as scist

from util import gen_util, math_util
from sess_util import sess_ntuple_util, sess_gen_util


N_BOOTSTRP = int(1e4)


#############################################
def get_sig_symbol(corr_p_val, ctrl=False, percentile=False, sensitivity=None, 
                   side=1, tails=2, p_thresh=0.05):
    """
    get_sig_symbol(corr_p_val)

    Return significance symbol.

    Required args:
        - corr_p_val (float): 
            corrected p-value (e.g., corrected for multiple comparisons and 
            tails)

    Optional args:
        - ctrl (bool): 
            if True, control symbol ("+") is used instead of "*"
            default: False
        - percentile (bool):
            if True, corr_p_val is a percentile (0-100) instead of a 
            p-value (0-1)
            default: False
        - sensitivity (float): 
            minimum p-value or percentile that can be measured (based on number 
            of permutations, comparisons and tails)  
            default: None
        - side (int): 
            side of the distribution
            1: above the median
            -1: below the median
            default: 1
        - tails (str or int):
            tails for significance assessment
            default: 2
    Returns:
        - sig_symbol (str): 
            significance symbol ("" if not significant)
    """
    
    if percentile:
        if corr_p_val > 50 and str(tails) in ["hi", "2"]:
            corr_p_val = 100 - corr_p_val
        corr_p_val = corr_p_val / 100
    
    # double check if side matches tail
    if str(tails) != "2":
        if (tails == "hi" and side == -1) or (tails == "lo" and side == 1):
            return ""

    # corrected for sensitivity
    if sensitivity is not None:
        corr_p_val = np.max([corr_p_val, sensitivity])

    sig_symbol = ""
    levels = [50, 5, 1]
    if corr_p_val < p_thresh / levels[0]: # 0.001 if p_thresh = 0.05
        sig_symbol = "***"
    elif corr_p_val < p_thresh / levels[1]: # 0.01 if p_thresh = 0.05
        sig_symbol = "**"
    elif corr_p_val < p_thresh / levels[2]: # 0.05 if p_thresh = 0.05
        sig_symbol = "*"
    if ctrl:
        sig_symbol = sig_symbol.replace("*", "+")

    return sig_symbol


#############################################
def get_corrected_p_val(p_val, permpar, raise_multcomp=True):
    """
    get_corrected_p_val(p_val, permpar)

    Returns p-value, Bonferroni corrected for number of tails and multiple 
    comparisons.
    
    Required args:
        - p_val (float): 
            raw p-value
        - permpar (PermPar or dict): 
            named tuple containing permutation parameters

    Optional args:
        - raise_multcomp (bool):
            if True, an error is raised if permpar.multcomp is False
            default: True

    Returns:
        - corr_p_val (float): 
            corrected p-value
    """

    if isinstance(permpar, dict):
        permpar = sess_ntuple_util.init_permpar(**permpar)

    n_tails = 1 if permpar.tails in ["lo", "hi"] else int(permpar.tails)
    corr_p_val = p_val * n_tails 
    
    if permpar.multcomp:
        corr_p_val *= permpar.multcomp
    elif raise_multcomp:
        raise ValueError("permpar.multcomp is set to False.")

    corr_p_val = np.min([corr_p_val, 1])

    return corr_p_val


#############################################
def add_corr_p_vals(df, permpar, raise_multcomp=True):
    """
    add_corr_p_vals(df, permpar)

    Returns dataframe with p-values, corrected for tails and multiple 
    comparisons, added, if permpar.multcomp is True. If any case, original 
    "p_vals" column names are returned as "raw_p_vals" instead.
    
    Required args:
        - df (pd.DataFrame): 
            dataframe with p-value columns ("p_vals" in column names)
        - permpar (PermPar or dict): 
            named tuple containing permutation parameters

    Optional args:
        - raise_multcomp (bool):
            if True, an error is raised if permpar.multcomp is False
            default: True

    Returns:
        - df (pd.DataFrame): 
            dataframe with raw p-value columns names changed to "raw_{}", 
            and corrected p-value columns added, if permpar.multcomp
    """

    if isinstance(permpar, dict):
        permpar = sess_ntuple_util.init_permpar(**permpar)

    p_val_cols = [col for col in df.columns if "p_vals" in col]

    if sum(["raw_p_vals" in col for col in p_val_cols]):
        raise ValueError(
            "Function converts 'p_vals' columns to 'raw_p_vals' columns. "
            "Dataframe should not already contain columns with 'raw_p_vals' "
            "in name.")

    new_col_names = {
        col: col.replace("p_vals", "raw_p_vals") for col in p_val_cols
        }
    df = df.rename(columns=new_col_names)
    # define function with arguments for use with .map()
    correct_p_val_fct = lambda x: get_corrected_p_val(
        x, permpar, raise_multcomp
        )

    for corr_p_val_col in p_val_cols:
        raw_p_val_col = corr_p_val_col.replace("p_vals", "raw_p_vals")
        df[corr_p_val_col] = df[raw_p_val_col].map(correct_p_val_fct)
    
    return df
    
    
#############################################
def get_binom_sensitivity(n_items, null_perc=50, side=1):
    """
    get_binom_sensitivity(n_items)

    Returns p-value sensitivity, i.e., the smallest non zero p-value that can 
    be measured given the discrete binomial distribution constructed from the 
    data.

    The sensitivity is measured for a specific side of the distribution 
    (above or below the median), given that the distribution may be 
    asymmetrical.
    
    Required args:
        - n_items (int): 
            number of items

    Optional args:
        - null_perc (float): 
            null percentage expected
            default: 50
        - side (int): 
            side of the distribution
            1: above the median
            -1: below the median
            default: 1

    Returns:
        - sensitivity (float): 
            minimum theoretical p-value
    """

    if side == 1:
        x = n_items - 1 # above the median
    elif side == -1:
        x = 1 # below the median
    else:
        raise ValueError("Expected 'side' to be an int, of value of either 1 "
            "(above the median) or -1 (below the median), only.")
    
    sensitivity = scist.binom.cdf(x, n_items, null_perc / 100)

    if side == 1:
        sensitivity = 1 - sensitivity

    return sensitivity


#############################################
def get_sensitivity(permpar):
    """
    get_sensitivity(permpar)

    Returns p-value sensitivity, i.e., the smallest non zero p-value that can 
    be measured given the number of permutations used and Bonferroni 
    corrections for number of tails and multiple comparisons.
    
    Required args:
        - permpar (PermPar or dict): 
            named tuple containing permutation parameters

    Returns:
        - sensitivity (float): 
            minimum theoretical p-value
    """

    if isinstance(permpar, dict):
        permpar = sess_ntuple_util.init_permpar(**permpar)

    n_tails = 1 if permpar.tails in ["lo", "hi"] else int(permpar.tails)
    sensitivity = n_tails / permpar.n_perms

    if permpar.multcomp:
        sensitivity *= permpar.multcomp


    return sensitivity


#############################################
def get_comp_info(permpar):
    """
    get_comp_info(permpar)

    Returns p-value correction information.
    
    Required args:
        - permpar (PermPar or dict): 
            named tuple containing permutation parameters

    Returns:
        - full_comp_info (str): 
            string containing tails and multiple comparisons information
    """

    if isinstance(permpar, dict):
        permpar = sess_ntuple_util.init_permpar(**permpar)

    if permpar.tails == "lo":
        comp_info = "one-tailed"
    elif permpar.tails == "hi":
        comp_info = "one-tailed"
    elif int(permpar.tails) == 2:
        comp_info = "two-tailed"
    else:
        gen_util.accepted_values_error(
            "permpar.tails", permpar.tails, ["lo", "hi", 2]
            )

    if permpar.multcomp:
        comp_info = f"{int(permpar.multcomp)} comparisons, {comp_info}"
        corr_str = "Corrected"
    else:
        corr_str = "Raw"
    
    full_comp_info = f"{corr_str} p-values ({comp_info})"

    return full_comp_info


#############################################
def get_check_sess_df(sessions, sess_df=None, analyspar=None, roi=True): 
    """
    get_check_sess_df(sessions)

    Checks a dataframe against existing sessions (that they match and are in 
    the same order), or returns a dataframe with session information if sess_df 
    is None.

    Required args:
        - sessions (list):
            Session objects

    Optional args:
        - sess_df (pd.DataFrame):
            dataframe containing session information (see keys under Returns)
            default: None
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters, used if sess_df is None
        - roi (bool):
            if True, ROI data is included in sess_df, used if sess_df is None


    Returns:
        - sess_df (pd.DataFrame):
            dataframe containing session information under the following keys:
            "mouse_ns", "mouseids", "sess_ns", "sessids", "lines", "planes"
            if datatype == "roi":
                "nrois", "twop_fps"
            if not rem_bad: 
                "bad_rois_{}" (depending on fluor)
    """

    sessions = gen_util.list_if_not(sessions)

    if sess_df is None:
        roi_kwargs = dict()
        if analyspar is None and roi:
            raise ValueError("If sess_df is None, must pass analyspar.")
        elif analyspar is not None:
            roi_kwargs["fluor"] = analyspar.fluor
            roi_kwargs["rem_bad"] = analyspar.rem_bad

        sess_df = sess_gen_util.get_sess_info(
            sessions, incl_roi=roi, return_df=True, **roi_kwargs
            )

    else:
        if len(sess_df) != len(sessions):
            raise ValueError(
                "'sess_df' should have as many rows as 'sessions'.")
        # check order
        sessids = np.asarray([sess.sessid for sess in sessions]).astype(int)
        sess_df_sessids = sess_df.sessids.to_numpy().astype(int)

        if len(sessids) != len(sess_df_sessids):
            raise ValueError("'sess_df' is not the same length at 'sessions'.")

        elif (np.sort(sessids) != np.sort(sess_df_sessids)).any():
            raise ValueError("Sessions do not match ids in 'sess_df'.")

        elif (sessids != sess_df_sessids).any():
            raise ValueError("Sessions do not appear in order in 'sess_df'.")

    return sess_df


#############################################
def get_sess_df_columns(session, analyspar, roi=True): 
    """
    get_sess_df_columns(session, analyspar)

    Returns basic session dataframe columns.

    Required args:
        - session (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters, used if sess_df is None

    Optional args:
        - roi (bool):
            if True, ROI data is included in sess_df, used if sess_df is None


    Returns:
        - sess_df_cols (list):
            session dataframe columns
    """

    sess_df = sess_gen_util.get_sess_info(
        [session], fluor=analyspar.fluor, incl_roi=roi, return_df=True, 
        rem_bad=analyspar.rem_bad
        )

    sess_df_cols = sess_df.columns.tolist()

    return sess_df_cols


#############################################
def check_sessions_complete(sessions, raise_err=False):
    """
    check_sessions_complete(sessions)

    Checks for mice for which session series are incomplete and removes them, 
    raising a warning.

    Required args:
        - sessions (list):
            Session objects

    Optional args:
        - raise_err (bool):
            if True, an error is raised if any mouse has missing sessions.
            default: False

    Returns:
        - sessions (list):
            Session objects, with sessions belonging to incomplete series 
            removed
    """

    mouse_ns = [sess.mouse_n for sess in sessions]
    sess_ns = [sess.sess_n for sess in sessions]

    unique_sess_ns = set(sess_ns)
    unique_mouse_ns = set(mouse_ns)

    remove_mouse = []
    remove_idxs = []
    for m in list(unique_mouse_ns):
        mouse_idxs = [i for i, mouse_n in enumerate(mouse_ns) if mouse_n == m]
        mouse_sess_ns = [sess_ns[i] for i in mouse_idxs]
        if set(mouse_sess_ns) != unique_sess_ns:
            remove_mouse.append(m)
            remove_idxs.extend(mouse_idxs)
    
    if len(remove_idxs):
        mice_str = ", ".join([str(m) for m in remove_mouse])
        sessions = [
            sess for i, sess in enumerate(sessions) if i not in remove_idxs
            ]
        message = f"missing sessions: {mice_str}"
        if raise_err:
            raise RuntimeError("The following mice have {}")
        warnings.warn(f"Removing the following mice, as they have {message}", 
            category=UserWarning, stacklevel=1)
    
    if len(sessions) == 0:
        raise RuntimeError(
            "All mice were removed, as all have missing sessions."
            )

    return sessions


#############################################
def aggreg_columns(source_df, targ_df, aggreg_cols, row_idx=0, 
                   sort_by="sessids", in_place=False, by_mouse=False):
    """
    aggreg_columns(source_df, targ_df, aggreg_cols)

    Required args:
        - source_df (pd.DataFrame):
            source dataframe
        - targ_df (pd.DataFrame):
            target dataframe
        - aggreg_cols
            columns to aggregate from source dataframe

    Optional args:
        - row_idx (int or str):
            target dataframe row to add values to
            default: 0
        - sort_by (str):
            column to sort by (must be on the columns to aggregate).
            False or None, for no sorting
            default: "sessids"
        - in_place (bool):
            if True, targ_df is modified in place. Otherwise, a deep copy is 
            modified. targ_df is returned in either case.
            default: False
        - by_mouse (bool):
            if True, data is understood to be aggregated by mouse. So, if 
            source_df contains only one row, its values are not placed in a 
            list.
            default: False

    Returns:
        - targ_df ()
    """
    
    if not in_place:
        targ_df = targ_df.copy(deep=True)

    retain_single = False
    if by_mouse:
        if "mouse_ns" in aggreg_cols:
            raise ValueError(
                "If 'by_mouse', 'mouse_ns' should not be a column in "
                "'aggreg_cols'.")
        if len(source_df) == 1:
            retain_single = True

    sort_order = None
    if sort_by:
        if sort_by not in aggreg_cols:
            raise ValueError(
                "If sort_by is not False or None, it must be in aggreg_cols."
                )
        sorter = source_df[sort_by].tolist()
        sort_order = np.argsort(sorter)

    for column in aggreg_cols:
        values = source_df[column].tolist()
        if retain_single:
            values = values[0]
        elif sort_order is not None:
            values = [values[v] for v in sort_order]
        targ_df.at[row_idx, column] = values # may throw an error if row doesn't yet exist

    return targ_df


#############################################
def get_sess_ns(sesspar, data_df):
    """
    get_sess_ns(sesspar, data_df)

    Returns array of session numbers, inferred from sesspar, if possible, or 
    from a dataframe.

    Required args:
        - sesspar (SessPar or dict): 
            named tuple containing session parameters
        - data_df (pd.DataFrame):
            dataframe with a 'sess_ns' column containing individual session 
            numbers for each row in the dataframe

    Returns:
        - sess_ns (1D array):
            array of session numbers, in order
    """

    if isinstance(sesspar, dict):
        sesspar = sess_ntuple_util.init_sesspar(**sesspar)

    if sesspar.sess_n in ["any", "all"]:
        if "sess_ns" not in data_df.columns:
            raise KeyError("data_df is expected to contain a 'sess_ns' column.")
        sess_ns = np.arange(data_df.sess_ns.min(), data_df.sess_ns.max() + 1)
    else:
        sess_ns = np.asarray(sesspar.sess_n).reshape(-1)

    return sess_ns


#############################################
def set_multcomp(permpar, sess_df, CIs=True, pairs=True, factor=1):
    """
    set_multcomp(permpar)

    Returns permpar updated with the number of comparisons computed from the 
    sessions.

    Required args:
        - permpar (PermPar): 
            named tuple containing permutation parameters
        - sess_df (pd.DataFrame):
            dataframe containing session information, including the following 
            keys: "sess_ns", "lines", "planes"
    
    Optional args:
        - CIs (bool):
            include comparisons to CIs comparisons
            default: True
        - pairs (bool):
            include paired comparisons
            default: True
        - factor (int): 
            additional factor by which to multiply the number of comparisons
            default: 1

    Returns:
        - permpar (PermPar): 
            updated permutation parameter named tuple
    """

    if isinstance(permpar, dict):
        permpar = sess_ntuple_util.init_permpar(**permpar)

    n_comps = 0    
    for _, sess_df_grp in sess_df.groupby(["lines", "planes"]):
        n_sess = len(sess_df_grp)

        # sessions compared to CIs
        if CIs:
            n_comps += n_sess

        # session pair comparisons
        if pairs:
            k = 2
            if n_sess >= k:
                fact = np.math.factorial
                n_comps += fact(n_sess) / (fact(k) * fact(n_sess - k))

    # multiplied by specified factor
    n_comps *= factor

    permpar = sess_ntuple_util.get_modif_ntuple(
            permpar, "multcomp", int(n_comps)
        )

    return permpar


#############################################
def get_snr(session, analyspar, datatype="snrs", block_size=100):
    """
    get_snr(session, analyspar)

    Returns SNR related information for the ROIs in a session.

    Required args:
        - session (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters

    Optional args:
        - datatype (str):
            type of data to retrieve ("snrs" or "signal_means")
            default: "snrs"
        - block_size (int):
            number of ROIs for which to load data at a time
            default: 100

    Returns:
        - data (1D array):
            data, depending on datatype, for each ROI in a session
    """
    
    if session.only_tracked_rois != analyspar.tracked:
        raise RuntimeError(
            "session.only_tracked_rois should match analyspar.tracked."
            )

    if analyspar.scale:
        raise ValueError("analyspar.scale must be False for SNR analysis.")

    if analyspar.tracked:
        keep_rois = session.tracked_rois
        if analyspar.rem_bad and len(session.get_bad_rois(fluor=analyspar.fluor)):
            raise NotImplementedError(
                "rem_bad not implemented for tracked ROIs."
                )
    else:
        keep_rois = np.arange(session.get_nrois(rem_bad=False))
        if analyspar.rem_bad:
            bad_rois = session.get_bad_rois(analyspar.fluor)
            if len(bad_rois):
                keep_rois = np.delete(
                    keep_rois, np.asarray(bad_rois)
                    ).astype(int)
    
    datatypes = ["snrs", "signal_means"]
    if datatype not in datatypes:
        gen_util.accepted_values_error("datatype", datatype, datatypes)
    index = 0 if datatype == "snrs" else -1

    data = np.full(len(keep_rois), np.nan)
    for i in range(len(keep_rois)):
        if not i % block_size: # load by block
            block_ns = keep_rois[i : i + block_size]
            roi_traces = session.get_roi_traces_by_ns(block_ns, analyspar.fluor)
        data[i] = math_util.calculate_snr(
            roi_traces[i % block_size],
            return_stats=True
            )[index]

    return data


#############################################
def get_correlation(session, analyspar, rolling_win=4):
    """
    get_correlation(session, analyspar)

    Returns ROI correlations for a session.

    Required args:
        - session (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters

    Optional args:
        - rolling_win (int):
            window to use in rolling mean over individual traces before 
            computing correlation between ROIs (None for no smoothing)
            default: 4 

    Returns:
        - corr_triu (1D array):
            all correlations
    """
    
    if session.only_tracked_rois != analyspar.tracked:
        raise RuntimeError(
            "session.only_tracked_rois should match analyspar.tracked."
            )

    if analyspar.scale:
        raise ValueError(
            "analyspar.scale must be False for correlation analysis."
            )
    
    full_traces_df = session.get_roi_traces(
        fluor=analyspar.fluor, 
        rem_bad=analyspar.rem_bad 
        )

    full_traces = gen_util.reshape_df_data(full_traces_df, squeeze_cols=True)
    
    if rolling_win is not None:
        full_traces = math_util.rolling_mean(full_traces, win=rolling_win)

    corrs = np.corrcoef(full_traces)

    corr_triu = corrs[np.triu_indices(len(corrs), k=1)]

    return corr_triu


#############################################
def get_all_correlations(sessions, analyspar, n_bins=40, rolling_win=4, 
                         parallel=False):
    """
    get_all_correlations(sessions, analyspar)

    Returns ROI correlation data for each line/plane/session.

    Required args:
        - session (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
    
    Optional args:
        - n_bins (int):
            number of bins for correlation data
            default: 40
        - parallel (bool):
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
        - rolling_win (int):
            window to use in rolling mean over individual traces before 
            computing correlation between ROIs (None for no smoothing)
            default: 4 
        
    Returns:
        - binned_cc_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the 
            following columns, in addition to the basic sess_df columns:
            - bin_edges (list): first and last bin edge
            - corrs_binned (list): number of correlation values per bin
    """

    all_corrs = gen_util.parallel_wrap(
        get_correlation, sessions, [analyspar, rolling_win], parallel=parallel
        )

    cc_df = get_check_sess_df(sessions, analyspar=analyspar)
    initial_columns = cc_df.columns

    cc_df["corrs"] = [cc.tolist() for cc in all_corrs]

    # group within line/plane
    group_columns = ["lines", "planes", "sess_ns"]

    columns = initial_columns.tolist() + ["bin_edges", "corrs_binned"]
    binned_cc_df = pd.DataFrame(columns=columns)
    aggreg_cols = [col for col in initial_columns if col not in group_columns]
    for grp_vals, grp_df in cc_df.groupby(group_columns):
        grp_df = grp_df.sort_values("mouse_ns")
        row_idx = len(binned_cc_df)
        for g, group_column in enumerate(group_columns):
            binned_cc_df.loc[row_idx, group_column] = grp_vals[g]

        # add aggregated values for initial columns
        binned_cc_df = aggreg_columns(
            grp_df, binned_cc_df, aggreg_cols, row_idx=row_idx, in_place=True
            )

        cc_data = np.concatenate(grp_df["corrs"].tolist())

        cc_data_binned, bin_edges = np.histogram(
            cc_data, bins=np.linspace(-1, 1, n_bins + 1)
            )

        binned_cc_df.at[row_idx, "corrs_binned"] = cc_data_binned.tolist()
        binned_cc_df.at[row_idx, "bin_edges"] = [bin_edges[0], bin_edges[-1]]

    binned_cc_df["sess_ns"] = binned_cc_df["sess_ns"].astype(int)

    return binned_cc_df

