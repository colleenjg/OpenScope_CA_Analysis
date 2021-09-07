"""
misc_analys.py

This script contains miscellaneous analysis functions.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

import numpy as np
import scipy.stats as scist

from util import gen_util, logger_util, math_util
from sess_util import sess_ntuple_util, sess_gen_util

logger = logging.getLogger(__name__)



#############################################
def get_sig_symbol(adj_p_val, ctrl=False, percentile=False, sensitivity=None, 
                   side=1, tails=2):
    """
    get_sig_symbol(adj_p_val)

    Return significance symbol.

    Required args:
        - adj_p_val (float): 
            adjusted p-value (e.g., corrected for multiple comparisons and 
            tails)

    Optional args:
        - ctrl (bool): 
            if True, control symbol ("+") is used instead of "*"
            default: False
        - percentile (bool):
            if True, adj_p_val is a percentile (0-100) instead of a 
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
        if adj_p_val > 50:
            adj_p_val = 100 - adj_p_val
        adj_p_val = 1 - adj_p_val / 100
    
    # check if side matches tail
    if str(tails) != "2":
        if (tails == "hi" and side == -1) or (tails == "lo" and side == 1):
            return ""

    # adjust for sensitivity
    if sensitivity is not None:
        adj_p_val = np.max([adj_p_val, sensitivity])

    sig_symbol = ""
    if adj_p_val < 0.001:
        sig_symbol = "***"
    elif adj_p_val < 0.01:
        sig_symbol = "**"
    elif adj_p_val < 0.05:
        sig_symbol = "*"
    if ctrl:
        sig_symbol = sig_symbol.replace("*", "+")

    return sig_symbol


#############################################
def get_adjusted_p_val(p_val, permpar):
    """
    get_adjusted_p_val(p_val, permpar)

    Returns p-value, Bonferroni corrected for number of tails and multiple 
    comparisons.
    
    Required args:
        - p_val (float): 
            raw p-value
        - permpar (PermPar): 
            named tuple containing permutation parameters

    Returns:
        - adj_p_val (float): 
            adjusted or corrected p-value
    """

    if isinstance(permpar, dict):
        permpar = sess_ntuple_util.init_permpar(**permpar)

    n_tails = 1 if permpar.tails in ["lo", "hi"] else int(permpar.tails)
    adj_p_val = p_val * n_tails 
    
    if permpar.multcomp:
        adj_p_val *= permpar.multcomp

    return adj_p_val


#############################################
def add_adj_p_vals(df, permpar):
    """
    add_adj_p_vals(df, permpar)

    Returns dataframe with p-values, adjusted for tails and multiple 
    comparisons, are added. Original p-values columns names have "raw" added 
    in them.
    
    Required args:
        - df (pd.DataFrame): 
            dataframe with p-value columns
        - permpar (PermPar): 
            named tuple containing permutation parameters

    Returns:
        - df (pd.DataFrame): 
            dataframe with adjusted p-value columns added
    """

    if isinstance(permpar, dict):
        permpar = sess_ntuple_util.init_permpar(**permpar)

    p_val_cols = [col for col in df.columns if "p_vals" in col]

    new_col_names = {
        col: col.replace("p_vals", "raw_p_vals") for col in p_val_cols
        }

    df = df.rename(columns=new_col_names)

    for p_val_col in p_val_cols:
        adj_p_val_col = p_val_col.replace("raw_p_vals", "p_vals")
        df[adj_p_val_col] = get_adjusted_p_val(df[p_val_col], permpar)

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
        - permpar (PermPar): 
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
            if not remnans: 
                "nanrois_{}" (depending on fluor)
    """

    sessions = gen_util.list_if_not(sessions)

    if sess_df is None:
        if analyspar is None:
            raise ValueError("If sess_df is None, must pass analyspar.")

        sess_df = sess_gen_util.get_sess_info(
            sessions, fluor=analyspar.fluor, incl_roi=roi, return_df=True, 
            remnans=analyspar.remnans
            )

    else:
        if len(sess_df) != len(sessions):
            raise ValueError(
                "'sess_df' should have as many rows as 'sessions'.")
        # check order
        sessids = np.asarray([sess.sessid for sess in sessions]).astype(int)
        sess_df_sessids = sess_df.sessids.to_numpy().astype(int)

        if np.sort(sessids) != np.sort(sess_df_sessids):
            raise ValueError("Sessions do not match ids in 'sess_df'.")

        elif sessids != sess_df_sessids:
            raise ValueError("Sessions do not appear in order in 'sess_df'.")

    return sess_df


#############################################
def set_multcomp(permpar, sess_df, pairs=True, factor=1):
    """
    set_multcomp(permpar)

    Returns permpar updated with the number of comparisons computed from the 
    sessions, if permpar.multcomp is True.

    Required args:
        - permpar (PermPar): 
            named tuple containing permutation parameters
        - sess_df (pd.DataFrame):
            dataframe containing session information, including the following 
            keys: "sess_ns", "lines", "planes"
    
    Optional args:
        - pairs (bool):
            include paired comparisons
            default: True
        - factor (int): 
            additional factor by which to multiply the number of comparisons
            default: 1

    Returns:
        - permpar (PermPar): updated permutation parameter named tuple
    """

    if isinstance(permpar, dict):
        permpar = sess_ntuple_util.init_permpar(**permpar)

    if not permpar.multcomp:
        return permpar

    n_comps = 0    
    for _, sess_df_grp in sess_df.groupby(["lines", "planes"]):
        n_sess = len(sess_df_grp)

        # sessions compared to CIs
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
def get_snr(session, analyspar, datatype="snrs"):
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

    Returns:
        - data (1D array):
            data, depending on datatype, for each ROI in a session
    """
    
    keep_rois = np.arange(session.nrois)
    if analyspar.remnans:
        nanrois = session.get_nanrois(analyspar.fluor)
        keep_rois = np.delete(keep_rois, np.asarray(nanrois)).astype(int)
    
    if analyspar.scale:
        raise ValueError("analyspar.scale must be False for SNR analysis.")

    datatypes = ["snrs", "signal_means"]
    if datatype not in datatypes:
        gen_util.accepted_values_error("datatype", datatype, datatypes)
    index = 0 if datatype == "snrs" else -1

    data = np.empty(len(keep_rois)) * np.nan
    for i, r in enumerate(keep_rois):
        data[i] = math_util.calculate_snr(
            session.get_single_roi_trace(r, fluor=analyspar.fluor), 
            return_stats=True
            )[index]

    return data

