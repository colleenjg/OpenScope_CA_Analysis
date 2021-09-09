"""
seq_analys.py

This script contains functions for sequence analysis.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging
from sess_util import sess_gen_util, sess_ntuple_util

import numpy as np
import pandas as pd
import scipy.stats as scist

from util import logger_util, gen_util, math_util
from analysis import misc_analys

logger = logging.getLogger(__name__)


#############################################
def get_data(stim, refs, analyspar, pre=0, post=1, ch_fl=None, integ=False,
             ref_type="segs"):
    """
    get_data(stim, refs, analyspar)

    Returns data for a specific stimulus around sequence references provided.

    Required args:
        - stim (Stim):
            Stimulus object
        - refs (1D array):
            Sequences references (either segments or frames, specified by 
            ref_type)
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters

    Optional args:
        - pre (num): 
            number of seconds to keep before refs
            default: 0
        - post (num): 
            number of seconds to keep after refs
            default: 1
        - ch_fl (list):
            flanks to check for discarding refs with insufficient flanks
            default: None
        - integ (bool):
            if True, sequence data is integrated
            default: False
        - ref_type (str):
            type of references provided ("segs", "twop_fr", "stim_fr")
            default: "segs"

    Returns:
        - data_arr (2-3D array):
            sequence data array
            dims: ROIs x seq (x frames)
        - time_values (1D array):
            values for each frame, in seconds
    """

    if ref_type == "segs":
        fr_ns = stim.get_twop_fr_by_seg(
            refs, first=True, last=True, ch_fl=ch_fl)["first_twop_fr"]

        if len(fr_ns) == 0:
            raise ValueError("No frames found given flank requirements.")
    elif ref_type == "twop_frs":
        fr_ns = stim.sess.check_flanks(refs, ch_fl, fr_type="twop")
    else:
        gen_util.accepted_values_error(
            "ref_type", ref_type, ["segs", "twop_fr"]
            )

    data_df = stim.get_roi_data(
        fr_ns, pre, post, remnans=analyspar.remnans, scale=analyspar.scale
        )
    
    time_values = data_df.index.unique("time_values").to_numpy()

    data_arr = gen_util.reshape_df_data(
        data_df["roi_traces"], squeeze_cols=True
        )

    if integ:
        nanpol = None if analyspar.remnans else "omit"
        data_arr = math_util.integ(
            data_arr, 1. / stim.sess.twop_fps, axis=-1, nanpol=nanpol)
    
    return data_arr, time_values
    

#############################################
def get_common_oris(stimpar, split="by_exp"):
    """
    get_common_oris(stimpar)

    Returns Gabor orientations for common orientations, and checks parameters. 

    Required args:
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - split (str): 
            how to split data:
            "by_exp" (all exp, all unexp)
            default: "by_exp"

    Returns:
        - gab_oris (list):
            Gabor orientations for [exp, unexp] sequences, respectively
    """

    if split != "by_exp":
        raise NotImplementedError("'common_oris' only implemented "
            "with 'split' set to 'by_exp'.")
    if stimpar.stimtype != "gabors":
        raise ValueError("Exp/unexp index analysis with common "
            "orientations can only be run on Gabors.")

    if (isinstance(stimpar.gab_ori, list) and (len(stimpar.gab_ori) == 2) 
        and isinstance(stimpar.gab_ori[0], list) 
        and isinstance(stimpar.gab_ori[1], list)):
        gab_oris = stimpar.gab_ori
        set_common_oris = False

    else:
        gab_oris = sess_gen_util.get_params(gab_ori=stimpar.gab_ori)
        gab_oris = sess_gen_util.gab_oris_common_U("D", "all")

    return gab_oris


#############################################
def get_by_exp_data(sess, analyspar, stimpar, integ=False, common_oris=False):
    """
    get_by_exp_data(sess, analyspar, stimpar)

    Returns data split into expected and unexpected sequences.

    Required args:
        - sess (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - integ (bool)
            if True, sequence data is integrated
            default: False
        - common_oris (bool): 
            if True, only Gabor stimulus orientations common to D and U frames 
            are included ("by_exp" split only)
            default: False

    Returns:
        - data_arr (nested list):
            sequence data array
            dims: split x ROIs x seq (x frames)
        - time_values (1D array):
            values for each frame, in seconds
    """

    stim = sess.get_stim(stimpar.stimtype)

    gab_oris = [stimpar.gab_ori] * 2
    if common_oris:
        gab_oris = get_common_oris(stimpar, split="by_exp")

    by_exp_data = []
    for e, exp in enumerate([0, 1]):
        
        segs = stim.get_segs_by_criteria(
            gabfr=stimpar.gabfr, gabk=stimpar.gabk, gab_ori=gab_oris[e],
            bri_dir=stimpar.bri_dir, bri_size=stimpar.bri_size, surp=exp, 
            remconsec=False, by="seg")

        data, time_values = get_data(
            stim, segs, analyspar, pre=stimpar.pre, post=stimpar.post, 
            integ=integ
            )
        by_exp_data.append(data.tolist())

    return by_exp_data, time_values


#############################################
def get_locked_data(sess, analyspar, stimpar, split="unexp_lock", integ=False):
    """
    get_locked_data(sess, analyspar, stimpar)

    Returns data locked to unexpected sequence onset or expected sequence onset.

    Required args:
        - sess (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - split (str): 
            how to split data:
            "unexp_lock" (unexp, preceeding exp), 
            "exp_lock" (exp, preceeding unexp),
            default: "unexp_lock"
        - integ (bool)
            if True, sequence data is integrated
            default: False

    Returns:
        - data_arr (nested list):
            sequence data array
            dims: split x ROIs x seq (x frames)
        - time_values (1D array):
            values for each frame, in seconds (for 0 to stimpar.post)
    """

    if split not in ["unexp_lock", "exp_lock"]:
        gen_util.accepted_values_error(
            "split", split, ["unexp_lock", "exp_lock"])

    stim = sess.get_stim(stimpar.stimtype)

    exp = 1 if split == "unexp_lock" else 0

    locked_data = []
    for i in range(2):

        segs = stim.get_segs_by_criteria(
            gabfr=stimpar.gabfr, gabk=stimpar.gabk, gab_ori=stimpar.gab_ori,
            bri_dir=stimpar.bri_dir, bri_size=stimpar.bri_size, surp=exp, 
            remconsec=True, by="seg")

        if i == 0:
            pre, post = [stimpar.pre, 0]
        else:
            pre, post = [0, stimpar.post]

        data, time_values = get_data(
            stim, segs, analyspar, pre=pre, post=post, 
            ch_fl=[stimpar.pre, stimpar.post], integ=integ
            )
        
        locked_data.append(data.tolist())

    return locked_data, time_values


#############################################
def get_stim_on_off_data(sess, analyspar, stimpar, split="stim_onset", 
                         integ=False):
    """
    get_stim_on_off_data(sess, analyspar, stimpar)

    Returns data locked to stimulus onset or stimulus offset.

    Required args:
        - sess (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - split (str): 
            how to split data:
            "stim_onset" (grayscr, stim on), 
            "stim_offset" (stim off, grayscr),
            default: "stim_onset"
        - integ (bool)
            if True, sequence data is integrated
            default: False

    Returns:
        - data_arr (nested list):
            sequence data array
            dims: split x ROIs x seq (x frames)
        - time_values (1D array):
            values for each frame, in seconds (for 0 to stimpar.post)
    """

    if split not in ["stim_onset", "stim_offset"]:
        gen_util.accepted_values_error(
            "split", split, ["stim_onset", "stim_offset"]
            )

    if stimpar.stimtype != "both":
        raise ValueError("stimpar.stimtype must be 'both', if analysing "
            "stimulus on/off data.")

    stim = None
    for stimtype in ["gabors", "bricks"]: # use any stimulus to retrieve data
        if hasattr(sess, stimtype):
            stim = sess.get_stim(stimtype)
            break
    
    if split == "stim_onset":
        stim_fr = sess.grayscr.get_last_nongab_stim_fr()["last_stim_fr"][:-1] + 1
    elif split == "stim_offset":
        stim_fr = sess.grayscr.get_first_nongab_stim_fr()["first_stim_fr"][1:]

    twop_fr = sess.stim2twopfr[stim_fr]

    stim_on_off_data = []
    for i in range(2):

        if i == 0:
            pre, post = [stimpar.pre, 0]
        else:
            pre, post = [0, stimpar.post]

        # ROI x seq (x frames)
        data, time_values = get_data(
            stim, twop_fr, analyspar, pre=pre, post=post, 
            ch_fl=[stimpar.pre, stimpar.post], integ=integ, ref_type="twop_frs"
            )
        
        # very few stim onset/offset sequences, so best to retain all
        axis = -1 if integ else -2
        if data.shape[axis] != len(twop_fr):
            raise ValueError("Not all sequences could be retained for "
                f"{split} with stimpar.pre={stimpar.pre} and "
                f"stimpar.post={stimpar.post}.")


        stim_on_off_data.append(data.tolist())
        
    return stim_on_off_data, time_values


#############################################
def split_data_by_sess(sess, analyspar, stimpar, split="by_exp", integ=False, 
                       baseline=0.0, common_oris=False):
    """
    split_data_by_sess(sess, analyspar, stimpar)

    Returns data for the session, split as requested.

    Required args:
        - sess (Session): 
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - split (str): 
            how to split data:
            "by_exp" (all exp, all unexp), 
            "unexp_lock" (unexp, preceeding exp), 
            "exp_lock" (exp, preceeding unexp),
            "stim_onset" (grayscr, stim on), 
            "stim_offset" (stim off, grayscr)
            default: "by_exp"
        - integ (bool)
            if True, sequence data is integrated
            default: False
        - baseline (bool or num): 
            if not False, number of second to use for baseline 
            (not implemented)
            default: 0.0
        - common_oris (bool): 
            if True, only Gabor stimulus orientations common to D and U frames 
            are included ("by_exp" split only)
            default: False

    Returns:
        - data (nested list): 
            list of data arrays
            dims: split x ROIs x seq (x frames)
        - time_values (1D array):
            values for each frame, in seconds 
            (only 0 to stimpar.post, unless split is "by_exp")
    """
    

    locks = ["exp_lock", "unexp_lock"]
    stim_on_offs = ["stim_onset", "stim_offset"] 

    if baseline != 0:
        raise NotImplementedError("Baselining not implemented here.")

    if common_oris:
        get_common_oris(stimpar, split=split) # checks if permitted

    arg_dict = {
        "sess"      : sess,
        "analyspar" : analyspar,
        "stimpar"   : stimpar,
        "integ"     : integ,
    }

    if split == "by_exp":
        data, time_values = get_by_exp_data(common_oris=common_oris, **arg_dict)
    elif split in locks:
        data, time_values = get_locked_data(split=split, **arg_dict)
    elif split in stim_on_offs:
        data, time_values = get_stim_on_off_data(split=split, **arg_dict)
    else:
        gen_util.accepted_values_error(
            "split", split, ["by_exp"] + locks + stim_on_offs
            )

    return data, time_values


#############################################
def get_sess_roi_trace_stats(sess, analyspar, stimpar, basepar, 
                             split="by_exp"):
    """
    get_sess_roi_trace_stats(sess, analyspar, stimpar, basepar)

    Returns ROI trace statistics for a specific session, split as requested.

    Required args:
        - sess (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters

    Optional args:
        - split (str): 
            how to split data:
            "by_exp" (all exp, all unexp), 
            "unexp_lock" (unexp, preceeding exp), 
            "exp_lock" (exp, preceeding unexp),
            "stim_onset" (grayscr, stim on), 
            "stim_offset" (stim off, grayscr)
            default: "by_exp"

    Returns:
        - stats (4D array): 
            ROI trace statistics for a sessions
            dims: exp, unexp x ROIs x frames x stats
        - time_values (1D array):
            values for each frame, in seconds 
            (only 0 to stimpar.post, unless split is "by_exp")
    """
    
    nanpol = None if analyspar.remnans else "omit"

    split_data, time_values = split_data_by_sess(
        sess, analyspar, stimpar, split=split, baseline=basepar.baseline
        )
    
    stats = []
    # split x ROIs x frames x stats
    for data in split_data:
        stats.append(
            np.transpose(
                math_util.get_stats(
                    data, stats=analyspar.stats, error=analyspar.error, 
                    axes=1, nanpol=nanpol
                    ), 
                [1, 2, 0])
            )
    stats = np.asarray(stats)

    return stats, time_values


#############################################
def get_sess_roi_trace_df(sessions, analyspar, stimpar, basepar, 
                          split="by_exp", parallel=False):
    """
    get_sess_roi_trace_df(sess, analyspar, stimpar, basepar)

    Returns ROI trace statistics for specific sessions, split as requested.

    Required args:
        - sessions (list): 
            session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters

    Optional args:
        - split (str): 
            how to split data:
            "by_exp" (all exp, all unexp), 
            "unexp_lock" (unexp, preceeding exp), 
            "exp_lock" (exp, preceeding unexp),
            "stim_onset" (grayscr, stim on), 
            "stim_offset" (stim off, grayscr)
            default: "by_exp"

    Returns:
        - trace_df (pd.DataFrame):
            dataframe with a row for each session, and the following 
            columns, in addition to the basic sess_df columns: 
            - roi_trace_stats (list): 
                ROI trace stats (split x ROIs x frames x stat (me, err))
            - time_values (list):
                values for each frame, in seconds
                (only 0 to stimpar.post, unless split is "by_exp")
    """

    trace_df = misc_analys.get_check_sess_df(sessions, None, analyspar)

    # retrieve ROI index information
    args_dict = {
        "analyspar": analyspar, 
        "stimpar"  : stimpar, 
        "basepar"  : basepar, 
        "split"    : split,
    }

    # sess x split x ROIs x frames
    roi_trace_stats, all_time_values = gen_util.parallel_wrap(
        get_sess_roi_trace_stats, sessions, 
        args_dict=args_dict, parallel=parallel, zip_output=True
        )

    trace_df["roi_trace_stats"] = [stats.tolist() for stats in roi_trace_stats]
    trace_df["time_values"] = [
        time_values.tolist() for time_values in all_time_values
        ]

    return trace_df


#############################################
def get_sess_grped_trace_df(sessions, analyspar, stimpar, basepar, 
                            split="by_exp", parallel=False):
    """
    get_sess_grped_trace_df(sessions, analyspar, stimpar, basepar)

    Returns ROI trace statistics for specific sessions, split as requested, 
    and grouped across mice.

    Required args:
        - sessions (list): 
            session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters

    Optional args:
        - split (str): 
            how to split data:
            "by_exp" (all exp, all unexp), 
            "unexp_lock" (unexp, preceeding exp), 
            "exp_lock" (exp, preceeding unexp),
            "stim_onset" (grayscr, stim on), 
            "stim_offset" (stim off, grayscr)
            default: "by_exp"

    Returns:
        - trace_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - trace_stats (list): 
                trace stats (split x frames x stat (me, err))
            - time_values (list):
                values for each frame, in seconds
              (only 0 to stimpar.post, unless split is "by_exp")
    """

    nanpol = None if analyspar.remnans else "omit"

    trace_df = get_sess_roi_trace_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        split=split, 
        parallel=parallel
        )
    
    columns = trace_df.columns.tolist()
    columns[columns.index("roi_trace_stats")] = "trace_stats"
    grped_trace_df = pd.DataFrame(columns=columns)

    group_columns = ["lines", "planes", "sess_ns"]
    for grp_vals, trace_grp_df in trace_df.groupby(group_columns):
        row_idx = len(grped_trace_df)
        for g, group_column in enumerate(group_columns):
            grped_trace_df.loc[row_idx, group_column] = grp_vals[g]

        for column in columns:
            if column not in group_columns + ["trace_stats", "time_values"]:
                values = trace_grp_df[column].tolist()
                grped_trace_df.at[row_idx, column] = values

        # group ROIs across mice
        n_fr = np.min(
            [len(time_values) for time_values in trace_grp_df["time_values"]]
        )
        
        if split == "by_exp":
            time_values = np.linspace(-stimpar.pre, stimpar.post, n_fr)
        else:
            time_values = np.linspace(0, stimpar.post, n_fr)

        all_roi_stats = np.concatenate(
            [np.asarray(roi_stats)[..., : n_fr, 0] 
             for roi_stats in trace_grp_df["roi_trace_stats"]], axis=1
        )

        # take stats across ROIs
        trace_stats = np.transpose(
            math_util.get_stats(
                all_roi_stats, stats=analyspar.stats, error=analyspar.error, 
                axes=1, nanpol=nanpol
            ), [1, 2, 0]
        )

        grped_trace_df.loc[row_idx, "trace_stats"] = trace_stats.tolist()
        grped_trace_df.loc[row_idx, "time_values"] = time_values.tolist()

    grped_trace_df["sess_ns"] = grped_trace_df["sess_ns"].astype(int)

    return grped_trace_df


#############################################
def get_sess_roi_split_stats(sess, analyspar, stimpar, basepar, split="by_exp"):
    """
    get_sess_roi_split_stats(sess, analyspar, stimpar, basepar)

    Returns ROI split stats for a specific session.

    Required args:
        - sess (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters

    Optional args:
        - split (str): 
            how to split data:
            "by_exp" (all exp, all unexp), 
            "unexp_lock" (unexp, preceeding exp), 
            "exp_lock" (exp, preceeding unexp),
            "stim_onset" (grayscr, stim on), 
            "stim_offset" (stim off, grayscr)
            default: "by_exp"

    Returns:
        - split_stats (2D array): 
            integrated ROI traces by split
            dims: split x ROIs
    """
    
    nanpol = None if analyspar.remnans else "omit"

    split_data, _ = split_data_by_sess(
        sess, analyspar, stimpar, split=split, baseline=basepar.baseline, 
        integ=True
        )
    
    split_stats = []
    # split x ROI
    for data in split_data:
        split_stats.append(
            math_util.get_stats(
                data, stats=analyspar.stats, error=analyspar.error, axes=1, 
                nanpol=nanpol
                )[0]
            )

    split_stats = np.asarray(split_stats)

    return split_stats


#############################################
def get_sess_grped_diffs_df(sessions, analyspar, stimpar, basepar, permpar,
                            split="by_exp", parallel=False):
    """
    get_sess_grped_diffs_df(sessions, analyspar, stimpar, basepar)

    Returns split difference statistics for specific sessions, grouped across 
    mice.

    Required args:
        - sessions (list): 
            session objects
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

    Returns:
        - diffs_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - diff_stats (list): split difference stats (me, err)
            - null_CIs (list): adjusted null CI for split differences
            - raw_p_vals (float): unadjusted p-value for differences within 
                sessions
            - p_vals (float): p-value for differences within sessions, 
                adjusted for multiple comparisons and tails
            for session comparisons, e.g. 1v2:
            - raw_p_vals_{}v{} (float): unadjusted p-value for differences
                between sessions 
            - p_vals_{}v{} (float): p-value for differences between sessions, 
                adjusted for multiple comparisons and tails
    """

    nanpol = None if analyspar.remnans else "omit"

    sess_diffs_df = misc_analys.get_check_sess_df(sessions, None, analyspar)
    initial_columns = sess_diffs_df.columns

    # retrieve ROI index information
    args_dict = {
        "analyspar": analyspar, 
        "stimpar"  : stimpar, 
        "basepar"  : basepar, 
        "split"    : split,
    }

    # sess x split x ROI
    data_diffs = gen_util.parallel_wrap(
        get_sess_roi_split_stats, sessions, 
        args_dict=args_dict, parallel=parallel, zip_output=False
        )
    sess_diffs_df["roi_split_stats"] = data_diffs

    columns = sess_diffs_df.columns.tolist() + ["null_CIs"]
    columns[columns.index("roi_split_stats")] = "diff_stats"
    diffs_df = pd.DataFrame(columns=columns)

    group_columns = ["lines", "planes", "sess_ns"]
    for lp_grp_vals, lp_grp_df in sess_diffs_df.groupby(["lines", "planes"]):
        sess_ns = sorted(lp_grp_df["sess_ns"].unique())
        sess_diffs = []
        row_indices = []
        for sess_n in sess_ns:
            row_idx = len(diffs_df)
            row_indices.append(row_idx)
            sess_grp_df = lp_grp_df.loc[lp_grp_df["sess_ns"] == sess_n]

            grp_vals = list(lp_grp_vals) + [sess_n]
            for g, group_column in enumerate(group_columns):
                diffs_df.loc[row_idx, group_column] = grp_vals[g]

            for column in initial_columns:
                if column not in group_columns:
                    values = sess_grp_df[column].tolist()
                    diffs_df.at[row_idx, column] = values

            # group ROI split stats across mice: split x ROIs
            split_stats = np.concatenate(
                sess_grp_df["roi_split_stats"].to_numpy(), axis=-1
                )

            # take diff and stats across ROIs
            diffs = split_stats[1] - split_stats[0]
            diff_stats = math_util.get_stats(
                diffs, stats=analyspar.stats, error=analyspar.error,
                nanpol=nanpol
            )
            diffs_df.at[row_idx, "diff_stats"] = diff_stats.tolist()

            p_val, null_CI = math_util.get_diff_p_val(
                split_stats, n_perms=permpar.n_perms, stats=analyspar.stats, 
                op="diff", return_CIs=True, p_thresh=permpar.p_val, 
                tails=permpar.tails, multcomp=permpar.multcomp)

            diffs_df.loc[row_idx, "p_vals"] = p_val
            diffs_df.at[row_idx, "null_CIs"] = null_CI

            sess_diffs.append(diffs)

        # calculate statistics across sessions (0-1, 0-2, 1-2...)
        p_vals = math_util.comp_vals_acr_groups(
            sess_diffs, n_perms=permpar.n_perms, stats=analyspar.stats
            )
        p = 0
        for i, sess_n in enumerate(sess_ns):
            for j, sess_n2 in enumerate(sess_ns[i + 1:]):
                key = f"p_vals_{int(sess_n)}v{int(sess_n2)}"
                diffs_df.loc[row_indices[i], key] = p_vals[p]
                diffs_df.loc[row_indices[j + 1], key] = p_vals[p]
                p += 1

    # adjust p-values
    diffs_df = misc_analys.add_adj_p_vals(diffs_df, permpar)

    diffs_df["sess_ns"] = diffs_df["sess_ns"].astype(int)

    return diffs_df



#############################################
def get_sess_ex_traces(sess, analyspar, stimpar, basepar):
    """
    get_sess_ex_traces(sess, analyspar, stimpar, basepar)

    Returns example traces selected for the session, based on SNR and Gabor 
    response pattern criteria. 

    Criteria:
    - Above median SNR
    - Sequence response cross-correlation above 75th percentile.
    - Mean sequence standard deviation above 75th percentile.
    - Mean sequence skew above 75th percentile.

    Required args:
        - sess (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters

    Returns:
        - selected_roi_data (dict):
            ["time_values"] (1D array): values for each frame, in seconds
                (only 0 to stimpar.post, unless split is "by_exp")
            ["roi_ns"] (1D array): selected ROI numbers
            ["roi_traces"] (3D array): selected ROI sequence traces,
                dims: ROIs x seq x frames
            ["roi_trace_stat"] (2D array): selected ROI trace mean or median, 
                dims: ROIs x frames
    """

    nanpol = None if analyspar.remnans else "omit"

    if stimpar.stimtype != "gabors":
        raise ValueError(
            "ROI selection criteria designed for Gabors, as they produce "
            "cyclical responses."
            )

    snr_analyspar = sess_ntuple_util.get_modif_ntuple(analyspar, "scale", False)
    snrs = misc_analys.get_snr(sess, snr_analyspar, "snrs")
    snr_median = np.median(snrs)

    traces, time_values = split_data_by_sess(
        sess,
        analyspar=analyspar,
        stimpar=stimpar,
        split="by_exp",
        baseline=basepar.baseline,
        )

    traces_exp = np.asarray(traces[0])
    traces_exp_stat = math_util.mean_med(
        traces_exp, stats=analyspar.stats, axis=1, nanpol=nanpol
        )

    # select for SNR threshold
    snr_thr_rois = np.where(snrs > snr_median)[0]
    
    # get upper diagonal indices for cross-correlations
    triu_idx = np.triu_indices(traces_exp[snr_thr_rois].shape[1])
    cc_medians = [
        np.median(np.corrcoef(roi_trace)[triu_idx]) 
        for roi_trace in traces_exp[snr_thr_rois]
        ]

    trace_stds = np.std(traces_exp_stat[snr_thr_rois], axis=1)
    trace_skews = scist.skew(traces_exp_stat[snr_thr_rois], axis=1)
    
    std_thr = np.percentile(trace_stds, 75)
    skew_thr = np.percentile(trace_skews, 75)
    cc_thr = np.percentile(cc_medians, 75)
    
    selected_idx = np.where(
        ((trace_stds > std_thr) * 
         (cc_medians > cc_thr) * 
         (trace_skews > skew_thr))
         )[0]
    
    roi_ns = snr_thr_rois[selected_idx]

    selected_roi_data = {
        "time_values"   : time_values,
        "roi_ns"        : roi_ns,
        "roi_traces"    : traces_exp[roi_ns], 
        "roi_trace_stat": traces_exp_stat[roi_ns]
    }

    return selected_roi_data


#############################################
def get_ex_traces_df(sessions, analyspar, stimpar, basepar, parallel=False, 
                     seed=None, n_ex=6):
    """
    get_ex_traces_df(sessions, analyspar, stimpar, basepar)

    Returns example ROI traces dataframe.

    Required args:
        - sessions (list):
            Session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters
    
    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - selected_roi_data (pd.DataFrame):
            dataframe with a row for each ROI, and the following columns, 
            in addition to the basic sess_df columns: 
            - time_values (list): values for each frame, in seconds
                (only 0 to stimpar.post, unless split is "by_exp")
            - roi_ns (list): selected ROI number
            - traces (list): selected ROI sequence traces, dims: seq x frames
            - trace_stat (list): selected ROI trace mean or median
    """

    retained_traces_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar
        )
    initial_columns = retained_traces_df.columns

    retained_roi_data = gen_util.parallel_wrap(
        get_sess_ex_traces, sessions, [analyspar, stimpar, basepar], 
        parallel=parallel
        )

    seed = gen_util.seed_all(seed, "cpu", log_seed=False)

    # add data to dataframe
    new_columns = retained_roi_data[0].keys()
    for column in new_columns:
        retained_traces_df[column] = np.nan
        retained_traces_df[column] = retained_traces_df[column].astype(object)

    for i, sess in enumerate(sessions):
        row_idx = retained_traces_df.loc[
            retained_traces_df["sessids"] == sess.sessid
        ].index

        if len(row_idx) != 1:
            raise ValueError(
                "Expected exactly one dataframe row to match session ID."
                )
        row_idx = row_idx[0]

        for column, value in retained_roi_data[i].items():
            retained_traces_df.at[row_idx, column] = value

    # select a few ROIs per line/plane/session
    columns = retained_traces_df.columns.tolist()
    columns = [column.replace("roi_trace", "trace") for column in columns]
    selected_traces_df = pd.DataFrame(columns=columns)

    group_columns = ["lines", "planes", "sess_ns"]
    for _, trace_grp_df in retained_traces_df.groupby(group_columns):
        grp_indices = trace_grp_df.index
        n_per = np.asarray([len(roi_ns) for roi_ns in trace_grp_df["roi_ns"]])
        roi_ns = np.concatenate(trace_grp_df["roi_ns"].tolist())
        concat_idxs = np.sort(
            np.random.choice(len(roi_ns), n_ex, replace=False)
            )

        for concat_idx in concat_idxs:
            row_idx = len(selected_traces_df)
            sess_idx = np.where(concat_idx < np.cumsum(n_per))[0][0]
            source_row = trace_grp_df.loc[grp_indices[sess_idx]]
            for column in initial_columns:
                selected_traces_df.at[row_idx, column] = source_row[column]

            selected_traces_df.at[row_idx, "time_values"] = \
                source_row["time_values"].tolist()
            
            roi_idx = concat_idx - n_per[: sess_idx].sum()
            for col in ["roi_ns", "traces", "trace_stat"]: 
                source_col = col.replace("trace", "roi_trace")
                selected_traces_df.at[row_idx, col] = \
                    source_row[source_col][roi_idx].tolist()

    for column in [
        "mouse_ns", "mouseids", "sess_ns", "sessids", "nrois", "roi_ns"
        ]:
        selected_traces_df[column] = selected_traces_df[column].astype(int)

    return selected_traces_df


