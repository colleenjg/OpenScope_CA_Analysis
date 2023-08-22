"""
seq_analys.py

This script contains functions for sequence analysis.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import numpy as np
import pandas as pd
import scipy.stats as scist

from util import gen_util, logger_util, math_util, rand_util
from sess_util import sess_ntuple_util
from analysis import basic_analys, misc_analys
from plot_fcts import plot_helper_fcts


TAB = "    "


logger = logger_util.get_module_logger(name=__name__)


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
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

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
        basic_analys.get_sess_roi_trace_stats, sessions, 
        args_dict=args_dict, parallel=parallel, zip_output=True
        )

    misc_analys.get_check_sess_df(sessions, trace_df)
    trace_df["roi_trace_stats"] = [stats.tolist() for stats in roi_trace_stats]
    trace_df["time_values"] = [
        time_values.tolist() for time_values in all_time_values
        ]

    return trace_df


#############################################
def get_sess_thirds_roi_trace_df(sessions, analyspar, stimpar, basepar, 
                                 split="by_exp", parallel=False):
    """
    get_sess_roi_trace_df(sess, analyspar, stimpar, basepar)

    Returns ROI trace statistics for specific sessions, split as requested, for 
    the first and last third.

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
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - thirds_trace_df (pd.DataFrame):
            dataframe with a row for each session, and the following 
            columns, in addition to the basic sess_df columns: 
            - roi_trace_third_{third}_stats (list): 
                ROI trace stats for each third 
                (split x ROIs x frames x stat (me, err))
            - time_values (list):
                values for each frame, in seconds
                (only 0 to stimpar.post, unless split is "by_exp")
    """

    thirds_trace_df = misc_analys.get_check_sess_df(sessions, None, analyspar)

    # retrieve ROI index information
    args_dict = {
        "analyspar": analyspar, 
        "stimpar"  : stimpar, 
        "basepar"  : basepar, 
        "split"    : split,
    }

    # sess x split x ROIs x frames
    for t, third in enumerate([0, 2]):
        args_dict["third"] = third
        roi_trace_stats, all_time_values = gen_util.parallel_wrap(
            basic_analys.get_sess_roi_trace_stats, sessions, 
            args_dict=args_dict, parallel=parallel, zip_output=True
            )

        misc_analys.get_check_sess_df(sessions, thirds_trace_df)

        thirds_trace_df[f"third_{third}_roi_trace_stats"] = [
            stats.tolist() for stats in roi_trace_stats
            ]

        if t == 0:
            thirds_trace_df["time_values"] = [
                time_values.tolist() for time_values in all_time_values
                ]

    return thirds_trace_df

#############################################
def get_sess_grped_trace_df(sessions, analyspar, stimpar, basepar, 
                            split="by_exp", thirds=False, parallel=False):
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
        - thirds (bool):
            if True, data is collected for the first and last thirds.
            Not implemented for stim onset or offset.
            default: False
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - trace_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            if thirds:
                - trace_third_0_stats (list): 
                    ROI trace stats for each third 
                    (split x frames x stat (me, err))
                - trace_third_2_stats (list): 
                    ROI trace stats for each third 
                    (split x frames x stat (me, err))
            else:
                - trace_stats (list): 
                    trace stats (split x frames x stat (me, err))
            - time_values (list):
                values for each frame, in seconds
              (only 0 to stimpar.post, unless split is "by_exp")
    """

    nanpol = None if analyspar.rem_bad else "omit"

    if thirds:
        get_sess_roi_trace_fct = get_sess_thirds_roi_trace_df
    else:
        get_sess_roi_trace_fct = get_sess_roi_trace_df 

    trace_df = get_sess_roi_trace_fct(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        split=split, 
        parallel=parallel
        )
    
    columns = trace_df.columns.tolist()
    for c, column in enumerate(columns):
        if "roi_trace" in column:
            columns[c] = column.replace("roi_trace_stats", "trace_stats")
    trace_stat_cols = [col for col in columns if "trace_stats" in col]
    grped_trace_df = pd.DataFrame(columns=columns)

    group_columns = ["lines", "planes", "sess_ns"]
    for grp_vals, trace_grp_df in trace_df.groupby(group_columns):
        trace_grp_df = trace_grp_df.sort_values("mouse_ns")
        row_idx = len(grped_trace_df)
        for g, group_column in enumerate(group_columns):
            grped_trace_df.loc[row_idx, group_column] = grp_vals[g]

        for column in columns:
            if column not in group_columns + trace_stat_cols + ["time_values"]:
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

        for trace_stat_col in trace_stat_cols:
            src_stat_col = trace_stat_col.replace(
                "trace_stats", "roi_trace_stats"
                )
            all_roi_stats = np.concatenate(
                [np.asarray(roi_stats)[..., : n_fr, 0] 
                for roi_stats in trace_grp_df[src_stat_col]], axis=1
            )

            # take stats across ROIs
            trace_stats = np.transpose(
                math_util.get_stats(
                    all_roi_stats, stats=analyspar.stats, error=analyspar.error, 
                    axes=1, nanpol=nanpol
                ), [1, 2, 0]
            )

            grped_trace_df.loc[row_idx, trace_stat_col] = trace_stats.tolist()
        grped_trace_df.loc[row_idx, "time_values"] = time_values.tolist()

    grped_trace_df["sess_ns"] = grped_trace_df["sess_ns"].astype(int)

    return grped_trace_df


#############################################
def get_sess_roi_split_stats(sess, analyspar, stimpar, basepar, split="by_exp", 
                             third=None, return_data=False):
    """
    get_sess_roi_split_stats(sess, analyspar, stimpar, basepar)

    Returns ROI split stats for a specific session (integrated data).

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
        - third (int):
            if not None, specifies the index of the third split to return.
            Not implemented for stim onset or offset.
            default: None
        - return_data (bool):
            if True, split_data is returned in addition to split_stats

    Returns:
        - split_stats (2D array): 
            integrated ROI traces by split
            dims: split x ROIs
        if return_data:
        - split_data (nested list): 
            list of data arrays
            dims: split x ROIs x seq
    """
    
    nanpol = None if analyspar.rem_bad else "omit"

    split_data, _ = basic_analys.get_split_data_by_sess(
        sess, analyspar, stimpar, split=split, baseline=basepar.baseline, 
        integ=True, third=third,
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

    if return_data:
        return split_stats, split_data
    else:
        return split_stats


#############################################
def get_rand_split_data(split_data, analyspar, permpar, randst=None):
    """
    get_rand_split_data(split_data, analyspar, permpar)

    Returns random split differences for the true split data.

    Required args:
        - split_data (nested list): 
            list of data arrays
            dims: split x ROIs x seq
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters

    Optional args:
        - randst (int or np.random.RandomState): 
            random state or seed value to use. (-1 treated as None)
            default: None

    Returns:
        - rand_diffs (2D array): 
            random split diffs
            dims: ROIs x perms
    """

    nanpol = None if analyspar.rem_bad else "omit"

    split_data = [np.asarray(data) for data in split_data]

    # collect random sequence mean/median split differences
    div = split_data[0].shape[1]

    rand_diffs = rand_util.permute_diff_ratio(
        np.concatenate(split_data, axis=1), 
        div=div, 
        n_perms=permpar.n_perms, 
        stats=analyspar.stats, 
        nanpol=nanpol, 
        op="diff",
        randst=randst
        )            

    return rand_diffs


#############################################
def get_sess_grped_diffs_df(sessions, analyspar, stimpar, basepar, permpar,
                            split="by_exp", randst=None, parallel=False):
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
        - randst (int or np.random.RandomState): 
            random state or seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - diffs_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - diff_stats (list): split difference stats (me, err)
            - null_CIs (list): adjusted null CI for split differences 
            - raw_p_vals (float): uncorrected p-value for differences within 
                sessions
            - p_vals (float): p-value for differences within sessions, 
                corrected for multiple comparisons and tails
            for session comparisons, e.g. 1v2:
            - raw_p_vals_{}v{} (float): uncorrected p-value for differences
                between sessions 
            - p_vals_{}v{} (float): p-value for differences between sessions, 
                corrected for multiple comparisons and tails
    """


    nanpol = None if analyspar.rem_bad else "omit"

    if analyspar.tracked:
        misc_analys.check_sessions_complete(sessions, raise_err=True)

    sess_diffs_df = misc_analys.get_check_sess_df(sessions, None, analyspar)
    initial_columns = sess_diffs_df.columns.tolist()

    # retrieve ROI index information
    args_dict = {
        "analyspar"  : analyspar, 
        "stimpar"    : stimpar, 
        "basepar"    : basepar, 
        "split"      : split,
        "return_data": True,
    }
        
    # sess x split x ROI
    split_stats, split_data = gen_util.parallel_wrap(
        get_sess_roi_split_stats, sessions, 
        args_dict=args_dict, parallel=parallel, zip_output=True
        )
    
    misc_analys.get_check_sess_df(sessions, sess_diffs_df)
    sess_diffs_df["roi_split_stats"] = list(split_stats)
    sess_diffs_df["roi_split_data"] = list(split_data)

    columns = initial_columns + ["diff_stats", "null_CIs"]
    diffs_df = pd.DataFrame(columns=columns)

    group_columns = ["lines", "planes", "sess_ns"]
    aggreg_cols = [col for col in initial_columns if col not in group_columns]
    for lp_grp_vals, lp_grp_df in sess_diffs_df.groupby(["lines", "planes"]):
        lp_grp_df = lp_grp_df.sort_values(["sess_ns", "mouse_ns"])
        line, plane = lp_grp_vals
        lp_name = plot_helper_fcts.get_line_plane_name(line, plane)
        logger.info(
            f"Running permutation tests for {lp_name} sessions...", 
            extra={"spacing": TAB}
            )

        # obtain ROI random split differences per session
        # done here to avoid OOM errors
        lp_rand_diffs = gen_util.parallel_wrap(
            get_rand_split_data, lp_grp_df["roi_split_data"].tolist(), 
            args_list=[analyspar, permpar, randst], parallel=parallel, 
            zip_output=False
            )

        sess_diffs = []
        row_indices = []
        sess_ns = sorted(lp_grp_df["sess_ns"].unique())
        for sess_n in sess_ns:
            row_idx = len(diffs_df)
            row_indices.append(row_idx)
            sess_grp_df = lp_grp_df.loc[lp_grp_df["sess_ns"] == sess_n]

            grp_vals = list(lp_grp_vals) + [sess_n]
            for g, group_column in enumerate(group_columns):
                diffs_df.loc[row_idx, group_column] = grp_vals[g]

            # add aggregated values for initial columns
            diffs_df = misc_analys.aggreg_columns(
                sess_grp_df, diffs_df, aggreg_cols, row_idx=row_idx, 
                in_place=True
                )

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
            sess_diffs.append(diffs)

            # group random ROI split diffs across mice, and take stat
            rand_idxs = [
                lp_grp_df.index.tolist().index(idx) 
                for idx in sess_grp_df.index
            ]
            rand_diffs = math_util.mean_med(
                np.concatenate([lp_rand_diffs[r] for r in rand_idxs], axis=0), 
                axis=0, stats=analyspar.stats, nanpol=nanpol
                )

            # get CIs and p-values
            p_val, null_CI = rand_util.get_p_val_from_rand(
                diff_stats[0], rand_diffs, return_CIs=True, 
                p_thresh=permpar.p_val, tails=permpar.tails, 
                multcomp=permpar.multcomp, nanpol=nanpol
                )
            diffs_df.loc[row_idx, "p_vals"] = p_val
            diffs_df.at[row_idx, "null_CIs"] = null_CI

        del lp_rand_diffs # free up memory

        # calculate p-values between sessions (0-1, 0-2, 1-2...)
        p_vals = rand_util.comp_vals_acr_groups(
            sess_diffs, n_perms=permpar.n_perms, stats=analyspar.stats,
            paired=analyspar.tracked, nanpol=nanpol, randst=randst
            )
        p = 0
        for i, sess_n in enumerate(sess_ns):
            for j, sess_n2 in enumerate(sess_ns[i + 1:]):
                key = f"p_vals_{int(sess_n)}v{int(sess_n2)}"
                diffs_df.loc[row_indices[i], key] = p_vals[p]
                diffs_df.loc[row_indices[j + 1], key] = p_vals[p]
                p += 1

    # add corrected p-values
    diffs_df = misc_analys.add_corr_p_vals(diffs_df, permpar)

    diffs_df["sess_ns"] = diffs_df["sess_ns"].astype(int)

    return diffs_df


#############################################
def get_sess_grped_diffs_by_thirds_df(sessions, analyspar, stimpar, basepar, 
                                      permpar, split="by_exp", randst=None, 
                                      parallel=False):
    """
    get_sess_grped_diffs_by_thirds_df(sessions, analyspar, stimpar, basepar)

    Returns split difference statistics for specific sessions, grouped across 
    mice, split into first and second halves.

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
            default: "by_exp"
        - randst (int or np.random.RandomState): 
            random state or seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - diffs_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - third_{third}_diff_stats (list): split difference stats (me, err)
            for early vs late comparisons, e.g. 0v1:
            - raw_p_vals_{}v{} (float): uncorrected p-value for differences
                between sessions 
            - p_vals_{}v{} (float): p-value for differences between sessions, 
                corrected for multiple comparisons and tails
    """


    nanpol = None if analyspar.rem_bad else "omit"

    if analyspar.tracked:
        misc_analys.check_sessions_complete(sessions, raise_err=True)

    sess_diffs_df = misc_analys.get_check_sess_df(sessions, None, analyspar)
    initial_columns = sess_diffs_df.columns.tolist()

    # retrieve ROI index information
    args_dict = {
        "analyspar"  : analyspar, 
        "stimpar"    : stimpar, 
        "basepar"    : basepar, 
        "split"      : split,
        "return_data": True,
    }
    
    thirds = [0, 2]

    use_sessions, loop_dicts = list(), list()
    for third in thirds:
        for sess in sessions:
            loop_dicts.append({"third": third})
            use_sessions.append(sess)
        
    # sess x split x ROI
    split_stats, split_data = gen_util.parallel_wrap(
        get_sess_roi_split_stats, use_sessions, 
        args_dict=args_dict, loop_dicts=loop_dicts, parallel=parallel, 
        zip_output=True, 
        )
    
    misc_analys.get_check_sess_df(sessions, sess_diffs_df)

    diff_stat_cols = list()
    for t, third in enumerate(thirds):
        st = t * len(sessions)
        end = (t + 1) * len(sessions)
        sess_diffs_df[f"third_{third}_roi_split_stats"] = list(
            split_stats[st: end]
            )
        sess_diffs_df[f"third_{third}_roi_split_data"] = list(
            split_data[st: end]
            )
        diff_stat_cols.append(f"third_{third}_diff_stats")

    columns = initial_columns + diff_stat_cols
    diffs_by_third_df = pd.DataFrame(columns=columns)

    group_columns = ["lines", "planes", "sess_ns"]
    aggreg_cols = [col for col in initial_columns if col not in group_columns]
    for lp_grp_vals, lp_grp_df in sess_diffs_df.groupby(["lines", "planes"]):
        lp_grp_df = lp_grp_df.sort_values(["sess_ns", "mouse_ns"])
        sess_ns = sorted(lp_grp_df["sess_ns"].unique())
        for sess_n in sess_ns:
            third_diffs = []
            row_indices = []
            for third in thirds:
                row_idx = len(diffs_by_third_df)
                row_indices.append(row_idx)
                sess_grp_df = lp_grp_df.loc[lp_grp_df["sess_ns"] == sess_n]

                grp_vals = list(lp_grp_vals) + [sess_n]
                for g, group_column in enumerate(group_columns):
                    diffs_by_third_df.loc[row_idx, group_column] = grp_vals[g]

                # add aggregated values for initial columns
                diffs_by_third_df = misc_analys.aggreg_columns(
                    sess_grp_df, diffs_by_third_df, aggreg_cols, row_idx=row_idx, 
                    in_place=True
                    )

                # group ROI split stats across mice: split x ROIs
                split_stats = np.concatenate(
                    sess_grp_df[f"third_{third}_roi_split_stats"].to_numpy(), 
                    axis=-1
                    )

                # take diff and stats across ROIs
                diffs = split_stats[1] - split_stats[0]
                diff_stats = math_util.get_stats(
                    diffs, stats=analyspar.stats, error=analyspar.error,
                    nanpol=nanpol
                )
                diffs_by_third_df.at[
                    row_idx, f"third_{third}_diff_stats"
                    ] = diff_stats.tolist()
                third_diffs.append(diffs)

            # calculate p-values between thirds
            p_vals = rand_util.comp_vals_acr_groups(
                third_diffs, n_perms=permpar.n_perms, stats=analyspar.stats,
                paired=True, nanpol=nanpol, randst=randst
                )
            p = 0
            for i, third in enumerate(thirds):
                for j, third2 in enumerate(thirds[i + 1:]):
                    key = f"p_vals_{int(third)}v{int(third)}"
                    diffs_by_third_df.loc[row_indices[i], key] = p_vals[p]
                    diffs_by_third_df.loc[row_indices[j + 1], key] = p_vals[p]
                    p += 1

    # add corrected p-values
    diffs_by_third_df = misc_analys.add_corr_p_vals(diffs_by_third_df, permpar)

    diffs_by_third_df["sess_ns"] = diffs_by_third_df["sess_ns"].astype(int)

    return diffs_by_third_df

#############################################
def identify_gabor_roi_ns(traces, analyspar, rolling_win=4):
    """
    identify_gabor_roi_ns(traces, analyspar)

    Identifies ROIs that meet Gabor sequence response pattern criteria.

    Criteria:
    - Sequence response correlation above 75th percentile.
    - Mean sequence standard deviation above 75th percentile.
    - Mean sequence skew above 75th percentile.

    Required args:
        - traces (3D array):
            ROI traces, with dims: ROIs x seq x frames 
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters

    Optional args:
        - rolling_win (int):
            window to use in rolling mean over individual trial traces before 
            computing correlation between trials (None for no smoothing).
            default: 4 

    Returns:
        - selected_idx (1D array):
            indices of ROIs that meet the criteria
    """

    nanpol = None if analyspar.rem_bad else "omit"

    # calculate std and skew over trace statistics
    traces_stat = math_util.mean_med(
        traces, stats=analyspar.stats, axis=1, nanpol=nanpol
        )
    trace_stat_stds = np.std(traces_stat, axis=1)
    trace_stat_skews = scist.skew(traces_stat, axis=1)

    # smooth, if applicable, to compute correlations
    if rolling_win is not None:
        traces = math_util.rolling_mean(traces, win=rolling_win)

    triu_idx = np.triu_indices(traces.shape[1], k=1)
    corr_medians = [
        np.median(np.corrcoef(roi_trace)[triu_idx]) for roi_trace in traces
        ]

    # identify indices that meet threshold
    std_thr = np.percentile(trace_stat_stds, 75)
    skew_thr = np.percentile(trace_stat_skews, 75)
    corr_thr = np.percentile(corr_medians, 75)
    
    selected_idx = np.where(
        ((trace_stat_stds > std_thr) * 
         (trace_stat_skews > skew_thr) * 
         (corr_medians > corr_thr))
         )[0]
        
    return selected_idx


#############################################
def identify_visflow_roi_ns(traces, analyspar, stimpar, rolling_win=4):
    """
    identify_visflow_roi_ns(traces, analyspar, stimpar)

    Identifies ROIs that meet visual flow response pattern criteria.

    Criteria:
    - Unexpected sequence response correlation above 75th percentile.
    - Mean difference in expected versus unexpected responses above 75th 
    percentile.

    Required args:
        - traces (3D array):
            ROI traces, with dims: ROIs x seq x frames 
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - rolling_win (int):
            window to use in rolling mean over individual trial traces before 
            computing correlation between trials (None for no smoothing).
            default: 4 

    Returns:
        - selected_idx (1D array):
            indices of ROIs that meet the criteria
    """

    nanpol = None if analyspar.rem_bad else "omit"

    # identify pre/post    
    num_frames_pre = int(
        np.ceil(traces.shape[2] * stimpar.pre / (stimpar.pre + stimpar.post))
        )

    # calculate statistics for pre/post
    traces_stat_exp = math_util.mean_med(
        traces[..., : num_frames_pre], 
        stats=analyspar.stats, axis=2, nanpol=nanpol
        )
    traces_stat_unexp = math_util.mean_med(
        traces[..., num_frames_pre :], 
        stats=analyspar.stats, axis=2, nanpol=nanpol
        )
    trace_stat_diffs = math_util.mean_med(
        traces_stat_unexp - traces_stat_exp, 
        stats=analyspar.stats, axis=1, nanpol=nanpol
        )

    # smooth, if applicable, to compute correlations
    if rolling_win is not None:
        traces = math_util.rolling_mean(traces, win=rolling_win)

    triu_idx = np.triu_indices(traces.shape[1], k=1)
    corr_medians = [
        np.median(np.corrcoef(roi_trace[:, num_frames_pre :])[triu_idx]) 
        for roi_trace in traces
        ]
    
    # identify indices that meet threshold
    diffs_thr = np.percentile(trace_stat_diffs, 75)
    corr_thr = np.percentile(corr_medians, 75)
    
    selected_idx = np.where(
        ((trace_stat_diffs > diffs_thr) * 
         (corr_medians > corr_thr))
         )[0]
        
    return selected_idx


#############################################
def get_sess_ex_traces(sess, analyspar, stimpar, basepar, rolling_win=4, 
                       unexp=0):
    """
    get_sess_ex_traces(sess, analyspar, stimpar, basepar)

    Returns example traces selected for the session, based on SNR and stimulus 
    response pattern criteria. 

    Criteria:
    - Above median SNR
    - see identify_gabor_roi_ns() or identify_visflow_roi_ns() for stimulus 
    specific criteria

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
        - rolling_win (int):
            window to use in rolling mean over individual trial traces before 
            computing correlation between trials (None for no smoothing).
            default: 4 
        - unexp (int):
            expectedness value for which to return traces (0 or 1), if 
            stimpar.stimtype is gabors.
            default: 0

    Returns:
        - selected_roi_data (dict):
            ["time_values"] (1D array): values for each frame, in seconds
                (only 0 to stimpar.post, unless split is "by_exp")
            ["roi_ns"] (1D array): selected ROI numbers
            ["roi_traces_sm"] (3D array): selected ROI sequence traces, 
                smoothed, with dims: ROIs x seq x frames
            ["roi_trace_stats"] (2D array): selected ROI trace mean or median, 
                dims: ROIs x frames
    """

    nanpol = None if analyspar.rem_bad else "omit"

    snr_analyspar = sess_ntuple_util.get_modif_ntuple(analyspar, "scale", False)

    snrs = misc_analys.get_snr(sess, snr_analyspar, "snrs")
    snr_median = np.median(snrs)

    # identify ROIs that meet the SNR threshold
    snr_thr_rois = np.where(snrs > snr_median)[0]

    if stimpar.stimtype == "gabors":
        split = "by_exp"
    else:
        split = "unexp_lock"

    # collect all data, and compute statistics
    traces, time_values = basic_analys.get_split_data_by_sess(
        sess,
        analyspar=analyspar,
        stimpar=stimpar,
        split=split,
        baseline=basepar.baseline,
        )

    if stimpar.stimtype == "gabors":
        traces = np.asarray(traces[unexp])
    else:
        traces = np.concatenate(traces, axis=2)[..., 1:] # concat along time dim
        time_values = np.concatenate([-time_values[::-1][:-1], time_values])


    # select indices and re-index into all ROIs
    if stimpar.stimtype == "gabors":
        selected_idx = identify_gabor_roi_ns(
            traces[snr_thr_rois], analyspar, rolling_win=rolling_win
            )
    else:
        selected_idx = identify_visflow_roi_ns(
            traces[snr_thr_rois], analyspar, stimpar, rolling_win=rolling_win
            )
    roi_ns = snr_thr_rois[selected_idx]

    # collect data to return
    traces_stat = math_util.mean_med(
       traces, stats=analyspar.stats, axis=1, nanpol=nanpol
    )

    # smooth individual traces, if applicable
    if rolling_win is not None:
        traces = math_util.rolling_mean(traces, win=rolling_win)


    # aggregate data to return
    selected_roi_data = {
        "time_values"    : time_values,
        "roi_ns"         : roi_ns,
        "roi_traces_sm"  : traces[roi_ns], 
        "roi_trace_stats": traces_stat[roi_ns]
    }

    return selected_roi_data


#############################################
def get_ex_traces_df(sessions, analyspar, stimpar, basepar, n_ex=6, 
                     rolling_win=4, unexp=0, randst=None, parallel=False):
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
        - n_ex (int):
            number of example traces to retain
            default: 6
        - rolling_win (int):
            window to use in rolling mean over individual trial traces
            default: 4 
        - unexp (int):
            expectedness value for which to return traces (0 or 1).
            default: 0
        - randst (int or np.random.RandomState): 
            random state or seed value to use. (-1 treated as None)
            default: None
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
            - traces_sm (list): selected ROI sequence traces, smoothed, with 
                dims: seq x frames
            - trace_stats (list): selected ROI trace mean or median
    """

    retained_traces_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar
        )
    initial_columns = retained_traces_df.columns

    logger.info(
        f"Identifying example ROIs for each session...", 
        extra={"spacing": TAB}
        )

    retained_roi_data = gen_util.parallel_wrap(
        get_sess_ex_traces, sessions, 
        [analyspar, stimpar, basepar, rolling_win, unexp], 
        parallel=parallel
        )
    
    randst = rand_util.get_np_rand_state(randst, set_none=True)
    
    # add data to dataframe
    new_columns = list(retained_roi_data[0])
    retained_traces_df = gen_util.set_object_columns(
        retained_traces_df, new_columns, in_place=True
    )

    for i, sess in enumerate(sessions):
        row_idx = retained_traces_df.loc[
            retained_traces_df["sessids"] == sess.sessid
        ].index

        if len(row_idx) != 1:
            raise RuntimeError(
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
        trace_grp_df = trace_grp_df.sort_values("mouse_ns")
        grp_indices = trace_grp_df.index
        n_per = np.asarray([len(roi_ns) for roi_ns in trace_grp_df["roi_ns"]])
        roi_ns = np.concatenate(trace_grp_df["roi_ns"].tolist())
        concat_idxs = np.sort(
            randst.choice(len(roi_ns), n_ex, replace=False)
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
            for col in ["roi_ns", "traces_sm", "trace_stats"]: 
                source_col = col.replace("trace", "roi_trace")
                selected_traces_df.at[row_idx, col] = \
                    source_row[source_col][roi_idx].tolist()

    for column in [
        "mouse_ns", "mouseids", "sess_ns", "sessids", "nrois", "roi_ns"
        ]:
        selected_traces_df[column] = selected_traces_df[column].astype(int)

    return selected_traces_df


#############################################
def get_sess_integ_resp_dict(sess, analyspar, stimpar):
    """
    get_sess_integ_resp_dict(sess, analyspar, stimpar)

    Returns dictionary with integrated ROI response stats for a session.

    Required args:
        - sess (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
    
    Returns:
        - data_dict (dict):
            data dictionary with response stats (2D array, ROI x stats) under 
            keys for expected ("exp") and unexpected ("unexp") data, separated 
            by Gabor frame (e.g., "exp_3", "unexp_G") 
            if stimpar.stimtype == "gabors".
    """

    nanpol = None if analyspar.rem_bad else "omit"

    # a few checks
    if stimpar.stimtype == "gabors":
        gabfrs = [[0, 1, 2], [3, 4]]
        if stimpar.gabfr != gabfrs:
            raise ValueError(f"Expected stimpar.gabfrs to be {gabfrs}")
        if stimpar.pre != 0 or stimpar.post != 0.3:
            raise ValueError(
                f"Expected stimpar.pre and post to be 0 and 0.3, respectively."
                )
    elif stimpar.stimtype == "visflow":
        if stimpar.pre != 0 or stimpar.post != 1:
            raise ValueError(
                f"Expected stimpar.pre and post to be 0 and 1, respectively."
                )
    else:
        gen_util.accepted_values_error(
            "stimpar.stimtype", stimpar.stimtype, ["gabors", "visflow"]
            )

    if analyspar.scale:
        raise ValueError("analyspar.scale should be set to False.")

    # collect data
    stim = sess.get_stim(stimpar.stimtype)
    data_dict = {}

    # retrieve integrated sequences for each frame, and return as dictionary
    for e, unexp in enumerate(["exp", "unexp"]):
        if stimpar.stimtype == "gabors":
            cycle_gabfr = gabfrs[e]
            if e == 0:
                cycle_gabfr = cycle_gabfr + gabfrs[1] # collect expected for all
        else:
            cycle_gabfr = [""] # dummy variable

        for g, gabfr in enumerate(cycle_gabfr):
            if stimpar.stimtype == "gabors":
                data_key = f"{unexp}_{gabfr}" 
            else:
                data_key = unexp
            
            refs = stim.get_segs_by_criteria(
                gabfr=gabfr, gabk=stimpar.gabk, gab_ori=stimpar.gab_ori,
                visflow_dir=stimpar.visflow_dir, 
                visflow_size=stimpar.visflow_size, unexp=e, remconsec=False, 
                by="seg")
            
            # ROI x seq
            data, _ = basic_analys.get_data(
                stim, refs, analyspar, pre=stimpar.pre, post=stimpar.post, 
                integ=True, ref_type="segs"
                )
            
            # take stats across sequences
            data_dict[data_key] = math_util.get_stats(
                data, stats=analyspar.stats, error=analyspar.error, 
                axes=1, nanpol=nanpol
                ).T

    return data_dict


############################################
def add_relative_resp_data(resp_data_df, analyspar, rel_sess=1, in_place=False):
    """
    add_relative_resp_data(resp_data_df, analyspar)

    Adds relative response data to input dataframe for any column with "exp" 
    in the name, optionally in place.

    Required args:
        - resp_data_df (pd.DataFrame):
            dataframe with one row per session, and response stats 
            (2D array, ROI x stats) under keys for expected ("exp") and 
            unexpected ("unexp") data, separated by Gabor frame 
            (e.g., "exp_3", "unexp_G") if applicable.
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters

    Optional args:
        - rel_sess (int):
            number of session relative to which data should be scaled, for each 
            mouse
            default: 1
        - in_place (bool):
            if True, dataframe is modified in place

    Returns:
        - resp_data_df (pd.DataFrame):
            input dataframe, with "rel_{}" columns added for each input column 
            with "exp" in its name
    """

    if not in_place:
        resp_data_df = resp_data_df.copy(deep=True)

    nanpol = None if analyspar.rem_bad else "omit"

    source_columns = [col for col in resp_data_df.columns if "exp" in col]
    rel_columns = [f"rel_{col}" for col in source_columns]
    resp_data_df = gen_util.set_object_columns(
        resp_data_df, rel_columns, in_place=True
    )

    # calculate relative value for each
    for mouse_n, resp_mouse_df in resp_data_df.groupby("mouse_ns"):
        resp_data_df = resp_data_df.sort_values("sess_ns")
        # find sess 1 and check that there is only 1
        rel_sess_idx = resp_mouse_df.loc[
            resp_mouse_df["sess_ns"] == rel_sess
            ].index
        mouse_n_idxs = resp_mouse_df.index
        if len(rel_sess_idx) != 1:
            raise RuntimeError(
                f"Expected to find session {rel_sess} data for each mouse, "
                f"but if is missing for mouse {mouse_n}.")
        
        mouse_row = resp_mouse_df.loc[rel_sess_idx[0]]
        for source_col in source_columns:
            rel_col = source_col.replace("unexp", "exp")
            rel_data = math_util.mean_med(
                mouse_row[rel_col], analyspar.stats, nanpol=nanpol
                )
            for mouse_n_idx in mouse_n_idxs:
                resp_data_df.at[mouse_n_idx, f"rel_{source_col}"] = \
                    resp_data_df.loc[mouse_n_idx, source_col] / rel_data
    
    return resp_data_df 
    

############################################
def get_resp_df(sessions, analyspar, stimpar, rel_sess=1, parallel=False):
    """
    get_resp_df(sessions, analyspar, stimpar)

    Returns relative response dataframe for requested sessions.

    Required args:
        - sessions (list): 
            session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - rel_sess (int):
            number of session relative to which data should be scaled, for each 
            mouse. If None, relative data is not added.
            default: 1
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - resp_data_df (pd.DataFrame):
            data dictionary with response stats (2D array, ROI x stats) under 
            keys for expected ("exp") and unexpected ("unexp") data, 
            separated by Gabor frame (e.g., "exp_3", "unexp_G") 
            if stimpar.stimtype == "gabors", and 
            with "rel_{}" columns added for each input column with "exp" in its 
            name if rel_sess is not None.
    """
    
    if analyspar.tracked:
        misc_analys.check_sessions_complete(sessions, raise_err=True)

    sessids = [sess.sessid for sess in sessions]
    resp_data_df = misc_analys.get_check_sess_df(sessions, analyspar=analyspar) 

    # double check that sessions are in correct order
    if resp_data_df["sessids"].tolist() != sessids:
        raise NotImplementedError(
            "Implementation error. Sessions must appear in correct order in "
            "resp_data_df."
            )

    logger.info(
        f"Loading data for each session...", 
        extra={"spacing": TAB}
        )
    data_dicts = gen_util.parallel_wrap(
        get_sess_integ_resp_dict, sessions, args_list=[analyspar, stimpar], 
        parallel=parallel
    )

    # add data to df
    misc_analys.get_check_sess_df(sessions, resp_data_df)
    for i, idx in enumerate(resp_data_df.index):
        for key, value in data_dicts[i].items():
            if i == 0:
                resp_data_df = gen_util.set_object_columns(
                    resp_data_df, [key], in_place=True
                )
            resp_data_df.at[idx, key] = value[:, 0] # retain stat only, not error
    
    # add relative data
    if rel_sess is not None:
        resp_data_df = add_relative_resp_data(
            resp_data_df, analyspar, rel_sess=rel_sess, in_place=True
            )

    return resp_data_df


############################################
def get_rel_resp_stats_df(sessions, analyspar, stimpar, permpar, rel_sess=1, 
                          randst=None, parallel=False):
    """
    get_rel_resp_stats_df(sessions, analyspar, stimpar, permpar)

    Returns relative response stats dataframe for requested sessions.

    Required args:
        - sessions (list): 
            session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters

    Optional args:
        - rel_sess (int):
            number of session relative to which data should be scaled, for each 
            mouse
            default: 1
        - randst (int or np.random.RandomState): 
            random state or seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - rel_reg or rel_exp (list): data stats for regular data (me, err)
            - rel_unexp (list): data stats for unexpected data (me, err)
            for reg/exp/unexp data types, session comparisons, e.g. 1v2:
            - {data_type}_raw_p_vals_{}v{} (float): uncorrected p-value for 
                data differences between sessions 
            - {data_type}_p_vals_{}v{} (float): p-value for data between 
                sessions, corrected for multiple comparisons and tails
    """

    nanpol = None if analyspar.rem_bad else "omit"
 
    initial_columns = misc_analys.get_sess_df_columns(sessions[0], analyspar)

    resp_data_df = get_resp_df(
        sessions, analyspar, stimpar, rel_sess=rel_sess, parallel=parallel
        )

    # prepare target dataframe
    source_cols = ["rel_exp", "rel_unexp"]
    if stimpar.stimtype == "gabors":
         # regular means only A, B, C are included
        targ_cols = ["rel_reg", "rel_unexp"]
    else:
        targ_cols = ["rel_exp", "rel_unexp"]
    rel_resp_data_df = pd.DataFrame(columns=initial_columns + targ_cols)

    group_columns = ["lines", "planes"]
    aggreg_cols = [
        col for col in initial_columns if col not in group_columns + ["sess_ns"]
        ]
    for grp_vals, resp_grp_df in resp_data_df.groupby(group_columns):
        sess_ns = sorted(resp_grp_df["sess_ns"].unique())

        # take stats across frame types
        for e, (data_col, source_col) in enumerate(zip(targ_cols, source_cols)):
            sess_data = []
            if e == 0:
                row_indices = []
            for s, sess_n in enumerate(sess_ns):
                sess_grp_df = resp_grp_df.loc[resp_grp_df["sess_ns"] == sess_n]
                sess_grp_df = sess_grp_df.sort_values("mouse_ns")
                if e == 0:
                    row_idx = len(rel_resp_data_df)
                    row_indices.append(row_idx)
                    rel_resp_data_df.loc[row_idx, "sess_ns"] = sess_n
                    for g, group_column in enumerate(group_columns):
                        rel_resp_data_df.loc[row_idx, group_column] = grp_vals[g]

                    # add aggregated values for initial columns
                    rel_resp_data_df = misc_analys.aggreg_columns(
                        sess_grp_df, rel_resp_data_df, aggreg_cols, 
                        row_idx=row_idx, in_place=True)
                else:
                    row_idx = row_indices[s]

                if stimpar.stimtype == "gabors":
                    # average across Gabor frames included in reg or unexp data
                    cols = [f"{source_col}_{fr}" for fr in stimpar.gabfr[e]]
                    data = sess_grp_df[cols].values.tolist()
                    # sess x frs x ROIs -> sess x ROIs
                    data = [
                        math_util.mean_med(
                            sub, stats=analyspar.stats, axis=0, nanpol=nanpol 
                        ) for sub in data
                    ]
                else:
                    # sess x ROIs
                    data = sess_grp_df[source_col].tolist()
                
                data = np.concatenate(data, axis=0)

                # take stats across ROIs, grouped
                rel_resp_data_df.at[row_idx, data_col] = \
                    math_util.get_stats(
                        data, 
                        stats=analyspar.stats,
                        error=analyspar.error,
                        nanpol=nanpol
                        ).tolist()

                sess_data.append(data) # for p-value calculation
            
            # calculate p-values between sessions (0-1, 0-2, 1-2...)
            p_vals = rand_util.comp_vals_acr_groups(
                sess_data, n_perms=permpar.n_perms, stats=analyspar.stats, 
                paired=analyspar.tracked, nanpol=nanpol, randst=randst
                )
            p = 0
            for i, sess_n in enumerate(sess_ns):
                for j, sess_n2 in enumerate(sess_ns[i + 1:]):
                    key = f"{data_col}_p_vals_{int(sess_n)}v{int(sess_n2)}"
                    rel_resp_data_df.loc[row_indices[i], key] = p_vals[p]
                    rel_resp_data_df.loc[row_indices[j + 1], key] = p_vals[p]
                    p += 1

    rel_resp_data_df["sess_ns"] = rel_resp_data_df["sess_ns"].astype(int)

    # corrected p-values
    rel_resp_data_df = misc_analys.add_corr_p_vals(rel_resp_data_df, permpar)

    return rel_resp_data_df

