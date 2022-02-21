"""
behav_analys.py

This script contains functions for running and pupil analysis.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import numpy as np
import pandas as pd

from util import gen_util, math_util, rand_util
from analysis import basic_analys, misc_analys


#############################################
def get_pupil_run_trace_df(sessions, analyspar, stimpar, basepar, 
                           split="by_exp", parallel=False):
    """
    get_pupil_run_trace_df(sessions, analyspar, stimpar, basepar)

    Returns pupil and running traces for specific sessions, split as 
    requested.

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
            - run_traces (list): 
                running velocity traces (split x seqs x frames)
            - run_time_values (list):
                values for each frame, in seconds
                (only 0 to stimpar.post, unless split is "by_exp")
            - pupil_traces (list): 
                pupil diameter traces (split x seqs x frames)
            - pupil_time_values (list):
                values for each frame, in seconds
                (only 0 to stimpar.post, unless split is "by_exp")    
    """

    trace_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar, roi=False
        )

    # retrieve ROI index information
    args_dict = {
        "analyspar": analyspar, 
        "stimpar"  : stimpar, 
        "baseline" : basepar.baseline, 
        "split"    : split,
    }

    misc_analys.get_check_sess_df(sessions, trace_df)
    for datatype in ["pupil", "run"]:
        args_dict["datatype"] = datatype
        # sess x split x seq x frames
        split_traces, all_time_values = gen_util.parallel_wrap(
            basic_analys.get_split_data_by_sess, sessions, 
            args_dict=args_dict, parallel=parallel, zip_output=True
            )
        
        # add columns to dataframe
        trace_df[f"{datatype}_traces"] = list(split_traces)
        trace_df[f"{datatype}_time_values"] = list(all_time_values)

    return trace_df


#############################################
def get_pupil_run_trace_stats_df(sessions, analyspar, stimpar, basepar, 
                                 split="by_exp", parallel=False):
    """
    get_pupil_run_trace_stats_df(sessions, analyspar, stimpar, basepar)

    Returns pupil and running trace statistics for specific sessions, grouped 
    across mice, split as requested.

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
            dataframe with one row per session number, and the following 
            columns, in addition to the basic sess_df columns: 
            - run_trace_stats (list): 
                running velocity trace stats (split x frames x stats (me, err))
            - run_time_values (list):
                values for each frame, in seconds
                (only 0 to stimpar.post, unless split is "by_exp")
            - pupil_trace_stats (list): 
                pupil diameter trace stats (split x frames x stats (me, err))
            - pupil_time_values (list):
                values for each frame, in seconds
                (only 0 to stimpar.post, unless split is "by_exp")    
    """

    nanpol = None if analyspar.rem_bad else "omit"

    all_trace_df = get_pupil_run_trace_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        split=split, 
        parallel=parallel
        )
    
    datatypes = ["pupil", "run"]

    columns = all_trace_df.columns.tolist()
    for datatype in datatypes:
        columns[columns.index(f"{datatype}_traces")] = f"{datatype}_trace_stats"
    trace_df = pd.DataFrame(columns=columns)

    group_columns = ["sess_ns"]
    for grp_vals, trace_grp_df in all_trace_df.groupby(group_columns):
        trace_grp_df = trace_grp_df.sort_values(["lines", "planes", "mouse_ns"])
        row_idx = len(trace_df)
        grp_vals = [grp_vals]
        for g, group_column in enumerate(group_columns):
            trace_df.loc[row_idx, group_column] = grp_vals[g]

        for column in columns:
            skip = np.max([datatype in column for datatype in datatypes])
            if column not in group_columns and not skip:
                values = trace_grp_df[column].tolist()
                trace_df.at[row_idx, column] = values

        for datatype in datatypes:
            # group sequences across mice
            n_fr = np.min(
                [len(time_values) 
                for time_values in trace_grp_df[f"{datatype}_time_values"]]
            )
            
            if split == "by_exp":
                time_values = np.linspace(-stimpar.pre, stimpar.post, n_fr)
            else:
                time_values = np.linspace(0, stimpar.post, n_fr)

            all_split_stats = []
            for s in range(2):
                split_data = np.concatenate(
                    [np.asarray(traces[s])[:, : n_fr] 
                    for traces in trace_grp_df[f"{datatype}_traces"]], axis=0
                )
                # take stats across sequences
                trace_stats = math_util.get_stats(
                    split_data, stats=analyspar.stats, error=analyspar.error, 
                    axes=0, nanpol=nanpol).T
                all_split_stats.append(trace_stats)
            all_split_stats = np.asarray(all_split_stats)

            # trace stats (split x frames x stat (me, err))
            trace_df.at[row_idx, f"{datatype}_trace_stats"] = \
                all_split_stats.tolist()
            trace_df.at[row_idx, f"{datatype}_time_values"] = \
                time_values.tolist()

    trace_df["sess_ns"] = trace_df["sess_ns"].astype(int)

    return trace_df



#############################################
def get_pupil_run_block_diffs_df(sessions, analyspar, stimpar, parallel=False):
    """
    get_pupil_run_block_diffs_df(sessions, analyspar, stimpar)

    Returns pupil and running statistic differences (unexp - exp) by block.

    Required args:
        - sessions (list): 
            session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Returns:
        - block_df (pd.DataFrame):
            dataframe with a row for each session, and the following 
            columns, in addition to the basic sess_df columns: 
            - run_block_diffs (1D array):
                split differences per block
            - run_block_stats (3D array): 
                block statistics (split x block x stats (me, err))
            - pupil_block_diffs (1D array):
                split differences per block
            - pupil_block_stats (3D array): 
                block statistics (split x block x stats (me, err))
    """

    block_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar, roi=False
        )

    # retrieve ROI index information
    args_dict = {
        "analyspar": analyspar, 
        "stimpar"  : stimpar, 
    }

    misc_analys.get_check_sess_df(sessions, block_df)
    for datatype in ["pupil", "run"]:
        args_dict["datatype"] = datatype
        # sess x split x block x stats
        block_stats = gen_util.parallel_wrap(
            basic_analys.get_block_data, sessions, args_dict=args_dict, 
            parallel=parallel
            )
        
        block_diffs = []
        for sess_block_data in block_stats:
            # take difference (unexp - exp statistic) for each block
            stat_diffs = sess_block_data[1, ..., 0] - sess_block_data[0, ..., 0]
            block_diffs.append(stat_diffs)
        
        block_df[f"{datatype}_block_stats"] = block_stats
        block_df[f"{datatype}_block_diffs"] = block_diffs


    return block_df


#############################################
def get_pupil_run_block_stats_df(sessions, analyspar, stimpar, permpar, 
                                 randst=None, parallel=False):
    """
    get_pupil_run_block_stats_df(sessions, analyspar, stimpar, permpar)

    Returns pupil and running block difference statistics for specific 
    sessions, grouped across mice.

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
        - randst (int or np.random.RandomState): 
            random state or seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - block_df (pd.DataFrame):
            dataframe with one row per session/line/plane, and the following 
            columns, in addition to the basic sess_df columns: 
            - run_block_diffs (list): 
                running velocity differences per block
            - run_raw_p_vals (float):
                uncorrected p-values for differences within sessions
            - run_p_vals (float):
                p-values for differences within sessions, 
                corrected for multiple comparisons and tails
            - pupil_block_diffs (list): 
                for pupil diameter differences per block
            - pupil_raw_p_vals (list):
                uncorrected p-value for differences within sessions
            - pupil_p_vals (list):
                p-value for differences within sessions, 
                corrected for multiple comparisons and tails
    """

    nanpol = None if analyspar.rem_bad else "omit"
    
    all_block_df = get_pupil_run_block_diffs_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        parallel=parallel
        )
    
    datatypes = ["pupil", "run"]

    columns = all_block_df.columns.tolist()
    for datatype in datatypes:
        columns.remove(f"{datatype}_block_stats")
    block_df = pd.DataFrame(columns=columns)

    group_columns = ["lines", "planes", "sess_ns"]
    for grp_vals, block_grp_df in all_block_df.groupby(group_columns):
        block_grp_df = block_grp_df.sort_values("mouse_ns")
        row_idx = len(block_df)
        for g, group_column in enumerate(group_columns):
            block_df.loc[row_idx, group_column] = grp_vals[g]

        for column in columns:
            skip = np.max([datatype in column for datatype in datatypes])
            if column not in group_columns and not skip:
                values = block_grp_df[column].tolist()
                block_df.at[row_idx, column] = values

        for datatype in datatypes:
            # group blocks across mice
            all_diffs = np.concatenate(
                block_grp_df[f"{datatype}_block_diffs"].tolist()
                )
            block_df.at[row_idx, f"{datatype}_block_diffs"] = all_diffs.tolist()

            # concatenate blocks per split, across mice: split x block
            all_split_block = np.concatenate(
                block_grp_df[f"{datatype}_block_stats"].tolist(),
                axis=1
            )[..., 0] # keep mean/median only

            block_df.loc[row_idx, f"{datatype}_p_vals"] = \
                rand_util.get_op_p_val(
                    all_split_block, 
                    n_perms=permpar.n_perms, 
                    stats=analyspar.stats, 
                    paired=True,
                    nanpol=nanpol, 
                    randst=randst
                )

    # add corrected p-values
    block_df = misc_analys.add_corr_p_vals(
        block_df, permpar, raise_multcomp=False
        )

    block_df["sess_ns"] = block_df["sess_ns"].astype(int)

    return block_df

