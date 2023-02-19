"""
seq_analys.py

This script contains functions for sequence analysis.

Authors: Colleen Gillon

Date: February 2023

Note: this code was aggregated from https://github.com/colleenjg/OpenScope_CA_Analysis.
"""

import numpy as np
import pandas as pd
import scipy.stats as scist

from util import gen_util, logger_util, math_util, sess_util
from analysis import basic_analys, misc_analys


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

    trace_df = misc_analys.get_check_sess_df(sessions, None)

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
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

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

    nanpol = None if analyspar.rem_bad else "omit"

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
        trace_grp_df = trace_grp_df.sort_values("mouse_ns")
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
                axis=1, nanpol=nanpol
            ), [1, 2, 0]
        )

        grped_trace_df.loc[row_idx, "trace_stats"] = trace_stats.tolist()
        grped_trace_df.loc[row_idx, "time_values"] = time_values.tolist()

    grped_trace_df["sess_ns"] = grped_trace_df["sess_ns"].astype(int)

    return grped_trace_df


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

    analyspar_dict = analyspar._asdict()
    analyspar_dict["scale"] = False
    snr_analyspar = sess_util.init_analyspar(**analyspar_dict)

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

    retained_traces_df = misc_analys.get_check_sess_df(sessions, None)
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

    # set random state
    if randst in [None, -1, np.random]:
        randst = np.random.RandomState(None)
    else:
        if isinstance(randst, np.random.RandomState):
            randst = randst
        else:
            randst = np.random.RandomState(randst)
    
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


