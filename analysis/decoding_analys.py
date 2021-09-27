"""
decoding_analys.py

This script contains functions for decoding analysis.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import itertools
import logging

import numpy as np
import pandas as pd
import scipy.stats as scist

from util import logger_util, gen_util, logreg_util, math_util
from sess_util import sess_gen_util
from analysis import misc_analys

logger = logging.getLogger(__name__)

MAX_SIMULT_RUNS = 25000
TAB = "    "


#############################################
def get_decoding_data(sess, analyspar, stimpar, comp="Dori", ctrl=False):
    """
    get_decoding_data(sess, analyspar, stimpar)

    Retrieves data for decoding.

    Required args:
        - sess (Session): 
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - comp (str):
            comparison used to define classes ("Dori" or "Eori")
            default: "Dori"
        - ctrl (bool):
            if True, the number of examples per class for the unexpected data 
            is returned (applies to "Dori" comp only).
            default: False

    Returns:
        - all_input_data (3D array):
            input data, dims: seq x frames x ROIs
        - all_target_data (1D array):
            class target for each input sequence
        - ctrl_ns (list):
            number of examples per class for the unexpected data 
            (None if it doesn't apply)
    """

    if stimpar.stimtype != "gabors":
        raise ValueError("Expected stimpar.stimtype to be 'gabors'.")

    if comp == "Dori":
        exp = 0
        ctrl_ns = []
    elif comp == "Uori":
        exp = 1
        ctrl = False
        ctrl_ns = False
    else:
        gen_util.accepted_values_error("comp", comp, ["Dori", "Uori"])

    gab_oris = sess_gen_util.get_params(gab_ori=stimpar.gab_ori)[-1]

    stim = sess.get_stim(stimpar.stimtype)

    all_input_data = []
    all_target_data = []
    for g, gab_ori in enumerate(gab_oris):
        segs = stim.get_segs_by_criteria(
            gabfr=stimpar.gabfr, gabk=stimpar.gabk, gab_ori=gab_ori, surp=exp, 
            remconsec=False, by="seg")
        fr_ns = stim.get_twop_fr_by_seg(segs, first=True)["first_twop_fr"]

        # sample as many sequences as are usable for unexpected data
        if ctrl:
            segs_ctrl = stim.get_segs_by_criteria(
                gabfr=stimpar.gabfr, gabk=stimpar.gabk, gab_ori=gab_ori, 
                surp=1, remconsec=False, by="seg")
            fr_ns_ctrl = stim.get_twop_fr_by_seg(
                segs_ctrl, first=True, ch_fl=[stimpar.pre, stimpar.post]
                )["first_twop_fr"]
            ctrl_ns.append(len(fr_ns_ctrl))

        ori_data_df = stim.get_roi_data(
            fr_ns, stimpar.pre, stimpar.post, remnans=analyspar.remnans, 
            scale=analyspar.scale
            )
        # seq x frames x ROIs
        ori_data = np.transpose(
            gen_util.reshape_df_data(ori_data_df, squeeze_cols=True),
            [1, 2, 0]
        )

        all_input_data.append(ori_data)
        all_target_data.append(np.full(len(ori_data), g))

    all_input_data = np.concatenate(all_input_data, axis=0)
    all_target_data = np.concatenate(all_target_data)

    return all_input_data, all_target_data, ctrl_ns
        

#############################################
def get_df_stats(scores_df, analyspar):
    """
    get_df_stats(scores_df, analyspar)

    Returns statistics (mean/median and error) for each data column.

    Required args:
        - scores_df (pd.DataFrame):
            dataframe where each column contains data for which statistics 
            should be measured
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
    
    Returns:
        - stats_df (pd.DataFrame):
            dataframe with only one data row containing data stats for each 
            original column under "{col}_stat" and "{col}_err"
    """

    # take statistics
    stats_df = pd.DataFrame()
    for col in scores_df.columns:

        # get stats
        stat =  math_util.mean_med(
            scores_df[col].to_numpy(), stats=analyspar.stats, 
            nanpol="omit"
            )
        err = math_util.error_stat(
            scores_df[col].to_numpy(), stats=analyspar.stats, 
            error=analyspar.error, nanpol="omit"
            )
        
        if isinstance(err, np.ndarray):
            err = err.tolist()
            stats_df[f"{col}_err"] = np.nan
            stats_df[f"{col}_err"] = stats_df[f"{col}_err"].astype(object)

        stats_df.loc[0, f"{col}_stat"] = stat
        stats_df.at[0, f"{col}_err"] = err

    return stats_df


#############################################
def add_CI_p_vals(shuffle_df, stats_data_df, permpar):
    """
    add_CI_p_vals(shuffle_df, stats_data_df, permpar)

    Returns confidence intervals from shuffled data, and p-values for real data.

    Required args:
        - shuffle_df (pd.DataFrame):
            dataframe where each row contains data for different data 
            shuffles, and each column contains data to use to construct null 
            distributions.
        - stats_data_df (pd.DataFrame):
            dataframe with only one data row containing real data stats for 
            each shuffle_df column. Columns should have the same names as 
            shuffle_df, as "{col}_stat" and "{col}_err".
        - permpar (PermPar): 
            named tuple containing permutation parameters

    Returns:
        - stats_df (pd.DataFrame):
            dataframe with only one data row containing real data stats, 
            shuffled data stats, and p-values for real data test set results.
    """

    if len(stats_data_df) != 1:
        raise ValueError("Expected stats_data_df to have length 1.")

    multcomp = 1 if not permpar.multcomp else permpar.multcomp
    p_thresh_corr = permpar.p_val / multcomp
    percs = math_util.get_percentiles(
        CI=(1 - p_thresh_corr), tails=permpar.tails
        )[0]
    percs = [percs[0], 50, percs[1]]

    stats_df = pd.DataFrame()
    for col in shuffle_df.columns:
        # add real data
        stat_key = f"{col}_stat"
        err_key = f"{col}_err"
        if (stat_key not in stats_data_df.columns or 
            err_key not in stats_data_df.columns):
            raise KeyError(
                f"{stat_key} and {err_key} not found stats_data_df."
                )
        stats_df[stat_key] = stats_data_df[stat_key]
        stats_df[err_key] = stats_data_df[err_key]

        # get and add null CI data
        shuffle_data = shuffle_df[col].to_numpy()
        shuffle_data = shuffle_data[~np.isnan(shuffle_data)] # remove NaN data
        
        math_util.check_n_rand(len(shuffle_data), p_thresh_corr)
        null_CI = [np.percentile(shuffle_data, p) for p in percs]

        null_key = f"{col}_null_CIs"
        stats_df[null_key] = np.nan
        stats_df[null_key] = stats_df[null_key].astype(object)
        stats_df.at[0, null_key] = null_CI

        # get and add p-value
        if "test" in col:
            perc = scist.percentileofscore(
                shuffle_data, stats_data_df.loc[0, stat_key], kind='mean'
                )
            if perc > 50:
                perc = 100 - perc
            
            p_val = perc / 100
            stats_df.loc[0, f"{col}_p_vals"] = p_val

    return stats_df


#############################################
def collate_results(sess_data_stats_df, shuffle_dfs, analyspar, permpar):
    """
    collate_results(sess_data_stats_df, shuffle_dfs, analyspar, permpar)

    Return results collated from real data and shuffled data dataframes, 
    with statistics, null distributions and p-values added.

    Required args:
        - sess_data_stats_df (pd.DataFrame):
            dataframe where each row contains statistics for a session, 
            and where columns include data descriptors, and logistic regression 
            scores for different data subsets 
            (e.g. "train", "val", "test").
        - shuffle_dfs (list):
            dataframes for each session, where each row contains data for 
            different data shuffles, and each column contains data to use to 
            construct null distributions.
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters
    
    Returns:
        - stats_df (pd.DataFrame):
            dataframe with real data statistics, shuffled data confidence 
            intervals and p-values for test set data.
    """

    # check shuffle_dfs
    shuffle_df_lengths = [len(shuffle_df) for shuffle_df in shuffle_dfs]
    if len(np.unique(shuffle_df_lengths)) != 1:
        raise ValueError("All shuffle_dfs must have the same length.")

    stat_cols = [
        col for col in shuffle_dfs[0].columns 
        if col.split("_")[0] in ["train", "val", "test"]
        ]

    main_cols = [
        col for col in shuffle_dfs[0].columns 
        if not (col in stat_cols + ["shuffle"])
        ]

    # take statistics across session scores
    data_stats_df = pd.DataFrame()
    for stat_col in stat_cols:
        data_stats_df[stat_col] = sess_data_stats_df[f"{stat_col}_stat"]
    data_stats_df = get_df_stats(data_stats_df, analyspar)

    # take statistics across session shuffles at the same index
    shuffle_dfs_concat = pd.concat(shuffle_dfs)
    stat_shuffle_dfs = shuffle_dfs_concat.loc[:, stat_cols]
    by_row_index = stat_shuffle_dfs.groupby(stat_shuffle_dfs.index)
    if analyspar.stats == "mean":
        shuffle_df = by_row_index.mean()
    elif analyspar.stats == "median":
        shuffle_df = by_row_index.median()
    else:
        gen_util.accepted_values_error(
            "analyspar.stats", analyspar.stats, ["mean", "median"]
            )
    temp_stats_df = add_CI_p_vals(shuffle_df, data_stats_df, permpar)
    
    # add in main data columns
    stats_df = pd.DataFrame(columns=main_cols + temp_stats_df.columns.tolist())
    sort_order = np.argsort(sess_data_stats_df["sessids"].tolist())
    for col in main_cols:
        data_df_values = sess_data_stats_df[col].unique().tolist()
        shuffle_df_values = shuffle_dfs_concat[col].unique().tolist()

        if data_df_values != shuffle_df_values:
            raise ValueError(
                "Expected data_df and shuffle_df non-statistic columns, "
                "except shuffle, to contain the same sets of values."
                )
        
        # sort by session ID
        values = sess_data_stats_df[col].tolist()
        stats_df.at[0, col] = values = [values[v] for v in sort_order]

    for col in temp_stats_df.columns:
        stats_df.at[0, col] = temp_stats_df.loc[0, col]

    return stats_df


#############################################
def run_sess_log_reg(sess, analyspar, stimpar, logregpar, n_splits=100, 
                     n_shuff_splits=300, seed=None, parallel=False):
    """
    run_sess_log_reg(sess, analyspar, stimpar, logregpar)

    Runs logistic regressions on a session (real data and shuffled), and 
    returns statistics dataframes.

    Required args:
        - sess (Session): 
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - logregpar (LogRegPar): 
            named tuple containing logistic regression parameters

    Optional args:
        - n_splits (int):
            number of data splits to run logistic regressions on
            default: 100
        - n_shuff_splits (int):
            number of shuffled data splits to run logistic regressions on
            default: 300
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - data_stats_df (pd.DataFrame):
            dataframe with only one data row containing data stats for each 
            score and data subset.
        - shuffle_df (pd.DataFrame):
            dataframe where each row contains data for different data 
            shuffles, and each column contains data for each score and data 
            subset.
    """
    
    seed = gen_util.seed_all(seed, log_seed=False, seed_now=False)

    # retrieve data
    input_data, target_data, ctrl_ns = get_decoding_data(
        sess, analyspar, stimpar, comp=logregpar.comp, ctrl=logregpar.ctrl)

    scores_df = misc_analys.get_check_sess_df([sess], None, analyspar)
    common_columns = scores_df.columns.tolist()
    logreg_columns = ["comp", "ctrl", "bal", "shuffle"]

    # do checks
    if logregpar.q1v4 or logregpar.regvsurp:
        raise NotImplementedError("q1v4 and regvsurp are not implemented.")
    if n_splits <= 0 or n_shuff_splits <= 0:
        raise ValueError("n_splits and n_shuff_splits must be greater than 0.")

    set_types = ["train", "test"]
    score_types = ["neg_log_loss", "accuracy", "balanced_accuracy"]
    set_score_types = list(itertools.product(set_types, score_types))

    extrapar = dict()
    for shuffle in [False, True]:
        n_runs = n_shuff_splits if shuffle else n_splits
        extrapar["shuffle"] = shuffle

        temp_dfs = []
        for b, n in enumerate(range(0, n_runs, MAX_SIMULT_RUNS)):
            extrapar["n_runs"] = int(np.min([MAX_SIMULT_RUNS, n_runs - n]))

            with logger_util.TempChangeLogLevel(level="warning"):
                mod_cvs, _, _ = logreg_util.run_logreg_cv_sk(
                    input_data, target_data, logregpar._asdict(), extrapar, 
                    analyspar.scale, ctrl_ns, seed=seed + b, parallel=parallel, 
                    save_models=False, catch_set_prob=False)

            temp_df = pd.DataFrame()
            for set_type, score_type in set_score_types:
                key = f"{set_type}_{score_type}"
                temp_df[key] = mod_cvs[key]
            temp_dfs.append(temp_df)
        
        # compile batch scores, and get session stats for non shuffled data
        temp_df = pd.concat(temp_dfs, ignore_index=True)
        if not shuffle:
            temp_df = get_df_stats(temp_df, analyspar)
    
        # add columns to df
        score_columns = temp_df.columns.tolist()
        for col in common_columns:
            temp_df[col] = scores_df.loc[0, col]
        for col in logreg_columns:
            if col != "shuffle":
                temp_df[col] = logregpar._asdict()[col]
            else:
                temp_df[col] = shuffle

        # re-sort columns
        temp_df = temp_df.reindex(
            common_columns + logreg_columns + score_columns, axis=1
            )
        
        if shuffle:
            shuffle_df = temp_df
        else:
            data_stats_df = temp_df
    
    return data_stats_df, shuffle_df


#############################################
def run_sess_log_regs(sessions, analyspar, stimpar, logregpar, permpar, 
                      n_splits=100, seed=None, parallel=False):
    """
    run_sess_log_regs(sessions, analyspar, stimpar, logregpar, permpar)

    Runs logistic regressions on sessions (real data and shuffled), and 
    returns statistics dataframe.

    Number of shuffles is determined by permpar.n_perms. 

    Required args:
        - sessions (list): 
            Session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - logregpar (LogRegPar): 
            named tuple containing logistic regression parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters

    Optional args:
        - n_splits (int):
            number of data splits to run logistic regressions on
            default: 100
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - score_stats_df (pd.DataFrame):
            dataframe with logistic regression score statistics, shuffled score 
            confidence intervals, and test set p-values for each 
            line/plane/session.
    """

    sessids = [sess.sessid for sess in sessions]
    sess_df = misc_analys.get_check_sess_df(sessions, None, analyspar)

    score_stats_dfs = []
    group_columns = ["lines", "planes", "sess_ns"]
    s = 0
    for _, lp_grp_df in sess_df.groupby(group_columns):
        lp_sessions = [
            sessions[sessids.index(sessid)] for sessid in lp_grp_df["sessids"]
            ]

        sess_data_stats_dfs, shuffle_dfs = [], []
        for sess in lp_sessions:
            logger.info(
                f"Running decoders for session {s + 1}/{len(sess_df)}...",
                extra={"spacing": f"\n{TAB}"}
                )
            sess_data_stats_df, shuffle_df = run_sess_log_reg(
                sess, 
                analyspar=analyspar, 
                stimpar=stimpar, 
                logregpar=logregpar, 
                n_splits=n_splits, 
                n_shuff_splits=permpar.n_perms,
                seed=seed, 
                parallel=parallel
                )

            sess_data_stats_dfs.append(sess_data_stats_df)
            shuffle_dfs.append(shuffle_df)
            s += 1

        sess_data_stats_df = pd.concat(sess_data_stats_dfs, ignore_index=True)

        # collect data
        lp_df = collate_results(
            sess_data_stats_df, shuffle_dfs, analyspar, permpar
            )
        score_stats_dfs.append(lp_df)
        
    score_stats_df = pd.concat(score_stats_dfs, ignore_index=True)
    score_stats_df = misc_analys.add_corr_p_vals(score_stats_df, permpar)

    # add splits information
    score_stats_df["n_splits_per"] = n_splits
    score_stats_df["n_shuffled_splits_per"] = permpar.n_perms

    # get unique (first) values for group_columns
    for col in group_columns + list(logregpar._asdict().keys()):
        if col not in score_stats_df.columns:
            continue
        score_stats_df[col] = score_stats_df[col].apply(lambda x: x[0])
    
    score_stats_df["sess_ns"] = score_stats_df["sess_ns"].astype(int)
    
    return score_stats_df

