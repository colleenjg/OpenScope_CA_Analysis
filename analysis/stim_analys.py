"""
stim_analys.py

This script contains functions for stimulus comparison analysis.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging
from sess_util import sess_ntuple_util

import numpy as np
import pandas as pd

from util import logger_util, gen_util, math_util, rand_util
from analysis import misc_analys, seq_analys, usi_analys

logger = logging.getLogger(__name__)

TAB = "    "



############################################
def collect_base_data(sessions, analyspar, stimpar, datatype="rel_unexp_resp", 
                      rel_sess=1, basepar=None, idxpar=None, abs_usi=True, 
                      parallel=False):
    """
    collect_base_data(sessions, analyspar, stimpar)

    Collects base data for which stimulus and sessions comparisons are to be 
    calculated.

    Required args:
        - sessions (list): 
            session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - datatype (str):
            type of data to retrieve ("rel_unexp_resp" or "usis")
            default: "rel_unexp_resp"
        - rel_sess (int):
            number of session relative to which data should be scaled, for each 
            mouse
            default: 1
        - basepar (BasePar): 
            named tuple containing baseline parameters 
            (needed if datatype is "usis")
            default: None
        - idxpar (IdxPar): 
            named tuple containing index parameters 
            (needed if datatype is "usis")
            default: None
        - abs_usi (bool): 
            if True, absolute USIs are returned (applies if datatype is "usis")
            default: True
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - data_df (pd.DataFrame):
            dataframe with one row per session, and the basic sess_df columns, 
            in addition to datatype column:
            - {datatype} (1D array): data per ROI
    """
    
    nanpol = None if analyspar.remnans else "omit"

    initial_columns = misc_analys.get_sess_df_columns(sessions[0], analyspar)

    if datatype == "rel_unexp_resp":
        data_df = seq_analys.get_resp_df(
            sessions, analyspar, stimpar, rel_sess=rel_sess, parallel=parallel
            )
        if stimpar.stimtype == "gabors":
            unexp_gabfrs = stimpar.gabfr[1]
            unexp_data = [data_df[f"rel_unexp_{fr}"] for fr in unexp_gabfrs]
            data_df[datatype] = [
                math_util.mean_med(
                    data, stats=analyspar.stats, axis=0, nanpol=nanpol
                    ) for data in zip(*unexp_data)
                ]
        else:
            data_df = data_df.rename(columns={"rel_unexp": datatype})
        

    elif datatype == "usis":
        if basepar is None or idxpar is None:
            raise ValueError(
                "If datatype is 'usis', must provide basepar and idxpar."
                )
        data_df = usi_analys.get_idx_only_df(
            sessions, analyspar, stimpar, basepar, idxpar, parallel=parallel
            )
        data_df = data_df.rename(columns={"roi_idxs": datatype})
        if abs_usi:
            # absolute USIs
            data_df[datatype] = data_df[datatype].map(np.absolute)        
    else:
        gen_util.accepted_values_error(
            "datatype", datatype, ["rel_unexp_resp", "usis"]
            )

    data_df = data_df[initial_columns + [datatype]]

    return data_df


############################################
def check_init_stim_data_df(data_df, sessions, stimpar, comp_sess=[1, 3], 
                            stim_data_df=None, analyspar=None):
    """
    check_init_stim_data_df(data_df, stimpar)

    Checks existing stimulus dataframe or creates one for each line/plane.

    Required args:
        - data_df (pd.DataFrame):
            dataframe with one row per session, and the basic sess_df columns
        - sessions (list): 
            session objects
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - comp_sess (int):
            sessions for which to obtain absolute fractional change 
            [x, y] => |(y - x) / x|
            default: [1, 3]
        - stim_data_df (pd.DataFrame):
            dataframe with one row per line/plane, and the basic sess_df 
            columns
            default: None
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
            default: None

    Returns:
        - stim_data_df (pd.DataFrame):
            dataframe with one row per line/plane, and the basic sess_df 
            columns, as well as stimulus columns for each comp_sess:
            - {stimpar.stimtype}_s{comp_sess[0]}: for first comp_sess data
            - {stimpar.stimtype}_s{comp_sess[1]}: for second comp_sess data
    """

    initial_columns = misc_analys.get_sess_df_columns(sessions[0], analyspar)
    
    stimtype_cols = [f"{stimpar.stimtype}_s{i}" for i in comp_sess]
    if stim_data_df is None:
        new_df = True
        if analyspar is None:
            raise ValueError(
                "If stim_data_df is None, analyspar must be provided."
                )
        columns = initial_columns + stimtype_cols
        stim_data_df = pd.DataFrame(columns=columns)
    else:
        new_df = False
        if stimpar.stimtype in stim_data_df:
            raise KeyError(
            f"{stimpar.stimtype} should not already be in stim_data_df."
            )
        stim_data_df = gen_util.set_object_columns(
            stim_data_df, stimtype_cols, in_place=True
        )
    
    group_columns = ["lines", "planes"]
    aggreg_cols = [
        col for col in initial_columns 
        if col not in group_columns + ["sess_ns"]
        ]

    # populate dataframe
    for grp_vals, grp_df in data_df.groupby(group_columns):
        grp_df = grp_df.sort_values(["sess_ns", "mouse_ns"])
        line, plane = grp_vals
        if new_df:
            row_idx = len(stim_data_df)
            for g, group_column in enumerate(group_columns):
                stim_data_df.loc[row_idx, group_column] = grp_vals[g]
        else:
            row_idxs = stim_data_df.loc[
                (stim_data_df["lines"] == line) & 
                (stim_data_df["planes"] == plane)
                ].index
            if len(row_idxs) != 1:
                raise ValueError(
                    "Expected exactly one row to match line/plane."
                    )
            row_idx = row_idxs[0]

        # add aggregated values for initial columns
        ext_stim_data_df = misc_analys.aggreg_columns(
            grp_df, stim_data_df, aggreg_cols, row_idx=row_idx, 
            in_place=new_df)
        
        # check data was added correctly
        if not new_df:
            for col in aggreg_cols:
                if (ext_stim_data_df.loc[row_idx, col] != 
                    stim_data_df.loc[row_idx, col]):
                    raise RuntimeError(
                        "If stim_data_df is not None, it must contain columns "
                        "generated from data_df. This does not appear to be "
                        f"the case, as the values in {col} do not match the "
                        "values that would be added if stim_data_df was None."
                        )

    if new_df:
        stim_data_df = ext_stim_data_df

    return stim_data_df


############################################
def get_stim_data_df(sessions, analyspar, stimpar, stim_data_df=None, 
                     comp_sess=[1, 3], datatype="rel_unexp_resp", rel_sess=1, 
                     basepar=None, idxpar=None, abs_usi=True, parallel=False):
    """
    get_stim_data_df(sessions, analyspar, stimpar)

    Returns dataframe with relative ROI data for one session relative 
    to another, for each line/plane. 

    Required args:
        - sessions (list): 
            session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - stim_data_df (pd.DataFrame):
            dataframe with one row per line/plane, and the basic sess_df 
            columns
            default: None
        - comp_sess (int):
            sessions for which to obtain absolute fractional change 
            [x, y] => |(y - x) / x|
            default: [1, 3]
        - datatype (str):
            type of data to retrieve
            default: "rel_unexp_resp"
        - rel_sess (int):
            number of session relative to which data should be scaled, for each 
            mouse
            default: 1
        - basepar (BasePar): 
            named tuple containing baseline parameters 
            (needed if datatype is "usis")
            default: None
        - idxpar (IdxPar): 
            named tuple containing index parameters 
            (needed if datatype is "usis")
            default: None
        - abs_usi (bool): 
            if True, absolute USIs are returned (applies if datatype is "usis")
            default: True
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - stim_data_df (pd.DataFrame):
            dataframe with one row per line/plane, and the basic sess_df 
            columns, as well as stimulus columns for each comp_sess:
            - {stimpar.stimtype}_s{comp_sess[0]}: 
                first comp_sess data for each ROI
            - {stimpar.stimtype}_s{comp_sess[1]}: 
                second comp_sess data for each ROI
    """

    data_df = collect_base_data(
        sessions, analyspar, stimpar, datatype=datatype, rel_sess=rel_sess, 
        basepar=basepar, idxpar=idxpar, abs_usi=abs_usi, parallel=parallel
        )

    stim_data_df = check_init_stim_data_df(
        data_df, sessions, stimpar, stim_data_df=stim_data_df, 
        analyspar=analyspar
        )

    # populate dataframe
    group_columns = ["lines", "planes"]
    for grp_vals, grp_df in data_df.groupby(group_columns):
        grp_df = grp_df.sort_values(["sess_ns", "mouse_ns"])
        line, plane = grp_vals

        row_idxs = stim_data_df.loc[
            (stim_data_df["lines"] == line) & (stim_data_df["planes"] == plane)
            ].index
        if len(row_idxs) != 1:
            raise ValueError("Expected exactly one row to match line/plane.")
        row_idx = row_idxs[0]
        
        sess_ns = sorted(grp_df["sess_ns"].unique())
        for sess_n in comp_sess:
            if int(sess_n) not in sess_ns:
                raise RuntimeError(f"Session {sess_n} missing in grp_df.")

        # obtain comparison data
        comp_data = [[], []]
        for mouse_n in sorted(grp_df["mouse_ns"].unique()):
            mouse_loc = (grp_df["mouse_ns"] == mouse_n)
            for i in range(2):
                sess_loc = (grp_df["sess_ns"] == comp_sess[i])
                data_row = grp_df.loc[mouse_loc & sess_loc]
                if len(data_row) != 1:
                    raise RuntimeError("Expected to find exactly one row")
                # retrieve ROI data
                data = data_row.loc[data_row.index[0], datatype]
                comp_data[i].append(data)
        
        # add data for each session to dataframe
        for n, data in zip(comp_sess, comp_data):
            stim_data_df.loc[row_idx, f"{stimpar.stimtype}_s{n}"] = \
                np.concatenate(data)

    return stim_data_df

############################################
def abs_fractional_diff(data):
    """
    abs_fractional_diff(data)
    """

    if len(data) != 2:
        raise ValueError("'data' must have length 2.")
    
    data_0, data_1 = data
    data_0 = np.asarray(data_0)
    data_1 = np.asarray(data_1)

    if data_0.shape != data_1.shape:
        raise ValueError("'data' must contain two arrays of the same size.")

    abs_fraction_diff = np.absolute((data_1 - data_0) / data_0)

    return abs_fraction_diff


############################################
def add_stim_pop_stats(stim_stats_df, sessions, analyspar, stimpar, permpar, 
                       comp_sess=[1, 3], in_place=False, randst=None):
    """
    add_stim_pop_stats(stim_stats_df, sessions, analyspar, stimpar, permpar)

    Adds to dataframe comparison of absolute fractional data changes 
    between sessions for different stimuli, calculated for population 
    statistics.

    Required args:
        - stim_stats_df (pd.DataFrame):
            dataframe with one row per line/plane, and the basic sess_df 
            columns, as well as stimulus columns for each comp_sess:
            - {stimpar.stimtype}_s{comp_sess[0]}: 
                first comp_sess data for each ROI
            - {stimpar.stimtype}_s{comp_sess[1]}: 
                second comp_sess data for each ROI
        - sessions (list): 
            session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters

    Optional args:
        - comp_sess (int):
            sessions for which to obtain absolute fractional change 
            [x, y] => |(y - x) / x|
            default: [1, 3]
        - in_place (bool):
            if True, targ_df is modified in place. Otherwise, a deep copy is 
            modified. targ_df is returned in either case.
            default: False
        - randst (int or np.random.RandomState): 
            random state or seed value to use. (-1 treated as None)
            default: None

    Returns:
        - stim_stats_df (pd.DataFrame):
            dataframe with one row per line/plane and one for all line/planes 
            together, and the basic sess_df columns, in addition to the input 
            columns, and for each stimtype:
            - {stimtype} (list): absolute fractional change statistics (me, err)
            - p_vals (float): p-value for data differences between stimulus 
                types, corrected for multiple comparisons and tails
    """

    nanpol = None if analyspar.remnans else "omit"

    if analyspar.tracked:
        misc_analys.check_sessions_complete(sessions, raise_err=False)

    if not in_place:
        stim_stats_df = stim_stats_df.copy(deep=True)

    stimtypes = gen_util.list_if_not(stimpar.stimtype)
    stim_stats_df = gen_util.set_object_columns(
        stim_stats_df, stimtypes, in_place=True
    )

    if analyspar.stats != "mean" or analyspar.error != "std":
        raise NotImplementedError(
            "For population statistics analysis, "
            "analyspar.stats must be set to 'mean', and "
            "analyspar.error must be set to 'std'."
            )

    # initialize arrays for all data
    n_linpla = len(stim_stats_df)
    n_stims = len(stimpar.stimtype)
    n_bootstrp = misc_analys.N_BOOTSTRP

    all_stats = np.full((n_linpla, n_stims), np.nan)
    all_btstrap_stats = np.full((n_linpla, n_stims, n_bootstrp), np.nan)
    all_rand_stat_diffs = np.full((n_linpla, permpar.n_perms), np.nan)

    for i, row_idx in enumerate(stim_stats_df.index):
        full_comp_data = [[], []]
        for s, stimtype in enumerate(stimpar.stimtype):
            comp_data, btstrap_comp_data = [], []
            choices = None
            for n in comp_sess:
                data_col = f"{stimtype}_s{n}"

                # get data
                data = stim_stats_df.loc[row_idx, data_col]

                # get session stats
                comp_data.append(
                    math_util.mean_med(data, analyspar.stats, nanpol=nanpol)
                )

                # get bootstrapped data 
                returns = rand_util.bootstrapped_std(
                    data, randst=randst, n_samples=n_bootstrp, return_rand=True, 
                    return_choices=analyspar.tracked, choices=choices, 
                    nanpol=nanpol
                    )
                
                btstrap_data = returns[1]
                if analyspar.tracked:
                    choices = returns[-1] # use same choices across sessions
                
                btstrap_comp_data.append(btstrap_data)
                full_comp_data[s].append(data) # retain full data

            # compute absolute fractional change stats (bootstrapped std)
            all_stats[i, s] = abs_fractional_diff(comp_data)
            all_btstrap_stats[i, s] = abs_fractional_diff(btstrap_comp_data)
            error = np.std(all_btstrap_stats[i, s])

            # add to dataframe
            stim_stats_df.at[row_idx, stimtype] = [all_stats[i, s], error]  

        # obtain p-values for real data wrt random data
        stim_stat_diff = all_stats[i, 1] - all_stats[i, 0]

        # permute data for each session across stimtypes
        sess_rand_stats = [] # sess x stim
        for j in range(len(comp_sess)):
            rand_concat = [stim_data[j] for stim_data in full_comp_data]
            rand_concat = np.stack(rand_concat).T
            rand_stats = rand_util.permute_diff_ratio(
                rand_concat, div=None, n_perms=permpar.n_perms, 
                stats=analyspar.stats, op="none", paired=True, # pair stimuli
                nanpol=nanpol, randst=randst
                )
            sess_rand_stats.append(rand_stats)
        
        # obtain stats per stimtypes, then differences between stimtypes
        stim_rand_stats = list(zip(*sess_rand_stats)) # stim x sess
        all_rand_stats = []
        for rand_stats in stim_rand_stats:
            all_rand_stats.append(abs_fractional_diff(rand_stats))
        all_rand_stat_diffs[i] = all_rand_stats[1] - all_rand_stats[0]

        # calculate p-value
        p_val = rand_util.get_p_val_from_rand(
            stim_stat_diff, all_rand_stat_diffs[i], tails=permpar.tails,
            nanpol=nanpol
            )
        stim_stats_df.loc[row_idx, "p_vals"] = p_val
    
    # collect stats for all line/planes
    row_idx = len(stim_stats_df)
    for col in stim_stats_df.columns:
        stim_stats_df.loc[row_idx, col] = "all"

    # average across line/planes
    all_data = []
    for data in [all_stats, all_btstrap_stats, all_rand_stat_diffs]:
        all_data.append(
            math_util.mean_med(data, analyspar.stats, nanpol=nanpol, axis=0)
        )
    stat, btstrap_stats, rand_stat_diffs = all_data

    for s, stimtype in enumerate(stimpar.stimtype):
        error = np.std(btstrap_stats[s])
        stim_stats_df.at[row_idx, stimtype] = [stat[s], error]

    p_val = rand_util.get_p_val_from_rand(
        stat[1] - stat[0], rand_stat_diffs, tails=permpar.tails, nanpol=nanpol
        )
    stim_stats_df.loc[row_idx, "p_vals"] = p_val

    return stim_stats_df


############################################
def add_stim_roi_stats(stim_stats_df, sessions, analyspar, stimpar, permpar, 
                       comp_sess=[1, 3], in_place=False, randst=None):
    """
    add_stim_roi_stats(stim_stats_df, sessions, analyspar, stimpar, permpar)

    Adds to dataframe comparison of absolute fractional data changes 
    between sessions for different stimuli, calculated for individual ROIs.

    Required args:
        - stim_stats_df (pd.DataFrame):
            dataframe with one row per line/plane, and the basic sess_df 
            columns, as well as stimulus columns for each comp_sess:
            - {stimpar.stimtype}_s{comp_sess[0]}: 
                first comp_sess data for each ROI
            - {stimpar.stimtype}_s{comp_sess[1]}: 
                second comp_sess data for each ROI
        - sessions (list): 
            session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters

    Optional args:
        - comp_sess (int):
            sessions for which to obtain absolute fractional change 
            [x, y] => |(y - x) / x|
            default: [1, 3]
        - in_place (bool):
            if True, targ_df is modified in place. Otherwise, a deep copy is 
            modified. targ_df is returned in either case.
            default: False
        - randst (int or np.random.RandomState): 
            random state or seed value to use. (-1 treated as None)
            default: None

    Returns:
        - stim_stats_df (pd.DataFrame):
            dataframe with one row per line/plane and one for all line/planes 
            together, and the basic sess_df columns, in addition to the input 
            columns, and for each stimtype:
            - {stimtype} (list): absolute fractional change statistics (me, err)
            - p_vals (float): p-value for data differences between stimulus 
                types, corrected for multiple comparisons and tails
    """

    nanpol = None if analyspar.remnans else "omit"

    if analyspar.tracked:
        misc_analys.check_sessions_complete(sessions, raise_err=True)
    else:
        raise ValueError(
            "If analysis is run for individual ROIs and not population "
            "statistics, analyspar.tracked must be set to True."
            )

    if not in_place:
        stim_stats_df = stim_stats_df.copy(deep=True)

    stimtypes = gen_util.list_if_not(stimpar.stimtype)
    stim_stats_df = gen_util.set_object_columns(
        stim_stats_df, stimtypes, in_place=True
    )

    # compile all data
    full_data = dict()
    for stimtype in stimpar.stimtype:
        for n in comp_sess:
            stim_col = f"{stimtype}_s{n}"
            full_data[stim_col] = np.concatenate(stim_stats_df[stim_col])

    row_idx = len(stim_stats_df)
    for col in stim_stats_df.columns:
        stim_stats_df.loc[row_idx, col] = "all"
        if col in full_data.keys():
            stim_stats_df.loc[row_idx, col] = full_data[col]

    # take statistics
    for row_idx in stim_stats_df.index:
        comp_data = [None, None]
        for s, stimtype in enumerate(stimpar.stimtype):
            stim_data = []
            for n in comp_sess:
                data_col = f"{stimtype}_s{n}"
                stim_data.append(stim_stats_df.loc[row_idx, data_col])
                abs_fractional_diff(stim_data)

            # get stats and add to dataframe
            stim_stats_df.at[row_idx, stimtype] = \
                math_util.get_stats(
                    comp_data[s], analyspar.stats, analyspar.error, 
                    nanpol=nanpol
                    ).tolist()

        # obtain p-values
        stim_stats_df.loc[row_idx, "p_vals"] = rand_util.get_op_p_val(
            comp_data, permpar.n_perms, stats=analyspar.stats, paired=True,
            nanpol=nanpol, randst=randst
            )

    # remove full data columns
    data_cols = []
    for s, stimtype in enumerate(stimpar.stimtype):
        for n in comp_sess:
            data_cols.append(f"{stimtype}_s{n}")    
    stim_stats_df = stim_stats_df.drop(data_cols, axis=1)

    return stim_stats_df


############################################
def get_stim_stats_df(sessions, analyspar, stimpar, permpar, comp_sess=[1, 3], 
                      datatype="rel_unexp_resp", rel_sess=1, basepar=None, 
                      idxpar=None, pop_stats=True, randst=None, 
                      parallel=False): 
    """
    get_stim_stats_df(sessions, analyspar, stimpar, permpar)

    Returns dataframe with comparison of absolute fractional data changes 
    between sessions for different stimuli.

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
        - comp_sess (int):
            sessions for which to obtain absolute fractional change 
            [x, y] => |(y - x) / x|
            default: [1, 3]
        - datatype (str):
            type of data to retrieve
            default: "rel_unexp_resp"
        - rel_sess (int):
            number of session relative to which data should be scaled, for each 
            mouse
            default: 1
        - basepar (BasePar): 
            named tuple containing baseline parameters 
            (needed if datatype is "usis")
            default: None
        - idxpar (IdxPar): 
            named tuple containing index parameters 
            (needed if datatype is "usis")
            default: None
        - pop_stats (bool): 
            if True, analyses are run on population statistics, and not 
            individual tracked ROIs
            default: True
        - randst (int or np.random.RandomState): 
            random state or seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - stim_stats_df (pd.DataFrame):
            dataframe with one row per line/plane and one for all line/planes 
            together, and the basic sess_df columns, in addition to, 
            for each stimtype:
            - {stimtype} (list): absolute fractional change statistics (me, err)
            - raw_p_vals (float): uncorrected p-value for data differences 
                between stimulus types 
            - p_vals (float): p-value for data differences between stimulus 
                types, corrected for multiple comparisons and tails
    """

    if not pop_stats:
        if analyspar.tracked:
            misc_analys.check_sessions_complete(sessions, raise_err=True)
        else:
            raise ValueError(
                "If analysis is run for individual ROIs and not population "
                "statistics, analyspar.tracked must be set to True."
                )

    if set(stimpar.stimtype) != set(["gabors", "bricks"]):
        raise ValueError(
            "Expected stimpar.stimtype to list 'gabors' and 'bricks'."
            )
    if (not (isinstance(stimpar.pre, list) and isinstance(stimpar.post, list))
        or not (len(stimpar.pre) == 2 and len(stimpar.post) == 2)):
        raise ValueError(
            "stimpar.pre and stimpar.post must be provided as lists of "
            "length 2 (one value per stimpar.stimtype, in order)."
            )
    
    if datatype == "usis":
        if (not isinstance(idxpar.feature, list) or 
            not len(idxpar.feature) == 2):
            raise ValueError(
                "idxpar.feature must be provided as a list of length 2 "
                "(one value per stimpar.stimtype, in order)."
                )            

    stim_stats_df = None
    for s, stimtype in enumerate(stimpar.stimtype):
        stim_stimpar = sess_ntuple_util.get_modif_ntuple(
            stimpar, 
            ["stimtype", "pre", "post"], 
            [stimtype, stimpar.pre[s], stimpar.post[s]]
        )

        stim_idxpar = idxpar
        if datatype == "usis":
            stim_idxpar = sess_ntuple_util.get_modif_ntuple(
                idxpar, "feature", idxpar.feature[s]
        )

        stim_stats_df = get_stim_data_df(
            sessions, analyspar, stim_stimpar, stim_data_df=stim_stats_df, 
            comp_sess=comp_sess, datatype=datatype, rel_sess=rel_sess, 
            basepar=basepar, idxpar=stim_idxpar, abs_usi=pop_stats, 
            parallel=parallel
            )

    # add statistics and p-values
    add_stim_stats = add_stim_pop_stats if pop_stats else add_stim_roi_stats 
    stim_stats_df = add_stim_stats(
        stim_stats_df, sessions, analyspar, stimpar, permpar, 
        comp_sess=comp_sess, in_place=True, randst=randst
        )

    # adjust data columns
    data_cols = []
    for s, stimtype in enumerate(stimpar.stimtype):
        for n in comp_sess:
            data_cols.append(f"{stimtype}_s{n}")    

    stim_stats_df = stim_stats_df.drop(data_cols, axis=1)

    stim_stats_df["sess_ns"] = f"comp{comp_sess[0]}v{comp_sess[1]}"
    stim_stats_df = stim_stats_df.rename(columns={"bricks": "visflow"})

    # corrected p-values
    stim_stats_df = misc_analys.add_corr_p_vals(stim_stats_df, permpar)

    return stim_stats_df
    
