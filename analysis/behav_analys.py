"""
behav_analys.py

This script contains functions for running and pupil analysis.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

import numpy as np
import pandas as pd

from util import logger_util, gen_util, math_util
from analysis import misc_analys, seq_analys

logger = logging.getLogger(__name__)




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

    for datatype in ["pupil", "run"]:
        args_dict["datatype"] = datatype
        # sess x split x seq x frames
        split_traces, all_time_values = gen_util.parallel_wrap(
            seq_analys.get_split_data_by_sess, sessions, 
            args_dict=args_dict, parallel=parallel, zip_output=True
            )
        
        # add columns to dataframe
        for col in [f"{datatype}_traces", f"{datatype}_time_values"]:
            trace_df[col] = np.nan
            trace_df[col] = trace_df[col].astype(object)

        # add data to dataframe
        for s, sess in enumerate(sessions):
            row_idx = trace_df.loc[trace_df["sessids"] == sess.sessid].index[0]
            trace_df.at[row_idx, f"{datatype}_traces"] = split_traces[s]
            trace_df.at[row_idx, f"{datatype}_time_values"] = all_time_values[s]

    return trace_df


#############################################
def get_pupil_run_trace_stats_df(sessions, analyspar, stimpar, basepar, 
                                 split="by_exp", parallel=False):
    """
    get_pupil_run_trace_stats_df(sessions, analyspar, stimpar, basepar)

    Returns pupil and running trace statistics for specific sessions, split as 
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

    Returns:
        - trace_df (pd.DataFrame):
            dataframe with one row per session number, and the following 
            columns, in addition to the basic sess_df columns: 
            dataframe with a row for each session, and the following 
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

    nanpol = None if analyspar.remnans else "omit"

    trace_df = get_pupil_run_trace_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        split=split, 
        parallel=parallel
        )
    
    datatypes = ["pupil", "run"]

    columns = trace_df.columns.tolist()
    for datatype in datatypes:
        columns[columns.index(f"{datatype}_traces")] = f"{datatype}_trace_stats"
    grped_trace_df = pd.DataFrame(columns=columns)

    group_columns = ["sess_ns"]
    for grp_vals, trace_grp_df in trace_df.groupby(group_columns):
        row_idx = len(grped_trace_df)
        grp_vals = gen_util.list_if_not(grp_vals)
        for g, group_column in enumerate(group_columns):
            grped_trace_df.loc[row_idx, group_column] = grp_vals[g]

        for column in columns:
            skip = np.sum([datatype in column for datatype in datatypes])
            if column not in group_columns and not skip:
                values = trace_grp_df[column].tolist()
                grped_trace_df.at[row_idx, column] = values

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
            grped_trace_df.at[row_idx, f"{datatype}_trace_stats"] = \
                all_split_stats.tolist()
            grped_trace_df.at[row_idx, f"{datatype}_time_values"] = \
                time_values.tolist()

    grped_trace_df["sess_ns"] = grped_trace_df["sess_ns"].astype(int)

    return grped_trace_df


#############################################
def pupil_run_diffs():

    return

#     data_df = "summary-behav-sel-df.h5" # no baseline
#     summ_behav_sel_df = pd.read_hdf(data_df)
#     fig = plot_behav_sel(summ_behav_sel_df, stimtype_list=["gab"], with_pval=True)

# def plot_behav_sel(df, stimtype_list=["brk", "gab"], norm=False, with_pval=False):
#     """
#     Plots the behvaioural selectivity as seaborn violin and strip plots.
    
#     Returns the figure handle.
    
#     Parameters:
#     ------------
#     df : pandas data frams
#         Summary behavioural data frame with trial differences
#         organized by stimulus and compartment
#     stimtype_list : array_like, optional
#         stimulus types to plot ("brk" or "gab")
#     norm : boolean, optional
#         True: you"re plotting behavioural data that has been 
#         normalized to be between 0 and 1 (so that differences
#         are between -1 and 1)
#         False: you"re plotting raw values
#     with_pval : boolean, optional
#         Determines whether to annotate p-values for each compartment
    
    
#     Returns:
#     ---------
#     fig : figure handle
        
#     """
#     stim_str = ["Visual flow", "Gabor seq."]
#     layers = {"L23-Cux2":"L2/3", "L5-Rbp4":"L5"}
#     lc = list(itertools.product(layers.values(), ["dend", "soma"]))


#     [_, planes, _, 
#      _, pla_col_names, _] = sess_plot_util.fig_linpla_pars(n_grps=4)
    
#     colors = []
#     for linpla in lc:
#         pla = "dendrites" if linpla[1] == "dend" else "somata"
#         pl = planes.index(pla)
#         color = get_colors(pla_col_names[pl], line=linpla[0])
#         colors.append(color)
        
#     # switching dictionaries
#     behav_dict = {"run speed":"run_sel__diff", "pupil diameter":"pd_sel__diff"}
#     pval_dict = {"run speed":"run_pval", "pupil diameter":"pd_pval"}
#     stim_title_dict = {"gab":"Gabor seq.", "brk":"Visual flow"}
#     ylabel_raw_dict = \
#         {"run speed":
#          "Velocity diff (cm/s)\nfor DvU frames",
#          "pupil diameter":
#          "Pupil diam. diff (mm)\nfor DvU frames"}

#     n_stims = len(stimtype_list)
#     fig, ax = plt.subplots(n_stims, 2, figsize=(11.2 * n_stims, 4.7))

#     for i, which_behav in enumerate(["run speed", "pupil diameter"]):
#         for j, stim in enumerate(stimtype_list):
#             mask = df["stim"] == stim
#             mouse_ns = df["mouse_ns"]

#             plot_row = i
#             plot_col = j
#             if len(stimtype_list) > 1:
#                 plot_idx = (i, j)
#             else:
#                 plot_idx = 2 * j + i

#             sub_ax = ax[plot_idx]
#             linpla = [linplas[0] for linplas in df[mask]["layer_compartment_array"]]

#             x = range(len(lc))    
#             x0 = np.hstack(df[mask]["layer_compartment_array"].values)
#             y0 = np.hstack(df[mask][behav_dict[which_behav]].values)
#             if "pupil" in which_behav: # convert to mm
#                 y0 *= MM_PER_PIXEL
#             pval = np.hstack(df[mask][pval_dict[which_behav]])

#             bplot = sns.violinplot(x=x0, y=y0, inner=None, linewidth=3.5, 
#                            color="white", ax=sub_ax)
#             # set violin edgecolors to black
#             NEARBLACK = "#656565"
#             for c, collec in enumerate(bplot.collections):
#                 collec.set_edgecolor(NEARBLACK)
#                 if lc[c][0] == "L5": 
#                     collec.set_linestyle(plot_acr.VDASH)
#             sns.stripplot(x=x0, y=y0, size=9, jitter=0.2,
#                 alpha=0.3, palette=colors, ax=sub_ax)
#             sub_ax.axhline(y=0, ls=plot_acr.HDASH, c="k", lw=3.0, alpha=0.5)

#             # p-values are 2-tailed, but already corrected 
#             # i.e., if pval < 0.5: pval = 1-pval; pval *= 2 in the initial calculation
#             n_comps = len(x) # for correction
#             alpha = 0.05
#             p_val_str = f"{which_behav}, {stim}: "
#             if with_pval:
#                 for i_x, xval in enumerate(x):
#                     p_val_str = f"{p_val_str}: {pval[i_x] * n_comps:.5f} ({linpla[i_x]:6})"
#                     lead = ""
#                     if i_x == 0:
#                         lead = "raw p = "
#                         xval = xval - 0.5
#                     if pval[i_x] <= alpha:
#                         text = "{}{:.2f}*".format(lead, pval[i_x])
#                     else:
#                         text = "{}{:.2f}".format(lead, pval[i_x])                        
#                     sub_ax.text(xval, sub_ax.get_ylim()[1], text, fontsize=20, 
#                                 weight="bold", ha="center")
#             print(p_val_str)
#             sub_ax.set_title("{} {}".format(stim_title_dict[stim], which_behav), y=1.1)
#             if norm:
#                 sub_ax.set_ylabel("Norm. {} violation \n changes for all trials".format(
#                     which_behav))
#             else:
#                 sub_ax.set_ylabel(ylabel_raw_dict[which_behav])
                

