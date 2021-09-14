"""
corr_analys.py

This script contains functions for running and pupil analysis.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

from util import logger_util

logger = logging.getLogger(__name__)




def pupil_run_responses():

    return

#     data_df = "behav_traces_sess_1_df.h5"
#     behav_traces_df = pd.read_hdf(data_df)
#     baseline = False
#     error = "sem"
#     fig = plot_all_behaviour(behav_traces_df, baseline=False, error="sem"):

#     def plot_behaviour(ax, df, behaviour_type, baseline=False, error="sem"):
#     """
#     Extracts data from df based on the columns specified by the behaviour type, computes
#     the mean and SEM over trials.
    
#     ax: Axis on which to plot
    
#     df: Data frame with the associated run velocity and pupil diameter traces
    
#     behaviour_type: plot run velocity ("run") or pupil diameter ("pd")
#     """
    
#     # Useful dicts etc.
#     behaviour_dict = {"run": "run vel\n(cm/s)", "pd": "pupil diam\n(mm)"}
#     stimtype = "gabors" if "gab" in behaviour_type else "bricks"
#     n_mice = df["mouse_n"].nunique()

#     behaviour_type = behaviour_type.split("_")[0]
    
#     stimtype_dict = {
#         "gabors": {"expec": "abcdg", "unexpec": "abceg"}, 
#         "bricks": {"expec": "cons_flow", "unexpec": "viol_flow"}}

#     if stimtype == "gabors":
#         nticks = 6
#         secs = [-0.9, 0.6]
#         if behaviour_type == "run":
#             base_idx = [46, 54] # -0.13 to 0s
#         elif behaviour_type == "pd":
#             base_idx = [23, 27] # -0.13 to 0s
#     else:
#         nticks = 7
#         sec = [0, 2.0]
    
#     datatypes = ["expec", "unexpec"]
#     colornames = ["gray", "red"]
#     means, errs = [], []
#     for datatype in ["expec", "unexpec"]:
#         data_col = behaviour_type + "_" + stimtype_dict[stimtype][datatype]
#         which_rows = np.array([len(df.iloc[i][data_col]) 
#             for i in range(df.shape[0])]).astype("bool")
#         data = np.concatenate(df[data_col][which_rows].values, axis=1)[0, :, :]
#         if behaviour_type == "pd":
#             data = data * MM_PER_PIXEL
#         if baseline:
#             if stimtype == "bricks":
#                 raise NotImplementedError("Baseline not applied for bricks, as the data "
#                     "to baseline from is not available.")
#             baseline_mean = np.nanmean(data[:, base_idx[0] : base_idx[1]], axis=1)
#             data = data - baseline_mean.reshape(-1, 1)
            
#         means.append(np.nanmean(data, axis=0))
#         if error == "sem":
#             errs.append(scist.sem(data, axis=0, nan_policy="omit"))
#         else:
#             errs.append(np.nanstd(data, axis=0))
        
#     # Main plot
#     for i in range(len(means)):
#         colorname = colornames[i]
#         color = plot_util.LINCLAB_COLS[colorname]
#         if colorname == "gray":
#             alpha = 0.4
#             alpha_line = 1.0
#         else:
#             alpha = 0.2
#             alpha_line = 0.75 
#         xran = np.linspace(*secs, len(means[i]))
#         plot_util.plot_traces(ax, xran, means[i], err=errs[i], 
#             alpha=alpha, n_xticks=nticks, #label=datatypes[i], 
#             alpha_line=alpha_line, color=color)
    
#     full_ax = np.asarray(ax).reshape(1)
            
#     if stimtype == "gabors":
#         ax.axvline(x=0, ls=plot_acr.VDASH, c="k", lw=3.0, alpha=0.5, zorder=-13)
        
#     if baseline:
#         ax.axhline(y=0, ls=plot_acr.HDASH, c="k", lw=3.0, alpha=0.5, zorder=-13)
        
#     plot_util.set_interm_ticks(full_ax, nticks, dim="x", share=False, weight="bold")
#     plot_util.set_interm_ticks(full_ax, 4, dim="y", share=False, weight="bold")

#     ax.set_xlim(secs)
#     ax.set_ylabel(behaviour_dict[behaviour_type], weight="bold")
    
    
# def plot_all_behaviour(behav_traces_df, baseline=False, error="sem"):

#     fig, ax = plt.subplots(nrows=2, figsize=[4.8, 4], sharex=True)
#     plot_behaviour(ax[0], behav_traces_df, "run_gab", baseline=baseline, error=error)
#     # at least trial has only NaNs in the baseline, hence warning
#     plot_behaviour(ax[1], behav_traces_df, "pd_gab", baseline=baseline, error=error)
    
#     return fig


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
                

