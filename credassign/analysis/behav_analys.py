"""
behav_analys.py

This script contains functions for running and pupil analysis.

Authors: Colleen Gillon

Date: February 2023

Note: this code was aggregated from https://github.com/colleenjg/OpenScope_CA_Analysis.
"""

import numpy as np
import pandas as pd

from credassign.util import gen_util, load_util, math_util
from credassign.analysis import misc_analys


#############################################
def get_pupil_run_full(sess, analyspar):
    """
    get_pupil_run_full(sess, analyspar)

    Returns pupil and running data for a full session.

    Required args:
        - sess (list): 
            session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters

    Returns:
        - duration_sec (float):
            duration of the session in seconds
        - pup_data (1D array):
            pupil data
        - pup_frames (list):
            start and stop pupil frame numbers for each stimulus type
        - run_data (1D array):
            running velocity data
        - run_frames (list):
            start and stop running frame numbers for each stimulus type
        - stims (list):
            stimulus types, in order
    """

    duration_sec = sess.tot_twop_fr / sess.twop_fps
    run_data = gen_util.reshape_df_data(
        sess.get_run_velocity(
            rem_bad=analyspar.rem_bad, scale=analyspar.scale
        ), squeeze_rows=True, squeeze_cols=True
    )

    pupil_data_pix = gen_util.reshape_df_data(
        sess.get_pup_data(
            rem_bad=analyspar.rem_bad, scale=analyspar.scale
        ), squeeze_rows=True, squeeze_cols=True
    )
    pupil_data = (pupil_data_pix * load_util.MM_PER_PIXEL)

    stims, pupil_frs, run_frs = [], [], []
    for stim in sess.stims:
        if stim.stimtype == "gabors":
            if len(stim.block_params) != 1:
                raise NotImplementedError("Expected 1 Gabor block..")
            stims.append("Gabor sequences")
        else:
            if len(stim.block_params) != 2:
                raise NotImplementedError("Expected 2 visual flow blocks..")
            directions = stim.block_params["main_flow_direction"].tolist()
            stims.extend(
                [f"Visual flow ({direction})" for direction in directions]
            )
        for bl_idx in stim.block_params.index:
            stim_frames, twop_frames = [[
                int(stim.block_params.loc[bl_idx, key]) 
                for key in [f"start_frame_{ftype}", f"stop_frame_{ftype}"]
                ] for ftype in ["stim", "twop"]
            ]
            pupil_frs.append(twop_frames)
            run_frs.append(stim_frames)

    return run_data, pupil_data, run_frs, pupil_frs, stims, duration_sec


#############################################
def get_pupil_run_full_df(sessions, analyspar, parallel=False):
    """
    get_pupil_run_full_df(sessions, analyspar)

    Returns pupil and running data for full sessions.

    Required args:
        - sessions (list): 
            session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters

    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - sess_df (pd.DataFrame):
            dataframe with one row per session, and the following columns, in 
            addition to the basic sess_df columns:
            - duration_sec (float):
                duration of the session in seconds
            - pup_data (list):
                pupil data
            - pup_frames (list):
                start and stop pupil frame numbers for each stimulus type
            - run_data (list):
                running velocity data
            - run_frames (list):
                start and stop running frame numbers for each stimulus type
            - stims (list):
                stimulus types, in order
    """

    sess_df = misc_analys.get_check_sess_df(sessions, None, roi=False)

    outputs = gen_util.parallel_wrap(
        get_pupil_run_full, sessions, args_list=[analyspar], parallel=parallel, 
        zip_output=True
        )
    run_data, pupil_data, run_frs, pupil_frs, stims, duration_sec = outputs

    misc_analys.get_check_sess_df(sessions, sess_df)
    sess_df["run_data"] = [data.tolist() for data in run_data]
    sess_df["pupil_data"] = [data.tolist() for data in pupil_data]
    sess_df["stims"] = list(stims)
    sess_df["run_frames"] = list(run_frs)
    sess_df["pupil_frames"] = list(pupil_frs)
    sess_df["duration_sec"] = list(duration_sec)

    return sess_df


#############################################
def get_pupil_run_histograms(sessions, analyspar, n_bins=40, parallel=False):
    """
    get_pupil_run_histograms(sessions, analyspar)
    
    Returns pupil and running histograms.

    Required args:
        - sessions (list): 
            session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters

    Optional args:
        - n_bins (int):
            number of bins for correlation data
            default: 40
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - hist_df (pd.DataFrame):
            dataframe with one row per session, and the following columns, in 
            addition to the basic sess_df columns:
            - pupil_bin_edges (float):
                bin edges for the pupil diameter data
            - pupil_vals_binned (list):
                pupil diameter values by bin, for each mouse
            - run_bin_edges (float):
                bin edges for the running data
            - run_vals_binned (list):
                run values by bin, for each mouse
    """
 
    sess_df = get_pupil_run_full_df(sessions, analyspar, parallel=parallel)
    misc_analys.get_check_sess_df(sessions, sess_df, roi=False)

    basic_df = misc_analys.get_check_sess_df(sessions, None, roi=False)
    initial_columns = basic_df.columns.tolist()

    n_sig = 1 # number of significant digits to round bin edges to

    # determine bin edges
    bin_fcts = [np.min, np.max] if analyspar.rem_bad else [np.nanmin, np.nanmax]
    all_bin_edges, new_columns = [], []
    for datatype in ["run", "pupil"]:
        new_columns.append(f"{datatype}_bin_edges")
        new_columns.append(f"{datatype}_vals_binned")

        data = sess_df[f"{datatype}_data"]
        concat_data = np.concatenate(data.tolist())
        edges = [fct(concat_data) for fct in bin_fcts]

        bin_edges = [
            math_util.round_by_order_of_mag(edge, n_sig, direc=direc)
            for edge, direc in zip(edges, ["down", "up"])
        ]

        if bin_edges[0] > edges[0] or bin_edges[1] < edges[1]:
            raise NotImplementedError(
                "Rounded bin edges do not enclose true bin edges."
                )
        all_bin_edges.append(bin_edges)

    columns = initial_columns + new_columns
    hist_df = pd.DataFrame(columns=columns)

    aggreg_cols = [col for col in initial_columns if col != "sess_ns"]
    sess_ns = sorted(sess_df["sess_ns"].unique())
    for sess_n in sess_ns:
        row_idx = len(hist_df)
        sess_grp_df = sess_df.loc[sess_df["sess_ns"] == sess_n]
        sess_grp_df = sess_grp_df.sort_values(["mouse_ns"])

        hist_df.loc[row_idx, "sess_ns"] = sess_n

        # add aggregated values for initial columns
        hist_df = misc_analys.aggreg_columns(
            sess_grp_df, hist_df, aggreg_cols, row_idx=row_idx, 
            sort_by="mouse_ns", in_place=True
            )

        # aggregate running and pupil values into histograms
        mouse_ns = hist_df.loc[row_idx, "mouse_ns"]
        for d, datatype in enumerate(["run", "pupil"]):
            bins = np.linspace(*all_bin_edges[d], n_bins + 1)  
            all_data = []          
            for mouse_n in mouse_ns:
                sess_lines = sess_grp_df.loc[sess_grp_df["mouse_ns"] == mouse_n]
                if len(sess_lines) != 1:
                    raise NotImplementedError("Expected exactly one line")
                sess_line = sess_lines.loc[sess_lines.index[0]]
                binned_vals, _ = np.histogram(
                    sess_line[f"{datatype}_data"], bins=bins
                    )
                all_data.append(binned_vals.tolist())

            hist_df.loc[row_idx, f"{datatype}_bin_edges"] = all_bin_edges[d]
            hist_df.loc[row_idx, f"{datatype}_vals_binned"] = all_data

    hist_df["sess_ns"] = hist_df["sess_ns"].astype(int)

    return hist_df

