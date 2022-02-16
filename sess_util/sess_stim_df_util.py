"""
sess_stim_df_util.py

This module contains the functions needed to generate the stimulus dataframe
for Allen Institute OpenScope experiments for the Credit Assignment Project.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

from pathlib import Path

import numpy as np
import pandas as pd
import pynwb

from util import file_util, gen_util, logger_util
from sess_util import sess_file_util, sess_sync_util


TAB = "    "


logger = logger_util.get_module_logger(name=__name__)


##########################################

STIMULUS_NAME_MAP = {"b": "visflow", "g": "gabors", -1: "grayscreen"}
FLIP_FRACTION = 0.25
SQUARE_SIZE_TO_NUMBER = {128: 105, 256: 26, -1: -1}
VISFLOW_DIR = {"right": "right (temp)", "left": "left (nasal)", -1: -1}

GABOR_ORI_RANGE = [0, 360]
GABORS_KAPPAS = [4, 16]
GABORS_NUMBER = 30
GABORS_SETS_TO_LETTER = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    "grayscreen": "G",
    -1: -1,
}
FINAL_COLUMN_ORDER = [
    "stimulus_type",
    "stimulus_template_name",
    "orig_stimulus_segment",  # to be dropped
    "unexpected",
    "gabor_frame",
    "gabor_kappa",
    "gabor_mean_orientation",
    "gabor_number",
    "gabor_locations_x",
    "gabor_locations_y",
    "gabor_sizes",
    "gabor_orientations",
    "main_flow_direction",
    "square_size",
    "square_number",
    "square_proportion_flipped",
    "square_locations_x",
    "square_locations_y",
    "start_frame_stim_template",
    "start_frame_stim",
    "stop_frame_stim",
    "num_frames_stim",
    "start_frame_twop",
    "stop_frame_twop",
    "num_frames_twop",
    "start_time_sec",
    "stop_time_sec",
    "duration_sec",
]
STIMPAR = {
    "gabor_kappa": "stimPar2",
    "gabor_mean_orientation": "stimPar1",
    "main_flow_direction": "stimPar2",
    "square_size": "stimPar1",
}

FULL_TABLE_COLUMNS = [
    "gabor_orientations",
    "square_locations_x",
    "square_locations_y",
]

NWB_ONLY_COLUMNS = [
    "start_frame_stim_template",
]

# ADDITIONAL HARD_CODED STIMULUS FEATURES (PRODUCTION ONLY) TO ACCOMPANY NWB DATA
DEG_PER_PIXEL = 0.06251912565744862
EXP_LEN_SEC = [30, 90]
WIN_SIZE = [1920, 1200] # pixels

GABORS_SEG_LEN_SEC = 0.3
GABORS_UNEXP_LEN_SEC = [3, 6]

GABORS_N_SEGS_PER_SEQ = 5
GABORS_PHASE = 0.25 # 0 to 1
GABORS_SIZE_RAN = [204, 408] # in pixels
GABORS_SPATIAL_FREQ = 0.04 # in cyc/pix

VISFLOW_SEG_LEN_SEC = 1
VISFLOW_UNEXP_LEN_SEC = [2, 4]

VISFLOW_SPEED = 799.7552664756905 # in pix/sec


#############################################
def load_gen_stim_properties(stim_dict, stimtype="gabors", runtype="prod"):
    """
    load_gen_stim_properties(stim_dict)

    Returns dictionary with general stimulus properties loaded from the 
    stimulus dictionary.
    
    Arguments:
        - stim_dict (dict)   : experiment stim dictionary, loaded from pickle

    Optional arguments:
        - stimtype (str): stimulus type
                          default: "gabors"
        - runtype (str) : runtype ("prod" or "pilot")
                          default: "prod"):
    
    Returns:
        - gen_stim_props (dict): dictionary with stimulus properties.
            ["deg_per_pix"]: deg / pixel conversion used to generate stimuli
            ["exp_len_s"]  : duration of an expected seq (sec) [min, max]
            ["seg_len_s"]  : duration of an expected seq (sec) [min, max]
            ["unexp_len_s"]: duration of an unexpected seq (sec) [min, max]
            ["win_size"]   : window size [wid, hei] (in pixels)

            if stimtype == "gabors":
            ["n_segs_per_seq"]: number of segments in a sequence (including G)
            ["phase"]         : phase (0-1)
            ["sf"]            : spatial frequency (cyc/pix)
            ["size_ran"]      : size range (in pixels)

            if stimtype == "visflow":
            ["speed"]         : visual flow speed (pix/sec)
    """

    if not isinstance(stim_dict, dict):
        stim_dict = file_util.loadfile(stim_dict, filetype="pickle")

    # run checks
    runtypes = ["prod", "pilot"]
    if runtype not in ["prod", "pilot"]:
        gen_util.accepted_values_error("runtype", runtype, runtypes)

    stimtypes = ["gabors", "visflow"]
    if stimtype not in stimtypes:
        gen_util.accepted_values_error("stimtype", stimtype, stimtypes)

    # find a stimulus of the correct type in dictionary
    stim_params_key = "stimParams" if runtype == "pilot" else "stim_params"
    stimtype_key = "gabor_params" if stimtype == "gabors" else "square_params"
    stim_ns = [
        stim_n for stim_n, all_stim_info in enumerate(stim_dict["stimuli"]) 
        if stimtype_key in all_stim_info[stim_params_key].keys()
        ]
    
    if len(stim_ns) == 0:
        raise RuntimeError(
            f"No {stimtype} stimulus found in stimulus dictionary."
            )
    else:
        stim_n = stim_ns[0] # same general stimulus properties expected for all

    # collect information
    all_stim_info = stim_dict["stimuli"][stim_n]

    if runtype == "prod":
        sess_par = all_stim_info[stim_params_key]["session_params"]
    else:
        sess_par = all_stim_info[stim_params_key]["subj_params"]

    gen_stim_props = {
        "win_size"   : sess_par["windowpar"][0],
        "deg_per_pix": sess_par["windowpar"][1],
    }

    stimtype_info = all_stim_info[stim_params_key][stimtype_key]
    gen_stim_props["exp_len_s"]   = stimtype_info["reg_len"]        
    gen_stim_props["unexp_len_s"] = stimtype_info["surp_len"]

    deg_per_pix = gen_stim_props["deg_per_pix"] 
    if stimtype == "gabors":
        gen_stim_props["seg_len_s"] = stimtype_info["im_len"]
        gen_stim_props["n_segs_per_seq"] = stimtype_info["n_im"] + 1 # for G 

        gen_stim_props["phase"]     = stimtype_info["phase"]
        gen_stim_props["sf"]        = stimtype_info["sf"]
        gen_stim_props["size_ran"]  = \
            [np.around(x / deg_per_pix) for x in stimtype_info["size_ran"]]

        # Gabor size conversion based on psychopy definition
        # full-width half-max -> 6 std
        size_conv = 1.0 / (2 * np.sqrt(2 * np.log(2))) * stimtype_info["sd"]
        gen_stim_props["size_ran"]  = \
            [int(np.around(x * size_conv)) for x in gen_stim_props["size_ran"]]

    else:
        gen_stim_props["seg_len_s"] = stimtype_info["seg_len"]
        gen_stim_props["speed"]     = stimtype_info["speed"] / deg_per_pix
    
    return gen_stim_props


#############################################
def load_hard_coded_stim_properties(stimtype="gabors", runtype="prod"):
    """
    load_hard_coded_stim_properties()

    Returns dictionary with general stimulus properties hard-coded.
    
    Optional arguments:
        - stimtype (str): stimulus type
                          default: "gabors"

        - runtype (str): runtype ("prod" or "pilot")
                         default: "prod"):
    
    Returns:
        - gen_stim_props (dict): dictionary with stimulus properties.
            ["deg_per_pix"]: deg / pixel conversion used to generate stimuli
            ["exp_len_s"]  : duration of an expected seq (sec) [min, max]
            ["seg_len_s"]  : duration of an expected seq (sec) [min, max]
            ["unexp_len_s"]: duration of an unexpected seq (sec) [min, max]
            ["win_size"]   : window size [wid, hei] (in pixels)

            if stimtype == "gabors":
            ["n_segs_per_seq"]: number of segments in a sequence (including G)
            ["phase"]         : phase (0-1)
            ["sf"]            : spatial frequency (cyc/pix)
            ["size_ran"]      : size range (in pixels)

            if stimtype == "visflow":
            ["speed"]         : visual flow speed (pix/sec)
    """

    if runtype != "prod":
        raise ValueError(
            "Hard-coded properties are only available for production data."
            )

    gen_stim_props = {
        "deg_per_pix": DEG_PER_PIXEL,
        "exp_len_s"  : EXP_LEN_SEC,
        "win_size"   : WIN_SIZE,
    }

    if stimtype == "gabors":
        gen_stim_props["seg_len_s"]      = GABORS_SEG_LEN_SEC
        gen_stim_props["unexp_len_s"]    = GABORS_UNEXP_LEN_SEC

        gen_stim_props["n_segs_per_seq"] = GABORS_N_SEGS_PER_SEQ
        gen_stim_props["phase"]          = GABORS_PHASE
        gen_stim_props["size_ran"]       = GABORS_SIZE_RAN
        gen_stim_props["sf"]             = GABORS_SPATIAL_FREQ

    elif stimtype == "visflow":
        gen_stim_props["seg_len_s"]    = VISFLOW_SEG_LEN_SEC
        gen_stim_props["unexp_len_s"]  = VISFLOW_UNEXP_LEN_SEC

        gen_stim_props["speed"]        = VISFLOW_SPEED

    else:
        gen_util.accepted_values_error(
            "stimtype", stimtype, ["gabors", "visflow"]
            )

    return gen_stim_props
    
    
#############################################
def load_basic_stimulus_table(stim_dict, stim_sync_h5, time_sync_h5, align_pkl, 
                              sessid, runtype="prod"):
    """
    load_basic_stimulus_table(stim_dict, stim_sync_h5, time_sync_h5, align_pkl, 
                              sessid)

    Creates the alignment dataframe (stim_df) and saves it as a pickle
    in the session directory, if it does not already exist. Returns dataframe.
    
    Arguments:
        - stim_dict (dict)   : experiment stim dictionary, loaded from pickle
        - stim_sync_h5 (Path): full path name of the experiment sync hdf5 file
        - time_sync_h5 (Path): full path name of the time synchronization hdf5 
                               file
        - align_pkl (Path)   : full path name of the output pickle file to 
                               create
        - sessid (int)       : session ID, needed the check whether this 
                               session needs to be treated differently 
                               (e.g., for alignment bugs)

    Optional arguments:
        - runtype (str): runtype ("prod" or "pilot")
                         default: "prod"):
    
    Returns:
        df (pandas): basic stimulus table.
        stim_align (1D array): stimulus to 2p alignment array
    """

    align_pkl = Path(align_pkl)
    sessdir = align_pkl.parent

    # create stim_df if doesn't exist
    if not align_pkl.is_file():
        logger.info(f"Stimulus alignment pickle not found in {sessdir}, and "
            "will be created.", extra={"spacing": TAB})
        sess_sync_util.get_stim_frames(
            stim_dict, stim_sync_h5, time_sync_h5, align_pkl, sessid, runtype, 
            )
        
    align_dict = file_util.loadfile(align_pkl)

    df = align_dict["stim_df"]
    stim_align = align_dict["stim_align"].astype(int)

    return df, stim_align


#############################################
def update_basic_stimulus_table(df):
    """
    update_basic_stimulus_table(df)
    
    Updates columns of basic stimulus table.

    Arguments:
        df (pandas): stimulus table, as saved.

    Returns:
        df (pandas): stimulus table, with modified columns FINAL_COLUMN_ORDER.
    """


    df = df.rename(
        columns={
            "stimType": "stimulus_type",
            "GABORFRAME": "gabor_frame",
            "surp": "unexpected",
            "stimSeg": "orig_stimulus_segment",
            "start_frame": "start_frame_twop",
            "end_frame": "stop_frame_twop",
            "num_frames": "num_frames_twop",
        }
    )

    df["stimulus_type"] = df["stimulus_type"].map(STIMULUS_NAME_MAP)

    df = df.sort_values("start_frame_twop").reset_index(drop=True)

    for column in FINAL_COLUMN_ORDER:
        if column not in df.columns:
            df[column] = -1

    # visual flow columns
    stimulus_location = df["stimulus_type"] == "visflow"

    # set parameters from stimPar1 and stimPar2
    for key in ["square_size", "main_flow_direction"]:
        df.loc[stimulus_location, key] = df.loc[stimulus_location, STIMPAR[key]]
    # add number of visual flow squares
    df["square_number"] = df["square_size"].map(SQUARE_SIZE_TO_NUMBER)
    df["main_flow_direction"] = df["main_flow_direction"].map(VISFLOW_DIR)
    
    # set proportion flipped
    square_flip_map = {0: 0, 1: FLIP_FRACTION, -1: -1}
    df.loc[stimulus_location, "square_proportion_flipped"] = df.loc[
        stimulus_location, "unexpected"
    ].map(square_flip_map)

    # Gabors columns
    stimulus_location = df["stimulus_type"] == "gabors"

    # set parameters from stimPar1 and stimPar2
    for key in ["gabor_kappa", "gabor_mean_orientation"]:
        df.loc[stimulus_location, key] = df.loc[stimulus_location, STIMPAR[key]]
    # add number of Gabors
    df.loc[
        stimulus_location & (df["gabor_frame"] != "G"), "gabor_number"
    ] = GABORS_NUMBER
    # update Gabor sets and mean orientations for U sets
    df["gabor_frame"] = df["gabor_frame"].map(GABORS_SETS_TO_LETTER)
    df.loc[
        (df["gabor_frame"] == "D") & (df["unexpected"] == 1), "gabor_frame"
    ] = "U"
    df.loc[df["gabor_frame"] == "U", "gabor_mean_orientation"] += 90

    # drop extra columns
    df = df[FINAL_COLUMN_ORDER]

    # drop non Gabors, visual flow rows and reindex
    df = (
        df.loc[df["stimulus_type"].isin(["visflow", "gabors"])]
        .sort_values("start_frame_twop")
        .reset_index(drop=True)
    )

    return df


#############################################
def _num_pre_blank_frames(stim_dict):
    """
    _num_pre_blank_frames(stim_dict)

    Retrieves number of blank frames before stimulus starts.

    Arguments:
        stim_dict (dict): An Allen Institute session stimulus dictionary.

    Returns:
        num_pre_blank_frames (int): number of blank frames before stimulus starts.
    """

    stimulus_fps = stim_dict["fps"]
    num_pre_blank_frames = int(stim_dict["pre_blank_sec"] * stimulus_fps)
    return num_pre_blank_frames


#############################################
def add_stimulus_frame_num(df, stim_dict, stim_align, runtype="prod"):
    """
    add_stimulus_frame_num(df, stim_dict, stim_align)

    Adds stimulus frame numbers to the Gabors and visual flow stimulus rows in 
    the stimulus table.

    Arguments:
        df (pandas): stimulus table
        stim_dict (dict): An Allen Institute session stimulus dictionary.
        stim_align (1D array): stimulus to 2p alignment array

    Optional arguments:
        runtype (str): deployment type (prod or pilot)
        default: "prod"

    Returns:
        df (pandas): stimulus table, with stimulus frame numbers updated.
    """

    df = df.copy()

    df = df.sort_values("start_frame_twop").reset_index(drop=True)

    stimulus_num_dict = {"gabors": [], "visflow": []}
    stimulus_start_frame = {"gabors": [], "visflow": []}

    if runtype == "pilot":
        key = "stimParams"
    elif runtype == "prod":
        key = "stim_params"
    else:
        gen_util.accepted_values_error("runtype", runtype, ["prod", "pilot"])

    for n, stimulus_dict in enumerate(stim_dict["stimuli"]):
        stimulus_type = STIMULUS_NAME_MAP[stimulus_dict[key]["elemParams"]["name"][0]]
        if stimulus_type not in stimulus_num_dict.keys():
            raise RuntimeError(f"Stimulus type {stimulus_type} not recognized.")
        start_frame = np.where(stimulus_dict["frame_list"] == 0)[0][0]

        stimulus_num_dict[stimulus_type].append(n)
        stimulus_start_frame[stimulus_type].append(start_frame)

    num_pre_blank_frames = _num_pre_blank_frames(stim_dict)

    df = df.reset_index()
    for stimulus in ["gabors", "visflow"]:
        order = np.argsort(stimulus_start_frame[stimulus])
        for dictionary in [stimulus_num_dict, stimulus_start_frame]:
            dictionary[stimulus] = [dictionary[stimulus][n] for n in order]

        stimulus_location = df["stimulus_type"] == stimulus
        stimulus_df_start_idx = df.loc[
            stimulus_location & (df["orig_stimulus_segment"] == 0), "index"
        ].tolist() + [len(df)]
        for i, n in enumerate(stimulus_num_dict[stimulus]):
            sub_stimulus_location = stimulus_location & df["index"].isin(
                range(stimulus_df_start_idx[i], stimulus_df_start_idx[i + 1])
            )
            segments = df.loc[
                sub_stimulus_location, "orig_stimulus_segment"
                ].to_numpy()
            frame_list = stim_dict["stimuli"][n]["frame_list"]
            segment_frames = [
                np.where(frame_list == val)[0] + num_pre_blank_frames
                for val in segments
            ]
            start_frames, stop_frames = [
                np.asarray(frames)
                for frames in zip(
                    *[[frames[0], frames[-1] + 1] for frames in segment_frames]
                )
            ]

            df.loc[sub_stimulus_location, "start_frame_stim"] = start_frames
            df.loc[sub_stimulus_location, "stop_frame_stim"] = stop_frames
            df.loc[sub_stimulus_location, "num_frames_stim"] = (
                stop_frames - start_frames
            )

            # double check the 2p frames match stimulus frames
            for key, frames in zip(
                ["start_frame_twop", "stop_frame_twop"], 
                [start_frames, stop_frames]
            ):
                if (
                    df.loc[sub_stimulus_location, key] != stim_align[frames]
                ).any():
                    raise NotImplementedError(
                        f"Values in `{key}` do not match expected values."
                    )

    df = df.drop(columns=["index"])

    return df


#############################################
def add_stimulus_locations(df, stim_dict, runtype="prod"):
    """
    add_stimulus_locations(df, stim_dict)

    Adds stimulus locations and sizes to the Gabors and visual flow stimulus 
    rows in the stimulus table.

    Arguments:
        df (pandas): stimulus table
        stim_dict (dict): An Allen Institute session stimulus dictionary.

    Optional arguments:
        runtype (str): deployment type (prod or pilot)
        default: "prod"

    Returns:
        df (pandas): stimulus table, with stimulus locations and sizes updated.
    """

    df = df.copy()

    df = df.sort_values("start_frame_twop").reset_index(drop=True)

    for column in [
        "gabor_orientations",
        "gabor_locations_x",
        "gabor_locations_y",
        "gabor_sizes",
        "square_locations_x",
        "square_locations_y",
    ]:
        df[column] = df[column].astype(object)
        df[column] = [[]] * len(df)

    stimulus_starts = df.loc[df["orig_stimulus_segment"] == 0].index
    stimulus_edges = stimulus_starts.to_list() + [len(df)]
    stimuli = stim_dict["stimuli"]
    stimulus_sorter = np.argsort(
        [stimulus["frame_list"].tolist().index(0) for stimulus in stimuli]
    )
    sorted_stimuli = [stimuli[s] for s in stimulus_sorter]

    if len(sorted_stimuli) != len(stimulus_starts):
        raise RuntimeError(
            "The dataframe should have as many stimulus start segments as "
            "the pickle has stimuli."
        )

    drop_columns = [
        "gabor_orientations", "square_locations_x", "square_locations_y"
        ]

    for s, (stimulus, stimulus_start) in enumerate(
        zip(sorted_stimuli, stimulus_starts)
    ):
        stimulus_type = df.loc[stimulus_start, "stimulus_type"]
        stimulus_end = stimulus_edges[s + 1] - 1
        basic_loc = df["stimulus_type"] == stimulus_type

        if runtype == "pilot":
            stim_params = stimulus["stimParams"]
            win_size, _  = stim_params["subj_params"]["windowpar"]
            square_locations = "posByFrame"
            gabor_orientations = "orisByImg"
            gabor_locations_sizes = "possizes"

        elif runtype == "prod":
            basic_loc = (
                basic_loc & 
                (df.index >= stimulus_start) & 
                (df.index <= stimulus_end)
            )
            stim_params = stimulus["stim_params"]["session_params"]
            win_size, _  = stim_params["windowpar"]
            square_locations = "posbyframe"
            gabor_orientations = "orisbyimg"
            gabor_locations_sizes = "possize"

        else:
            gen_util.accepted_values_error(
                "runtype", runtype, ["prod", "pilot"]
                )

        if stimulus_type == "gabors":
            # may be omitted for size
            if gabor_orientations in stim_params.keys():
                orientations = np.asarray(stim_params[gabor_orientations])

                # adjust for Us, then set all to within pre-set range
                U_rows = np.where(
                    (df.loc[basic_loc, "gabor_frame"] == "U").to_numpy()
                    )[0]
                orientations[U_rows] += 90
                orientations[orientations < GABOR_ORI_RANGE[0]] += 360
                orientations[orientations > GABOR_ORI_RANGE[1]] -= 360

                df.loc[basic_loc, "gabor_orientations"] = pd.Series(
                    list(orientations), index=df.loc[basic_loc].index
                )

                # ensure column is kept
                if "gabor_orientations" in drop_columns:
                    drop_columns.remove("gabor_orientations")

            locations, sizes = [
                np.asarray(sub)
                for sub in list(zip(*stim_params[gabor_locations_sizes]))
            ]

            for i, gabor_set in enumerate(["A", "B", "C", "D", "U"]):
                loc = basic_loc & (df["gabor_frame"] == gabor_set)
                n_rows = len(df.loc[loc])

                df.loc[loc, "gabor_locations_x"] = pd.Series(
                    list(locations[i : i + 1, :, 0]) * n_rows, 
                    index=df.loc[loc].index,
                )
                df.loc[loc, "gabor_locations_y"] = pd.Series(
                    list(locations[i : i + 1, :, 1]) * n_rows, 
                    index=df.loc[loc].index,
                )
                df.loc[loc, "gabor_sizes"] = pd.Series(
                    list(sizes[i].astype(int).reshape(1, -1)) * n_rows,
                    index=df.loc[loc].index,
                )

        elif stimulus_type == "visflow":
            if runtype == "pilot":
                # different stimulus blocks are consecutive in locations array for pilot
                seg_nbrs = list(filter(lambda x: x != -1, stimulus["frame_list"]))
                start_frame_stim_within = np.insert(
                    np.where(np.diff(seg_nbrs) != 0)[0] + 1, 0, 0
                )

            elif runtype == "prod":
                start_frame_stim = df.loc[basic_loc, "start_frame_stim"].to_numpy()
                start_frame_stim_within = start_frame_stim - start_frame_stim.min()

            duration = np.diff(start_frame_stim_within)[0]
            if (np.diff(start_frame_stim_within) != duration).all():
                raise RuntimeError(
                    "Expected all start_frame_stim_within values to be "
                    "separated by the same number of frames."
                )

            frames_stim_within = np.tile(
                start_frame_stim_within, (duration, 1)
            ) + np.arange(duration).reshape(-1, 1)

            # may be omitted for size
            if square_locations in stim_params.keys():
                locations = np.asarray(stim_params[square_locations]).astype(int)

                df.loc[basic_loc, "square_locations_x"] = pd.Series(
                    list(np.transpose(locations[frames_stim_within, :, 0], [1, 2, 0])),
                    index=df.loc[basic_loc].index,
                )
                df.loc[basic_loc, "square_locations_y"] = pd.Series(
                    list(np.transpose(locations[frames_stim_within, :, 1], [1, 2, 0])),
                    index=df.loc[basic_loc].index,
                )

                # ensure columns are kept
                for col in ["square_locations_x", "square_locations_y"]:
                    if col in drop_columns:
                        drop_columns.remove(col)

        else:
            raise RuntimeError(
                f"`stimulus_type` value {stimulus_type} not recognized."
                )

    if len(drop_columns):
        df = df.drop(columns=drop_columns)

    return df


#############################################
def modify_segment_num(df):
    """
    modify_segment_num(df)

    Modifies stimulus segment numbers for the Gabors and visual flow if they 
    repeat.

    Arguments:
        df (pandas): stimulus table

    Returns:
        df (pandas): stimulus table, with updated stimulus segment numbers.
    """

    df = df.copy()

    df = df.sort_values("start_frame_twop").reset_index(drop=True)
    df = df.reset_index()

    stimulus_types = ["gabors", "visflow"]
    for stimulus in stimulus_types:
        stimulus_location = df["stimulus_type"] == stimulus
        stimulus_df_start_idx = df.loc[
            stimulus_location & (df["orig_stimulus_segment"] == 0), "index"
        ].tolist() + [len(df)]
        add_segment_num = 0
        for i in range(len(stimulus_df_start_idx[:-1])):
            sub_stimulus_location = stimulus_location & df["index"].isin(
                range(stimulus_df_start_idx[i], stimulus_df_start_idx[i + 1])
            )
            df.loc[
                sub_stimulus_location, "orig_stimulus_segment"
                ] += add_segment_num
            add_segment_num = (
                df.loc[
                    sub_stimulus_location, "orig_stimulus_segment"
                    ].max() + 1
            )

    df = df.drop(columns=["index"])

    return df


#############################################
def add_gabor_grayscreen_rows(df, stim_align):
    """
    add_gabor_grayscreen_rows(df, stim_align)

    Updates dataframe with grayscreen rows added after the Gabor D/U segments.

    Arguments:
        df (pandas): stimulus dataframe.
        stim_align (1D array): stimulus to 2p alignment array

    Returns:
        df (pandas): updated stimulus table with grayscreen rows.
    """

    df = df.copy()

    df = df.sort_values("start_frame_twop").reset_index(drop=True)

    kappas_ordered = list(
        filter(lambda k: k in GABORS_KAPPAS, df["gabor_kappa"].unique())
    )

    if (df["start_frame_stim"].to_numpy() == -1).any():
        raise RuntimeError(
            "Stimulus frame values must be set to run "
            "`add_gabor_grayscreen_rows()`."
        )

    num_stimulus_frames = len(stim_align)
    for kappa in kappas_ordered:
        gabor_DU_location = (df["stimulus_type"] == "gabors") & df["gabor_frame"].isin(
            ["D", "U"]
        )

        kappa_location = df["gabor_kappa"] == kappa
        DU_segment_idx = df.loc[gabor_DU_location & kappa_location].index.to_numpy()
        next_segment_idx = DU_segment_idx + 1

        # final D/U segment may not have an end within the kappa block
        fix_final = False
        if next_segment_idx[-1] > df.loc[kappa_location].index.max():
            next_segment_idx[-1] = next_segment_idx[-1] - 1
            fix_final = True

        grayscreen_rows = (
            df.loc[gabor_DU_location & kappa_location].copy().reset_index(drop=True)
        )
        grayscreen_rows["gabor_frame"] = "G"

        keep_info = ["stimulus_type", "gabor_frame", "gabor_kappa", 
            "gabor_mean_orientation", "unexpected"]
        for column in grayscreen_rows.columns:
            if column not in keep_info:
                grayscreen_rows[column] = -1

        grayscreen_rows["start_frame_stim"] = df.loc[
            DU_segment_idx, "stop_frame_stim"
        ].to_numpy()
        grayscreen_rows["stop_frame_stim"] = df.loc[
            next_segment_idx, "start_frame_stim"
        ].to_numpy()
        grayscreen_rows["num_frames_stim"] = (
            grayscreen_rows["stop_frame_stim"] - grayscreen_rows["start_frame_stim"]
        )

        # check what the stop frame for the final row should be
        if fix_final:
            final_row = len(grayscreen_rows) - 1
            final_start_frame = grayscreen_rows["start_frame_stim"].tolist()[-1]
            mean_num_frames = grayscreen_rows["num_frames_stim"][:-1].max()
            potential_stop_frame = final_start_frame + mean_num_frames
            limits = np.append(df["start_frame_stim"].to_numpy(), num_stimulus_frames)
            closest_limit = limits[np.where((limits - final_start_frame) > 0)[0][0]]
            if potential_stop_frame < closest_limit:
                final_stop_frame = potential_stop_frame
            else:
                final_stop_frame = closest_limit
            grayscreen_rows.loc[final_row, "stop_frame_stim"] = final_stop_frame
            grayscreen_rows.loc[final_row, "num_frames_stim"] = (
                final_stop_frame - final_start_frame
            )

        grayscreen_rows["start_frame_twop"] = stim_align[
            grayscreen_rows["start_frame_stim"]
        ]
        grayscreen_rows["stop_frame_twop"] = stim_align[
            grayscreen_rows["stop_frame_stim"]
        ]
        grayscreen_rows["num_frames_twop"] = (
            grayscreen_rows["stop_frame_twop"] - grayscreen_rows["start_frame_twop"]
        )

        df = (
            df.append(grayscreen_rows)
            .sort_values("start_frame_twop")
            .reset_index(drop=True)
        )

    return df


#############################################
def add_grayscreen_stimulus_rows(df, stim_align):
    """
    add_grayscreen_stimulus_rows(df, stim_align)

    Updates dataframe with grayscreen stimulus rows.

    Arguments:
        df (pandas): stimulus dataframe.
        stim_align (1D array): stimulus to 2p alignment array

    Returns:
        df (pandas): updated stimulus table with grayscreen stimulus rows.
    """

    df = df.copy()

    df = df.sort_values("start_frame_twop").reset_index(drop=True)

    num_stimulus_frames = len(stim_align)

    df_start_frames = np.append(df["start_frame_stim"].to_numpy(), num_stimulus_frames)
    df_stop_frames = np.insert(df["stop_frame_stim"].to_numpy(), 0, 0)
    missing_rows = np.where(df_start_frames != df_stop_frames)[0]

    grayscreen_row = df.loc[0].copy()
    for idx in grayscreen_row.index:
        grayscreen_row[idx] = -1
    grayscreen_row["stimulus_type"] = "grayscreen"

    for missing_row in missing_rows:
        grayscreen_row = grayscreen_row.copy()
        # start_frame obtained from df_stop_frames and v.v.
        start_frame, stop_frame = [
            frames[missing_row] for frames in [df_stop_frames, df_start_frames]
        ]
        if not (stop_frame > start_frame):
            raise NotImplementedError(
                "Unexpected `stop_frame` greater than `start_frame`."
            )

        grayscreen_row["start_frame_stim"] = start_frame
        grayscreen_row["stop_frame_stim"] = stop_frame
        grayscreen_row["num_frames_stim"] = stop_frame - start_frame

        if stop_frame == len(stim_align):
            stop_frame_twop = stim_align[-1] + 1
        else:
            stop_frame_twop = stim_align[stop_frame]

        grayscreen_row["start_frame_twop"] = stim_align[start_frame]
        grayscreen_row["stop_frame_twop"] = stop_frame_twop
        grayscreen_row["num_frames_twop"] = (
            stop_frame_twop - grayscreen_row["start_frame_twop"]
        )

        df = (
            df.append(grayscreen_row)
            .sort_values("start_frame_twop")
            .reset_index(drop=True)
        )

    return df


#############################################
def add_time(df, stim_dict, stimulus_timestamps):
    """
    add_time(df, stim_dict, stimulus_timestamps)

    Updates dataframe with time columns.

    Arguments:
        df (pandas): stimulus dataframe.
        stim_dict (dict): experiment stim dictionary, loaded from pickle
        stimulus_timestamps (1D array): timem stamps for each stimulus frames.

    Returns:
        df (pandas): updated stimulus table with time.
    """

    df = df.copy()

    df = df.sort_values("start_frame_twop").reset_index(drop=True)

    df["start_time_sec"] = stimulus_timestamps[
        df["start_frame_stim"].values.astype(int)
    ]

    non_final = range(0, len(df) - 1)
    df.loc[non_final, "stop_time_sec"] = stimulus_timestamps[
        df["stop_frame_stim"].values[:-1].astype(int)
    ]

    # final line
    final_row = len(df) - 1
    last_duration = (
        df.loc[final_row, "stop_frame_stim"] - 
        df.loc[final_row, "start_frame_stim"]
    ) / stim_dict["fps"]
    df.loc[final_row, "stop_time_sec"] = \
        df.loc[final_row, "start_time_sec"] + last_duration

    df["duration_sec"] = df["stop_time_sec"] - df["start_time_sec"]

    return df


#############################################
def add_stimulus_template_names_and_frames(df, runtype="prod"):
    """
    add_stimulus_template_names_and_frames(df)

    Updates dataframe with stimulus template names and frame numbers.

    Arguments:
        df (pandas): stimulus dataframe.

    Optional arguments:
        runtype (str): deployment type (prod or pilot)
        default: "prod"

    Returns:
        df (pandas): updated stimulus table with stimulus template names and 
                     frame numbers.
    """

    df = df.copy()

    df = df.sort_values("start_frame_twop").reset_index(drop=True)

    df["stimulus_template_name"] = df["stimulus_type"].values

    if runtype == "pilot":
        # visual flow
        stimulus = "visflow"
        for square_size in df["square_size"].unique():
            for direction in df["main_flow_direction"].unique():
                if direction == -1:
                    continue
                loc = (
                    (df["stimulus_type"] == stimulus)
                    & (df["square_size"] == square_size)
                    & (df["main_flow_direction"] == direction)
                )
                direction_short = direction.split(" ")[0]
                df.loc[
                    loc, "stimulus_template_name"
                ] = f"{stimulus}_{direction_short}_size_{int(square_size)}"
        # gabors
        stimulus = "gabors"
        for gabor_kappa in df["gabor_kappa"].unique():
            if gabor_kappa == -1 or np.isnan(gabor_kappa):
                continue
            loc = (df["stimulus_type"] == stimulus) & (df["gabor_kappa"] == gabor_kappa)
            df.loc[
                loc, "stimulus_template_name"
            ] = f"{stimulus}_kappa_{int(gabor_kappa)}"

        # fix gabor G frame information based on preceeding sets
        gabor_grayscreen_rows_loc = (df["stimulus_type"] == stimulus) & (
            df["gabor_frame"] == "G"
        )
        prev_row_idx = df.loc[gabor_grayscreen_rows_loc].index - 1

        df.loc[gabor_grayscreen_rows_loc, "stimulus_template_name"] = df.loc[
            prev_row_idx, "stimulus_template_name"
        ].values

    elif runtype == "prod":
        # visual flow
        stimulus = "visflow"
        for direction in df["main_flow_direction"].unique():
            if direction == -1:
                continue
            loc = (df["stimulus_type"] == stimulus) & (
                df["main_flow_direction"] == direction
            )
            direction_short = direction.split(" ")[0]
            df.loc[loc, "stimulus_template_name"] = \
                f"{stimulus}_{direction_short}"

    else:
        gen_util.accepted_values_error("runtype", runtype, ["prod", "pilot"])

    # stimulus template frame numbers
    for stimulus_template_name in df["stimulus_template_name"].unique():
        loc = df["stimulus_template_name"] == stimulus_template_name
        if stimulus_template_name == "grayscreen":
            df.loc[loc, "start_frame_stim_template"] = 0
        elif "gabors" in stimulus_template_name:
            df.loc[loc, "start_frame_stim_template"] = range(len(df.loc[loc]))
        elif "visflow" in stimulus_template_name:
            start_frames = df.loc[loc, "start_frame_stim"]
            df.loc[loc, "start_frame_stim_template"] = start_frames - start_frames.min()

    return df


#############################################
def load_stimulus_table(stim_dict, stim_sync_h5, time_sync_h5, align_pkl, 
                        sessid, runtype="prod"):
    """
    load_stimulus_table(stim_dict, stim_sync_h5, time_sync_h5, align_pkl, 
                        sessid)

    Retrieves and expands stimulus dataframe.

    Arguments:
        stim_dict (dict)   : experiment stim dictionary, loaded from pickle 
                             (or full path to load it from)
        stim_sync_h5 (Path): full path name of the experiment sync hdf5 file
        time_sync_h5 (Path): full path name of the time synchronization hdf5 
                             file
        align_pkl (Path)   : full path name of the output pickle file to 
                             create
        sessid (int)       : session ID, needed the check whether this 
                             session needs to be treated differently 
                             (e.g., for alignment bugs)

    Optional args:
        runtype (str): runtype ("prod" or "pilot")
                       default: "prod"):

    Returns:
        df (pandas): stimulus table.
        stim_align (1D array): stimulus to 2p alignment array
    """

    # PRE-LOAD EVERYTHING TO AVOID RE-LOADING
    # read the pickle file and call it "pkl"
    if not isinstance(stim_dict, dict):
        stim_dict = file_util.loadfile(stim_dict, filetype="pickle")

    # load dataframe as is (or trigger creation, if it doesn't exist)
    df, stim_align = load_basic_stimulus_table(
        stim_dict, stim_sync_h5, time_sync_h5, align_pkl, sessid, 
        runtype=runtype
    )

    # load stimulus timestamps
    stimulus_timestamps = sess_sync_util.get_stim_fr_timestamps(
        stim_sync_h5, time_sync_h5=time_sync_h5, stim_align=stim_align
        )
    
    # CREATE DATAFRAME
    # load dataframe with updated column names
    df = update_basic_stimulus_table(df)

    # add stimulus frame numbers
    df = add_stimulus_frame_num(df,  stim_dict, stim_align, runtype=runtype)

    # add stimulus names
    df = add_stimulus_locations(df,  stim_dict, runtype=runtype)

    # adjust stimulus segment numbers for each stimulus type
    df = modify_segment_num(df)

    # add gabor grayscreens (G)
    df = add_gabor_grayscreen_rows(df, stim_align)

    # add intervening grayscreen stimuli
    df = add_grayscreen_stimulus_rows(df, stim_align)

    # update time columns
    df = add_time(df, stim_dict, stimulus_timestamps)

    # add stimulus names
    df = add_stimulus_template_names_and_frames(df, runtype=runtype)

    # reorder rows
    df = df.sort_values("start_frame_twop").reset_index(drop=True)

    # verify that no stray -1s are left
    check_for_values = [
        "stimulus_template_name",
        "stimulus_type",
        "start_frame_stim_template",
        "start_frame_stim",
        "stop_frame_stim",
        "num_frames_stim",
        "start_frame_twop",
        "stop_frame_twop",
        "num_frames_twop",
        "start_time_sec",
        "stop_time_sec",
        "duration_sec",
    ]

    for column in check_for_values:
        if len(df.loc[df[column] == -1]) != 0:
            raise NotImplementedError(
                f"Incorrect implemmentation. -1 found in {column} column of "
                "stimulus table."
            )

    # replace -1s with ""/np.nan/[]
    str_dtype_columns = ["gabor_frame", "main_flow_direction"]
    list_dtype_columns = [
        "gabor_orientations",
        "gabor_sizes",
        "gabor_locations_x",
        "gabor_locations_y",
        "square_locations_x",
        "square_locations_y",
    ]
    df[str_dtype_columns] = df[str_dtype_columns].replace([-1, "-1"], "")
    for column in df.columns:
        if column in list_dtype_columns:
            df[column] = df[column].apply(
                lambda d: d if isinstance(d, np.ndarray) else np.asarray([])
            )
        else:
            df[column] = df[column].replace([-1, "-1"], np.nan)

    # drop stimulus_segment
    df = df.drop(columns="orig_stimulus_segment")

    return df, stim_align


#############################################
def add_frames_from_timestamps(df, twop_timestamps, stim_timestamps):
    """
    add_frames_from_timestamps(df, twop_timestamps, stim_timestamps)

    Add stimulus and two-photon frame numbers to the dataframe

    Arguments:
        df (pandas): stimulus table.
        twop_timestamps (1D array): time stamp for each two-photon frame
        stim_timestamps (1D array): time stamp for each stimulus frame
    """

    df = df.copy()

    start_times = df["start_time_sec"].to_numpy()

    start_times = np.append(start_times, df.loc[len(df) - 1, "stop_time_sec"])

    for fr_type, timestamps in zip(
        ["twop", "stim"], [twop_timestamps, stim_timestamps]
        ):

        frame_ns = gen_util.get_closest_idx(timestamps, start_times)

        df[f"start_frame_{fr_type}"] = frame_ns[:-1]
        df[f"stop_frame_{fr_type}"] = frame_ns[1:]
        df[f"num_frames_{fr_type}"] = np.diff(frame_ns)


#############################################
def load_stimulus_table_nwb(sess_files, full_table=True):
    """ 
    load_stimulus_table_nwb(sess_files)
    Retrieves stimulus dataframe.

    Arguments:
        sess_files (Path): full path names of the session files
    
    Returns:
        df (pandas): stimulus table.

    """

    sess_file = sess_file_util.select_nwb_sess_path(sess_files)

    exclude = set() if full_table else set(FULL_TABLE_COLUMNS)
 
    # a bit long
    with pynwb.NWBHDF5IO(str(sess_file), "r") as f:
        nwbfile_in = f.read()
        df = nwbfile_in.trials.to_dataframe(exclude=exclude)
 
    # rename time columns
    df = df.rename(
        columns={"start_time": "start_time_sec", 
         "stop_time": "stop_time_sec"}
    )
    df["duration_sec"] = df["stop_time_sec"] - df["start_time_sec"]

    # add 2p and stimulus frames back in, if necessary
    if "start_frame_stim" not in df.columns:
        twop_timestamps, stim_timestamps = \
            sess_sync_util.get_frame_timestamps_nwb(sess_files)
        df = add_frames_from_timestamps(df, twop_timestamps, stim_timestamps)

    # sort columns
    column_order = [col for col in FINAL_COLUMN_ORDER if col in df.columns]

    df = df[column_order]

    return df

