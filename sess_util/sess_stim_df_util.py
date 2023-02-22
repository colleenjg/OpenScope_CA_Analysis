"""
sess_stim_df_util.py

This module contains the functions needed to generate the stimulus dataframe
for Allen Institute OpenScope experiments for the Credit Assignment Project.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import numpy as np
import pynwb

from util import gen_util, logger_util
from sess_util import sess_file_util, sess_load_util


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
            sess_load_util.get_frame_timestamps_nwb(sess_files)
        df = add_frames_from_timestamps(df, twop_timestamps, stim_timestamps)

    # sort columns
    column_order = [col for col in FINAL_COLUMN_ORDER if col in df.columns]

    df = df[column_order]

    return df

