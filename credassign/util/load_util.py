"""
load_util.py

This module contains loading utility functions.

Authors: Colleen Gillon

Date: February 2023

Note: this code was aggregated from https://github.com/colleenjg/util.

"""

import copy
from pathlib import Path
import warnings

import glob
import json
import numpy as np
import pandas as pd
import pynwb

from credassign.util import gen_util, logger_util


logger = logger_util.get_module_logger(name=__name__)

TAB = "    "

MASK_THRESHOLD = 0.1 # value used in ROI extraction
MIN_N_PIX = 3 # value used in ROI extraction

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

FULL_TABLE_COLUMNS = [
    "gabor_orientations",
    "square_locations_x",
    "square_locations_y",
]



# ADDITIONAL HARD_CODED STIMULUS FEATURES (PRODUCTION ONLY) TO ACCOMPANY NWB DATA
MM_PER_PIXEL = 10.2 / 1000.0

DEG_PER_PIXEL = 0.06251912565744862
EXP_LEN_SEC = [30, 90]
WIN_SIZE = [1920, 1200] # pixels

GABOR_ORI_RANGE = [0, 360]
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
def select_nwb_sess_path(sess_files, ophys=False, behav=False, stim=False, 
                         warn_multiple=False):
    """
    select_nwb_sess_path(sess_files)

    Returns an NWB session data path name, selected from a list according to 
    the specified criteria.
 
    Required arguments:
        - sess_files (list): full path names of the session files

    Optional arguments
        - ophys (bool)        : if True, only session files with optical 
                                physiology data are retained
                                default: False
        - behav (bool)        : if True, only session files with behaviour 
                                data are retained
                                default: False
        - stim (bool)         : if True, only session files with stimulus 
                                images are retained
                                default: False
        - warn_multiple (bool): if True, a warning if thrown if multiple 
                                matching session files are found
                                default: False

    Returns:
        - sess_file (Path): full path name of the selected session file
    """

    if not isinstance(sess_files, list):
        sess_files = [sess_files]
    
    criterion_dict = {
        "ophys"   : [ophys, "optical physiology"],
        "behavior": [behav, "behavioral"],
        "image"   : [stim, "stimulus template"],        
        }

    data_names = []
    for data_str, [data_bool, data_name] in criterion_dict.items():
        if data_bool:
            sess_files = [
                sess_file for sess_file in sess_files 
                if data_str in str(sess_file)
            ]
            data_names.append(data_name)
    
    tog_str = "" if len(data_names) < 2 else " together"
    data_names = ", and ".join(data_names).lower()

    if len(sess_files) == 0:
        raise RuntimeError(
            f"{data_names.capitalize()} data not included{tog_str} in this "
            "session's NWB files."
            )
    
    sess_file = sess_files[0]
    if len(sess_files) > 1 and warn_multiple:
        data_names_str = f" with {data_names} data" if len(data_names) else ""
        warnings.warn(
            f"Several session files{data_names_str} found{tog_str}. "
            f"Using the first listed: {sess_file}."
            )

    return sess_file
    

#############################################
def load_roi_data_nwb(sess_files):
    """
    load_roi_data_nwb(sess_files)

    Returns ROI data from NWB files. 

    Required args:
        - sess_files (Path): full path names of the session files

    Returns:
        - roi_ids (list)   : ROI IDs
        - nrois (int)      : total number of ROIs
        - tot_twop_fr (int): total number of two-photon frames recorded
    """

    ophys_file = select_nwb_sess_path(sess_files, ophys=True)

    with pynwb.NWBHDF5IO(str(ophys_file), "r") as f:
        nwbfile_in = f.read()
        ophys_module = nwbfile_in.get_processing_module("ophys")
        main_field = "ImageSegmentation"
        data_field = "PlaneSegmentation"
        try:
            plane_seg = ophys_module.get_data_interface(
                main_field).get_plane_segmentation(data_field
                )
        except KeyError as err:
            raise KeyError(
                "Could not find plane segmentation data in image segmentation "
                f"for {ophys_file} due to: {err}"
                )

        # ROI IDs
        roi_ids = list(plane_seg["id"].data)

        # Number of ROIs and frames
        main_field = "DfOverF"
        data_field = "RoiResponseSeries"
        try:
            roi_resp_series = ophys_module.get_data_interface(
                main_field).get_roi_response_series(data_field
                )
        except KeyError as err:
            raise KeyError(
                "Could not find ROI response series data in image segmentation "
                f"for {ophys_file} due to: {err}"
                )

        tot_twop_fr, nrois = roi_resp_series.data.shape


    return roi_ids, nrois, tot_twop_fr
    
    
#############################################
def get_nwb_sess_paths(maindir, sess_id, mouseid=None):
    """
    get_nwb_sess_paths(maindir, sess_id)

    Returns a list of NWB session data path names for the DANDI Credit 
    Assignment session requested.

    Several files may be found if they contain different types of information 
    (e.g., behavior, image, ophys).
 
    Required arguments:
        - maindir (str): path of the main data directory
        - sess_id (str): session ID on Dandi

    Optional arguments
        - mouseid (str) : mouse 6-digit ID string optionally used to check 
                          whether files are for the expected mouse number
                          e.g. "389778"

    Returns:
        - sess_files (list): full path names of the session files
    """

    
    dandi_form = f"*ses-{sess_id}*.nwb"
    if mouseid is not None:
        dandi_form = f"sub-{mouseid}_{dandi_form}"
    dandi_glob_path = Path(maindir, "**", dandi_form)
    sess_files = sorted(glob.glob(str(dandi_glob_path), recursive=True))

    if len(sess_files) == 0:
        raise RuntimeError(
            "Found no NWB sessions of the expected form "
            f"{dandi_form} under {maindir}."
            )

    else:
        sess_files = [Path(sess_file) for sess_file in sess_files]
        return sess_files
        
        

    
#############################################
def load_max_projection_nwb(sess_files):
    """
    load_max_projection_nwb(sess_files)

    Returns maximum projection image of downsampled z-stack as an array, from 
    NWB files. 

    Required args:
        - sess_files (Path): full path names of the session files

    Returns:
        - max_proj (2D array): maximum projection image across downsampled 
                               z-stack (hei x wei), with pixel intensity 
                               in 0 (incl) to 256 (excl) range 
                               ("uint8" datatype).
    """

    ophys_file = select_nwb_sess_path(sess_files, ophys=True)

    with pynwb.NWBHDF5IO(str(ophys_file), "r") as f:
        nwbfile_in = f.read()
        ophys_module = nwbfile_in.get_processing_module("ophys")
        main_field = "PlaneImages"
        data_field = "max_projection"
        try:
            max_proj = ophys_module.get_data_interface(
                main_field).get_image(data_field)[()].astype("uint8")
        except KeyError as err:
            raise KeyError(
                "Could not find a maximum projection plane image "
                f"for {ophys_file} due to: {err}"
                )

    return max_proj


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
def get_frame_timestamps_nwb(sess_files):
    """
    get_frame_timestamps_nwb(sess_files)

    Returns time stamps for stimulus and two-photon frames.

    Required args:
        - sess_files (Path): full path names of the session files

    Returns:
        - stim_timestamps (1D array): time stamp for each stimulus frame
        - twop_timestamps (1D array): time stamp for each two-photon frame
    """

    behav_file = select_nwb_sess_path(sess_files, behav=True)

    use_ophys = False
    with pynwb.NWBHDF5IO(str(behav_file), "r") as f:
        nwbfile_in = f.read()
        behav_module = nwbfile_in.get_processing_module("behavior")
        main_field = "BehavioralTimeSeries"
        data_field = "running_velocity"
        try:
            run_time_series = behav_module.get_data_interface(
                main_field).get_timeseries(data_field)
        except KeyError as err:
            raise KeyError(
                "Could not find running velocity data in behavioral time "
                f"series for {behav_module} due to: {err}"
                )

        stim_timestamps = np.asarray(run_time_series.timestamps)

        main_field = "PupilTracking"
        data_field = "pupil_diameter"
        try:
            twop_timestamps = np.asarray(behav_module.get_data_interface(
                main_field).get_timeseries(data_field).timestamps)
        except KeyError as err:
            use_ophys = True            

    # if timestamps weren't found with pupil data, look for optical physiology 
    # data
    if use_ophys:
        ophys_file = select_nwb_sess_path(sess_files, ophys=True)

        with pynwb.NWBHDF5IO(str(ophys_file), "r") as f:
            nwbfile_in = f.read()
            ophys_module = nwbfile_in.get_processing_module("ophys")
            main_field = "DfOverF"
            data_field = "RoiResponseSeries"
            try:
                roi_resp_series = ophys_module.get_data_interface(
                    main_field).get_roi_response_series(data_field
                    )
            except KeyError:
                file_str = f"{behav_file} or {ophys_file}"
                if behav_file == ophys_file:
                    file_str = behav_file
                raise OSError(
                    "Two-photon timestamps cannot be collected, as no "
                    f"pupil or ROI series data was found in {file_str}."
                    )
            twop_timestamps = roi_resp_series.timestamps
    

    return stim_timestamps, twop_timestamps
    
    
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

    sess_file = select_nwb_sess_path(sess_files)

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
            get_frame_timestamps_nwb(sess_files)
        df = add_frames_from_timestamps(df, twop_timestamps, stim_timestamps)

    # sort columns
    column_order = [col for col in FINAL_COLUMN_ORDER if col in df.columns]

    df = df[column_order]

    return df


#############################################
def get_registration_transform_params(nway_match_path, sessid, targ_sess_idx=0):
    """
    get_registration_transform_params(nway_match_path, sessid)

    Returns cv2.warpPerspective registration transform parameters used to 
    register session planes to one another, saved in the n-way match files. 

    (cv2.warpPerspective should be used with flags cv2.INTER_LINEAR and 
    cv2.WARP_INVERSE_MAP)

    Required args:
        - nway_match_path (Path): full path name of the n-way registration path 
                                  (should be a local path in a directory that 
                                  other session registrations are also stored)
        - sessid (int)          : session ID

    Optional args:
        - targ_sess_idx (int): session that the registration transform should 
                               be targetted to
                               default: 0  

    Returns:
        - transform_params (3D array): registration transformation parameters 
                                       for the session (None if the session was 
                                       the registration target)
    """

    if not Path(nway_match_path).is_file():
        raise OSError(f"{nway_match_path} does not exist.")

    with open(nway_match_path, "r") as f:
        nway_metadata = pd.DataFrame().from_dict(json.load(f)["metadata"])

    if len(nway_metadata) != 1:
        raise NotImplementedError(
            "Metadata dataframe expected to only have one line."
            )
    nway_row = nway_metadata.loc[nway_metadata.index[0]]
    if sessid not in nway_row["sess_ids"]:
        raise RuntimeError(
            "sessid not found in the n-way match metadata dataframe."
            )
    sess_idx = nway_row["sess_ids"].index(sessid)
    sess_n = nway_row["sess_ns"][sess_idx]
    n_reg_sess = len(nway_row["sess_ids"])
    if targ_sess_idx >= n_reg_sess:
        raise ValueError(
            f"targ_sess_idx is {targ_sess_idx}, but only {n_reg_sess} "
            "sessions were registered to one another.")

    targ_sess_id = nway_row["sess_ids"][targ_sess_idx]
    targ_sess_n = nway_row["sess_ns"][targ_sess_idx]
    if targ_sess_id == sessid:
        return None # no transform needed

    # get transform from the target session's nway file
    if str(sessid) not in str(nway_match_path):
        raise ValueError(
            "Expected the n-way_match_path to contain the session ID."
            )
    targ_nway_match_path = Path(
        str(nway_match_path).replace(str(sessid), str(targ_sess_id))
        )

    if not Path(targ_nway_match_path).is_file():
        raise RuntimeError(f"Expected to find {targ_nway_match_path} to "
        "retrieve registration transform, but file does not exist.")

    with open(targ_nway_match_path, "r") as f:
        target_nway_metadata = pd.DataFrame().from_dict(json.load(f)["metadata"])

    column_name = f"sess_{sess_n}_to_sess_{targ_sess_n}_transformation_matrix"

    if column_name not in target_nway_metadata.columns:
        raise RuntimeError(
            f"Expected to find {column_name} column in the metadata "
            "dataframe for the target session."
            )

    if len(nway_metadata) != 1:
        raise NotImplementedError(
            "Target session metadata dataframe expected to only have one line."
            )

    target_nway_row = target_nway_metadata.loc[target_nway_metadata.index[0]]
    
    transform_params = np.asarray(target_nway_row[column_name])
        
    return transform_params


#############################################
def apply_registration_transform(nway_match_path, sessid, image, 
                                 targ_sess_idx=0):
    """
    apply_registration_transform(nway_match_path, sessid, image)

    Returns an image transformed using registration transform parameters, saved 
    in the n-way match files. 

    Required args:
        - nway_match_path (Path): full path name of the n-way registration path 
                                  (should be a local path in a directory that 
                                  other session registrations are also stored)
        - sessid (int)          : session ID
        - image (2 or 3D array) : image to transform, with dimensions 
                                  (item x) hei x wid. Only certain datatypes 
                                  are supported by the OpenCV function used, 
                                  e.g. float or uint8.
         
    Optional args:
        - targ_sess_idx (int)   : session that the registration transform 
                                  should be targetted to
                                  default: 0
    
    Returns:
        - registered_image (2 or 3D array) : transformed image, with dimensions 
                                             (item x) hei x wid.
    """

    import cv2

    registration_transform_params = get_registration_transform_params(
        nway_match_path, sessid, targ_sess_idx=targ_sess_idx
        )

    if registration_transform_params is None:
        registered_image = image
    else:
        len_image_shape = len(image.shape)
        if len_image_shape == 2:
            image = np.asarray([image])
        elif len_image_shape != 3:
            raise ValueError("image must be a 2D or 3D array.")

        image = np.asarray(image)

        if registration_transform_params.shape != (3, 3):
            raise RuntimeError(
                "registration_transform_params retrieved is expected to have "
                "shape (3, 3), but found shape "
                f"{registration_transform_params.shape}."
                )

        try:
            registered_image = np.asarray(
                [cv2.warpPerspective(
                    sub, 
                    registration_transform_params, 
                    dsize=sub.shape, 
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                    )
                    for sub in image]
                )
        except cv2.error as err:
            # try to capture and clarify obscure datatype errors raised by 
            # OpenCV
            if "ifunc" in str(err) or "argument 'src'" in str(err):
                raise RuntimeError(
                    "The following error was raised by OpenCV during image "
                    f"warping: {err}May be due to the use of an unsupported "
                    "datatype. Supported datatypes include uint8, int16, "
                    "uint16, float32, and float64."
                    )
            else:
                raise err

        if len_image_shape == 2:
            registered_image = registered_image[0]

    return registered_image


#############################################
def get_tracked_rois(nway_match_path, idx_after_rem_bad=False):
    """
    get_tracked_rois(nway_match_path)

    Returns ROI tracking indices.

    Required args:
        - nway_match_path (Path): path to nway matching file
        
    Optional args:
        - idx_after_rem_bad (bool): if True, the ROI indices are shifted to 
                                    as if bad ROIs did not exist
                                    (bad ROIs computed for dF/F only)
                                    default: False

    Returns:
        - tracked_rois (1D array): ordered indices of tracked ROIs
    """

    with open(nway_match_path, "r") as f:
        nway_dict = json.load(f)

    tracked_rois_df = pd.DataFrame().from_dict(nway_dict["rois"])
    tracked_rois = tracked_rois_df["dff-ordered_roi_index"].values
    
    if idx_after_rem_bad:
        bad_rois_df = pd.DataFrame().from_dict(nway_dict["bad_rois"])
        bad_rois = bad_rois_df["dff_local_bad_roi_idx"].values

        # shift indices to ignore bad ROIs
        adj_tracked_rois = []
        for n in tracked_rois:
            adj_tracked_rois.append(n - np.sum(bad_rois < n))
        tracked_rois = np.asarray(adj_tracked_rois)

    return tracked_rois
    
    
#############################################
def get_tracked_rois_nwb(sess_files):
    """
    get_tracked_rois_nwb(sess_files)

    Returns ROI tracking indices.

    Required args:
        - sess_files (list): full path names of the session files
        
    Returns:
        - tracked_rois (1D array): ordered indices of tracked ROIs
    """

    ophys_file = select_nwb_sess_path(sess_files, ophys=True)

    with pynwb.NWBHDF5IO(str(ophys_file), "r") as f:
        nwbfile_in = f.read()
        ophys_module = nwbfile_in.get_processing_module("ophys")
        main_field = "ImageSegmentation"
        data_field = "PlaneSegmentation"
        try:
            plane_seg = ophys_module.get_data_interface(
                main_field).get_plane_segmentation(data_field
                )
        except KeyError as err:
            raise KeyError(
                "Could not find plane segmentation data in image segmentation "
                f"for {ophys_file} due to: {err}"
                )

        tracking_key = "tracking_id"
        if tracking_key not in plane_seg.colnames:
            raise RuntimeError(f"No tracking data in {ophys_file}.")

        # tracking index for each ROI (np.nan for non tracked ROIs)
        roi_tracking = np.asarray(plane_seg[tracking_key].data)

    tracked_idxs = np.where(np.isfinite(roi_tracking))[0]
    tracking_order = np.argsort(roi_tracking[tracked_idxs])

    # ordered indices of tracked ROIs
    tracked_rois = tracked_idxs[tracking_order]

    return tracked_rois


#############################################
def process_roi_masks(roi_masks, mask_threshold=MASK_THRESHOLD, 
                      min_n_pix=MIN_N_PIX, make_bool=True):
    """
    process_roi_masks(roi_masks)

    Processes ROI masks, setting to 0 those that do not meet the input 
    criteria.

    Required args:
        - roi_masks (3D): ROI masks (ROI x hei x wid)

    Optional args:
        - mask_threshold (float): minimum value in non-boolean mask to
                                  retain a pixel in an ROI mask
                                  default: MASK_THRESHOLD
        - min_n_pix (int)       : minimum number of pixels in an ROI below 
                                  which, ROI is set to be empty (calculated 
                                  before conversion to boolean, using weighted 
                                  pixels)
                                  default: MIN_N_PIX
        - make_bool (bool)      : if True, ROIs are converted to boolean 
                                  before being returned
                                  default: True 

    Returns:
        - roi_masks (3D): processed ROI masks (ROI x hei x wid)
    """

    roi_masks = copy.deepcopy(roi_masks)

    if len(roi_masks.shape) != 3:
        raise ValueError("roi_masks should have 3 dimensions.")

    roi_masks[roi_masks < mask_threshold] = 0

    if min_n_pix != 0:
        set_empty = np.sum(roi_masks, axis=(1, 2)) < min_n_pix
        roi_masks[set_empty] = 0

    if make_bool:
        roi_masks = roi_masks.astype(bool)
    
    return roi_masks


#############################################
def get_roi_masks_nwb(sess_files, mask_threshold=MASK_THRESHOLD, 
                      min_n_pix=MIN_N_PIX, make_bool=True):
    """
    get_roi_masks_nwb(sess_files)

    Returns tracked ROIs, optionally converted to boolean.

    Required args:
        - sess_files (Path): full path names of the session files

    Optional args:
        - mask_threshold (float): minimum value in non-boolean mask to
                                  retain a pixel in an ROI mask
                                  default: MASK_THRESHOLD
        - min_n_pix (int)       : minimum number of pixels in an ROI below 
                                  which, ROI is set to be empty (calculated 
                                  before conversion to boolean, using weighted 
                                  pixels)
                                  default: MIN_N_PIX
        - make_bool (bool)      : if True, ROIs are converted to boolean 
                                  before being returned
                                  default: True 
        
    Returns:
        - roi_masks (3D array): ROI masks, structured as 
                                ROI x height x width
        - roi_ids (list)      : ID for each ROI
    """

    ophys_file = select_nwb_sess_path(sess_files, ophys=True)

    with pynwb.NWBHDF5IO(str(ophys_file), "r") as f:
        nwbfile_in = f.read()
        ophys_module = nwbfile_in.get_processing_module("ophys")
        main_field = "ImageSegmentation"
        data_field = "PlaneSegmentation"
        try:
            plane_seg = ophys_module.get_data_interface(
                main_field).get_plane_segmentation(data_field
                )
        except KeyError as err:
            raise KeyError(
                "Could not find plane segmentation data in image segmentation "
                f"for {ophys_file} due to: {err}"
                )

        roi_masks = np.asarray(plane_seg["image_mask"].data)
        roi_ids = list(plane_seg["id"].data)

    roi_masks = process_roi_masks(
        roi_masks, mask_threshold=mask_threshold, min_n_pix=min_n_pix, 
        make_bool=make_bool
        )

    return roi_masks, roi_ids


#############################################
def load_traces_optimally(roi_data_handle, roi_ns=None, frame_ns=None, 
                          rois_first=True):
    """
    load_traces_optimally(roi_data_handle)

    Updates indices, possibly reordered, for optimal loading of ROI traces.

    Optional args:
        - roi_ns (int or array-like)  : ROIs to load (None for all)
                                        default: None
        - frame_ns (int or array-like): frames to load (None for all) 
                                        default: None
        - rois_first (bool)           : if True, ROIs are stored as 
                                        ROIs x frames, else frames x ROIs
                                        default: True

    Returns:
        - roi_traces (1 or 2D array): ROI traces (ROI x frame)
    """

    # if no ROIs are specified
    if roi_ns is None and frame_ns is None:
        roi_traces = roi_data_handle[()]


    # if ROI_ns is an int
    elif isinstance(roi_ns, int):
        if rois_first:
            roi_traces = roi_data_handle[roi_ns]
        else:
            roi_traces = roi_data_handle[:, roi_ns]
        if frame_ns is not None:
            roi_traces = roi_traces[frame_ns]
        

    # if frame_ns is an int
    elif isinstance(frame_ns, int):
        if rois_first:
            roi_traces = roi_data_handle[:, frame_ns]
        else:
            roi_traces = roi_data_handle[frame_ns]
        if roi_ns is not None:
            roi_traces = roi_traces[roi_ns]


    # if both are vectors, if possible, load frames, then select ROIs
    elif frame_ns is not None and (len(np.unique(frame_ns)) == len(frame_ns)):
        if roi_ns is None:
            roi_ns = slice(None, None, None)
        frame_ns = np.asarray(frame_ns)
        resort = None
        if (np.sort(frame_ns) != frame_ns).any(): # sort if not sorted
            resort = np.argsort(np.argsort(frame_ns))
            frame_ns = np.sort(frame_ns)
        if rois_first:
            roi_traces = roi_data_handle[:, frame_ns][roi_ns]
            if resort is not None:
                roi_traces = roi_traces[:, resort]
        else:
            roi_traces = roi_data_handle[frame_ns][..., roi_ns]
            if resort is not None:
                roi_traces = roi_traces[resort]


    # alternatively, if possible, load ROIs, then select frames    
    elif roi_ns is not None and len(np.unique(roi_ns)) == len(roi_ns):
        if frame_ns is None:
            frame_ns = slice(None, None, None)
        roi_ns = np.asarray(roi_ns)
        resort = None
        if (np.sort(roi_ns) != roi_ns).any(): # sort if not sorted
            resort = np.argsort(np.argsort(roi_ns))
            roi_ns = np.sort(roi_ns)
        if rois_first:
            roi_traces = roi_data_handle[roi_ns][:, frame_ns]
            if resort is not None:
                roi_traces = roi_traces[resort]
        else:
            roi_traces = roi_data_handle[:, roi_ns][frame_ns]
            if resort is not None:
                roi_traces = roi_traces[:, resort]


    # load fully and select
    else:
        if roi_ns is None:
            roi_ns = slice(None, None, None)
        if frame_ns is None:
            frame_ns = slice(None, None, None)
        if rois_first:
            roi_traces = roi_data_handle[()][roi_ns][:, frame_ns]
        else:
            roi_traces = roi_data_handle[()][:, roi_ns][frame_ns]


    if not rois_first:
        roi_traces = roi_traces.T

    return roi_traces


#############################################
def load_roi_traces_nwb(sess_files, roi_ns=None, frame_ns=None):
    """
    load_roi_traces_nwb(sess_files)

    Returns ROI traces from NWB files (stored as frames x ROIs). 

    Required args:
        - sess_files (list): full path names of the session files

    Optional args:
        - roi_ns (int or array-like)  : ROIs to load (None for all)
                                        default: None
        - frame_ns (int or array-like): frames to load (None for all) 
                                        default: None

    Returns:
        - roi_traces (1 or 2D array): ROI traces (ROI x frame)
    """

    ophys_file = select_nwb_sess_path(sess_files, ophys=True)

    with pynwb.NWBHDF5IO(str(ophys_file), "r") as f:
        nwbfile_in = f.read()
        ophys_module = nwbfile_in.get_processing_module("ophys")
        main_field = "DfOverF"
        data_field = "RoiResponseSeries"
        try:
            roi_resp_series = ophys_module.get_data_interface(
                main_field).get_roi_response_series(data_field
                )
        except KeyError as err:
            raise KeyError(
                "Could not find ROI response series data in image segmentation "
                f"for {ophys_file} due to: {err}"
                )

        roi_data_handle = roi_resp_series.data

        roi_traces = load_traces_optimally(
            roi_data_handle, roi_ns=roi_ns, frame_ns=frame_ns, 
            rois_first=False,
            )


    return roi_traces
    
    
#############################################
def _warn_nans_diff_thr(run, min_consec=5, n_pre_existing=None, sessid=None):
    """
    _warn_nans_diff_thr(run)

    Checks for NaNs in running velocity, and logs a warning about the total 
    number of NaNs, and the consecutive NaNs. Optionally indicates the number 
    of pre-existing NaNs, versus number of NaNs resulting from the difference 
    threshold. 

    Required args:
        - run (1D array): array of running velocities in cm/s

    Optional args:
        - min_consec (num)    : minimum number of consecutive NaN running 
                                values to warn aboout
                                default: 5
        - n_pre_existing (num): number of pre-existing NaNs (before difference 
                                thresholding was used)
                                default: None
        - sessid (int)        : session ID to include in the log or error
                                default: None 
    """

    n_nans = np.sum(np.isnan(run))

    if n_nans == 0:
        return

    split_str = ""
    if n_pre_existing is not None:
        if n_pre_existing == n_nans:
            split_str = " (in pre-processing)"
        elif n_pre_existing == 0:
            split_str = " (using diff thresh)"
        else:
            split_str = (f" ({n_pre_existing} in pre-processing, "
                f"{n_nans - n_pre_existing} more using diff thresh)")

    mask = np.concatenate(([False], np.isnan(run), [False]))
    idx = np.nonzero(mask[1 : ] != mask[ : -1])[0]
    n_consec = np.sort(idx[1 :: 2] - idx[ :: 2])[::-1]

    n_consec_above_min_idx = np.where(n_consec > min_consec)[0]
    
    n_consec_str = ""
    if len(n_consec_above_min_idx) > 0:
        n_consec_str = ", ".join(
            [str(n) for n in n_consec[n_consec_above_min_idx]])
        n_consec_str = (f"\n{TAB}This includes {n_consec_str} consecutive "
            "dropped running values.")

    prop = n_nans / len(run)
    sessstr = "" if sessid is None else f"Session {sessid}: "
    
    logger.warning(f"{sessstr}{n_nans} dropped running frames "
        f"(~{prop * 100:.1f}%){split_str}.{n_consec_str}", 
        extra={"spacing": TAB})

    return


#############################################
def nan_large_run_differences(run, diff_thr=50, warn_nans=True, 
                              drop_tol=0.0003, sessid=None):
    """
    nan_large_run_differences(run)

    Returns running velocity with outliers replaced with NaNs.

    Required args:
        - run (1D array): array of running velocities in cm/s

    Optional args:
        - diff_thr (int)    : threshold of difference in running velocity to 
                              identify outliers
                              default: 50
        - warn_nans (bool)  : if True, a warning is logged 
                              default: True
        - drop_tol (num)    : the tolerance for proportion running frames 
                              dropped. A warning is produced only if this 
                              condition is not met. 
                              default: 0.0003 
        - sessid (int)      : session ID to include in the log or error
                              default: None 

    Returns:
        - run (1D array): updated array of running velocities in cm/s
    """

    # temorarily remove preexisting NaNs (to be reinserted after)
    original_length = len(run)
    not_nans_idx = np.where(~np.isnan(run))[0]
    run = run[not_nans_idx]
    n_pre_existing = original_length - len(run)

    run_diff = np.diff(run)
    out_idx = np.where((run_diff < -diff_thr) | (run_diff > diff_thr))[0]
    at_idx = -1
    for idx in out_idx:
        if idx > at_idx:
            if idx == 0:
                # in case the first value is completely off
                comp_val = 0
                if np.absolute(run[0]) > diff_thr:
                    run[0] = np.nan
            else:
                comp_val = run[idx]
            while np.absolute(run[idx + 1] - comp_val) > diff_thr:
                run[idx + 1] = np.nan
                idx += 1
            at_idx = idx

    # reinsert pre-existing NaNs
    prev_run = copy.deepcopy(run)
    run = np.full(original_length, np.nan)
    run[not_nans_idx] = prev_run

    prop_nans = np.sum(np.isnan(run)) / len(run)
    if warn_nans and prop_nans > drop_tol:
        _warn_nans_diff_thr(
            run, min_consec=5, n_pre_existing=n_pre_existing, sessid=sessid
            )

    return run


#############################################
def load_run_data_nwb(sess_files, diff_thr=50, drop_tol=0.0003, sessid=None):
    """
    load_run_data_nwb(sess_files)

    Returns pre-processed running velocity from NWB files. 

    Required args:
        - sess_files (Path): full path names of the session files

    Optional args:
        - diff_thr (int): threshold of difference in running velocity to 
                          identify outliers
                          default: 50
        - drop_tol (num): the tolerance for proportion running frames 
                          dropped. A warning is produced only if this 
                          condition is not met. 
                          default: 0.0003 
        - sessid (int)  : session ID to include in the log or error
                          default: None 
    Returns:
        - run_velocity (1D array): array of running velocities in cm/s for each 
                                   recorded stimulus frames

    """

    behav_file = select_nwb_sess_path(sess_files, behav=True)

    with pynwb.NWBHDF5IO(str(behav_file), "r") as f:
        nwbfile_in = f.read()
        behav_module = nwbfile_in.get_processing_module("behavior")
        main_field = "BehavioralTimeSeries"
        data_field = "running_velocity"
        try:
            behav_time_series = behav_module.get_data_interface(
                main_field).get_timeseries(data_field)
        except KeyError as err:
            raise KeyError(
                "Could not find running velocity data in behavioral time "
                f"series for {behav_module} due to: {err}"
                )
        
        run_velocity = np.asarray(behav_time_series.data)

    run_velocity = nan_large_run_differences(
        run_velocity, diff_thr, warn_nans=True, drop_tol=drop_tol, 
        sessid=sessid
        )

    return run_velocity


#############################################
def get_center_dist_diff(center_x, center_y):
    """
    get_center_dist_diff(center_x, center_y)
    Returns the change in pupil center between each pupil frame. All in pixels.
    Required args:
        - center_x (1D array): pupil center position in x at each pupil frame 
        - center_y (1D array): pupil center position in y at each pupil frame
    Returns:
        - center_dist_diff (1D array): change in pupil center between each 
                                       pupil frame
    """

    center = np.stack([center_x, center_y])
    center_diff = np.diff(center, axis=1)
    center_dist_diff = np.sqrt(center_diff[0]**2 + center_diff[1]**2)

    return center_dist_diff


#############################################
def load_pup_data_nwb(sess_files):
    """
    load_pup_data_nwb(sess_files)

    Returns pre-processed pupil data from NWB files. 

    Required args:
        - sess_files (Path): full path names of the session files

    Returns:
        - pup_data_df (pd DataFrame): pupil data dataframe with columns:
            - frames (int)        : frame number
            - pup_diam (float)    : median pupil diameter in pixels
            if found in NWB file:
            - pup_center_x (float): pupil center position for x at 
                                    each pupil frame in pixels
            - pup_center_y (float): pupil center position for y at 
                                    each pupil frame in pixels
    """

    behav_file = select_nwb_sess_path(sess_files, behav=True)

    srcs = ["pupil_diameter", "pupil_position_x", "pupil_position_y"]
    targs = ["pup_diam", "pup_center_x", "pup_center_y"]

    pup_data_df = pd.DataFrame()
    with pynwb.NWBHDF5IO(str(behav_file), "r") as f:
        nwbfile_in = f.read()
        behav_module = nwbfile_in.get_processing_module("behavior")
        main_field = "PupilTracking"
        pupil_series = behav_module.get_data_interface(
            main_field).fields["time_series"].keys()
        if not "pupil_position_x" in pupil_series:
            srcs = ["pupil_diameter"]
            targs = ["pup_diam"]
            warnings.warn(
                "NWB file does not include pupil center position data. "
                "Likely an older version of the data file."
                )
        for src, targ in zip(srcs, targs):
            try:
                behav_time_series = behav_module.get_data_interface(
                    main_field).get_timeseries(src)
            except KeyError as err:
                raise KeyError(
                    f"Could not find {src.replace('_', '')} data in behavioral "
                    f"time series for {behav_module} due to: {err}"
                    )

            pup_data = np.asarray(behav_time_series.data)
            pup_data_df[targ] = pup_data / MM_PER_PIXEL

    pup_data_df.insert(0, "frames", value=range(len(pup_data_df)))

    return pup_data_df


#############################################
def get_local_nway_match_path_from_sessid(sessid):
    """
    get_local_nway_match_path_from_sessid(sessid)

    Returns the full path name for the nway match file stored in the repository 
    main directory for the specified session.

    Required args:
        - sessid (int)  : session ID

    Returns:
        - nway_match_path (path): n-way match path
    """

    tracking_dir = Path(Path(__file__).resolve().parent.parent, "roi_tracking")

    if tracking_dir.exists():
        nway_path_pattern = Path(
            tracking_dir, "**", f"*session_{sessid}__nway_matched_rois.json"
            )
        matching_files = glob.glob(str(nway_path_pattern), recursive=True)
        if len(matching_files) == 0:
            raise RuntimeError(
                f"Found no local nway match file for session {sessid} in "
                f"{tracking_dir}."
                )
        elif len(matching_files) > 1:
            raise NotImplementedError(
                f"Found multiple local nway match files for session {sessid} "
                f"in {tracking_dir}."
                )
        else:
            nway_match_path =  Path(matching_files[0])
    else:
        raise RuntimeError(
            "Expected to find the 'roi_tracking' directory in the main "
            f"repository folder: {tracking_dir}"
            )

    return nway_match_path
    
    
#############################################
def get_tracking_perm_example_df(nway_match_path, sessid=None, 
                                 idx_after_rem_bad=False):
    """
    get_tracking_perm_example_df(nway_match_path)

    Returns dataframe with tracking permutation example data. (Only a few mice 
    have this data included in their nway-match files.)

    Required args:
        - nway_match_path (Path): full path name of the n-way registration path 
                                  (should be a local path in a directory that 
                                  other session registrations are also stored)         
    Optional args:
        - sessid (int)            : session ID, for error message if file does 
                                    not contain the tracking permutation 
                                    example key
                                    default: None
        - idx_after_rem_bad (bool): if True, the ROI indices (not IDs, however) 
                                    are shifted to as if bad ROIs did not exist
                                    (bad ROIs computed for dF/F only)
                                    default: False

    Returns:
        - nway_tracking_ex_df (pd.DataFrame): dataframe listing ROI tracking 
            matches that were yielded using different session permutations, 
            with columns:
            ['match_level'] (str): 
                whether this permutation produces the 'most' or 'fewest' 
                matches, or whether the row reflects the 'union'
            ['n_total'] (int):
                total number of ROIs for the match level

            if 'match_level' is 'most' or 'fewest' (NaN if 'union')
            ['sess_order'] (list):
                session number order for this permutation
            ['dff_local_missing_roi_idx'] (list):
                indices of ROIs that are included in the final tracked ROIs for 
                the session, but were not identified with this permutation
            ['dff_local_extra_roi_idx'] (list):
                indices of ROIs that are not included in the final tracked ROIs 
                for the session, but were identified with this permutation
            ['sess{}_missing_roi_id'] (list):
                ROI IDs/names corresponding to 'dff_local_missing_roi_idx'
            ['sess{}_extra_roi_id'] (list):
                ROI IDs/names corresponding to 'dff_local_extra_roi_idx'
    """

    if not Path(nway_match_path).is_file():
        raise OSError(f"{nway_match_path} does not exist.")

    with open(nway_match_path, "r") as f:
        nway_dict = json.load(f)
    
    match_key = "match_perm_examples"
    if match_key not in nway_dict.keys():
        sess_str = "" if sessid is None else f" for session {sessid}"
        raise RuntimeError(f"nway-match file{sess_str} does not contain "
            f"example tracking permutation data under {match_key}."
            )
    
    nway_tracking_ex_df = pd.DataFrame().from_dict(nway_dict[match_key])

    # check that missing ROI indices are all tracked, and extra ROI indices are 
    # all untracked for the session
    rois_df = pd.DataFrame().from_dict(nway_dict["rois"])
    for col in nway_tracking_ex_df.columns:
        if "roi_idx" not in col:
            continue
        targ_vals = rois_df["dff-ordered_roi_index"].tolist()
        for row_idx in nway_tracking_ex_df.index:
            if nway_tracking_ex_df.loc[row_idx, "match_level"] == "union":
                continue

            roi_idxs = nway_tracking_ex_df.loc[row_idx, col]
            for n in roi_idxs:
                if n in targ_vals and "extra" in col:
                    raise RuntimeError(
                        "Some ROIs identified as 'extra' are in fact tracked."
                        )
                elif n not in targ_vals and "missing" in col:
                    raise RuntimeError(
                        "Some ROIs identified as 'missing' are not in fact "
                        "tracked."
                        )

    # shift ROI indices to as if bad ROIs did not exist
    if idx_after_rem_bad:
        bad_rois_df = pd.DataFrame().from_dict(nway_dict["bad_rois"])
        bad_rois = bad_rois_df["dff_local_bad_roi_idx"].values

        idx_cols = ["dff_local_missing_roi_idx", "dff_local_extra_roi_idx"]
        for row_idx in nway_tracking_ex_df.index:
            if nway_tracking_ex_df.loc[row_idx, "match_level"] == "union":
                continue
            for col in idx_cols:
                roi_idxs = nway_tracking_ex_df.loc[row_idx, col]
                if len(roi_idxs) == 0:
                    continue
                
                # shift indices to ignore bad ROIs
                adj_roi_idxs = []
                for n in roi_idxs:
                    adj_roi_idxs.append(n - np.sum(bad_rois < n))
                nway_tracking_ex_df.at[row_idx, col] = adj_roi_idxs

    return nway_tracking_ex_df

