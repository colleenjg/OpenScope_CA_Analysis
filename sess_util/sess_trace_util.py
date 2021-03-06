"""
sess_trace_util.py

This module contains functions for processing ROI trace data from files 
generated by the Allen Institute OpenScope experiments for the Credit 
Assignment Project.

Authors: Colleen Gillon

Date: August, 2018

Note: this code uses python 3.7.

"""

import copy
import logging
import os

import h5py
import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sparse
import scipy.linalg as linalg
from allensdk.brain_observatory import dff, r_neuropil, roi_masks, demixer
from allensdk.internal.brain_observatory import mask_set

from util import file_util, gen_util, logger_util
from sess_util import sess_file_util

logger = logging.getLogger(__name__)

EXCLUSION_LABELS = ["motion_border", "union", "duplicate", "empty", 
    "empty_neuropil"]


#############################################
def get_roi_locations(roi_extract_dict):
    """
    get_roi_locations(roi_extract_dict)

    Returns ROI locations, extracted from ROI extraction dictionary.

    Required args:
        - roi_extract_dict (dict): ROI extraction dictionary

    Returns 
        - roi_locations (pd DataFrame): ROI locations dataframe
    """

    if not isinstance(roi_extract_dict, dict):
        roi_extract_dict = file_util.loadfile(roi_extract_dict)

    # get data out of json and into dataframe
    rois = roi_extract_dict["rois"]
    roi_locations_list = []
    for i in range(len(rois)):
        roi = rois[i]
        #if roi["mask"][0] == "{":
        #    mask = _parse_mask_string(roi["mask"])
        #else:
        mask = roi["mask"]
        roi_locations_list.append(
            [roi["id"], roi["x"], roi["y"], roi["width"], roi["height"],
            roi["valid"], mask])
    roi_locations = pd.DataFrame(
        data=roi_locations_list,
        columns=["id", "x", "y", "width", "height", "valid", "mask"])
    
    return roi_locations


#############################################
def add_cell_specimen_ids_to_roi_metrics(roi_metrics, roi_locations):
    """
    Returns ROI metrics dataframe with ROI IDs added.

    Required args:
        - roi_metrics (pd DataFrame)  : ROI metrics dataframe
        - roi_locations (pd DataFrame): ROI locations dataframe
    
    Returns:
        - roi_metrics (pd DataFrame): ROI metrics dataframe updated with 
                                      locations information
    """

    # add roi ids to objectlist
    roi_metrics = roi_metrics.copy(deep=True)
    ids = []
    for row in roi_metrics.index:
        minx = roi_metrics.iloc[row][" minx"]
        miny = roi_metrics.iloc[row][" miny"]
        wid  = roi_metrics.iloc[row][" maxx"] - minx + 1
        hei  = roi_metrics.iloc[row][" maxy"] - miny + 1
        id_vals = roi_locations[
            (roi_locations.x == minx) & (roi_locations.y == miny) &
            (roi_locations.width == wid) & (roi_locations.height == hei)
                ].id.values
        if len(id_vals) != 1:
            if len(id_vals) > 1:
                msg = f"Multiple ROI matches found ({len(id_vals)})."
            else:
                msg = "No ROI matches found."
            raise ValueError(msg)
        ids.append(id_vals[0])
    roi_metrics["cell_specimen_id"] = ids
    
    return roi_metrics


#############################################
def get_motion_border(roi_extract_dict):
    """
    get_motion_border(roi_extract_dict)

    Returns motion border for motion corrected stack.

    Required args:
        - roi_extract_dict (dict): ROI extraction dictionary

    Returns:
        - motion border (list): motion border values for [x0, x1, y1, y0]
                                (right, left, down, up shifts)
    """

    if not isinstance(roi_extract_dict, dict):
        roi_extract_dict = file_util.loadfile(roi_extract_dict)

    coords = ["x0", "x1", "y0", "y1"]
    motion_border = [roi_extract_dict["motion_border"][coord] 
        for coord in coords] 

    return motion_border


#############################################
def get_roi_metrics(roi_extract_dict, objectlist_txt):
    """
    get_roi_metrics(roi_extract_dict, objectlist_txt)

    Returns ROI metrics loaded from object list file and updated based on 
    ROI extraction dictionary.

    Required args:
        - roi_extract_dict (dict): ROI extraction dictionary
        - objectlist_txt (str)   : path to object list containing ROI metrics
    
    Returns:
        - roi_metrics (pd DataFrame): dataframe containing ROI metrics
    """

    roi_locations = get_roi_locations(roi_extract_dict)
    roi_metrics = pd.read_csv(objectlist_txt)

    # get roi_locations and add unfiltered cell index
    roi_names = np.sort(roi_locations.id.values)
    roi_locations["unfiltered_cell_index"] = [
        np.where(roi_names == roi_id)[0][0] 
        for roi_id in roi_locations.id.values]

    # add cell ids to roi_metrics from roi_locations
    roi_metrics = add_cell_specimen_ids_to_roi_metrics(
        roi_metrics, roi_locations)

    # merge roi_metrics and roi_locations
    roi_metrics["id"] = roi_metrics.cell_specimen_id.values
    roi_metrics = pd.merge(roi_metrics, roi_locations, on="id")

    # add filtered cell index
    cell_index = [
        np.where(np.sort(roi_metrics.cell_specimen_id.values) == roi_id)[0][0]
        for roi_id in roi_metrics.cell_specimen_id.values]

    roi_metrics["cell_index"] = cell_index

    return roi_metrics


#############################################
def get_roi_masks(mask_file=None, roi_extract_json=None, objectlist_txt=None, 
                  mask_threshold=0.1, min_n_pix=3, make_bool=True):
    """
    get_roi_masks()

    Returns ROI masks, loaded either from an h5 or json file, and converted
    to boolean.

    Optional args:
        - mask_file (str)       : ROI mask h5. If None, roi_extract_json and
                                  objectlist_txt are used.
                                  default: None
        - roi_extract_json (str): ROI extraction json (only needed is mask_file 
                                  is None)
        - objectlist_txt (str)  : ROI object list txt (only needed if mask_file 
                                  is None)
                                  default: None
        - mask_threshold (float): minimum value in non-boolean mask to
                                  retain a pixel in an ROI mask
                                  default: 0.1
        - min_n_pix (int)       : minimum number of pixels in an ROI below 
                                  which, ROI is set to be empty
                                  default: 3
        - make_bool (bool)      : if True, ROIs are converted to boolean 
                                  before being returned
                                  default: True 
        
    Returns:
        - roi_masks (3D array): ROI masks, structured as 
                                ROI x height x width
        - roi_ids (list)      : ID for each ROI
    """

    if (mask_file is None and 
        (roi_extract_json is None or objectlist_txt is None)):
        raise ValueError("Must provide 'mask_file' or both "
            "'roi_extract_json' and 'objectlist_txt'.")

    if mask_file is None:
        roi_extract_dict = file_util.loadfile(roi_extract_json)
        h = roi_extract_dict["image"]["height"]         
        w = roi_extract_dict["image"]["width"]
        
        roi_metrics = get_roi_metrics(roi_extract_dict, objectlist_txt)
        roi_ids = np.sort(roi_metrics.cell_specimen_id.values)
        nrois = len(roi_ids)

        roi_masks = np.full([nrois, h, w], False).astype(bool)
        for i, roi_id in enumerate(roi_ids):
            m = roi_metrics[roi_metrics.id == roi_id].iloc[0]
            mask = np.asarray(m["mask"])
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            binary_mask[
                int(m.y): int(m.y) + int(m.height),
                int(m.x): int(m.x) + int(m.width)] = mask
            roi_masks[i] = binary_mask
        
    else:
        with h5py.File(mask_file, "r") as f:
            roi_masks = f["data"][()]

        roi_ids = list(range(len(roi_masks)))

    roi_masks[roi_masks < mask_threshold] = 0

    set_empty = np.sum(np.sum(roi_masks, axis=1), axis=1) < min_n_pix
    roi_masks[set_empty] = 0

    if make_bool:
        roi_masks = roi_masks.astype(bool)


    return roi_masks, roi_ids


#############################################
def get_valid_mask(roi_objs, neuropil_trace=None):
    """
    validate_masks(roi_objs)

    Returns a boolean mask for valid ROIs using the following exclusion 
    criteria: duplicate, empty, motion_border, union, empty_neuropil (optional).

    Required args:
        - roi_objs (ROI objects): ROI objects

    Optional args:
        - neuropil_traces (list) : neuropil traces from which to infer empty
                                   neuropil masks. If none provided, this 
                                   exclusion criterion is omitted.
                                   default: None
    
    Returns: 
        - valid_mask (1D array): boolean array of valid masks
    """
    
    excl_mask_dict = validate_masks(roi_objs, neuropil_trace)

    valid_mask = np.ones(len(roi_objs)).astype(bool)
    for _, excl_mask in excl_mask_dict.items():
        valid_mask[np.where(excl_mask)] = False

    return valid_mask


#############################################
def validate_masks(roi_objs, neuropil_traces=None):
    """
    validate_masks(roi_objs)

    Returns a dictionary with exclusion ROI masks for each exclusion criterion 
    ("duplicate", "empty", "motion_border", "union", "empty_neuropil"). 

    Required args:
        - roi_objs (ROI objects): ROI objects

    Optional args:
        - neuropil_traces (list) : neuropil traces from which to infer empty
                                   neuropil masks. If none provided, this 
                                   exclusion label is omitted
                                   default: None

    Returns:
        - excl_mask_dict (dict): dictionary of masks for different exclusion 
                                  criteria, where ROIs labeled by the exclusion
                                  criterion are marked as True, with keys:
            ["duplicate"]    : mask for duplicate ROIs
            ["empty"]        : mask for empty ROIs
            ["motion_border"]: mask for motion border overlapping ROIs
            ["union"]        : mask for union ROIs
    """

    exclusion_labels = EXCLUSION_LABELS

    if "empty_neuropil" in exclusion_labels and neuropil_traces is None:
        logger.warning("Empty_neuropil label will be omitted as "
            "neuropil_traces is None.")
        _ = exclusion_labels.remove("empty_neuropil")

    excl_mask_dict = dict()
    for lab in exclusion_labels:
        excl_mask_dict[lab] = np.zeros(len(roi_objs)).astype(bool)

    other_expl = []
    for r, roi_obj in enumerate(roi_objs):
        if roi_obj.mask is None:
            excl_mask_dict["empty"][r] = True
            continue
        if roi_obj.overlaps_motion_border:
            excl_mask_dict["motion_border"][r] = True
            other_expl.append(r)
        if "union" in roi_obj.labels:
            excl_mask_dict["union"][r] = True
            other_expl.append(r)
        if "duplicate" in roi_obj.labels:
            excl_mask_dict["duplicate"][r] = True
            other_expl.append(r)

    if "empty_neuropil" in exclusion_labels:
        nan_idx = np.where(np.isnan(np.sum(neuropil_traces, axis=1)))[0]
        empty_inferred = np.asarray(list(set(nan_idx) - set(other_expl)))
        if len(empty_inferred) != 0:
            excl_mask_dict["empty_neuropil"][empty_inferred] = True

    return excl_mask_dict


#############################################
def label_unions_and_duplicates(roi_objs, masks=None, duplicate_threshold=0.9, 
    union_threshold=0.7, max_dist=10, set_size=2):
    """
    
    Modified from allensdk.internal.brain_observatory.roi_filter.py
    
    Returns ROI objects with unions and duplicates labelled.

    Required args:
        - roi_objs (ROI objects): ROI objects

    Optional args:
        - masks (3D array)           : ROI mask arrays. If None provided, they 
                                       are recreated from the ROI objects
                                       default: None
        - duplicate_threshold (float): threshold for identifying ROI duplicated
                                       (only the first of each set is labelled 
                                       a duplicate)
                                       default: 0.9
        - union_threshold (float)    : threshold for identifying ROIs that are 
                                       unions of several ROIs
                                       default: 0.7
        - set_size (int)             : number of ROIs forming sets to be checked
                                       for possibly being unions
                                       default: 2
        - max_dist (num)             : max distance between ROIs to be checked
                                       for possibly being unions
                                       default: 10

    Returns:
        - roi_objs (ROI objects): ROI objects labelled for union, duplicate,
                                  empty and border overlapping mask conditions

    """

    roi_objs = copy.deepcopy(roi_objs)

    if masks is None:
        masks = roi_masks.create_roi_mask_array(roi_objs)

    # get indices for non empty ROIs
    non_empty_mask = np.asarray([
        roi_obj.mask is not None for roi_obj in roi_objs]).astype(bool)
    non_empty_idx = np.where(non_empty_mask)[0]
    
    # label empty ROIs
    for idx in np.where(~non_empty_mask)[0]:
        roi_objs[idx].labels.append("empty")

    ms = mask_set.MaskSet(masks=masks[non_empty_idx])

    # detect and label duplicates
    duplicates = ms.detect_duplicates(duplicate_threshold)
    for duplicate in duplicates:
        orig_idx = non_empty_idx[duplicate[0]]
        if "duplicate" not in roi_objs[orig_idx].labels:
            roi_objs[orig_idx].labels.append("duplicate")

    # detect and label unions
    unions = ms.detect_unions(set_size, max_dist, union_threshold)

    if unions:
        union_idxs = list(unions.keys())
        for idx in union_idxs:
            orig_idx = non_empty_idx[idx]
            if "union" not in roi_objs[orig_idx].labels:
                roi_objs[orig_idx].labels.append("union")
    
    return roi_objs


#############################################
def create_mask_objects(masks, motion_border, roi_ids, union_threshold=0.7):
    """
    create_mask_objects(masks, motion_border, roi_ids)

    Returns mask objects, labeled for overlapping the motion border, as well
    as for labels, duplicates and being empty.

    Required args:
        - masks (3D array)    : ROI masks, structured as ROI x height x width
        - motion border (list): motion border values for [x0, x1, y1, y0]
                                (right, left, down, up shifts)
        - roi_ids (list)      : ID for each ROI

    Returns:
        - all_mask_objs (list) : list of ROI Mask objects, with exclusion labels
                                 (allensdk roi_masks.py)
    """

    all_mask_objs = []

    hei, wid = masks.shape[1:]
    for _, (mask, roi_id) in enumerate(zip(masks, roi_ids)):
        all_mask_objs.append(roi_masks.create_roi_mask(
            wid, hei, motion_border, roi_mask=mask, label=str(roi_id)))
        # add empty list of labels
        all_mask_objs[-1].labels = []
    all_mask_objs = label_unions_and_duplicates(all_mask_objs, masks, 
        union_threshold=0.7)        

    return all_mask_objs


#############################################
def save_roi_dataset(data, save_path, roi_names, data_name="data", 
                     excl_dict=None, replace=True, compression=None):
    """
    save_roi_dataset(save_path, roi_names)

    Saves ROI dataset.

    Required args:
        - data (nd array)       : ROI data, where first dimension are ROIs
        - save_path (str)       : path for saving the dataset
        - roi_names (array-like): list of names for each ROI
    
    Optional args:
        - data_name (str)  : main dataset name
                             default: "data"
        - excl_dict (dict) : dictionary of exclusion masks for different 
                             criteria
                             default: None
        - replace (bool)   : if True, an existing file is replaced
                             default: True
        - compression (str): type of compression to use when saving h5 
                             file (e.g., "gzip")
                             default: None
    """

    if len(data) != len(roi_names):
        raise ValueError("'roi_names' must be as long as the first dimension "
            "of 'data'.")

    if os.path.isfile(save_path) and not replace:
        logger.info("ROI traces already exist.")
        return

    file_util.createdir(os.path.dirname(save_path), log_dir=False)

    with h5py.File(save_path, "w") as hf:
        hf.create_dataset(data_name, data=data, compression=compression)
        hf.create_dataset("roi_names", data=np.asarray(roi_names, dtype="S"))
        if excl_dict is not None:
            for key, item in excl_dict.items():
                hf.create_dataset(key, data=np.asarray(item, dtype="u1"))


#############################################
def demix_rois(raw_traces, h5path, masks, excl_dict):
    """
    demix_rois(raw_traces, h5path, masks, excl_dict)

    Returns time-dependent demixed traces.

    Required args:
        - raw_traces (2D array): extracted traces, structured as ROI x frames
        - h5path (str)         : path to full movie, structured as 
                                 time x height x width
        - masks (3D array)     : ROI mask, structured as ROI x height x width
        - excl_dict (dict)     : dictionary of masks for different exclusion 
                                 criteria, where ROIs labeled by the exclusion
                                 criterion are marked as True, with keys:
            ["duplicate"]    : mask for duplicate ROIs
            ["empty"]        : mask for empty ROIs
            ["motion_border"]: mask for motion border overlapping ROIs
            ["union"]        : mask for union ROIs

    Returns:
        - demixed_traces (2D array): demixed traces, with excluded ROIs set to 
                                     np.nan, structured as ROI x frames 
        - drop_frames (list)       : list of boolean values for each frame, 
                                     recording whether it is dropped
    """

    exclusion_labels = EXCLUSION_LABELS
    valid_mask = np.ones(len(masks)).astype(bool)
    for lab in exclusion_labels:
        if lab not in excl_dict.keys():
            if lab == "empty_neuropil":
                logger.warning("ROIs with empty neuropil not checked for "
                    "before demixing.")
            else:
                raise ValueError(f"{lab} missing from excl_dict keys.")
        valid_mask *= ~(excl_dict[lab].astype(bool))

    if len(valid_mask) != len(raw_traces):
        raise ValueError("'valid_mask' must be as long as the first dimension "
            "of 'raw_traces'.")

    # omit invalid ROIs from demixing process
    raw_traces_valid = raw_traces[valid_mask.astype(bool)]
    masks_valid = masks[valid_mask.astype(bool)]

    with h5py.File(h5path, "r") as f:
        demix_traces, drop_frames = demixer.demix_time_dep_masks(
            raw_traces_valid, f["data"], masks_valid)

    # put NaNs in for dropped ROIs
    demix_traces_all = np.full(raw_traces.shape, np.nan)
    demix_traces_all[valid_mask.astype(bool)] = demix_traces

    return demix_traces_all, drop_frames


############################################
def get_neuropil_subtracted_traces(roi_traces, neuropil_traces):
    """
    get_neuropil_subtracted_traces(roi_traces, neuropil_traces)

    Returns ROI traces with neuropil subtracted, as well as the contamination 
    ratio for each ROI.

    Required args:
        - roi_traces (2D array)     : ROI traces, structured as ROI x frame
        - neuropil_traces (2D array): neuropil traces, structured as ROI x frame

    Returns:
        - neuropilsub_traces (2D array): ROI traces with neuropil subtracted, 
                                         structured as ROI x frame
        - r (1D array)                 : contamination ratio (0-1) for each ROI
    """
    
    r = np.full(len(roi_traces), 0.)
    for i, (roi_trace, neuropil_trace) in enumerate(zip(
        roi_traces, neuropil_traces)):
        if np.isfinite(roi_trace).all():
            r[i] = r_neuropil.estimate_contamination_ratios(
                roi_trace, neuropil_trace, iterations=3)["r"]

    neuropilsub_traces = roi_traces - neuropil_traces * r[:, np.newaxis]

    return neuropilsub_traces, r


############################################
def create_traces_from_masks(datadir, sessid, runtype="prod", h5dir=None,
                             savedir="trace_proc_dir", dendritic=False, 
                             mask_threshold=0.1, min_n_pix=3, 
                             compression=None):
    """
    create_traces_from_masks(datadir, sessid)

    Extracts traces from masks, applies correction (neuropil traces, demixed 
    traces, corrected traces, dF/F traces) and saves them. 
    
    WARNING: Will replace any existing files.

    Required args:
        - datadir (str): name of the data directory
        - sessid (int) : session ID (9 digits)

    Optional args:
        - runtype (str)         : "prod" (production) or "pilot" data
                                  default: "prod"
        - h5dir (str)           : name of the h5 data directory. If None, 
                                  datadir is used.
                                  default: None
        - savedir (str)         : name of the directory in which to save new 
                                  files. If None, datadir is used.
                                  default: "trace_proc_dir"
        - dendritic (bool)      : if True, paths are changed to EXTRACT 
                                  version dendritic
                                  default: False
        - mask_threshold (float): minimum value in non-boolean mask to
                                  retain a pixel in an ROI mask
                                  default: 0.1 
        - min_n_pix (int)       : minimum number of pixels in an ROI
                                  default: 3
        - compression (str)     : type of compression to use when saving data 
                                  to h5 files (e.g., "gzip")
                                  default: None
    """

    # Retrieve all file paths
    file_dict = sess_file_util.get_file_names_from_sessid(
        datadir, sessid, runtype, check=False)[1]

    roi_extract_json = file_dict["roi_extract_json"]
    objectlist_path  = file_dict["roi_objectlist_txt"]
    h5path           = file_dict["correct_data_h5"]

    # remove extra slashes
    dirnames = [datadir, h5dir, savedir]
    datadir, h5dir, savedir = [os.path.normpath(dirname) 
        for dirname in dirnames]

    if h5dir is not None:
        h5path = h5path.replace(datadir, h5dir)

    mask_path = None
    if dendritic:
        mask_path = sess_file_util.get_dendritic_mask_path_from_sessid(
            datadir, sessid, runtype, check=True)

    # use data directory, as requires some files to be present
    roi_trace_dict = sess_file_util.get_roi_trace_paths_from_sessid(
        datadir, sessid, runtype, dendritic=dendritic, check=False)
    if savedir is not None:
        for key, item in roi_trace_dict.items():
            roi_trace_dict[key] = item.replace(datadir, savedir)

    logger.info("Extracting ROI masks.")
    masks_bool, roi_ids = get_roi_masks(
        mask_path, roi_extract_json, objectlist_path, 
        mask_threshold=mask_threshold, min_n_pix=min_n_pix, make_bool=True, )
    motion_border = get_motion_border(roi_extract_json)
    all_mask_objs = create_mask_objects(masks_bool, motion_border, roi_ids, 
        union_threshold=0.7)


    logger.info("Creating ROI and neuropil traces.")
    [roi_traces, 
    neuropil_traces, _] = roi_masks.calculate_roi_and_neuropil_traces(
        h5path, all_mask_objs, motion_border=motion_border)
    
    excl_dict = validate_masks(all_mask_objs, neuropil_traces=neuropil_traces)

    logger.info("Saving raw ROI traces.")
    save_roi_dataset(
        roi_traces, roi_trace_dict["unproc_roi_trace_h5"], roi_ids, 
        excl_dict=excl_dict, replace=True, compression=compression)

    logger.info("Saving neuropil traces.")
    save_roi_dataset(
        neuropil_traces, roi_trace_dict["neuropil_trace_h5"], roi_ids, 
        excl_dict=excl_dict, replace=True, compression=compression)


    logger.info("Demixing ROI traces.")
    demixed_traces, _ = demix_rois(roi_traces, h5path, masks_bool, excl_dict)

    logger.info("Saving demixed traces.")
    save_roi_dataset(
        demixed_traces, roi_trace_dict["demixed_trace_h5"], roi_ids, 
        excl_dict=excl_dict, replace=True, compression=compression)


    logger.info("Subtracting neuropil from demixed ROI traces.")
    raw_processed_traces, r = get_neuropil_subtracted_traces(
        demixed_traces, neuropil_traces)

    logger.info("Saving processed traces")
    save_roi_dataset(
        raw_processed_traces, roi_trace_dict["roi_trace_h5"], roi_ids, 
        data_name="FC", excl_dict=excl_dict, replace=True, 
        compression=compression)
    # also save contamination ratios
    with h5py.File(roi_trace_dict["roi_trace_h5"], "r+") as hf:
        hf.create_dataset("r", data=r, compression=compression)
    

    logger.info("Calculating dF/F")
    dff_traces = dff.calculate_dff(raw_processed_traces)
    
    logger.info("Saving dF/F traces.")
    save_roi_dataset(
        dff_traces, roi_trace_dict["roi_trace_dff_h5"], roi_ids, 
        excl_dict=excl_dict, replace=True, compression=compression)


    return

