"""
roi_analys.py

This script contains functions for ROI mask and projection analysis.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import numpy as np
import pandas as pd

from util import gen_util
from sess_util import sess_load_util
from analysis import misc_analys


# PRESET CROPPING PARAMETERS FOR SPECIFIC MICE
# (crop factor, shift prop hei, shift prop wid)
SMALL_CROP_DICT = {
    4: (2.5, 0, 1), # L5-S
    11: (5.5, 0.74, 0.84), # L5-D
}

LARGE_CROP_DICT = {
    3: (3.0, 0.30, 0.85), # L23-S
    4: (3.0, 0.08, 0.95), # L5-S
    6: (3.0, 0.50, 0.50), # L23-D
    11: (3.0, 0.90, 0.90), # L5-D
}


#############################################
def get_sess_reg_mask_info(sess, analyspar, reg=True, proj=True):
    """
    get_sess_reg_mask_info(sess, analyspar)

    Returns a dictionary with ROI mask and projection information for the 
    session.

    Required args:
         - sess (session.Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters

    Optional args:
        - reg (bool):
            if True, registered ROI masks, and projection if proj is True, are 
            included in the returned dictionary
            default: True
        - proj (bool):
            if True, max projections are included in the returned dictionary
            default: True

    Returns
        - sess_dict (dict):
            dictionary with the following keys:
            ["roi_masks"] (3D array): ROI masks (ROI, hei, wid)

            if reg:
            ["registered_roi_mask_idxs"] (3D array): ROI masks (ROI, hei, wid), 
                registered across sessions

            if proj:
            ["max_projection"] (2D array): pixel intensities of maximum 
                projection for the plane (hei x wid)

            if reg and proj:
            ["registered_max_projection"] (2D array): pixel intensities of 
                maximum projection for the plane (hei x wid), registered across 
                sessions
    """

    if reg and not analyspar.tracked:
        raise ValueError("analyspar.tracked must be True for this analysis.")

    sess_dict = dict()

    if sess.only_tracked_rois != analyspar.tracked:
        raise ValueError(
            "sess.only_tracked_rois must match analyspar.tracked."
            )

    # get unregistered ROI masks
    sess_dict["roi_masks"] = sess.get_roi_masks(
        fluor=analyspar.fluor, rem_bad=analyspar.rem_bad
        )

    if reg:
        # get registered ROI masks
        registered_roi_masks = sess.get_registered_roi_masks(
            fluor=analyspar.fluor, rem_bad=analyspar.rem_bad
            )
        sess_dict["registered_roi_masks"] = registered_roi_masks

    if proj:
        sess_dict["max_projection"] = sess.max_proj
        if reg:
            sess_dict["registered_max_projection"] = \
                sess.get_registered_max_proj()


    return sess_dict    


#############################################
def get_roi_tracking_df(sessions, analyspar, reg_only=False, proj=False, 
                        crop_info=False, parallel=False):
    """
    get_roi_tracking_df(sessions, analyspar)

    Return ROI tracking information for the requested sessions.

    Required args:
        - sessions (list): 
            Session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters

    Optional args:
        - proj (bool):
            if True, max projections are included in the output dataframe
            default: False
        - reg_only (bool):
            if True, only registered masks, and projections if proj is True, 
            are included in the output dataframe
            default: False
        - crop_info (bool or str):
            if not False, the type of cropping information to include 
            ("small" for the small plots, "large" for the large plots)
            default: False
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False

    Returns:
        - roi_mask_df (pd.DataFrame in dict format):
            dataframe with a row for each mouse, and the following 
            columns, in addition to the basic sess_df columns: 

            - "registered_roi_mask_idxs" (list): list of mask indices, 
                registered across sessions, for each session 
                (flattened across ROIs) ((sess, hei, wid) x val)
            - "roi_mask_shapes" (list): shape into which ROI mask indices index 
                (sess x hei x wid)

            if not reg_only:
            - "roi_mask_idxs" (list): list of mask indices for each session, 
                and each ROI (sess x ((ROI, hei, wid) x val)) (not registered)

            if proj:
            - "registered_max_projections" (list): pixel intensities of maximum 
                projection for the plane (hei x wid), after registration across 
                sessions

            if proj and not reg_only:
            - "max_projections" (list): pixel intensities of maximum projection 
                for the plane (hei x wid)
                
            if crop_info:
            - "crop_fact" (num): factor by which to crop masks (> 1) 
            - "shift_prop_hei" (float): proportion by which to shift cropped 
                mask center vertically from left edge [0, 1]
            - "shift_prop_wid" (float): proportion by which to shift cropped 
                mask center horizontally from left edge [0, 1]
    """

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked must be True for this analysis.")

    misc_analys.check_sessions_complete(sessions, raise_err=True)

    sess_df = misc_analys.get_check_sess_df(sessions, analyspar=analyspar)

    # if cropping, check right away for dictionary with the preset parameters
    if crop_info:
        if crop_info == "small":
            crop_dict = SMALL_CROP_DICT
        elif crop_info == "large":
            crop_dict = LARGE_CROP_DICT
        else:
            gen_util.accepted_values_error(
                "crop_info", crop_info, ["small", "large"]
                )
        for mouse_n in sess_df["mouse_ns"].unique():
            if int(mouse_n) not in crop_dict.keys():
                raise NotImplementedError(
                    f"No preset cropping information found for mouse {mouse_n}."
                    )

    # collect ROI mask data
    sess_dicts = gen_util.parallel_wrap(
        get_sess_reg_mask_info, sessions, args_list=[analyspar, True, proj], 
        parallel=parallel
        )
    all_sessids = [sess.sessid for sess in sessions]

    group_columns = ["planes", "lines", "mouse_ns"]
    initial_columns = sess_df.columns.tolist()
    obj_columns = ["registered_roi_mask_idxs", "roi_mask_shapes"]
    if not reg_only:
        obj_columns.append("roi_mask_idxs")
    if proj:
        obj_columns.append("registered_max_projections")
        if not reg_only:
            obj_columns.append("max_projections")

    roi_mask_df = pd.DataFrame(columns=initial_columns + obj_columns)

    aggreg_cols = [col for col in initial_columns if col not in group_columns]
    for grp_vals, grp_df in sess_df.groupby(group_columns):
        row_idx = len(roi_mask_df)
        for g, group_column in enumerate(group_columns):
            roi_mask_df.loc[row_idx, group_column] = grp_vals[g]

        # add aggregated values for initial columns
        roi_mask_df = misc_analys.aggreg_columns(
            grp_df, roi_mask_df, aggreg_cols, row_idx=row_idx, 
            in_place=True, by_mouse=True
            )

        sessids = sorted(grp_df["sessids"].tolist())
        reg_roi_masks, roi_mask_idxs = [], []
        if proj:
            reg_max_projs, max_projs = [], []

        roi_mask_shape = None
        for sessid in sessids:
            sess_dict = sess_dicts[all_sessids.index(sessid)]
            reg_roi_mask = sess_dict["registered_roi_masks"]
            # flatten masks across ROIs
            reg_roi_masks.append(np.max(reg_roi_mask, axis=0))
            if roi_mask_shape is None:
                roi_mask_shape = reg_roi_mask.shape
            elif roi_mask_shape != reg_roi_mask.shape:
                raise RuntimeError(
                    "ROI mask shapes across sessions should match, for the "
                    "same mouse."
                    )
            if not reg_only:
                roi_mask_idxs.append(
                    [idxs.tolist() for idxs in np.where(sess_dict["roi_masks"])]
                    )
            if proj:
                reg_max_projs.append(
                    sess_dict["registered_max_projection"].tolist()
                    )
                if not reg_only:
                    max_projs.append(sess_dict["max_projection"].tolist())

        # add to the dataframe
        roi_mask_df.at[row_idx, "registered_roi_mask_idxs"] = \
            [idxs.tolist() for idxs in np.where(reg_roi_masks)]
        roi_mask_df.at[row_idx, "roi_mask_shapes"] = roi_mask_shape

        if not reg_only:
            roi_mask_df.at[row_idx, "roi_mask_idxs"] = roi_mask_idxs
        if proj:
            roi_mask_df.at[row_idx, "registered_max_projections"] = \
                reg_max_projs
            if not reg_only:
                roi_mask_df.at[row_idx, "max_projections"] = max_projs

        # add cropping info
        if crop_info:
            mouse_n = grp_vals[group_columns.index("mouse_ns")]
            crop_fact, shift_prop_hei, shift_prop_wid = crop_dict[mouse_n]
            roi_mask_df.at[row_idx, "crop_fact"] = crop_fact
            roi_mask_df.at[row_idx, "shift_prop_hei"] = shift_prop_hei
            roi_mask_df.at[row_idx, "shift_prop_wid"] = shift_prop_wid
    
    roi_mask_df["mouse_ns"] = roi_mask_df["mouse_ns"].astype(int)

    return roi_mask_df


#############################################
def collect_roi_tracking_example_data(sessions):
    """
    collect_roi_tracking_example_data(sessions)

    Returns information on example ROI tracking permutations that yielded 
    either the most or fewest matches for a set of sessions, as well as the 
    union of all permutations.

    Runs a lot of checks to verify that the stored example data fits with the 
    tracking data retrieved from the dataset.

    Only sessions from certain mice have the requisit data stored in their 
    nway-match files.

    Required args:
        - sessions (list): 
            Session objects (of a single mouse)

    Returns:
        - masks (dict): 
            dictionary with masks (sess x hei x wid) under keys 
            ["fewest"] and ["most"] for ROIs included in the list of tracked 
            ROIs for the given permutation, registered to first session in the 
            permutation order, and listed in the same order
        - ordered_sess_ns (dict):
            dictionary with session numbers ordered for the given permutation 
            under keys ["fewest"] and ["most"]
        - n_tracked (dict):
            dictionary with the number of tracked ROIs for different cases, 
            i.e., ["fewest"], ["most"], ["union"] and ["conflicts"]. 
    """

    # sort sessions, and run some checks
    sessids = [sess.sessid for sess in sessions]
    sessions = [sessions[i] for i in np.argsort(sessids)]
    mouse_ns = [sess.mouse_n for sess in sessions]
    if len(np.unique(mouse_ns)) != 1:
        raise ValueError("Sessions must all come from the same mouse.")

    # Prepare potential subsequent error messages    
    lead_error_msg = (
        "Error in the tracking permutation example dataframe for this "
        "session: "
    )

    # collect data for different permutation types
    perm_types = ["fewest", "most"]

    ordered_sess_ns = {"fewest": None, "most": None}
    n_tracked = {"fewest": None, "most": None, "union": None, "conflict": None}
    masks = {"fewest": [], "most": []}

    for sess in sessions:
        if not sess.only_tracked_rois:
            raise ValueError(
                "Sessions should all be set to use only tracked ROIs."
                )
        
        tracked_roi_masks = sess.get_roi_masks()
        n_tracked_rois = len(tracked_roi_masks)

        rem_bad = sess.nwb # index without bad ROIs, if using NWB
        ex_df = sess_load_util.get_tracking_perm_example_df(
            sess.get_local_nway_match_path(), sessid=sess.sessid,
            idx_after_rem_bad=rem_bad
            )

        # check that union information makes sense
        union_row = ex_df.loc[ex_df["match_level"] == "union"].index
        n_union = ex_df.loc[union_row[0], "n_total"]
        if n_tracked["union"] is None:
            n_tracked["union"] = n_union
        elif n_tracked["union"] != n_union:
            raise RuntimeError(
                f"{lead_error_msg}The number of union ROI matches is not "
                "consistent across sessions for the same mouse."
                )
        
        if n_union < n_tracked_rois:
            raise RuntimeError(
                f"{lead_error_msg}The number of union ROI matches is smaller "
                "than the number of tracked ROIs for the session."
                )
        
        # check that conflict information makes sense
        if n_tracked["conflict"] is None:
            n_tracked["conflict"] = n_union - n_tracked_rois
        elif n_tracked["conflict"] != n_union - n_tracked_rois:
            raise RuntimeError(
                f"{lead_error_msg}The overall number of conflicting ROI "
                "matches is not consistent across sessions for the same mouse."
                )

        # retrieve masks for fewest and most match level
        min_n_conflicts = 0
        for perm_type in perm_types:
            ex_df_row_idx = ex_df.loc[ex_df["match_level"] == perm_type].index
            if len(ex_df_row_idx) != 1:
                raise RuntimeError(
                    f"{lead_error_msg}Expected exactly one row with "
                    f"match_level '{perm_type}'."
                    )
            ex_df_row_idx = ex_df_row_idx[0]
            
            sess_order = ex_df.loc[ex_df_row_idx, "sess_order"]
            if ordered_sess_ns[perm_type] is None:
                ordered_sess_ns[perm_type] = ex_df.loc[
                    ex_df_row_idx, "sess_order"
                    ]
            elif sess_order != ordered_sess_ns[perm_type]:
                raise RuntimeError(
                    f"{lead_error_msg}The session order for match_level "
                    f"'{perm_type}' is not recorded consistently across "
                    "sessions for the same mouse."
                    )
            targ_sess_idx = sorted(sess_order).index(sess_order[0])
                        
            # get tracked ROIs missing for this permutation
            missing_roi_ns =  ex_df.loc[
                ex_df_row_idx, "dff_local_missing_roi_idx"
                ]
            track_idx = [
                sess.tracked_rois.tolist().index(n) for n in missing_roi_ns
                ]
            keep_track_idx = np.arange(n_tracked_rois)
            if len(track_idx):
                keep_track_idx = np.delete(keep_track_idx, track_idx)
            perm_masks = [tracked_roi_masks[np.asarray(keep_track_idx)]]

            # get untracked ROIs identified in this permutation
            # index into original ROI list, excluding bad ROIs
            extra_roi_ns = ex_df.loc[ex_df_row_idx, "dff_local_extra_roi_idx"]
            min_n_conflicts = np.max([min_n_conflicts, len(extra_roi_ns)])
            if len(extra_roi_ns):
                try:
                    sess.set_only_tracked_rois(False)
                    extra_roi_masks = sess.get_roi_masks(
                        rem_bad=rem_bad
                        )[np.asarray(extra_roi_ns)]
                    perm_masks.append(extra_roi_masks)
                finally: 
                    # make sure to reset, even if an error occurs
                    sess.set_only_tracked_rois(True)

            # apply registration transform
            perm_masks = sess_load_util.apply_registration_transform(
                sess.get_local_nway_match_path(), 
                sess.sessid, 
                image=np.concatenate(perm_masks, axis=0).astype("uint8"), 
                targ_sess_idx=targ_sess_idx
            ).astype(bool)

            masks[perm_type].append(perm_masks)
            
            # run some checks
            if n_tracked[perm_type] is None:
                n_tracked[perm_type] = len(perm_masks)

            elif len(perm_masks) != n_tracked[perm_type]:
                raise RuntimeError(
                    f"{lead_error_msg}The number of ROI matches for "
                    f"match_level '{perm_type}' is not consistent across "
                    "sessions for the same mouse."
                    )
            
            if n_tracked[perm_type] != ex_df.loc[ex_df_row_idx, "n_total"]:
                raise RuntimeError(
                    f"{lead_error_msg}The number of ROI matches for "
                    f"match_level '{perm_type}' does not match the number of "
                    "ROIs retrieved from the data."
                    )
        
    # sum across tracked ROIs and reorder sessions
    fewest_order = np.argsort(np.argsort(ordered_sess_ns["fewest"]))
    masks["fewest"] = np.sum(masks["fewest"], axis=1)[fewest_order]

    most_order = np.argsort(np.argsort(ordered_sess_ns["most"]))
    masks["most"] = np.sum(masks["most"], axis=1)[most_order]

    # final number checks
    if n_tracked["fewest"] > n_tracked["most"]:
        raise RuntimeError(
            f"{lead_error_msg}The permutation with the fewest matches contains "
            "more matches than permutation with the most matches."
            )
    if n_tracked["most"] > n_tracked["union"]:
        raise RuntimeError(
            f"{lead_error_msg}The number of union matches is smaller than "
            "the number of matches of the permutation with the most matches."
            )
    if n_tracked["conflict"] < min_n_conflicts:
        raise RuntimeError(
            f"{lead_error_msg}The number of conflicts is smaller than the "
            "minimum calculated from the different permutations included in "
            "the data."
            )        

    return masks, ordered_sess_ns, n_tracked


#############################################
def get_roi_tracking_ex_df(sessions, analyspar, parallel=False):
    """
    get_roi_tracking_ex_df(sessions, analyspar)

    Return ROI tracking example information for the requested sessions, showing 
    the different ROI matches identified depending on the orderin which the 
    sessions are matched.

    Only sessions from certain mice have the requisit data stored in their 
    nway-match files.

    Required args:
        - sessions (list): 
            Session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters

    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    
    Returns:
        - roi_mask_df (pd.DataFrame in dict format):
            dataframe with a row for each mouse, and the following 
            columns, in addition to the basic sess_df columns: 
            - "roi_mask_shapes" (list): shape into which ROI mask indices index 
                (sess x hei x wid)
            - "union_n_conflicts" (int): number of conflicts after union
            for "union", "fewest" and "most" tracked ROIs:
            - "{}_registered_roi_mask_idxs" (list): list of mask indices, 
                registered across sessions, for each session 
                (flattened across ROIs) ((sess, hei, wid) x val), 
                ordered by {}_sess_ns if "fewest" or "most"
            - "{}_n_tracked" (int): number of tracked ROIs
            for "fewest", "most" tracked ROIs:
            - "{}_sess_ns" (list): ordered session number 
    """
    
    perm_types = ["fewest", "most"]
    add_cols = ["union_n_conflicts"]
    for perm_type in perm_types:
        add_cols.append(f"{perm_type}_registered_roi_mask_idxs")
        add_cols.append(f"{perm_type}_n_tracked")
        add_cols.append(f"{perm_type}_sess_ns")

    # collect ROI mask information
    roi_mask_df = get_roi_tracking_df(
        sessions, analyspar, reg_only=True, parallel=parallel
        )
    roi_mask_df = gen_util.set_object_columns(
        roi_mask_df, add_cols, in_place=True
        )
    roi_mask_df = roi_mask_df.rename(
        columns={"registered_roi_mask_idxs": "union_registered_roi_mask_idxs"}
        )
    
    all_sessids = [sess.sessid for sess in sessions]
    for row_idx in roi_mask_df.index:
        sess_ns = roi_mask_df.loc[row_idx, "sess_ns"]
        sessids = roi_mask_df.loc[row_idx, "sessids"]

        mouse_sessions = [
            sessions[all_sessids.index(sessid)] for sessid in sessids
            ]
        
        masks, ordered_sess_ns, n_tracked = collect_roi_tracking_example_data(
            mouse_sessions
            )

        roi_mask_df.loc[row_idx, "union_n_tracked"] = n_tracked["union"]
        roi_mask_df.loc[row_idx, "union_n_conflicts"] = n_tracked["conflict"]
        for perm_type in perm_types:
            if set(ordered_sess_ns[perm_type]) != set(sess_ns):
                raise RuntimeError("Session number do not match.")

            roi_mask_df.at[
                row_idx, f"{perm_type}_registered_roi_mask_idxs"
                ] = [idxs.tolist() for idxs in np.where(masks[perm_type])]
            roi_mask_df.at[row_idx, f"{perm_type}_sess_ns"] = \
                ordered_sess_ns[perm_type]
            roi_mask_df.loc[row_idx, f"{perm_type}_n_tracked"] = \
                n_tracked[perm_type]

    int_cols = [col for col in roi_mask_df.columns if "_n_" in col]
    for col in int_cols:
        roi_mask_df[col] = roi_mask_df[col].astype(int)

    return roi_mask_df


