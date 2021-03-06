"""
session.py

Classes to store, extract, and analyze an Allen Institute OpenScope session for
the Credit Assignment Project.

Authors: Colleen Gillon, Blake Richards

Date: August, 2018

Note: this code uses python 3.7.

"""
import copy
import glob
import json
import logging
import os
import sys
import warnings

import h5py
import itertools
import numpy as np
import pandas as pd
import pickle
import scipy.stats as st
import scipy.signal as scsig

from util import file_util, gen_util, logger_util, math_util
from sess_util import sess_data_util, sess_file_util, sess_gen_util, \
    sess_load_util, sess_pupil_util, sess_sync_util, sess_trace_util

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
#############################################
class Session(object):
    """
    The Session object is the top-level object for analyzing a session from the 
    OpenScope Credit Assignment Project. All that needs to be provided to 
    create the object is the directory in which the session data directories 
    are located and the ID for the session to analyze/work with. The Session 
    object that is created will contain all of the information relevant to a 
    session, including stimulus information, behaviour information and 
    pointers to the 2p data.
    """
    
    def __init__(self, datadir, sessid, runtype="prod", droptol=0.0003, 
                 verbose=False, only_matched_rois=False):
        """
        self.__init__(datadir, sessid)

        Initializes and returns the new Session object using the specified data 
        directory and ID.

        Calls:
            - self._init_directory()

        Attributes:
            - droptol (num)  : dropped frame tolerance (proportion of total)
            - home (str)     : path of the master data directory
            - runtype (str)  : "prod" (production) or "pilot" data
            - sessid (int)   : session ID (9 digits), e.g. "712483302"
        
        Required args:
            - datadir (str): full path to the directory where session 
                             folders are stored.
            - sessid (int) : the ID for this session.

        Optional args:
            - runtype (str)           : the type of run, either "pilot" or 
                                        "prod"
                                        default: "prod"
            - droptol (num)           : the tolerance for percentage stimulus 
                                        frames dropped, create a Warning if  
                                        this condition isn't met.
                                        default: 0.0003 
            - verbose (bool)          : if True, will log instructions on next 
                                        steps to load all necessary data
                                        default: True
            - only_matched_rois (bool): if True, only data from ROIs matched 
                                        across sessions (1-3) are included when 
                                        data is returned
                                        default: False
        """


        self.home   = datadir
        self.sessid = int(sessid)
        if runtype not in ["pilot", "prod"]:
            gen_util.accepted_values_error(
                "runtype", runtype, ["pilot", "prod"])
        self.runtype = runtype
        self.droptol = droptol
        self._init_directory()
        self.only_matched_rois = only_matched_rois

        if verbose:
            print("To load mouse database information into the session, "
                "run 'self.extract_sess_attribs()'.\nTo load stimulus, "
                "behaviour and ophys data, run 'self.extract_info()'")


    #############################################
    def __repr__(self):
        return f"{self.__class__.__name__} ({self.sessid})"

    def __str__(self):
        return repr(self)


    #############################################
    def _init_directory(self):
        """
        self._init_directory()

        Checks that the session data directory obeys the expected organization
        scheme and sets attributes.

        Attributes:        
            - align_pkl (str)         : path name of the stimulus alignment 
                                        pickle file
            - behav_video_h5 (str)    : path name of the behavior hdf5 file
            - correct_data_h5 (str)   : path name of the motion corrected 2p 
                                        data hdf5 file
            - date (str)              : session date (i.e., yyyymmdd)
            - dir (str)               : path of session directory
            - expdir (str)            : path name of experiment directory
            - expid (int)             : experiment ID (8 digits)
            - mouseid (int)           : mouse ID (6 digits)
            - mouse_dir (bool)        : whether path includes a mouse directory
            - procdir (str)           : path name of the processed data 
                                        directory
            - pup_video_h5 (str)      : path name of the pupil hdf5 file
            - roi_extract_json (str)  : path name of the ROI extraction json
            - roi_mask_file (str)     : path name of the ROI mask file (None, 
                                        as allen masks must be created during 
                                        loading)
            - roi_objectlist_txt (str): path name of the ROI object list file
            - roi_trace_h5 (str)      : path name of the ROI raw processed 
                                        fluorescence trace hdf5 file
            - roi_trace_dff_h5 (str)  : path name of the ROI dF/F trace 
                                        hdf5 file
            - stim_pkl (str)          : path name of the stimulus pickle file
            - stim_sync_h5 (str)      : path name of the stimulus 
                                        synchronisation hdf5 file
            - time_sync_h5 (str)      : path name of the time synchronization 
                                        hdf5 file
            - zstack_h5 (str)         : path name of the z-stack 2p hdf5 file
        """

        # check that the high-level home directory exists
        file_util.checkdir(self.home)

        sessdir, mouse_dir = sess_file_util.get_sess_dir_path(
            self.home, self.sessid, self.runtype)
        self.dir       = sessdir
        self.mouse_dir = mouse_dir

        mouseid, date = sess_file_util.get_mouseid_date(self.dir, self.sessid)
        self.mouseid = mouseid
        self.date    = date

        self.expid = sess_file_util.get_expid(self.dir)
        self.segid = sess_file_util.get_segid(self.dir)
        
        dirpaths, filepaths = sess_file_util.get_file_names(
            self.home, self.sessid, self.expid, self.segid, self.date, 
            self.mouseid, self.runtype, self.mouse_dir, check=True)  
        
        self.expdir           = dirpaths["expdir"]
        self.procdir          = dirpaths["procdir"]
        self.stim_pkl         = filepaths["stim_pkl"]
        self.stim_sync_h5     = filepaths["stim_sync_h5"]
        self.behav_video_h5   = filepaths["behav_video_h5"]
        self.pup_video_h5     = filepaths["pupil_video_h5"]
        self.time_sync_h5     = filepaths["time_sync_h5"]
        self.roi_extract_json = filepaths["roi_extract_json"]
        self.roi_objectlist   = filepaths["roi_objectlist_txt"]
        self.roi_mask_file    = None

        # existence not checked
        self.align_pkl        = filepaths["align_pkl"]
        self.roi_trace_h5     = filepaths["roi_trace_h5"]
        self.roi_trace_dff_h5 = filepaths["roi_trace_dff_h5"]
        self.zstack_h5        = filepaths["zstack_h5"]
        self.correct_data_h5  = filepaths["correct_data_h5"]
    

    #############################################
    @property
    def pup_data_h5(self):
        """
        self.pup_data_h5

        Returns:
            - _pup_data_h5 (list or str): single pupil data file path if one is 
                                          found, a list if several are found 
                                          and "none" if none is found
        """
        
        if not hasattr(self, "_pup_data_h5"):
            self._pup_data_h5 = sess_file_util.get_pupil_data_h5_path(self.dir)

        return self._pup_data_h5


    ############################################
    @ property
    def roi_masks(self):
        """
        self.roi_masks()

        Loads boolean ROI masks

        Returns:
            - _roi_masks (3D array): boolean ROI masks, structured as 
                                     ROI x height x width
        """

        mask_threshold = 0.1 # value used in ROI extraction
        min_n_pix = 3 # value used in ROI extraction

        if not hasattr(self, "_dend"):
            raise ValueError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        if not hasattr(self, "_roi_masks"):
            self._roi_masks, _ = sess_trace_util.get_roi_masks(
                self.roi_mask_file, self.roi_extract_json, self.roi_objectlist, 
                mask_threshold=mask_threshold, min_n_pix=min_n_pix, 
                make_bool=True)

        return self._roi_masks


    ############################################
    @ property
    def dend(self):
        """
        self.dend()

        Returns:
            - _dend (str): type of dendrites loaded ("allen" or "extr")
        """

        if not hasattr(self, "_dend"):
            raise ValueError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        return self._dend


    #############################################
    def _load_stim_dict(self, fulldict=True):
        """
        self._load_stim_dict()

        Loads the stimulus dictionary from the stimulus pickle file, checks
        whether the dropped stimulus frames exceeds the drop tolerance and
        logs a warning if it does. 
        
        Attributes: 
            - drop_stim_fr (list)    : list of dropped stimulus frames
            - n_drop_stim_fr (int)   : number of dropped stimulus frames
            - post_blank (num)       : number of blank screen seconds after 
                                       the stimulus end
            - pre_blank (num)        : number of blank screen seconds before 
                                       the stimulus start 
            - stim_dict (dict)       : stimulus dictionary
            - stim_fps (num)         : stimulus frames per second
            - tot_stim_fr (int)      : number of stimulus frames
            - _stim_dict_loaded (str): which stimulus dictionary has been 
                                       loaded

        Optional args:
            - fulldict (bool)  : if True, the full stim_dict is loaded,
                                 else the small stim_dict is loaded
                                 (does not contain "posbyframe" for Bricks)
                                 default: True
        """

        # check which stim dict needs to be loaded
        loading = "fulldict" if fulldict else "smalldict"

        # check if correct stim dict has already been loaded
        if hasattr(self, "_stim_dict_loaded"):
            if self._stim_dict_loaded == loading:
                return
            else:
                logging.info("Stimulus dictionary being reloaded with "
                    f"fulldict as {fulldict}.", extra={"spacing": TAB})
        else:
            self._stim_dict_loaded = False

        if fulldict:
            self.stim_dict = file_util.loadfile(self.stim_pkl)
        else:
            self.stim_dict = sess_load_util.load_small_stim_pkl(
                self.stim_pkl, self.runtype)

        if not self._stim_dict_loaded:
            self.stim_fps       = self.stim_dict["fps"]
            self.pre_blank      = self.stim_dict["pre_blank_sec"]  # seconds
            self.post_blank     = self.stim_dict["post_blank_sec"] # seconds
            
            pre_blank_stim_fr   = int(np.around(self.pre_blank * self.stim_fps))
            post_blank_stim_fr  = int(np.around(self.post_blank * self.stim_fps))
            self.tot_stim_fr    = (self.stim_dict["total_frames"] + 
                pre_blank_stim_fr + post_blank_stim_fr)
            self.drop_stim_fr   = self.stim_dict["droppedframes"]
            self.n_drop_stim_fr = len(self.drop_stim_fr[0])
            self.sess_stim_seed = sess_load_util.load_sess_stim_seed(
                self.stim_dict, runtype=self.runtype)

            sess_sync_util.check_drop_tolerance(
                self.n_drop_stim_fr, self.tot_stim_fr, self.droptol, 
                raise_exc=False)

        self._stim_dict_loaded = loading


    #############################################
    def _load_align_df(self):
        """
        self._load_align_df()

        Loads stimulus dataframe and alignment information.

        Attributes:
            - stim_df (pd DataFrame) : stimulus alignment dataframe with 
                                       columns:
                                         "stimType", "stimPar1", "stimPar2", 
                                         "surp", "stimSeg", "gabfr", 
                                         "start2pfr", "end2pfr", "num2pfr"
            - stimtype_order (list)  : stimulus type order
            - stim2twopfr (1D array) : 2p frame numbers for each stimulus frame, 
                                       as well as the flanking
                                       blank screen frames 
            - twop_fps (num)         : mean 2p frames per second
            - twop_fr_stim (int)     : number of 2p frames recorded while stim
                                       was playing
            - _align_df_loaded (bool): if True, alignment dataframe has been 
                                       loaded
        """

        if hasattr(self, "_align_df_loaded"):
            return

        [stim_df, stimtype_order, stim2twopfr, twop_fps, twop_fr_stim] = \
            sess_load_util.load_stim_df_info(
                self.stim_pkl, self.stim_sync_h5, self.align_pkl, self.dir, 
                self.runtype)

        self.stim_df        = stim_df
        self.stimtype_order = stimtype_order
        self.stim2twopfr    = stim2twopfr
        self.twop_fps       = twop_fps
        self.twop_fr_stim   = twop_fr_stim

        self._align_df_loaded = True


    #############################################
    def _load_sync_h5_data(self):
        """
        self._load_sync_h5_data()

        Loads the synchronisation hdf5 files for behavior and pupil.

        Attributes:
            - pup_fps (num)           : average pupil frame rate (frames per 
                                        sec)
            - pup_fr_interv (1D array): interval in sec between each pupil 
                                        frame
            - tot_pup_fr (int)        : total number of pupil frames
            - twop2bodyfr (1D array)  : body-tracking video (video-0) frame 
                                        numbers for each 2p frame
            - twop2pupfr (1D array)   : eye-tracking video (video-1) frame 
                                        numbers for each 2p frame
            - _sync_h5_loaded (bool)  : if True, info from synchronisation hdf5 
                                        files has already been loaded 
        """

        if hasattr(self, "_sync_h5_data_loaded"):
            logging.info("Sync h5 info already loaded.", 
                    extra={"spacing": TAB})
            return

        pup_fr_interv, twop2bodyfr, twop2pupfr, _ = \
            sess_load_util.load_sync_h5_data(
                self.pup_video_h5, self.time_sync_h5)
        self.pup_fr_interv = pup_fr_interv
        self.twop2bodyfr   = twop2bodyfr
        self.twop2pupfr    = twop2pupfr

        self.pup_fps = 1/(np.mean(self.pup_fr_interv))
        self.tot_pup_fr = len(self.pup_fr_interv + 1)

        self._sync_h5_loaded = True


    #############################################
    def load_run_data(self, filter_ks=5, diff_thr=50, replace=False):
        """
        self.load_run_data()

        Loads running data.

        Sets attribute with running with outliers replaced with NaNs.
            - run_data (pd Dataframe): multi-level dataframe containing pupil 
                                       data in cm/s, organized by: 
                hierarchical columns:
                    - "datatype"    : type of running data ("velocity")
                    - "filter_ks"   : kernel size used in median filtering the 
                                      running velocity
                    - "diff_thr"    : difference threshold used to identify 
                                      outliers
                    - "interpolated": whether NaNs (outliers) in 
                                      data are interpolated ("yes", "no")
                hierarchical rows:
                    - "info"        : type of information contained 
                                      ("frames": values for each frame, 
                                       "factors": scaling values for 
                                      each factor)
                    - "specific"    : specific type of information contained 
                                      (frame number, scaling factor 
                                      name)
            - tot_run_fr (1D array)  : number of running velocity frames

        Optional args:
            - filter_ks (int): kernel size to use in median filtering the 
                               running velocity (0 to skip filtering).
                               default: 5
            - diff_thr (int) : threshold of difference in running  
                               velocity to identify outliers
                               default: 50
            - replace (bool) : if True, running data is recalculated
                               default: False
        """


        if hasattr(self, "run_data"):
            prev_filter_ks = self.run_data.columns.get_level_values(
                level="filter_ks").values[0]
            prev_diff_thr = self.run_data.columns.get_level_values(
                level="diff_thr").values[0]

            modifications = []
            if filter_ks != prev_filter_ks:
                modifications.append(f"filter kernelsize {filter_ks}")

            if diff_thr != prev_diff_thr:
                modifications.append(f"difference threshold {diff_thr}")

            # check if any modifications need to be made, and if they are 
            # allowed
            if len(modifications) > 0:
                modif_str = "running dataframe using {}".format(
                    " and ".join(modifications))
                if not replace:
                    warnings.warn("Running dataframe not updated. Must set "
                        f"'replace' to True to update {modif_str}.")
                    return

                logger.info(f"Updating {modif_str}.", extra={"spacing": TAB})
            else:
                return
        
        velocity = sess_load_util.load_run_data(
            self.stim_dict, self.stim_sync_h5, filter_ks, diff_thr)
        
        row_index = pd.MultiIndex.from_product(
            [["frames"], range(len(velocity))], names=["info", "specific"])

        col_index = pd.MultiIndex.from_product(
            [["run_velocity"], ["no"], [filter_ks], [diff_thr]], 
            names=["datatype", "interpolated", "filter_ks", "diff_thr"]) 

        self.run_data = pd.DataFrame(velocity, index=row_index, 
            columns=col_index)

        sc_type = "stand_rob"
        extrem = "reg"
        sub, div = math_util.scale_fact_names(sc_type, extrem)

        self.run_data.loc[("frames", ), 
            ("run_velocity", "yes", filter_ks, diff_thr)] = \
            math_util.lin_interp_nan(self.run_data.loc[("frames", ), 
            ("run_velocity", "no", filter_ks, diff_thr)])
        for interp in ["yes", "no"]:
            run_data_array = gen_util.reshape_df_data(
                self.run_data.loc[("frames", ), ("run_velocity", interp)], 
                squeeze_rows=False, squeeze_cols=True)
            subval, divval = math_util.scale_facts(
                run_data_array, sc_type=sc_type, extrem=extrem, nanpol="omit"
                )[0:2]
            self.run_data.loc[
                ("factors", f"sub_{sub}"), ("run_velocity", interp)] = subval
            self.run_data.loc[
                ("factors", f"div_{div}"), ("run_velocity", interp)] = divval

        self.run_data = self.run_data.sort_index(axis="columns")
    
        self.tot_run_fr = len(velocity)


    #############################################
    def load_pup_data(self):
        """
        self.load_pup_data()

        If it exists, loads the pupil tracking data. Extracts pupil diameter
        and position information in pixels.

        Attributes:
            - pup_data (pd Dataframe): multi-level dataframe containing pupil 
                                       data in pixels, organized by: 
                hierarchical columns:
                    - "datatype"    : type of pupil data ("pup_diam", 
                                      "pup_center_x", "pup_center_y", 
                                      "pup_center_diff")
                    - "interpolated": whether NaNs (blinks and outliers) in 
                                      data are interpolated ("yes", "no")
                hierarchical rows:
                    - "info"        : type of information contained 
                                      ("frames": values for each frame, 
                                       "factors": scaling values for 
                                      each factor)
                    - "specific"    : specific type of information contained 
                                      (frame number, scaling factor 
                                      name)
        """

        if hasattr(self, "pup_data"):
            return
        
        pup_data = sess_load_util.load_pup_data(self.pup_data_h5)

        pup_center_diff = sess_pupil_util.get_center_dist_diff(
            pup_data["pup_center_x"], pup_data["pup_center_y"])
        pup_data["pup_center_diff"] = np.insert(pup_center_diff, 0, np.nan)

        row_index = pd.MultiIndex.from_product(
            [["frames"], pup_data["frames"]], names=["info", "specific"])

        pup_data = pup_data.drop("frames", axis="columns")

        col_index = pd.MultiIndex.from_product(
            [pup_data.columns.tolist(), ["no"]], 
            names=["datatype", "interpolated"]) 

        pup_data = pd.DataFrame(
            pup_data.values, index=row_index, columns=col_index)

        sc_type = "stand_rob"
        extrem = "reg"
        sub, div = math_util.scale_fact_names(sc_type, extrem)

        datatypes = pup_data.columns.unique(level="datatype").tolist()
        for col in datatypes:
            pup_data.loc[("frames", ), (col, "yes")] = math_util.lin_interp_nan(
                pup_data.loc[("frames", ), (col, "no")])
            for interp in ["yes", "no"]:
                subval, divval = math_util.scale_facts(
                    pup_data.loc[("frames", ), (col, interp)], 
                    sc_type=sc_type, extrem=extrem, nanpol="omit")[0:2]
                pup_data.loc[("factors", f"sub_{sub}"), (col, interp)] = subval
                pup_data.loc[("factors", f"div_{div}"), (col, interp)] = divval

        self.pup_data = pup_data.sort_index(axis="columns")
    

    #############################################
    def _set_nanrois(self, fluor="dff"):
        """
        self._set_nanrois()

        Sets attributes with the indices of ROIs containing NaNs or Infs in the
        raw or dff data.

        Attributes:
            if fluor is "dff":
                - nanrois_dff (list): list of ROIs containing NaNs or Infs in
                                      the ROI dF/F traces
            if fluor is "raw":
                - nanrois (list)    : list of ROIs containing NaNs or Infs in
                                      the ROI raw processed traces

        Optional args:
            - fluor (str): if "dff", a nanrois attribute is added for dF/F 
                           traces. If "raw, it is created for raw processed 
                           traces.
                           default: "dff"
        """
        
        rem_noisy = True

        if not hasattr(self, "_dend"):
            raise ValueError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        if fluor == "dff":
            full_trace_file = self.roi_trace_dff_h5
            dataset_name = "data"
        elif fluor == "raw":
            full_trace_file = self.roi_trace_h5
            dataset_name = "FC"
        else:
            gen_util.accepted_values_error("fluor", fluor, ["raw", "dff"])
        
        if not os.path.isfile(full_trace_file):
            raise ValueError("Specified ROI traces file does not exist: "
                             f"{full_trace_file}")
        
        with h5py.File(full_trace_file, "r") as f:
            traces = f[dataset_name][()]

        nan_arr = np.isnan(traces).any(axis=1) + np.isinf(traces).any(axis=1)

        if rem_noisy:
            min_roi = np.min(traces, axis=1)
            high_med = ((np.median(traces, axis=1) - min_roi)/\
                (np.max(traces, axis=1) - min_roi) > 0.5)

            nanmean_filt_warn = gen_util.temp_filter_warnings(np.nanmean)
            sub0_mean = nanmean_filt_warn(traces, axis=1, 
                msgs=["Mean of empty slice"], categs=[RuntimeWarning]) < 0
            
            warn_str = "None"
            roi_ns = np.where(high_med + sub0_mean)[0]
            if len(roi_ns) != 0:
                warn_str = ", ".join([str(x) for x in roi_ns])
            logger.warning("Noisy ROIs (mean below 0, median above "
                "midrange) are also included in NaN ROI attributes "
                f"(but not set to NaN): {warn_str}.", extra={"spacing": TAB})
            
            nan_arr += high_med + sub0_mean

        nan_rois = np.where(nan_arr)[0].tolist()

        if fluor == "dff":
            self.nanrois_dff = nan_rois
        elif fluor == "raw":
            self.nanrois = nan_rois


#############################################
    def _set_matched_rois(self):
        """
        self._set_matched_rois()

        Sets attribute with the indices of ROIs that have been matched across 
        sessions 

        Attributes:
            - matched_rois (1D array): ordered indices of ROIs matched across 
                                       sessions
        """

        if self.plane == "dend" and self.dend != "extr":
            raise UserWarning("ROIs not matched for Allen extracted dendritic "
                "ROIs.")

        try:
            nway_match_path = sess_file_util.get_nway_match_path_from_sessid(
                self.home, self.sessid, self.runtype, check=True)
        except Exception as err:
            if "not exist" in str(err):
                raise UserWarning(f"No matched ROIs file found for {self}.")
            else:
                raise err


        with open(nway_match_path, 'r') as fp:
            matched_rois_df = pd.DataFrame(json.load(fp)['rois'])

        self.matched_rois = matched_rois_df['dff-ordered_roi_index'].values


    #############################################
    def _init_roi_facts_df(self, fluor="dff"):
        """
        self._init_roi_facts_df()

        Initializes the ROIs factors dataframe.

        Attributes:
            - roi_facts_df (pd DataFrame): multi-level dataframe containing ROI 
                                           scaling factors, organized 
                                           by: 
                hierarchical columns:
                    - "datatype"    : type of ROI data ("roi_traces")
                    - "fluorescence": type of ROI trace ("dff", "raw")
                hierarchical rows:
                    - "factors"     : scaling factor name
                    - "ROIs"        : ROI index

        Optional args:
            - fluor (str or list): the fluorescence column(s) to initialize.
                                   default: "dff"
        """

        if not hasattr(self, "roi_facts_df"):
            fluor = gen_util.list_if_not(fluor)
            sc_type = "stand_rob"
            extrem = "perc"
            
            sub, div = math_util.scale_fact_names(sc_type, extrem)

            col_index = pd.MultiIndex.from_product(
                [["roi_traces"], fluor], names=["datatype", "fluorescence"])
            row_index = pd.MultiIndex.from_product(
                [[f"sub_{sub}", f"div_{div}"], range(self.nrois)], 
                names=["factors", "ROIs"])
    
            self.roi_facts_df = pd.DataFrame(
                None, index=row_index, columns=col_index)
 

    ############################################
    def _set_roi_attributes(self, fluor="dff"):
        """
        self._set_roi_attributes()

        Loads the processed trace information, and updates self.roi_facts_df
        for the requested fluorescence data.

        Calls:
            - self._init_roi_facts_df()
            - self._set_nanrois()
            if self.only_matched_rois:
            - self._set_matched_rois()
        """

        if not hasattr(self, "nrois"):
            raise ValueError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        if hasattr(self, "roi_facts_df"):
            if fluor in self.roi_facts_df.columns.get_level_values(
                level="fluorescence"):
                return
        
        self._init_roi_facts_df(fluor=fluor)

        if fluor == "dff":
            roi_trace_use = self.roi_trace_dff_h5
            dataset = "data"
        elif fluor == "raw":
            roi_trace_use = self.roi_trace_h5
            dataset = "FC"
        else:
            gen_util.accepted_values_error("fluor", fluor, ["raw", "dff"])

        file_util.checkfile(roi_trace_use)

        # get scaling facts (factor x ROI)
        with h5py.File(roi_trace_use, "r") as f:
            
            # obtain scaling facts while filtering All-NaN warning.
            scale_facts_filt_warn = gen_util.temp_filter_warnings(
                math_util.scale_facts)

            self.roi_facts_df[("roi_traces", fluor)] = np.asarray(
                scale_facts_filt_warn(f[dataset][()], axis=1, 
                sc_type="stand_rob", extrem="perc", nanpol="omit",
                allow_0=True, msgs="All-NaN", categs=RuntimeWarning
                )[0:2]).reshape(-1)

        self._set_nanrois(fluor)
        if self.only_matched_rois:
            self._set_matched_rois()


    #############################################
    def _get_roi_facts(self, fluor="dff"):
        """
        self._get_roi_facts()

        Returns scaling factors dataframe for ROIs for specified
        fluorescence type(s).

        Calls:
            - self._set_roi_attributes()

        Optional args:
            - fluor (str or list): type(s) of fluorescence for which to return 
                                   scaling values ("dff", "raw")
                                   default: "dff"

        Returns:
            - specific_roi_facts (pd DataFrame): multi-level dataframe 
                                                 containing ROI scaling 
                                                 factors, organized by: 
                hierarchical columns (dummy):
                    - "datatype"    : type of ROI data ("roi_traces")
                    - "fluorescence": type of ROI trace ("dff", "raw")
                hierarchical rows:
                    - "factors"     : scaling factor name
                    - "ROIs"        : ROI index
        """

        fluors = gen_util.list_if_not(fluor)

        for fluor in fluors:
            if hasattr(self, "roi_facts_df"):
                if fluor in self.roi_facts_df.columns.get_level_values(
                    level="fluorescence"):
                    continue
            self._set_roi_attributes(fluor)
        
        specific_roi_facts = self.roi_facts_df.copy(deep=True)
        
        drop_fluors = list(filter(lambda x: x not in fluors, 
            self.roi_facts_df.columns.unique("fluorescence")))

        if len(drop_fluors) != 0:
            specific_roi_facts = self.roi_facts_df.drop(
                drop_fluors, axis="columns", level="fluorescence")
        
        return specific_roi_facts


    #############################################
    def _set_dend_type(self, dend="extr", fluor="dff"):
        """
        self._set_dend_type()

        Sets the dendritic type based on the requested type, plane and 
        whether the corresponding files are found.

        NOTE: "fluor" parameter should not make a difference to the content of 
        the attributes set. It just allows for the code to run when only "raw" 
        or  "dff" traces are present in the data directory.

        NOTE: Once the 

        Attributes:
            - _dend (str)           : type of dendrites loaded 
                                      ("allen" or "extr")
            if EXTRACT dendrites are used, updates:
            - roi_mask_file (str)   : path to ROI mask h5
            - roi_trace_h5 (str)    : full path name of the ROI raw 
                                      processed fluorescence trace hdf5 file
            - roi_trace_dff_h5 (str): full path name of the ROI dF/F
                                      fluorescence trace hdf5 file


        Optional args:
            - dend (str) : dendritic traces to use ("allen" for the 
                           original extracted traces and "extr" for the
                           ones extracted with Hakan's EXTRACT code, if
                           available)
                           default: "extr"
            - fluor (str): if "dff", ROI information is collected from dF/F 
                           trace file. If "raw", based on the raw processed 
                           trace file. 
                           default: "dff"
        
        Raises:
            - ValueError if 'dend' has already been set and is being changed 
            (only checked for dendritic plane data.)
        """

        if hasattr(self, "_dend"):
            if self.plane == "dend" and self.dend != dend:
                raise ValueError("Cannot change dendrite type. Already set "
                f"to {self.dend} traces.")
            return

        if dend not in ["extr", "allen"]:
            gen_util.accepted_values_error("dend", dend, ["extr", "allen"])

        self._dend = "allen"
        if self.plane == "dend" and dend == "extr":
            try:
                dend_roi_trace_h5 = sess_file_util.get_dendritic_trace_path(
                    self.roi_trace_h5, check=(fluor=="raw"))
                dend_roi_trace_dff_h5 = sess_file_util.get_dendritic_trace_path(
                    self.roi_trace_dff_h5, check=(fluor=="dff"))
                dend_mask_file = sess_file_util.get_dendritic_mask_path(
                    self.home, self.sessid, self.expid, self.mouseid, 
                    self.runtype, self.mouse_dir, check=True)
                
                self._dend = "extr"
                self.roi_trace_h5     = dend_roi_trace_h5
                self.roi_trace_dff_h5 = dend_roi_trace_dff_h5
                self.roi_mask_file    = dend_mask_file

            except Exception as e:
                warnings.warn(f"{e}.\Allen extracted dendritic ROIs "
                    "will be used instead.")


    ############################################
    def get_stim_twop_fr_ns(self):
        """
        get_stim_twop_fr_ns()

        Returns the 2p frame numbers that occur during stimuli (excluding all 
        grayscreen presentations, including Gabor grayscreen frames)
        
        Returns:
            - all_stim_twop_fr_ns (1D array): 2p frame numbers during which 
                                              stimuli are presented
        """

        start_twop_frs = self.stim_df["start2pfr"]
        end_twop_frs = self.stim_df["end2pfr"]

        all_stim_twop_fr_ns = np.concatenate([
            np.arange(start, end) 
            for start, end in zip(start_twop_frs, end_twop_frs)])

        return all_stim_twop_fr_ns


    #############################################
    def get_frames_timestamps(self, pre, post, fr_type="twop"):
        """
        self.get_frames_timestamps(pre, post)

        Returns range of frames and time stamps for the given pre to post 
        time values.

        Require args:
            - pre (num)         : range of frames to include before each 
                                  reference frame number (in s)
            - post (num)        : range of frames to include after each 
                                  reference frame number (in s)

        Optional args:
            - fr_type (str) : type of frames to calculate for
                              default: "stim"

        Returns:
            - ran_fr (list)        : relative frame range [-pre, post]
            - timestamps (1D array): time stamps for each frame
        """

        if fr_type == "twop":
            fps = self.twop_fps
        elif fr_type == "stim":
            fps = self.stim_fps
        elif fr_type == "pup":
            fps = self.pup_fps

        ran_fr = [np.around(x * fps) for x in [-pre, post]]
        timestamps = np.linspace(-pre, post, int(np.diff(ran_fr)[0]))

        return ran_fr, timestamps 


    #############################################
    def load_roi_info(self, fluor="dff", dend="extr"): 
        """
        self.load_roi_info()

        Sets the attributes below based on the specified processed ROI traces.
        
        NOTE: "fluor" parameter should not make a difference to the content of 
        the attributes set. It just allows not both "raw" and "dff" traces to
        be present in the data directory.
        
        Calls:
            - self._set_dend_type()
            - self._set_roi_attributes()

        Attributes:
            - nrois (int)         : number of ROIs in traces
            - roi_names (list)    : list of ROI names (9 digits)
            - tot_twop_fr (int)   : number of 2p frames recorded

        Optional args:
            - fluor (str): if "dff", ROI information is collected from dF/F 
                           trace file. If "raw", based on the raw processed 
                           trace file. 
                           default: "dff"
            - dend (str) : dendritic traces to use ("allen" for the 
                           original extracted traces and "extr" for the
                           ones extracted with Hakan's EXTRACT code, if
                           available)
                           default: "extr"
        """

        self._set_dend_type(dend=dend, fluor=fluor)

        if not hasattr(self, "roi_names"): # do this only first time
            if fluor == "raw":
                use_roi_file = self.roi_trace_h5
                dataset_name = "FC"

            elif fluor == "dff":
                use_roi_file = self.roi_trace_dff_h5
                dataset_name = "data"
            else:
                gen_util.accepted_values_error("fluor", fluor, ["raw", "dff"])
            
            try:
                # open the roi file and get the info
                with h5py.File(use_roi_file, "r") as f:
                    # get the names of the rois
                    self.roi_names = f["roi_names"][()].tolist()

                    # get the number of rois
                    self.nrois = f[dataset_name].shape[0]

                    # get the number of data points in the traces
                    self.tot_twop_fr = f[dataset_name].shape[1]

            except Exception as err:
                raise OSError(f"Could not open {use_roi_file} for reading: "
                    f"{err}.")
        
        self._set_roi_attributes(fluor)


    #############################################
    def _load_stims(self):
        """
        self._load_stims()
        
        Initializes attributes, including Stim objects (Gabors, Bricks, Grayscr)

        Attributes:
            - bricks (list or Bricks object): session bricks object, if 
                                              runtype is "pilot" or list
                                              of session bricks objects if 
                                              runtype is "prod"
            - gabors (Gabors object)        : session gabors object
            - grayscr (Grayscr object)      : session grayscreen object
            - n_stims (int)                 : number of stimulus objects in
                                              the session (2 bricks stims
                                              in production data count as one)
            - stimtypes (list)              : list of stimulus type names 
                                              (i.e., "gabors", "bricks")
            - stims (list)                  : list of stimulus objects in the
                                              session
        """

        if hasattr(self, "stimtypes"):
            return

        # create the stimulus fields and objects
        self.stimtypes = []
        self.n_stims    = len(self.stim_dict["stimuli"])
        self.stims      = []
        if self.runtype == "prod":
            n_bri = []
        for i in range(self.n_stims):
            stim = self.stim_dict["stimuli"][i]
            if self.runtype == "pilot":
                stimtype = stim["stimParams"]["elemParams"]["name"]
            elif self.runtype == "prod":
                stimtype = stim["stim_params"]["elemParams"]["name"]
            # initialize a Gabors object
            if stimtype == "gabors":
                self.stimtypes.append(stimtype)
                self.gabors = Gabors(self, i)
                self.stims.append(self.gabors)
            # initialize a Bricks object
            elif stimtype == "bricks":
                if self.runtype == "prod":
                    n_bri.append(i)
                    # 2 brick stimuli are recorded in the production data, but 
                    # are merged to initialize one stimulus object
                    if len(n_bri) == 2:
                        self.stimtypes.append(stimtype)
                        self.bricks = Bricks(self, n_bri)
                        self.stims.append(self.bricks)
                        self.n_stims = self.n_stims - 1
                        n_bri = []
                elif self.runtype == "pilot":
                    self.stimtypes.append(stimtype)
                    self.bricks = Bricks(self, i)
                    self.stims.append(self.bricks)
                
            else:
                logger.info(f"{stimtype} stimulus type not recognized. No Stim " 
                    "object created for this stimulus. \n", 
                    extra={"spacing": TAB})

        # initialize a Grayscr object
        self.grayscr = Grayscr(self)

    
    #############################################
    def extract_sess_attribs(self, mouse_df="mouse_df.csv"):
        """
        self.extract_sess_attribs(mouse_df)

        This function should be run immediately after creating a Session 
        object. It loads the dataframe containing information on each session,
        and sets attributes.

        Attributes:
            - all_files (bool) : if True, all files have been acquired for
                                 the session
            - any_files (bool) : if True, some files have been acquired for
                                 the session
            - depth (int)      : recording depth 
            - plane (str)      : recording plane ("soma" or "dend")
            - line (str)       : mouse line (e.g., "L5-Rbp4")
            - mouse_n (int)    : mouse number (e.g., 1)
            - notes (str)      : notes from the dataframe on the session
            - pass_fail (str)  : whether session passed "P" or failed "F" 
                                 quality control
            - sess_gen (int)   : general session number (e.g., 1)
            - sess_within (int): within session number (session number within
                                 the sess_gen) (e.g., 1)
            - sess_n (int)     : overall session number (e.g., 1)

        Required args:
        - mouse_df (str): path name of dataframe containing information on each 
                          session.
        """

        df_data = sess_load_util.load_info_from_mouse_df(self.sessid, mouse_df)

        self.mouse_n      = df_data["mouse_n"]
        self.depth        = df_data["depth"]
        self.plane        = df_data["plane"]
        self.line         = df_data["line"]
        self.sess_gen     = df_data["sess_gen"]
        self.sess_n       = df_data["sess_n"]
        self.sess_within  = df_data["sess_within"]
        self.pass_fail    = df_data["pass_fail"]
        self.all_files    = df_data["all_files"]
        self.any_files    = df_data["any_files"]
        self.notes        = df_data["notes"]


    #############################################
    def extract_info(self, fulldict=True, fluor="dff", dend="extr", roi=True, 
                     run=False, pupil=False):
        """
        self.extract_info()

        This function should be run immediately after creating a Session 
        object and running self.extract_sess_attribs(). It creates the 
        stimulus objects attached to the Session, and loads the ROI traces, 
        running data, synchronization data, etc. If stimtypes have not been 
        initialized, also initializes stimtypes.

        Calls:
            self._load_stim_dict()
            self._load_align_df()
            self._load_sync_h5_data()
            self._load_stims()

            optionally:
            self.load_roi_info()
            self.load_run_data()
            self.load_pup_data()

        Optional args:
            - fulldict (bool): if True, the full stim_dict is loaded,
                               else the small stim_dict is loaded
                               (which contains everything, except "posbyframe" 
                               for Bricks)
                               default: True
            - fluor (str)    : if "dff", ROI information is loaded from dF/F 
                               trace file. If "raw", based on the raw processed 
                               trace file. 
                               default: "dff"
            - dend (str)     : dendritic traces to use ("allen" for the 
                               original extracted traces and "extr" for the
                               ones extracted with Hakan's EXTRACT code, if
                               available). Can only be set the first time 
                               ROIs are loaded to the session. 
                               default: "extr"
            - roi (bool)     : if True, ROI data is loaded
                               default: True
            - run (bool)     : if True, running data is loaded
                               default: False
            - pup (bool)     : if True, pupil data is loaded
                               default: False
        """

        if not hasattr(self, "plane"):
            raise ValueError("Session attributes missing to extract info. "
                "Make sure to run self.extract_sess_attribs() first.")

        # load the stimulus, running, alignment and trace information         
        logger.info("Loading stimulus dictionary...", extra={"spacing": "\n"})
        self._load_stim_dict(fulldict=fulldict)
        
        logger.info(f"Loading alignment dataframe...")
        self._load_align_df()
         
        logger.info("Loading sync h5 info...")
        self._load_sync_h5_data()

        logger.info("Creating stimulus objects...")
        self._load_stims()

        if roi:
            logger.info("Loading ROI trace info...")
            self.load_roi_info(fluor=fluor, dend=dend)

        if run:
            logger.info("Loading running info...")
            self.load_run_data()

        if pupil:
            logger.info("Loading pupil info...")
            self.load_pup_data()


    #############################################
    def get_stim(self, stimtype="gabors"):
        """
        self.get_stim()

        Returns the requested Stim object, if it is an attribute of the 
        Session.

        Required args:
            - sess (Session): Session object

        Optional args:
            - stimtype (str): stimulus type to return ("bricks", "gabors" or 
                              "grayscr")
                              default: "gabors"

        Return:
            - stim (Stim): Stim object (either Gabors or Bricks)
        """


        if stimtype == "gabors":
            if hasattr(self, "gabors"):
                stim = self.gabors
            else:
                raise ValueError("Session object has no gabors stimulus.")
        elif stimtype == "bricks":
            if hasattr(self, "bricks"):
                stim = self.bricks
            else:
                raise ValueError("Session object has no bricks stimulus.")
        elif stimtype == "grayscr":
            if hasattr(self, "grayscr"):
                stim = self.grayscr
            else:
                raise ValueError("Session object has no grayscr stimulus.")
        else:
            gen_util.accepted_values_error("stimtype", stimtype, 
                ["gabors", "bricks", "grayscr"])
        
        return stim


    #############################################
    def get_pup_data(self, datatype="pup_diam", remnans=True, scale=False):
        """
        self.get_pup_data()

        Returns the correct full pupil data array based on whether NaNs are
        to be removed or not. 

        Optional args:
            - datatype (str): type of pupil data to return ("pup_diam", 
                              "pup_center_x", "pup_center_y", "pup_center_diff")
                              default: "pup_diam"
            - remnans (bool): if True, the full pupil array in which NaN 
                              values have been removed using linear 
                              interpolation is returned. If False, the non
                              interpolated pupil array is returned.
                              default: True
            - scale (bool)  : if True, pupil data is scaled using full 
                              data array
                              default: False      

        Returns:
            - pup_data_df (pd DataFrame): dataframe containing pupil diameter 
                                          values (in pixels) for the frames
                                          of interest, organized by:
                hierarchical columns (all dummy):
                    - datatype    : type of data (e.g., "pup_diam")
                    - interpolated: whether data is interpolated ("yes", "no")
                    - scaled      : whether data is scaled ("yes", "no")
                hierarchical rows:
                    - info        : type of information contained 
                                    ("frames": values for each frame)
                    - "specific"  : specific type of information contained 
                                    (frame number)
        """

        if not hasattr(self, "pup_data"):
            self.load_pup_data()

        interpolated = "no"
        if remnans:
            interpolated ="yes"

        datatypes = self.pup_data.columns.unique(level="datatype").tolist()

        if datatype not in datatypes:
            if f"pup_{datatype}" in datatypes:
                datatype = f"pup_{datatype}"
            else:
                gen_util.accepted_values_error("datatype", datatype, datatypes)
        
        index = pd.MultiIndex.from_product(
            [[datatype], [interpolated], ["no"]], 
            names=["datatype", "interpolated", "scaled"])

        pup_data_df = pd.DataFrame(self.pup_data, columns=index)
        
        if scale:
            pup_data_df = sess_data_util.scale_data_df(
                pup_data_df, datatype, interpolated)
            pup_data_df = pup_data_df.drop(
                labels="no", axis="columns", level="scaled")

        pup_data_df = pup_data_df.drop("factors", axis="index")

        return pup_data_df


    #############################################
    def get_run_velocity(self, remnans=True, scale=False):
        """
        self.get_run_velocity()

        Returns the correct full running velocity array based on whether NaNs 
        are to be removed or not. 

        Optional args:
            - remnans (bool): if True, the full running array in which NaN 
                              values have been removed using linear 
                              interpolation is returned. If False, the non
                              interpolated running array is returned.
                              default: True
            - scale (bool)  : if True, running is scaled based on 
                              full trace array
                              default: False

        Returns:
            - run_data_df (pd DataFrame): dataframe containing running velocity 
                                          values (in cm/s) for the frames
                                          of interest, organized by:
                hierarchical columns (all dummy):
                    - datatype    : type of data (e.g., "run_velocity")
                    - interpolated: whether data is interpolated ("yes", "no")
                    - scaled      : whether data is scaled ("yes", "no")
                    - filter_ks   : kernel size used to median filter running 
                                    velocity data
                    - diff_thr    : threshold of difference in running velocity 
                                    used to identify outliers
                hierarchical rows:
                    - "info"      : type of information contained 
                                    ("frames": values for each frame)
                    - "specific   : specific type of information contained 
                                    (frame number)
        """

        if not hasattr(self, "run_data"):
            raise ValueError("Run 'self.load_run_data()' to load the running "
                "data correctly.")

        interpolated = "no"
        if remnans:
            interpolated ="yes"
        
        datatype = "run_velocity"
        filter_ks = self.run_data[
            (datatype, interpolated)].columns.get_level_values(
            level="filter_ks")[0]

        diff_thr = self.run_data[
            (datatype, interpolated)].columns.get_level_values(
            level="diff_thr")[0]

        names = ["datatype", "interpolated", "filter_ks", "diff_thr", "scaled"]
        reorder = [names[i] for i in [0, 1, 4, 2, 3]]
        index = pd.MultiIndex.from_product(
            [[datatype], [interpolated], [filter_ks], [diff_thr], ["no"]], 
            names=names)

        run_data_df = pd.DataFrame(self.run_data, columns=index).reorder_levels(
            reorder, axis="columns")

        if scale:
            run_data_df = sess_data_util.scale_data_df(
                run_data_df, datatype, interpolated, 
                other_vals=[filter_ks, diff_thr])
            run_data_df = run_data_df.drop(
                labels="no", axis="columns", level="scaled")

        run_data_df = run_data_df.drop("factors", axis="index")

        return run_data_df


    #############################################
    def get_run_velocity_by_fr(self, fr, fr_type="stim", remnans=True, 
                               scale=False):
        """
        self.get_run_velocity_by_fr(fr)

        Returns the running velocity for the given frames, either stimulus 
        frames or two-photon imaging frames using linear interpolation.

        Required args:
            - fr (array-like): set of frames for which to get running velocity
        
        Optional args:
            - fr_type (str) : type of frames passed ("stim" or "twop" frames)
                              default: "stim"
            - remnans (bool): if True, NaN values are removed using linear 
                              interpolation.
                              default: True
            - scale (bool)  : if True, running is scaled based on 
                              full trace array
                              default: False
        Returns:
            - run_data_df (pd DataFrame): dataframe containing running velocity 
                                          values (in cm/s) for the frames
                                          of interest, organized by:
                hierarchical columns (all dummy):
                    - datatype    : type of data (e.g., "run_velocity")
                    - interpolated: whether data is interpolated ("yes", "no")
                    - scaled      : whether data is scaled ("yes", "no")
                hierarchical rows:
                    - "sequences" : sequence numbers
                    - "frames"    : frame numbers
        """

        if not hasattr(self, "run_data"):
            raise ValueError("Run 'self.load_run_data()' to load the running "
                "data correctly.")

        fr = np.asarray(fr)

        if fr_type == "stim":
            max_val = self.tot_run_fr
        elif fr_type == "twop":
            max_val = self.tot_twop_fr
        else:
            gen_util.accepted_values_error(
                "fr_type", fr_type, ["stim", "twop"])

        if (fr >= max_val).any() or (fr < 0).any():
            raise UserWarning("Some of the specified frames are out of range")
        
        run_data = self.get_run_velocity(remnans=remnans, scale=scale)

        if fr_type == "stim":
            velocity = run_data.to_numpy()[fr]
        elif fr_type == "twop":
            velocity = np.interp(fr, self.stim2twopfr, run_data.to_numpy())

        index = pd.MultiIndex.from_product(
            [range(velocity.shape[0]), range(velocity.shape[1])], 
            names=["sequences", "frames"])

        run_data_df = pd.DataFrame(
            velocity.reshape(-1), columns=run_data.columns, index=index)

        return run_data_df


    #############################################
    def get_nanrois(self, fluor="dff"):
        """
        self.get_nanrois()

        Returns as a list the indices of ROIs containing NaNs or Infs.

        Optional args:
            - fluor (str): if "dff", remnans is assessed on ROIs using dF/F 
                           traces. If "raw", on raw processed traces.
                           default: "dff"
        Returns:
            - (list): indices of ROIs containing NaNs or Infs
        """

        if fluor == "dff":
            if not hasattr(self, "nanrois_dff"):
                self._set_nanrois(fluor)
            return self.nanrois_dff
        elif fluor == "raw":
            if not hasattr(self, "nanrois"):
                self._set_nanrois(fluor)
            return self.nanrois
        else:
            gen_util.accepted_values_error("fluor", fluor, ["raw", "dff"])


    #############################################
    def get_matched_rois(self):
        """
        self.get_matched_rois()

        Returns as a numpy array the indices of ROIs that have been
        matched across sessions (currently, across all sessions for
        which we have data).
        """

        if not hasattr(self, 'matched_rois'):
            self._set_matched_rois()
        
        return self.matched_rois


    #############################################
    def get_nrois(self, remnans=True, fluor="dff"):
        """
        self.get_nrois()

        Returns the number of ROIs according to the specified criteria.

        Optional args:
            - remnans (bool): if True, ROIs with NaN/Inf values are excluded
                              from number.
                              default: True
            - fluor (str)   : if "dff", the indices of ROIs with NaNs or Infs 
                              in the dF/F traces are returned. If "raw", for 
                              raw processed traces.
                              default: "dff"
        Returns:
            - nrois (int): number of ROIs fitting criteria
        """

        if not hasattr(self, "nrois"):
            raise ValueError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        nrois = self.nrois
        if remnans:
            rem_rois = len(self.get_nanrois(fluor))
            nrois = nrois - rem_rois

        return nrois
        

    #############################################
    def get_active_rois(self, fluor="dff", stimtype=None, remnans=True):
        """
        self.active_rois()

        Returns as a list the indices of ROIs that have calcium transients 
        (defined as median + 3 std), optionally during a specific stimulus type.

        Optional args:
            - fluor (str)   : if "dff", the indices of ROIs with NaNs or Infs 
                              in the dF/F traces are returned. If "raw", for 
                              raw processed traces.
                              default: "dff"
            - stimtype (str): stimulus type during which to check for 
                              transients ("bricks", "gabors" or None). If None,
                              the entire session is checked.
                              default: None
            - remnans (bool): if True, the indices ignore ROIs containg NaNs or 
                              Infs
                              default: True
        Returns:
            - active_roi_indices (list): indices of active ROIs
        """

        if not hasattr(self, "nrois"):
            raise ValueError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        logger.info("Identifying active ROIs.", extra={"spacing": "\n"})

        win = [1, 5]
        
        full_data = self.get_roi_traces(None, fluor, remnans)

        full_data_sm = scsig.medfilt(
            gen_util.reshape_df_data(
                full_data, squeeze_rows=False, squeeze_cols=True
                ), win)
        med = np.nanmedian(full_data_sm, axis=1) # smooth full data median
        std = np.nanstd(full_data_sm, axis=1) # smooth full data std

        if stimtype is None:
            stim_data = full_data
            stim_data_sm = full_data_sm
        else:
            stim = self.get_stim(stimtype)
            twop_fr = []
            blocks = stim.block_params.index.unique("block_n")
            for b in blocks:
                row = stim.block_params.loc[pd.IndexSlice[:, b]]
                twop_fr.extend([row["start_twop_fr"][0], row["end_twop_fr"][0]])
            stim_data = self.get_roi_traces(twop_fr, fluor, remnans)
            stim_data_sm = scsig.medfilt(stim_data, win)

        # count how many calcium transients occur in the data of interest for
        # each ROI and identify inactive ROIs
        diff = stim_data_sm - (med + 3 * std)[:, np.newaxis]
        counts = np.sum(diff > 0, axis=1)
        active = np.where(counts != 0)[0]

        active_roi_indices = full_data.index.unique("ROIs")[active].tolist()

        return active_roi_indices


    #############################################
    def get_plateau_roi_traces(self, n_consec=4, thr_ratio=3, fluor="dff", 
                               remnans=True, replace=False):
        """
        self.get_plateau_roi_traces()

        Returns modified ROI traces thresholded, so that values that do not 
        reach criteria are set to median.

        Attributes:
            - plateau_traces (2D array): ROI traces converted to plateau traces, 
                                         ROI x frames


        Optional args:
            - n_consec (int)   : number of consecutive above threshold (3 std) 
                                 frames to be considered a plateau potential
                                 default: 4
            - thr_ratio (float): number of standard deviations above median 
                                 at which threshold is set for identifying 
                                 calcium transients
                                 default: 3
            - fluor (str)      : if "dff", then dF/F traces are returned, if 
                                 "raw", raw processed traces are returned
                                 default: "dff"
            - remnans (bool)   : if True, the indices ignore ROIs containg NaNs 
                                 or Infs
                                 default: True

        Returns:
            - plateau_traces: modified ROI traces where frames below 
                              threshold, or where trace does not remain above
                              threshold for minimum number of frames are set to
                              the median. Frames reaching criteria are 
                              converted to number of standard deviations above 
                              median.
        """

        if not hasattr(self, "nrois"):
            raise ValueError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        if not hasattr(self, "plateau_traces"):
            logger.info("Retrieving plateau traces.", extra={"spacing": "\n"})

            plateau_traces = gen_util.reshape_df_data(
                self.get_roi_traces(None, fluor, remnans), squeeze_cols=True)
            med = np.nanmedian(plateau_traces, axis=1)
            std = np.nanstd(plateau_traces, axis=1)

            for r, roi_data in enumerate(plateau_traces):
                roi_bool = ((roi_data - med[r])/std[r] >= thr_ratio)
                idx = np.where(roi_bool)[0]
                each_first_idx = np.where(np.insert(np.diff(idx), 0, 100) > 1)[0]
                drop_break_pts = np.where(np.diff(each_first_idx) < n_consec)[0]
                for d in drop_break_pts: 
                    set_zero_indices = np.arange(
                        idx[each_first_idx[d]], 
                        idx[each_first_idx[d + 1] - 1] + 1)
                    roi_bool[set_zero_indices] = False
                plateau_traces[r, ~roi_bool] = 1.0
                plateau_traces[r, roi_bool] = \
                    (plateau_traces[r, roi_bool] - med[r])/std[r]

            self.plateau_traces = plateau_traces

        return self.plateau_traces


    #############################################
    def get_roi_traces(self, frames=None, fluor="dff", remnans=True, 
                       scale=False):
        """
        self.get_roi_traces()

        Returns the processed ROI traces for the given two-photon imaging
        frames and specified ROIs.

        Optional args:
            - frames (int array): set of 2p imaging frames (1D) to give ROI 
                                  dF/F for. The order is not changed, so frames
                                  within a sequence should already be properly 
                                  sorted (likely ascending). If None, then all 
                                  frames are returned. 
                                  default: None
            - fluor (str)       : if "dff", then dF/F traces are returned, if 
                                  "raw", raw processed traces are returned
                                  default: "dff"
            - remnans (bool)    : if True, ROIs with NaN/Inf values anywhere 
                                  in session are excluded. 
                                  default: True
            - scale (bool)      : if True, each ROIs is scaled 
                                  based on full data array
                                  default: False
        Returns:
            - roi_data_df (pd DataFrame): dataframe containing ROI trace data  
                                          for the frames of interest, organized 
                                          by:
                hierarchical columns (all dummy):
                    - datatype        : type of data (e.g., "roi_traces")
                    - nan_rois_removed: whether ROIs with NaNs/Infs were 
                                        removed ("yes", "no")
                    - scaled          : whether ROI data is scaled 
                                        ("yes", "no")
                    - fluorescence    : type of data ("raw" or "dff")
                hierarchical rows:
                    - ROIs            : ROI indices
                    - frames          : last frames dimensions
        """

        if not hasattr(self, "nrois"):
            raise ValueError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        # check whether the frames to retrieve are within range
        if frames is None:
            frames = np.arange(self.tot_twop_fr)
        elif max(frames) >= self.tot_twop_fr or min(frames) < 0:
            raise UserWarning("Some of the specified frames are out of range")
        else:
            frames = np.asarray(frames)

        remnans_str = "yes" if remnans else "no"
        scale_str = "yes" if scale else "no"

        # read the data points into the return array
        if fluor == "dff":
            roi_trace_h5 = self.roi_trace_dff_h5
            dataset_name = "data"
        elif fluor == "raw":
            roi_trace_h5 = self.roi_trace_h5
            dataset_name = "FC"
        else:
            gen_util.accepted_values_error("fluor", fluor, ["raw", "dff"])

        with h5py.File(roi_trace_h5, "r") as f:
            try:
                # avoid loading full dataset if frames are strictly increasing
                if np.min(np.diff(frames)) > 0:
                    traces = f[dataset_name][:, frames]
                else:
                    traces = f[dataset_name][()][:, frames]
            except Exception as err:
                raise OSError(f"Could not read {self.roi_trace_h5}: {err}")

        if scale:
            factors = self._get_roi_facts(fluor)
            factor_names = factors.index.unique(level="factors")
            sub_names =  list(filter(lambda x: "sub" in x, factor_names))
            if len(sub_names) != 1:
                raise ValueError("Only one factor should contain 'sub'.")
            div_names =  list(filter(lambda x: "div" in x, factor_names))
            if len(div_names) != 1:
                raise ValueError("Only one row should contain 'div'.")
            traces = (traces - factors.loc[sub_names[0]].values)/factors.loc[
                div_names[0]].values

        # do this BEFORE building dataframe - much faster
        if self.only_matched_rois:
            ROI_ids = self.get_matched_rois()
            traces = traces[ROI_ids]
        else:
            ROI_ids = np.arange(self.nrois)
            if remnans:
                rem_rois = self.get_nanrois(fluor)
                # remove ROIs with NaNs or Infs in dataframe
                if len(rem_rois):
                    ROI_ids = np.asarray(sorted(set(ROI_ids) - set(rem_rois)))
                    traces = traces[ROI_ids]

        # initialize the return dataframe
        index_cols = pd.MultiIndex.from_product(
            [["roi_traces"], [remnans_str], [scale_str], [fluor]], 
            names=["datatype", "nan_rois_removed", "scaled", 
            "fluorescence"])
        index_rows = pd.MultiIndex.from_product(
            [ROI_ids, *[range(dim) for dim in frames.shape]], 
            names=["ROIs", "frames"])
        
        roi_data_df = pd.DataFrame(
            traces.reshape(-1), index=index_rows, columns=index_cols)

        return roi_data_df


    #############################################
    def get_twop_fr_ran(self, twop_ref_fr, pre, post, pad=(0, 0)):
        """
        self.get_twop_fr_ran(twop_ref_fr, pre, post)
        
        Returns an array of 2p frame numbers, where each row is a sequence and
        each sequence ranges from pre to post around the specified reference 
        2p frame numbers. 

        Required args:
            - twop_ref_fr (list): 1D list of 2p frame numbers 
                                  (e.g., all 1st seg frames)
            - pre (num)         : range of frames to include before each 
                                  reference frame number (in s)
            - post (num)        : range of frames to include after each 
                                  reference frame number (in s)
        
        Optional args:
            - pad (tuple): number of frame to use as padding (before, after)
                           default: (0, 0)

        Returns:
            - frame_n_df (pd DataFrame): Dataframe of 2-photon frame numbers, 
                                         organized with:
                columns: 
                    - twop_fr_n: 2-photon frame numbers
                hierarchical rows:
                    - sequences  : sequence number
                    - time_values: time values for each frame
        """

        if not hasattr(self, "twop_fps"):
            raise ValueError("Run 'self.load_roi_info()' to load the ROI "
                "attributes correctly.")    

        ran_fr, xran = self.get_frames_timestamps(pre, post, fr_type="twop")

        # adjust for padding
        if len(pad) != 2:
            raise ValueError("Padding must have length 2.")
        if min(pad) < 0:
            raise ValueError("Padding cannot be negative")
        if pad != (0, 0):
            if sum(pad) > len(xran)/10.:
                warnings.warn("Proportionally high padding values may distort "
                    "time values as method is designed to preserve 'pre' and "
                    "'post' values in time stamps.")
            pad = [int(val) for val in pad]
            ran_fr = [ran_fr[0] - pad[0], ran_fr[1] + pad[1]]
            diff = np.diff(xran)[0]
            pre, post = [pre + diff * pad[0], post + diff * pad[1]] 
            xran = np.linspace(-pre, post, int(np.diff(ran_fr)[0]))

        if len(twop_ref_fr) == 0:
            raise ValueError("No frames: frames list must include at least 1 "
                "frame.")

        if isinstance(twop_ref_fr[0], (list, np.ndarray)):
            raise ValueError("Frames must be passed as a 1D list, not by "
                "block.")

        # get sequences x frames
        fr_idx = gen_util.num_ranges(
            twop_ref_fr, pre=-ran_fr[0], leng=len(xran))
                     
        # remove sequences with negatives or values above total number of stim 
        # frames
        neg_idx  = np.where(fr_idx[:,0] < 0)[0].tolist()
        over_idx = np.where(fr_idx[:,-1] >= self.tot_twop_fr)[0].tolist()
        
        num_ran = gen_util.remove_idx(fr_idx, neg_idx + over_idx, axis=0)

        if len(num_ran) == 0:
            raise ValueError("No frames: All frames were removed from list.")

        row_index = pd.MultiIndex.from_product([range(num_ran.shape[0]), xran], 
            names=["sequences", "time_values"])

        frame_n_df = pd.DataFrame(
            num_ran.reshape(-1), index=row_index, columns=["twop_fr_n"])

        return frame_n_df


    #############################################
    def get_roi_seqs(self, twop_fr_seqs, padding=(0,0), fluor="dff", 
                     remnans=True, scale=False, use_plateau=False):
        """
        self.get_roi_seqs(twop_fr_seqs)

        Returns the processed ROI traces for the given stimulus sequences.
        Frames around the start and end of the sequences can be requested by 
        setting the padding argument.

        If the sequences are different lengths the ends of the shorter 
        sequences are nan padded.

        Required args:
            - twop_fr_seqs (list of arrays): list of arrays of 2p frames,
                                             structured as sequences x frames. 
                                             If any frames are out of range, 
                                             then NaNs returned.

        Optional args:
            - padding (2-tuple of ints): number of additional 2p frames to 
                                         include from start and end of 
                                         sequences
                                         default: (0, 0)
            - fluor (str)              : if "dff", then dF/F traces are 
                                         returned, if "raw", raw processed 
                                         traces are returned
                                         default: "dff"
            - remnans (bool)           : if True, ROIs with NaN/Inf values 
                                         anywhere in session are excluded. 
                                         default: True
            - scale (bool)             : if True, each ROIs is scaled 
                                         based on full data array
                                         default: False

        Returns:
            - roi_data_df (pd DataFrame): dataframe containing ROI trace data  
                                          for the frames of interest, organized 
                                          by:
                hierarchical columns (all dummy):
                    - datatype        : type of data (e.g., "roi_traces")
                    - nan_rois_removed: whether ROIs with NaNs/Infs were 
                                        removed ("yes", "no")
                    - scaled          : whether ROI data is scaled 
                                        ("yes", "no")
                    - fluorescence    : type of data ("raw" or "dff")
                hierarchical rows:
                    - ROIs          : ROI indices
                    - sequences     : sequence numbers
                    - frames        : frame numbers
        """

        # extend values with padding
        if padding[0] != 0:
            min_fr       = np.asarray([min(x) for x in twop_fr_seqs])
            st_padd      = np.tile(
                np.arange(-padding[0], 0), (len(twop_fr_seqs), 1)) + \
                    min_fr[:,None]
            twop_fr_seqs = [np.concatenate((st_padd[i], x)) 
                for i, x in enumerate(twop_fr_seqs)]
        if padding[1] != 0:
            max_fr       = np.asarray([max(x) for x in twop_fr_seqs])
            end_padd     = np.tile(
                np.arange(1, padding[1]+1), (len(twop_fr_seqs), 1)) + \
                    max_fr[:,None]
            twop_fr_seqs = [np.concatenate((x, end_padd[i])) 
                for i, x in enumerate(twop_fr_seqs)]
        if padding[0] < 0 or padding[1] < 0:
            raise ValueError("Negative padding not supported.")

        # get length of each padded sequence
        pad_seql = np.array([len(s) for s in twop_fr_seqs])

        # flatten the sequences into one list of frames, removing any sequences
        # with unacceptable frame values (< 0 or > self.tot_twop_fr) 
        frames_flat = np.empty([sum(pad_seql)])
        last_idx = 0
        seq_rem, seq_rem_l = [], []
        for i in range(len(twop_fr_seqs)):
            if (max(twop_fr_seqs[i]) >= self.tot_twop_fr or 
                min(twop_fr_seqs[i]) < 0):
                seq_rem.extend([i])
                seq_rem_l.extend([pad_seql[i]])
            else:
                frames_flat[last_idx : last_idx + pad_seql[i]] = twop_fr_seqs[i]
                last_idx += pad_seql[i]

        # Warn about removed sequences and update pad_seql and twop_fr_seqs 
        # to remove these sequences
        if len(seq_rem) != 0 :
            logging.warning("Some of the specified frames for sequences "
                f"{seq_rem} are out of range so the sequence will not be "
                "included.", extra={"spacing": "\n"})
            pad_seql     = np.delete(pad_seql, seq_rem)
            twop_fr_seqs = np.delete(twop_fr_seqs, seq_rem).tolist()

        # sanity check that the list is as long as expected
        if last_idx != len(frames_flat):
            if last_idx != len(frames_flat) - sum(seq_rem_l):
                raise ValueError(f"Concatenated frame array is {last_idx} long "
                    f"instead of expected {len(frames_flat - sum(seq_rem_l))}.")
            else:
                frames_flat = frames_flat[: last_idx]

        traces_flat = self.get_roi_traces(
            frames_flat.astype(int), fluor, remnans, scale=scale)
        if use_plateau:
            traces_flat_fill = self.get_plateau_roi_traces(
                fluor=fluor, remnans=remnans
                )[:, frames_flat.astype(int)].reshape(-1, 1)
            traces_flat[:] = traces_flat_fill

        index_rows = pd.MultiIndex.from_product(
            [traces_flat.index.unique("ROIs").tolist(), 
            range(len(twop_fr_seqs)), range(max(pad_seql))], 
            names=["ROIs", "sequences", "frames"])
        
        traces_df = pd.DataFrame(
            None, index=index_rows, columns=traces_flat.columns)

        if max(pad_seql) == min(pad_seql):
            traces_df["roi_traces"] = traces_flat.values
        else:
            # chop back up into sequences padded with Nans
            traces_flat = gen_util.reshape_df_data(
                traces_flat, squeeze_rows=False, squeeze_cols=True)

            traces = np.full((traces_flat.shape[0], len(twop_fr_seqs), 
                max(pad_seql)), np.nan)
            last_idx = 0
            for i in range(len(twop_fr_seqs)):
                traces[:, i, : pad_seql[i]] = traces_flat[
                    :, last_idx : last_idx + pad_seql[i]]
                last_idx += pad_seql[i]
            traces_df["roi_traces"] = traces.reshape(-1)

        return traces_df


    #############################################
    def check_flanks(self, frs, ch_fl, fr_type="twop", ret_idx=False):
        """
        self.check_flanks(self, frs, ch_fl)

        Required args:
            - frs (arraylike): list of frames values
            - ch_fl (list)   : flanks in sec [pre sec, post sec] around frames 
                               to check for removal if out of bounds

        Optional args:
            - fr_type (str) : time of frames ("twop", "stim", "pup")
                              default: "twop"
            - ret_idx (bool): if True, indices of frames retained are also 
                              returned
                              default: False
        Returns:
            - frs (1D array): list of frames values within bounds
            if ret_idx:
            - all_idx (list): list of original indices retained
        """

        if not hasattr(self, "twop_fps"):
            raise ValueError("Run 'self.load_roi_info()' to load the ROI "
                "attributes correctly.")

        if not isinstance(ch_fl, list) or len(ch_fl) != 2:
            raise ValueError("'ch_fl' must be a list of length 2.")

        if fr_type == "twop":
            fps = self.twop_fps
            max_val = self.tot_twop_fr        
        elif fr_type == "stim":
            fps = self.stim_fps
            max_val = self.tot_stim_fr
        elif fr_type == "pup":
            fps = self.pup_fps
            max_val = self.tot_pup_fr
        else:
            gen_util.accepted_values_error(
                "fr_type", fr_type, ["twop", "stim", "pup"])

        ran_fr = [np.around(x * fps) for x in [-ch_fl[0], ch_fl[1]]]
        frs = np.asarray(frs)

        neg_idx  = np.where((frs + ran_fr[0]) < 0)[0].tolist()
        over_idx = np.where((frs + ran_fr[1]) >= max_val)[0].tolist()
        all_idx = sorted(set(range(len(frs))) - set(neg_idx + over_idx))

        if len(all_idx) == 0:
            frs = np.asarray([])
        else:
            frs = frs[np.asarray(all_idx)]

        if ret_idx:
            return frs, all_idx
        else:
            return frs

    
    #############################################
    def get_pup_fr_by_twop_fr(self, twop_fr, ch_fl=None):
        """
        self.get_pup_fr_by_twop_fr(twop_fr)

        Returns pupil frames corresponding to 2p frames, taking into
        account the delay to display.

        Required args:
            - twop_fr (array-like): the 2p frames for which to get 
                                    pupil frames

        Optional args:
            - ch_fl (list): if provided, flanks in sec [pre sec, post sec] 
                            around frames to check for removal if out of bounds
                            default: None

        Returns:
            - pup_fr (array-like): the pupil frames corresponding to the 2p
                                   frames
        """
 
        # delay of ~0.1s to display on screen
        delay = int(np.round(self.twop_fps * 0.1))
        pup_fr = self.twop2pupfr[list(twop_fr)] + delay

        if ch_fl is not None:
            pup_fr = self.check_flanks(pup_fr, ch_fl, fr_type="pup").tolist()

        return pup_fr


#############################################
#############################################
class Stim(object):
    """
    The Stim object is a higher level class for describing stimulus properties.
    For production data, both brick stimuli are initialized as one stimulus 
    object.

    It should be not be initialized on its own, but via a subclass in which
    stimulus specific information is initialized.
    """

    def __init__(self, sess, stim_n, stimtype):
        """
        self.__init__(sess, stim_n, stimtype)

        Initializes and returns a stimulus object, and sets attributes. 
        
        USE: Only initialize subclasses of Stim, not the stim class itself.

        Calls:
            - self._set_block_params()

            and if stimulus is a bricks stimulus from the production data:
            - self._check_brick_prod_params()

        Attributes:
            - act_n_blocks (int)          : nbr of blocks (where an overarching 
                                            parameter is held constant)
            - blank_per (int)             : period at which a blank segment 
                                            occurs
            - exp_block_len_s (int)       : expected length of each block in 
                                            seconds
            - exp_n_blocks (int)          : expected number of blocks of the 
                                            stimulus
            - reg_max_s (int)             : max duration of a regular seq 
            - reg_min_s (int)             : min duration of a regular seq
            - seg_len_s (sec)             : length of each segment 
                                            (1 sec for bricks, 0.3 sec for 
                                            gabors)
            - seg_ps_nobl (num)           : average number of segments per 
                                            second in a block, excluding blank 
                                            segments
            - seg_ps_wibl (num)           : average number of segments per 
                                            second in a block, including blank 
                                            segments
            - sess (Session object)       : session to which the stimulus 
                                            belongs
            - stim_fps (int)              : fps of the stimulus
            - stim_n (int)                : stimulus number in session (or 
                                            first stimulus number for 
                                            production bricks)
            - stimtype (str)              : "gabors" or "bricks"
            - surp_max_s (int)            : max duration of a surprise seq 
            - surp_min_s (int)            : min duration of a surprise seq

            if stimtype == "gabors":
                - n_seg_per_set (int)     : number of segments per set (4)
            if stimtype == "bricks" and sess.runtype == "prod":
                - stim_n_all (list)       : both stim numbers
            
        Required args:
            - sess (Session object): session to which the stimulus belongs
            - stim_n (int or list) : number of stimulus in session pickle
                                     (2 numbers if production Bricks stimulus)
            - stimtype (str)       : type of stimulus ("gabors" or "bricks")  

        """

        self.sess      = sess
        self.stimtype = stimtype
        self.stim_fps  = self.sess.stim_fps
        self.stim_n    = stim_n
    
        # for production Bricks, check that both stimulus dictionaries
        # are identical where necessary
        if self.sess.runtype == "prod" and self.stimtype == "bricks":
            self._check_brick_prod_params()
            self.stim_n_all = copy.deepcopy(self.stim_n)
            self.stim_n = self.stim_n[0]
        
        stim_info = self.sess.stim_dict["stimuli"][self.stim_n]

        # get segment parameters
        # seg is equivalent to a sweep, as defined in camstim 
        if self.sess.runtype == "pilot":
            stim_par = stim_info["stimParams"]
        if self.sess.runtype == "prod":
            stim_par = stim_info["stim_params"]

        if self.stimtype == "gabors":
            params = "gabor_params"
            dur_key = "gab_dur"
            # segment length (sec) (0.3 sec)
            self.seg_len_s     = stim_par[params]["im_len"] 
            # num seg per set (4: A, B, C D/U)
            self.n_seg_per_set = stim_par[params]["n_im"] 
            if self.sess.runtype == "pilot":
                # 2 blocks (1 per kappa) are expected.
                self.exp_n_blocks = 2 
            elif self.sess.runtype == "prod":
                self.exp_n_blocks = 1
        elif self.stimtype == "bricks":
            params = "square_params"
            dur_key = "sq_dur"
            # segment length (sec) (1 sec)
            self.seg_len_s     = stim_par[params]["seg_len"]
            if self.sess.runtype == "pilot":
                # 4 blocks (1 per direction/size) are expected.
                self.exp_n_blocks = 4 
            elif self.sess.runtype == "prod":
                self.exp_n_blocks = 2
        else:
            raise ValueError(f"{self.stimtype} stim type not recognized. Stim "
                "object cannot be initialized.")
        
        # blank period (i.e., 1 blank every _ segs)
        self.blank_per     = stim_info["blank_sweeps"] 
        # num seg per sec (blank segs count) 
        self.seg_ps_wibl   = 1/self.seg_len_s 
        # num seg per sec (blank segs do not count)
        if self.blank_per != 0:
            self.seg_ps_nobl = self.seg_ps_wibl * \
                self.blank_per/(1. + self.blank_per) 
        else:
            self.seg_ps_nobl = self.seg_ps_wibl
        
        # sequence parameters
        self.surp_min_s  = stim_par[params]["surp_len"][0]
        self.surp_max_s  = stim_par[params]["surp_len"][1]
        self.reg_min_s   = stim_par[params]["reg_len"][0]
        self.reg_max_s   = stim_par[params]["reg_len"][1]

        # expected length of a block (sec) where an overarching parameter is 
        # held constant
        if self.sess.runtype == "pilot":
            self.exp_block_len_s = stim_par[params]["block_len"] 
        elif self.sess.runtype == "prod":
            self.exp_block_len_s = stim_par["session_params"][dur_key]
                                                                                                
        self._set_block_params()


    #############################################
    def __repr__(self):
        return (f"{self.__class__.__name__} (stimulus {self.stim_n} of "
            f"session {self.sess.sessid})")

    def __str__(self):
        return repr(self)

    #############################################
    def _check_brick_prod_params(self):
        """
        self._check_brick_prod_params()

        Checks for Bricks production stimuli whether both specific components 
        of the stimulus dictionaries are identical. Specifically:
            ["stim_params"]["elemParams"]
            ["stim_params"]["session_params"]
            ["stim_params"]["square_params"]
            ["blank_sweeps"]

        If differences are found, throws an error specifying which components
        are different.
        """

        if self.stimtype != "bricks" or self.sess.runtype != "prod":
            raise ValueError("Checking whether 2 stimulus dictionaries "
                             "contain the same parameters is only for "
                             "production Bricks stimuli.")
        
        stim_n = gen_util.list_if_not(self.stim_n)
        if len(stim_n) != 2:
            raise ValueError("Expected 2 stimulus numbers, "
                             f"but got {len(stim_n)}")
        
        stim_dict_1 = self.sess.stim_dict["stimuli"][self.stim_n[0]]
        stim_dict_2 = self.sess.stim_dict["stimuli"][self.stim_n[1]]
        
        # check elemParams and square_params dictionaries
        error = False
        diff_dicts = []
        overall_dict = "stim_params"
        sub_dicts = ["elemParams", "square_params"]
        for dict_name in sub_dicts: 
            if (stim_dict_1[overall_dict][dict_name] != 
                stim_dict_2[overall_dict][dict_name]):
                diff_dicts.append(dict_name)
                error = True

        if error:
            diff_str = ", ".join(diff_dicts)
            dict_str = (f"\n- different values in the {diff_str} "
                f"(under {overall_dict}).")
        else:
            dict_str = ""

        # check blank_sweeps
        if stim_dict_1["blank_sweeps"] != stim_dict_2["blank_sweeps"]:
            error = True
            sweep_str = "\n- different values in the blank_sweeps key."
        else:
            sweep_str = ""
        
        # check sq_dur
        if (stim_dict_1["stim_params"]["session_params"]["sq_dur"] !=
            stim_dict_2["stim_params"]["session_params"]["sq_dur"]):
            error = True
            sq_str = ("\n- different values in the sq_dur key under "
                "stim_params, session_params.")
        else:
            sq_str = ""

        if error:
            raise ValueError("Cannot initialize production Brick stimuli "
                f"together, due to:{dict_str}{sweep_str}{sq_str}")


    #############################################
    def _set_block_params(self):
        """
        self._set_block_params

        Set attributes related to blocks and display sequences. Also checks
        whether expected number of blocks were shown and whether they 
        comprised the expected number of segments.

        NOTE: A block is a sequence of stimulus presentations of the same 
        stimulus type, and there can be multiple blocks in one experiment. 
        For Gabors, segments refer to each gabor frame (lasting 0.3 s). For 
        Bricks, segments refer to 1s of moving bricks. 
        
        NOTE: Grayscr segments are not omitted when a session's segments are 
        numbered.

        Calls:
            - self._set_stim_fr()
            - self._set_twop_fr()


        Attributes:
            - act_n_blocks (int)          : actual number of blocks of the 
                                            stimulus
            - disp_seq (2D array)         : display start and end times in sec, 
                                            structured as 
                                                display sequence x [start, end]
            - extra_segs (int)            : number of additional segments shown,
                                            if any
            - block_params (pd DataFrame): dataframe containing stimulus 
                                           parameters for each display sequence 
                                           and block:
                hierarchical columns:
                    - "parameters": parameter names 
                                    ("start_seg", "end_seg", "len_seg")
                hierarchical rows:
                    - "display_sequence_n": display sequence number
                    - "block_n"           : block number (across display 
                                            sequences)
        """

        stim_info = self.sess.stim_dict["stimuli"][self.stim_n]
        
        self.disp_seq = stim_info["display_sequence"].tolist()
        
        if self.stimtype == "bricks" and self.sess.runtype == "prod":
            stim_info2    = self.sess.stim_dict["stimuli"][self.stim_n_all[1]]
            self.disp_seq = self.disp_seq + \
                stim_info2["display_sequence"].tolist()

        tot_disp = int(sum(np.diff(self.disp_seq)))

        if self.stimtype == "gabors":
            # block length is correct, as it was set to include blanks
            block_len = self.exp_block_len_s
        elif self.stimtype == "bricks":
            # block length was not set to include blanks, so must be adjusted
            block_len = self.exp_block_len_s * \
                float(self.seg_ps_wibl)/self.seg_ps_nobl

        row_index = pd.MultiIndex.from_product(
            [[0], [0]], names=["display_sequence_n", "block_n"])
        col_index = pd.MultiIndex.from_product(
            [["start_seg", "end_seg"]], names=["parameters"])
        self.block_params = pd.DataFrame(
            None, index=row_index, columns=col_index)

        # calculate number of blocks that started and checking whether it is as 
        # expected
        self.act_n_blocks = int(np.ceil(float(tot_disp)/block_len))
        self.extra_segs = 0
        if self.act_n_blocks != self.exp_n_blocks:
            logger.warning(f"{self.act_n_blocks} {self.stimtype} blocks "
                f"started instead of the expected {self.exp_n_blocks}.", 
                extra={"spacing": TAB})
            if self.act_n_blocks > self.exp_n_blocks:
                self.extra_segs = (float(tot_disp) - \
                    self.exp_n_blocks*block_len)*self.seg_ps_wibl 
                logger.warning(f"In total, {self.extra_segs} "
                    "segments were shown, including blanks.", 
                    extra={"spacing": TAB})
    
        # calculate uninterrupted segment ranges for each block and check for 
        # incomplete or split blocks
        rem_sec_all = 0
        start = 0
        b = 0
        for d, i in enumerate(range(len(self.disp_seq))):
            # useable length is reduced if previous block was incomplete
            length = np.diff(self.disp_seq)[i] - rem_sec_all
            n_bl = int(np.ceil(float(length)/block_len))
            rem_sec_all += float(n_bl) * block_len - length
            rem_seg = int(np.around((
                float(n_bl) * block_len - length) * self.seg_ps_wibl))
            
            # collect block starts and ends (in segment numbers)
            for _ in range(n_bl - 1):
                end = start + int(np.around(block_len * self.seg_ps_nobl))
                self.block_params.loc[(d, b), (["start_seg", "end_seg"])] = \
                    [start, end]
                b += 1
                start = end
            # 1 removed because last segment is a blank
            end = start + int(np.around(block_len*self.seg_ps_nobl)) - \
                np.max([0, rem_seg - 1])
            self.block_params.loc[(d, b), (["start_seg", "end_seg"])] = \
                [start, end]
            start = end + np.max([0, rem_seg - 1])
            
            if rem_seg == 1:
                if i == len(self.disp_seq)-1:
                    logger.warning("During last sequence of "
                        f"{self.stimtype}, the last blank segment of the "
                        f"{n_bl}. block was omitted.", extra={"spacing": TAB})
                else:
                    logger.warning(f"During {i+1}. sequence of "
                        f"{self.stimtype}, the last blank segment of the "
                        f"{n_bl}. block was pushed to the start of the next "
                        "sequence.", extra={"spacing": TAB})
            elif rem_seg > 1:
                if i == len(self.disp_seq)-1:
                    logger.warning("During last sequence of "
                        f"{self.stimtype}, {rem_seg} segments (incl. blanks) "
                        f"from the {n_bl}. block were omitted.", 
                        extra={"spacing": TAB})
                else:
                    logger.warning(f"During {i+1}. sequence of "
                        f"{self.stimtype}, {rem_seg} segments (incl. blanks) "
                        f"from the {n_bl}. block were pushed to the next "
                        "sequence. These segments will be omitted from "
                        "analysis.", extra={"spacing": TAB})
            b += 1 # keep increasing across display sequences

        # get the actual length in segments of each block
        self.block_params[("len_seg", )] = \
            self.block_params[("end_seg", )] - \
                self.block_params[("start_seg", )]

        self._add_stim_fr_info()
        self._add_twop_fr_info()
        self.block_params = self.block_params.astype(int)


    #############################################
    def _add_stim_fr_info(self):
        """
        self._add_stim_fr_info()

        Sets and updates attributes related to stimulus frames.

        Attributes:
            - block_params (pd DataFrame): updates dataframe with stimulus frame 
                                           parameters for each display sequence 
                                           and block:
                hierarchical columns:
                    - "parameters": parameter names 
                                    ("start_stim_fr", "end_stim_fr", 
                                    "len_stim_fr")
                hierarchical rows:
                    - "display_sequence_n": display sequence number
                    - "block_n"           : block number (across display 
                                            sequences)
            - stim_seg_list (list)       : full list of stimulus segment 
                                           numbers for each stimulus frame
        """

        stim_info = self.sess.stim_dict["stimuli"][self.stim_n]

        # n blank frames pre/post stimulus
        bl_fr_pre = int(self.sess.pre_blank * self.stim_fps)
        bl_fr_post = int(self.sess.post_blank * self.stim_fps)
        
        # recorded stimulus frames
        stim_fr = stim_info["frame_list"].tolist()

        # combine the stimulus frame lists
        if self.stimtype == "bricks" and self.sess.runtype == "prod":
            stim_info2 = self.sess.stim_dict["stimuli"][self.stim_n_all[1]]
            stim_fr2   = stim_info2["frame_list"].tolist()
            # update seg numbers
            add = np.max(stim_fr) + 1
            for i in range(len(stim_fr2)):
                if stim_fr2[i] != -1:
                    stim_fr2[i] = stim_fr2[i] + add
            # collect all seg numbers together
            all_stim_fr = np.full(len(stim_fr2), -1)
            all_stim_fr[:len(stim_fr)] = stim_fr
            stim_fr = (all_stim_fr + np.asarray(stim_fr2) + 1).tolist()
            
        # unrecorded stim frames (frame list is only complete for the last 
        # stimulus shown)
        add_bl_fr = int(
            self.sess.tot_stim_fr - (len(stim_fr) + bl_fr_pre + bl_fr_post)
            )

        # fill out the stimulus segment list to be the same length as running 
        # array
        self.stim_seg_list = bl_fr_pre * [-1] + stim_fr + \
            add_bl_fr * [-1] + bl_fr_post * [-1] 

        # (skip last element, since it is ignored in stimulus frames as well
        self.stim_seg_list = self.stim_seg_list[:-1]

        for d in self.block_params.index.unique("display_sequence_n"):
            for b in self.block_params.loc[d].index.unique("block_n"):
                row = self.block_params.loc[(d, b)]
                # get first occurrence of first segment
                min_idx = self.stim_seg_list.index(row["start_seg"][0])
                max_idx = len(self.stim_seg_list)-1 - \
                    self.stim_seg_list[::-1].index(row["end_seg"][0] - 1) + 1 
                self.block_params.loc[(d, b), ("start_stim_fr", )] = min_idx
                self.block_params.loc[(d, b), ("end_stim_fr", )] = max_idx
                self.block_params.loc[
                    (d, b), ("len_stim_fr", )] = max_idx - min_idx
                # update the stimulus dataframe
                row_index = (
                    (self.sess.stim_df["stimType"] == self.stimtype[0]) &
                    (self.sess.stim_df["stimSeg"] >= row["start_seg"][0]) &
                    (self.sess.stim_df["stimSeg"] < row["end_seg"][0]))
                self.sess.stim_df.loc[row_index, "display_sequence_n"] = d
                self.sess.stim_df.loc[row_index, "block_n"] = b


    #############################################
    def _add_twop_fr_info(self):
        """
        self._add_twop_fr_info()

        Updates attributes related to twop frames.

        Attributes:
            - block_params (pd DataFrame): updates dataframe containing 
                                           two-photon frame parameters for each 
                                           display sequence and block:
                hierarchical columns:
                    - "parameters": parameter names 
                                    ("start_twop_fr", "end_twop_fr", 
                                    "len_twop_fr")
                hierarchical rows:
                    - "display_sequence_n": display sequence number
                    - "block_n"           : block number (across display 
                                            sequences)
        """

        for d in self.block_params.index.unique("display_sequence_n"):
            for b in self.block_params.loc[d].index.unique("block_n"):
                row = self.block_params.loc[(d, b)]
                # get first occurrence of first segment
                min_idx = int(self.sess.stim_df.loc[
                    (self.sess.stim_df["stimType"] == self.stimtype[0]) &
                    (self.sess.stim_df["stimSeg"] == row["start_seg"][0])
                    ]["start2pfr"].tolist()[0])
                max_idx = int(self.sess.stim_df.loc[
                    (self.sess.stim_df["stimType"] == self.stimtype[0]) &
                    (self.sess.stim_df["stimSeg"] == row["end_seg"][0] - 1)
                    ]["end2pfr"].tolist()[0] + 1)
                # 1 added as range end is excluded
                self.block_params.loc[(d, b), ("start_twop_fr", )] = min_idx
                self.block_params.loc[(d, b), ("end_twop_fr", )] = max_idx
                self.block_params.loc[
                    (d, b), ("len_twop_fr", )] = max_idx - min_idx


    #############################################
    def get_stim_beh_sub_df(self, pre, post, stats="mean", fluor="dff", 
                            remnans=True, gabfr="any", gabk="any", 
                            gab_ori="any", bri_size="any", bri_dir="any", 
                            pupil=False, run=False, scale=False):
        """
        self.get_stim_beh_sub_df(pre, post)

        Returns a stimulus and behaviour dataframe for the specific stimulus 
        (gabors or bricks) with grayscreen rows added in if requested and 
        plane, line and sessid added in.

        Required args:
            - pre (num) : range of frames to include before each reference 
                          frame number (in s)
            - post (num): range of frames to include after each reference 
                          frame number (in s)

        Optional args:
            - fluor (str)           : if "dff", dF/F is used, if "raw", ROI 
                                      traces
                                      default: "dff"
            - stats (str)           : statistic to use for baseline, mean 
                                      ("mean") or median ("median") (NaN values 
                                      are omitted)
                                      default: "mean"
            - remnans (bool)        : if True, NaN values are removed from data, 
                                      either through interpolation for pupil 
                                      and running data or ROI exclusion for ROI
                                      data
                                      default: True
            - gabfr (int or list)   : 0, 1, 2, 3, "gray", "G", "any"
            - gabk (int or list)    : 4, 16, or "any"
                                      default: "any"
            - gab_ori (int or list) : 0, 45, 90, 135, or "any"
                                      default: "any"
            - bri_size (int or list): 128, 256, or "any"
                                      default: "any"
            - bri_dir (str or list) : "right", "left", "temp", "nasal" or "any"
                                      default: "any"
            - pupil (bool)          : if True, pupil data is added in
                                      default: False
            - run (bool)            : if True, run data is added in
                                      default: False
            - scale (bool)          : if True, data is scaled
                                      default: False

        Returns:
            - sub_df (pd DataFrame): extended stimulus dataframe containing
                                     grayscreen rows if requested, modified 
                                     column names, plane, line and sessid info 
        """

        retain = ["stimPar1", "stimPar2", "surp", "stimSeg", 
            "start2pfr", "end2pfr"]
        drop = ["stimSeg", "start2pfr", "end2pfr", "start_stim_fr", 
            "end_stim_fr"] # drop at end

        ret_gabfr = gabfr
        if self.stimtype == "gabors":
            retain.append("gabfr")
            get_gray = (("gray" in gen_util.list_if_not(gabfr)) or 
                ("G" in gen_util.list_if_not(gabfr)) or
                (gabfr in ["any", "all"]))
            if get_gray:
                ret_gabfr = "any"
    
        sub_df = self.get_stim_df_by_criteria(
            gabfr=ret_gabfr, gabk=gabk, gab_ori=gab_ori, bri_size=bri_size, 
            bri_dir=bri_dir)[retain]
        
        stim_fr_df = self.get_stim_fr_by_seg(
            sub_df["stimSeg"], first=True, last=True)
        sub_df["start_stim_fr"] = stim_fr_df["first_stim_fr"].astype(int).values
        sub_df["end_stim_fr"] = stim_fr_df["last_stim_fr"].astype(int).values
        if self.stimtype == "gabors":
            sub_df = sub_df.rename(
                columns={"stimPar1": "gab_ori", "stimPar2": "gabk"})
            if get_gray:
                sub_df = sess_data_util.add_G_rows_gabors(sub_df)
                # set non 3 or "G" gabors to surp = 0
                sub_df.loc[~(sub_df["gabfr"].isin([3, "G"])), "surp"] = 0
                gabfr_vals = gen_util.get_df_label_vals(sub_df, "gabfr", gabfr)
                sub_df = sub_df.loc[(sub_df["gabfr"].isin(gabfr_vals))]
        elif self.stimtype == "bricks":
            sub_df = sub_df.rename(
                columns={"stimPar1": "bri_size", "stimPar2": "bri_dir"})
        else:
            raise NotImplementedError("Extended stimulus subdataframe only "
                "implemented for Gabor and Brick stimuli, "
                f"not '{self.stimtype}'.")
        
        sub_df = sub_df.reset_index(drop=True) # reset index
        sub_df["plane"] = self.sess.plane
        sub_df["line"]  = self.sess.line
        sub_df["sessid"] = self.sess.sessid

        if pupil:
            pup_fr = self.sess.get_pup_fr_by_twop_fr(
                sub_df["start2pfr"].to_numpy())
            pup_data = gen_util.reshape_df_data(
                self.get_pup_diam_data(
                    pup_fr, pre, post, remnans=remnans, scale=scale
                    )["pup_diam"], squeeze_rows=False, squeeze_cols=True)
            sub_df["pup_diam_data"] = math_util.mean_med(
                pup_data, stats=stats, axis=-1)
        if run:
            run_data = gen_util.reshape_df_data(
                self.get_run_data(sub_df["start_stim_fr"].to_numpy(), 
                    pre, post, remnans=remnans, scale=scale
                    )["run_velocity"], squeeze_rows=False, squeeze_cols=True)
            sub_df["run_data"] = math_util.mean_med(
                run_data, stats=stats, axis=-1)
        
        # add ROI data
        logger.info("Adding ROI data to dataframe.")
        roi_data = self.get_roi_data(
            sub_df["start2pfr"].to_numpy(), pre, post, remnans=remnans, 
            fluor=fluor, scale=scale)["roi_traces"]
        targ = [len(roi_data.index.unique(dim)) for dim in roi_data.index.names]
        roi_data = math_util.mean_med(roi_data.to_numpy().reshape(targ), 
            stats=stats, axis=-1, nanpol="omit")
        cols = [f"roi_data_{i}" for i in range(len(roi_data))]
        all_roi = pd.DataFrame(columns=cols, data=roi_data.T)
        sub_df = sub_df.join(all_roi)

        sub_df = sub_df.drop(columns=drop)

        return sub_df


    #############################################
    def get_stim_fr_by_seg(self, seglist, first=False, last=False, ch_fl=None):
        """
        self.get_stim_fr_by_seg(seglist)

        Returns a list of arrays containing the stimulus frame numbers that 
        correspond to a given set of stimulus segments provided in a list 
        for a specific stimulus.

        Required args:
            - seglist (list of ints): the stimulus segments for which to get 
                                      stim frames

        Optional args:
            - first (bool): instead returns the first frame for each seg.
                            default: False
            - last (bool) : instead returns the last for each seg (excl).
                            default: False
            - ch_fl (list): if provided, flanks in sec [pre sec, post sec] 
                            around frames to check for removal if out of bounds
                            default: None

        Returns:
            if first or last is True:
                - frames (pd DataFrame)      : frames dataframe with
                    columns:
                    - "first_stim_fr": first stimulus frame for each segment
                    - "last_stim_fr" : last stimulus frame for each segment 
                                       (excl)
            else:
                - frames (list of int arrays): a list (one entry per segment) 
                                               of arrays containing the stim 
                                               frame
        """
        
        stim_seg_list_array = np.asarray(self.stim_seg_list)
        all_fr = [np.where(stim_seg_list_array == val)[0] for val in seglist]

        firsts = [fr[0] for fr in all_fr]

        if ch_fl is not None:
            firsts, ret_idx = self.sess.check_flanks(
                firsts, ch_fl, fr_type="stim", ret_idx=True)
            firsts = firsts.tolist()
            if not first or last:
                # trim all_fr, if needed
                all_fr = [all_fr[i] for i in ret_idx]
        
        if first or last:
            frames = pd.DataFrame()
            if first:
                frames["first_stim_fr"] = firsts
            if last:
                frames["last_stim_fr"] = [fr[-1] + 1 for fr in all_fr]
        else:
            frames = all_fr

        return frames
        
        
    #############################################
    def get_twop_fr_by_seg(self, seglist, first=False, last=False, 
                           ch_fl=None):
        """
        self.get_twop_fr_by_seg(seglist)

        Returns a list of arrays containing the 2-photon frame numbers that 
        correspond to a given set of stimulus segments provided in a list 
        for a specific stimulus.

        Required args:
            - seglist (list of ints): the stimulus segments for which to get 
                                      2p frames

        Optional args:
            - first (bool): instead, return first frame for each seg
                            default: False
            - last (bool) : instead return last frame for each seg (excl)
                            default: False
            - ch_fl (list): if provided, flanks in sec [pre sec, post sec] 
                            around frames to check for removal if out of bounds
                            default: None

        Returns:
            if first or last is True:
                - frames (pd DataFrame)      : frames dataframe with
                    columns:
                    - "first_twop_fr": first two-photon frame for each segment
                    - "last_twop_fr" : last two-photon frame for each segment
                                       (excl)
            else:
                - frames (list of int arrays): a list (one entry per segment) 
                                               of arrays containing the stim 
                                               frame
        """

        # get the rows in the alignment dataframe that correspond to the 
        # segments
        rows = self.sess.stim_df.loc[
            (self.sess.stim_df["stimType"] == self.stimtype[0]) &
            (self.sess.stim_df["stimSeg"].isin(seglist))]

        # get the start frames and end frames from each row
        start2pfrs = rows["start2pfr"].values

        if ch_fl is not None:
            start2pfrs = self.sess.check_flanks(
                start2pfrs, ch_fl, fr_type="twop")

        if not first or last:
            end2pfrs = rows.loc[rows["start2pfr"].isin(start2pfrs), 
                "end2pfr"].values

        if first or last:
            frames = pd.DataFrame()
            if first:
                frames["first_twop_fr"] = start2pfrs
            if last:
                frames["last_twop_fr"] = end2pfrs
        else:
            frames = [np.arange(st, end) for st, end in zip(
                start2pfrs, end2pfrs)]

        return frames


    #############################################
    def get_n_twop_fr_by_seg(self, segs):
        """
        self.get_n_twop_fr_by_seg(segs)

        Returns a list with the number of twop frames for each seg passed.    

        Required args:
            - segs (list): list of segments

        Returns:
            - n_fr_sorted (list): list of number of frames in each segment
        """

        segs = gen_util.list_if_not(segs)

        segs_unique = sorted(set(segs))
        
        # number of frames will be returned in ascending order of seg number
        n_fr = self.sess.stim_df.loc[
            (self.sess.stim_df["stimType"] == self.stimtype[0]) &
            (self.sess.stim_df["stimSeg"].isin(segs_unique))
            ]["num2pfr"].tolist()
        
        # resort based on order in which segs were passed and include any 
        # duplicates
        n_fr_sorted = [n_fr[segs_unique.index(seg)] for seg in segs]
        
        return n_fr_sorted


    #############################################
    def get_stim_df_by_criteria(self, stimPar1="any", stimPar2="any", 
                                surp="any", stimSeg="any", gabfr="any", 
                                start2pfr="any", end2pfr="any", 
                                num2pfr="any", gabk=None, gab_ori=None, 
                                bri_size=None, bri_dir=None):
        """
        self.get_stim_df_by_criteria()

        Returns a subset of the stimulus dataframe based on the criteria 
        provided.    

        Will return lines only for the current stim object.

        Optional args:
            - stimPar1 (str, int or list)  : stimPar1 value(s) of interest 
                                             (sizes: 128, 256, 
                                             oris: 0, 45, 90, 135)
                                             default: "any"
            - stimPar2 (str, int or list)  : stimPar2 value(s) of interest 
                                             ("right", "left", "temp", 
                                             "nasal", 4, 16)
                                             default: "any"
            - surp (str, int or list)      : surp value(s) of interest (0, 1)
                                             default: "any"
            - stimSeg (str, int or list)   : stimSeg value(s) of interest
                                             default: "any"
            - gabfr (str, int or list)     : gaborframe value(s) of interest 
                                             (0, 1, 2, 3)
                                             default: "any"
            - start2pfr (str or list)      : 2p start frames range of interest
                                             [min, max (excl)] 
                                             default: "any"
            - end2pfr (str or list)        : 2p end frames (excluded ends) 
                                             range of interest [min, max (excl)]
                                             default: "any"
            - num2pfr (str or list)        : 2p num frames range of interest
                                             [min, max (excl)]
                                             default: "any"
            - gabk (int or list)           : if not None, will overwrite 
                                             stimPar2 (4, 16, or "any")
                                             default: None
            - gab_ori (int or list)        : if not None, will overwrite 
                                             stimPar1 (0, 45, 90, 135, or "any")
                                             default: None
            - bri_size (int or list)       : if not None, will overwrite 
                                             stimPar1 (128, 256, or "any")
                                             default: None
            - bri_dir (str or list)        : if not None, will overwrite 
                                             stimPar2 ("right", "left", "temp", 
                                             "nasal", or "any")
                                             default: None
        
        Returns:
            - sub_df (pd DataFrame): subset of the stimulus dataframe 
                                     fitting the criteria provided
        """

        pars = sess_data_util.format_stim_criteria(
            self.sess.stim_df, self.stimtype, stimPar1, stimPar2, surp, 
            stimSeg, gabfr, start2pfr, end2pfr, num2pfr, gabk, gab_ori, 
            bri_size, bri_dir)

        [stimPar1, stimPar2, surp, stimSeg, gabfr, start2pfr_min, 
         start2pfr_max, end2pfr_min, end2pfr_max, num2pfr_min, 
         num2pfr_max] = pars

        sub_df = self.sess.stim_df.loc[
            (self.sess.stim_df["stimType"]==self.stimtype[0]) & 
            (self.sess.stim_df["stimPar1"].isin(stimPar1))    &
            (self.sess.stim_df["stimPar2"].isin(stimPar2))    &
            (self.sess.stim_df["surp"].isin(surp))            &
            (self.sess.stim_df["stimSeg"].isin(stimSeg))      &
            (self.sess.stim_df["gabfr"].isin(gabfr))          &
            (self.sess.stim_df["start2pfr"] >= start2pfr_min) &
            (self.sess.stim_df["start2pfr"] < start2pfr_max)  &
            (self.sess.stim_df["end2pfr"] >= end2pfr_min)     &
            (self.sess.stim_df["end2pfr"] < end2pfr_max)      &
            (self.sess.stim_df["num2pfr"] >= num2pfr_min)     &
            (self.sess.stim_df["num2pfr"] < num2pfr_max)]
        
        return sub_df


    #############################################
    def get_segs_by_criteria(self, stimPar1="any", stimPar2="any", surp="any", 
                             stimSeg="any", gabfr="any", start2pfr="any", 
                             end2pfr="any", num2pfr="any", gabk=None, 
                             gab_ori=None, bri_size=None, bri_dir=None, 
                             remconsec=False, by="block"):
        """
        self.get_segs_by_criteria()

        Returns a list of stimulus seg numbers that have the specified values 
        in specified columns in the stimulus dataframe.    

        Will return segs only for the current stim object.

        Optional args:
            - stimPar1 (str, int or list)  : stimPar1 value(s) of interest 
                                             (sizes: 128, 256, 
                                             oris: 0, 45, 90, 135)
                                             default: "any"
            - stimPar2 (str, int or list)  : stimPar2 value(s) of interest 
                                             ("right", "left", "temp", "nasal", 
                                             4, 16)
                                             default: "any"
            - surp (str, int or list)      : surp value(s) of interest (0, 1)
                                             default: "any"
            - stimSeg (str, int or list)   : stimSeg value(s) of interest
                                             default: "any"
            - gabfr (str, int or list)     : gaborframe value(s) of interest 
                                             (0, 1, 2, 3)
                                             default: "any"
            - start2pfr (str or list)      : 2p start frames range of interest
                                             [min, max (excl)] 
                                             default: "any"
            - end2pfr (str or list)        : 2p end frames (excluded ends) 
                                             range of interest [min, max (excl)]
                                             default: "any"
            - num2pfr (str or list)        : 2p num frames range of interest
                                             [min, max (excl)]
                                             default: "any"
            - gabk (int or list)           : if not None, will overwrite 
                                             stimPar2 (4, 16, or "any")
                                             default: None
            - gab_ori (int or list)        : if not None, will overwrite 
                                             stimPar1 (0, 45, 90, 135, or "any")
                                             default: None
            - bri_size (int or list)       : if not None, will overwrite 
                                             stimPar1 (128, 256, or "any")
                                             default: None
            - bri_dir (str or list)        : if not None, will overwrite 
                                             stimPar2 ("right", "left", "temp", 
                                             "nasal", or "any")
                                             default: None
            - remconsec (bool)             : if True, consecutive segments are 
                                             removed within a block
                                             default: False
            - by (str)                     : determines whether segment numbers
                                             are returned in a flat list 
                                             ("seg"), grouped by block 
                                             ("block"), or further grouped by 
                                             display sequence ("disp")
                                             default: "block"
        
        Returns:
            - segs (list): list of seg numbers that obey the criteria, 
                           optionally arranged by block or display sequence
        """

        sub_df = self.get_stim_df_by_criteria(stimPar1, stimPar2, surp, 
            stimSeg, gabfr, start2pfr, end2pfr, num2pfr, gabk, gab_ori, 
            bri_size, bri_dir)
        
        segs = []
        disp_grps = sub_df.groupby("display_sequence_n")
        for disp_grp in disp_grps:
            block_grps = disp_grp[1].groupby("block_n")
            temp = []
            for block_grp in block_grps:
                add_segs = block_grp[1]["stimSeg"].tolist()
                if remconsec and len(add_segs) != 0: 
                    idx_keep = np.where(
                        np.insert(np.diff(add_segs), 0, 4) > 1)[0]
                    add_segs = (np.asarray(add_segs)[idx_keep]).tolist()
                if len(add_segs) != 0: # check for empty
                    temp.append(add_segs)  
            if len(temp) != 0: # check for empty
                segs.append(temp)
        
        # check for empty
        if len(segs) == 0:
             raise ValueError("No segments fit these criteria.")

        # if not returning by disp
        if by == "block" or by == "seg":
            segs = [x for sub in segs for x in sub]
            if by == "seg":
                segs = [x for sub in segs for x in sub]
        elif by != "disp":
            gen_util.accepted_values_error("by", by, ["block", "disp", "seg"])
        
        return segs


    #############################################
    def get_stim_fr_by_criteria(self, stimPar1="any", stimPar2="any", 
                                surp="any", stimSeg="any", gabfr="any", 
                                start2pfr="any", end2pfr="any", 
                                num2pfr="any", gabk=None, gab_ori=None, 
                                bri_size=None, bri_dir=None, first_fr=True, 
                                remconsec=False, by="block"):
        """
        self.get_stim_fr_by_criteria()

        Returns a list of stimulus frames numbers that have the specified 
        values in specified columns in the stimulus dataframe. 
        
        Will return frame numbers only for the current stim object.

        NOTE: grayscreen frames are NOT returned

        Optional args:
            - stimPar1 (str, int or list)  : stimPar1 value(s) of interest 
                                             (sizes: 128, 256, 
                                             oris: 0, 45, 90, 135)
                                             default: "any"
            - stimPar2 (str, int or list)  : stimPar2 value(s) of interest 
                                             ("right", "left", "temp", "nasal", 
                                             4, 16)
                                             default: "any"
            - surp (str, int or list)      : surp value(s) of interest (0, 1)
                                             default: "any"
            - stimSeg (str, int or list)   : stimSeg value(s) of interest
                                             default: "any"
            - gabfr (str, int or list)     : gaborframe value(s) of interest 
                                             (0, 1, 2, 3)
                                             default: "any"
            - start2pfr (str or list)      : 2p start frames range of interest
                                             [min, max (excl)] 
                                             default: "any"
            - end2pfr (str or list)        : 2p end frames (excluded ends) 
                                             range of interest [min, max (excl)]
                                             default: "any"
            - num2pfr (str or list)        : 2p num frames range of interest
                                             [min, max (excl)]
                                             default: "any"         
            - gabk (int or list)           : if not None, will overwrite 
                                             stimPar2 (4, 16, or "any")
                                             default: None
            - gab_ori (int or list)        : if not None, will overwrite 
                                             stimPar1 (0, 45, 90, 135, or "any")
                                             default: None
            - bri_size (int or list)       : if not None, will overwrite 
                                             stimPar1 (128, 256, or "any")
                                             default: None
            - bri_dir (str or list)        : if not None, will overwrite 
                                             stimPar2 ("right", "left" or "any")
                                             default: None
            - remconsec (bool)               if True, consecutive segments are 
                                             removed within a block
                                             default: False
            - by (str)                     : determines whether frame numbers 
                                             are returned in a flat list 
                                             ("frame"), grouped by block 
                                             ("block"), or further grouped by 
                                             display sequence ("disp")
                                             default: "block"
        
        Returns:
            - frames (list): list of stimulus frame numbers that obey the 
                             criteria, optionally arranged by block or display 
                             sequence
        """


        segs = self.get_segs_by_criteria(
            stimPar1, stimPar2, surp, stimSeg, gabfr, start2pfr, end2pfr, 
            num2pfr, gabk, gab_ori, bri_size, bri_dir, remconsec, by="disp")

        frames = []
        for i in segs:
            temp = []
            for idxs in i:
                temp2 = self.get_stim_fr_by_seg(idxs, first=first_fr)
                if first_fr:
                    temp2 = temp2["first_stim_fr"]
                else:
                    temp2 = np.concatenate(temp2).tolist()
                if len(temp2) != 0:
                    temp.append(temp2)
            # check for empty      
            if len(temp) != 0:
                frames.append(temp)
        
        # check for empty
        if len(frames) == 0:
             raise ValueError("No segments fit these criteria.")

        # if not returning by disp
        if by == "block" or by == "frame":
            frames = [x for sub in frames for x in sub]
            if by == "frame":
                frames = [x for sub in frames for x in sub]
        elif by != "disp":
            gen_util.accepted_values_error("by", by, ["block", "disp", "frame"])
        
        return frames


    #############################################
    def get_first_surp_segs(self, by="block"):
        """
        self.get_first_surp_segs()

        Returns two lists of stimulus segment numbers, the first is a list of 
        all the first surprise segments for the stimulus type at transitions 
        from regular to surprise sequences. The second is a list of all the 
        first regular segements for the stimulus type at transitions from 
        surprise to regular sequences.

        Optional args:
            - by (str): determines whether segment numbers are returned in a 
                        flat list ("seg"), grouped by block ("block"), or 
                        further grouped by display sequence ("disp")
                        default: "block"

        Returns:
            - reg_segs (list) : list of first regular segment numbers at 
                                surprise to regular transitions for stimulus 
                                type
            - surp_segs (list): list of first surprise segment numbers at 
                                regular to surprise transitions for stimulus 
                                type
        """

        reg_segs  = self.get_segs_by_criteria(surp=0, remconsec=True, by=by)
        surp_segs = self.get_segs_by_criteria(surp=1, remconsec=True, by=by)

        return reg_segs, surp_segs


    #############################################
    def get_all_surp_segs(self, by="block"):
        """
        self.get_all_surp_segs()

        Returns two lists of stimulus segment numbers. The first is a list of 
        all the surprise segments for the stimulus type. The second is a list 
        of all the regular segments for the stimulus type.

        Optional args:
            - by (str): determines whether segment numbers are returned in a 
                        flat list ("seg"), grouped by block ("block"), or 
                        further grouped by display sequence ("disp")
                        default: "block"

        Returns:
            - reg_segs (list) : list of regular segment numbers for stimulus 
                                type
            - surp_segs (list): list of surprise segment numbers for stimulus 
                                type
        """

        reg_segs  = self.get_segs_by_criteria(surp=0, by=by)
        surp_segs = self.get_segs_by_criteria(surp=1, by=by)

        return reg_segs, surp_segs
    

    #############################################
    def get_first_surp_stim_fr_trans(self, by="block"):
        """
        self.get_first_surp_stim_fr_trans()

        Returns two lists of stimulus frame numbers, the first is a list of all 
        the first surprise frames for the stimulus type at transitions from 
        regular to surprise sequences. The second is a list of all the first 
        regular frames for the stimulus type at transitions from surprise to 
        regular sequences.

        Optional args:
            - by (str): determines whether frames are returned in a flat list 
                        ("frame"), grouped by block ("block"), or further 
                        grouped by display sequence ("disp")
                        default: "block"
        
        Returns:
            - reg_fr (list) : list of first regular stimulus frame numbers at 
                              surprise to regular transitions for stimulus type
            - surp_fr (list): list of first surprise stimulus frame numbers at 
                              regular to surprise transitions for stimulus type
        """
    
        reg_fr  = self.get_stim_fr_by_criteria(surp=0, remconsec=True, by=by)
        surp_fr = self.get_stim_fr_by_criteria(surp=1, remconsec=True, by=by)

        return reg_fr, surp_fr


    #############################################
    def get_all_surp_stim_fr(self, by="block"):
        """
        self.get_all_surp_stim_fr()

        Returns two lists of stimulus frame numbers, the first is a list of all 
        surprise frames for the stimulus type. The second is a list of all 
        regular frames for the stimulus type.

        Optional args:
            - by (str): determines whether frame numbers are returned in a flat 
                        list ("frame"), grouped by block ("block"), or further 
                        grouped by display sequence ("disp")
                        default: "block"

        Returns:
            - reg_fr (list) : list of all regular frame numbers for stimulus 
                              type
            - surp_fr (list): list of all surprise frame numbers for stimulus 
                              type
        """

        surp_fr = self.get_stim_fr_by_criteria(surp=1, first_fr=False, by=by)
        reg_fr  = self.get_stim_fr_by_criteria(surp=0, first_fr=False, by=by)

        return reg_fr, surp_fr
    

    #############################################
    def get_stats_df(self, data_df, ret_arr=False, dims="sequences", 
                     stats="mean", error="std", nanpol=None):
        """
        self.get_stats_df(data_df)

        Returns dataframe with stats (mean and std or median and quartiles) for 
        arrays of running, pupil or roi traces. If sequences of unequal length 
        are passed, they are cut down to the same length to obtain statistics.

        Required args:
            - data_df (pd DataFrame): dataframe containing data organized in 
                hierarchical columns:
                    - datatype   : the specific datatype
                hierarchical rows: 
                    - ROIs       : each ROI, if ROI data (optional) 
                    - sequences  : each sequence (trial)
                    - time_values: time values for each sequence, if data is 
                                   not integrated (optional)
            
        Optional args:
            - ret_arr (bool)    : also return data array, not just statistics 
                                  default: False 
            - dims (str or list): dimensions along which to take statistics 
                                  (see data_df rows). If None, axes are ordered 
                                  reverse sequentially (-1 to 1).
                                  default: "sequences"
            - stats (str)       : return mean ("mean") or median ("median")
                                  default: "mean"
            - error (str)       : return std dev/quartiles ("std") or SEM/MAD 
                                  ("sem")
                                  default: "sem"
            - nanpol (str)      : policy for NaNs, "omit" or None
                                  default: None
         
        Returns:
            - data_stats_df (pd DataFrame): dataframe containing data statistics 
                                            organized in 
                hierarchical columns (dummy):
                    - datatype   : the specific datatype
                hierarchical rows:
                    - general    : type of data ("data" (if ret_arr), "stats")
                    - ROIs       : each ROI, if ROI data (optional) 
                    - sequences  : each sequence (trial)
                    - time_values: time values for each sequence, if data is 
                                   not integrated (optional)
        """

        # check dataframe structure
        possible_index_names = ["ROIs", "sequences", "time_values"]
        required = [False, True, False]
        cum = 0
        incls = []
        for name, req in zip(possible_index_names, required):
            if name in data_df.index.names:
                if list(data_df.index.names).index(name) != cum:
                    raise ValueError("'data_df' row indices must occur in "
                        "the following order: (ROIs), sequences, (time_values)")
                cum += 1
                incls.append(True)
            elif req:
                raise ValueError("'data_df' row indices must include "
                    "'sequences'.")
            else:
                incls.append(False)
        rois, _, not_integ = incls

        if "datatype" not in data_df.columns.names:
            raise ValueError("data_df column must include 'datatype'.")

        # retrieve datatype
        datatypes = data_df.columns.get_level_values("datatype")
        if len(datatypes) != 1:
            raise ValueError("Expected only one datatype in data_df.")
        datatype = datatypes[0]

        # get data array from which to get statistics
        index_slice = pd.IndexSlice[:]
        if not_integ:
            use_data_df = data_df
            if rois:
                use_data_df = data_df.loc[data_df.index.unique("ROIs")[0]]
            seq_vals = data_df.index.unique(level="sequences")
            time_value_sets = [use_data_df.loc[(seq_val, )].index.unique(
                "time_values").tolist() for seq_val in seq_vals]
            shared_time_vals = sorted(set(time_value_sets[0]).intersection(
                *time_value_sets[0:]))
            all_time_vals = sorted(set(time_value_sets[0]).union(
                *time_value_sets[0:]))
            if len(set(shared_time_vals) - set(all_time_vals)):
                if rois:
                    index_slice = pd.IndexSlice[:, :, shared_time_vals]
                else:
                    index_slice = pd.IndexSlice[:, shared_time_vals]

        sub_df = data_df.loc[index_slice, (datatype)]
        targ_dims = [len(sub_df.index.unique(row)) 
            for row in sub_df.index.names]

        data_array = sub_df.to_numpy().reshape(targ_dims)

        # convert dims to axis numbers
        dims = gen_util.list_if_not(dims)
        if set(dims) - set(data_df.index.names):
            raise ValueError("'dims' can only include: {}".format(
                ", ".join(data_df.index.names)))
        else:
            axes = [data_df.index.names.index(val) for val in dims]
            if len(axes) == 0:
                raise ValueError("Must provide at least one 'dims'.")

        # get stats
        data_stats = math_util.get_stats(
            data_array.astype(float), stats, error, axes=axes, nanpol=nanpol)

        if rois and "ROIs" not in dims:
            # place ROI dimension first
            data_stats = data_stats.transpose(
                1, 0, *range(len(data_stats.shape))[2:])

        # retrieve the level values for the data statistics
        err_name = [f"error_{name}" for name in gen_util.list_if_not(
            math_util.error_stat_name(stats=stats, error=error))]
        stat_names = [f"stat_{stats}", *err_name]

        # prepare dataframe
        level_vals = [["stats"]] + gen_util.get_df_unique_vals(
            sub_df, axis="index")
        stat_names_set = False
        for l, level_name in enumerate(data_df.index.names):
            if level_name in dims:
                if "sequences" in dims:
                    if level_name == "sequences":
                        level_vals[l+1] = stat_names
                        continue
                elif not stat_names_set:
                    level_vals[l+1] = stat_names
                    stat_names_set = True
                    continue
                level_vals[l+1] = ["None"]
        

        # append stats
        row_index = pd.MultiIndex.from_product(level_vals, 
            names=["general"] + data_df.index.names)

        data_stats_df = pd.DataFrame(data_stats.reshape(-1), 
            index=row_index, columns=data_df.columns)

        if ret_arr:
            data_stats_df = pd.concat([data_stats_df, pd.concat([data_df], 
                keys=["data"], names=["general"])])
    
        return data_stats_df


    #############################################
    def get_pup_diam_data(self, pup_ref_fr, pre, post, integ=False, 
                           remnans=False, baseline=None, stats="mean", 
                           scale=False):
        """
        self.get_pup_diam_data(pup_ref_fr, pre, post)

        Returns array of pupil data around specific pupil frame numbers. NaNs
        are omitted in calculating statistics.

        Required args:
            - pup_ref_fr (list): 1D list of reference pupil frame numbers
                                  around which to retrieve running data 
                                  (e.g., all 1st Gabor A frames)
            - pre (num)         : range of frames to include before each 
                                  reference frame number (in s)
            - post (num)        : range of frames to include after each 
                                  reference frame number (in s)
        
        Optional args:
            - integ (bool)    : if True, pupil diameter is integrated over 
                                frames
                                default: False
            - remnans (bool)  : if True, NaN values are removed using linear 
                                interpolation. If False, NaN values (but
                                not Inf values) are omitted in calculating the 
                                data statistics.
                                default: False
            - baseline (num)  : number of seconds from beginning of 
                                sequences to use as baseline. If None, data 
                                is not baselined.
                                default: None
            - stats (str)     : statistic to use for baseline, mean ("mean") or 
                                median ("median") (NaN values are omitted)
                                default: "mean"
            - scale (bool)    : if True, pupil diameter is scaled using 
                                full data array
                                default: False      

        Returns:
            - pup_data_df (pd DataFrame): dataframe containing pupil diameter 
                                          values (in pixels) for the frames
                                          of interest, organized by:
                hierarchical columns (all dummy):
                    - datatype    : type of data ("pup_diam")
                    - interpolated: whether data is interpolated ("yes", "no")
                    - scaled      : whether data is scaled ("yes", "no")
                    - baseline    : baseline used ("no", value)
                    - integrated  : whether data is integrated over sequences 
                                    ("yes", "no")
                hierarchical rows:
                    - sequences   : sequence numbers for each datapoint or 
                                    dataset
                    - time_values : time values in seconds for each datapoint 
                                    (optional)
        """

        datatype = "pup_diam"

        if not hasattr(self.sess, "pup_data"):
            self.sess.load_pup_data()

        ran_fr, xran = self.sess.get_frames_timestamps(
            pre, post, fr_type="pup")

        if isinstance(pup_ref_fr[0], (list, np.ndarray)):
            raise ValueError("Frames must be passed as a 1D list, not by "
                "block.")

        # get corresponding running subblocks sequences x frames
        fr_idx = gen_util.num_ranges(
            pup_ref_fr, pre=-ran_fr[0], leng=len(xran))
                     
        # remove sequences with negatives or values above total number of stim 
        # frames
        neg_idx  = np.where(fr_idx[:,0] < 0)[0].tolist()
        over_idx = np.where(fr_idx[:,-1] >= self.sess.tot_pup_fr)[0].tolist()
        
        fr_idx = gen_util.remove_idx(fr_idx, neg_idx + over_idx, axis=0)

        if len(fr_idx) == 0:
            raise ValueError("No frames in list.")

        pup_data = self.sess.get_pup_data(
            datatype=datatype, remnans=remnans, scale=scale)

        data_array = pup_data.to_numpy().squeeze()[fr_idx]

        if remnans:
            nanpol = None 
        else:
            nanpol = "omit"

        baseline_str = "no"
        if baseline is not None:
            baseline_str = baseline
            baseline_fr = int(np.around(baseline * self.sess.pup_fps))
            baseline_data = data_array[:, : baseline_fr]
            data_array_base = math_util.mean_med(
                baseline_data, stats=stats, axis=-1, nanpol=nanpol
                )[:, np.newaxis]
            data_array = data_array - data_array_base

        row_indices = [range(len(data_array))]
        row_names = ["sequences"]
        if integ:
            integ_str = "yes"
            data_array = math_util.integ(
                data_array, 1./self.sess.pup_fps, axis=1, nanpol=nanpol)
        else:
            integ_str = "no"
            row_indices.append(xran)
            row_names.append("time_values")

        interp_str = pup_data.columns.unique("interpolated")[0]
        scale_str = pup_data.columns.unique("scaled")[0]

        col_index = pd.MultiIndex.from_product(
            [[datatype], [interp_str], [scale_str], [baseline_str], [integ_str]], 
            names=["datatype", "interpolated", "scaled", "baseline", 
                "integrated"])
        row_index = pd.MultiIndex.from_product(row_indices, names=row_names)

        pup_data_df = pd.DataFrame(
            data_array.reshape(-1), index=row_index, columns=col_index) 

        return pup_data_df


    #############################################
    def get_pup_diam_stats_df(self, pup_ref_fr, pre, post, integ=False, 
                              remnans=False, ret_arr=False, stats="mean", 
                              error="std", baseline=None, scale=False):
        """
        self.get_pup_diam_stats_df(pup_ref_fr, pre, post)

        Returns stats (mean and std or median and quartiles) for sequences of 
        pupil diameter data around specific pupil frame numbers. NaNs
        are omitted in calculating statistics.

        Required args:
            - pup_ref_fr (list): 1D list of reference pupil frame numbers
                                 around which to retrieve running data 
                                 (e.g., all 1st Gabor A frames)
            - pre (num)        : range of frames to include before each 
                                 reference frame number (in s)
            - post (num)       : range of frames to include after each 
                                 reference frame number (in s)

        Optional args:
            - integ (bool)    : if True, dF/F is integrated over sequences
                                default: False
            - remnans (bool)  : if True, NaN values are removed using linear 
                                interpolation. If False, NaN values (but
                                not Inf values) are omitted in calculating the 
                                data statistics.
                                default: False
            - ret_arr (bool)  : also return running data array, not just  
                                statistics
                                default: False 
            - stats (str)     : return mean ("mean") or median ("median")
                                default: "mean"
            - error (str)     : return std dev/quartiles ("std") or SEM/MAD 
                                ("sem")
                                default: "sem"
            - baseline (num)  : number of seconds from beginning of 
                                sequences to use as baseline. If None, data 
                                is not baselined.
                                default: None
            - scale (bool)    : if True, pupil diameter is scaled using 
                                full data array
                                default: False      

        Returns:
            - stats_df (pd DataFrame): dataframe containing pupil diameter 
                                       statistics organized in 
                hierarchical columns (dummy):
                    - datatype    : the specific datatype
                    - interpolated: whether data is interpolated ("yes", "no")
                    - scaled      : whether data is scaled ("yes", "no")
                    - baseline    : baseline used ("no", value)
                    - integrated  : whether data is integrated over sequences 
                                    ("yes", "no")
                hierarchical rows:
                    - general    : type of data ("data", "stats") (if ret_arr)
                    - sequences  : each sequence (trials, statistics)
                    - time_values: time values for each sequence, if data is 
                                   not integrated (optional)
        """

        pup_data_df = self.get_pup_diam_data(
            pup_ref_fr, pre, post, integ, remnans=remnans, baseline=baseline, 
            stats=stats, scale=scale)

        if remnans:
            nanpol = None 
        else:
            nanpol = "omit"

        stats_df = self.get_stats_df(
            pup_data_df, ret_arr, dims="sequences", stats=stats, error=error, 
            nanpol=nanpol)

        return stats_df


    #############################################
    def get_run_data(self, stim_ref_fr, pre, post, integ=False, remnans=True, 
                      baseline=None, stats="mean", scale=False):
        """
        self.get_run_data(stim_ref_fr, pre, post)

        Returns array of run data around specific stimulus frame numbers. 

        Required args:
            - stim_ref_fr (list): 1D list of reference stimulus frame numbers
                                  around which to retrieve running data 
                                  (e.g., all 1st Gabor A frames)
            - pre (num)         : range of frames to include before each 
                                  reference frame number (in s)
            - post (num)        : range of frames to include after each 
                                  reference frame number (in s)
        
        Optional args:
            - integ (bool)    : if True, running is integrated over frames
                                default: False
            - remnans (bool)  : if True, NaN values are removed using linear 
                                interpolation. If False, NaN values (but
                                not Inf values) are omitted in calculating the 
                                data statistics.
                                default: True
            - baseline (num)  : number of seconds from beginning of 
                                sequences to use as baseline. If None, data 
                                is not baselined.
                                default: None
            - stats (str)     : statistic to use for baseline, mean ("mean") or 
                                median ("median")
                                default: "mean"
            - scale (bool)    : if True, each ROI is scaled based on 
                                full trace array
                                default: False            
        Returns:            
            - run_data_df (pd DataFrame): dataframe containing running velocity 
                                          (in cm/s) for the frames of interest, 
                                          organized by:
                hierarchical columns (all dummy):
                    - datatype    : type of data ("run_velocity")
                    - interpolated: whether data is interpolated ("yes", "no")
                    - scaled      : whether data is scale ("yes", "no")
                    - baseline    : baseline used ("no", value)
                    - integrated  : whether data is integrated over sequences 
                                    ("yes", "no")
                    - filter_ks   : kernel size used to median filter running 
                                    velocity data
                    - diff_thr    : threshold of difference in running velocity 
                                    used to identify outliers
                hierarchical rows:
                    - sequences   : sequence numbers for each datapoint or 
                                    dataset
                    - time_values : time values in seconds for each datapoint 
                                    (optional)
        """

        datatype = "run_velocity"

        if not hasattr(self.sess, "run_data"):
            raise ValueError("Run 'self.load_run_data()' to load the running "
                "data correctly.")

        ran_fr, xran = self.sess.get_frames_timestamps(
            pre, post, fr_type="stim")

        if isinstance(stim_ref_fr[0], (list, np.ndarray)):
            raise ValueError("Frames must be passed as a 1D list, not by "
                "block.")

        # get corresponding running subblocks sequences x frames
        fr_idx = gen_util.num_ranges(
            stim_ref_fr, pre=-ran_fr[0], leng=len(xran))
                     
        # remove sequences with negatives or values above total number of stim 
        # frames
        neg_idx  = np.where(fr_idx[:, 0] < 0)[0].tolist()
        over_idx = np.where(fr_idx[:, -1] >= self.sess.tot_run_fr)[0].tolist()
        
        fr_idx = gen_util.remove_idx(fr_idx, neg_idx + over_idx, axis=0)

        if len(fr_idx) == 0:
            raise ValueError("No frames in list.")

        run_data = self.sess.get_run_velocity_by_fr(
            fr_idx, fr_type="stim", remnans=remnans, scale=scale)

        data_array = gen_util.reshape_df_data(
            run_data, squeeze_rows=False, squeeze_cols=True
            )

        if remnans:
            nanpol = None 
        else:
            nanpol = "omit"

        baseline_str = "no"
        if baseline is not None: # calculate baseline and subtract
            baseline_str = baseline
            if baseline > pre + post:
                raise ValueError("Baseline greater than sequence length.")
            baseline_fr = int(np.around(baseline * self.sess.stim_fps))
            baseline_data = data_array[:, : baseline_fr]
            data_array_base = math_util.mean_med(
                baseline_data, stats=stats, axis=-1, nanpol="omit"
                )[:, np.newaxis]
            data_array = data_array - data_array_base

        row_indices = [range(len(data_array))]
        row_names = ["sequences"]
        if integ:
            integ_str = "yes"
            data_array = math_util.integ(
                data_array, 1./self.sess.stim_fps, axis=1, nanpol=nanpol)
        else:
            integ_str = "no"
            row_indices.append(xran)
            row_names.append("time_values")

        interp_str = run_data.columns.unique("interpolated")[0]
        scale_str = run_data.columns.unique("scaled")[0]
        filter_ks = run_data.columns.unique("filter_ks")[0]
        diff_thr = run_data.columns.unique("diff_thr")[0]

        col_index = pd.MultiIndex.from_product(
            [[datatype], [interp_str], [scale_str], [baseline_str], 
            [integ_str], [filter_ks], [diff_thr]], 
            names=["datatype", "interpolated", "scaled", "baseline", 
            "integrated", "filter_ks", "diff_thr"])
        row_index = pd.MultiIndex.from_product(row_indices, names=row_names)

        run_data_df = pd.DataFrame(
            data_array.reshape(-1), index=row_index, columns=col_index) 

        return run_data_df


    #############################################
    def get_run_stats_df(self, stim_ref_fr, pre, post, integ=False,
                         remnans=True, ret_arr=False, stats="mean", 
                         error="std", baseline=None, scale=False):
        """
        self.get_run_stats_df(stim_ref_fr, pre, post)

        Returns stats (mean and std or median and quartiles) for sequences of 
        running data around specific stimulus frames.

        Required args:
            - stim_ref_fr (list): 1D list of reference stimulus frames numbers
                                  around which to retrieve running data 
                                  (e.g., all 1st Gabor A frames)
            - pre (num)         : range of frames to include before each 
                                  reference frame number (in s)
            - post (num)        : range of frames to include after each 
                                  reference frame number (in s)

        Optional args:
            - integ (bool)    : if True, dF/F is integrated over sequences
                                default: False
            - remnans (bool)  : if True, NaN values are removed using linear 
                                interpolation. If False, NaN values (but
                                not Inf values) are omitted in calculating the 
                                data statistics.
                                default: True
            - ret_arr (bool)  : also return running data array, not just  
                                statistics
                                default: False 
            - stats (str)     : return mean ("mean") or median ("median")
                                default: "mean"
            - error (str)     : return std dev/quartiles ("std") or SEM/MAD 
                                ("sem")
                                default: "sem"
            - baseline (num)  : number of seconds from beginning of 
                                sequences to use as baseline. If None, data 
                                is not baselined.
                                default: None
            - scale (bool)    : if True, running is scaled based on 
                                full trace array
                                default: False

        Returns:
            - stats_df (pd DataFrame): dataframe containing run velocity 
                                       statistics organized in 
                hierarchical columns (dummy):
                    - datatype    : the specific datatype
                    - interpolated: whether data is interpolated ("yes", "no")
                    - scaled      : whether data is scaled ("yes", "no")
                    - baseline    : baseline used ("no", value)
                    - integrated  : whether data is integrated over sequences 
                                    ("yes", "no")
                    - filter_ks   : kernel size used to median filter running 
                                    velocity data
                    - diff_thr    : threshold of difference in running velocity 
                                    used to identify outliers
                hierarchical rows:
                    - general    : type of data ("data", "stats") (if ret_arr)
                    - sequences  : each sequence (trials, statistics)
                    - time_values: time values for each sequence, if data is 
                                   not integrated (optional)
        """

        run_data_df = self.get_run_data(
            stim_ref_fr, pre, post, integ, baseline=baseline, stats=stats, 
            remnans=remnans, scale=scale)

        if remnans:
            nanpol = None 
        else:
            nanpol = "omit"

        stats_df = self.get_stats_df(
            run_data_df, ret_arr, dims="sequences", stats=stats, error=error, 
            nanpol=nanpol)

        return stats_df


    #############################################
    def get_roi_data(self, twop_ref_fr, pre, post, fluor="dff", integ=False, 
                     remnans=True, baseline=None, stats="mean", 
                     transients=False, scale=False, pad=(0, 0), smooth=False):
        """
        self.get_roi_data(twop_ref_fr, pre, post)

        Returns an array of 2p trace data around specific 2p frame numbers. 

        Required args:
            - twop_ref_fr (list): 1D list of 2p frame numbers 
                                  (e.g., all 1st Gabor A frames)
            - pre (num)         : range of frames to include before each 
                                  reference frame number (in s)
            - post (num)        : range of frames to include after each 
                                  reference frame number (in s)

        Optional args:
            - fluor (str)         : if "dff", dF/F is used, if "raw", ROI traces
                                    default: "dff"
            - integ (bool)        : if True, dF/F is integrated over frames
                                    default: False
            - remnans (bool)      : if True, ROIs with NaN/Inf values anywhere
                                    in session are excluded. If False, NaN 
                                    values (but not Inf values) are omitted in 
                                    calculating the data statistics.
                                    default: True
            - baseline (num)      : number of seconds from beginning of 
                                    sequences to use as baseline. If None, data 
                                    is not baselined.
                                    default: None
            - stats (str)         : statistic to use for baseline, mean ("mean") 
                                    or median ("median")
                                    default: "mean"
            - transients (bool)   : if True, only ROIs with transients are 
                                    retained
                                    default: False
            - scale (bool)        : if True, each ROI is scaled based on 
                                    full trace array
                                    default: False 
            - pad (tuple)         : number of frames to use as padding 
                                    (before, after)
                                    default: (0, 0)
            - smooth (bool or int): if not False, specifies the window length 
                                    to use in smoothing 
                                    default: False
        
        Returns:
            - roi_data_df (pd DataFrame): dataframe containing ROI trace data  
                                          for the frames of interest, organized 
                                          by:
                hierarchical columns (all dummy):
                    - datatype        : type of data (e.g., "roi_traces")
                    - nan_rois_removed: whether ROIs with NaNs/Infs were 
                                        removed ("yes", "no")
                    - scaled          : whether ROI data is scaled 
                                        ("yes", "no")
                    - baseline        : baseline used ("no", value)
                    - integrated      : whether data is integrated over 
                                        sequences ("yes", "no")
                    - smoothing       : smoothing padding ("(x, x)", "no")
                    - fluorescence    : type of data ("raw" or "dff")
                hierarchical rows:
                    - ROIs          : ROI indices
                    - sequences     : sequences numbers
                    - time_values   : time values in seconds for each datapoint 
                                      (optional)
        """

        datatype = "roi_traces"

        use_pad = pad
        if smooth:
            add_pad = np.ceil(smooth/2.0).astype(int)
            use_pad = [sub + add_pad for sub in use_pad]

        frame_n_df = self.sess.get_twop_fr_ran(twop_ref_fr, pre, post, use_pad)

        # get dF/F: ROI x seq x fr
        roi_data_df = self.sess.get_roi_seqs(
            gen_util.reshape_df_data(
                frame_n_df, squeeze_rows=False, squeeze_cols=True
                ), fluor=fluor, remnans=remnans, scale=scale)

        if transients:
            keep_rois = self.sess.get_active_rois(
                fluor=fluor, stimtype=None, remnans=remnans)
            drop_rois = set(roi_data_df.index.unique("ROIs")) - set(keep_rois)
            if len(drop_rois) != 0:
                roi_data_df = roi_data_df.drop(
                    drop_rois, axis="index", level="ROIs")
        
        dims = [len(roi_data_df.index.unique(row)) 
            for row in roi_data_df.index.names]

        data_array = roi_data_df.to_numpy().reshape(dims)

        if remnans:
            nanpol = None
        else:
            nanpol = "omit"

        row_indices = [roi_data_df.index.unique("ROIs"), 
            range(data_array.shape[1])]
        
        row_names = ["ROIs", "sequences"]
        baseline_str = "no"
        if baseline is not None: # calculate baseline and subtract
            baseline_str = baseline
            if baseline > pre + post:
                raise ValueError("Baseline greater than sequence length.")
            baseline_fr = int(np.around(baseline * self.sess.twop_fps))
            baseline_data = data_array[:, :, : baseline_fr]
            data_array_base = math_util.mean_med(
                baseline_data, stats=stats, axis=-1, nanpol=nanpol
                )[:, :, np.newaxis]
            data_array = data_array - data_array_base

        smooth_str = "no"
        if smooth:
            smooth_str = f"{smooth}"
            data_array = math_util.rolling_mean(
                data_array.astype(float), win=add_pad)
            # cut down based on pad
            data_array = data_array[:, :, add_pad:-add_pad]
        
        if integ:
            integ_str = "yes"
            data_array = math_util.integ(
                data_array, 1./self.sess.twop_fps, axis=-1, nanpol=nanpol)
        else:
            integ_str = "no"
            row_indices.append(frame_n_df.index.unique(
                level="time_values").tolist())
            row_names.append("time_values")

        remnans_str = roi_data_df.columns.unique("nan_rois_removed")[0]
        scale_str = roi_data_df.columns.unique("scaled")[0]
        fluor_str = roi_data_df.columns.unique("fluorescence")[0]

        col_index = pd.MultiIndex.from_product(
            [[datatype], [remnans_str], [scale_str], [baseline_str], 
            [integ_str], [smooth_str], [fluor_str], ], 
            names=["datatype", "nan_rois_removed", "scaled", "baseline", 
            "integrated", "smoothing", "fluorescence"])
        row_index = pd.MultiIndex.from_product(row_indices, names=row_names)

        roi_data_df = pd.DataFrame(
            data_array.reshape(-1), index=row_index, columns=col_index) 

        return roi_data_df
    
    
    #############################################
    def get_roi_stats_df(self, twop_ref_fr, pre, post, byroi=True, 
                         fluor="dff", integ=False, remnans=True, 
                         ret_arr=False, stats="mean", error="std", 
                         baseline=None, transients=False, scale=False, 
                         smooth=False):
        """
        self.get_roi_stats_df(twop_ref_fr, pre, post)

        Returns stats (mean and std or median and quartiles) for sequences of 
        roi traces centered around specific 2p frame numbers.

        Required args:
            - twop_ref_fr (list): 1D list of 2p frame numbers 
                                  (e.g., all 1st Gabor A frames)
            - pre (num)         : range of frames to include before each 
                                  reference frame number (in s)
            - post (num)        : range of frames to include after each  
                                  reference frame number (in s)

        Optional args:
            - byroi (bool)        : if True, returns statistics for each ROI. 
                                    If False, returns statistics across ROIs
                                    default: True 
            - fluor (str)         : if "dff", dF/F is used, if "raw", ROI traces
                                    default: "dff"
            - integ (bool)        : if True, dF/F is integrated over sequences
                                    default: False
            - remnans (bool)      : if True, ROIs with NaN/Inf values anywhere
                                    in session are excluded. If False, NaN 
                                    values (but not Inf values) are omitted in 
                                    calculating the data statistics.
                                    default: True
            - ret_arr (bool)      : also return ROI trace data array, not just  
                                    statistics.
            - stats (str)         : return mean ("mean") or median ("median")
                                    default: "mean"
            - error (str)         : return std dev/quartiles ("std") or SEM/MAD 
                                    ("sem")
                                    default: "sem"
            - baseline (num)      : number of seconds from beginning of 
                                    sequences to use as baseline. If None, data 
                                    is not baselined.
                                    default: None
            - transients (bool)   : if True, only ROIs with transients are 
                                   retained
                                   default: False
            - scale (bool)        : if True, each ROI is scaled based on 
                                    full trace array
                                    default: False
            - smooth (bool or int): if not False, specifies the window length 
                                    to use in smoothing 
                                    default: False

        Returns:
            - stats_df (pd DataFrame): dataframe containing run velocity 
                                       statistics organized in 
                hierarchical columns (dummy):
                    - datatype    : the specific datatype
                    - interpolated: whether data is interpolated ("yes", "no")
                    - scaled      : whether data is scaled ("yes", "no")
                    - baseline    : baseline used ("no", value)
                    - integrated  : whether data is integrated over sequences 
                                    ("yes", "no")
                    - filter_ks   : kernel size used to median filter running 
                                    velocity data
                    - diff_thr    : threshold of difference in running velocity 
                                    used to identify outliers
                hierarchical rows:
                    - general    : type of data ("data", "stats") (if ret_arr)
                    - ROIs       : ROI indices
                    - sequences  : each sequence (trials, statistics)
                    - time_values: time values for each sequence, if data is 
                                   not integrated (optional)
        """
        
        roi_data_df = self.get_roi_data(
            twop_ref_fr, pre, post, fluor, integ, remnans=remnans, 
            baseline=baseline, stats=stats, transients=transients, scale=scale, 
            smooth=smooth)
            
        # order in which to take statistics on data
        dims = ["sequences", "ROIs"]
        if byroi:
            dims = ["sequences"]

        if remnans:
            nanpol = None
        else:
            nanpol = "omit"

        stats_df = self.get_stats_df(
            roi_data_df, ret_arr, dims=dims, stats=stats, error=error, 
            nanpol=nanpol)

        return stats_df


    #############################################
    def get_run(self, by="block", remnans=True, scale=False):
        """
        self.get_run()

        Returns run values for each stimulus frame of each stimulus block.

        Optional args:
            - by (str)      : determines whether run values are returned in a  
                              flat list ("frame"), grouped by block ("block"), 
                              or further grouped by display sequence ("disp")
                              default: "block"
            - remnans (bool): if True, NaN values are removed using linear 
                              interpolation.
                              default: True
            - scale (bool)  : if True, each ROI is scaled based on 
                              full trace array
                              default: False
        Returns:
            - sub_run_df (pd DataFrame): dataframe containing running velocity 
                                         values (in cm/s) for the frames
                                         of interest, and optionally display or 
                                         block numbers, organized by:
                hierarchical columns (all dummy):
                    - datatype    : type of data (e.g., "run_velocity", 
                                    "block_n" or "display_sequence_n")
                    - interpolated: whether data is interpolated ("yes", "no")
                    - scaled      : whether data is scaled ("yes", "no")
                    - filter_ks   : kernel size used to median filter running 
                                    velocity data
                    - diff_thr    : threshold of difference in running velocity 
                                    used to identify outliers
                hierarchical rows:
                    - "info"      : type of information contained 
                                    ("frames": values for each frame)
                    - "specific"  : specific type of information contained 
                                    (frame number)
        """
        
        run_df = self.sess.get_run_velocity(remnans=remnans, scale=scale)

        if by not in ["block", "frame", "disp"]:
            gen_util.accepted_values_error("by", by, ["block", "frame", "disp"])

        for d in self.block_params.index.unique("display_sequence_n"):
            for b in self.block_params.loc[d].index.unique("block_n"):
                row = self.block_params.loc[(d, b)]
                # pd.IndexSlice: slice end is included
                idx = pd.IndexSlice["frames", 
                    row["start_seg"][0] : row["end_seg"][0] + 1]
                if by in ["block", "frame"]:
                    col = "block_n"
                    run_df.loc[idx, col] = b
                elif by == "disp":
                    col = "display_sequence_n"
                    run_df.loc[idx, col] = d

        sub_run_df = run_df.loc[~run_df[col].isna()]
        if by == "frame":
            sub_run_df = sub_run_df.drop("block_n", axis="columns")
    
        return sub_run_df

    
    #############################################
    def get_segs_by_twopfr(self, twop_fr):
        """
        self.get_segs_by_twopfr(twop_fr)

        Returns the stimulus segment numbers for the given two-photon imaging
        frames using linear interpolation, and rounds the segment numbers.

        Required args:
            - twop_fr (array-like): set of 2p imaging frames for which 
                                    to get stimulus seg numbers
        
        Returns:
            - segs (nd array): segment numbers (int), with same dimensions 
                               as input array
        """

        twop_fr = np.asarray(twop_fr)

        # make sure the frames are within the range of 2p frames
        if (twop_fr >= self.sess.tot_twop_fr).any() or (twop_fr < 0).any():
            raise UserWarning("Some of the specified frames are out of range")

        # perform linear interpolation on the running velocity
        segs = np.interp(twop_fr, self.sess.stim2twopfr, self.stim_seg_list)

        segs = segs.astype(int)

        return segs

    
#############################################
#############################################
class Gabors(Stim):
    """
    The Gabors object inherits from the Stim object and describes gabor 
    specific properties.
    """

    def __init__(self, sess, stim_n):
        """
        self.__init__(sess, stim_n)
        
        Initializes and returns a Gabors object, and the attributes below. 
        
        Calls:
            - self._update_block_params()
        
        Attributes:
            - deg_per_pix (num)       : degrees per pixels used in conversion
                                        to generate stimuli
            - n_patches (int)         : number of gabors 
            - ori_kaps (float or list): orientation kappa (calculated from std) 
                                        for each gabor block (only one value 
                                        for production data), not ordered
            - ori_std (float or list) : orientation standard deviation for each
                                        gabor block (only one value for 
                                        production data) (rad)
            - oris (list)             : mean orientations through which the 
                                        gabors cycle (in deg)
            - oris_pr (2D array)      : specific orientations for each segment 
                                        of each gabor (in deg, -180 to 180), 
                                        structured as:
                                            segments x gabor
            - phase (num)             : phase of the gabors (0-1)
            - pos (3D array)          : gabor positions for each segment type
                                        (A, B, C, D, U), in pixels with window
                                        center being (0, 0), structured as:
                                            segment type x gabor x coord (x, y) 
            - post (num)              : number of seconds from frame A that are
                                        included in a set (G, A, B, C, D/U)
            - pre (num)               : number of seconds before frame A that
                                        are included in a set 
                                        (G, A, B, C, D/U)
            - set_len_s (num)         : length of a set in seconds
                                        (set: G, A, B, C, D/U)
            - sf (num)                : spatial frequency of the gabors 
                                        (in cyc/pix)
            - size_pr (2D array)      : specific gabor sizes for each segment
                                        types (A, B, C, D, U) (in pix), 
                                        structured as:
                                            segment type x gabor
            - size_ran (list)         : range of gabor sizes (in pix)
            - units (str)             : units used to create stimuli in 
                                        PsychoPy (e.g., "pix")
        
        Required args:
            - sess (Session)  : session to which the gabors belongs
            - stim_n (int)    : this stimulus" number, x in 
                                sess.stim_dict["stimuli"][x]
        """

        super().__init__(sess, stim_n, stimtype="gabors")

        stim_info = self.sess.stim_dict["stimuli"][self.stim_n]
        
        # gabor specific parameters
        if self.sess.runtype == "pilot":
            gabor_par = stim_info["stimParams"]["gabor_params"]
            sess_par  = stim_info["stimParams"]["subj_params"]
            self.ori_std = copy.deepcopy(gabor_par["ori_std"])
            oris_pr = np.asarray(stim_info["stimParams"]["orisByImg"])
        elif self.sess.runtype == "prod":
            gabor_par = stim_info["stim_params"]["gabor_params"]
            sess_par  = stim_info["stim_params"]["session_params"]
            self.ori_std = gabor_par["ori_std"]
            oris_pr = np.asarray(sess_par["orisbyimg"])

        self.win_size = sess_par["windowpar"][0]
        self.deg_per_pix = sess_par["windowpar"][1]
        self.n_patches = gabor_par["n_gabors"]
        self.oris      = sorted(gabor_par["oris"])
        self.phase     = gabor_par["phase"]  
        self.sf        = gabor_par["sf"]
        self.units     = gabor_par["units"]
        self.pos_x     = np.asarray(list(zip(*sess_par["possize"]))[0])[:, :, 0]
        self.pos_y     = np.asarray(list(zip(*sess_par["possize"]))[0])[:, :, 1]
        self.sizes_pr  = np.asarray(list(zip(*sess_par["possize"]))[1])

        self.pos_x_ran = [-self.win_size[0]/2., self.win_size[0]/2.]
        self.pos_y_ran = [-self.win_size[1]/2., self.win_size[1]/2.]
        self.ori_ran = [-180, 180]
        
        # modify self.oris_pr U frames, as they are rotated 90 deg from what is 
        # recorded
        seg_surps = np.asarray(self.sess.stim_df.loc[
            (self.sess.stim_df["stimType"] == "g")]["surp"])
        seg_gabfr = np.asarray(self.sess.stim_df.loc[
            (self.sess.stim_df["stimType"] == "g")]["gabfr"])
        seg_surp_gabfr = np.asarray((seg_surps == 1) * (seg_gabfr == 3))
        self.oris_pr = oris_pr + seg_surp_gabfr[:, np.newaxis] * 90
        # in case some U frames values are now above upper range, so fix
        ori_hi = np.where(self.oris_pr > self.ori_ran[1])
        new_vals = self.ori_ran[0] + self.oris_pr[ori_hi] - self.ori_ran[1]
        self.oris_pr[ori_hi] = new_vals

        size_ran = copy.deepcopy(gabor_par["size_ran"])
        if self.units == "pix":
            self.sf = gabor_par["sf"] * self.deg_per_pix 
            size_ran = [x / self.deg_per_pix for x in size_ran]
        else:
             raise ValueError("Expected self.units to be pix.")

        # Convert to size as recorded in PsychoPy
        gabor_modif = 1. / (2 * np.sqrt(2 * np.log(2))) * gabor_par["sd"]
        self.size_ran = [np.around(x * gabor_modif) for x in size_ran]

        # kappas calculated as 1/std**2
        if self.sess.runtype == "pilot":
            self.ori_kaps = [1. / x ** 2 for x in self.ori_std] 
        elif self.sess.runtype == "prod":
            self.ori_kaps = 1. / self.ori_std ** 2

        # seg sets (hard-coded, based on the repeating structure  we are 
        # interested in, namely: blank, A, B, C, D/U)
        self.pre  = 1 * self.seg_len_s # 0.3 s blank
        self.post = self.n_seg_per_set * self.seg_len_s # 1.2 ms gabors
        self.set_len_s = self.pre + self.post
        
        # get parameters for each block
        self._update_block_params()


    #############################################
    def _update_block_params(self):
        """
        self._update_block_params()

        Updates self.block_params with stimulus parameter information.

        Attributes:
            - block_params (pd DataFrame): updates dataframe with gabor 
                                           parameter for each display sequence 
                                           and block:
                hierarchical columns:
                    - parameters  : parameter names 
                                    ("direction", "size", "number")
                hierarchical rows:
                    - "display_sequence_n": display sequence number
                    - "block_n"           : block number (across display 
                                            sequences)
        """


        for d in self.block_params.index.unique("display_sequence_n"):
            for b in self.block_params.loc[d].index.unique("block_n"):
                row = self.block_params.loc[(d, b)]
                segs = self.sess.stim_df.loc[
                    (self.sess.stim_df["stimType"]==self.stimtype[0]) & 
                    (self.sess.stim_df["stimSeg"] >= row["start_seg"][0]) & 
                    (self.sess.stim_df["stimSeg"] < row["end_seg"][0])]
                # skipping stimPar1 which indicates gabor orientations which 
                # change at each gabor sequence presentation
                stimPar2 = segs["stimPar2"].unique().tolist()
                if len(stimPar2) > 1:
                    raise ValueError(f"Block {b} of {self.stimtype} "
                        "comprises segments with different "
                        f"stimPar2 values: {stimPar2}")
                self.block_params.loc[(d, b), ("kappa", )] = stimPar2[0]


    #############################################
    def get_A_segs(self, by="block"):
        """
        self.get_A_segs()

        Returns lists of A gabor segment numbers.

        Optional args:
            - by (str): determines whether segment numbers are returned in a 
                        flat list ("seg"), grouped by block ("block"), or 
                        further grouped by display sequence ("disp")
                        default: "block"
        Returns:
            - A_segs (list): list of A gabor segment numbers.
        """
        
        A_segs = self.get_segs_by_criteria(gabfr=0, by=by)

        return A_segs


    #############################################
    def get_A_frame_1s(self, by="block"):
        """
        self.get_A_frame_1s()

        Returns list of first frame number for each A gabor segment number.

        Optional args:
            - by (str): determines whether frame numbers are returned in a flat 
                        list ("frame"), grouped by block ("block"), or further 
                        grouped by display sequence ("disp")
                        default: "block"
     
        Returns:
            - A_segs (list) : lists of first frame number for each A gabor 
                              segment number
        """
        
        A_frames = self.get_stim_fr_by_criteria(gabfr=0, by=by)

        return A_frames
    

    #############################################
    def get_stim_par_by_seg(self, segs, pos=True, ori=True, size=True, 
                            scale=False):
        """
        self.get_stim_par_by_seg(segs)

        Returns stimulus parameters for specified segments (0s for any segments 
        out of range).

        NOTE: A warning will be logged if any of the segments are out of 
        range, unless they are -1. (-1 parameter values will be returned for 
        these segments, as if they were grayscreen frames.)

        Required args:
            - segs (nd array): array of segments for which parameters are
                               requested
        
        Optional args:
            - pos (bool)  : if True, the positions of each Gabor are returned
                            (in x and y separately)
                            default: True
            - ori (bool)  : if True, the orientations of each Gabor are returned
                            (in deg, -180 to 180)
                            default: True
            - size (bool) : if True, the sizes of each Gabor are returned
                            default: True
            - scale (bool): if True, values are scaled to between -1 and 1 
                            (each parameter type separately, based on full 
                            possible ranges)
                            default: False
     
        Returns:
            - full_param_df (pd DataFrame): dataframe containing gabor parameter
                                            values for each segment,
                                            organized by:
                hierarchical columns:
                    - parameters  : parameter names ("pos_x", "pos_y", "size", 
                                    "ori")
                hierarchical rows:
                    (- "sequence" : sequence number for first dimension of segs
                     - "sub_sequence", ...)
                    - "seg_n"     : segment number
                    - "gabor_n"   : gabor number
        """

        segs = np.asarray(segs)

        # a few checks that implementation is appropriate based on stimulus info        
        if self.block_params.loc[(0, 0), "start_seg", ][0] != 0:
            raise NotImplementedError("Function not properly implemented if "
                "the minimum segment is not 0.")
        if self.block_params["end_seg", ].max() != self.oris_pr.shape[0]:
            raise NotImplementedError("Function not properly implemented if "
                "the maximum segment is not the same as the number of "
                "orientations recorded.")

        # check that at least one parameter type is requested
        if not(pos or ori or size):
            raise ValueError("At least one of the following must be True: "
                "pos, ori, size.")
        
        sub_df = self.sess.stim_df.loc[
            (self.sess.stim_df["stimType"] == "g") &
            (self.sess.stim_df["stimSeg"].isin(np.asarray(segs).reshape(-1)))
            ][["stimSeg", "gabfr", "surp"]]

        miss_segs = set(np.unique(segs).tolist()) - \
            set(sub_df["stimSeg"].unique())
        if len(miss_segs) != 0 and miss_segs != {-1}:
            logger.warning("Some of the segments requested are out of "
                "range for Gabors stimulus (No warning for -1 segs.).", 
                extra={"spacing": TAB})

        for miss_seg in miss_segs:
            sub_df = sub_df.append(
                {"stimSeg": miss_seg, "gabfr": -1, "surp": -1}, 
                ignore_index=True)

        sub_df.loc[(sub_df["gabfr"] == 3) & (sub_df["surp"] == 1), "gabfr"] = 4
        
        row_index = pd.MultiIndex.from_product(
            [sub_df["stimSeg"], range(self.n_patches)], 
            names=["seg_n", "gabor_n"])

        param_df = pd.DataFrame(None, index=row_index, columns=[])

        sub_df.loc[sub_df["stimSeg"].isin(miss_segs), "stimSeg"] = -1

        pos_x_dict = {"name": "pos_x",
                      "bool": pos,
                      "attribs": self.pos_x, 
                      "extr": self.pos_x_ran}

        pos_y_dict = {"name": "pos_y",
                      "bool": pos, 
                      "attribs": self.pos_y, 
                      "extr": self.pos_y_ran}

        size_dict = {"name": "size",
                     "bool": size, 
                     "attribs": self.sizes_pr, 
                     "extr": self.size_ran}

        ori_dict = {"name": "ori",
                    "bool": ori, 
                    "attribs": self.oris_pr, 
                    "extr": self.ori_ran}

        for par_dict in [pos_x_dict, pos_y_dict, size_dict, ori_dict]:
            if par_dict["bool"]:
                vals = np.asarray(par_dict["attribs"])
                if scale:
                    sub = min(par_dict["extr"])
                    div = max(par_dict["extr"]) - sub
                    vals = math_util.scale_data(vals, facts=[sub, div, 2, -1])
                # add 0s for -1 segments
                vals = np.append(vals, np.full([1, self.n_patches], 0), axis=0)
                col = "gabfr"
                if par_dict["name"] == "ori":
                    col = "stimSeg"
                ref_ns = sub_df[col].to_numpy().astype(int)
                param_df[par_dict["name"]] = np.asarray(
                    vals[ref_ns]).reshape(-1)

        # create a dataframe organized like 'segs' and transfer data
        names = ["{}sequence".format("".join(["sub_"] * i)) 
            for i in range(len(segs.shape) - 1)]

        idx_tup = np.asarray(list(itertools.product(*[
            range(n) for n in list(segs.shape) + [self.n_patches]])))
        idx_tup[:, -2] = np.repeat(segs.reshape(-1), self.n_patches)

        col_index = pd.MultiIndex.from_product(
            [param_df.columns], names=["parameters"])

        row_index = pd.MultiIndex.from_tuples([tuple(val) for val in idx_tup], 
            names=names + ["seg_n", "gabor_n"])

        full_param_df = pd.DataFrame(
            None, index=row_index, columns=col_index)

        use_idx = [vals[-2:] for vals in full_param_df.index.values]
        for col in param_df.columns:
            full_param_df[col, ] = param_df.loc[use_idx, col].values

        return full_param_df


    #############################################
    def get_stim_par_by_twopfr(self, twop_ref_fr, pre, post, pos=True, 
                               ori=True, size=True, scale=False):
        """
        self.get_stim_par_by_seg(segs)

        Returns stimulus parameters for 2p frame sequences specified by the 
        reference frame numbers and pre and post ranges.

        NOTE: A warning will be logged if any of the 2p frame sequences occur
        during Bricks frames. 
        (-1 parameter values will be returned for these frames, as if they
        were grayscreen frames.)

        Required args:
            - twop_ref_fr (list): 1D list of 2p frame numbers 
                                  (e.g., all 1st Gabor A frames)
            - pre (num)         : range of frames to include before each 
                                  reference frame number (in s)
            - post (num)        : range of frames to include after each 
                                  reference frame number (in s)
                    
        Optional args:
            - pos (bool)  : if True, the positions of each Gabor are returned
                            (in x and y separately)
                            default: True
            - ori (bool)  : if True, the orientations of each Gabor are returned
                            default: True
            - size (bool) : if True, the sizes of each Gabor are returned
                            default: True
            - scale (bool): if True, values are scaled to between -1 and 1 
                            (each parameter type separately, to its full 
                            possible range)
                            default: False
     
        Returns:
            - full_param_df (pd DataFrame): dataframe containing gabor parameter
                                            values for each two-photon frame,
                                            organized by:
                hierarchical columns:
                    - parameters  : parameter names ("pos_x", "pos_y", "size", 
                                    "ori")
                hierarchical rows:
                    (- "sequence" : sequence number for first dimension of segs
                     - "sub_sequence", ...)
                    - "twop_fr_n" : two-photon frame number
                    - "gabor_n"   : gabor number
        """

        twop_fr_seqs = gen_util.reshape_df_data(
            self.sess.get_twop_fr_ran(
                twop_ref_fr, pre, post).loc[:, ("twop_fr_n", )], 
            squeeze_cols=True)

        # check whether any of the segments occur during Bricks
        if hasattr(self.sess, "bricks"):
            bri_segs = self.sess.bricks.get_segs_by_twopfr(twop_fr_seqs)
            if not (bri_segs == -1).all():
                logger.warning("Some of the frames requested occur while "
                      "Bricks are presented.", extra={"spacing": TAB})

        # get seg numbers for each twopfr in each sequence
        seq_segs = self.get_segs_by_twopfr(twop_fr_seqs)

        full_param_df = self.get_stim_par_by_seg(
            seq_segs, pos=pos, ori=ori, size=size, scale=scale)

        full_param_df.index.set_levels(
            twop_fr_seqs.reshape(-1), level="seg_n", inplace=True)

        full_param_df.index.rename("twop_fr_n", level="seg_n", inplace=True)

        return full_param_df


#############################################
#############################################
class Bricks(Stim):
    """
    The Bricks object inherits from the Stim object and describes bricks 
    specific properties. For production data, both brick stimuli are 
    initialized as one stimulus object.
    """

    def __init__(self, sess, stim_n):
        """
        self.__init__(sess, stim_n)
        
        Initializes and returns a bricks object, and the attributes below. 
        
        Calls:
            - self._update_block_params()

        Attributes:
            - deg_per_pix (num)       : degrees per pixels used in conversion
                                        to generate stimuli
            - direcs (list)           : main brick direction for each block
            - flipfrac (num)          : fraction of bricks that flip direction 
                                        at each surprise
            - n_bricks (float or list): n_bricks for each brick block (only one
                                        value for production data), not ordered
            - sizes (int or list)     : brick size for each brick block (only
                                        one value for production data) (in pix), 
                                        not ordered
            - speed (num)             : speed at which the bricks are moving 
                                        (in pix/sec)
            - units (str)             : units used to create stimuli in 
                                        PsychoPy (e.g., "pix")
        
        Required args:
            - sess (Session)  : session to which the bricks belongs
            - stim_n (int)    : this stimulus" number (2 in the case of
                                production bricks): x in 
                                sess.stim_dict["stimuli"][x]
        """

        super().__init__(sess, stim_n, stimtype="bricks")
            
        stim_info = self.sess.stim_dict["stimuli"][self.stim_n]

        # initialize brick specific parameters
        if self.sess.runtype == "pilot":
            sqr_par     = stim_info["stimParams"]["square_params"]
            self.units  = sqr_par["units"]
            self.deg_per_pix = stim_info["stimParams"]["subj_params"]["windowpar"][1]
            self.direcs = sqr_par["direcs"]
            self.sizes  = copy.deepcopy(sqr_par["sizes"])
            
            # calculate n_bricks, as wasn"t explicitly recorded
            max_n_brick   = stim_info["stimParams"]["elemParams"]["nElements"]
            prod          = float(max_n_brick) * min(self.sizes)**2
            self.n_bricks = [int(prod/size**2) for size in self.sizes]

            if self.units == "pix":
                # sizes recorded in deg, so converting to pix (only for pilot)
                self.sizes = [np.around(x/self.deg_per_pix) for x in self.sizes]
            
        elif self.sess.runtype == "prod":
            sqr_par       = stim_info["stim_params"]["square_params"]
            stim_info2    = self.sess.stim_dict["stimuli"][self.stim_n_all[1]]
            self.units    = sqr_par["units"]
            self.deg_per_pix = stim_info[
                "stim_params"]["session_params"]["windowpar"][1]
            self.direcs   = [stim_info["stim_params"]["direc"], 
                             stim_info2["stim_params"]["direc"]]
            self.sizes    = stim_info["stim_params"]["elemParams"]["sizes"]
            self.n_bricks = stim_info["stim_params"]["elemParams"]["nElements"]
        
        self.direcs = [sess_gen_util.get_bri_screen_mouse_direc(direc) 
            for direc in self.direcs]

        self.speed = sqr_par["speed"]
        if self.units == "pix":
            # recorded in deg, so converting to pix
            self.speed = self.speed/self.deg_per_pix
        else:
            raise ValueError("Expected self.units to be pix.")
       
        self.flipfrac = sqr_par["flipfrac"]

        # set parameters for each block
        self._update_block_params()


    #############################################
    def _update_block_params(self):
        """
        self._update_block_params()

        Updates self.block_params with stimulus parameter information.

        Attributes:
            - block_params (pd DataFrame): updates dataframe with brick 
                                           parameters for each display sequence 
                                           and block:
                hierarchical columns:
                    - parameters  : parameter names 
                                    ("direction", "size", "number")
                hierarchical rows:
                    - "display_sequence_n": display sequence number
                    - "block_n"           : block number (across display 
                                            sequences)
        """

        for d in self.block_params.index.unique("display_sequence_n"):
            for b in self.block_params.loc[d].index.unique("block_n"):
                row = self.block_params.loc[(d, b)]
                segs = self.sess.stim_df.loc[
                    (self.sess.stim_df["stimType"]==self.stimtype[0]) & 
                    (self.sess.stim_df["stimSeg"] >= row["start_seg"][0]) & 
                    (self.sess.stim_df["stimSeg"] < row["end_seg"][0])]
                for source_name, par_name in zip(
                    ["stimPar2", "stimPar1"], ["direction", "size"]):
                    stimPar = segs[source_name].unique().tolist()
                    if len(stimPar) > 1:
                        raise ValueError(f"Block {b} of {self.stimtype} "
                            "comprises segments with different "
                            f"{source_name} values: {stimPar}")
                    self.block_params.loc[(d, b), (par_name, )] = stimPar[0]
                
                # add n_bricks info
                if self.sess.runtype == "prod":
                    self.block_params.loc[(d, b), ("number", )] = self.n_bricks
                else:
                    if (self.block_params.loc[(d, b), ("size", )] == \
                        min(self.sizes)):
                        self.block_params.loc[(d, b), ("number", )] = \
                            max(gen_util.list_if_not(self.n_bricks))
                    else:
                        self.block_params.loc[(d, b), ("number", )] = \
                            min(gen_util.list_if_not(self.n_bricks))


    #############################################
    def get_dir_segs_reg(self, by="block"):
        """
        self.get_dir_segs_reg()

        Returns two lists of stimulus segment numbers, the first is a list of 
        the temporal moving segments. The second is a list of nasal 
        moving segments. Both lists exclude surprise segments.

        Optional args:
            - by (str): determines whether segment numbers are returned in a 
                        flat list ("seg"), grouped by block ("block"), or 
                        further grouped by display sequence ("disp")
                        default: "block"  
        Returns:
            - temp_segs (list) : list of temporal (head to tail) moving segment 
                                 numbers, excluding surprise segments.
            - nasal_segs (list): list of nasal (tail to head) moving segment 
                                 numbers, excluding surprise segments.
        """

        temp_segs = self.get_segs_by_criteria(bri_dir="temp", surp=0, by=by)
        nasal_segs  = self.get_segs_by_criteria(bri_dir="nasal", surp=0, by=by)

        return temp_segs, nasal_segs


#############################################
#############################################
class Grayscr():
    """
    The Grayscr object describes describes grayscreen specific properties.

    NOTE: Not well fleshed out, currently.
    """

    
    def __init__(self, sess):
        """
        self.__init__(sess)
        
        Initializes and returns a grayscr object, and the attributes below. 

        Attributes:
            - sess (Session object): session to which the grayscr belongs
            - gabors (bool): if True, the session to which the grayscreen 
                             belongs has a gabors attribute
        
        Required args:
            - sess (Session object): session to which the grayscr belongs
            - stim_n (int): this stimulus" number (2 in the case of
                            production bricks): x in 
                            sess.stim_dict["stimuli"][x]
        """

        self.sess = sess
        if hasattr(self.sess, "gabors"):
            self.gabors = True
        else:
            self.gabors = False
        

    #############################################
    def __repr__(self):
        return (f"{self.__class__.__name__} (session {self.sess.sessid})")

    def __str__(self):
        return repr(self)


    #############################################        
    def get_all_nongab_stim_fr(self):
        """
        self.get_all_nongab_stim_fr()

        Returns a lists of grayscreen stimulus frame numbers, excluding 
        grayscreen stimulus frames occurring during gabor stimulus blocks, 
        including grayscreen stimulus frames flanking gabor stimulus blocks.
        
        Returns:
            - grays (list): list of grayscreen stimulus frames.
        """

        frames = []
        if self.gabors:
            frames_gab = np.asarray(self.sess.gabors.stim_seg_list)
            for b in self.sess.gabors.block_params.index.unique("block_n"):
                row = self.sess.gabors.block_params.loc[pd.IndexSlice[:, b], ]
                frames_gab[row["start_stim_fr", ].values[0]: 
                    row["end_stim_fr", ].values[0]] = 0
            frames.append(frames_gab)
        if hasattr(self.sess, "bricks"):
            frames.append(self.sess.bricks.stim_seg_list)
        length = len(frames)
        if length == 0:
            raise ValueError("No frame lists were found for either stimulus "
                " type (gabors, bricks).")
        elif length == 1:
            frames_sum = np.asarray(frames)
        else:
            frames_sum = np.sum(np.asarray(frames), axis=0)
        grays = np.where(frames_sum == length * -1)[0].tolist()

        if len(grays) == 0:
            raise ValueError("No grayscreen frames were found outside of "
                "gabor stimulus sequences.")

        return grays


    #############################################
    def get_first_nongab_stim_fr(self):
        """
        self.get_first_nongab_stim_fr()

        Returns every first grayscreen stimulus frame number for every 
        grayscreen sequence occuring outside of gabor stimulus blocks, and 
        the number of consecutive grayscreen stimulus frames for each sequence. 
                
        NOTE: any grayscreen stimulus frames for sequences flanking gabor 
        stimulus blocks are included in the returned list.
        
        Returns:
            - first_grays_df (pd DataFrame): dataframe containing stimulus 
                                             frame information on each first 
                                             grayscreen sequence, outside of 
                                             gabor stimulus blocks, with 
                columns:
                    - "first_stim_fr": first stimulus frame number for the 
                                       grayscreen sequence
                    - "n_stim_fr"    : length of grayscreen sequence
        """

        first_grays_df = pd.DataFrame()
        
        grays_all = np.asarray(self.get_all_nongab_stim_fr())
        first_grays_idx = [0] + \
            (np.where(np.diff(grays_all) != 1)[0] + 1).tolist() + \
            [len(grays_all)]
        
        first_grays_df["first_stim_fr"] = grays_all[
            np.asarray(first_grays_idx)[:-1]]
        
        first_grays_df["n_stim_fr"] = np.diff(first_grays_idx).tolist()

        return first_grays_df


    #############################################
    def get_all_gabG_stim_fr(self, by="block"):
        """
        self.get_all_gabG_stim_fr()

        Returns a list of grayscreen stimulus frame numbers for every 
        grayscreen (G) segment during a gabor block, excluding grayscreen 
        segments flanking the gabor blocks.

        Optional args:
            - by (str): determines whether frame numbers are returned in a 
                        flat list ("frame"), grouped by block ("block"), or 
                        further grouped by display sequence ("disp")
                        default: "block"    

        Returns:
            - gab_Gs (list): nested list of grayscreen stimulus frame 
                             numbers for every grayscreen (G) segment 
                             during gabors
        """
        
        if not self.gabors:
            raise ValueError("Session does not have a Gabors stimulus.")
        gabors = self.sess.gabors
        frames_gab = np.asarray(self.sess.gabors.stim_seg_list)
        
        gab_Gs = []
        for d in gabors.block_params.index.unique("display_sequence_n"):
            temp = []
            for b in gabors.block_params.loc[d].index.unique("block_n"):
                row = gabors.block_params.loc[(d, b)].astype(int)
                Gs = np.where(frames_gab[
                    row["start_stim_fr"][0] : row["end_stim_fr"][0]
                    ] == -1)[0] + row["start_stim_fr"][0]
                temp.append(Gs.tolist())
            gab_Gs.append(temp)

        # if not returning by disp
        if by == "block" or by == "frame":
            gab_Gs = [x for sub in gab_Gs for x in sub]
            if by == "frame":
                gab_Gs = [x for sub in gab_Gs for x in sub]
        elif by != "disp":
            raise ValueError("'by' can only take the values 'disp', "
                "'block' or 'frame'.")
        
        return gab_Gs
            

    #############################################    
    def get_gabG_stim_fr(self, by="block"):
        """
        self.get_gabG_stim_fr()

        Returns every first grayscreen stimulus frame number for every 
        grayscreen sequence occuring during a gabor stimulus blocks, and the 
        number of consecutive grayscreen stimulus frames for each sequence. 
                
        NOTE: any grayscreen stimulus frames for sequences flanking gabor 
        stimulus blocks are excluded in the returned list.

        Optional args:
            - by (str): determines whether frame numbers are returned in a 
                        flat list ("frame"), grouped by block ("block"), or 
                        further grouped by display sequence ("disp")
                        default: "block"    

        Returns:
            - first_gabGs_df (pd DataFrame): dataframe containing stimulus 
                                             frame information on each first 
                                             grayscreen (G) segment, within 
                                             gabor stimulus blocks, with 
                columns:
                    - "first_stim_fr": first stimulus frame number for the 
                                       grayscreen (G) segment
                    - "n_stim_fr"    : length of grayscreen (G) segment
                    if by == "block":
                    - "block_n"           : block number during which sequence 
                                            occurs
                    if by == "disp":
                    - "display_sequence_n": display sequence number during 
                                            which sequence occurs
        """

        first_gabGs_df = pd.DataFrame()
        
        gabGs_all = np.asarray(self.get_all_gabG_stim_fr(by="frame"))
        gabors = self.sess.gabors
        first_gabGs_idx = [0] + \
            (np.where(np.diff(gabGs_all) != 1)[0] + 1).tolist() + \
            [len(gabGs_all)]
        
        first_gabGs_df["first_stim_fr"] = gabGs_all[
            np.asarray(first_gabGs_idx)[:-1]]
        
        first_gabGs_df["n_stim_fr"] = np.diff(first_gabGs_idx).tolist()

        if by == "disp":
            col = "display_sequence_n"
            pd_values = [d for d in gabors.block_params.index.unique(col)]
            pd_slices = [pd.IndexSlice[d] for d in pd_values]
        elif by == "block":
            col = "block_n"
            pd_values = [b for b in gabors.block_params.index.unique(col)]
            pd_slices = [pd.IndexSlice[:, b] for b in pd_values]

        if by in ["disp", "block"]:
            for val, pd_slice in zip(pd_values, pd_slices):
                min_stim_fr = gabors.block_params.loc[pd_slice, ][
                    "start_stim_fr", ].min()
                max_stim_fr = gabors.block_params.loc[pd_slice, ][
                    "end_stim_fr", ].max()
                first_gabGs_df.loc[
                    (first_gabGs_df["first_stim_fr"] >= min_stim_fr) &
                    (first_gabGs_df["first_stim_fr"] < max_stim_fr), 
                    col] = val
        elif by != "frame":
            raise ValueError("'by' can only take the values 'disp', "
                             "'block' or 'frame'.")

        first_gabGs_df = first_gabGs_df.astype(int)

        return first_gabGs_df


