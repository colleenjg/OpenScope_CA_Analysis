"""
session.py

Classes to store, extract, and analyze an Allen Institute OpenScope session for
the Credit Assignment Project.

Authors: Colleen Gillon, Blake Richards

Date: August, 2018

Note: this code uses python 3.7.

"""

import glob
import json
import logging
import warnings
from pathlib import Path

import h5py
import itertools
import numpy as np
import pandas as pd
import scipy.signal as scsig

from util import file_util, gen_util, logger_util, math_util
from sess_util import sess_data_util, sess_stim_df_util, sess_file_util, \
    sess_load_util, sess_pupil_util, sess_sync_util, sess_trace_util

logger = logging.getLogger(__name__)

# check pandas version
from packaging import version
PD_MIN_VERSION = "1.1.1"
if not version.parse(pd.__version__) >= version.parse(PD_MIN_VERSION):
    raise OSError(f"Please update pandas package >= {PD_MIN_VERSION}.")

DEFAULT_DATADIR = Path("..", "data", "OSCA")
TAB = "    "

#### ALWAYS SET TO FALSE - CHANGE ONLY FOR TESTING PURPOSES
TEST_USE_PLATEAU = False


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
    
    def __init__(self, datadir=None, sessid=None, mouse_df="mouse_df.csv", 
                 runtype="prod", drop_tol=0.0003, verbose=False, 
                 only_tracked_rois=False, mouse_n=1, sess_n=1):
        """
        self.__init__(datadir, sessid)

        Initializes and returns the new Session object using the specified data 
        directory and ID.

        Calls:
            - self._extract_sess_attribs()
            - self._init_directory()

        Attributes:
            - drop_tol (num)          : dropped frame tolerance 
                                        (proportion of total)
            - home (Path)             : path of the main data directory
            - mouse_df (Path)         : path to dataframe containing 
                                        information on each session.
            - nwb (bool)              : if True, data is in NWB format.
            - only_tracked_rois (bool): if True, only tracked ROIs will be 
                                        loaded
            - runtype (str)           : "prod" (production) or "pilot" data
            - sessid (int)            : session ID (9 digits), e.g. "712483302"
            if sessid is None:
            - mouse_n (int)           : mouse number
            - sess_n (int)            : session number
        
        Required args:
            - datadir (Path)          : full path to the directory where session 
                                        folders are stored. If None, default 
                                        data directory is used.
                                        default: None
            - sessid (int)            : the ID for this session.
                                        default: None
            - mouse_df (Path)         : path to the mouse dataframe
                                        default: "mouse_df.csv"
            - runtype (str)           : the type of run, either "pilot" or 
                                        "prod" (ignored if sessid is provided)
                                        default: "prod"
            - drop_tol (num)          : the tolerance for proportion frames 
                                        dropped (stimulus or running). Warnings 
                                        are produced when this condition 
                                        isn't met.
                                        default: 0.0003 
            - verbose (bool)          : if True, will log instructions on next 
                                        steps to load all necessary data.
                                        default: True
            - only_tracked_rois (bool): if True, only data from ROIs tracked 
                                        across sessions (1-3) are included when 
                                        data is returned.
                                        default: False
            - mouse_n (int)           : mouse number, used only if sessid is 
                                        None.
                                        default: 1
            - sess_n (int)            : session number, used only if sessid is 
                                        None.
                                        default: 1
        """

        if datadir is None:
            datadir = DEFAULT_DATADIR
        self.home     = Path(datadir)

        if not isinstance(mouse_df, pd.DataFrame):
            mouse_df = Path(mouse_df) 
        self.mouse_df = mouse_df
        
        self.sessid   = sessid
        if self.sessid is None:
            self.mouse_n = mouse_n
            self.sess_n = sess_n
            if runtype not in ["pilot", "prod"]:
                gen_util.accepted_values_error(
                    "runtype", runtype, ["pilot", "prod"])
            self.runtype = runtype

        self.drop_tol = drop_tol

        self.nwb

        self._extract_sess_attribs()
        self._init_directory()
        self.set_only_tracked_rois(only_tracked_rois)

        if verbose:
            print("To load stimulus, behaviour and ophys data, "
                "run 'self.extract_info()'")


    #############################################
    def __repr__(self):
        return f"{self.__class__.__name__} ({self.sessid})"

    def __str__(self):
        return repr(self)


    #############################################
    def _extract_sess_attribs(self):
        """
        self._extract_sess_attribs()

        This function loads the dataframe containing information on each 
        session,and sets attributes.

        Attributes:
            if self.nwb:
            - dandi_id (str)   : Dandi archive session ID

            - all_files (bool) : if True, all files have been acquired for
                                 the session
            - any_files (bool) : if True, some files have been acquired for
                                 the session
            - date (int)       : session date (i.e., yyyymmdd)
            - depth (int)      : recording depth 
            - plane (str)      : recording plane ("soma" or "dend")
            - line (str)       : mouse line (e.g., "L5-Rbp4")
            - mouse_n (int)    : mouse number (e.g., 1)
            - mouseid (int)    : mouse ID (6 digits)
            - notes (str)      : notes from the dataframe on the session
            - pass_fail (str)  : whether session passed "P" or failed "F" 
                                 quality control
            - runtype (str)    : "prod" (production) or "pilot" data
            - sess_n (int)     : overall session number (e.g., 1)
            - stim_seed (int)  : random seed used to generated stimulus
        """

        if self.sessid is None:
            self.sessid = sess_load_util.get_sessid_from_mouse_df(
                mouse_n=self.mouse_n, sess_n=self.sess_n, runtype=self.runtype, 
                mouse_df=self.mouse_df
                )

        self.sessid = int(self.sessid)

        df_data = sess_load_util.load_info_from_mouse_df(
            self.sessid, self.mouse_df
            )
        
        if self.nwb:
            self.dandi_id = df_data["dandi_id"]

        self.mouse_n      = df_data["mouse_n"]
        self.date         = df_data["date"]
        self.depth        = df_data["depth"]
        self.plane        = df_data["plane"]
        self.line         = df_data["line"]
        self.mouseid      = df_data["mouseid"]
        self.runtype      = df_data["runtype"]
        self.sess_n       = df_data["sess_n"]
        self.stim_seed    = df_data["stim_seed"]
        self.pass_fail    = df_data["pass_fail"]
        self.all_files    = df_data["all_files"]
        self.any_files    = df_data["any_files"]
        self.notes        = df_data["notes"]


    #############################################
    def _init_directory(self):
        """
        self._init_directory()

        Checks that the session data directory obeys the expected organization
        scheme and sets attributes.

        Attributes:
            if self.nwb:
            - sess_files (list): paths names to session files

            else:
            - align_pkl (Path)         : path name of the stimulus alignment 
                                         pickle file
            - behav_video_h5 (Path)    : path name of the behavior hdf5 file
            - correct_data_h5 (Path)   : path name of the motion corrected 2p 
                                         data hdf5 file
            - dir (Path)               : path of session directory
            - expdir (Path)            : path name of experiment directory
            - expid (int)              : experiment ID (8 digits)
            - max_proj_png (Path)      : full path to max projection of stack
                                          in png format
            - mouse_dir (bool)         : whether path includes a mouse directory
            - procdir (Path)           : path name of the processed data 
                                         directory
            - pup_video_h5 (Path)      : path name of the pupil hdf5 file
            - roi_extract_json (Path)  : path name of the ROI extraction json
            - roi_mask_file (Path)     : path name of the ROI mask file (None, 
                                         as allen masks must be created during 
                                         loading)
            - roi_objectlist_txt (Path): path name of the ROI object list file
            - roi_trace_h5 (Path)      : path name of the ROI raw processed 
                                         fluorescence trace hdf5 file
            - roi_trace_dff_h5 (Path)  : path name of the ROI dF/F trace 
                                         hdf5 file
            - stim_pkl (Path)          : path name of the stimulus pickle file
            - stim_sync_h5 (Path)      : path name of the stimulus 
                                         synchronisation hdf5 file
            - time_sync_h5 (Path)      : path name of the time synchronization 
                                         hdf5 file
            - zstack_h5 (Path)         : path name of the z-stack 2p hdf5 file
        """

        file_util.checkdir(self.home)
        
        if self.nwb:
            # find a directory and potential file names
            self.sess_files = sess_file_util.get_nwb_sess_paths(
                self.home, self.dandi_id, mouseid=self.mouseid
                )        
        
        else:
            # check that the high-level home directory exists
            sessdir, mouse_dir = sess_file_util.get_sess_dir_path(
                self.home, self.sessid, self.runtype)
            self.dir       = sessdir
            self.mouse_dir = mouse_dir

            mouseid, date = sess_file_util.get_mouseid_date(
                self.dir, self.sessid
                )
            if self.mouseid != int(mouseid):
                raise RuntimeError(
                    f"Mouse ID from session directory ({mouseid}) does not "
                    f"match attribute ({self.mouseid})."
                    )
            if self.date != int(date):
                raise RuntimeError(
                    f"Date from session directory ({date}) does not "
                    f"match attribute ({self.date})."
                    )

            self.expid = sess_file_util.get_expid(self.dir)
            self.segid = sess_file_util.get_segid(self.dir)
            dirpaths, filepaths = sess_file_util.get_file_names(
                self.home, self.sessid, self.expid, self.segid, self.date, 
                self.mouseid, self.runtype, self.mouse_dir, check=True)  
        
            self.expdir           = dirpaths["expdir"]
            self.procdir          = dirpaths["procdir"]
            self.max_proj_png     = filepaths["max_proj_png"]
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
    def nwb(self):
        """
        self.nwb()

        Returns:
            - _nwb (bool): whether data is provided in NWB format or in the 
                           internal Allen Institute data structure. 
                           This attribute should NOT be updated after 
                           initialization.
        """

        if not hasattr(self, "_nwb"):
            # check for any nwb files
            path_style = str(Path(self.home, "**" "*.nwb"))
            self._nwb = bool(len(glob.glob(path_style, recursive=True)))

        return self._nwb


    #############################################
    @property
    def max_proj(self):
        """
        self.max_proj()

        Returns:
            - _max_proj (2D array): maximum projection image across downsampled 
                                    z-stack (hei x wei), with pixel intensity 
                                    in 0 (incl) to 256 (excl) range.
        """

        if self.nwb:
            raise NotImplementedError(
                "Retrieving max projection not implemented for NWB data."
                )

        if not hasattr(self, "_max_proj"):
            self._max_proj = sess_load_util.load_max_projection(
                self.max_proj_png
                )

        return self._max_proj


    #############################################
    @property
    def pup_data_h5(self):
        """
        self.pup_data_h5

        Returns:
            - _pup_data_h5 (list or Path): single pupil data file path if one is 
                                           found, a list if several are found 
                                           and "none" if none is found.
        """
        
        if self.nwb:
            raise ValueError(
                "self.pup_data_h5 does not exist if self.nwb is True."
                )

        if not hasattr(self, "_pup_data_h5"):
            self._pup_data_h5 = sess_file_util.get_pupil_data_h5_path(self.dir)

        return self._pup_data_h5


    ############################################
    @property
    def roi_masks(self):
        """
        self.roi_masks()

        Loads boolean ROI masks

        Returns:
            - _roi_masks (3D array): boolean ROI masks, structured as 
                                     ROI x height x width
        """

        if not hasattr(self, "_nrois"):
            raise RuntimeError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        if not hasattr(self, "_roi_masks"):
            if self.nwb:
                self._roi_masks, _ = sess_trace_util.get_roi_masks_nwb(
                    self.sess_files, make_bool=True
                    )
            else:
                mask_threshold = 0.1 # value used in ROI extraction
                min_n_pix = 3 # value used in ROI extraction

                self._roi_masks, _ = sess_trace_util.get_roi_masks(
                    self.roi_mask_file, self.roi_extract_json, 
                    self.roi_objectlist, mask_threshold=mask_threshold, 
                    min_n_pix=min_n_pix, make_bool=True)

        return self._roi_masks


    ############################################
    @property
    def dend(self):
        """
        self.dend()

        Returns:
            - _dend (str): type of dendrites loaded ("allen" or "extr")
        """

        if not hasattr(self, "_nrois"):
            raise RuntimeError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        return self._dend


    #############################################
    @property
    def only_tracked_rois(self):
        """
        self.only_tracked_rois()

        Returns whether session is currently set to use only tracked ROIs.
        """

        if not hasattr(self, "_only_tracked_rois"):
            raise RuntimeError("self._only_tracked_rois does not exist, but "
                "should be set in __init__.")
        
        return self._only_tracked_rois


    #############################################
    @property
    def tracked_rois(self):
        """
        self.tracked_rois()

        Returns as a numpy array the indices of ROIs that have been
        tracked across sessions (currently, across all sessions for
        which we have data).
        """

        if not hasattr(self, '_tracked_rois'):
            self._set_tracked_rois()
        
        return self._tracked_rois


    #############################################
    @property
    def stim2twopfr(self):
        """
        self.stim2twopfr()

        Returns as a numpy array the indices of the two-photon frame numbers 
        for each stimulus number.
        """

        if not hasattr(self, '_stim2twopfr'):
            if self.nwb:
                stim_timestamps, twop_timestamps = \
                    sess_sync_util.get_frame_timestamps_nwb(self.sess_files)
                self._stim2twopfr = gen_util.get_closest_idx(
                    twop_timestamps, stim_timestamps
                    )
            else:
                raise RuntimeError(
                    "self._stim2twopfr should be initialized when stimulus "
                    "dataframe is created."
                    )

        
        return self._stim2twopfr


    #############################################
    @property
    def twop2stimfr(self):
        """
        self.twop2stimfr()

        Returns as a numpy array the indices of the stimulus frame numbers 
        for each two-photon frame number.

        Array contains np.nan for frames that are out of range for the stimulus.
        """

        if not hasattr(self, '_twop2stimfr'):
            if self.nwb:
                stim_timestamps, twop_timestamps = \
                    sess_sync_util.get_frame_timestamps_nwb(self.sess_files)

                self._twop2stimfr = gen_util.get_closest_idx(
                    stim_timestamps, twop_timestamps
                    ).astype(float)
                
                start = int(self.stim2twopfr[0])
                end = int(self.stim2twopfr[-1]) + 1
                self._twop2stimfr[ : start] = np.nan
                self._twop2stimfr[end :] = np.nan

            else:
                if not hasattr(self, "tot_twop_fr"):
                    raise RuntimeError("Run 'self.load_roi_info()' or "
                        "self.load_pup_data() to set two-photon "
                        "attributes correctly.")                    
                self._twop2stimfr = sess_sync_util.get_twop2stimfr(
                    self.stim2twopfr, self.tot_twop_fr, sessid=self.sessid
                    )

        return self._twop2stimfr


    #############################################
    def _load_stim_dict(self, full_table=True):
        """
        self._load_stim_dict()

        Returns stimulus dictionary, if applicable.

        Optional args:
            - full_table (bool): if True, the full stimulus information is 
                                 loaded. Otherwise, exact Gabor orientations 
                                 and visual flow square positions per frame are 
                                 omitted
                                 default: True
        """

        if self.nwb:
            raise RuntimeError(
                "Cannot load stimulus dictionary if data is in NWB format."
                )

        if full_table:
            stim_dict = file_util.loadfile(self.stim_pkl)
        else:
            stim_dict = sess_load_util.load_small_stim_pkl(
                self.stim_pkl, self.runtype
            )

        return stim_dict


    #############################################
    def _load_stim_df(self, full_table=True):
        """
        self._load_stim_df()

        Loads the stimulus dataframe.
        
        Attributes: 
            - stim_fps (num)        : stimulus frames per second
            - twop_fps (num)        : two-photon frames per second
            - tot_stim_fr (num)     : total number of stimulus frames

            - stim_df (pd.DataFrame): stimulus dataframe, with columns 
                                      indicating for each segment:
                stimulus_type            : type of stimulus 
                                           ("gabors", "visflow", "grayscreen")
                unexpected               : whether segment is expected or not
                                           for Gabors, full sequences (A to G) 
                                           are marked as unexpected, if a U 
                                           occurs
                gabor_frame              : Gabors frame 
                                           ("A", "B", "C", "D", "U" or "G")
                gabor_kappa              : Gabors orientation dispersion
                gabor_mean_orientation   : mean Gabor orientation (deg)
                gabor_number             : number of Gabors
                gabor_locations_x        : x locations for each Gabor patch (pix) 
                gabor_locations_y        : y locations for each Gabor patch (pix)
                gabor_sizes              : size of each Gabor patch (pix)
                main_flow_direction      : main visual flow direction
                square_size              : visual flow square sizes (in pix)
                square_number            : number of visual flow squares
                square_proportion_flipped: proportion of visual flow squares 
                                           going in direction opposite to main 
                                           flow
                start_frame_stim         : first stimulus frame number (incl)
                stop_frame_stim          : last stimulus frame number (excl)
                num_frames_stim          : number of stimulus frames
                start_frame_twop         : first twop-photon frame number (incl)
                stop_frame_twop          : last twop-photon frame number (excl)
                num_frames_twop          : number of twop-photon frames
                start_time_sec           : starting time (sec, incl)
                stop_time_sec            : stopping time (sec, excl)
                duration_sec             : duration (sec)

                if full_table:
                gabor_orientations       : orientation of each Gabor patch (deg)
                square_locations_x       : x locations for each square 
                                           (square x stimulus frame) (pix) 
                square_locations_y       : y locations for each square 
                                           (square x stimulus frame) (pix)

                if self.nwb:
                stimulus_template_name   : name of the template image stored

        Optional args:
            - full_table (bool): if True, the full stimulus information is 
                                 loaded. Otherwise, exact Gabor orientations 
                                 and visual flow square positions per frame are 
                                 omitted
                                 default: True
        
        Returns:
            - stim_dict (dict): stimulus dictionary 
                                (None returned if self.nwb if True)
        """

        reloading = False
        if hasattr(self, "stim_df"):
            if full_table == self._full_table:
                return

            if not full_table:
                logger.info(
                    "Dropping full table columns from stimulus dataframe .", 
                    extra={"spacing": TAB})
                
                self.stim_df = self.stim_df.drop(
                    columns=sess_stim_df_util.FULL_TABLE_COLUMNS
                    )
                self._full_table = full_table
                return
            
            reloading = True
            logger.info("Stimulus dataframe being reloaded with "
                f"full table columns.", extra={"spacing": TAB})


        self._full_table = full_table
        if self.nwb:
            self.stim_df = sess_stim_df_util.load_stimulus_table_nwb(
                self.sess_files, full_table=full_table
                )
            stim_dict = None
        
        else:
            stim_dict = self._load_stim_dict(full_table=full_table)

            stim_df, stim2twopfr = sess_stim_df_util.load_stimulus_table(
                stim_dict, self.stim_sync_h5, self.time_sync_h5, 
                self.align_pkl, self.sessid, self.runtype
                )
            
            self.stim_df = stim_df.drop(
                columns=sess_stim_df_util.NWB_ONLY_COLUMNS
                )
            self._stim2twopfr = stim2twopfr

            if not reloading:
                drop_stim_fr = stim_dict["droppedframes"]
                n_drop_stim_fr = len(drop_stim_fr[0])
                sess_sync_util.check_stim_drop_tolerance(
                    n_drop_stim_fr, stim_df["stop_frame_stim"].max(), 
                    self.drop_tol, self.sessid, raise_exc=False)

            
        if not reloading:
            start_row = stim_df.loc[0]
            last_row = stim_df.loc[len(stim_df) - 1]
            num_sec = start_row["start_time_sec"] - last_row["stop_time_sec"]
            num_stim_fr = (
                start_row["start_frame_stim"] - last_row["stop_frame_stim"]
            )
            num_twop_fr = (
                start_row["start_frame_twop"] - last_row["stop_frame_twop"]
                )
            
            self.stim_fps = num_stim_fr / num_sec
            self.twop_fps = num_twop_fr / num_sec
            self.tot_stim_fr = last_row["stop_frame_stim"]

        return stim_dict


    #############################################
    def data_loaded(self):
        """
        self.data_loaded()

        Returns:
            - roi_loaded (bool)  : whether ROI data is loaded
            - run_loaded (bool)  : whether running data is loaded
            - pupil_loaded (bool): whether pupil data is loaded
        """
        
        roi_loaded = hasattr(self, "_dend")
        run_loaded = hasattr(self, "run_data")
        pupil_loaded = hasattr(self, "pup_data")

        return roi_loaded, run_loaded, pupil_loaded


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
            - tot_stim_fr (num): number of stimulus velocity frames

        Optional args:
            - filter_ks (int): kernel size to use in median filtering the 
                               running velocity (0 to skip filtering). 
                               Does not apply to NWB data which is 
                               pre-processed.
                               default: 5
            - diff_thr (int) : threshold of difference in running  
                               velocity to identify outliers.
                               default: 50
            - replace (bool) : if True, running data is recalculated.
                               default: False
        """

        if self.nwb:
            if filter_ks != sess_load_util.NWB_FILTER_KS:
                raise ValueError(
                    f"Cannot use filter_ks={filter_ks}, as running data from "
                    "NWB files is pre-processed with "
                    f"{sess_load_util.NWB_FILTER_KS} filter kernel size."
                    )

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
                modif_str = ("running dataframe using "
                    f"{' and '.join(modifications)}")
                if not replace:
                    warnings.warn("Running dataframe not updated. Must set "
                        f"'replace' to True to update {modif_str}.", 
                        category=RuntimeWarning, stacklevel=1)
                    return

                logger.info(
                    f"Updating {modif_str}.", extra={"spacing": TAB}
                    )
            else:
                return
            
        if self.nwb:
            velocity = sess_load_util.load_run_data_nwb(
                self.sess_files, diff_thr, self.drop_tol, self.sessid
                )
        else:
            velocity = sess_load_util.load_run_data(
                self.stim_pkl, self.stim_sync_h5, filter_ks, diff_thr, 
                self.drop_tol, self.sessid)
        
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
    
        self.tot_stim_fr = len(velocity)

        if self.stim_df["stop_frame_stim"].max() > self.tot_stim_fr:
            raise RuntimeError(
                "Number of stimulus frames in stimulus dataframe "
                "is higher than number of running frames."
                )


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
                    - "datatype"    : type of pupil data: "pup_diam"
                                      if not self.nwb: 
                                          "pup_center_x", "pup_center_y", 
                                          "pup_center_diff"
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
        
        if self.nwb:
            pup_data = sess_load_util.load_pup_data_nwb(self.sess_files)
        else:
            pup_data = sess_load_util.load_pup_data(
                self.pup_data_h5, self.time_sync_h5
                )

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
        n_frames = len(pup_data)

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

        if not hasattr(self, "tot_twop_fr"):
            self.tot_twop_fr = n_frames
        elif n_frames != self.tot_twop_fr:
            raise RuntimeError(
                "Number of pupil frames expected to match number of "
                "two-photon frames."
                )


    #############################################
    def _set_nanrois(self, fluor="dff", roi_traces=None):
        """
        self._set_nanrois()

        Sets attributes with the indices of ROIs containing NaNs or Infs in the
        raw or dff data.

        Attributes:
            if fluor is "dff":
                - _nanrois_dff (list): list of ROIs containing NaNs or Infs in
                                       the ROI dF/F traces
            if fluor is "raw":
                - _nanrois (list)    : list of ROIs containing NaNs or Infs in
                                       the ROI raw processed traces

        Optional args:
            - fluor (str)         : if "dff", a nanrois attribute is added for 
                                    dF/F traces. If "raw, it is created for raw 
                                    processed traces.
                                    default: "dff"
            - roi_trace (2D array): ROI traces (can be passed to avoid 
                                    re-loading them into memory, for example).
                                    Assumes that fluorescence type matches 
                                    'fluor'.
                                    default: None
        """
        
        
        if not hasattr(self, "_nrois"):
            raise RuntimeError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        if self.nwb:
            if fluor != "dff":
                raise ValueError("NWB session files only include dF/F data.")
            self._nanrois_dff = []

        else:
            rem_noisy = True
            if roi_traces is None:
                roi_trace_path = self.get_roi_trace_path(fluor)
                roi_traces = sess_trace_util.load_roi_traces(roi_trace_path)

            else:
                expected_shape = (self._nrois, self.tot_twop_fr)
                if roi_traces.shape != (self._nrois, self.tot_twop_fr):
                    raise RuntimeError(
                        f"Expected roi_traces to have shape {expected_shape}, "
                        f"but found {roi_traces.shape}."
                        )

            nan_arr = (np.isnan(roi_traces).any(axis=1) + 
                np.isinf(roi_traces).any(axis=1))

            if rem_noisy:
                min_roi = np.min(roi_traces, axis=1)

                # suppress a few NaN-related warnings
                msgs = ["Mean of empty slice", "invalid value"]
                categs = [RuntimeWarning, RuntimeWarning]
                with gen_util.TempWarningFilter(msgs, categs):
                    high_med = (
                        ((np.median(roi_traces, axis=1) - min_roi)/
                        (np.max(roi_traces, axis=1) - min_roi)) 
                        > 0.5)            
                    sub0_mean = np.nanmean(roi_traces, axis=1) < 0
                
                roi_ns = np.where(high_med + sub0_mean)[0]

                n_noisy_rois = len(roi_ns)
                if n_noisy_rois != 0:
                    warn_str = ", ".join([str(x) for x in roi_ns])
                    logger.warning(f"Session {self.sessid}: {n_noisy_rois} "
                        "noisy ROIs (mean below 0 or median above midrange) "
                        "are also included in the NaN ROI attributes (but not "
                        f"set to NaN): {warn_str}.", 
                        extra={"spacing": TAB})
                
                nan_arr += high_med + sub0_mean

            nan_rois = np.where(nan_arr)[0].tolist()

            if fluor == "dff":
                self._nanrois_dff = nan_rois
            elif fluor == "raw":
                self._nanrois = nan_rois


    #############################################
    def _set_nanrois_tracked(self):
        """
        self._set_nanrois_tracked()

        Sets attributes with the indices of tracked ROIs containing NaNs or 
        Infs in the raw and/or dff data.

        Attributes:
            if fluor is "dff":
                - _nanrois_dff_tracked (list): 
                                      list of ROIs (indexed for the full ROI 
                                      list) containing NaNs or Infs in the ROI 
                                      dF/F traces
            if fluor is "raw":
                - nanrois_tracked (list): 
                                      list of ROIs (indexed for the full ROI 
                                      list) containing NaNs or Infs in the ROI 
                                      raw processed traces
        """

        if not hasattr(self, "_nanrois_dff_tracked"):
            if hasattr(self, "_nanrois_dff"):
                all_nanrois = self.get_nanrois("dff")
                self._nanrois_dff_tracked = [
                    nanroi for nanroi in all_nanrois 
                    if nanroi in self.tracked_rois
                    ]

        if not hasattr(self, "_nanrois_tracked"):
            if hasattr(self, "_nanrois"):
                all_nanrois = self.get_nanrois("raw")
                self._nanrois_tracked = [
                    nanroi for nanroi in all_nanrois 
                    if nanroi in self.tracked_rois
                    ]
    

    #############################################
    def set_only_tracked_rois(self, only_tracked_rois=True):
        """
        self.set_only_tracked_rois()

        Sets only_tracked_rois attribute, and updates related attributes.

        Attributes:
            - tracked_rois (1D array): ordered indices of ROIs tracked across 
                                       sessions
                                       default: True
        """

        self._only_tracked_rois = bool(only_tracked_rois)
        
        if not hasattr(self, "_nrois"): # ROIs not yet loaded, anyway
            return

        if self.only_tracked_rois:
            self._set_tracked_rois()


    #############################################
    def _set_tracked_rois(self):
        """
        self._set_tracked_rois()

        Sets attribute with the indices of ROIs that have been tracked across 
        sessions.

        Attributes:
            - tracked_rois (1D array): ordered indices of ROIs tracked across 
                                       sessions
        """

        if self.nwb:
            raise NotImplementedError("Not implemented for NWB!")

        if self.plane == "dend" and self.dend != "extr":
            raise UserWarning("ROIs not tracked for Allen extracted dendritic "
                "ROIs.")

        if hasattr(self, "_tracked_rois"):
            self._set_nanrois_tracked()
            return

        try:
            nway_match_path = sess_file_util.get_nway_match_path_from_sessid(
                self.home, self.sessid, self.runtype, check=True)
        except Exception as err:
            if "not exist" in str(err):
                raise UserWarning(f"No tracked ROIs file found for {self}.")
            else:
                raise err

        with open(nway_match_path, 'r') as fp:
            tracked_rois_df = pd.DataFrame(json.load(fp)['rois'])

        self._tracked_rois = tracked_rois_df['dff-ordered_roi_index'].values
        self._set_nanrois_tracked()


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
                [[f"sub_{sub}", f"div_{div}"], range(self._nrois)], 
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
            if self.only_tracked_rois:
            - self._set_tracked_rois()
        """

        if not hasattr(self, "_nrois"):
            raise RuntimeError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        if hasattr(self, "roi_facts_df"):
            if fluor in self.roi_facts_df.columns.get_level_values(
                level="fluorescence"):
                return
        
        self._init_roi_facts_df(fluor=fluor)

        if self.nwb:
            if fluor != "dff":
                raise ValueError("NWB session files only include dF/F data.")
            roi_traces = sess_trace_util.load_roi_traces_nwb(self.sess_files)

        else:
            roi_trace_path = self.get_roi_trace_path(fluor)
            roi_traces = sess_trace_util.load_roi_traces(roi_trace_path)

        
        # obtain scaling facts while filtering All-NaN warning.
        with gen_util.TempWarningFilter("All-NaN", RuntimeWarning):

            self.roi_facts_df[("roi_traces", fluor)] = np.asarray(
                math_util.scale_facts(roi_traces, axis=1, 
                sc_type="stand_rob", extrem="perc", nanpol="omit",
                allow_0=True
                )[0:2]).reshape(-1)

        self._set_nanrois(fluor, roi_traces=roi_traces) # avoid re-loading
        if self.only_tracked_rois:
            self._set_tracked_rois()


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

        Attributes:
            - _dend (str)            : type of dendrites loaded 
                                       ("allen" or "extr")
            if not self.nwb and EXTRACT dendrites are used, updates:
            - roi_mask_file (Path)   : path to ROI mask h5
            - roi_trace_h5 (Path)    : full path name of the ROI raw 
                                       processed fluorescence trace hdf5 file
            - roi_trace_dff_h5 (Path): full path name of the ROI dF/F
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
                raise NotImplementedError(
                    "Cannot change dendrite type. "
                    f"Already set to {self.dend} traces."
                    )
            return

        if dend not in ["extr", "allen"]:
            gen_util.accepted_values_error("dend", dend, ["extr", "allen"])

        if self.nwb:
            if dend != "extr":
                raise ValueError(
                    "NWB session files include only 'extr' dendrites."
                    )
            if fluor != "dff":
                raise ValueError("NWB session files only include dF/F data.")
            self._dend = dend

        else:
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
                        "will be used instead.", category=UserWarning, 
                        stacklevel=1)


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
            - nrois (int)      : number of ROIs in traces
            - roi_names (list) : list of ROI names (9 digits)
            - tot_twop_fr (int): total number of two-photon frames

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
            if self.nwb:
                roi_ids, nrois, tot_twop_fr = \
                    sess_trace_util.load_roi_data_nwb(self.sess_files)

            else:
                roi_trace_path = self.get_roi_trace_path(fluor)
                roi_ids, nrois, tot_twop_fr = \
                    sess_trace_util.load_roi_data(roi_trace_path)

            self.roi_names = roi_ids
            self._nrois = nrois
            self.tot_twop_fr = tot_twop_fr

            if self.stim_df["stop_frame_twop"].max() > self.tot_twop_fr:
                raise RuntimeError(
                    "Number of two-photon frames in stimulus dataframe "
                    "is higher than number of recorded frames."
                    )
        
        self._set_roi_attributes(fluor)


    #############################################
    def _load_stims(self, full_table=True):
        """
        self._load_stims()
        
        Initializes attributes, including Stim objects 
        (Gabors, Visflow, Grayscr)

        Attributes:
            - visflow (Visflow object): session visflow object
            - gabors (Gabors object)  : session gabors object
            - grayscr (Grayscr object): session grayscreen object
            - n_stims (int)           : number of stimulus objects in
                                        the session (2 visflow stims
                                        in production data count as one)
            - stimtypes (list)        : list of stimulus type names 
                                        (i.e., "gabors", "visflow")
            - stims (list)            : list of stimulus objects in the
                                        session
        """

        stim_dict = self._load_stim_df(full_table=full_table)

        if hasattr(self, "stimtypes"):
            return

        # create the stimulus fields and objects
        self.stimtypes = []
        self.stims = []

        for stimtype in self.stim_df["stimulus_type"].unique():
            if stimtype == "grayscreen":
                continue
            elif stimtype == "gabors":
                self.stimtypes.append(stimtype)
                self.stims.append(Gabors(self, stim_dict))
                self.gabors = self.stims[-1]
            elif stimtype == "visflow":
                self.stimtypes.append(stimtype)
                self.stims.append(Visflow(self, stim_dict))
                self.visflow = self.stims[-1]
            else:
                logger.info(f"{stimtype} stimulus type not recognized. No Stim " 
                    "object created for this stimulus. \n", 
                    extra={"spacing": TAB})

        self.n_stims = len(self.stims)

        # initialize a Grayscr object
        self.grayscr = Grayscr(self)


    #############################################
    def extract_info(self, full_table=True, fluor="dff", dend="extr", roi=True, 
                     run=False, pupil=False):
        """
        self.extract_info()

        This function should be run immediately after creating a Session 
        object. It creates the stimulus objects attached to the Session, and 
        loads the stimulus dataframe, ROI traces, running data, etc. If 
        stimtypes have not been initialized, also initializes stimtypes.

        Calls:
            self._load_stims()

            optionally:
            self.load_roi_info()
            self.load_run_data()
            self.load_pup_data()

        Optional args:
            - full_table (bool): if True, the full stimulus information is 
                                 loaded. Otherwise, exact Gabor orientations 
                                 and visual flow square positions per frame are 
                                 omitted
                                 default: True
            - fluor (str)      : if "dff", ROI information is loaded from dF/F 
                                 trace file. If "raw", based on the raw processed 
                                 trace file. 
                                 default: "dff"
            - dend (str)       : dendritic traces to use ("allen" for the 
                                 original extracted traces and "extr" for the
                                 ones extracted with Hakan's EXTRACT code, if
                                 available). Can only be set the first time 
                                 ROIs are loaded to the session. 
                                 default: "extr"
            - roi (bool)       : if True, ROI data is loaded
                                 default: True
            - run (bool)       : if True, running data is loaded
                                 default: False
            - pup (bool)       : if True, pupil data is loaded
                                 default: False
        """

        # load the stimulus dataframe         
        logger.info("Loading stimulus and alignment info...")
        self._load_stims(full_table=full_table)
   
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
            - stimtype (str ): stimulus type to return ("visflow", "gabors" or 
                               "grayscr")
                               default: "gabors"

        Return:
            - stim (Stim): Stim object (either Gabors or Visflow)
        """


        if stimtype == "gabors":
            if hasattr(self, "gabors"):
                stim = self.gabors
            else:
                raise RuntimeError("Session object has no gabors stimulus.")
        elif stimtype == "visflow":
            if hasattr(self, "visflow"):
                stim = self.visflow
            else:
                raise RuntimeError("Session object has no visual flow stimulus.")
        elif stimtype == "grayscr":
            if hasattr(self, "grayscr"):
                stim = self.grayscr
            else:
                raise RuntimeError("Session object has no grayscr stimulus.")
        else:
            gen_util.accepted_values_error("stimtype", stimtype, 
                ["gabors", "visflow", "grayscr"])
        
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
            raise RuntimeError("Run 'self.load_run_data()' to load the running "
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
    def convert_frames(self, fr, src_fr_type="stim", targ_fr_type="twop", 
                       raise_nans=True):
        """
        self.convert_frames(fr)

        Returns frames converted from stimulus to two-photon or vice versa.

        Required args:
            - fr (1D array): frame numbers
        
        Optional args:
            - src_fr_type (str) : source frame type
                                  default: "stim"
            - targ_fr_type (str): target frame type
                                  default: "twop"
            - raise_nans (bool) : if True, NaNs in the converted frames are 
                                  raised. Otherwise, they are removed.
                                  default: True
        
        Returns:
            - targ_fr (1D array): converted frame numbers
        """

        fr = np.asarray(fr)

        if src_fr_type == targ_fr_type:
            return fr

        fr_types = ["stim", "twop"]
        if src_fr_type not in fr_types:
            gen_util.accepted_values_error("src_fr_type", src_fr_type, fr_types)
        if targ_fr_type not in fr_types:
            gen_util.accepted_values_error(
                "targ_fr_type", targ_fr_type, fr_types
                )
        
        if not hasattr(self, "tot_twop_fr") :
            raise RuntimeError(
                "Must load ROI info or pupil data to set two-photon "
                "attributes correctly."
                )

        if fr.min() < 0: 
            raise ValueError("Frames cannot be < 0.")

        max_fr = self.tot_twop_fr if src_fr_type == "twop" else self.tot_stim_fr
        if fr.max() > max_fr - 1:
            raise ValueError(
                f"Some frames are beyond the maximum number of {src_fr_type} "
                f"frames: {max_fr}."
                )

        # time stamps for stimulus frames
        if src_fr_type == "stim" and targ_fr_type == "twop":
            targ_fr = self.stim2twopfr[fr.astype(int)]
        if src_fr_type == "twop" and targ_fr_type == "stim":
            targ_fr = self.twop2stimfr[fr.astype(int)]

        not_nan_idxs = np.where(np.isnan(targ_fr))[0]
        if len(not_nan_idxs) != len(targ_fr):
            if raise_nans:
                raise RuntimeError("Some frames are out of range.")
            else:
                targ_fr = targ_fr[not_nan_idxs]

        return targ_fr                


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
            raise RuntimeError("Run 'self.load_run_data()' to load the running "
                "data correctly.")

        fr = np.asarray(fr)

        if fr_type == "twop":
            fr = self.convert_frames(
                fr, src_fr_type="twop", targ_fr_type="stim", raise_nans=True
                )
        elif fr_type != "stim":
            gen_util.accepted_values_error("fr_type", fr_type, "stim", "twop")

        if (fr >= self.tot_stim_fr).any() or (fr < 0).any():
            raise UserWarning("Some of the specified frames are out of range")
        
        run_data = self.get_run_velocity(remnans=remnans, scale=scale)

        velocity = run_data.to_numpy()[fr]

        index = pd.MultiIndex.from_product(
            [range(velocity.shape[0]), range(velocity.shape[1])], 
            names=["sequences", "frames"])

        run_data_df = pd.DataFrame(
            velocity.reshape(-1), columns=run_data.columns, index=index)

        return run_data_df


    #############################################
    def get_roi_trace_path(self, fluor="dff", check_exists=True):
        """
        self.get_roi_trace_path()

        Returns correct ROI trace path.

        Optional args:
            - fluor (str)        : if "dff", remnans is assessed on ROIs using 
                                   dF/F traces. If "raw", on raw processed 
                                   traces.
                                   default: "dff"
            - check_exists (bool): if True, checks whether file exists before 
                                   returning the path, and raises an error if 
                                   it doesn't.
                                   default: True
        Returns:
            - roi_trace_path (Path): indices of ROIs containing NaNs or Infs 
                       (indexed into full ROI array, even if 
                       self.only_tracked_rois)
        """

        if self.nwb:
            raise RuntimeError(
                "get_roi_trace_path() is not applicable for NWB data."
                )

        if fluor == "dff":
            roi_trace_path = self.roi_trace_dff_h5
        elif fluor == "raw":
            roi_trace_path = self.roi_trace_h5
        else:
            gen_util.accepted_values_error("fluor", fluor, ["raw", "dff"])

        if check_exists and not Path(roi_trace_path).is_file():
            raise OSError(f"No {fluor} traces found under {roi_trace_path}.")

        return roi_trace_path


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
            - nanrois (1D array): indices of ROIs containing NaNs or Infs 
                                  (indexed into full ROI array, even if 
                                  self.only_tracked_rois)
        """

        if fluor == "dff":
            if not hasattr(self, "_nanrois_dff"):
                self._set_nanrois(fluor)
            if self.only_tracked_rois and hasattr(self, "_nanrois_dff_tracked"):
                self._set_nanrois_tracked()
                nanrois = self._nanrois_dff_tracked
            else:
                nanrois = self._nanrois_dff
        elif fluor == "raw":
            if not hasattr(self, "_nanrois"):
                self._set_nanrois(fluor)
            if self.only_tracked_rois and hasattr(self, "_nanrois_tracked"):
                self._set_nanrois_tracked()
                nanrois = self._nanrois_tracked
            else:
                nanrois = self._nanrois
        else:
            gen_util.accepted_values_error("fluor", fluor, ["raw", "dff"])

        return nanrois


    #############################################
    def get_roi_masks(self, fluor="dff", remnans=True):
        """
        self.get_roi_masks()

        Returns ROI masks, optionally removing those that contain NaNs or Infs.

        Optional args:
            - fluor (str)   : if "dff", remnans is assessed on ROIs using dF/F 
                              traces. If "raw", on raw processed traces.
                              default: "dff"
            - remnans (bool): if True, ROIs containing NaNs/Infs are removed.
                              default: "dff"
        Returns:
            - roi_masks (3D array): boolean ROI masks, restricted to tracked 
                                    ROIs if self.only_tracked_rois, and 
                                    structured as 
                                    ROI x height x width
        """

        roi_masks = self.roi_masks

        if self.only_tracked_rois:
            roi_masks = roi_masks[self.tracked_rois]
            if remnans and len(self.get_nanrois(fluor)):
                raise NotImplementedError(
                    "remnans not implemented for tracked ROIs."
                    )

        elif remnans:
            rem_idx = self.get_nanrois(fluor)
            roi_masks = np.delete(roi_masks, rem_idx, axis=0)

        return roi_masks


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

        if not hasattr(self, "_nrois"):
            raise RuntimeError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        if self.only_tracked_rois:
            nrois = len(self.tracked_rois)
        else:
            nrois = self._nrois

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
                              transients ("visflow", "gabors" or None). 
                              If None, the entire session is checked.
                              default: None
            - remnans (bool): if True, the indices ignore ROIs containg NaNs or 
                              Infs
                              default: True
        Returns:
            - active_roi_indices (list): indices of active ROIs 
                                         (indexed into the full ROI array)
        """

        if not hasattr(self, "_nrois"):
            raise RuntimeError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        logger.info("Identifying active ROIs...", extra={"spacing": "\n"})

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
                twop_fr.extend(
                    [row["start_frame_twop"][0], row["stop_frame_twop"][0]
                    ])
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

        if not hasattr(self, "_nrois"):
            raise RuntimeError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        if replace or not hasattr(self, "plateau_traces"):

            calc_str = "Calculating"
            if hasattr(self, "plateau_traces"):
                calc_str = "Recalculating"
            logger.info(f"{calc_str} plateau traces.", extra={"spacing": "\n"})

            plateau_traces = gen_util.reshape_df_data(
                self.get_roi_traces(None, fluor, remnans), squeeze_cols=True)
            med = np.nanmedian(plateau_traces, axis=1)
            std = np.nanstd(plateau_traces, axis=1)

            for r, roi_data in enumerate(plateau_traces):
                roi_bool = ((roi_data - med[r])/std[r] >= thr_ratio)
                idx = np.where(roi_bool)[0]
                each_start_idx = np.where(np.insert(np.diff(idx), 0, 100) > 1)[0]
                drop_break_pts = np.where(np.diff(each_start_idx) < n_consec)[0]
                for d in drop_break_pts: 
                    set_zero_indices = np.arange(
                        idx[each_start_idx[d]], 
                        idx[each_start_idx[d + 1] - 1] + 1)
                    roi_bool[set_zero_indices] = False
                plateau_traces[r, ~roi_bool] = 1.0
                plateau_traces[r, roi_bool] = \
                    (plateau_traces[r, roi_bool] - med[r])/std[r]

            self.plateau_traces = plateau_traces

        return self.plateau_traces


    #############################################
    def get_single_roi_trace(self, n, fluor="dff"):
        """
        self.get_single_roi_trace(n)

        Returns a single ROI trace, indexed by n (must index the original 
        array). self.only_tracked_rois is ignored here.

        Required args:
            - n (int): ROI index

        Optional args:
            - fluor (str): if "dff", then dF/F traces are returned, if 
                           "raw", raw processed traces are returned
                           default: "dff"

        Returns:
            - trace (1D array): full ROI trace
        """

        if not hasattr(self, "_nrois"):
            raise RuntimeError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        if n >= self._nrois:
            raise ValueError(f"ROI {n} does not exist.")

        n = int(n)

        # read the data points into the return array
        if self.nwb:
            trace = sess_trace_util.load_roi_traces_nwb(self.sess_files, n=n)
        else:
            roi_trace_path = self.get_roi_trace_path(fluor)
            trace = sess_trace_util.load_roi_traces(roi_trace_path, roi_ns=n)

        return trace


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

        if not hasattr(self, "_nrois"):
            raise RuntimeError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        # check whether the frames to retrieve are within range
        if frames is None:
            frames = frames
        elif max(frames) >= self.tot_twop_fr or min(frames) < 0:
            raise UserWarning("Some of the specified frames are out of range")
        else:
            frames = np.asarray(frames)

        remnans_str = "yes" if remnans else "no"
        scale_str = "yes" if scale else "no"

        roi_ids = None
        if self.only_tracked_rois:
            roi_ids = self.tracked_rois

        if remnans:
            nanrois = self.get_nanrois(fluor)
            if len(nanrois):
                if self.only_tracked_rois:
                    raise ValueError("remnans not implemented for tracked ROIs.")
                roi_ids = np.delete(np.arange(self._nrois), nanrois)

        if self.nwb:
            if fluor != "dff":
                raise ValueError("NWB session files only include dF/F data.")
            traces = sess_trace_util.load_roi_traces_nwb(
                self.sess_files, roi_ns=roi_ids, frame_ns=frames
                )

        else:
            roi_trace_path = self.get_roi_trace_path(fluor)
            traces = sess_trace_util.load_roi_traces(
                roi_trace_path, roi_ns=roi_ids, frame_ns=frames
            )

        if roi_ids is None:
            roi_ids = np.arange(self._nrois)
        if frames is None:
            frames = np.arange(self.tot_twop_fr)

        if scale:
            factors = self._get_roi_facts(fluor)

            factor_names = factors.index.unique(level="factors")
            sub_names =  list(filter(lambda x: "sub" in x, factor_names))
            if len(sub_names) != 1:
                raise RuntimeError("Only one factor should contain 'sub'.")
            div_names =  list(filter(lambda x: "div" in x, factor_names))
            if len(div_names) != 1:
                raise RuntimeError("Only one row should contain 'div'.")

            traces = (
                (traces - factors.loc[sub_names[0]].loc[roi_ids].values) /
                factors.loc[div_names[0]].loc[roi_ids].values
                )

        # initialize the return dataframe
        index_cols = pd.MultiIndex.from_product(
            [["roi_traces"], [remnans_str], [scale_str], [fluor]], 
            names=["datatype", "nan_rois_removed", "scaled", 
            "fluorescence"])
        index_rows = pd.MultiIndex.from_product(
            [roi_ids, *[range(dim) for dim in frames.shape]], 
            names=["ROIs", "frames"])
        
        roi_data_df = pd.DataFrame(
            traces.reshape(-1), index=index_rows, columns=index_cols)

        return roi_data_df


    #############################################
    def get_fr_ran(self, ref_fr, pre, post, pad=(0, 0), fr_type="twop"):
        """
        self.get_fr_ran(ref_fr, pre, post)
        
        Returns an array of frame numbers, where each row is a sequence and
        each sequence ranges from pre to post around the specified reference 
        frame numbers. 

        Required args:
            - ref_fr (list): 1D list of frame numbers 
                             (e.g., all 1st seg frames)
            - pre (num)    : range of frames to include before each 
                             reference frame number (in s)
            - post (num)   : range of frames to include after each 
                             reference frame number (in s)
        
        Optional args:
            - pad (tuple)   : number of frame to use as padding (before, after)
                              default: (0, 0)
            - fr_type (str) : time of frames ("twop", "stim")
                              default: "twop"
        Returns:
            - frame_n_df (pd DataFrame): Dataframe of frame numbers, 
                                         organized with:
                columns: 
                    - twop_fr_n: 2-photon frame numbers
                    or
                    - stim_fr_n: stimulus frame numbers
                hierarchical rows:
                    - sequences  : sequence number
                    - time_values: time values for each frame
        """

        ran_fr, xran = self.get_frames_timestamps(pre, post, fr_type=fr_type)

        # adjust for padding
        if len(pad) != 2:
            raise ValueError("Padding must have length 2.")
        if min(pad) < 0:
            raise ValueError("Padding cannot be negative")
        if pad != (0, 0):
            if sum(pad) > len(xran) / 10.:
                warnings.warn("Proportionally high padding values may distort "
                    "time values as method is designed to preserve 'pre' and "
                    "'post' values in time stamps.", category=RuntimeWarning, 
                    stacklevel=1)
            pad = [int(val) for val in pad]
            ran_fr = [ran_fr[0] - pad[0], ran_fr[1] + pad[1]]
            diff = np.diff(xran)[0]
            pre, post = [pre + diff * pad[0], post + diff * pad[1]] 
            xran = np.linspace(-pre, post, int(np.diff(ran_fr)[0]))

        ref_fr = np.asarray(ref_fr)
        if len(ref_fr) == 0:
            raise RuntimeError("No frames: frames list must include at "
                "least 1 frame.")

        if isinstance(ref_fr[0], (list, np.ndarray)):
            raise ValueError("Frames must be passed as a 1D list, not by "
                "block.")

        # get sequences x frames
        fr_idx = gen_util.num_ranges(ref_fr, pre=-ran_fr[0], leng=len(xran))
                     
        # remove sequences with negatives or values above total number of stim 
        # frames
        max_fr = self.tot_twop_fr if fr_type == "twop" else self.tot_stim_fr
        neg_idx  = np.where(fr_idx[:,0] < 0)[0].tolist()
        over_idx = np.where(fr_idx[:,-1] >= max_fr)[0].tolist()
        
        num_ran = gen_util.remove_idx(fr_idx, neg_idx + over_idx, axis=0)

        if len(num_ran) == 0:
            raise RuntimeError("No frames: All frames were removed from list.")

        row_index = pd.MultiIndex.from_product([range(num_ran.shape[0]), xran], 
            names=["sequences", "time_values"])

        frame_n_df = pd.DataFrame(
            num_ran.reshape(-1), index=row_index, columns=[f"{fr_type}_fr_n"])

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

        # for plateau test analyses
        if TEST_USE_PLATEAU:
            logger.warning(
                "Setting `use_plateau` to True for testing purposes.")
            use_plateau = True

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
            raise NotImplementedError("Negative padding not supported.")

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
            logger.warning("Some of the specified frames for sequences "
                f"{seq_rem} are out of range so the sequence will not be "
                "included.", extra={"spacing": "\n"})
            pad_seql     = np.delete(pad_seql, seq_rem)
            twop_fr_seqs = np.delete(twop_fr_seqs, seq_rem).tolist()

        # sanity check that the list is as long as expected
        if last_idx != len(frames_flat):
            if last_idx != len(frames_flat) - sum(seq_rem_l):
                raise RuntimeError(f"Concatenated frame array is {last_idx} "
                    "long instead of expected "
                    f"{len(frames_flat - sum(seq_rem_l))}.")
            else:
                frames_flat = frames_flat[: last_idx]

        if use_plateau:
            traces_flat = self.get_plateau_roi_traces(
                fluor=fluor, remnans=remnans
                )[:, frames_flat.astype(int)].reshape(-1, 1)
        else:
            traces_flat = self.get_roi_traces(
                frames_flat.astype(int), fluor, remnans, scale=scale)

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
            - fr_type (str) : time of frames ("twop", "stim")
                              default: "twop"
            - ret_idx (bool): if True, indices of frames retained are also 
                              returned
                              default: False
        Returns:
            - frs (1D array): list of frames values within bounds
            if ret_idx:
            - all_idx (list): list of original indices retained
        """

        if not hasattr(self, "tot_twop_fr"):
            raise RuntimeError(
                "Run 'self.load_roi_info()' or 'self.load_pup_data()' to load "
                "two-photon attributes correctly.")

        if not isinstance(ch_fl, list) or len(ch_fl) != 2:
            raise ValueError("'ch_fl' must be a list of length 2.")

        if fr_type == "twop":
            fps = self.twop_fps
            max_val = self.tot_twop_fr        
        elif fr_type == "stim":
            fps = self.stim_fps
            max_val = self.tot_stim_fr
        else:
            gen_util.accepted_values_error(
                "fr_type", fr_type, ["twop", "stim"])

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
#############################################
class Stim(object):
    """
    The Stim object is a higher level class for describing stimulus properties.
    For production data, both visual flow stimuli are initialized as one 
    stimulus object.

    It should be not be initialized on its own, but via a subclass in which
    stimulus specific information is initialized.
    """

    def __init__(self, sess, stimtype, stim_dict=None):
        """
        self.__init__(sess, stimtype)

        Initializes and returns a stimulus object, and sets attributes. 
        
        USE: Only initialize subclasses of Stim, not the stim class itself.

        Calls:
            - self._set_block_params()

        Attributes:
            - deg_per_pix (float)  : deg / pixel conversion used to generate 
                                     stimuli
            - exp_max_s (int)      : max duration of an expected seq (sec)
            - exp_min_s (int)      : min duration of an expected seq (sec)
            - seg_len_s (sec)      : length of each segment (sec)
                                     (1 sec for visual flow, 
                                     0.3 sec for gabors)
            - sess (Session object): session to which the stimulus belongs
            - stim_fps (int)       : fps of the stimulus
            - stimtype (str)       : "gabors" or "visflow"
            - unexp_max_s (int)    : max duration of an unexpected seq (sec)
            - unexp_min_s (int)    : min duration of an unexpected seq (sec)
            - win_size (list)      : window size (in pixels) [wid, hei]

            if stimtype == "gabors":
            - n_segs_per_seq (int) : number of segments per set (5)
            
        Required args:
            - sess (Session object): session to which the stimulus belongs
            - stimtype (str)       : type of stimulus ("gabors" or "visflow")  

        Optional args:
            - stim_dict (dict): stimulus dictionary 
                                (only applicable if self.sess.nwb is False, in 
                                which case it will be loaded if not passed)
                                default: None
        """

        self.sess      = sess
        self.stimtype  = stimtype
        self.stim_fps  = self.sess.stim_fps

        if self.sess.nwb:
            if self.stimtype == "gabors":
                # segment length (sec) (0.3 sec)
                self.seg_len_s = 0.3 
                # num seg per set (4: A, B, C, D/U, G)
                self.n_segs_per_seq = 5 

                unexp_len = [3, 6]
                exp_len = [30, 90]
            
            elif self.stimtype == "visflow":
                # segment length (sec) (1 sec)
                self.seg_len_s = 1

                unexp_len = [2, 4]
                exp_len = [30, 90]

            else:
                raise ValueError(f"{self.stimtype} stim type not recognized. "
                    "Stim object cannot be initialized.")
        
            # sequence parameters
            self.unexp_min_s = unexp_len[0]
            self.unexp_max_s = unexp_len[1]
            self.exp_min_s   = exp_len[0]
            self.exp_max_s   = exp_len[1]
        else:
            if stim_dict is None:
                stim_dict = self.sess._load_stim_dict(full_table=False)
            
            gen_stim_props = sess_stim_df_util.load_gen_stim_properties(
                stim_dict, stimtype=stimtype, runtype=self.sess.runtype
                )
            self.seg_len_s = gen_stim_props["seg_len_s"]
            if stimtype == "gabors":            
                self.n_segs_per_seq = gen_stim_props["n_segs_per_seq"]
            
            # sequence parameters
            self.unexp_min_s = gen_stim_props["unexp_len_s"][0]
            self.unexp_max_s = gen_stim_props["unexp_len_s"][1]
            self.exp_min_s   = gen_stim_props["exp_len_s"][0]
            self.exp_max_s   = gen_stim_props["exp_len_s"][1]

            self.win_size    = gen_stim_props["win_size"]
            self.deg_per_pix = gen_stim_props["deg_per_pix"]

        self._set_block_params()


    #############################################
    def __repr__(self):
        return (f"{self.__class__.__name__} (stimulus from "
            f"session {self.sess.sessid})")

    def __str__(self):
        return repr(self)

    #############################################
    def _set_block_params(self):
        """
        self._set_block_params

        Set attributes related to blocks.

        NOTE: A block, here, is a sequence of stimulus presentations of the 
        same stimulus type, and there can be multiple blocks in one experiment. 

        They are almost always separated by a grayscreen stimulus.
        
        For Gabors, segments refer to the stim_df index, where each gabor frame 
        (lasting 0.3 s) and each visual flow segment (lasting 1s) occupies a 
        separate row.

        Attributes:
            - block_params (pd DataFrame): dataframe containing stimulus 
                                           parameters and start (incl), 
                                           stop (excl) info for each block
        """

        if self.stimtype == "gabors":
            stimulus_type = "gabors"
            stim_columns = ["gabor_kappa"]
        elif self.stimtype == "visflow":
            stimulus_type = "visflow"
            stim_columns = \
                ["main_flow_direction", "square_size", "square_number"]

        segments = self.sess.stim_df.loc[
            self.sess.stim_df["stimulus_type"] == stimulus_type
            ].index.to_numpy()

        start_idxs = np.where(np.diff(np.insert(segments, 0, -2)) != 1)[0]
        stop_idxs_incl = np.append(start_idxs[1:], len(segments)) - 1

        self.block_params = pd.DataFrame()
        for start_idx, stop_idx_incl in zip(start_idxs, stop_idxs_incl):
            row_idx = len(self.block_params)
            start_seg = segments[start_idx]
            stop_seg = segments[stop_idx_incl] + 1

            self.block_params.loc[row_idx, "start_seg"] = start_seg
            self.block_params.loc[row_idx, "stop_seg"] = stop_seg
            self.block_params.loc[row_idx, "num_segs"] = stop_seg - start_seg

            start_row = self.sess.stim_df.loc[start_seg]
            stop_row_incl = self.sess.stim_df.loc[stop_seg - 1]                

            for datatype in ["time_sec", "frame_stim", "frame_twop"]:
                start_val = start_row[f"start_{datatype}"]
                stop_val = stop_row_incl[f"stop_{datatype}"]
                diff_val = stop_val - start_val

                self.block_params.loc[row_idx, f"start_{datatype}"] = \
                    start_val
                self.block_params.loc[row_idx, f"stop_{datatype}"] = \
                    stop_val
                
                if datatype == "time_sec":
                    column = "duration_sec"
                else:
                    column = f"num_{datatype}".replace("frame", "frames")
                    
                self.block_params.loc[row_idx, column] = diff_val
            
            for stim_column in stim_columns:
                self.block_params.loc[row_idx, stim_column] = \
                    start_row[stim_column]

        for col in self.block_params.columns:
            if "_sec" not in col and "direction" not in col: 
                self.block_params[col] = self.block_params[col].astype(int)


    #############################################
    def get_stim_beh_sub_df(self, pre, post, stats="mean", fluor="dff", 
                            remnans=True, gabfr="any", gabk="any", 
                            gab_ori="any", visflow_size="any", 
                            visflow_dir="any", pupil=False, run=False, 
                            scale=False):
        """
        self.get_stim_beh_sub_df(pre, post)

        Returns a stimulus and behaviour dataframe for the specific stimulus 
        (gabors or visflow) with plane, line and sessid added in.

        Required args:
            - pre (num) : range of frames to include before each reference 
                          frame number (in s)
            - post (num): range of frames to include after each reference 
                          frame number (in s)

        Optional args:
            - fluor (str)               : if "dff", dF/F is used, if "raw", ROI 
                                          traces
                                          default: "dff"
            - stats (str)               : statistic to use for baseline, mean 
                                          ("mean") or median ("median") (NaN 
                                          values are omitted)
                                          default: "mean"
            - remnans (bool)            : if True, NaN values are removed from 
                                          data, either through interpolation 
                                          for pupil and running data or ROI 
                                          exclusion for ROIdata
                                          default: True
            - gabfr (int or list)       : 0, 1, 2, 3, "G", "any"
                                          default: "any"
            - gabk (int or list)        : 4, 16, or "any"
                                          default: "any"
            - gab_ori (int or list)     : 0, 45, 90, 135, 180, 225, or "any"
                                          default: "any"
            - visflow_size (int or list): 128, 256, or "any"
                                          default: "any"
            - visflow_dir (str or list) : "right", "left", "temp", "nasal" or 
                                          "any"
                                          default: "any"
            - pupil (bool)              : if True, pupil data is added in
                                          default: False
            - run (bool)                : if True, run data is added in
                                          default: False
            - scale (bool)              : if True, data is scaled
                                          default: False

        Returns:
            - sub_df (pd DataFrame): extended stimulus dataframe containing
                                     behaviour, and line/plane/sessid info, if 
                                     requested
        """

        drop = ["start_frame_twop", "stop_frame_twop", 
            "start_frame_stim", "end_stim_fr"] # drop at end
    
        sub_df = self.get_stim_df_by_criteria(
            gabfr=gabfr, gabk=gabk, gab_ori=gab_ori, 
            visflow_size=visflow_size, visflow_dir=visflow_dir
            ).copy().reset_index(drop=True)
        
        sub_df["plane"] = self.sess.plane
        sub_df["line"]  = self.sess.line
        sub_df["sessid"] = self.sess.sessid

        twop_fr = sub_df["start_frame_twop"].to_numpy()
        if pupil:
            pup_data = gen_util.reshape_df_data(
                self.get_pup_diam_data(
                    twop_fr, pre, post, remnans=remnans, scale=scale
                    )["pup_diam"], squeeze_rows=False, squeeze_cols=True)
            sub_df["pup_diam_data"] = math_util.mean_med(
                pup_data, stats=stats, axis=-1)
        if run:
            stim_fr = sub_df["start_frame_stim"].to_numpy()
            run_data = gen_util.reshape_df_data(
                self.get_run_data(
                    stim_fr, pre, post, remnans=remnans, scale=scale
                    )["run_velocity"], squeeze_rows=False, squeeze_cols=True)
            sub_df["run_data"] = math_util.mean_med(
                run_data, stats=stats, axis=-1)
        
        # add ROI data
        logger.info("Adding ROI data to dataframe.")
        roi_data = self.get_roi_data(
            twop_fr, pre, post, remnans=remnans, fluor=fluor, scale=scale
            )["roi_traces"]
        targ = [len(roi_data.index.unique(dim)) for dim in roi_data.index.names]
        roi_data = math_util.mean_med(roi_data.to_numpy().reshape(targ), 
            stats=stats, axis=-1, nanpol="omit")
        cols = [f"roi_data_{i}" for i in range(len(roi_data))]
        all_roi = pd.DataFrame(columns=cols, data=roi_data.T)
        sub_df = sub_df.join(all_roi)

        # drop columns
        drop = []
        for col in sub_df.columns:
            if ("frame" in col) or ("time" in col) or (col == "duration"):
                drop.append(col)

        sub_df = sub_df.drop(columns=drop)

        return sub_df


    #############################################
    def get_fr_by_seg(self, seg_list, fr_type="twop", start=False, stop=False, 
                      ch_fl=None):
        """
        self.get_fr_by_seg(seg_list)

        Returns a list of arrays containing the stimulus frame numbers that 
        correspond to a given set of stimulus segments provided in a list 
        for a specific stimulus.

        Required args:
            - seg_list (list of ints): the stimulus dataframe segments for 
                                       which to get frames

        Optional args:
            - fr_type (str): type of frames to return ("stim" or "twop")
                             return: "twop"
            - start (bool) : instead returns the start frame for each seg.
                             default: False
            - stop (bool)  : instead returns the stop frame for each seg (excl).
                             default: False
            - ch_fl (list) : if provided, flanks in sec [pre sec, post sec] 
                             around frames to check for removal if out of bounds
                             default: None

        Returns:
            if start or stop is True:
                - frames (pd DataFrame)      : frames dataframe with
                    columns:
                    - "start_frame_{fr_type}": start frame for each segment
                    - "stop_frame_{fr_type}" : stop frame for each segment (excl)
            else:
                - frames (list of int arrays): a list (one entry per segment) 
                                               of arrays containing the stim 
                                               frame
        """
        
        seg_list = np.asarray(seg_list)
        if seg_list.min() < 0:
            raise ValueError("Segments cannot be < 0.")
        if seg_list.max() >= len(self.sess.stim_df):
            raise ValueError(
                "Some segments requested are beyond the length of stim_df."
                )

        fr_types = ["twop", "stim"]
        if fr_type not in fr_types:
            gen_util.accepted_values_error("fr_type", fr_type, fr_types)

        starts = self.sess.stim_df.loc[seg_list, f"start_frame_{fr_type}"]

        ret_idx = None
        if ch_fl is not None:
            starts, ret_idx = self.sess.check_flanks(
                starts, ch_fl, fr_type=fr_type, ret_idx=True)

        if stop or (not start and not stop):
            stops = self.sess.stim_df.loc[seg_list, f"stop_frame_{fr_type}"]
            if ret_idx is not None:
                stops = stops.to_numpy()[ret_idx]


        if start or stop:
            frames = pd.DataFrame()
            if start:
                frames[f"start_frame_{fr_type}"] = starts
            if stop:
                frames[f"stop_frame_{fr_type}"] = stops
        else:
            frames = [
                np.arange(start_fr, stop_fr) 
                for start_fr, stop_fr in zip(starts, stops)
                ]

        return frames
        
        
    #############################################
    def get_n_fr_by_seg(self, seg_list, fr_type="twop"):
        """
        self.get_n_fr_by_seg(seg_list)

        Returns a list with the number of twop frames for each seg passed.    

        Required args:
            - seg_list (list of ints): the stimulus dataframe segments for 
                                       which to get frames

        Optional args:
            - fr_type (str): type of frames to return ("stim" or "twop")
                             return: "twop"

        Returns:
            - num_frames (list): list of number of frames in each segment
        """

        frames = self.get_fr_by_seg(
            seg_list, fr_type=fr_type, start=True, stop=True
            )

        start_frames = frames[f"start_frame_{fr_type}"].to_numpy()
        stop_frames = frames[f"stop_frame_{fr_type}"].to_numpy()
        
        num_frames = stop_frames - start_frames

        return num_frames


    #############################################
    def get_stim_df_by_criteria(self, unexp="any", stim_seg="any", gabfr="any", 
                                gabk="any", gab_ori="any", visflow_size="any", 
                                visflow_dir="any", start_frame_twop="any", 
                                stop_frame_twop="any", num_frames_twop="any"):
        """
        self.get_stim_df_by_criteria()

        Returns a subset of the stimulus dataframe based on the criteria 
        provided.    

        Will return lines only for the current stim object.

        Optional args:
            - unexp (str, int or list)      : unexp value(s) of interest (0, 1)
                                              default: "any"
            - stim_seg (str, int or list)   : stim_seg value(s) of interest
                                              default: "any"
            - gabfr (str, int or list)      : gaborframe value(s) of interest 
                                              (0, 1, 2, 3)
                                              default: "any"
            - gabk (int or list)            : if not None, will overwrite 
                                              (4, 16, or "any")
                                              default: "any"
            - gab_ori (int or list)         : if not None, will overwrite 
                                              (0, 45, 90, 135, 180, 225, or "any")
                                              default: "any"
            - visflow_size (int or list)    : if not None, will overwrite 
                                              (128, 256, or "any")
                                              default: "any"
            - visflow_dir (str or list)     : if not None, will overwrite 
                                              ("right", "left", "temp", 
                                              "nasal", or "any")
                                              default: "any"
            - start_frame_twop (str or list): 2p start frames range of interest
                                              [min, max (excl)] 
                                              default: "any"
            - stop_frame_twop (str or list) : 2p end frames (excluded ends) 
                                              range of interest [min, max (excl)]
                                              default: "any"
            - num_frames_twop (str or list) : 2p num frames range of interest
                                              [min, max (excl)]
                                              default: "any"
        
        Returns:
            - sub_df (pd DataFrame): subset of the stimulus dataframe 
                                     fitting the criteria provided
        """

        stimtypes = self.sess.stim_df["stimulus_type"].unique()
        if self.stimtype not in stimtypes:
            raise RuntimeError(
                f"Stimulus {self.stimtype} not found amoung dataframe stimuli: "
                f"{', '.join(stimtypes)}."
                )

        pars = sess_data_util.format_stim_criteria(
            self.sess.stim_df, self.stimtype, unexp, stim_seg=stim_seg, 
            gabfr=gabfr, gabk=gabk, gab_ori=gab_ori, visflow_size=visflow_size, 
            visflow_dir=visflow_dir, start2pfr=start_frame_twop, 
            end2pfr=stop_frame_twop, num2pfr=num_frames_twop, 
            )

        [unexp, stim_seg, gabfr, gabk, gab_ori, visflow_size, visflow_dir, 
         start_frame_twop_min, start_frame_twop_max, stop_frame_twop_min, 
         stop_frame_twop_max, num_frames_twop_min, num_frames_twop_max] = pars

        sub_df = self.sess.stim_df.loc[
            (self.sess.stim_df.index.isin(stim_seg))                        &
            (self.sess.stim_df["stimulus_type"]==self.stimtype)             & 
            (self.sess.stim_df["unexpected"].isin(unexp))                   &
            (self.sess.stim_df["gabor_frame"].isin(gabfr))                  &
            (self.sess.stim_df["gabor_kappa"].isin(gabk))                   &
            (self.sess.stim_df["gabor_mean_orientation"].isin(gab_ori))     &
            (self.sess.stim_df["square_size"].isin(visflow_size))           &
            (self.sess.stim_df["main_flow_direction"].isin(visflow_dir))    &
            (self.sess.stim_df["start_frame_twop"] >= start_frame_twop_min) &
            (self.sess.stim_df["start_frame_twop"] < start_frame_twop_max)  &
            (self.sess.stim_df["stop_frame_twop"] >= stop_frame_twop_min)   &
            (self.sess.stim_df["stop_frame_twop"] < stop_frame_twop_max)    &
            (self.sess.stim_df["num_frames_twop"] >= num_frames_twop_min)   &
            (self.sess.stim_df["num_frames_twop"] < num_frames_twop_max)]

        return sub_df


    #############################################
    def get_segs_by_criteria(self, unexp="any", stim_seg="any", gabfr="any", 
                             gabk="any", gab_ori="any", visflow_size="any", 
                             visflow_dir="any", start_frame_twop="any", 
                             stop_frame_twop="any", num_frames_twop="any",  
                             remconsec=False, by="seg"):
        """
        self.get_segs_by_criteria()

        Returns a list of stimulus seg numbers that have the specified values 
        in specified columns in the stimulus dataframe.    

        Will return segs only for the current stim object.

        Optional args:
            see self.get_stim_df_by_criteria()
            - remconsec (bool): if True, consecutive segments are 
                                removed within a block
                                default: False
            - by (str)        : determines whether segment numbers
                                are returned in a flat list 
                                ("seg"), grouped by block ("block")
                                default: "seg"
        
        Returns:
            - segs (list): list of seg numbers that obey the criteria, 
                           optionally arranged by block or display sequence
        """

        sub_df = self.get_stim_df_by_criteria(unexp, stim_seg=stim_seg, 
            gabfr=gabfr, gabk=gabk, gab_ori=gab_ori, visflow_size=visflow_size, 
            visflow_dir=visflow_dir, start_frame_twop=start_frame_twop, 
            stop_frame_twop=stop_frame_twop, num_frames_twop=num_frames_twop 
            )
        
        segs = np.sort(sub_df.index).tolist()

        # check for empty
        if len(segs) == 0:
             raise RuntimeError("No segments fit these criteria.")
        
        if by == "block":
            segs_by_block = []
            for block_n in self.block_params.index:
                block_row = self.block_params.loc[block_n]
                min_seg = block_row["start_seg"]
                max_seg = block_row["stop_seg"]
                block_segs = [
                    seg for seg in segs if seg >= min_seg and seg < max_seg
                    ]
                segs_by_block.append(block_segs)
            n_segs = [len(block_segs) for block_segs in segs_by_block]
            if n_segs > segs:
                raise RuntimeError(
                    "Should not be more segments after splitting into blocks."
                    )
            if n_segs < segs:
                raise RuntimeError(
                    "Segments should not be missing after splitting into blocks."
                    )
            segs = segs_by_block
        
        if remconsec:
            if by == "seg":
                segs = [segs]
            keep_segs = []
            
            for block_segs in segs:
                block_segs = np.asarray(block_segs)
                keep_idx = np.where(np.diff(np.insert(block_segs, 0, -2)) != 1)[0]
                keep_segs.append(block_segs[keep_idx].tolist())

            if by == "seg":
                keep_segs = keep_segs[0]
            segs = keep_segs
        
        if by not in ["seg", "block"]:
            gen_util.accepted_values_error("by", by, ["seg", "block"])
        
        return segs


    #############################################
    def get_frames_by_criteria(self, unexp="any", stim_seg="any", gabfr="any", 
                               gabk=None, gab_ori=None, visflow_size=None, 
                               visflow_dir=None, start_frame_twop="any", 
                               stop_frame_twop="any", num_frames_twop="any",  
                               remconsec=False, start_fr=True, fr_type="twop",
                               by="frame"):
        """
        self.get_frames_by_criteria()

        Returns a list of frames numbers that have the specified 
        values in specified columns in the stimulus dataframe. 
        
        Will return frame numbers only for the current stim object.

        Optional args:
            see self.get_stim_df_by_criteria()
            - remconsec (bool): if True, consecutive segments are 
                                removed within a block
                                default: False
            - start_fr (bool) : if True, only start stimulus frames are 
                                returned per segment
                                default: True
            - fr_type (str)   : type of frame to return ("twop" or "stim")
                                default: "twop"
            - by (str)        : determines whether segment numbers
                                are returned in a flat list 
                                ("frame"), grouped by block ("block")
                                default: "frame"
        
        Returns:
            - frames (list): list of frame numbers that obey the criteria, 
                             optionally arranged by block
        """


        segs = self.get_segs_by_criteria(
            unexp, stim_seg=stim_seg, gabfr=gabfr, gabk=gabk, gab_ori=gab_ori, 
            visflow_size=visflow_size, visflow_dir=visflow_dir, 
            start_frame_twop=start_frame_twop, stop_frame_twop=stop_frame_twop, 
            num_frames_twop=num_frames_twop, remconsec=remconsec, by=by
            )

        if by == "frame":
            segs = [segs]

        frames = []
        for block_segs in segs:
            block_frames = self.get_fr_by_seg(
                block_segs, fr_type=fr_type, start=start_fr
                )
            if start_fr:
                block_frames = block_frames[f"start_frame_{fr_type}"].tolist()
            else:
                block_frames = [sub.tolist() for sub in block_frames]
            
            frames.append(block_frames)
        
        if by == "frame":
            frames = frames[0]
        
        return frames


    #############################################
    def get_start_unexp_segs(self, by="seg"):
        """
        self.get_start_unexp_segs()

        Returns two lists of stimulus segment numbers, the start is a list of 
        all the start unexpected segments for the stimulus type at transitions 
        from expected to unexpected sequences. The second is a list of all the 
        start expected segements for the stimulus type at transitions from 
        unexpected to expected sequences.

        Optional args:
            - by (str): determines whether segment numbers are returned in a 
                        flat list ("seg") or grouped by block ("block")
                        default: "seg"

        Returns:
            - exp_segs (list)  : list of start expected segment numbers at 
                                 unexpected to expected transitions for 
                                 stimulus type
            - unexp_segs (list): list of start unexpected segment numbers at 
                                 expected to unexpected transitions for stimulus 
                                 type
        """

        exp_segs  = self.get_segs_by_criteria(unexp=0, remconsec=True, by=by)
        unexp_segs = self.get_segs_by_criteria(unexp=1, remconsec=True, by=by)

        return exp_segs, unexp_segs


    #############################################
    def get_all_unexp_segs(self, by="seg"):
        """
        self.get_all_unexp_segs()

        Returns two lists of stimulus segment numbers. The first is a list of 
        all the unexpected segments for the stimulus type. The second is a list 
        of all the expected segments for the stimulus type.

        Optional args:
            - by (str): determines whether segment numbers are returned in a 
                        flat list ("seg") or grouped by block ("block")
                        default: "seg"

        Returns:
            - exp_segs (list)  : list of expected segment numbers for stimulus 
                                 type
            - unexp_segs (list): list of unexpected segment numbers for 
                                 stimulus type
        """

        exp_segs  = self.get_segs_by_criteria(unexp=0, by=by)
        unexp_segs = self.get_segs_by_criteria(unexp=1, by=by)

        return exp_segs, unexp_segs
    

    #############################################
    def get_start_unexp_stim_fr_trans(self, fr_type="twop", by="frame"):
        """
        self.get_start_unexp_stim_fr_trans()

        Returns two lists of frame numbers, the start is a list of all 
        the start unexpected frames for the stimulus type at transitions from 
        expected to unexpected sequences. The second is a list of all the start 
        expected frames for the stimulus type at transitions from unexpected to 
        expected sequences.

        Optional args:
            - fr_type (str): type of frame to return ("twop" or "stim")
                             default: "twop"
            - by (str)     : determines whether frames are returned in a flat 
                             list ("frame") or grouped by block ("block")
                             default: "frame"
        
        Returns:
            - exp_fr (list)  : list of start expected stimulus frame numbers at 
                               unexpected to expected transitions for stimulus 
                               type
            - unexp_fr (list): list of start unexpected stimulus frame numbers 
                               at expected to unexpected transitions for 
                               stimulus type
        """
    
        exp_fr  = self.get_frames_by_criteria(
            unexp=0, remconsec=True, fr_type=fr_type, by=by
            )
        unexp_fr = self.get_frames_by_criteria(
            unexp=1, remconsec=True, fr_type=fr_type, by=by
            )

        return exp_fr, unexp_fr


    #############################################
    def get_all_unexp_stim_fr(self, fr_type="twop", by="frame"):
        """
        self.get_all_unexp_stim_fr()

        Returns two lists of frame numbers, the first is a list of all 
        unexpected frames for the stimulus type. The second is a list of all 
        expected frames for the stimulus type.

        Optional args:
            - fr_type (str): type of frame to return ("twop" or "stim")
                             default: "twop"
            - by (str)     : determines whether frames are returned in a flat 
                             list ("frame") or grouped by block ("block")
                             default: "frame"

        Returns:
            - exp_fr (list)  : list of all expected frame numbers for stimulus 
                               type
            - unexp_fr (list): list of all unexpected frame numbers for 
                               stimulus type
        """

        exp_fr  = self.get_frames_by_criteria(
            unexp=0, start_fr=False, fr_type=fr_type, by=by
            )
        unexp_fr = self.get_frames_by_criteria(
            unexp=1, start_fr=False, fr_type=fr_type, by=by
            )

        return exp_fr, unexp_fr
    

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
                raise KeyError("'data_df' row indices must include "
                    "'sequences'.")
            else:
                incls.append(False)
        rois, _, not_integ = incls

        if "datatype" not in data_df.columns.names:
            raise KeyError("data_df column must include 'datatype'.")

        # retrieve datatype
        datatypes = data_df.columns.get_level_values("datatype")
        if len(datatypes) != 1:
            raise RuntimeError("Expected only one datatype in data_df.")
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
            raise ValueError("'dims' can only include: "
                f"{', '.join(data_df.index.names)}")
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
    def get_pup_diam_data(self, twop_ref_fr, pre, post, integ=False, 
                           remnans=False, baseline=None, stats="mean", 
                           scale=False, metric="mm"):
        """
        self.get_pup_diam_data(pup_ref_fr, pre, post)

        Returns array of pupil data around specific pupil frame numbers. NaNs
        are omitted in calculating statistics.

        Required args:
            - twop_ref_fr (list): 1D list of reference two-photon frame numbers
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
            - metric (str)    : metric to return data in, e.g., "pixel" or "mm".
                                Only applies if scale is False.
                                default: "mm"

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

        frame_n_df = self.sess.get_fr_ran(
            twop_ref_fr, pre, post, fr_type="twop"
            )

        frame_ns = gen_util.reshape_df_data(
                frame_n_df, squeeze_rows=False, squeeze_cols=True
                )

        pup_data = self.sess.get_pup_data(
            datatype=datatype, remnans=remnans, scale=scale)

        data_array = pup_data.to_numpy().squeeze()[frame_ns]

        if not scale:
            if metric == "mm":
                data_array = data_array * sess_sync_util.MM_PER_PIXEL
            elif metric != "pixel":
                gen_util.accepted_values_error(
                    "metric", metric, ["pixel", "mm"]
                    )

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
            row_indices.append(frame_n_df.index.unique(
                level="time_values").tolist())
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
                              error="std", baseline=None, scale=False, 
                              metric="mm"):
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
            - metric (str)    : metric to return data in, e.g., "pixel" or "mm"
                                Only applies if scale is False.
                                default: "mm"

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
            stats=stats, scale=scale, metric=metric)

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
            raise RuntimeError("Run 'self.load_run_data()' to load the "
                "running data correctly.")

        frame_n_df = self.sess.get_fr_ran(
            stim_ref_fr, pre, post, fr_type="stim"
            )

        frame_ns = gen_util.reshape_df_data(
                frame_n_df, squeeze_rows=False, squeeze_cols=True
                )

        run_data = self.sess.get_run_velocity_by_fr(
            frame_ns, fr_type="stim", remnans=remnans, scale=scale)

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
            row_indices.append(frame_n_df.index.unique(
                level="time_values").tolist())
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

        if not hasattr(self.sess, "_nrois"):
            raise RuntimeError("Run 'self.load_roi_info()' to set ROI "
                "attributes correctly.")

        use_pad = pad
        if smooth:
            add_pad = np.ceil(smooth/2.0).astype(int)
            use_pad = [sub + add_pad for sub in use_pad]

        frame_n_df = self.sess.get_fr_ran(
            twop_ref_fr, pre, post, pad=use_pad, fr_type="twop"
            )

        frame_ns = gen_util.reshape_df_data(
                frame_n_df, squeeze_rows=False, squeeze_cols=True
                )

        # get dF/F: ROI x seq x fr
        roi_data_df = self.sess.get_roi_seqs(
            frame_ns, fluor=fluor, remnans=remnans, scale=scale
            )

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
    def get_run(self, by="frame", remnans=True, scale=False):
        """
        self.get_run()

        Returns run values for each stimulus frame of each stimulus block.

        Optional args:
            - by (str)      : determines whether run values are returned in a  
                              flat list ("frame") or grouped by block ("block")
                              default: "frame"
            - remnans (bool): if True, NaN values are removed using linear 
                              interpolation.
                              default: True
            - scale (bool)  : if True, each ROI is scaled based on 
                              full trace array
                              default: False
        Returns:
            - sub_run_df (pd DataFrame): dataframe containing running velocity 
                                         values (in cm/s) for the frames
                                         of interest, and optionally block 
                                         numbers, organized by:
                hierarchical columns (all dummy):
                    - datatype    : type of data (e.g., "run_velocity" 
                                    or "block_n")
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

        if by not in ["block", "frame"]:
            gen_util.accepted_values_error("by", by, ["block", "frame"])

        for b in self.block_params.index:
            row = self.block_params.loc[b]
            # pd.IndexSlice: slice end is included
            idx = pd.IndexSlice["frames", 
                row["start_frame_stim"] : row["stop_frame_stim"] + 1]
            run_df.loc[idx, "block_n"] = b
        
        # keep rows that are within blocks
        sub_run_df = run_df.loc[~run_df["block_n"].isna()]
    
        return sub_run_df

    
    #############################################
    def get_segs_by_frame(self, fr, fr_type="twop"):
        """
        self.get_segs_by_frame(fr)

        Returns the stimulus segment numbers for the given frames.

        Required args:
            - fr (array-like): set of frames for which to get stimulus seg 
                               numbers
        
        Returns:
            - segs (nd array): segment numbers (int), with same dimensions 
                               as input array
        """

        fr = np.asarray(fr)

        min_fr = self.stim_df[f"start_frame_{fr_type}"].min()
        max_fr = self.stim_df[f"stop_frame_{fr_type}"].max()
        if fr_type not in ["twop", "stim"]:
            gen_util.accepted_values_error("fr_type", fr_type, ["twop", "stim"])

        # make sure the frames are within the range
        if (fr >= max_fr).any() or (fr < min_fr).any():
            raise UserWarning("Some of the specified frames are out of range")

        targ_frames = self.stim_df[f"start_frame_{fr_type}"].to_numpy()
        segs = np.searchsorted(targ_frames, fr).astype(int)

        return segs

    
#############################################
#############################################
class Gabors(Stim):
    """
    The Gabors object inherits from the Stim object and describes gabor 
    specific properties.
    """

    def __init__(self, sess, stim_dict=None):
        """
        self.__init__(sess)
        
        Initializes and returns a Gabors object, and the attributes below. 
        
        Attributes:
            - n_patches (int)   : number of gabor patches
            - ori_ran (list)    : Gabor orientation range (deg)
            - phase (num)       : phase of the gabors (0-1)
            - sf (num)          : spatial frequency of the gabors (cyc/pix)
            - size_ran (list)   : Gabor patch size range (deg)
            
            Block parameters:
            - kappas (list)     : Gabor kappas (unordered)

            Gabor frame attributes
            - all_gabfr (list)  : Gabor frames 
                                  (["A", "B", "C", "D", "U", "G"])
            - exp_gabfr (list)  : expected Gabor frames 
                                  (["A", "B", "C", "D", "G"])
            - unexp_gabfr (list): unexpected Gabor frames 
                                  (["U"])
            - all_gabfr_mean_oris (list)  : possible mean orientations for all 
                                            frames (deg)
            - exp_gabfr_mean_oris (list)  : possible mean orientations for 
                                            expected frames (deg)
            - unexp_gabfr_mean_oris (list): possible mean orientations for 
                                            unexpected frames (deg)

        Required args:
            - sess (Session)  : session to which the gabors belongs
        
        Optional args:
            - stim_dict (dict): stimulus dictionary 
                                (only applicable if self.sess.nwb is False, in 
                                which case it will be loaded if not passed)
                                default: None
        """

        super().__init__(sess, stimtype="gabors", stim_dict=stim_dict)
        
        if self.sess.nwb:
            self.sf        = 0.04
            self.phase     = 0.25
            self.size_ran  = [10, 20]
        else:
            if stim_dict is None:
                stim_dict = self.sess._load_stim_dict(full_table=False)

            gen_stim_props = sess_stim_df_util.load_gen_stim_properties(
                stim_dict, stimtype="gabors", runtype=self.sess.runtype
                )
        
            self.sf        = gen_stim_props["sf"]
            self.phase     = gen_stim_props["phase"]
            self.size_ran  = gen_stim_props["size_ran"]
        
        # collect information from sess.stim_df
        stimtype_df = self.sess.stim_df.loc[
            self.sess.stim_df["stimulus_type"] == "gabors"
            ]

        # number of Gabor patches
        n_patches = stimtype_df.loc[
            stimtype_df["gabor_frame"] != "G"
            ]["gabor_number"].unique()

        if len(n_patches) != 1:
            n_patch_info = n_patches if len(n_patches) else "none" 
            raise RuntimeError(
                "Expected exactly one value for number of Gabor patches, "
                f"but found {n_patch_info}."
                )
        self.n_patches = int(n_patches[0]) 

        # Gabor frames and mean orientations
        self.ori_ran = sess_stim_df_util.GABOR_ORI_RANGE
        self.exp_gabfr = ["A", "B", "C", "D", "G"]
        self.unexp_gabfr = ["U"]
        self.all_gabfr = self.exp_gabfr + self.unexp_gabfr

        all_mean_oris = []
        for gab_fr in [self.exp_gabfr, self.unexp_gabfr, self.all_gabfr]:
            if "G" in gab_fr:
                gab_fr.remove("G")

            mean_oris = stimtype_df.loc[
                stimtype_df["gabor_frame"].isin(gab_fr)
                ]["gabor_mean_orientation"].unique()
            all_mean_oris.append(np.sort(mean_oris).tolist())

        self.exp_gabfr_mean_oris   = all_mean_oris[0]
        self.unexp_gabfr_mean_oris = all_mean_oris[1]
        self.all_gabfr_mean_oris   = all_mean_oris[2]

        # collect Gabor kappas from block parameters
        self.kappas = sorted(self.block_params["gabor_kappa"].unique())


    #############################################
    def get_A_segs(self, by="seg"):
        """
        self.get_A_segs()

        Returns lists of A gabor segment numbers.

        Included here as a basic example.

        Optional args:
            - by (str): determines whether segment numbers are returned in a 
                        flat list ("seg") or grouped by block ("block")
                        default: "seg"
        Returns:
            - A_segs (list): list of A gabor segment numbers.
        """
        
        A_segs = self.get_segs_by_criteria(gabfr=0, by=by)

        return A_segs


    #############################################
    def get_A_frame_1s(self, fr_type="twop", by="frame"):
        """
        self.get_A_frame_1s()

        Returns list of start frame number for each A gabor segment number.

        Included here as a basic example.

        Optional args:
            - fr_type (str): type of frame to return 
                             default: "twop"
            - by (str)     : determines whether frame numbers are returned in a 
                             flat list ("frame") or grouped by block ("block")
                             default: "frame"
     
        Returns:
            - A_segs (list) : lists of start frame number for each A gabor 
                              segment number
        """
        
        A_frames = self.get_frames_by_criteria(
            gabfr=0, fr_type=fr_type, by=by, start_fr=True
            )

        return A_frames
    

    #############################################
    def get_stim_par_by_seg(self, segs, pos=True, ori=True, size=True, 
                            scale=False):
        """
        self.get_stim_par_by_seg(segs)

        Returns stimulus parameters for specified segments.

        NOTE: An error is raised if segments are out of range.

        Required args:
            - segs (nd array): array of segments for which parameters are
                               requested
        
        Optional args:
            - pos (bool)  : if True, the positions of each Gabor are returned
                            (in x and y separately)
                            default: True
            - ori (bool)  : if True, the orientations of each Gabor are returned
                            (in deg, 0 to 360)
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

        # check that at least one parameter type is requested
        if not(pos or ori or size):
            raise ValueError("At least one of the following must be True: "
                "pos, ori, size.")
        
        if ori and not self.sess._full_table:
            self.sess._load_stim_df(full_table=True)
        
        if (segs < 0).any():
            raise NotImplementedError("segs should not contain negative values.")
        if segs.max() >= len(self.sess.stim_df):
            raise ValueError(
                "segs contains values above length of stimulus dataframe."
                )

        sub_df = self.sess.stim_df.loc[segs]

        main_gab_segs = sub_df.loc[
            (sub_df["stimulus_type"] == "gabors") & 
            (sub_df["gabor_frame"] != "G")
            ].index
        gray_gab_segs = sub_df.loc[sub_df["gabor_frame"] == "G"].index
        non_gab_segs = sub_df.loc[sub_df["stimulus_type"] != "gabors"].index

        if len(non_gab_segs):
            raise ValueError("Some of the segments requested are out of "
                "range for Gabors stimulus.")
        
        row_index = pd.MultiIndex.from_product(
            [sub_df.index, range(self.n_patches)], 
            names=["seg_n", "gabor_n"])

        param_df = pd.DataFrame(None, index=row_index, columns=[])

        for param, param_name in zip(
            [pos, pos, size, ori], ["pos_x", "pos_y", "size", "ori"]
            ):
            if not param:
                continue
            if param_name == "ori":
                column = "gabor_orientations"
                ran = self.ori_ran
            elif param_name == "size":
                column = "gabor_sizes"
                ran = self.size_ran
            elif param_name == "pos_x":
                column = "gabor_locations_x"
                ran = self.win_size
            elif param_name == "pos_y":
                column = "gabor_locations_y"
                ran = self.win_size
            
            vals = np.asarray(sub_df.loc[main_gab_segs, column].to_list())
            if vals.max() > ran[1] or vals.min < ran[0]:
                raise RuntimeError(
                    f"Expected {column} data to be between {ran[0]} and "
                    f"{ran[1]}, but found values outside that range."
                    )

            if scale:
                sub = ran[0]
                div = ran[1] - ran[0]
                vals = math_util.scale_data(vals, facts=[sub, div, 2, -1])
            param_df.loc[main_gab_segs, column] = vals.reshape(-1)
            
            # add 0s for grayscreen
            param_df.loc[gray_gab_segs, column] = \
                np.full([len(gray_gab_segs), self.n_patches], 0).reshape(-1)

        # create a dataframe organized like 'segs' and transfer data
        names = [f"{''.join(['sub_'] * i)}sequence" 
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
    def get_stim_par_by_frame(self, fr, pre, post, pos=True, ori=True, 
                              size=True, scale=False, fr_type="twop"):
        """
        self.get_stim_par_by_frame(fr, pre, post)

        Returns stimulus parameters for frame sequences specified by the 
        reference frame numbers and pre and post ranges.

        NOTE: An error is raised if frames are out of range.

        Required args:
            - fr (list) : 1D list of frame numbers 
                          (e.g., all 1st Gabor A frames)
            - pre (num) : range of frames to include before each 
                          reference frame number (in s)
            - post (num): range of frames to include after each 
                          reference frame number (in s)
                    
        Optional args:
            - pos (bool)   : if True, the positions of each Gabor are returned
                             (in x and y separately)
                             default: True
            - ori (bool)   : if True, the orientations of each Gabor are returned
                             default: True
            - size (bool)  : if True, the sizes of each Gabor are returned
                             default: True
            - scale (bool) : if True, values are scaled to between -1 and 1 
                             (each parameter type separately, to its full 
                             possible range)
                             default: False
            - fr_type (str): type of frame to return 
                             default: "twop"

        Returns:
            - full_param_df (pd DataFrame): dataframe containing gabor parameter
                                            values for each frame,
                                            organized by:
                hierarchical columns:
                    - parameters  : parameter names ("pos_x", "pos_y", "size", 
                                    "ori")
                hierarchical rows:
                    (- "sequence" : sequence number for first dimension of segs
                     - "sub_sequence", ...)
                    - "{fr_type}_fr_n" : two-photon frame number
                    - "gabor_n"   : gabor number
        """



        fr_seqs = gen_util.reshape_df_data(
            self.sess.get_fr_ran(
                fr, pre, post, fr_type=fr_type
                ).loc[:, (f"{fr_type}_fr_n", )], 
            squeeze_cols=True)

        segs = self.get_segs_by_fr(fr_seqs, fr_type=fr_type)

        full_param_df = self.get_stim_par_by_seg(
            segs, pos=pos, ori=ori, size=size, scale=scale)

        full_param_df.index.set_levels(
            fr_seqs.reshape(-1), level="seg_n", inplace=True)

        full_param_df.index.rename(
            f"{fr_type}_fr_n", level="seg_n", inplace=True
            )

        return full_param_df


#############################################
#############################################
class Visflow(Stim):
    """
    The Visflow object inherits from the Stim object and describes visual flow 
    specific properties.
    """

    def __init__(self, sess, stim_dict=None):
        """
        self.__init__(sess)
        
        Initializes and returns a visual flow object, and the attributes below. 

        Attributes:
            - prop_flipped (float): proportion of squares that flip direction 
                                    following unexpected sequence onset (0-1)
            - speed (num)         : speed at which the visual flow squares 
                                    are moving (pix/sec)

            Block parameters:
            - main_flow_direcs (list): main directions of flow (unordered)
            - n_squares (list)       : number of squares (unordered)
            - square_sizes (list)    : square sizes (unordered) (pix)

        Required args:
            - sess (Session)  : session to which the visual flow stimulus 
                                belongs

        Optional args:
            - stim_dict (dict): stimulus dictionary 
                                (only applicable if self.sess.nwb is False, in 
                                which case it will be loaded if not passed)
                                default: None
        """

        super().__init__(sess, stimtype="visflow", stim_dict=stim_dict)

        if self.sess.nwb:
            self.speed = 50 # pix/sec
        else:
            if stim_dict is None:
                stim_dict = self.sess._load_stim_dict(full_table=False)
            gen_stim_props = sess_stim_df_util.load_gen_stim_properties(
                stim_dict, stimtype="visflow", runtype=self.sess.runtype
                )
        
            self.speed = gen_stim_props["speed"]

        # collect visual flow information from sess.stim_df
        stimtype_df = self.sess.stim_df.loc[
            self.sess.stim_df["stimulus_type"] == "visflow"
            ]
            
        prop_flipped = stimtype_df.loc[
                stimtype_df["unexpected"] == 1
                ]["square_proportion_flipped"].unique()

        if len(prop_flipped) != 1:
            prop_info = prop_flipped if len(prop_flipped) else "none" 
            raise RuntimeError(
                "Expected exactly one value for number of proportion flipped "
                f"squares, but found {prop_info}."
                )
        self.prop_flipped = prop_flipped[0] 

        # collect visual flow information from block parameters
        self.main_flow_direcs = sorted(
            self.block_params["main_flow_direction"].unique()
            )
        self.square_sizes = sorted(self.block_params["square_size"].unique())
        self.n_squares = sorted(self.block_params["square_number"].unique())


    #############################################
    def get_dir_segs_exp(self, by="seg"):
        """
        self.get_dir_segs_exp()

        Returns two lists of stimulus segment numbers, the first is a list of 
        the temporal moving segments. The second is a list of nasal 
        moving segments. Both lists exclude unexpected segments.

        Optional args:
            - by (str): determines whether segment numbers are returned in a 
                        flat list ("seg") or grouped by block ("block")
                        default: "seg"  
        Returns:
            - temp_segs (list) : list of temporal (head to tail) moving segment 
                                 numbers, excluding unexpected segments.
            - nasal_segs (list): list of nasal (tail to head) moving segment 
                                 numbers, excluding unexpected segments.
        """

        temp_segs = self.get_segs_by_criteria(
            visflow_dir="temp", unexp=0, by=by
            )
        nasal_segs = self.get_segs_by_criteria(
            visflow_dir="nasal", unexp=0, by=by
            )

        return temp_segs, nasal_segs


#############################################
#############################################
class Grayscr():
    """
    The Grayscr object describes describes grayscreen specific properties.
    """

    
    def __init__(self, sess):
        """
        self.__init__(sess)
        
        Initializes and returns a grayscr object, and the attributes below. 

        Attributes:
            - sess (Session object): session to which the grayscr belongs
        
        Required args:
            - sess (Session object): session to which the grayscr belongs
        """

        self.sess = sess
        

    #############################################
    def __repr__(self):
        return (f"{self.__class__.__name__} (session {self.sess.sessid})")

    def __str__(self):
        return repr(self)


    #############################################        
    def get_all_fr(self, fr_type="twop"):
        """
        self.get_all_fr()

        Returns a lists of grayscreen stimulus frame numbers, excluding 
        grayscreen Gabor frames.

        Optional args:
            - fr_type (str): type of frame to return 
                             default: "twop"

        Returns:
            - grayscr_stim_frs (list): list of grayscreen stimulus frames.
        """

        grayscr_loc = (self.sess.stim_df["stimulus_type"] == "grayscreen")
        
        start_frames  = self.sess.stim_df.loc[
            grayscr_loc, f"start_frame_{fr_type}"
            ]
        stop_frames  = self.sess.stim_df.loc[
            grayscr_loc, f"stop_frame_{fr_type}"
            ]
        
        all_frames = []
        for start, stop in zip(start_frames, stop_frames):
            all_frames.append(np.arange(start, stop))
        
        grayscr_stim_frs = np.concatenate(all_frames).tolist()

        return grayscr_stim_frs


    #############################################
    def get_start_fr(self, fr_type="twop"):
        """
        self.get_start_fr()

        Returns every start grayscreen stimulus frame number for every 
        grayscreen sequence that is not a Gabor frame. 

        Optional args:
            - fr_type (str): type of frame to return 
                             default: "twop"

        Returns:
            - start_grays_df (pd DataFrame): dataframe containing stimulus 
                                             frame information on each start 
                                             grayscreen sequence, with cols: 
                - "start_frame_stim": start stimulus frame number for the 
                                       grayscreen sequence
                - "num_frames_stim" : number of frames
        """

        start_grays_df = pd.DataFrame()
        
        grayscr_loc = (self.sess.stim_df["stimulus_type"] == "grayscreen")
        
        start_frames  = self.sess.stim_df.loc[
            grayscr_loc, f"start_frame_{fr_type}"
            ].tolist()
        stop_frames  = self.sess.stim_df.loc[
            grayscr_loc, f"stop_frame_{fr_type}"
            ]

        num_frames = []
        for start, stop in zip(start_frames, stop_frames):
            num_frames.append(stop - start)

        start_grays_df[f"start_frame_{fr_type}"] = start_frames
        start_grays_df[f"num_frames_{fr_type}"] = num_frames

        return start_grays_df


    #############################################
    def get_stop_fr(self, fr_type="twop"):
        """
        self.get_stop_fr()

        Returns every stop grayscreen stimulus frame number for every 
        grayscreen sequence that is not a Gabor frame. 

        Optional args:
            - fr_type (str): type of frame to return 
                             default: "twop"

        Returns:
            - stop_grays_df (pd DataFrame): dataframe containing stimulus 
                                            frame information on each stop 
                                            grayscreen sequence
                - "stop_frame_stim": stop stimulus frame number for the 
                                     grayscreen sequence (excl)
                - "num_frames_stim": number of frames
        """

        stop_grays_df = pd.DataFrame()
        
        grayscr_loc = (self.sess.stim_df["stimulus_type"] == "grayscreen")
        
        start_frames  = self.sess.stim_df.loc[
            grayscr_loc, f"start_frame_{fr_type}"
            ].tolist()
        stop_frames  = self.sess.stim_df.loc[
            grayscr_loc, f"stop_frame_{fr_type}"
            ].tolist()

        num_frames = []
        for start, stop in zip(start_frames, stop_frames):
            num_frames.append(stop - start)

        stop_grays_df[f"stop_frame_{fr_type}"] = stop_frames
        stop_grays_df[f"num_frames_{fr_type}"] = num_frames

        return stop_grays_df

