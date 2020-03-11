"""
session.py

Classes to store, extract, and analyze an AIBS OpenScope session for
the Credit Assignment Project.

Authors: Colleen Gillon, Blake Richards

Date: August, 2018

Note: this code uses python 3.7.

"""
import copy
import glob
import os
import pdb
import sys
import warnings

import h5py
import numpy as np
import pandas as pd
import pickle
import scipy.stats as st
import scipy.signal as scsig

from allensdk.brain_observatory import dff, roi_masks

from sess_util import sess_data_util, sess_file_util, sess_load_util, \
                      sess_pupil_util, sess_sync_util
from util import file_util, gen_util, math_util



#############################################
#############################################
class Session(object):
    """
    The Session object is the top-level object for analyzing a session from the 
    AIBS OpenScope Credit Assignment Project. All that needs to be provided to 
    create the object is the directory in which the session data directories 
    are located and the ID for the session to analyze/work with. The Session 
    object that is created will contain all of the information relevant to a 
    session, including stimulus information, behaviour information and 
    pointers to the 2p data.
    """
    
    def __init__(self, datadir, sessid, runtype='prod', droptol=0.0003, 
                 verbose=False):
        """
        self.__init__(datadir, sessid)

        Initializes and returns the new Session object using the specified data 
        directory and ID.

        Sets attributes:
            - droptol (num)  : dropped frame tolerance (proportion of total)
            - home (str)     : path of the master data directory
            - runtype (str)  : 'prod' (production) or 'pilot' data
            - sessid (int)   : session ID (9 digits), e.g. '712483302'
        
        and calls self._init_directory()


        Required args:
            - datadir (str): full path to the directory where session 
                             folders are stored.
            - sessid (int) : the ID for this session.

        Optional args:
            - runtype (str) : the type of run, either 'pilot' or 'prod'
                              default: 'prod'
            - droptol (num) : the tolerance for percentage stimulus frames 
                              dropped, create a Warning if this condition 
                              isn't met.
                              default: 0.0003 
            - verbose (bool): if True, will print instructions on next steps to 
                              load all necessary data
                              default: True
        """


        self.home   = datadir
        self.sessid = int(sessid)
        if runtype not in ['pilot', 'prod']:
            gen_util.accepted_values_error('runtype', runtype, 
                                           ['pilot', 'prod'])
        self.runtype = runtype
        self.droptol = droptol
        self._init_directory()

        if verbose:
            print('To load mouse database information into the session, run '
                  '`sess.extract_sess_attribs()`.\nTo load stimulus, '
                  'behaviour and ophys data, run `self.extract_info()`')


    #############################################
    def __repr__(self):
        return f"{self.__class__.__name__}('{self.sessid}')"

    def __str__(self):
        return repr(self)


    #############################################
    def _init_directory(self):
        """
        self._init_directory()

        Checks that the session data directory obeys the expected organization
        scheme and sets the following attributes:
        
            - align_pkl (str)       : path name of the stimulus alignment pickle 
                                      file
            - behav_video_h5 (str)  : path name of the behavior hdf5 file
            - correct_data_h5 (str) : path name of the motion corrected 2p data 
                                      hdf5 file
            - date (str)            : session date (i.e., yyyymmdd)
            - dir (str)             : path of session directory
            - expdir (str)          : path name of experiment directory
            - expid (int)           : experiment ID (8 digits)
            - mouseid (int)         : mouse ID (6 digits)
            - mouse_dir (bool)      : whether path includes a mouse directory
            - procdir (str)         : path name of the processed data directory
            - pup_video_h5 (str)    : path name of the pupil hdf5 file
            - roi_trace_h5 (str)    : path name of the ROI raw fluorescence trace 
                                      hdf5 file
            - roi_trace_dff_h5 (str): path name of the ROI raw dF/F trace 
                                      hdf5 file
            - stim_pkl (str)        : path name of the stimulus pickle file
            - stim_sync_h5 (str)    : path name of the stimulus synchronisation 
                                      hdf5 file
            - time_sync_h5 (str)    : path name of the time synchronization hdf5 
                                      file
            - zstack_h5 (str)       : path name of the z-stack 2p hdf5 file
        """

        # check that the high-level home directory exists
        file_util.checkdir(self.home)

        sessdir, mouse_dir = sess_file_util.get_sess_dir_path(self.home, 
                                            self.runtype, self.sessid)
        self.dir       = sessdir
        self.mouse_dir = mouse_dir

        mouseid, date = sess_file_util.get_mouse_date(self.dir, self.sessid)
        self.mouseid = mouseid
        self.date    = date

        expid = sess_file_util.get_expid(self.dir)
        self.expid = expid

        (expdir, procdir, filepaths) = \
            sess_file_util.get_file_names(self.home, self.sessid, self.expid, 
                                          self.date, self.mouseid, 
                                          self.runtype, self.mouse_dir)  
        self.expdir           = expdir
        self.procdir          = procdir
        self.stim_pkl         = filepaths['stim_pkl']
        self.stim_sync_h5     = filepaths['stim_sync_h5']
        self.align_pkl        = filepaths['align_pkl']
        self.behav_video_h5   = filepaths['behav_video_h5']
        self.pup_video_h5     = filepaths['pupil_video_h5']
        self.time_sync_h5     = filepaths['time_sync_h5']
        self.roi_trace_h5     = filepaths['roi_trace_h5']
        self.roi_trace_dff_h5 = filepaths['roi_trace_dff_h5']

        # existence not checked in get_file_names()
        # self.zstack_h5 = filepaths['zstack_h5']
        # self.correct_data_h5 = filepaths['correct_data_h5']
    

    #############################################
    @property
    def pup_data_h5(self):
        """
        self.pup_data_h5

        Returns:
            - _pup_data_h5 (list or str): single pupil data file path if one is 
                                          found, a list if several are found 
                                          and 'none' if none is found
        """
        
        if not hasattr(self, '_pup_data_h5'):
            self._pup_data_h5 = sess_file_util.get_pupil_data_h5_path(self.dir)

        return self._pup_data_h5


    #############################################
    def _load_stim_dict(self, fulldict=True):
        """
        self._load_stim_dict()

        Loads the stimulus dictionary from the stimulus pickle file, checks
        whether the dropped stimulus frames exceeds the drop tolerance and
        prints a warning if it does. 
        
        Sets the following attributes: 
            - drop_stim_fr (list) : list of dropped stimulus frames
            - n_drop_stim_fr (int): number of dropped stimulus frames
            - post_blank (num)    : number of blank screen seconds after the 
                                    stimulus end
            - pre_blank (num)     : number of blank screen seconds before the
                                    stimulus start 
            - stim_dict (dict)    : stimulus dictionary
            - stim_fps (num)      : stimulus frames per second
            - tot_stim_fr (int)   : number of stimulus frames

        Optional args:
            - fulldict (bool)  : if True, the full stim_dict is loaded,
                                 else the small stim_dict is loaded
                                 (does not contain 'posbyframe' for Bricks)
                                 default: True
        """

        if fulldict:
            self.stim_dict = file_util.loadfile(self.stim_pkl)

        else:
            self.stim_dict = sess_load_util.load_small_stim_pkl(self.stim_pkl, 
                                            self.runtype)
        
        self.stim_fps       = self.stim_dict['fps']
        self.tot_stim_fr    = self.stim_dict['total_frames']
        self.pre_blank      = self.stim_dict['pre_blank_sec']  # seconds
        self.post_blank     = self.stim_dict['post_blank_sec'] # seconds
        self.drop_stim_fr   = self.stim_dict['droppedframes']
        self.n_drop_stim_fr = len(self.drop_stim_fr[0])

        sess_sync_util.check_drop_tolerance(self.n_drop_stim_fr, 
                                            self.tot_stim_fr, self.droptol, 
                                            raise_exc=False)

    #############################################
    def _load_align_df(self):
        """
        self._load_align_df()

        Loads stimulus dataframe and alignment information.

        Sets the following attributes:
            - stim_df (pd DataFrame): stimlus alignment dataframe with columns:
                                        'stimType', 'stimPar1', 'stimPar2', 
                                        'surp', 'stimSeg', 'gabfr', 
                                        'start2pfr', 'end2pfr', 'num2pfr'
            - stim2twopfr (1D array): 2p frame numbers for each stimulus frame, 
                                      as well as the flanking
                                      blank screen frames 
            - twop_fps (num)        : mean 2p frames per second
            - twop_fr_stim (int)    : number of 2p frames recorded while stim
                                      was playing
        """

        [stim_df, stim2twopfr, 
         twop_fps, twop_fr_stim] = sess_load_util.load_stim_df_info(
                self.stim_pkl, self.stim_sync_h5, self.align_pkl, self.dir, 
                self.runtype)

        self.stim_df      = stim_df
        self.stim2twopfr  = stim2twopfr
        self.twop_fps     = twop_fps
        self.twop_fr_stim = twop_fr_stim


    #############################################
    def _load_sync_h5_data(self):
        """
        self._load_sync_h5_data()

        Loads the synchronisation hdf5 files for behavior and pupil.

        Sets the following attributes:
            - pup_fps (num)           : average pupil frame rate (frames per 
                                        sec)
            - pup_fr_interv (1D array): interval in sec between each pupil 
                                        frame
            - stim2twopfr2 (1D array) : 2p frame numbers for each stimulus 
                                        frame, as well as the flanking
                                        blank screen frames (second 
                                        version, very similar to stim2twopfr 
                                        with a few differences)
            - tot_pup_fr (int)        : total number of pupil frames
            - twop2bodyfr (1D array)  : body-tracking video (video-0) frame 
                                        numbers for each 2p frame
            - twop2pupfr (1D array)   : eye-tracking video (video-1) frame 
                                        numbers for each 2p frame
            - twop2stimfr (1D array): stimulus frame numbers for the beginning
                                      of each 2p frame (np.nan when no stimulus
                                      appears)
        """

        [pup_fr_interv, twop2bodyfr, 
         twop2pupfr, stim2twopfr2] = sess_load_util.load_sync_h5_data(
                                         self.pup_video_h5, self.time_sync_h5)
        self.pup_fr_interv = pup_fr_interv
        self.twop2bodyfr   = twop2bodyfr
        self.twop2pupfr    = twop2pupfr
        self.stim2twopfr2  = stim2twopfr2
        self.twop2stimfr   = sess_sync_util.get_twop2stimfr(self.stim2twopfr, 
                                            len(self.twop2pupfr))


        self.pup_fps = 1/(np.mean(self.pup_fr_interv))
        self.tot_pup_fr = len(self.pup_fr_interv + 1)

    
    #############################################
    def load_run_data(self, diff_thr=100, interp=True, replace=False):
        """
        self.load_run_data()

        Sets attribute with running with outliers replaced with NaNs.
            - run (1D array)       : array of running velocities in cm/s for 
                                     each recorded stimulus frames with outliers 
                                     replaced with NaNs
            - run_facts (list)     : stand. factors (IQR, median) for running
            - tot_run_fr (1D array): number of running velocity frames

            if interp:
            - run_interp (1D array)  : array of running velocities in cm/s for 
                                       each recorded stimulus frames, where 
                                       NaNs are reinterpolated
            - run_interp_facts (list): stand. factors (IQR, median) for 
                                       interpolated running

        Optional args:
            - diff_thr (int): threshold of difference in running velocity to 
                              identify outliers
                              default: 100
            - interp (bool) : if True, interpolated running data is also loaded
                              default: True
            - replace (bool): if True, running data is replaced
                              default: False
        """

        if hasattr(self, 'run') and replace:
            print(f'Replacing running values using {diff_thr} difference '
                   'threshold. All running attributes will also be reset.')
            delattr(self, 'run')

            if hasattr(self, 'run_interp'):
                delattr(self, 'run_interp')
                interp = True

        if not hasattr(self, 'run'):
            self.run        = sess_load_util.load_run_data(self.stim_dict, 
                                             diff_thr)
            self.run_facts  =  math_util.scale_facts(self.run, 
                                         sc_type='stand_rob', extrem='reg',
                                         nanpol='omit')[0:2]
            self.tot_run_fr =  len(self.run)

        if not hasattr(self, '_run_interp') and interp:
            self.run_interp       = math_util.lin_interp_nan(self.run)
            self.run_interp_facts =  math_util.scale_facts(self.run_interp, 
                                            sc_type='stand_rob', extrem='reg',
                                            nanpol='omit')[0:2]


    #############################################
    def load_pup_data(self):
        """
        self.load_pup_data()

        If it exists, loads the pupil tracking data. Extracts pupil diameter
        and position information in pixels.

        Sets the following attributes:
            where blinks and outlier values have been replaced with NaNs:
            - pup_center_x (1D array)     : pupil center position for x at 
                                            each pupil frame in pixels
            - pup_center_x_facts (list)   : stand. factors (std, mean) for 
                                            pupil center position for x
            - pup_center_y (1D array)     : pupil center position for y at 
                                            each pupil frame in pixels
            - pup_center_y_facts (list)   : stand. factors (std, mean) for 
                                            pupil center position for y
            - pup_center_diff (1D array)  : change in pupil center between 
                                            each pupil frames in pixels
            - pup_center_diff_facts (list): stand. factors (std, mean) for 
                                            change in pupil center between
                                            each pupil frames
            - pup_diam_facts (list)       : stand. factors (std, mean) for 
                                            pupil diameters

            same, as above but with blinks and outlier values linearly 
            interpolated:            
            - pup_center_x_interp (1D array)
            - pup_center_x_interp_facts (list)
            - pup_center_y_interp (1D array)
            - pup_center_y_interp_facts (list)
            - pup_center_diff_interp (1D array)
            - pup_center_diff_interp_facts (list)
            - pup_diam_interp (1D array)
            - pup_diam_interp_facts (list)
        """

        if hasattr(self, 'pup_diam'):
            return
        
        [self.pup_diam, self.pup_center_x, 
            self.pup_center_y] = sess_load_util.load_pup_data(self.pup_data_h5)

        self.pup_center_diff = sess_pupil_util.get_center_dist_diff(
                                    self.pup_center_x, self.pup_center_y)

        self.pup_diam_interp     = math_util.lin_interp_nan(self.pup_diam)
        self.pup_center_x_interp = math_util.lin_interp_nan(self.pup_center_x)
        self.pup_center_y_interp = math_util.lin_interp_nan(self.pup_center_y)
        self.pup_center_diff_interp = math_util.lin_interp_nan(
                                                self.pup_center_diff)

        data_arrs = [self.pup_diam, self.pup_center_x, self.pup_center_y, 
                     self.pup_center_diff, self.pup_diam_interp, 
                     self.pup_center_x_interp, self.pup_center_y_interp,
                     self.pup_center_diff_interp]
        all_facts = []
        for data in data_arrs:
            all_facts.append(math_util.scale_facts(data, sc_type='stand_rob', 
                                       extrem='reg', nanpol='omit')[0:2])
        [self.pup_diam_facts, self.pup_center_x_facts, self.pup_center_y_facts, 
         self.pup_center_diff_facts, self.pup_diam_interp_facts, 
         self.pup_center_x_interp_facts, self.pup_center_y_interp_facts,
         self.pup_center_diff_interp_facts] = all_facts


    #############################################
    def _set_nanrois(self, fluor='dff'):
        """
        self._set_nanrois()

        Sets attributes the indices of ROIs containing NaNs or Infs in the
        raw or dff data.

            if fluor is 'dff':
                - nanrois_dff (list): list of ROIs containing NaNs or Infs in
                                      the ROI dF/F traces
            if fluor is 'raw':
                - nanrois (list)    : list of ROIs containing NaNs or Infs in
                                      the ROI raw traces


        Optional args:
            - fluor (str): if 'dff', a nanrois attribute is added for dF/F 
                           traces. If 'raw, it is created for raw traces.
                           default: 'dff'
        """
        
        # generate attribute listing ROIs with NaNs or Infs (for dff traces)
        if fluor == 'dff':
            full_trace_file = self.roi_trace_dff_h5
        elif fluor == 'raw':
            full_trace_file = self.roi_trace_h5
        else:
            gen_util.accepted_values_error('fluor', fluor, ['raw', 'dff'])
        
        if not os.path.exists(full_trace_file):
            raise ValueError('Specified ROI traces file does not exist: '
                             f'{full_trace_file}')
        
        with h5py.File(full_trace_file, 'r') as f:
            traces = f['data'].value

        nan_arr = np.isnan(traces).any(axis=1) + np.isinf(traces).any(axis=1)
        nan_rois = np.where(nan_arr)[0].tolist()

        if fluor == 'dff':
            self.nanrois_dff = nan_rois
        elif fluor == 'raw':
            self.nanrois = nan_rois


    #############################################
    def _get_roi_facts(self, fluor='dff'):
        """
        self._get_roi_facts()

        Returns standardization factors for ROIs.

        Optional args:
            - fluor (str)   : if 'dff', a nanrois attribute is added for dF/F 
                              traces. If 'raw, it is created for raw traces.
                              default: 'dff'

        Returns:
            - roi_facts (2D array): ROI standardization factors (IQR, med) for 
                                    each ROI, structured as fact x ROI
        """
        
        # generate attribute listing ROIs with NaNs or Infs (for dff traces)
        if fluor == 'dff':
            roi_facts = self.roi_dff_facts
        elif fluor == 'raw':
            roi_facts = self.roi_facts
        else:
            gen_util.accepted_values_error('fluor', fluor, ['raw', 'dff'])
        
        return roi_facts


    ############################################
    def _load_dff(self, replace=False, basewin=1000):
        """
        self._load_dff()

        Loads the dF/F trace information, and sets the attributes below.

        Also calls self._set_nanrois().

            - roi_dff_facts (2D array): stand. factors for ROIs, structured as 
                                        facts (std, mean) x ROI
            
        Required args:
            - replace (bool): if True, replaces pre-existing dF/F traces. If
                              False, no new dF/F traces are created if they
                              already exist.
                              default: False
            - basewin (int) : basewin factor for compute_dff function
                              default: 1000
        
        """

        sess_load_util.create_dff(self.roi_trace_h5, self.roi_trace_dff_h5, 
                       self.sessid, replace=replace, basewin=basewin)

        # get standardization facts (factor x ROI)
        with h5py.File(self.roi_trace_dff_h5, 'r') as f:
            self.roi_dff_facts = np.asarray(math_util.scale_facts(f['data'][()], 
                                            axis=1, sc_type='stand_rob', 
                                            extrem='perc', allow_0=True)[0:2])

        # generate attribute listing ROIs with NaNs or Infs (for dF/F traces)
        self._set_nanrois('dff')


    #############################################
    def _load_roi_trace_info(self, basewin=1000, dend='extr'):
        """
        self._load_roi_trace_info()

        Sets the attributes below based on the raw ROI traces.
        
        Also calls self._set_nanrois().

            - dend (str)          : type of dendrites loaded ('aibs' or 'extr')
            - nrois (int)         : number of ROIs in traces
            - roi_facts (2D array): stand. factors for ROIs, structured as 
                                    facts (std, mean) x ROI
            - roi_names (list)    : list of ROI names (9 digits)
            - tot_twop_fr (int)   : number of 2p frames recorded

        Optional args:
            - basewin (int): window length for calculating baseline 
                             fluorescence
                             default: 1000
            - dend (str)   : dendritic traces to use ('aibs' for the 
                             original extracted traces and 'extr' for the
                             ones extracted with Hakan's EXTRACT code, if
                             available)
                             default: 'extr'
        """
        self.dend = 'aibs'

        if self.plane == 'dend' and dend == 'extr':
            [roi_trace_h5, 
             roi_trace_h5_dff] = sess_file_util.get_dendritic_trace_paths(
                                                self.roi_trace_h5)
            
            if roi_trace_h5 == 'none':
                print('    No extr extracted dendrites found. AIBS '
                          'extracted dendrites will be used instead.')
            else:
                self.dend = 'extr'
                [self.roi_trace_h5, 
                 self.roi_trace_dff_h5] = roi_trace_h5, roi_trace_h5_dff

        self._load_dff(basewin=basewin)

        try:
            # open the roi file and get the info
            with h5py.File(self.roi_trace_h5, 'r') as f:
                # get the names of the rois
                try:
                    self.roi_names = f['roi_names'].value.tolist()
                except:
                    self.roi_names = None

                # get the number of rois
                self.nrois = f['data'].shape[0]

                # get the number of data points in the traces
                self.tot_twop_fr = f['data'].shape[1]

                # get standardization facts (factor x ROI)
                self.roi_facts = np.asarray(math_util.scale_facts(f['data'][()], 
                                            axis=1, sc_type='stand_rob', 
                                            extrem='perc', allow_0=True)[0:2])
    
        except:
            raise OSError(f'Could not open {self.roi_trace_h5} for reading.')

        # generate attribute listing ROIs with NaNs or Infs (for raw traces)
        self._set_nanrois('raw')


    #############################################
    def _load_stims(self):
        """
        self._load_stims()
        
        Initializes the following attributes, including Stim objects (Gabors, 
        Bricks, Grayscr):
            - bricks (list or Bricks object): session bricks object, if 
                                              runtype is 'pilot' or list
                                              of session bricks objects if 
                                              runtype is 'prod'
            - gabors (Gabors object)        : session gabors object
            - grayscr (Grayscr object)      : session grayscreen object
            - n_stims (int)                 : number of stimulus objects in
                                              the session (2 bricks stims
                                              in production data count as one)
            - stimtypes (list)              : list of stimulus type names 
                                              (i.e., 'gabors', 'bricks')
            - stims (list)                  : list of stimulus objects in the
                                              session
        """

        if hasattr(self, 'stimtypes'):
            return

        # create the stimulus fields and objects
        self.stimtypes = []
        self.n_stims    = len(self.stim_dict['stimuli'])
        self.stims      = []
        if self.runtype == 'prod':
            n_bri = []
        for i in range(self.n_stims):
            stim = self.stim_dict['stimuli'][i]
            if self.runtype == 'pilot':
                stimtype = stim['stimParams']['elemParams']['name']
            elif self.runtype == 'prod':
                stimtype = stim['stim_params']['elemParams']['name']
            # initialize a Gabors object
            if stimtype == 'gabors':
                self.stimtypes.append(stimtype)
                self.gabors = Gabors(self, i)
                self.stims.append(self.gabors)
            # initialize a Bricks object
            elif stimtype == 'bricks':
                if self.runtype == 'prod':
                    n_bri.append(i)
                    # 2 brick stimuli are recorded in the production data, but 
                    # are merged to initialize one stimulus object
                    if len(n_bri) == 2:
                        self.stimtypes.append(stimtype)
                        self.bricks = Bricks(self, n_bri)
                        self.stims.append(self.bricks)
                        self.n_stims = self.n_stims - 1
                        n_bri = []
                elif self.runtype == 'pilot':
                    self.stimtypes.append(stimtype)
                    self.bricks = Bricks(self, i)
                    self.stims.append(self.bricks)
                
            else:
                print(f'    {stimtype} stimulus type not recognized. No Stim ' 
                      'object created for this stimulus. \n')

        # initialize a Grayscr object
        self.grayscr = Grayscr(self)

    
    #############################################
    def extract_sess_attribs(self, mouse_df='mouse_df.csv'):
        """
        self.extract_sess_attribs(mouse_df)

        This function should be run immediately after creating a Session 
        object. It loads the dataframe containing information on each session,
        and sets the following attributes:

            - all_files (bool) : if True, all files have been acquired for
                                 the session
            - any_files (bool) : if True, some files have been acquired for
                                 the session
            - depth (int)      : recording depth 
            - plane (str)      : recording plane ('soma' or 'dend')
            - line (str)       : mouse line (e.g., 'L5-Rbp4')
            - mouse_n (int)    : mouse number (e.g., 1)
            - notes (str)      : notes from the dataframe on the session
            - pass_fail (str)  : whether session passed 'P' or failed 'F' 
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

        self.mouse_n      = df_data['mouse_n']
        self.depth        = df_data['depth']
        self.plane        = df_data['plane']
        self.line         = df_data['line']
        self.sess_gen     = df_data['sess_gen']
        self.sess_n       = df_data['sess_n']
        self.sess_within  = df_data['sess_within']
        self.pass_fail    = df_data['pass_fail']
        self.all_files    = df_data['all_files']
        self.any_files    = df_data['any_files']
        self.notes        = df_data['notes']


    #############################################
    def extract_info(self, fulldict=True, basewin=1000, dend='extr', roi=True, 
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
            self._load_sync_h5s()
            self._load_stims()

            optionally:
            self._load_roi_trace_info()
            self._load_run_data()
            self._load_pup_data()

        Optional args:
            - fulldict (bool): if True, the full stim_dict is loaded,
                               else the small stim_dict is loaded
                               (which contains everything, except 'posbyframe' 
                               for Bricks)
                               default: True
            - basewin (int)  : window length for calculating baseline 
                               fluorescence
                               default: 1000
            - dend (str)     : dendritic traces to use ('aibs' for the 
                               original extracted traces and 'extr' for the
                               ones extracted with Hakan's EXTRACT code, if
                               available)
                               default: 'extr'
            - roi (bool)     : if True, ROI data is loaded
                               default: True
            - run (bool)     : if True, running data is loaded
                               default: False
            - pup (bool)     : if True, pupil data is loaded
                               default: False
        """

        if not hasattr(self, 'plane'):
            raise ValueError('Session attributes missing to extract info. '
                             'Make sure to run self.extract_sess_attribs() '
                             'first')

        # load the stimulus, running, alignment and trace information 
        print('\nLoading stimulus dictionary...')
        self._load_stim_dict(fulldict=fulldict)
    
        print('Loading alignment dataframe...')
        self._load_align_df()
    
        if roi:
            print('Loading ROI trace info...')
            self._load_roi_trace_info(basewin=basewin, dend=dend)
        
        print('Loading sync h5 info...')
        self._load_sync_h5_data()

        print('Creating stimulus objects...')
        self._load_stims()

        if run:
            print('Loading running info...')
            self.load_run_data()

        if pupil:
            print('Loading pupil info...')
            self.load_pup_data()


    ############################################
    def extract_traces_from_masks(self, h5_dir=None, replace=False, 
                                  block_size=100, compression=None):
        """
        self.extract_traces_from_masks()

        Extracts traces from masks generated with Hakan's EXTRACT code (using 
        boolean masks) and saves them.

        Optional args:
            - h5_dir (str)     : path to full corrected twop data. If None, then 
                                 it should already be an attribute of the 
                                 Session object.
                                 default: None
            - replace (bool)   : if True, existing extracted EXTRACT traces are 
                                 replaced (and existing EXTRACT dF/F traces are 
                                 deleted)
                                 default: False
            - block_size (int) : number of frames to load and extract traces for
                                 at a time 
                                 default: 100
            - compression (str): type of compression to use when saving h5 file
                                 (e.g., 'gzip')
                                 default: None
        """

        act_names, prev_names = sess_file_util.get_dendritic_trace_paths(
                                               self.roi_trace_h5)
            
        for names in [act_names, prev_names]:
            if os.path.exists(names[0]) and not replace:
                print('\nExtract traces already exist')
                return
        
        extr_tr_file, extr_tr_dff_file = act_names
            
        # Raise error if not dendritic data
        if self.plane != 'dend':
            raise ValueError('Extracting traces from masks is meant to be '
                             'used with dendritic data, not somatic.')

        # Retrieve masks generated with EXTRACT
        self.masks = sess_file_util.get_mask_path(self.home, self.sessid, 
                                    self.expid, self.mouseid, self.runtype, 
                                    self.mouse_dir) 
        with h5py.File(self.masks, 'r') as f:
            masks = f['data'][:]

        # Get the full video information
        if h5_dir is not None:
            self.correct_data_h5 = h5_dir
        elif not hasattr(self, 'correct_data_h5'):
            raise ValueError('No path to full corrected twop data '
                             '(self.correct_data_h5).')

        with h5py.File(self.correct_data_h5, 'r') as f:
            _, wid, hei = f['data'].shape

        # extract ROI traces using masks
        all_masks = [] 
        border = [0, 0, 0, 0]
        for mask in masks:
            all_masks.append(roi_masks.create_roi_mask(wid, hei, border, 
                             roi_mask=mask.astype(bool)))

        with h5py.File(self.correct_data_h5, 'r') as f:
            traces = roi_masks.calculate_traces(f['data'], all_masks, 
                                                block_size=block_size)
        
        # Save traces
        with h5py.File(extr_tr_file, 'w') as hf:
            hf.create_dataset('data', data=traces, compression=compression)

        # Removes pre-existing EXTRACT dF/F file
        if os.path.exists(extr_tr_dff_file):
            print('\nRemoving pre-existing EXTRACT dF/F file.')
            os.remove(extr_tr_dff_file)
        

    #############################################
    def get_stim(self, stimtype='gabors'):
        """
        self.get_stim()

        Returns the requested Stim object, if it is an attribute of the 
        Session.

        Required args:
            - sess (Session): Session object

        Optional args:
            - stimtype (str): stimulus type to return ('bricks', 'gabors' or 
                              'grayscr')
                              default: 'gabors'

        Return:
            - stim (Stim): Stim object (either Gabors or Bricks)
        """


        if stimtype == 'gabors':
            if hasattr(self, 'gabors'):
                stim = self.gabors
            else:
                raise ValueError('Session object has no gabors stimulus.')
        elif stimtype == 'bricks':
            if hasattr(self, 'bricks'):
                stim = self.bricks
            else:
                raise ValueError('Session object has no bricks stimulus.')
        elif stimtype == 'grayscr':
            if hasattr(self, 'grayscr'):
                stim = self.grayscr
            else:
                raise ValueError('Session object has no grayscr stimulus.')
        else:
            gen_util.accepted_values_error('stimtype', stimtype, 
                                           ['gabors', 'bricks', 'grayscr'])
        return stim


    #############################################
    def get_pup_data(self, datatype='diam', remnans=True, stand=False):
        """
        self.get_pup_data()

        Returns the correct full pupil data array based on whether NaNs are
        to be removed or not. 

        Optional args:
            - datatype (str): type of pupil data to return ('diam', 'center_x', 
                              'center_y', 'center_diff')
                              default: 'diam'
            - remnans (bool): if True, the full pupil array in which NaN 
                              values have been removed using linear 
                              interpolation is returned. If False, the non
                              interpolated pupil array is returned.
                              default: True
            - stand (bool)  : if True, pupil data is standardized using full 
                              data array
                              default: False      

        Returns:
            - pup (nd array): full pupil data array (in pixels)
        """

        if datatype == 'diam':
            if remnans:
                pup = self.pup_diam_interp
                facts = self.pup_diam_interp_facts
            else:
                pup = self.pup_diam
                facts = self.pup_diam_facts
        elif datatype == 'center_x':
            if remnans:
                pup = self.pup_center_x_interp
                facts = self.pup_center_x_interp_facts
            else:
                pup = self.pup_center_x
                facts = self.pup_center_x_facts
        elif datatype == 'center_y':
            if remnans:
                pup = self.pup_center_y_interp
                facts = self.pup_center_y_interp_facts
            else:
                pup = self.pup_center_y
                facts = self.pup_center_y_facts
        elif datatype == 'center_diff':
            if remnans:
                pup = self.pup_center_diff_interp
                facts = self.pup_center_diff_interp_facts
            else:
                pup = self.pup_center_diff
                facts = self.pup_center_diff_facts
        else:
            gen_util.accepted_values_error('datatype', datatype, 
                     ['diam', 'center_x', 'center_y', 'center_diff'])

        if stand:
            pup = (pup - facts[0])/facts[1]

        return pup


    #############################################
    def get_run_velocity(self, remnans=True, stand=False):
        """
        self.get_run_velocity()

        Returns the correct full running velocity array based on whether NaNs are
        to be removed or not. 

        Optional args:
            - remnans (bool): if True, the full running array in which NaN 
                              values have been removed using linear 
                              interpolation is returned. If False, the non
                              interpolated running array is returned.
                              default: True
            - stand (bool)  : if True, running is standardized based on 
                              full trace array
                              default: False

        Returns:
            - run (nd array): full running velocity array (in cm/s)
        """

        if remnans:
            run = self.run_interp
            facts = self.run_interp_facts
        else:
            run = self.run
            facts = self.run_facts

        if stand:
            run = (run - facts[0])/facts[1]
        
        return run


    #############################################
    def get_run_velocity_by_fr(self, fr, fr_type='stim', remnans=True, 
                               stand=False):
        """
        self.get_run_velocity_by_fr(fr)

        Returns the running velocity for the given frames, either stimulus 
        frames or two-photon imaging frames using linear interpolation.

        Required args:
            - fr (array-like): set of frames for which to get running velocity
        
        Optional args:
            - fr_type (str) : type of frames passed ('stim' or 'twop' frames)
                              default: 'stim'
            - remnans (bool): if True, NaN values are removed using linear 
                              interpolation.
                              default: True
            - stand (bool)  : if True, running is standardized based on 
                              full trace array
                              default: False
        Returns:
            - velocity (nd array): running velocity (in cm/s), with same 
                                   dimensions as input array
        """

        fr = np.asarray(fr)

        if fr_type == 'stim':
            max_val = self.tot_run_fr
        elif fr_type == 'twop':
            max_val = self.tot_twop_fr
        else:
            gen_util.accepted_values_error('fr_type', fr_type, 
                                           ['stim', 'twop'])

        if (fr >= max_val).any() or (fr < 0).any():
                raise UserWarning('Some of the specified frames are out of '
                                  'range')
        
        run = self.get_run_velocity(remnans=remnans, stand=stand)

        if fr_type == 'stim':
            velocity = run[fr]
        elif fr_type == 'twop':
            velocity = np.interp(fr, self.stim2twopfr, run)

        return velocity


    #############################################
    def get_nanrois(self, fluor='dff'):
        """
        self.get_nanrois()

        Returns as a list the indices of ROIs containing NaNs or Infs.

        Optional args:
            - fluor (str): if 'dff', remnans is assessed on ROIs using dF/F 
                           traces. If 'raw', on raw traces.
                           default: 'dff'
        Returns:
            - (list): indices of ROIs containing NaNs or Infs
        """

        if fluor == 'dff' or self.dend == 'extr':
            return self.nanrois_dff
        elif fluor == 'raw':
            return self.nanrois
        else:
            gen_util.accepted_values_error('fluor', fluor, ['raw', 'dff'])


    #############################################
    def get_nrois(self, remnans=True, fluor='dff'):
        """
        self.get_nrois()

        Returns the number of ROIs according to the specified criteria.

        Optional args:
            - remnans (bool): if True, ROIs with NaN/Inf values are excluded
                              from number.
                              default: True
            - fluor (str)   : if 'dff', the indices of ROIs with NaNs or Infs 
                              in the dF/F traces are returned. If 'raw', for 
                              raw traces.
                              default: 'dff'
        Returns:
            - nrois (int): number of ROIs fitting criteria
        """

        nrois = self.nrois
        if remnans:
            rem_rois = len(self.get_nanrois(fluor))
            nrois = nrois - rem_rois

        return nrois
        

    #############################################
    def get_active_rois(self, fluor='dff', stimtype=None, remnans=True):
        """
        self.active_rois()

        Returns as a list the indices of ROIs that have calcium transients 
        (defined as median + 3 std), optionally during a specific stimulus type.

        Optional args:
            - fluor (str)   : if 'dff', the indices of ROIs with NaNs or Infs 
                              in the dF/F traces are returned. If 'raw', for 
                              raw traces.
                              default: 'dff'
            - stimtype (str): stimulus type during which to check for 
                              transients ('bricks', 'gabors' or None). If None,
                              the entire session is checked.
                              default: None
            - remnans (bool): if True, the indices ignore ROIs containg NaNs or 
                              Infs
                              default: True
        Returns:
            - active_rois (list): indices of active ROIs
        """

        print('\nIdentifying active ROIs.')

        win = [1, 5]
        
        full_data = self.get_roi_traces(None, fluor, remnans)
        full_data_sm = scsig.medfilt(full_data, win)

        if stimtype is None:
            stim_data = full_data
            stim_data_sm = full_data_sm
        else:
            stim = self.get_stim(stimtype)
            twop_fr = []
            all_ran = [ran for dispran in stim.block_ran_seg for ran in dispran]
            for bl in all_ran:
                stfr = self.stim_df.loc[(self.stim_df['stimType'] == stimtype[0]) &
                                        (self.stim_df['stimSeg'] == bl[0])]['start2pfr'].values[0]
                endfr = self.stim_df.loc[(self.stim_df['stimType'] == stimtype[0]) &
                                         (self.stim_df['stimSeg'] == bl[1]-1)]['end2pfr'].values[0]               
                twop_fr.extend(list(range(stfr, endfr + 1)))
            stim_data = self.get_roi_traces(twop_fr, fluor, remnans)
            stim_data_sm = scsig.medfilt(stim_data, win)

        med = np.nanmedian(full_data_sm, axis=1) # smooth full data median
        std = np.nanstd(full_data_sm, axis=1) # smooth full data std

        # count how many calcium transients occur in the data of interest for
        # each ROI and identify inactive ROIs
        diff = stim_data_sm - (med + 3 * std)[:, np.newaxis]
        counts = np.sum(diff > 0, axis=1)
        inactive = np.where(counts == 0)[0]
        n_rois = len(stim_data)
        active_rois = list(set(range(n_rois)) - set(inactive))

        return active_rois


    #############################################
    def get_roi_traces(self, frames=None, fluor='dff', remnans=True, 
                       basewin=1000, stand=False):
        """
        self.get_roi_traces()

        Returns the processed ROI traces for the given two-photon imaging
        frames and specified ROIs.

        Optional args:
            - frames (int array): set of 2p imaging frames to give ROI dF/F 
                                  for, if any frames are out of range then NaNs 
                                  returned. The order is not changed, so frames
                                  within a sequence should already be properly 
                                  sorted (likely ascending). If None, then all 
                                  frames are returned. 
                                  default: None
            - fluor (str)       : if 'dff', then traces are converted into dF/F
                                  before return, using a sliding window of 
                                  length basewin (see below). 
                                  default: 'dff'
            - remnans (bool)    : if True, ROIs with NaN/Inf values anywhere 
                                  in session are excluded. 
                                  default: True
            - basewin (int)     : window length for calculating baseline 
                                  fluorescence
                                  default: 1000
            - stand (bool)      : if True, each ROIs is standardized 
                                  based on full data array
                                  default: False
        Returns:
            - traces (float array): array of dF/F for the specified frames,
                                    (ROI x frames)
        """

        # check whether the frames to retrieve are within range
        if frames is None:
            frames = np.arange(self.tot_twop_fr)
        elif max(frames) >= self.tot_twop_fr or min(frames) < 0:
            raise UserWarning('Some of the specified frames are out of range')

        # initialize the return array
        traces    = np.full((self.nrois, len(frames)), np.nan)

        # read the data points into the return array
        if fluor == 'dff':
            roi_trace_h5 = self.roi_trace_dff_h5
        elif fluor == 'raw':
            roi_trace_h5 = self.roi_trace_h5
        else:
            gen_util.accepted_values_error('fluor', fluor, ['raw', 'dff'])

        with h5py.File(roi_trace_h5, 'r') as f:
            try:
                traces = f['data'].value[:,frames]
            except:
                raise OSError(f'Could not read {self.roi_trace_h5}')
        
        if stand:
            facts = self._get_roi_facts(fluor)
            traces = (traces - facts[0:1].T)/facts[1:2].T

        if remnans:
            rem_rois = self.get_nanrois(fluor)
            # remove ROIs with NaNs or Infs in full session traces
            traces = gen_util.remove_idx(traces, rem_rois, axis=0)

        return traces


    #############################################
    def get_twop_fr_ran(self, twop_ref_fr, pre, post):
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
         
        Returns:
            - num_ran (2D array): array of frame numbers, structured as:
                                      sequence x frames
            - xran (1D array)   : time values for the 2p frames
            
        """

        ran_fr = [np.around(x*self.twop_fps) for x in [-pre, post]]
        xran   = np.linspace(-pre, post, int(np.diff(ran_fr)[0]))

        if len(twop_ref_fr) == 0:
            raise ValueError('No frames: frames list must include at least 1 '
                             'frame.')

        if isinstance(twop_ref_fr[0], (list, np.ndarray)):
            raise ValueError('Frames must be passed as a 1D list, not by block.')

        # get sequences x frames
        fr_idx = gen_util.num_ranges(twop_ref_fr, pre=-ran_fr[0], 
                                     leng=len(xran))
                     
        # remove sequences with negatives or values above total number of stim 
        # frames
        neg_idx  = np.where(fr_idx[:,0] < 0)[0].tolist()
        over_idx = np.where(fr_idx[:,-1] >= self.tot_twop_fr)[0].tolist()
        
        num_ran = gen_util.remove_idx(fr_idx, neg_idx + over_idx, axis=0)

        if len(num_ran) == 0:
            raise ValueError('No frames: All frames were removed from list.')

        return num_ran, xran


    #############################################
    def get_roi_seqs(self, twop_fr_seqs, padding=(0,0), fluor='dff', 
                     remnans=True, basewin=1000, stand=False):
        """
        self.get_roi_seqs(twop_fr_seqs)

        Returns the processed ROI traces for the given stimulus sequences.
        Frames around the start and end of the sequences can be requested by 
        setting the padding argument.

        If the sequences are different lengths the array is nan padded

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
            - fluor (str)              : if 'dff', then traces are converted 
                                         into dF/F before return, using a 
                                         sliding window of length basewin 
                                         (see below). 
                                         default: 'dff'
            - remnans (bool)           : if True, ROIs with NaN/Inf values 
                                         anywhere in session are excluded. 
                                         default: True
            - basewin (int)            : window length for calculating baseline 
                                         fluorescence
                                         default: 1000
            - stand (bool)             : if True, each ROIs is standardized 
                                         based on full data array
                                         default: False

        Returns:
            - traces (3D array): array of traces for the specified 
                                 ROIs and sequences, structured as: 
                                 ROIs x sequences x frames
        """

        # extend values with padding
        if padding[0] != 0:
            min_fr       = np.asarray([min(x) for x in twop_fr_seqs])
            st_padd      = np.tile(np.arange(-padding[0], 0), 
                                   (len(twop_fr_seqs), 1)) + min_fr[:,None]
            twop_fr_seqs = [np.concatenate((st_padd[i], x)) 
                           for i, x in enumerate(twop_fr_seqs)]
        if padding[1] != 0:
            max_fr       = np.asarray([max(x) for x in twop_fr_seqs])
            end_padd     = np.tile(np.arange(1, padding[1]+1), 
                                   (len(twop_fr_seqs), 1)) + max_fr[:,None]
            twop_fr_seqs = [np.concatenate((x, end_padd[i])) 
                            for i, x in enumerate(twop_fr_seqs)]
        if padding[0] < 0 or padding[1] < 0:
            raise ValueError('Negative padding not supported.')

        # get length of each padded sequence
        pad_seql = np.array([len(s) for s in twop_fr_seqs])

        # flatten the sequences into one list of frames, removing any sequences
        # with unacceptable frame values (< 0 or > self.tot_twop_fr) 
        frames_flat = np.empty([sum(pad_seql)])
        last_idx    = 0
        seq_rem     = []
        seq_rem_l   = []
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
            print(f'\nSome of the specified frames for sequences {seq_rem} are '
                  'out of range so the sequence will not be included.')
            pad_seql     = np.delete(pad_seql, seq_rem)
            twop_fr_seqs = np.delete(twop_fr_seqs, seq_rem).tolist()

        # sanity check that the list is as long as expected
        if last_idx != len(frames_flat):
            if last_idx != len(frames_flat) - sum(seq_rem_l):
                raise ValueError(f'Concatenated frame array is {last_idx} long '
                                 'instead of expected'
                                 f' {len(frames_flat - sum(seq_rem_l))}.')
            else:
                frames_flat = frames_flat[: last_idx]

        # convert to int
        frames_flat = frames_flat.astype(int)

        # load the traces
        traces_flat = self.get_roi_traces(frames_flat.tolist(), fluor, 
                                          remnans, basewin, stand=stand)
        
        if remnans:
            nrois = self.nrois - len(self.get_nanrois(fluor))
        else:
            nrois = self.nrois

        # chop back up into sequences padded with Nans
        traces = np.full((nrois, len(twop_fr_seqs), max(pad_seql)), np.nan)

        last_idx = 0
        for i in range(len(twop_fr_seqs)):
            traces[:, i, 
                   :pad_seql[i]] = traces_flat[:, last_idx:last_idx+pad_seql[i]]
            last_idx += pad_seql[i]

        return traces


    #############################################
    def check_flanks(self, frs, ch_fl, fr_type='twop'):
        """
        self.check_flanks(self, frs, ch_fl)

        Required args:
            - frs (arraylike): list of frames values
            - ch_fl (list)   : flanks in sec [pre sec, post sec] around frames 
                               to check for removal if out of bounds

        Optional args:
            - fr_type (str): time of frames ('twop', 'stim')
                             default: 'twop'
        
        """

        if not isinstance(ch_fl, list) or len(ch_fl) != 2:
            raise ValueError('`ch_fl` must be a list of length 2.')

        if fr_type == 'twop':
            fps = self.twop_fps
            max_val = self.tot_twop_fr        
        elif fr_type == 'stim':
            fps = self.stim_fps
            max_val = self.tot_stim_fr
        elif fr_type == 'pup':
            fps = self.pup_fps
            max_val = self.tot_pup_fr
        else:
            gen_util.accepted_values_error('fr_type', fr_type, 
                                           ['twop', 'stim', 'pup'])

        ran_fr = [np.around(x * fps) for x in [-ch_fl[0], ch_fl[1]]]
        frs = np.asarray(frs)

        neg_idx  = np.where((frs + ran_fr[0]) < 0)[0].tolist()
        over_idx = np.where((frs + ran_fr[1]) >= max_val)[0].tolist()
        all_idx = set(range(len(frs))) - set(neg_idx + over_idx)

        if len(all_idx) == 0:
            frs = np.asarray([])
        else:
            frs = frs[np.asarray(sorted(all_idx))]

        return frs

    
    #############################################
    def get_pup_fr_by_twop_fr(self, twop_fr, ch_fl=None):
        """
        self.get_pup_fr_by_twop_fr(twop_fr)

        Returns pupil frames corresponding to 2p frames, taking into
        account the delay to display.

        Required args:
            - twop_fr (array-like): the 2p frame segments for which to get 
                                    pupil frames

        Optional args:
            - ch_fl (list): if provided, flanks in sec [pre sec, post sec] 
                            around frames to check for removal if out of bounds
                            default: None

        Returns:
            - pup_fr (array-like): the pupil frames corresponding to the 2p
                                   frames
        """
 
        if ch_fl is not None:
            firsts = self.check_flanks(firsts, ch_fl, fr_type='pup').tolist()

        # delay of ~0.1s to display on screen
        delay = int(np.round(self.twop_fps * 0.1))
        pup_fr = self.twop2pupfr[list(twop_fr)] + delay

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

        Initializes and returns a stimulus object, and the attributes below. 
        
        USE: Only initialize subclasses of Stim, not the stim class itself.

        Also calls
            self._set_blocks()
            self._set_stim_fr()
            self._set_twop_fr()

            and if stimulus is a bricks stimulus from the production data:
            self._check_brick_prod_params()

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
            - stimtype (str)              : 'gabors' or 'bricks'
            - surp_max_s (int)            : max duration of a surprise seq 
            - surp_min_s (int)            : min duration of a surprise seq

            if stimtype == 'gabors':
                - n_seg_per_set (int)     : number of segments per set (4)
            if stimtype == 'bricks' and sess.runtype == 'prod':
                - stim_n_all (list)       : both stim numbers
            
        Required args:
            - sess (Session object): session to which the stimulus belongs
            - stim_n (int or list) : number of stimulus in session pickle
                                     (2 numbers if production Bricks stimulus)
            - stimtype (str)      : type of stimulus ('gabors' or 'bricks')  

        """

        self.sess      = sess
        self.stimtype = stimtype
        self.stim_fps  = self.sess.stim_fps
        self.stim_n    = stim_n
    
        # for production Bricks, check that both stimulus dictionaries
        # are identical where necessary
        if self.sess.runtype == 'prod' and self.stimtype == 'bricks':
            self._check_brick_prod_params()
            self.stim_n_all = copy.deepcopy(self.stim_n)
            self.stim_n = self.stim_n[0]
        
        stim_info = self.sess.stim_dict['stimuli'][self.stim_n]

        # get segment parameters
        # seg is equivalent to a sweep, as defined in camstim 
        if self.sess.runtype == 'pilot':
            stim_par = stim_info['stimParams']
        if self.sess.runtype == 'prod':
            stim_par = stim_info['stim_params']

        if self.stimtype == 'gabors':
            params = 'gabor_params'
            dur_key = 'gab_dur'
            # segment length (sec) (0.3 sec)
            self.seg_len_s     = stim_par[params]['im_len'] 
            # num seg per set (4: A, B, C D/E)
            self.n_seg_per_set = stim_par[params]['n_im'] 
            if self.sess.runtype == 'pilot':
                # 2 blocks (1 per kappa) are expected.
                self.exp_n_blocks = 2 
            elif self.sess.runtype == 'prod':
                self.exp_n_blocks = 1
        elif self.stimtype == 'bricks':
            params = 'square_params'
            dur_key = 'sq_dur'
            # segment length (sec) (1 sec)
            self.seg_len_s     = stim_par[params]['seg_len']
            if self.sess.runtype == 'pilot':
                # 4 blocks (1 per direction/size) are expected.
                self.exp_n_blocks = 4 
            elif self.sess.runtype == 'prod':
                self.exp_n_blocks = 2
        else:
            raise ValueError(f'{self.stimtype} stim type not recognized. Stim '
                             'object cannot be initialized.')
        
        # blank period (i.e., 1 blank every _ segs)
        self.blank_per     = stim_info['blank_sweeps'] 
        # num seg per sec (blank segs count) 
        self.seg_ps_wibl   = 1/self.seg_len_s 
        # num seg per sec (blank segs do not count)
        if self.blank_per != 0:
            self.seg_ps_nobl = self.seg_ps_wibl * \
                               self.blank_per/(1. + self.blank_per) 
        else:
            self.seg_ps_nobl = self.seg_ps_wibl
        
        # sequence parameters
        # min duration of each surprise sequence (sec)
        self.surp_min_s  = stim_par[params]['surp_len'][0]
        # max duration of each surprise sequence (sec)
        self.surp_max_s  = stim_par[params]['surp_len'][1]
        # min duration of each regular sequence (sec)
        self.reg_min_s   = stim_par[params]['reg_len'][0]
        # max duration of each regular sequence (sec)
        self.reg_max_s   = stim_par[params]['reg_len'][1]

        # expected length of a block (sec) where an overarching parameter is 
        # held constant
        if self.sess.runtype == 'pilot':
            self.exp_block_len_s = stim_par[params]['block_len'] 
        elif self.sess.runtype == 'prod':
            self.exp_block_len_s = stim_par['session_params'][dur_key]
                                                                                                
        self._set_blocks()
        self._set_stim_fr()
        self._set_twop_fr()

    #############################################
    def _check_brick_prod_params(self):
        """
        self._check_brick_prod_params()

        Checks for Bricks production stimuli whether both specific components 
        of the stimulus dictionaries are identical. Specifically:
            ['stim_params']['elemParams']
            ['stim_params']['session_params']
            ['stim_params']['square_params']
            ['blank_sweeps']

        If differences are found, throws an error specifying which components
        are different.
        """

        if self.stimtype != 'bricks' or self.sess.runtype != 'prod':
            raise ValueError('Checking whether 2 stimulus dictionaries '
                             'contain the same parameters is only for '
                             'production Bricks stimuli.')
        
        stim_n = gen_util.list_if_not(self.stim_n)
        if len(stim_n) != 2:
            raise ValueError('Expected 2 stimulus numbers, '
                             f'but got {len(stim_n)}')
        
        stim_dict_1 = self.sess.stim_dict['stimuli'][self.stim_n[0]]
        stim_dict_2 = self.sess.stim_dict['stimuli'][self.stim_n[1]]
        
        error = False
        
        # check elemParams and square_params dictionaries
        diff_dicts = []
        overall_dict = 'stim_params'
        sub_dicts = ['elemParams', 'square_params']
        for dict_name in sub_dicts: 
            if (stim_dict_1[overall_dict][dict_name] != 
                stim_dict_2[overall_dict][dict_name]):
                diff_dicts.append(dict_name)
                error = True

        if error:
            diff_str = ', '.join(diff_dicts)
            dict_str = (f'\n- different values in the {diff_str} '
                        f'(under {overall_dict}).')
        else:
            dict_str = ''

        # check blank_sweeps
        if stim_dict_1['blank_sweeps'] != stim_dict_2['blank_sweeps']:
            error = True
            sweep_str = '\n- different values in the blank_sweeps key.'
        else:
            sweep_str = ''
        
        # check sq_dur
        if (stim_dict_1['stim_params']['session_params']['sq_dur'] !=
            stim_dict_2['stim_params']['session_params']['sq_dur']):
            error = True
            sq_str = ('\n- different values in the sq_dur key under '
                      'stim_params, session_params.')
        else:
            sq_str = ''

        if error:
            raise ValueError('Cannot initialize production Brick stimuli '
                             f'together, due to:{dict_str}{sweep_str}{sq_str}')


    #############################################
    def _set_blocks(self):
        """
        self._set_blocks

        Set attributes related to blocks and display sequences. Also checks
        whether expected number of blocks were shown and whether they 
        comprised the expected number of segments.

        NOTE: A block is a sequence of stimulus presentations of the same 
        stimulus type, and there can be multiple blocks in one experiment. 
        For Gabors, segments refer to each gabor frame (lasting 0.3 s). For 
        Bricks, segments refer to 1s of moving bricks. 
        
        NOTE: Grayscr segments are not omitted when a session's segments are 
        numbered.

            - act_n_blocks (int)          : actual number of blocks of the 
                                            stimulus
            - block_len_seg (nested list) : len of blocks in segments each 
                                            sublist contains the length of each 
                                            block for a display sequence,
                                            structured as 
                                                display sequence x block
            - block_ran_seg (nested lists): segment tuples (start, end) for 
                                            each block (end is EXCLUDED) each
                                            sublist contains display sequence
                                            list, structured as 
                                                display sequence x block x 
                                                [start, end] 
            - disp_seq (2D array)         : display start and end times in sec, 
                                            structured as 
                                                display sequence x [start, end]
            - extra_segs (int)            : number of additional segments shown,
                                            if any
        """

        stim_info        = self.sess.stim_dict['stimuli'][self.stim_n]
        
        self.disp_seq = stim_info['display_sequence'].tolist()
        
        if self.stimtype == 'bricks' and self.sess.runtype == 'prod':
            stim_info2    = self.sess.stim_dict['stimuli'][self.stim_n_all[1]]
            self.disp_seq = self.disp_seq + stim_info2['display_sequence'].tolist()

        tot_disp = int(sum(np.diff(self.disp_seq)))

        if self.stimtype == 'gabors':
            # block length is correct, as it was set to include blanks
            block_len = self.exp_block_len_s
        elif self.stimtype == 'bricks':
            # block length was not set to include blanks, so must be adjusted
            block_len = self.exp_block_len_s * float(self.seg_ps_wibl) \
                        /self.seg_ps_nobl

        # calculate number of blocks that started and checking whether it is as 
        # expected
        self.act_n_blocks = int(np.ceil(float(tot_disp)/block_len))
        self.extra_segs = 0
        if self.act_n_blocks != self.exp_n_blocks:
            print(f'    WARNING: {self.act_n_blocks} {self.stimtype} blocks '
                  f'started instead of the expected {self.exp_n_blocks}.')
            if self.act_n_blocks > self.exp_n_blocks:
                self.extra_segs = (float(tot_disp) - \
                                   self.exp_n_blocks*block_len)*self.seg_ps_wibl 
                print(f'    WARNING: In total, {self.extra_segs} '
                       'segments were shown, including blanks.')
    
        # calculate uninterrupted segment ranges for each block and check for 
        # incomplete or split blocks
        rem_sec_all         = 0
        self.block_ran_seg  = []
        start               = 0
        for i in range(len(self.disp_seq)):
            # useable length is reduced if previous block was incomplete
            length = np.diff(self.disp_seq)[i]-rem_sec_all
            n_bl = int(np.ceil(float(length)/block_len))
            rem_sec_all += float(n_bl)*block_len - length
            rem_seg = int(np.around((float(n_bl)*block_len - \
                                     length)*self.seg_ps_wibl))
            
            # collect block starts and ends (in segment numbers)
            temp = []
            for _ in range(n_bl-1):
                end = start + int(np.around(block_len*self.seg_ps_nobl))
                temp.append([start, end])
                start = end
            # 1 removed because last segment is a blank
            end = start + int(np.around(block_len*self.seg_ps_nobl)) - \
                  np.max([0, rem_seg-1])
            temp.append([start, end])
            self.block_ran_seg.append(temp)
            start = end + np.max([0, rem_seg-1])
            
            if rem_seg == 1:
                if i == len(self.disp_seq)-1:
                    print('    WARNING: During last sequence of '
                          f'{self.stimtype}, the last blank segment of the '
                          f'{n_bl}. block was omitted.')
                else:
                    print(f'    WARNING: During {i+1}. sequence of '
                          f'{self.stimtype}, the last blank segment of the '
                          f'{n_bl}. block was pushed to the start of the next '
                          'sequence.')
            elif rem_seg > 1:

                if i == len(self.disp_seq)-1:
                    print('    WARNING: During last sequence of '
                          f'{self.stimtype}, {rem_seg} segments (incl. blanks) '
                          f'from the {n_bl}. block were omitted.')
                else:
                    print(f'    WARNING: During {i+1}. sequence of '
                          f'{self.stimtype}, {rem_seg} segments (incl. blanks) '
                          f'from the {n_bl}. block were pushed to the next '
                          'sequence. These segments will be omitted from '
                          'analysis.')
            # get the actual length in segments of each block
            self.block_len_seg = np.diff(self.block_ran_seg).squeeze(2).tolist()


    #############################################
    def _set_stim_fr(self):
        """
        self._set_stim_fr()

        Sets attributes related to stimulus frames.

            - block_len_fr (nested lists): same as block_ran_len but in
                                           stimulus frame numbers instead,
                                           with the flanking grayscreens
                                           omitted
            - block_ran_fr (nested lists): same as block_ran_seg but in 
                                           stimulus frame numbers instead,
                                           with the flanking grayscreens
                                           omitted
            - stim_seg_list (list)       : full list of stimulus segment 
                                           numbers for each stimulus frame
        """

        stim_info = self.sess.stim_dict['stimuli'][self.stim_n]

        # n blank frames pre/post stimulus
        bl_fr_pre = int(self.sess.pre_blank*self.stim_fps)
        bl_fr_post = int(self.sess.post_blank*self.stim_fps)
        
        # recorded stimulus frames
        stim_fr = stim_info['frame_list'].tolist()

        # combine the stimulus frame lists
        if self.stimtype == 'bricks' and self.sess.runtype == 'prod':
            stim_info2 = self.sess.stim_dict['stimuli'][self.stim_n_all[1]]
            stim_fr2   = stim_info2['frame_list'].tolist()
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
        add_bl_fr = int(self.sess.tot_stim_fr - len(stim_fr))

        # fill out the stimulus segment list to be the same length as running 
        # array
        self.stim_seg_list = bl_fr_pre*[-1] + stim_fr + \
                             add_bl_fr*[-1] + bl_fr_post*[-1] 

        # (skip last element, since it is ignored in stimulus frames as well
        self.stim_seg_list = self.stim_seg_list[:-1]

        self.block_ran_fr = []
        for i in self.block_ran_seg:
            temp = []
            for j in i:
                # get first occurrence of first segment
                min_idx = self.stim_seg_list.index(j[0])
                max_idx = len(self.stim_seg_list)-1 - \
                              self.stim_seg_list[::-1].index(j[1]-1) + 1 
                temp.append([min_idx, max_idx])
            self.block_ran_fr.append(temp)
        
        self.block_len_fr = np.diff(self.block_ran_fr).squeeze(2).tolist()


    #############################################
    def _set_twop_fr(self):
        """
        self._set_twop_fr()

        Sets attributes related to twop frames.

            - block_len_twop_fr (nested lists): same as block_ran_len but in
                                                twop frame numbers instead,
                                                with the flanking grayscreens
                                                omitted
            - block_ran_twop_fr (nested lists): same as block_ran_seg but in 
                                                twop frame numbers instead,
                                                with the flanking grayscreens
                                                omitted
        """

        self.block_ran_twop_fr = []
        for i in self.block_ran_seg:
            temp = []
            for j in i:
                # get first occurrence of first segment
                try:
                    min_idx = self.sess.stim_df.loc[(self.sess.stim_df['stimType'] == self.stimtype[0]) &
                                                    (self.sess.stim_df['stimSeg'] == j[0])]['start2pfr'].tolist()[0]
                    max_idx = self.sess.stim_df.loc[(self.sess.stim_df['stimType'] == self.stimtype[0]) &
                                                    (self.sess.stim_df['stimSeg'] == j[1]-1)]['end2pfr'].tolist()[0] + 1
                except Exception as err:
                    print(err)
                    pdb.set_trace()
                # 1 added as range end is excluded
                
                temp.append([min_idx, max_idx])
            self.block_ran_twop_fr.append(temp)
        
        self.block_len_twop_fr = np.diff(self.block_ran_twop_fr).squeeze(2).tolist()


    #############################################
    def _format_stim_criteria(self, stimPar1='any', stimPar2='any', surp='any', 
                              stimSeg='any', gabfr='any', start2pfr='any', 
                              end2pfr='any', num2pfr='any', gabk=None, 
                              gab_ori=None, bri_size=None, bri_dir=None):
        """
        self._format_stim_criteria()

        Returns a list of stimulus parameters formatted correctly to use
        as criteria when searching through the stim dataframe. 

        Will strip any criteria not related to the current stim object.

        Optional args:
            - stimPar1 (str, int or list)  : stimPar1 value(s) of interest 
                                             (sizes: 128, 256, 
                                             oris: 0, 45, 90, 135)
                                             default: 'any'
            - stimPar2 (str, int or list)  : stimPar2 value(s) of interest 
                                             ('right', 'left', 4, 16)
                                             default: 'any'
            - surp (str, int or list)      : surp value(s) of interest (0, 1)
                                             default: 'any'
            - stimSeg (str, int or list)   : stimSeg value(s) of interest
                                             default: 'any'
            - gabfr (str, int or list)     : gaborframe value(s) of interest 
                                             (0, 1, 2, 3)
                                             default: 'any'
            - start2pfr (str or list)      : 2p start frames range of interest
                                             [min, max (excl)] 
                                             default: 'any'
            - end2pfr (str or list)        : 2p end frames range of interest
                                             [min, max (excl)]
                                             default: 'any'
            - num2pfr (str or list)        : 2p num frames range of interest
                                             [min, max (excl)]
                                             default: 'any'
            - gabk (int or list)           : if not None, will overwrite 
                                             stimPar2 (4, 16, or 'any')
                                             default: None
            - gab_ori (int or list)        : if not None, will overwrite 
                                             stimPar1 (0, 45, 90, 135, or 'any')
                                             default: None
            - bri_size (int or list)       : if not None, will overwrite 
                                             stimPar1 (128, 256, or 'any')
                                             default: None
            - bri_dir (str or list)        : if not None, will overwrite 
                                             stimPar2 ('right', 'left' or 'any')
                                             default: None
        
        Returns:
            - stimPar1 (list)    : stimPar1 value(s) of interest 
            - stimPar2 (list)    : stimPar2 value(s) of interest 
            - surp (list)        : surp value(s) of interest (0, 1)
            - stimSeg (list)     : stimSeg value(s) of interest
            - gabfr (list)       : gaborframe value(s) of interest 
            - start2pfr_min (int): minimum of 2p start2pfr range of interest 
            - start2pfr_max (int): maximum of 2p start2pfr range of interest 
                                   (excl)
            - end2pfr_min (int)  : minimum of 2p end2pfr range of interest
            - end2pfr_max (int)  : maximum of 2p end2pfr range of interest 
                                   (excl)
            - num2pfr_min (int)  : minimum of num2pfr range of interest
            - num2pfr_max (int)  : maximum of num2pfr range of interest 
                                   (excl)
        """

        # remove brick criteria for gabors and vv
        if self.stimtype == 'gabors':
            bri_size = None
            bri_dir = None
        elif self.stimtype == 'bricks':
            gabfr = None
            gabk = None
            gab_ori = None

        # if passed, replace StimPar1 and StimPar2 with the gabor and brick
        # arguments
        pars = [gabk, gab_ori, bri_size, bri_dir]
        stimpar_names = ['stimPar2', 'stimPar1', 'stimPar1', 'stimPar2']
        sp1 = []
        sp2 = []

        for i in range(len(pars)):
            if pars[i] == 'any':
                pars[i] = gen_util.get_df_vals(self.sess.stim_df, 'stimType', 
                                               self.stimtype, stimpar_names[i])
            if pars[i] is not None:
                pars[i] = gen_util.list_if_not(pars[i])
                if stimpar_names[i] == 'stimPar1':
                    sp1.extend(pars[i])
                elif stimpar_names[i] == 'stimPar2':
                    sp2.extend(pars[i])
        
        if len(sp1) != 0:
            stimPar1 = sp1
        if len(sp2) != 0:
            stimPar2 = sp2 

        # converts values to lists or gets all possible values, if 'any'
        stimPar1 = gen_util.get_df_label_vals(self.sess.stim_df, 
                                              'stimPar1', stimPar1)
        stimPar2 = gen_util.get_df_label_vals(self.sess.stim_df, 
                                              'stimPar2', stimPar2)
        surp     = gen_util.get_df_label_vals(self.sess.stim_df, 
                                              'surp', surp)
        stimSeg  = gen_util.get_df_label_vals(self.sess.stim_df, 
                                              'stimSeg', stimSeg)
        # here, ensure that the -1s are removed
        stimSeg = gen_util.remove_if(stimSeg, -1)
        gabfr   = gen_util.get_df_label_vals(self.sess.stim_df, 'gabfr', gabfr)
        
        if start2pfr in ['any', None]:
            start2pfr_min = int(self.sess.stim_df['start2pfr'].min())
            start2pfr_max = int(self.sess.stim_df['start2pfr'].max()+1)
        elif len(start2pfr) == 2:
            start2pfr_min, start2pfr_max = start2pfr
        else:
            raise ValueError('`start2pfr` must be of length 2 if passed.')

        if end2pfr in ['any', None]:
            end2pfr_min = int(self.sess.stim_df['end2pfr'].min())
            end2pfr_max = int(self.sess.stim_df['end2pfr'].max()+1)
        elif len(start2pfr) == 2:
            end2pfr_min, end2pfr_max = end2pfr
        else:
            raise ValueError('`end2pfr` must be of length 2 if passed.')

        if num2pfr in ['any', None]:
            num2pfr_min = int(self.sess.stim_df['num2pfr'].min())
            num2pfr_max = int(self.sess.stim_df['num2pfr'].max()+1)
        elif len(start2pfr) == 2:
            num2pfr_min, num2pfr_max = num2pfr
        else:
            raise ValueError('`num2pfr` must be of length 2 if passed.')

        return [stimPar1, stimPar2, surp, stimSeg, gabfr, start2pfr_min, 
                start2pfr_max, end2pfr_min, end2pfr_max, num2pfr_min, 
                num2pfr_max] 


    #############################################
    def get_stim_beh_sub_df(self, pre, post, stats='mean', fluor='dff', 
                            gabfr='any', gabk='any', gab_ori='any', 
                            bri_size='any', bri_dir='any', pupil=False, 
                            run=False):
        """
        self.get_stim_beh_sub_df()

        Returns an stimulus and behaviour dataframe for the specific stimulus 
        (gabors or bricks) with grayscreen rows added in if requested and 
        plane, line and sessid added in.

        Required args:
            - pre (num) : range of frames to include before each reference 
                          frame number (in s)
            - post (num): range of frames to include after each reference 
                          frame number (in s)

        Optional args:
            - fluor (str)           : if 'dff', dF/F is used, if 'raw', ROI 
                                      traces
                                      default: 'dff'
            - stats (str)           : statistic to use for baseline, mean 
                                      ('mean') or median ('median') (NaN values 
                                      are omitted)
                                      default: 'mean'
            - gabfr (int or list)   : 0, 1, 2, 3, 'gray', 'any'
            - gabk (int or list)    : 4, 16, or 'any'
                                      default: 'any'
            - gab_ori (int or list) : 0, 45, 90, 135, or 'any'
                                      default: 'any'
            - bri_size (int or list): 128, 256, or 'any'
                                      default: 'any'
            - bri_dir (str or list) : 'right', 'left' or 'any'
                                      default: 'any'
            - pupil (bool)          : if True, pupil data is added in
                                      default: False
            - run (bool)            : if True, run data is added in
                                      default: False

        Returns:
            - sub_df (pd DataFrame): extended stimulus dataframe containing
                                     grayscreen rows if requested, modified 
                                     column names, plane, line and sessid info 
        """

        retain = ['stimPar1', 'stimPar2', 'surp', 'stimSeg', 
                  'start2pfr', 'end2pfr']
        drop = ['stimSeg', 'start2pfr', 'end2pfr', 'start_stim_fr', 
                'end_stim_fr'] # drop at end

        ret_gabfr = gabfr
        if self.stimtype == 'gabors':
            retain.append('gabfr')
            get_gray = ('gray' in gen_util.list_if_not(gabfr) or \
                        gabfr in ['any', 'all'])
            if get_gray:
                ret_gabfr = 'any'
    
        sub_df = self.get_stim_df_by_criteria(gabfr=ret_gabfr, gabk=gabk, 
                      gab_ori=gab_ori, bri_size=bri_size, 
                      bri_dir=bri_dir)[retain]
        
        start_stim_fr, end_stim_fr = self.get_stim_fr_by_seg(sub_df['stimSeg'], 
                                          first=True, last=True)
        sub_df['start_stim_fr'] = start_stim_fr
        sub_df['end_stim_fr'] = end_stim_fr
        if self.stimtype == 'gabors':
            sub_df = sub_df.rename(columns={'stimPar1': 'gab_ori', 
                                            'stimPar2': 'gabk'})
            if get_gray:
                sub_df = sess_data_util.add_grayscreen_rows_gabors(sub_df)
                gabfr_vals = gen_util.get_df_label_vals(sub_df, 'gabfr', gabfr)
                sub_df = sub_df.loc[(sub_df['gabfr'].isin(gabfr_vals))]
        elif self.stimtype == 'bricks':
            sub_df = sub_df.rename(columns={'stimPar1': 'bri_size', 
                                            'stimPar2': 'bri_dir'})
        else:
            raise NotImplementedError('Extended stimulus subdataframe only '
                     'implemented for Gabor and Brick stimuli, '
                     f'not `{self.stimtype}`.')
        
        sub_df = sub_df.reset_index(drop=True) # reset index
        sub_df['plane'] = self.sess.plane
        sub_df['line']  = self.sess.line
        sub_df['sessid'] = self.sess.sessid

        if pupil:
            pup_fr = self.sess.get_pup_fr_by_twop_fr(
                               sub_df['start2pfr'].tolist())
            pup_data = self.get_pup_diam_array(pup_fr, pre, post, 
                                               remnans=True)[1]
            sub_df['pup_diam_data'] = math_util.mean_med(pup_data, stats=stats, 
                                                         axis=-1)
        if run:
            run_data = self.get_run_array(sub_df['start_stim_fr'].tolist(), 
                                          pre, post, remnans=True)[1]
            sub_df['run_data'] = math_util.mean_med(run_data, stats=stats, 
                                                    axis=-1)
        # add ROI data
        roi_data = self.get_roi_trace_array(sub_df['start2pfr'].tolist(), 
                        pre, post, remnans=True, fluor=fluor)[1]
        roi_data = math_util.mean_med(roi_data, stats=stats, 
                                      axis=-1, nanpol='omit')
        cols = [f'roi_data_{i}' for i in range(len(roi_data))]
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
            - last (bool) : instead returns the last for each seg.
                            default: False
            - ch_fl (list): if provided, flanks in sec [pre sec, post sec] 
                            around frames to check for removal if out of bounds
                            default: None

        Returns:
            if first and last are True:
                - frames (nested list): list of the first and last stim frames 
                                        numbers for each segment, structured
                                        as [first, last]
            if first or last is True, but not both:
                - frames (array-like) : a list of first or last stim frames 
                                        numbers for each segment
            else:
                - frames (list of int arrays): a list (one entry per segment) 
                                               of arrays containing the stim 
                                               frame
        """

        stim_seg_list_array = np.asarray(self.stim_seg_list)
        
        all_fr = [np.where(stim_seg_list_array == val)[0].tolist() 
                                                for val in seglist]

        firsts = [fr[0] for fr in all_fr]

        if ch_fl is not None:
            firsts = self.sess.check_flanks(firsts, ch_fl, 
                                            fr_type='stim').tolist()

        frames = []
        if first:
            frames.append(firsts)
        if last:
            frames.append([fr[-1] for fr in all_fr])

        if first or last:
            if not (first and last):
                frames = np.asarray(frames[0])
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
            - last (bool) : instead return last frame for each seg
                            default: False
            - ch_fl (list): if provided, flanks in sec [pre sec, post sec] 
                            around frames to check for removal if out of bounds
                            default: None

        Returns:
            if first and last are True:
                - frames (nested list): list of the first and last 2p frames 
                                        numbers for each segment, structured
                                        as [first, last]
            if first or last is True, but not both:
                - frames (array-like) : a list of first or last 2p frames 
                                        numbers for each segment
            else:
                - frames (list of int arrays): a list (one entry per segment) 
                                               of arrays containing the 2p 
                                               frame
        """

        # initialize the frames list
        frames = []

        # get the rows in the alignment dataframe that correspond to the segments
        rows = self.sess.stim_df.loc[(self.sess.stim_df['stimType'] == self.stimtype[0]) &
                                      (self.sess.stim_df['stimSeg'].isin(seglist))]

        # get the start frames and end frames from each row
        start2pfrs = rows['start2pfr'].values


        if ch_fl is not None:
            start2pfrs = self.sess.check_flanks(start2pfrs, ch_fl, fr_type='twop')

        if not first or last:
            end2pfrs = rows['end2pfr'].values

        if not first and not last:
            # build arrays for each segment
            for r in range(start2pfrs.shape[0]):
                frames.append(np.arange(start2pfrs[r], end2pfrs[r]))
        else:
            if first:
                frames.append(start2pfrs)
            if last:
                frames.append(end2pfrs)
            frames = gen_util.delist_if_not(frames)


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
        n_fr = self.sess.stim_df.loc[(self.sess.stim_df['stimType'] == self.stimtype[0]) &
                                      (self.sess.stim_df['stimSeg'].isin(segs_unique))]['num2pfr'].tolist()
        
        # resort based on order in which segs were passed and include any 
        # duplicates
        n_fr_sorted = [n_fr[segs_unique.index(seg)] for seg in segs]
        
        return n_fr_sorted


    #############################################
    def get_stim_df_by_criteria(self, stimPar1='any', stimPar2='any', 
                                surp='any', stimSeg='any', gabfr='any', 
                                start2pfr='any', end2pfr='any', 
                                num2pfr='any', gabk=None, gab_ori=None, 
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
                                             default: 'any'
            - stimPar2 (str, int or list)  : stimPar2 value(s) of interest 
                                             ('right', 'left', 4, 16)
                                             default: 'any'
            - surp (str, int or list)      : surp value(s) of interest (0, 1)
                                             default: 'any'
            - stimSeg (str, int or list)   : stimSeg value(s) of interest
                                             default: 'any'
            - gabfr (str, int or list)     : gaborframe value(s) of interest 
                                             (0, 1, 2, 3)
                                             default: 'any'
            - start2pfr (str or list)      : 2p start frames range of interest
                                             [min, max (excl)] 
                                             default: 'any'
            - end2pfr (str or list)        : 2p end frames range of interest
                                             [min, max (excl)]
                                             default: 'any'
            - num2pfr (str or list)        : 2p num frames range of interest
                                             [min, max (excl)]
                                             default: 'any'
            - gabk (int or list)           : if not None, will overwrite 
                                             stimPar2 (4, 16, or 'any')
                                             default: None
            - gab_ori (int or list)        : if not None, will overwrite 
                                             stimPar1 (0, 45, 90, 135, or 'any')
                                             default: None
            - bri_size (int or list)       : if not None, will overwrite 
                                             stimPar1 (128, 256, or 'any')
                                             default: None
            - bri_dir (str or list)        : if not None, will overwrite 
                                             stimPar2 ('right', 'left' or 'any')
                                             default: None
        
        Returns:
            - sub_df (pd DataFrame): subset of the stimulus dataframe 
                                     fitting the criteria provided
        """

        pars = self._format_stim_criteria(stimPar1, stimPar2, surp, 
                                stimSeg, gabfr, start2pfr, end2pfr, 
                                num2pfr, gabk, gab_ori, bri_size, bri_dir)

        [stimPar1, stimPar2, surp, stimSeg, gabfr, start2pfr_min, 
         start2pfr_max, end2pfr_min, end2pfr_max, num2pfr_min, 
         num2pfr_max] = pars

        sub_df = self.sess.stim_df.loc[(self.sess.stim_df['stimType']==self.stimtype[0])     & 
                                       (self.sess.stim_df['stimPar1'].isin(stimPar1))        &
                                       (self.sess.stim_df['stimPar2'].isin(stimPar2))        &
                                       (self.sess.stim_df['surp'].isin(surp))                &
                                       (self.sess.stim_df['stimSeg'].isin(stimSeg))          &
                                       (self.sess.stim_df['gabfr'].isin(gabfr))              &
                                       (self.sess.stim_df['start2pfr'] >= start2pfr_min) &
                                       (self.sess.stim_df['start2pfr'] < start2pfr_max)  &
                                       (self.sess.stim_df['end2pfr'] >= end2pfr_min)     &
                                       (self.sess.stim_df['end2pfr'] < end2pfr_max)      &
                                       (self.sess.stim_df['num2pfr'] >= num2pfr_min)   &
                                       (self.sess.stim_df['num2pfr'] < num2pfr_max)]
        
        return sub_df


    #############################################
    def get_segs_by_criteria(self, stimPar1='any', stimPar2='any', surp='any', 
                             stimSeg='any', gabfr='any', start2pfr='any', 
                             end2pfr='any', num2pfr='any', gabk=None, 
                             gab_ori=None, bri_size=None, bri_dir=None, 
                             remconsec=False, by='block'):
        """
        self.get_segs_by_criteria()

        Returns a list of stimulus seg numbers that have the specified values 
        in specified columns in the stimulus dataframe.    

        Will return segs only for the current stim object.

        Optional args:
            - stimPar1 (str, int or list)  : stimPar1 value(s) of interest 
                                             (sizes: 128, 256, 
                                             oris: 0, 45, 90, 135)
                                             default: 'any'
            - stimPar2 (str, int or list)  : stimPar2 value(s) of interest 
                                             ('right', 'left', 4, 16)
                                             default: 'any'
            - surp (str, int or list)      : surp value(s) of interest (0, 1)
                                             default: 'any'
            - stimSeg (str, int or list)   : stimSeg value(s) of interest
                                             default: 'any'
            - gabfr (str, int or list)     : gaborframe value(s) of interest 
                                             (0, 1, 2, 3)
                                             default: 'any'
            - start2pfr (str or list)      : 2p start frames range of interest
                                             [min, max (excl)] 
                                             default: 'any'
            - end2pfr (str or list)        : 2p end frames range of interest
                                             [min, max (excl)]
                                             default: 'any'
            - num2pfr (str or list)        : 2p num frames range of interest
                                             [min, max (excl)]
                                             default: 'any'
            - gabk (int or list)           : if not None, will overwrite 
                                             stimPar2 (4, 16, or 'any')
                                             default: None
            - gab_ori (int or list)        : if not None, will overwrite 
                                             stimPar1 (0, 45, 90, 135, or 'any')
                                             default: None
            - bri_size (int or list)       : if not None, will overwrite 
                                             stimPar1 (128, 256, or 'any')
                                             default: None
            - bri_dir (str or list)        : if not None, will overwrite 
                                             stimPar2 ('right', 'left' or 'any')
                                             default: None
            - remconsec (bool)             : if True, consecutive segments are 
                                             removed within a block
                                             default: False
            - by (str)                     : determines whether segment numbers
                                             are returned in a flat list 
                                             ('seg'), grouped by block 
                                             ('block'), or further grouped by 
                                             display sequence ('disp')
                                             default: 'block'
        
        Returns:
            - segs (list): list of seg numbers that obey the criteria
        """

        pars = self._format_stim_criteria(stimPar1, stimPar2, surp, 
                                stimSeg, gabfr, start2pfr, end2pfr, 
                                num2pfr, gabk, gab_ori, bri_size, bri_dir)

        [stimPar1, stimPar2, surp, stimSeg, gabfr, start2pfr_min, 
         start2pfr_max, end2pfr_min, end2pfr_max, num2pfr_min, 
         num2pfr_max] = pars
        
        segs = []
        for i in self.block_ran_seg:
            temp = []
            for j in i:
                idxs = self.sess.stim_df.loc[(self.sess.stim_df['stimType']==self.stimtype[0])     & 
                                             (self.sess.stim_df['stimPar1'].isin(stimPar1))        &
                                             (self.sess.stim_df['stimPar2'].isin(stimPar2))        &
                                             (self.sess.stim_df['surp'].isin(surp))                &
                                             (self.sess.stim_df['stimSeg'].isin(stimSeg))          &
                                             (self.sess.stim_df['gabfr'].isin(gabfr))              &
                                             (self.sess.stim_df['start2pfr'] >= start2pfr_min) &
                                             (self.sess.stim_df['start2pfr'] < start2pfr_max)  &
                                             (self.sess.stim_df['end2pfr'] >= end2pfr_min)     &
                                             (self.sess.stim_df['end2pfr'] < end2pfr_max)      &
                                             (self.sess.stim_df['num2pfr'] >= num2pfr_min)   &
                                             (self.sess.stim_df['num2pfr'] < num2pfr_max)    &
                                             (self.sess.stim_df['stimSeg'] >= j[0])                &
                                             (self.sess.stim_df['stimSeg'] < j[1])]['stimSeg'].tolist()
                
                # if removing consecutive values
                if remconsec and len(idxs) != 0: 
                    idx_keep = np.where(np.insert(np.diff(idxs), 0, 4) > 1)[0]
                    idxs = (np.asarray(idxs)[idx_keep]).tolist()
                # check for empty
                if len(idxs) != 0:
                    temp.append(idxs)
            # check for empty      
            if len(temp) != 0:
                segs.append(temp)
        
        # check for empty
        if len(segs) == 0:
             raise ValueError('No segments fit these criteria.')

        # if not returning by disp
        if by == 'block' or by == 'seg':
            segs = [x for sub in segs for x in sub]
            if by == 'seg':
                segs = [x for sub in segs for x in sub]
        elif by != 'disp':
            gen_util.accepted_values_error('by', by, ['block', 'disp', 'seg'])
        
        return segs


    #############################################
    def get_stim_fr_by_criteria(self, stimPar1='any', stimPar2='any', 
                                surp='any', stimSeg='any', gabfr='any', 
                                start2pfr='any', end2pfr='any', 
                                num2pfr='any', gabk=None, gab_ori=None, 
                                bri_size=None, bri_dir=None, first_fr=True, 
                                remconsec=False, by='block'):
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
                                             default: 'any'
            - stimPar2 (str, int or list)  : stimPar2 value(s) of interest 
                                             ('right', 'left', 4, 16)
                                             default: 'any'
            - surp (str, int or list)      : surp value(s) of interest (0, 1)
                                             default: 'any'
            - stimSeg (str, int or list)   : stimSeg value(s) of interest
                                             default: 'any'
            - gabfr (str, int or list)     : gaborframe value(s) of interest 
                                             (0, 1, 2, 3)
                                             default: 'any'
            - start2pfr (str or list)      : 2p start frames range of interest
                                             [min, max (excl)] 
                                             default: 'any'
            - end2pfr (str or list)        : 2p end frames range of interest
                                             [min, max (excl)]
                                             default: 'any'
            - num2pfr (str or list)        : 2p num frames range of interest
                                             [min, max (excl)]
                                             default: 'any'         
            - gabk (int or list)           : if not None, will overwrite 
                                             stimPar2 (4, 16, or 'any')
                                             default: None
            - gab_ori (int or list)        : if not None, will overwrite 
                                             stimPar1 (0, 45, 90, 135, or 'any')
                                             default: None
            - bri_size (int or list)       : if not None, will overwrite 
                                             stimPar1 (128, 256, or 'any')
                                             default: None
            - bri_dir (str or list)        : if not None, will overwrite 
                                             stimPar2 ('right', 'left' or 'any')
                                             default: None
            - remconsec (bool)               if True, consecutive segments are 
                                             removed within a block
                                             default: False
            - by (str)                     : determines whether frame numbers 
                                             are returned in a flat list 
                                             ('frame'), grouped by block 
                                             ('block'), or further grouped by 
                                             display sequence ('disp')
                                             default: 'block'
        
        Returns:
            - frames (list): list of stimulus frame numbers that obey the 
                             criteria
        """


        segs = self.get_segs_by_criteria(stimPar1, stimPar2, surp, stimSeg, 
                                         gabfr, start2pfr, end2pfr, 
                                         num2pfr, gabk, gab_ori, bri_size, 
                                         bri_dir, remconsec, by='disp')

        frames = []
        for i in segs:
            temp = []
            for idxs in i:
                temp2 = self.get_stim_fr_by_seg(idxs, first=first_fr)
                temp2 = [val for vals in temp2 for val in vals] # flatten
                # check for empty
                if len(temp2) != 0:
                    temp.append(temp2)
            # check for empty      
            if len(temp) != 0:
                frames.append(temp)
        
        # check for empty
        if len(frames) == 0:
             raise ValueError('No segments fit these criteria.')

        # if not returning by disp
        if by == 'block' or by == 'frame':
            frames = [x for sub in frames for x in sub]
            if by == 'frame':
                frames = [x for sub in frames for x in sub]
        elif by != 'disp':
            gen_util.accepted_values_error('by', by, ['block', 'disp', 'frame'])
        
        return frames


    #############################################
    def get_first_surp_segs(self, by='block'):
        """
        self.get_first_surp_segs()

        Returns two lists of stimulus segment numbers, the first is a list of 
        all the first surprise segments for the stimulus type at transitions 
        from regular to surprise sequences. The second is a list of all the 
        first regular segements for the stimulus type at transitions from 
        surprise to regular sequences.

        Optional args:
            - by (str): determines whether segment numbers are returned in a 
                        flat list ('seg'), grouped by block ('block'), or 
                        further grouped by display sequence ('disp')
                        default: 'block'

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
    def get_all_surp_segs(self, by='block'):
        """
        self.get_all_surp_segs()

        Returns two lists of stimulus segment numbers. The first is a list of 
        all the surprise segments for the stimulus type. The second is a list 
        of all the regular segments for the stimulus type.

        Optional args:
            - by (str): determines whether segment numbers are returned in a 
                        flat list ('seg'), grouped by block ('block'), or 
                        further grouped by display sequence ('disp')
                        default: 'block'

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
    def get_first_surp_stim_fr_trans(self, by='block'):
        """
        self.get_first_surp_stim_fr_trans()

        Returns two lists of stimulus frame numbers, the first is a list of all 
        the first surprise frames for the stimulus type at transitions from 
        regular to surprise sequences. The second is a list of all the first 
        regular frames for the stimulus type at transitions from surprise to 
        regular sequences.

        Optional args:
            - by (str): determines whether frames are returned in a flat list 
                        ('frame'), grouped by block ('block'), or further 
                        grouped by display sequence ('disp')
                        default: 'block'
        
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
    def get_all_surp_stim_fr(self, by='block'):
        """
        self.get_all_surp_stim_fr()

        Returns two lists of stimulus frame numbers, the first is a list of all 
        surprise frames for the stimulus type. The second is a list of all 
        regular frames for the stimulus type.

        Optional args:
            - by (str): determines whether frame numbers are returned in a flat 
                        list ('frame'), grouped by block ('block'), or further 
                        grouped by display sequence ('disp')
                        default: 'block'

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
    def get_array_stats(self, xran, data, ret_arr=False, axes=0, stats='mean', 
                        error='std', integ=False, nanpol=None):
        """
        self.get_array_stats(xran, data)

        Returns stats (mean and std or median and quartiles) for arrays of 
        running or roi traces. If sequences of unequal length are passed, they 
        are cut down to the same length, including xran.

        Required args:
            - xran (array-like)  : x values for the frames
            - data (nested lists): list of datapoints, (no more than 3 dim),
                                   structured as 
                                       dim x (dim x) (frames if not integ)
            
        Optional args:
            - ret_arr (bool)    : also return data array, not just statistics 
                                  default: False 
            - axes (int or list): axes along which to  take statistics. If a 
                                  list is passed.
                                  If None, axes are ordered reverse 
                                  sequentially (-1 to 1).
                                  default: 0
            - stats (str)       : return mean ('mean') or median ('median')
                                  default: 'mean'
            - error (str)       : return std dev/quartiles ('std') or SEM/MAD 
                                  ('sem')
                                  default: 'sem'
            - integ (bool)      : if True, data is integrated across frames
                                  default: False
            - nanpol (str)      : policy for NaNs, 'omit' or None
                                  default: None
         
        Returns:
            - xran (array-like)         : x values for the frames
                                          (length is equal to last data 
                                          dimension) 
            - data_stats (2 to 3D array): array of data statistics, structured 
                                          as: stats [me, err] (x dim) x frames
            if ret_arr, also:
            - data_array (2 to 3D array): data array, structured as:
                                              dim x (dim x) frames
        """

        if integ:
            data_array = np.asarray(data)
        
        # find minimum sequence length and cut all sequences down
        else:
            all_leng = []
            for subdata in data:
                if isinstance(subdata[0], (list, np.ndarray)):
                    for subsub in subdata:
                        all_leng.append(len(subsub))
                else:
                    all_leng.append(len(subdata))

            len_fr = np.min(all_leng + [len(xran)])
            xran   = xran[:len_fr]

            if np.all(all_leng == len_fr):
                data_array = np.asarray(data)
            else:
                # cut data down to minimum sequence length  
                data_array = []
                for subdata in data:
                    if isinstance(subdata[0], (list, np.ndarray)):
                        sub_array = []
                        for subsub in subdata:
                            sub_array.append(subsub[:len_fr])
                        data_array.append(sub_array)
                    else:
                        data_array.append(subdata[:len_fr])
                data_array = np.asarray(data_array)
        
        data_stats = math_util.get_stats(data_array, stats, error, axes=axes, 
                                         nanpol=nanpol)
     
        if ret_arr:
            return xran, data_stats, data_array
        else:
            return xran, data_stats


    #############################################
    def get_pup_diam_array(self, pup_ref_fr, pre, post, integ=False, 
                           remnans=False, baseline=None, stats='mean', 
                           stand=False):
        """
        self.get_pup_diam_array(pup_ref_fr, pre, post)

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
            - baseline (num)  : number of seconds to use as baseline. If None,
                                data is not baselined.
                                default: None
            - stats (str)     : statistic to use for baseline, mean ('mean') or 
                                median ('median') (NaN values are omitted)
                                default: 'mean'
            - stand (bool)    : if True, pupil diameter is standardized using 
                                full data array
                                default: False      

        Returns:
            - xran (1D array)           : time values for the stimulus frames
            - data_array (1 to 2D array): running data array, structured as:
                                          sequences (x frames)
        """

        if not hasattr(self.sess, 'pup_nan_diam'):
            self.sess._load_pup_data()

        ran_fr = [np.around(x*self.sess.pup_fps) for x in [-pre, post]]
        xran  = np.linspace(-pre, post, int(np.diff(ran_fr)[0]))

        if isinstance(pup_ref_fr[0], (list, np.ndarray)):
            raise OSError('Frames must be passed as a 1D list, not by block.')

        # get corresponding running subblocks sequences x frames
        fr_idx = gen_util.num_ranges(pup_ref_fr, pre=-ran_fr[0], 
                                     leng=len(xran))
                     
        # remove sequences with negatives or values above total number of stim 
        # frames
        neg_idx  = np.where(fr_idx[:,0] < 0)[0].tolist()
        over_idx = np.where(fr_idx[:,-1] >= self.sess.tot_pup_fr)[0].tolist()
        
        fr_idx = gen_util.remove_idx(fr_idx, neg_idx + over_idx, axis=0)

        pup_data = self.sess.get_pup_data(datatype='diam', remnans=remnans, 
                             stand=stand)
        data_array = pup_data[fr_idx]

        if remnans:
            nanpol = None 
        else:
            nanpol = 'omit'

        if baseline is not None:
            baseline_fr = int(np.around(baseline * self.sess.pup_fps))
            baseline_data = data_array[:, : baseline_fr]
            data_array_base = math_util.mean_med(baseline_data, stats=stats, 
                                                 axis=-1, 
                                                 nanpol=nanpol)[:, np.newaxis]
            data_array = data_array - data_array_base

        if integ:
            data_array = math_util.integ(data_array, 1./self.sess.pup_fps, 
                                         axis=1, nanpol=nanpol)

        return xran, data_array


    #############################################
    def get_pup_diam_stats(self, pup_ref_fr, pre, post, integ=False, 
                           remnans=False, ret_arr=False, stats='mean', 
                           error='std', baseline=None, stand=False):
        """
        self.get_pup_diam_stats(pup_ref_fr, pre, post)

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
            - stats (str)     : return mean ('mean') or median ('median')
                                default: 'mean'
            - error (str)     : return std dev/quartiles ('std') or SEM/MAD 
                                ('sem')
                                default: 'sem'
            - baseline (num)  : number of seconds to use as baseline. If None,
                                data is not baselined.
                                default: None
            - stand (bool)    : if True, pupil diameter is standardized using 
                                full data array
                                default: False      

        Returns:
            - xran (1D array)           : time values for the pupil frames 
                                          (length is equal to last data 
                                          dimension) 
            - data_stats (1 to 2D array): array of pupil diameter statistics, 
                                          structured as:
                                             stats [me, err] (x frames)
            if ret_arr, also:
            - data_array (1 to 2D array): puil diameter data array, structured 
                                          as: sequences (x frames)
        """

        xran, data_array = self.get_pup_diam_array(pup_ref_fr, pre, post, 
                                integ, remnans=remnans, baseline=baseline, 
                                stats=stats, stand=stand)

        if remnans:
            nanpol = None 
        else:
            nanpol = 'omit'

        all_data = self.get_array_stats(xran, data_array, ret_arr, axes=0, 
                                        stats=stats, error=error, integ=integ, 
                                        nanpol=nanpol)

        if ret_arr:
            xran, data_stats, data_array = all_data
            return xran, data_stats, data_array
        
        else:
            xran, data_stats = all_data
            return xran, data_stats


    #############################################
    def get_run_array(self, stim_ref_fr, pre, post, integ=False, remnans=True, 
                      baseline=None, stats='mean', stand=False):
        """
        self.get_run_array(stim_ref_fr, pre, post)

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
            - baseline (num)  : number of seconds to use as baseline. If None,
                                data is not baselined.
                                default: None
            - stats (str)     : statistic to use for baseline, mean ('mean') or 
                                median ('median')
                                default: 'mean'
            - stand (bool)    : if True, each ROI is standardized based on 
                                full trace array
                                default: False            
        Returns:
            - xran (1D array)           : time values for the stimulus frames
            - data_array (1 to 2D array): running data array, structured as:
                                          sequences (x frames)
        """

        ran_fr = [np.around(x*self.stim_fps) for x in [-pre, post]]
        xran  = np.linspace(-pre, post, int(np.diff(ran_fr)[0]))

        if isinstance(stim_ref_fr[0], (list, np.ndarray)):
            raise OSError('Frames must be passed as a 1D list, not by block.')

        # get corresponding running subblocks sequences x frames
        fr_idx = gen_util.num_ranges(stim_ref_fr, pre=-ran_fr[0], 
                                     leng=len(xran))
                     
        # remove sequences with negatives or values above total number of stim 
        # frames
        neg_idx  = np.where(fr_idx[:,0] < 0)[0].tolist()
        over_idx = np.where(fr_idx[:,-1] >= self.sess.tot_run_fr)[0].tolist()
        
        fr_idx = gen_util.remove_idx(fr_idx, neg_idx + over_idx, axis=0)

        data_array = self.sess.get_run_velocity_by_fr(fr_idx, fr_type='stim', 
                               remnans=remnans, stand=stand)

        if remnans:
            nanpol = None 
        else:
            nanpol = 'omit'

        if baseline is not None: # calculate baseline and subtract
            if baseline > pre + post:
                raise ValueError('Baseline greater than sequence length.')
            baseline_fr = int(np.around(baseline * self.sess.stim_fps))
            baseline_data = data_array[:, : baseline_fr]
            data_array_base = math_util.mean_med(baseline_data, stats=stats, 
                                        axis=-1, nanpol='omit')[:, np.newaxis]
            data_array = data_array - data_array_base

        if integ:
            data_array = math_util.integ(data_array, 1./self.sess.stim_fps, 
                                         axis=1, nanpol=nanpol)

        return xran, data_array


    #############################################
    def get_run_array_stats(self, stim_ref_fr, pre, post, integ=False,
                            remnans=True, ret_arr=False, stats='mean', 
                            error='std', baseline=None, stand=False):
        """
        self.get_run_array_stats(stim_ref_fr, pre, post)

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
            - stats (str)     : return mean ('mean') or median ('median')
                                default: 'mean'
            - error (str)     : return std dev/quartiles ('std') or SEM/MAD 
                                ('sem')
                                default: 'sem'
            - baseline (num)  : number of seconds to use as baseline. If None,
                                data is not baselined.
                                default: None
            - stand (bool)    : if True, running is standardized based on 
                                full trace array
                                default: False

        Returns:
            - xran (1D array)           : time values for the stimulus frames 
                                          (length is equal to last data 
                                          dimension) 
            - data_stats (1 to 2D array): array of running data statistics, 
                                          structured as:
                                             stats [me, err] (x frames)
            if ret_arr, also:
            - data_array (1 to 2D array): running data array, structured as:
                                              sequences (x frames)
        """

        xran, data_array = self.get_run_array(stim_ref_fr, pre, post, integ, 
                                              baseline=baseline, stats=stats, 
                                              remnans=remnans, stand=stand)

        if remnans:
            nanpol = None
        else:
            nanpol = 'omit'

        all_data = self.get_array_stats(xran, data_array, ret_arr, axes=0, 
                                        stats=stats, error=error, integ=integ, 
                                        nanpol=nanpol)

        if ret_arr:
            xran, data_stats, data_array = all_data
            return xran, data_stats, data_array
        
        else:
            xran, data_stats = all_data
            return xran, data_stats


    #############################################
    def get_roi_trace_array(self, twop_ref_fr, pre, post, fluor='dff', 
                            integ=False, remnans=True, baseline=None, 
                            stats='mean', transients=False, stand=False, 
                            pad=(0, 0), smooth=False):
        """
        self.get_roi_trace_array(twop_ref_fr, pre, post)

        Returns an array of 2p trace data around specific 2p frame numbers. 

        Required args:
            - twop_ref_fr (list): 1D list of 2p frame numbers 
                                  (e.g., all 1st Gabor A frames)
            - pre (num)         : range of frames to include before each 
                                  reference frame number (in s)
            - post (num)        : range of frames to include after each 
                                  reference frame number (in s)

        Optional args:
            - fluor (str)         : if 'dff', dF/F is used, if 'raw', ROI traces
                                    default: 'dff'
            - integ (bool)        : if True, dF/F is integrated over frames
                                    default: False
            - remnans (bool)      : if True, ROIs with NaN/Inf values anywhere
                                    in session are excluded. If False, NaN 
                                    values (but not Inf values) are omitted in 
                                    calculating the data statistics.
                                    default: True
            - baseline (num)      : number of seconds to use as baseline. If 
                                    None, data is not baselined.
                                    default: None
            - stats (str)         : statistic to use for baseline, mean ('mean') 
                                    or median ('median')
                                    default: 'mean'
            - transients (bool)   : if True, only ROIs with transients are 
                                    retained
                                    default: False
            - stand (bool)        : if True, each ROI is standardized based on 
                                    full trace array
                                    default: False 
            - pad (tuple)         : number of frame to use as padding 
                                    (before, after)
                                    default: (0, 0)
            - smooth (bool or int): if not False, specifies the window length 
                                    to use in smoothing 
                                    default: False
        
        Returns:
            - xran (1D array)           : time values for the 2p frames
            - data_array (2 or 3D array): roi trace data, structured as 
                                          ROI x sequences (x frames)
        """
        
        fr_idx, xran = self.sess.get_twop_fr_ran(twop_ref_fr, pre, post)

        use_pad = pad
        if smooth:
            add_pad = np.ceil(smooth/2.0).astype(int)
            use_pad = [sub + add_pad for sub in use_pad]

        # get dF/F: ROI x seq x fr
        data_array = self.sess.get_roi_seqs(fr_idx, fluor=fluor, 
                                            remnans=remnans, stand=stand, 
                                            padding=use_pad)
        
        if remnans:
            nanpol = None
        else:
            nanpol = 'omit'

        if baseline is not None: # calculate baseline and subtract
            if baseline > pre + post:
                raise ValueError('Baseline greater than sequence length.')
            baseline_fr = int(np.around(baseline * self.sess.twop_fps))
            baseline_data = data_array[:, :, : baseline_fr]
            data_array_base = math_util.mean_med(baseline_data, stats=stats, 
                                     axis=-1, nanpol=nanpol)[:, :, np.newaxis]
            data_array = data_array - data_array_base

        if transients:
            keep_rois = self.sess.get_active_rois(fluor=fluor, stimtype=None, 
                                                  remnans=remnans)
            data_array = data_array[keep_rois]

        if smooth:
            data_array = math_util.rolling_mean(data_array, win=add_pad)
            # cut down based on pad
            data_array = data_array[:, :, add_pad:-add_pad]

        if integ:
            data_array = math_util.integ(data_array, 1./self.sess.twop_fps, 
                                         axis=2, nanpol=nanpol)

        return xran, data_array
    
    
    #############################################
    def get_roi_trace_stats(self, twop_ref_fr, pre, post, byroi=True, 
                            fluor='dff', integ=False, remnans=True, 
                            ret_arr=False, stats='mean', error='std', 
                            baseline=None, transients=False, stand=False, 
                            smooth=False):
        """
        self.get_roi_trace_stats(twop_ref_fr, pre, post)

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
            - fluor (str)         : if 'dff', dF/F is used, if 'raw', ROI traces
                                    default: 'dff'
            - integ (bool)        : if True, dF/F is integrated over sequences
                                    default: False
            - remnans (bool)      : if True, ROIs with NaN/Inf values anywhere
                                    in session are excluded. If False, NaN 
                                    values (but not Inf values) are omitted in 
                                    calculating the data statistics.
                                    default: True
            - ret_arr (bool)      : also return ROI trace data array, not just  
                                    statistics.
            - stats (str)         : return mean ('mean') or median ('median')
                                    default: 'mean'
            - error (str)         : return std dev/quartiles ('std') or SEM/MAD 
                                    ('sem')
                                    default: 'sem'
            - baseline (num)      : number of seconds to use as baseline. If 
                                    None, data is not baselined.
                                    default: None
            - transients (bool)   : if True, only ROIs with transients are 
                                   retained
                                   default: False
            - stand (bool)        : if True, each ROI is standardized based on 
                                    full trace array
                                    default: False
            - smooth (bool or int): if not False, specifies the window length 
                                    to use in smoothing 
                                    default: False

        Returns:
            - xran (1D array)           : time values for the 2p frames
                                          (length is equal to last data array 
                                          dimension) 
            - data_stats (1 to 3D array): array of trace data statistics, 
                                          structured as:
                                              stats [me, err] (x ROI) (x frames)
            if ret_arr, also:
            - data_array (2 to 3D array): roi trace data array, structured as:
                                            ROIs x sequences (x frames)
        """
        
        # array is ROI x seq (x fr)
        xran, data_array = self.get_roi_trace_array(twop_ref_fr, pre, post, 
                                fluor, integ, remnans=remnans, 
                                baseline=baseline, stats=stats, 
                                transients=transients, stand=stand, 
                                smooth=smooth)
            
        # order in which to take statistics on data
        axes = [1, 0]
        if byroi:
            axes = 1

        if remnans:
            nanpol = None
        else:
            nanpol = 'omit'

        all_data = self.get_array_stats(xran, data_array, ret_arr, axes=axes, 
                                        stats=stats, error=error, 
                                        nanpol=nanpol, integ=integ)
        
        if ret_arr:
            xran, data_stats, data_array = all_data
            return xran, data_stats, data_array
        
        else:
            xran, data_stats = all_data
            return xran, data_stats


    #############################################
    def get_run(self, by='block', remnans=True, stand=False):
        """
        self.get_run()

        Returns run values for each stimulus frame of each stimulus block.

        Optional args:
            - by (str)      : determines whether run values are returned in a  
                              flat list ('frame'), grouped by block ('block'), 
                              or further grouped by display sequence ('disp')
                              default: 'block'
            - remnans (bool): if True, NaN values are removed using linear 
                              interpolation.
                              default: True
            - stand (bool)  : if True, each ROI is standardized based on 
                              full trace array
                              default: False
        Returns:
            - run (list): list of running values for stimulus blocks
        """
        
        full_run = self.sess.get_run_velocity(remnans=remnans, stand=stand)
        
        run = []
        for i in self.block_ran_fr:
            temp = []
            for j in i:
                temp.append(full_run[j[0]: j[1]].tolist())
            run.append(temp)

        # if not returning by disp
        if by == 'block' or by == 'frame':
            run = [x for sub in run for x in sub]
            if by == 'frame':
                run = [x for sub in run for x in sub]
        elif by != 'disp':
            raise ValueError('`by` can only take the values `disp`, '
                             '`block` or `frame`.')
    
        return run

    
    #############################################
    def get_segs_by_twopfr(self, twop_fr):
        """
        self.get_segs_by_twopfr(twop_fr)

        Returns the stimulus segment numbers for the given two-photon imaging
        frames using linear interpolation, and round the segment numbers.

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
            raise UserWarning('Some of the specified frames are out of range')

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
        
        Initializes and returns a gabors object, and the attributes below. 
        
        Also calls
            self._set_block_params()
        
            - deg_per_pix (num)       : degrees per pixels used in conversion
                                        to generate stimuli
            - n_patches (int)         : number of gabors 
            - ori_kaps (float or list): orientation kappa (calculated from std) 
                                        for each gabor block (only one value 
                                        for production data)
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
                                        (A, B, C, D, E), in pixels with window
                                        center being (0, 0), structured as:
                                            segment type x gabor x coord (x, y) 
            - post (num)              : number of seconds from frame A that are
                                        included in a set (gray, A, B, C, D/E)
            - pre (num)               : number of seconds before frame A that
                                        are included in a set 
                                        (gray, A, B, C, D/E)
            - set_len_s (num)         : length of a set in seconds
                                        (set: gray, A, B, C, D/E)
            - sf (num)                : spatial frequency of the gabors 
                                        (in cyc/pix)
            - size_pr (2D array)      : specific gabor sizes for each segment
                                        types (A, B, C, D, E) (in pix), 
                                        structured as:
                                            segment type x gabor
            - size_ran (list)         : range of gabor sizes (in pix)
            - units (str)             : units used to create stimuli in 
                                        PsychoPy (e.g., 'pix')
        
        Required args:
            - sess (Session)  : session to which the gabors belongs
            - stim_n (int)    : this stimulus' number, x in 
                                sess.stim_dict['stimuli'][x]
        """

        super().__init__(sess, stim_n, stimtype='gabors')

        stim_info = self.sess.stim_dict['stimuli'][self.stim_n]
        
        # gabor specific parameters
        if self.sess.runtype == 'pilot':
            gabor_par = stim_info['stimParams']['gabor_params']
            sess_par  = stim_info['stimParams']['subj_params']
            self.ori_std = copy.deepcopy(gabor_par['ori_std'])
            oris_pr = np.asarray(stim_info['stimParams']['orisByImg'])
        elif self.sess.runtype == 'prod':
            gabor_par = stim_info['stim_params']['gabor_params']
            sess_par  = stim_info['stim_params']['session_params']
            self.ori_std = gabor_par['ori_std']
            oris_pr = np.asarray(sess_par['orisbyimg'])

        self.win_size = sess_par['windowpar'][0]
        self.deg_per_pix = sess_par['windowpar'][1]
        self.n_patches = gabor_par['n_gabors']
        self.oris      = sorted(gabor_par['oris'])
        self.phase     = gabor_par['phase']  
        self.sf        = gabor_par['sf']
        self.units     = gabor_par['units']
        self.pos_x     = np.asarray(list(zip(*sess_par['possize']))[0])[:, :, 0]
        self.pos_y     = np.asarray(list(zip(*sess_par['possize']))[0])[:, :, 1]
        self.sizes_pr  = np.asarray(list(zip(*sess_par['possize']))[1])

        self.pos_x_ran = [-self.win_size[0]/2., self.win_size[0]/2.]
        self.pos_y_ran = [-self.win_size[1]/2., self.win_size[1]/2.]
        self.ori_ran = [-180, 180]
        
        # modify self.oris_pr E frames, as they are rotated 90 deg from what is 
        # recorded
        seg_surps = np.asarray(self.sess.stim_df.loc[(self.sess.stim_df['stimType'] == 'g')]['surp'])
        seg_gabfr = np.asarray(self.sess.stim_df.loc[(self.sess.stim_df['stimType'] == 'g')]['gabfr'])
        seg_surp_gabfr = np.asarray((seg_surps == 1) * (seg_gabfr == 3))
        self.oris_pr = oris_pr + seg_surp_gabfr[:, np.newaxis] * 90
        # in case some E frames values are now above upper range, so fix
        ori_hi = np.where(self.oris_pr > self.ori_ran[1])
        new_vals = self.ori_ran[0] + self.oris_pr[ori_hi] - self.ori_ran[1]
        self.oris_pr[ori_hi] = new_vals

        size_ran = copy.deepcopy(gabor_par['size_ran'])
        if self.units == 'pix':
            self.sf = gabor_par['sf']*self.deg_per_pix 
            size_ran = [x/self.deg_per_pix for x in size_ran]
        else:
             raise ValueError('Expected self.units to be pix.')

        # Convert to size as recorded in PsychoPy
        gabor_modif = 1./(2*np.sqrt(2*np.log(2))) * gabor_par['sd']
        self.size_ran = [np.around(x*gabor_modif) for x in size_ran]

        # kappas calculated as 1/std**2
        if self.sess.runtype == 'pilot':
            self.ori_kaps = [1./x**2 for x in self.ori_std] 
        elif self.sess.runtype == 'prod':
            self.ori_kaps = 1./self.ori_std**2

        # seg sets (hard-coded, based on the repeating structure  we are 
        # interested in, namely: blank, A, B, C, D/E)
        self.pre  = 1 * self.seg_len_s # 0.3 s blank
        self.post = self.n_seg_per_set * self.seg_len_s # 1.2 ms gabors
        self.set_len_s = self.pre + self.post
        
        # get parameters for each block
        self._set_block_params()


    #############################################
    def _set_block_params(self):
        """
        self._set_block_params()

        Sets the following attributes related to parameters that change across 
        blocks:
            - block_params (nested_list): gabor kappa parameter for each block,
                                          structured as: 
                                              display sequence x block x param
        """

        self.block_params = []
        for i, disp in enumerate(self.block_ran_seg):
            block_par = []
            for j, block in enumerate(disp):
                segs = self.sess.stim_df.loc[(self.sess.stim_df['stimType']==self.stimtype[0]) & 
                                              (self.sess.stim_df['stimSeg'] >= block[0]) & 
                                              (self.sess.stim_df['stimSeg'] < block[1])]
                # skipping stimPar1 which indicates gabor orientations which 
                # change at each gabor sequence presentation
                stimPar2 = segs['stimPar2'].unique().tolist()

                if len(stimPar2) > 1:
                    block_n = i*len(self.block_ran_seg)+j+1
                    raise ValueError(f'Block {block_n} of {self.stimtype} '
                                     'comprises segments with different '
                                     f'stimPar2 values: {stimPar2}')
                block_par.append([stimPar2[0]])
            self.block_params.append(block_par)


    #############################################
    def get_A_segs(self, by='block'):
        """
        self.get_A_segs()

        Returns lists of A gabor segment numbers.

        Optional args:
            - by (str): determines whether segment numbers are returned in a 
                        flat list ('seg'), grouped by block ('block'), or 
                        further grouped by display sequence ('disp')
                        default: 'block'
        Returns:
            - A_segs (list): list of A gabor segment numbers.
        """
        A_segs = self.get_segs_by_criteria(gabfr=0, by=by)

        return A_segs


    #############################################
    def get_A_frame_1s(self, by='block'):
        """
        self.get_A_frame_1s()

        Returns list of first frame number for each A gabor segment number.

        Optional args:
            - by (str): determines whether frame numbers are returned in a flat 
                        list ('frame'), grouped by block ('block'), or further 
                        grouped by display sequence ('disp')
                        default: 'block'
     
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

        Returns stimulus parameters for specified segments.

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
            - pars (list): list of requested parameter arrays for the requested 
                           segments with:
                if pos:
                - (nd array): x position array, structured as:
                                segs dims x gabors
                - (nd array): y position array, structured as:
                                segs dims x gabors
                if ori:
                -(nd array) : orientation array, structured as:
                                segs dims x gabors
                if size:
                - (nd array): size array, structured as:
                                segs dims x gabors
        """

        # a few checks, as the orientations are retrieved on the assumption
        # that the stimulus seg numbers are consecutive within the stimulus, 
        # starting at 0        
        min_seg = np.min(self.block_ran_seg)
        max_seg = np.max(self.block_ran_seg)

        if min_seg != 0:
            raise NotImplementedError('Function not properly implemented if '
                                      'the minimum segment is not 0.')
        if max_seg != self.oris_pr.shape[0]:
            raise NotImplementedError('Function not properly implemented if '
                                      'the maximum segment is not the same '
                                      'as the number of orientations recorded.')

        # check that at least one parameter type is requested
        if not(pos or ori or size):
            raise ValueError('At least one of the following must be True: '
                             'pos, ori, size.')
        
        segs = np.asarray(segs)

        if pos or size:
            segs_max = np.max(segs)
            
            # values will be returned in ascending order of seg number
            gabfr = self.sess.stim_df.loc[(self.sess.stim_df['stimType'] == 'g') &
                                        (self.sess.stim_df['stimSeg'] < segs_max)]['gabfr'].tolist()
            surps = self.sess.stim_df.loc[(self.sess.stim_df['stimType'] == 'g') &
                                        (self.sess.stim_df['stimSeg'] < segs_max)]['surp'].tolist()

            # reintroduce -1s
            gabfr.append(-1)
            surps.append(-1)

            if len(gabfr) != segs_max + 1 or len(surps) != segs_max + 1:
                raise ValueError('Something went wrong.')

            # change gabfr 3 to 4 if surprise sequence
            gabfr = np.asarray(gabfr)
            gabfr[np.where((np.asarray(surps) + gabfr) == 4)[0]] = 4

            # fill new array with gabor frame values at each requested segment
            # and reshape to requested input
            seq_gabfr = gabfr[segs]
        else:
            seq_gabfr = None
        
        # for each parameter:
        # boolean (whether to include or not)
        par_bools = [pos, pos, ori, size] 
        # attributes
        par_atts  = [self.pos_x, self.pos_y, self.oris_pr, self.sizes_pr]
        # use gabfr or seg nbrs as indices 
        par_idxs  = [seq_gabfr, seq_gabfr, segs, seq_gabfr] 
        # full ranges
        par_extr  = [self.pos_x_ran, self.pos_y_ran, self.ori_ran, 
                     self.size_ran] 

        pars = []        
        for b, idx, att, extr in zip(par_bools, par_idxs, par_atts, par_extr):
            if b:
                vals = att
                if scale:
                    extr_fl = [float(e) for e in extr]
                    vals = 2 * (vals - extr_fl[0])/(extr_fl[1] - extr_fl[0]) - 1
                # pad the end of first axis with 0s (for segs or gabfr = -1)
                vals_pad = np.concatenate([vals, 
                                      np.full([1] + list(vals.shape[1:]), 0)], 
                                      axis=0)
                seq_vals = vals_pad[idx]
                pars.append(seq_vals)

        return pars


    #############################################
    def get_stim_par_by_twopfr(self, twop_ref_fr, pre, post, pos=True, 
                               ori=True, size=True, scale=False):
        """
        self.get_stim_par_by_seg(segs)

        Returns stimulus parameters for 2p frame sequences specified by the 
        reference frame numbers and pre and post ranges.

        NOTE: A warning will be thrown if any of the 2p frame sequences occur
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
            - pars (list): list of requested parameter arrays for the specified 
                           frame sequences with:
                if pos:
                - (nd array): x position array, structured as:
                                segs dims x gabors
                - (nd array): y position array, structured as:
                                segs dims x gabors
                if ori:
                -(nd array) : orientation array, structured as:
                                segs dims x gabors
                if size:
                - (nd array): size array, structured as:
                                segs dims x gabors
        """

        twopfr_seqs, _ = self.sess.get_twop_fr_ran(twop_ref_fr, pre, post)

        # check whether any of the segments occur during Bricks
        if hasattr(self.sess, 'bricks'):
            bri_segs = self.sess.bricks.get_segs_by_twopfr(twopfr_seqs)
            if not (bri_segs == -1).all():
                print('    WARNING: some of the frames requested occur while '
                      'Bricks are presented.')

        # get seg numbers for each twopfr in each sequence
        seq_segs = self.get_segs_by_twopfr(twopfr_seqs)

        pars = self.get_stim_par_by_seg(seq_segs, pos=pos, ori=ori, size=size, 
                                        scale=scale)

        return pars


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
        
        Also calls
            self._set_block_params()

            - deg_per_pix (num)       : degrees per pixels used in conversion
                                        to generate stimuli

            - direcs (list)           : main brick direction for each block
            - flipfrac (num)          : fraction of bricks that flip direction 
                                        at each surprise
            - n_bricks (float or list): n_bricks for each brick block (only one
                                        value for production data)
            - sizes (int or list)     : brick size for each brick block (only
                                        one value for production data) (in pix)
            - speed (num)             : speed at which the bricks are moving 
                                        (in pix/sec)
            - units (str)             : units used to create stimuli in 
                                        PsychoPy (e.g., 'pix')
        
        Required args:
            - sess (Session)  : session to which the bricks belongs
            - stim_n (int)    : this stimulus' number (2 in the case of
                                production bricks): x in 
                                sess.stim_dict['stimuli'][x]
        """

        super().__init__(sess, stim_n, stimtype='bricks')
            
        stim_info = self.sess.stim_dict['stimuli'][self.stim_n]

        # initialize brick specific parameters
        if self.sess.runtype == 'pilot':
            sqr_par     = stim_info['stimParams']['square_params']
            self.units  = sqr_par['units']
            self.deg_per_pix = stim_info['stimParams']['subj_params']['windowpar'][1]
            self.direcs = sqr_par['direcs']
            self.sizes  = copy.deepcopy(sqr_par['sizes'])
            
            # calculate n_bricks, as wasn't explicitly recorded
            max_n_brick   = stim_info['stimParams']['elemParams']['nElements']
            prod          = float(max_n_brick) * min(self.sizes)**2
            self.n_bricks = [int(prod/size**2) for size in self.sizes]

            if self.units == 'pix':
                # sizes recorded in deg, so converting to pix (only for pilot)
                self.sizes = [np.around(x/self.deg_per_pix) for x in self.sizes]
            
        elif self.sess.runtype == 'prod':
            sqr_par       = stim_info['stim_params']['square_params']
            stim_info2    = self.sess.stim_dict['stimuli'][self.stim_n_all[1]]
            self.units    = sqr_par['units']
            self.deg_per_pix = stim_info['stim_params']['session_params']['windowpar'][1]
            self.direcs   = [stim_info['stim_params']['direc'], 
                             stim_info2['stim_params']['direc']]
            self.sizes    = stim_info['stim_params']['elemParams']['sizes']
            self.n_bricks = stim_info['stim_params']['elemParams']['nElements']
        
        self.speed = sqr_par['speed']
        if self.units == 'pix':
            # recorded in deg, so converting to pix
            self.speed = self.speed/self.deg_per_pix
        else:
            raise ValueError('Expected self.units to be pix.')
       
        self.flipfrac = sqr_par['flipfrac']

        # set parameters for each block
        self._set_block_params()


    #############################################
    def _set_block_params(self):
        """
        self._set_block_params()

        Sets the following attributes related to parameters that change across 
        blocks:
            - block_params (nested_list): brick direction, size and number
                                          parameter for each block,
                                          structured as: 
                                              display sequence x block x param
        """

        self.block_params = []
        for i, disp in enumerate(self.block_ran_seg):
            block_par = []
            for j, block in enumerate(disp):
                segs = self.sess.stim_df.loc[(self.sess.stim_df['stimType']==self.stimtype[0]) & 
                                              (self.sess.stim_df['stimSeg'] >= block[0]) & 
                                              (self.sess.stim_df['stimSeg'] < block[1])]
                stimPars = []
                for par_name in ['stimPar2', 'stimPar1']:
                    stimPar = segs[par_name].unique().tolist()
                    if len(stimPar) > 1:
                        block_n = i*len(self.block_ran_seg)+j+1
                        raise ValueError(f'Block {block_n} of {self.stimtype} '
                                        'comprises segments with different '
                                        f'{par_name} values: {stimPar}')
                    stimPars.append(stimPar[0])
                
                # add n_bricks info
                if self.sess.runtype == 'prod':
                    stimPars.append(self.n_bricks)
                else:
                    if stimPars[0] == min(self.sizes):
                        stimPars.append(max(self.n_bricks))
                    else:
                        stimPars.append(min(self.n_bricks))
                block_par.append(stimPars)
            self.block_params.append(block_par)


    #############################################
    def get_dir_segs_reg(self, by='block'):
        """
        self.get_dir_segs_reg()

        Returns two lists of stimulus segment numbers, the first is a list of 
        the right moving segments. The second is a list of left moving 
        segments. Both lists exclude surprise segments.

        Optional args:
            - by (str): determines whether segment numbers are returned in a 
                        flat list ('seg'), grouped by block ('block'), or 
                        further grouped by display sequence ('disp')
                        default: 'block'  
        Returns:
            - right_segs (list): list of right moving segment numbers, 
                                 excluding surprise segments.
            - left_segs (list) : list of left moving segment numbers, 
                                 excluding surprise segments.
        """

        right_segs = self.get_segs_by_criteria(bri_dir='right', surp=0, by=by)
        left_segs  = self.get_segs_by_criteria(bri_dir='left', surp=0, by=by)

        return right_segs, left_segs


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

            - sess (Session object): session to which the grayscr belongs
            - gabors (bool)        : if True, the session to which the grayscr 
                                     belongs has a gabors attribute
        
        Required args:
            - sess (Session object): session to which the grayscr belongs
            - stim_n (int)    : this stimulus' number (2 in the case of
                                production bricks): x in 
                                sess.stim_dict['stimuli'][x]
        """

        self.sess = sess
        if hasattr(self.sess, 'gabors'):
            self.gabors = True
        else:
            self.gabors = False
        

    #############################################        
    def get_all_nongab_stim_fr(self):
        """
        self.get_all_nongab_stim_fr()

        Returns a lists of grayscreen stimulus frame numbers, excluding 
        grayscreen stimulus frames occurring during gabor stimulus blocks, 
        including grayscreen stimulus frames flanking gabor stimulus blocks.
        
        Returns:
            grays (list): list of grayscreen stimulus frames.
        """

        frames = []
        if self.gabors:
            frames_gab = np.asarray(self.sess.gabors.stim_seg_list)
            gab_blocks = self.sess.gabors.block_ran_fr
            for i in gab_blocks:
                for j in i:
                    frames_gab[j[0]:j[1]] = 0
            frames.append(frames_gab)
        if hasattr(self.sess, 'bricks'):
            frames.append(np.asarray(self.sess.bricks.stim_seg_list))
        length = len(frames)
        if length == 0:
            raise ValueError('No frame lists were found for either stimulus '
                             ' types (gabors, bricks).')
        elif length == 1:
            frames_sum = np.asarray(frames)
        else:
            frames_sum = np.sum(np.asarray(frames), axis=0)
        grays = np.where(frames_sum==length*-1)[0].tolist()

        if len(grays) == 0:
            raise ValueError('No grayscreen frames were found outside of '
                            'gabor stimulus sequences.')

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
            first_grays (list) : list of first grayscreen stimulus frame 
                                 numbers for every grayscreen sequence
            n_grays (list)     : list of number of grayscreen stimulus frames 
                                 for every grayscreen sequence
        """

        grays_all = self.get_all_nongab_stim_fr()
        first_grays = []
        n_grays = []
        k=0

        for i, val in enumerate(grays_all):
            if i == 0:
                first_grays.extend([val])
                k=1
            elif val != grays_all[i-1]+1:
                n_grays.extend([k])
                first_grays.extend([val])
                k = 1
            else:
                k +=1
        n_grays.extend([k])

        return first_grays, n_grays


    #############################################
    def get_all_gab_stim_fr(self, by='block'):
        """
        self.get_all_gab_stim_fr()

        Returns a list of grayscreen stimulus frame numbers for every 
        grayscreen sequence during a gabor block, excluding grayscreen 
        sequences flanking the gabor blocks.

        Optional args:
            - by (str): determines whether frame numbers are returned in a 
                        flat list ('frame'), grouped by block ('block'), or 
                        further grouped by display sequence ('disp')
                        default: 'block'    

        Returns:
            - gab_grays (list): list of grayscreen stimulus frame numbers for 
                                every grayscreen sequence during gabors
        """
        
        if self.gabors:
            frames_gab = np.asarray(self.sess.gabors.stim_seg_list)
            gab_blocks = self.sess.gabors.block_ran_fr
            gab_grays = []
            for i in gab_blocks:
                temp = []
                for j in i:
                    grays = np.where(frames_gab[j[0]:j[1]]==-1)[0] + j[0]
                    temp.append(grays.tolist())
                gab_grays.append(temp)

            # if not returning by disp
            if by == 'block' or by == 'frame':
                gab_grays = [x for sub in gab_grays for x in sub]
                if by == 'frame':
                    gab_grays = [x for sub in gab_grays for x in sub]
            elif by != 'disp':
                raise ValueError('`by` can only take the values `disp`, '
                                 '`block` or `frame`.')
            
            return gab_grays
        else:
            raise ValueError('Session does not have a gabors attribute. Be '
                             'sure to extract stim info and check that '
                             'session contains a gabor stimulus.')


    #############################################    
    def get_gab_gray_stim_fr(self, by='block'):
        """
        self.get_gab_gray_stim_fr()

        Returns every first grayscreen stimulus frame number for every 
        grayscreen sequence occuring during a gabor stimulus blocks, and the 
        number of consecutive grayscreen stimulus frames for each sequence. 
                
        NOTE: any grayscreen stimulus frames for sequences flanking gabor 
        stimulus blocks are excluded in the returned list.

        Optional args:
            - by (str): determines whether frame numbers are returned in a 
                        flat list ('frame'), grouped by block ('block'), or 
                        further grouped by display sequence ('disp')
                        default: 'block'    

        Returns:
            - first_gab_grays (list): list of first grayscreen stimulus frame 
                                      numbers for every grayscreen sequence 
                                      during gabors
            - n_gab_grays (list)    : list of number of grayscreen stimulus 
                                      frames for every grayscreen sequence 
                                      during gabors
        """

        grays_gab = self.get_all_gab_stim_fr(by='disp')
        first_gab_grays = []
        n_gab_grays = []

        for i in grays_gab:
            temp_first = []
            temp_n = []
            k=0
            for j in i:
                temp2_first = []
                temp2_n = []
                for l, val in enumerate(j): 
                    if l == 0:
                        temp2_first.extend([val])
                        k = 1
                    elif val != j[l-1]+1:
                        temp2_n.extend([k])
                        temp2_first.extend([val])
                        k = 1
                    else:
                        k += 1
                temp2_n.extend([k])
                temp_first.append(temp2_first)
                temp_n.append(temp2_n)
            first_gab_grays.append(temp_first)
            n_gab_grays.append(temp_n)

        # if not returning by disp
        if by == 'block' or by == 'frame':
            first_gab_grays = [x for sub in first_gab_grays for x in sub]
            n_gab_grays     = [x for sub in n_gab_grays for x in sub]
            if by == 'frame':
                first_gab_grays = [x for sub in first_gab_grays for x in sub]
                n_gab_grays     = [x for sub in n_gab_grays for x in sub]
        elif by != 'disp':
            raise ValueError('`by` can only take the values `disp`, '
                             '`block` or `frame`.')

        return first_gab_grays, n_gab_grays


