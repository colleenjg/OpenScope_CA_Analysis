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
import pandas
import pickle
import scipy.stats as st

from allensdk.brain_observatory import dff

from sess_util import sess_file_util, sess_sync_util
from util import file_util, gen_util, math_util



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
    
    def __init__(self, datadir, sessid, runtype='prod', droptol=0.0003):
        """
        self.__init__(datadir, sessid)

        Initializes and returns the new Session object using the specified data 
        directory and ID.

        Sets attributes:
            - droptol (float): dropped frame tolerance (proportion of total)
            - home (str)     : path of the master data directory
            - runtype (str)  : 'prod' (production) or 'pilot' data
            - sessid (int)   : session ID (9 digits), e.g. '712483302'
        
        and calls self._init_directory()


        Required args:
            - datadir (str): full path to the directory where session 
                             folders are stored.
            - sessid (int) : the ID for this session.

        Optional args:
            - runtype (str)   : the type of run, either 'pilot' or 'prod'
                                default: 'prod'
            - droptol (float) : the tolerance for percentage stimulus frames 
                                dropped, create a Warning if this condition 
                                isn't met.
                                default: 0.0003 
        """

        self.home   = datadir
        self.sessid = int(sessid)
        if runtype not in ['pilot', 'prod']:
            gen_util.accepted_values_error('runtype', runtype, 
                                           ['pilot', 'prod'])
        self.runtype = runtype
        self.droptol = droptol
        self._init_directory()
        

    #############################################
    def _init_directory(self):
        """
        self._init_directory()

        Checks that the session data directory obeys the expected organization
        scheme and sets the following attributes:
        
            - align_pkl (str)     : path name of the stimulus alignment pickle 
                                    file
            - corrected (str)     : path name of the motion corrected 2p data 
                                    hdf5 file
            - date (str)          : session date (i.e., yyyymmdd)
            - dir (str)           : path of session directory
            - expdir (str)        : path name of experiment directory
            - expid (int)         : experiment ID (8 digits)
            - mouseid (int)       : mouse ID (6 digits)
            - procdir (str)       : path name of the processed data directory
            - roi_traces (str)    : path name of the ROI raw fluorescence trace 
                                    hdf5 file
            - roi_traces_dff (str): path name of the ROI raw dF/F trace 
                                    hdf5 file
            - stim_pkl (str)      : path name of the stimulus pickle file
            - stim_sync (str)     : path name of the stimulus synchronisation 
                                    hdf5 file
            - z_stack (str)       : path name of the z-stack 2p hdf5 file
        """

        # check that the high-level home directory exists
        file_util.checkdir(self.home)

        # set the session directory (full path)
        wild_dir  = os.path.join(self.home, self.runtype, 'mouse_*', 
                                 'ophys_session_{}'.format(self.sessid))
        name_dir  = glob.glob(wild_dir)
        mouse_dir = True
        
        # pilot data may not be in a 'mouse_' folder
        if len(wild_dir) == 0:
            wild_dir  = os.path.join(self.home, self.runtype,  
                                     'ophys_session_{}'.format(self.sessid))
            name_dir  = glob.glob(wild_dir)
            mouse_dir = False
        
        if len(name_dir) == 0:
            raise OSError(('Could not find directory for session {} in {}'
                           ' subfolders').format(self.sessid, self.home))
        self.dir = name_dir[0]

        # extract the mouse ID, and date from the stim pickle file
        pklglob = glob.glob(os.path.join(self.dir, 
                                         '{}*stim.pkl'.format(self.sessid)))
        
        if len(pklglob) == 0:
            raise OSError('Could not find stim pkl file in {}'.format(self.dir))
        else:
            pklinfo = os.path.basename(pklglob[0]).split('_')
        
        self.mouseid = int(pklinfo[1]) # mouse 6 digit nbr
        self.date    = pklinfo[2]

        # extract the experiment ID from the experiment directory name
        expglob = glob.glob(os.path.join(self.dir,'ophys_experiment*'))
        if len(expglob) == 0:
            raise OSError(('Could not find experiment directory '
                           'in {}').format(self.dir))
        else:
            expinfo = os.path.basename(expglob[0]).split('_')
        self.expid = int(expinfo[2])

        # create the filenames
        (self.expdir, self.procdir, self.stim_pkl, self.stim_sync, 
        self.align_pkl, self.corrected, self.roi_traces, self.roi_traces_dff, 
            self.zstack) = \
            sess_file_util.get_file_names(self.home, self.sessid, self.expid, 
                                          self.date, self.mouseid, 
                                          self.runtype, mouse_dir)       
    

    #############################################
    def _create_small_stim_pkl(self, small_stim_pkl):
        """
        self._create_small_stim_pkl(small_stim_pkl)

        Creates and saves a smaller stimulus dictionary from the stimulus  
        pickle file in which 'posbyframe' for bricks stimuli is not included. 
        Reduces the pickle size about 10 fold.

        Required args:
            - small_stim_pkl (str): full path name for the small stimulus
                                    pickle file
        """
    
        print('Creating smaller stimulus pickle.')

        self.stim_dict = file_util.loadfile(self.stim_pkl)

        if self.runtype == 'pilot':
            stim_par = 'stimParams'
        elif self.runtype == 'prod':
            stim_par = 'stim_params'

        for i in range(len(self.stim_dict['stimuli'])):
            stim_keys = self.stim_dict['stimuli'][i][stim_par].keys()
            stim_par = self.stim_dict['stimuli'][i][stim_par]
            if self.runtype == 'pilot' and 'posByFrame' in stim_keys:
                _ = stim_par.pop('posByFrame')
            elif self.runtype == 'prod' and 'square_params' in stim_keys:
                _ = stim_par['session_params'].pop('posbyframe')
                
        file_util.saveinfo(self.stim_dict, small_stim_pkl)


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
            - stim_fps (float)    : stimulus frames per second
            - tot_stim_fr (int)   : number of stimulus frames
            - run (1D array)      : array of running speeds in cm/s for each 
                                    recorded stimulus frames

        Optional args:
            - fulldict (bool)  : if True, the full stim_dict is loaded,
                                 else the small stim_dict is loaded
                                 (does not contain 'posbyframe' for Bricks)
                                 default: True
        """

        if fulldict:
            self.stim_dict = file_util.loadfile(self.stim_pkl)

        else:
            # load the smaller dict
            small_stim_pkl = '{}_small.pkl'.format(self.stim_pkl[0:-4])
            if not os.path.exists(small_stim_pkl):
                self._create_small_stim_pkl(small_stim_pkl[0:-4])
            else:
                self.stim_dict = file_util.loadfile(small_stim_pkl)
            print('Using smaller stimulus pickle.')

        # store some variables for easy access
        self.stim_fps       = self.stim_dict['fps']
        self.tot_stim_fr    = self.stim_dict['total_frames']
        self.pre_blank      = self.stim_dict['pre_blank_sec']  # seconds
        self.post_blank     = self.stim_dict['post_blank_sec'] # seconds
        self.drop_stim_fr   = self.stim_dict['droppedframes']
        self.n_drop_stim_fr = len(self.drop_stim_fr[0])

        # running speed per stimulus frame in cm/s
        self.run = sess_sync_util.get_run_speed(stim_dict=self.stim_dict)

        # check our drop tolerance
        if np.float(self.n_drop_stim_fr)/self.tot_stim_fr > self.droptol:
            print('WARNING: {} dropped stimulus frames out of {}.'
                  .format(self.n_drop_stim_fr, self.tot_stim_fr))
        

    #############################################
    def _load_sync_h5(self):
        """
        self._load_sync_h5()

        Loads the stimulus synchronisation hdf5 file.
        """

        raise NotImplementedError(('Loading h5 sync file of a session has '
                                   'not been implemented yet.'))
    

    #############################################
    def _load_align_df(self):
        """
        self._load_align_df()

        Creates the alignment dataframe (align_df) and stores it as a pickle
        in the session directory, if it does not already exist. Loads
        it if it exists, along with the stimulus alignment list (stim_align).
        Sets the following attributes:
        
            - align_df (pandas df) : stimlus alignment dataframe
            - stim_align (1D array): list of 2p frame numbers for each
                                     stimulus frame, as well as the flanking
                                     blank screen frames 
            - twop_fps (float)     : mean 2p frames per second
            - twop_fr_stim (int)   : number of 2p frames recorded while stim
                                     was playing
        """

        # create align_df if doesn't exist
        if not os.path.exists(self.align_pkl):
            sess_sync_util.get_stim_frames(self.stim_pkl, self.stim_sync, 
                                           self.align_pkl, self.runtype)
            
        else:
            print('NOTE: Stimulus alignment pickle already exists in {}'
                  .format(self.dir))

        align = file_util.loadfile(self.align_pkl)

        self.align_df     = align['stim_df']
        self.stim_align   = align['stim_align']
        self.twop_fps     = sess_sync_util.get_frame_rate(self.stim_sync)[0] 
        self.twop_fr_stim = int(max(align['stim_align']))
    

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
        - fluor (str): if 'dff', a nanrois attribute is added for dF/F traces. 
                       If 'raw, it is created for raw traces.
                       default: 'dff'
        """
        
        # generate attribute listing ROIs with NaNs or Infs (for dff traces)
        if fluor == 'dff':
            full_trace_file = self.roi_traces_dff
        elif fluor == 'raw':
            full_trace_file = self.roi_traces
        else:
            gen_util.accepted_values_error('fluor', fluor, ['raw', 'dff'])
        
        if not os.path.exists(full_trace_file):
            raise ValueError(('Specified ROI traces file does not exist: '
                              '{}').format(full_trace_file))
        
        with h5py.File(full_trace_file, 'r') as f:
            traces = f['data'].value

        nan_arr = np.isnan(traces).any(axis=1) + np.isinf(traces).any(axis=1)
        nan_rois = np.where(nan_arr)[0].tolist()

        if fluor == 'dff':
            self.nanrois_dff = nan_rois
        elif fluor == 'raw':
            self.nanrois = nan_rois


    ############################################
    def _create_dff(self, replace=False, basewin=1000):
        """
        self._create_dff()

        Creates and saves the dF/F traces.

        Also calls self._set_nanrois().

        Required args:
            - replace (bool): if True, replaces pre-existing dF/F traces. If
                              False, no new dF/F traces are created if they
                              already exist.
                              default: False
            - basewin (int) : basewin factor for compute_dff function
                              default: 1000
        
        """
        
        if not os.path.exists(self.roi_traces_dff) or replace:
            print(('Creating dF/F files using {} basewin '
                   'for session {}').format(basewin, self.sessid))
            # read the data points into the return array
            with h5py.File(self.roi_traces,'r') as f:
                try:
                    traces = f['data'].value
                except:
                    pdb.set_trace()
                    raise OSError('Could not read {}'.format(self.roi_traces))
            
            traces = dff.compute_dff(traces, mode_kernelsize=2*basewin, 
                                     mean_kernelsize=basewin)
                
            with h5py.File(self.roi_traces_dff, 'w') as hf:
                hf.create_dataset('data',  data=traces)
        
        # generate attribute listing ROIs with NaNs or Infs (for dF/F traces)
        self._set_nanrois('dff')


    #############################################
    def _load_roi_trace_info(self, basewin=1000):
        """
        self._load_roi_trace_info()

        Sets the attributes below based on the raw ROI traces.
        
        Also calls self._set_nanrois().

            - nrois (int)      : number of ROIs in traces
            - roi_names (list) : list of ROI names (9 digits)
            - tot_twop_fr (int): number of 2p frames recorded

        Optional args:
            - basewin (int): window length for calculating baseline 
                             fluorescence
                             default: 1000
        """
        
        self._create_dff(basewin=basewin)
        
        try:
            # open the roi file and get the info
            with h5py.File(self.roi_traces,'r') as f:
                
                # get the names of the rois
                self.roi_names = f['roi_names'].value.tolist()

                # get the number of rois
                self.nrois = len(self.roi_names)

                # get the number of data points in the traces
                self.tot_twop_fr = f['data'].shape[1]
    
        except:
            raise OSError(('Could not open {} for '
                        'reading').format(self.roi_traces))

        # generate attribute listing ROIs with NaNs or Infs (for raw traces)
        self._set_nanrois('raw')


    #############################################
    def _modif_bri_segs(self):
        """
        self._modif_bri_segs()

        Modifies brick segment numbers in align_df attribute to ensure that
        they are different for the two brick stimuli in the production data.

        Also updates the block segment ranges and both brick stim segment lists 
        to contain all brick segment numbers, including updated segment numbers.
        """
        
        if hasattr(self, '_bri_segs_modified'):
            return

        elif self.runtype == 'prod':
            bri_st_fr = gen_util.get_df_vals(self.align_df, 'stimType', 'b', 
                                             'start_frame', unique=False)
            bri_num_fr = np.diff(bri_st_fr)
            num_fr = gen_util.get_df_vals(self.align_df, 'stimType', 'b', 
                                          'num_frames', unique=False)[:-1]
            break_idx = np.where(num_fr != bri_num_fr)[0]
            n_br = len(break_idx)
            if n_br != 1:
                raise ValueError(('Expected only one break in the bricks '
                                  'stimulus, but found {}.'.format(n_br)))
            
            # last start frame and seg for the first brick stim
            last_fr1 = bri_st_fr[break_idx[0]] 
            last_seg1 = gen_util.get_df_vals(self.align_df, 
                                             ['stimType', 'start_frame'], 
                                             ['b', last_fr1], 'stimSeg')[0]
            
            seg_idx = ((self.align_df['stimType'] == 'b') & 
                       (self.align_df['start_frame'] > last_fr1))

            new_idx = self.align_df.loc[seg_idx]['stimSeg'] + last_seg1 + 1
            self.align_df = gen_util.set_df_vals(self.align_df, seg_idx, 
                                                 'stimSeg', new_idx)

            self._bri_segs_modified = True
                

    #############################################
    def extract_sess_attribs(self, mouse_df='mouse_df.csv'):
        """
        self.extract_sess_attribs(mouse_df)

        Loads the dataframe containing information on each session,
        and sets the following attributes:

            - all_files (bool) : if True, all files have been acquired for
                                 the session
            - any_files (bool) : if True, some files have been acquired for
                                 the session
            - depth (int)      : recording depth 
            - layer (str)      : recording layer ('soma' or 'dend')
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
                          session. Dataframe should have the following columns:
                              sessid, mouse_n, depth, layer, line, sess_gen, 
                              sess_within, sess_n, pass_fail, all_files, 
                              any_files, notes
        """

        if isinstance(mouse_df, str):
            mouse_df = file_util.loadfile(mouse_df)

        df_line = gen_util.get_df_vals(mouse_df, 'sessid', self.sessid)
        self.mouse_n      = int(df_line['mouse_n'].tolist()[0])
        self.depth        = df_line['depth'].tolist()[0]
        self.layer        = df_line['layer'].tolist()[0]
        self.line         = df_line['line'].tolist()[0]
        self.sess_gen     = int(df_line['sess_gen'].tolist()[0])
        self.sess_n       = int(df_line['sess_n'].tolist()[0])
        self.sess_within  = int(df_line['sess_within'].tolist()[0])
        self.pass_fail    = df_line['pass_fail'].tolist()[0]
        self.all_files    = bool(int(df_line['all_files'].tolist()[0]))
        self.any_files    = bool(int(df_line['any_files'].tolist()[0]))
        self.notes        = df_line['notes'].tolist()[0]


    #############################################
    def extract_info(self, fulldict=True, basewin=1000):
        """
        self.extract_info()

        This function should be run immediately after creating a Session 
        object. It creates the stimulus objects attached to the Session 

        If the attributes they set have not been initialized yet, calls:
            self._load_align_df()
            self._load_roi_trace_info()
            self._load_stim_dict()

        If the runtype is 'prod', calls self._modif_bri_segs().

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


        Optional args:
            - fulldict (bool): if True, the full stim_dict is loaded,
                               else the small stim_dict is loaded
                               (which contains everything, except 'posbyframe' 
                               for Bricks)
                               default: True
            - basewin (int)  : window length for calculating baseline 
                               fluorescence
                               default: 1000
        """

        # load the stimulus, running, alignment and trace information 
        if not hasattr(self, 'stim_dict'):
            print("Loading stimulus dictionary...")
            self._load_stim_dict(fulldict=fulldict)
        if not hasattr(self, 'align_df'):
            print("Loading alignment dataframe...")
            self._load_align_df()
        if not hasattr(self, 'roi_names'):
            print("Loading ROI trace info...")
            self._load_roi_trace_info(basewin=basewin)
        # TODO: load h5 sync
        
        # create the stimulus fields and objects
        self.stimtypes = []
        self.n_stims    = len(self.stim_dict['stimuli'])
        self.stims      = []
        if self.runtype == 'prod':
            n_bri = []
            # modify segment numbers for second bricks block as they are 
            # the same for both in production in the align_df
            self._modif_bri_segs()
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
                print(('{} stimulus type not recognized. No Stim object ' 
                      'created for this stimulus. \n').format(stimtype))

        # initialize a Grayscr object
        self.grayscr = Grayscr(self)


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
    def get_run_speed(self, twop_fr):
        """
        self.get_run_speed(twop_fr)

        Returns the running speed for the given two-photon imaging
        frames using linear interpolation.

        Required args:
            - twop_fr (array-like): set of 2p imaging frames for which 
                                    to get running speed
        
        Returns:
            - speed (nd array): running speed (in cm/s), with same dimensions 
                                as input array
        """

        twop_fr = np.asarray(twop_fr)

        # make sure the frames are within the range of 2p frames
        if (twop_fr >= self.tot_twop_fr).any() or (twop_fr < 0).any():
            raise UserWarning('Some of the specified frames are out of range')

        # perform linear interpolation on the running speed
        speed = np.interp(twop_fr, self.stim_align, self.run)

        return speed


    #############################################
    def get_nanrois(self, fluor='dff'):
        """
        self.get_nanrois()

        Returns as a list the indices of ROIs containing NaNs or Infs.

        Optional args:
            - fluor (str): if 'dff', the indices of ROIs with NaNs or Infs in 
                           the dF/F traces are returned. If 'raw', for raw 
                           traces.
                           default: 'dff'
        Returns:
            - (list): indices of ROIs containing NaNs or Infs
        """

        if fluor == 'dff':
            return self.nanrois_dff
        elif fluor == 'raw':
            return self.nanrois
        else:
            gen_util.accepted_values_error('fluor', fluor, ['raw', 'dff'])


    #############################################
    def get_roi_traces(self, frames=None, fluor='dff', remnans=True, 
                       basewin=1000):
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
            - remnans (bool)    : if True, removes ROIs with NaN/Inf values 
                                  anywhere in session. 
                                  default: True
            - basewin (int)     : window length for calculating baseline 
                                  fluorescence
                                  default: 1000

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
        self._load_roi_trace_info(basewin=basewin)
        if fluor == 'dff':
            roi_traces = self.roi_traces_dff
        elif fluor == 'raw':
            roi_traces = self.roi_traces
        else:
            gen_util.accepted_values_error('fluor', fluor, ['raw', 'dff'])

        with h5py.File(roi_traces, 'r') as f:
            try:
                traces = f['data'].value[:,frames]
            except:
                pdb.set_trace()
                raise OSError('Could not read {}'.format(self.roi_traces))
        
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
            - pre (float)       : range of frames to include before each 
                                  reference frame number (in s)
            - post (float)      : range of frames to include after each 
                                  reference frame number (in s)
         
        Returns:
            - num_ran (2D array): array of frame numbers, structured as:
                                      sequence x frames
            - xran (1D array)   : time values for the 2p frames
            
        """

        ran_fr = [np.around(x*self.twop_fps) for x in [-pre, post]]
        xran   = np.linspace(-pre, post, int(np.diff(ran_fr)[0]))

        if isinstance(twop_ref_fr[0], (list, np.ndarray)):
            raise OSError('Frames must be passed as a 1D list, not by block.')

        # get sequences x frames
        fr_idx = gen_util.num_ranges(twop_ref_fr, pre=-ran_fr[0], 
                                     leng=len(xran))
                     
        # remove sequences with negatives or values above total number of stim 
        # frames
        neg_idx  = np.where(fr_idx[:,0] < 0)[0].tolist()
        over_idx = np.where(fr_idx[:,-1] >= self.tot_twop_fr)[0].tolist()
        
        fr_idx = gen_util.remove_idx(fr_idx, neg_idx + over_idx, axis=0)

        return fr_idx, xran


    #############################################
    def get_roi_seqs(self, stim_fr_seqs, padding=(0,0), fluor='dff', 
                     remnans=True, basewin=1000):
        """
        self.get_roi_seqs(stim_fr_seqs)

        Returns the processed ROI traces for the given stimulus sequences.
        Frames around the start and end of the sequences can be requested by 
        setting the padding argument.

        If the sequences are different lengths the array is nan padded

        Required args:
            - stim_fr_seqs (list of arrays): list of arrays of 2p frames,
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
            - remnans (bool)           : if True, removes ROIs with NaN/Inf 
                                         values anywhere in session. 
                                         default: True
            - basewin (int)            : window length for calculating baseline 
                                         fluorescence
                                         default: 1000
        
        Returns:
            - traces (3D array): array of traces for the specified 
                                 ROIs and sequences, structured as: 
                                 ROIs x sequences x frames
        """

        # extend values with padding
        if padding[0] != 0:
            min_fr       = np.asarray([min(x) for x in stim_fr_seqs])
            st_padd      = np.tile(np.arange(-padding[0], 0), 
                                   (len(stim_fr_seqs), 1)) + min_fr[:,None]
            stim_fr_seqs = [np.concatenate((st_padd[i], x)) 
                           for i, x in enumerate(stim_fr_seqs)]
        if padding[1] != 0:
            max_fr       = np.asarray([max(x) for x in stim_fr_seqs])
            end_padd     = np.tile(np.arange(1, padding[1]+1), 
                                   (len(stim_fr_seqs), 1)) + max_fr[:,None]
            stim_fr_seqs = [np.concatenate((x, end_padd[i])) 
                            for i, x in enumerate(stim_fr_seqs)]
        if padding[0] < 0 or padding[1] < 0:
            raise ValueError('Negative padding not supported.')

        # get length of each padded sequence
        pad_seql = np.array([len(s) for s in stim_fr_seqs])

        # flatten the sequences into one list of frames, removing any sequences
        # with unacceptable frame values (< 0 or > self.tot_twop_fr) 
        frames_flat = np.empty([sum(pad_seql)])
        last_idx    = 0
        seq_rem     = []
        seq_rem_l   = []
        for i in range(len(stim_fr_seqs)):
            if (max(stim_fr_seqs[i]) >= self.tot_twop_fr or 
                min(stim_fr_seqs[i]) < 0):
                seq_rem.extend([i])
                seq_rem_l.extend([pad_seql[i]])
            else:
                frames_flat[last_idx : last_idx + pad_seql[i]] = stim_fr_seqs[i]
                last_idx += pad_seql[i]

        # Warn about removed sequences and update pad_seql and stim_fr_seqs 
        # to remove these sequences
        if len(seq_rem) != 0 :
            print(('Some of the specified frames for sequences {} are out of '
                   'range so the sequence will not be '
                   'included.').format(seq_rem))
            pad_seql     = np.delete(pad_seql, seq_rem)
            stim_fr_seqs = np.delete(stim_fr_seqs, seq_rem).tolist()

        # sanity check that the list is as long as expected
        if last_idx != len(frames_flat):
            if last_idx != len(frames_flat) - sum(seq_rem_l):
                raise ValueError(('Concatenated frame array is {} long '
                                  'instead of expected {}.')
                                  .format(last_idx, 
                                          len(frames_flat - sum(seq_rem_l))))
            else:
                frames_flat = frames_flat[: last_idx]

        # convert to int
        frames_flat = frames_flat.astype(int)

        # load the traces
        traces_flat = self.get_roi_traces(frames_flat.tolist(), fluor, 
                                          remnans, basewin)
        
        if remnans:
            nrois = self.nrois - len(self.get_nanrois(fluor))
        else:
            nrois = self.nrois

        # chop back up into sequences padded with Nans
        traces = np.full((nrois, len(stim_fr_seqs), max(pad_seql)), np.nan)

        last_idx = 0
        for i in range(len(stim_fr_seqs)):
            traces[:, i, 
                   :pad_seql[i]] = traces_flat[:, last_idx:last_idx+pad_seql[i]]
            last_idx += pad_seql[i]

        return traces

 

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
            - seg_ps_nobl (float)         : average number of segments per 
                                            second in a block, excluding blank 
                                            segments
            - seg_ps_wibl (float)         : average number of segments per 
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
            raise ValueError(('{} stim type not recognized. Stim object cannot '
                             'be initialized.').format(self.stimtype))
        
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
            raise ValueError(('Expected 2 stimulus numbers, '
                              'but got {}'.format(len(stim_n))))
        
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
            dict_str = ('\n- different values in the {} '
                        '(under {}).'.format(diff_str, overall_dict))
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
            raise ValueError(('Cannot initialize production Brick stimuli '
                              'together, due to:'
                              '{}{}{}').format(dict_str, sweep_str, sq_str))


    #############################################
    def _set_blocks(self):
        """
        self._set_blocks

        Set attributes related to blocks and display sequences. Also checks
        whether expected number of blocks were shown and whether they 
        comprised the expected number of segments.

        Note: A block is a sequence of stimulus presentations of the same 
        stimulus type, and there can be multiple blocks in one experiment. 
        For Gabors, segments refer to each gabor frame (lasting 0.3 s). For 
        Bricks, segments refer to 1s of moving bricks. 
        
        Note: Grayscr segments are not omitted when a session's segments are 
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
                                            list, structures as 
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
            print('WARNING: {} {} blocks started instead of the expected {}. \n'
                  .format(self.act_n_blocks, self.stimtype, self.exp_n_blocks))            
            if self.act_n_blocks > self.exp_n_blocks:
                self.extra_segs = (float(tot_disp) - \
                                   self.exp_n_blocks*block_len)*self.seg_ps_wibl 
                print(('WARNING: In total, {} extra segments were shown, '
                       'including blanks. \n').format(self.extra_segs))
    
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
                    print(('WARNING: During last sequence of {}, the last '
                          'blank segment of the {}. block was omitted. \n')
                          .format(self.stimtype, n_bl))
                else:
                    print(('WARNING: During {}. sequence of {}, the last blank '
                          'segment of the {}. block was pushed to the start of '
                          'the next sequence. \n').format(i+1, self.stimtype, 
                                                          n_bl))
            elif rem_seg > 1:

                if i == len(self.disp_seq)-1:
                    print(('WARNING: During last sequence of {}, {} segments '
                         '(incl. blanks) from the {}. block were omitted. \n')
                         .format(self.stimtype, rem_seg, n_bl))
                else:
                    print(('WARNING: During {}. sequence of {}, {} segments '
                          '(incl. blanks) from the {}. block were pushed to '
                          'the next sequence. These segments will be omitted '
                          'from analysis. \n').format(i+1, self.stimtype, 
                                                      rem_seg, n_bl))
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
                
                try:
                    min_idx = self.stim_seg_list.index(j[0])
                    max_idx = len(self.stim_seg_list)-1 - \
                                self.stim_seg_list[::-1].index(j[1]-1) + 1 
                                # 1 added as range end is excluded
                except:
                    pdb.set_trace()
                temp.append([min_idx, max_idx])
            self.block_ran_fr.append(temp)
        
        self.block_len_fr = np.diff(self.block_ran_fr).squeeze(2).tolist()


    #############################################
    def get_twop_fr_per_seg(self, seglist, first=False):
        """
        self.get_twop_fr_per_seg(seglist)

        Returns a list of arrays containing the 2-photon frame numbers that 
        correspond to a given set of stimulus segments provided in a list 
        for a specific stimulus.

        Required args:
            - seglist (list of ints): the stimulus segments for which to get 
                                      2p frames

        Optional args:
            - first (bool): return only first frame for each seg
                            default: Falsle
        Returns:
            - frames (list of int arrays): a list (one entry per segment) of
                                           arrays containing the 2p frame 
                                           numbers OR (if first is True) a 
                                           list of first 2p frames numbers for 
                                           each segment
        """

        # initialize the frames list
        frames = []

        # get the rows in the alignment dataframe that correspond to the segments
        rows = self.sess.align_df.loc[(self.sess.align_df['stimType'] == self.stimtype[0]) &
                                      (self.sess.align_df['stimSeg'].isin(seglist))]

        # get the start frames and end frames from each row
        start_frames = rows['start_frame'].values
        if not first:
            end_frames = rows['end_frame'].values

            # build arrays for each segment
            for r in range(start_frames.shape[0]):
                frames.append(np.arange(start_frames[r],end_frames[r]))
        else:
            frames = start_frames

        return frames


    #############################################
    def get_n_twop_fr_per_seg(self, segs):
        """
        self.get_n_twop_fr_per_seg(segs)

        Returns a list with the number of twop frames for each seg passed.    

        Required args:
            - segs (list): list of segments

        Returns:
            - n_fr_sorted (list): list of number of frames in each segment
        """

        segs = gen_util.list_if_not(segs)

        segs_unique = sorted(set(segs))
        
        # number of frames will be returned in ascending order of seg number
        n_fr = self.sess.align_df.loc[(self.sess.align_df['stimType'] == self.stimtype[0]) &
                                      (self.sess.align_df['stimSeg'].isin(segs_unique))]['num_frames'].tolist()
        
        # resort based on order in which segs were passed and include any 
        # duplicates
        n_fr_sorted = [n_fr[segs_unique.index(seg)] for seg in segs]
        
        return n_fr_sorted


    #############################################
    def get_segs_by_criteria(self, stimPar1='any', stimPar2='any', surp='any', 
                             stimSeg='any', gabfr='any', start_frame='any', 
                             end_frame='any', num_frames='any', gabk=None, 
                             gab_ori=None, bri_size=None, bri_dir=None, 
                             remconsec=False, by='block'):
        """
        self.get_segs_by_criteria()

        Returns a list of stimulus seg numbers that have the specified values 
        in specified columns in the alignment dataframe.    

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
            - gabfr (str, int or list)    : gaborframe value(s) of interest 
                                             (0, 1, 2, 3)
                                             default: 'any'
            - start_frame_min (str or int): minimum of 2p start_frame range of 
                                            interest 
                                            default: 'any'
            - start_frame_max (str or int): maximum of 2p start_frame range of 
                                            interest (excl)
                                            default: 'any'
            - end_frame_min (str or int)  : minimum of 2p end_frame range of 
                                            interest
                                            default: 'any'
            - end_frame_max (str or int)  : maximum of 2p end_frame range of 
                                            interest (excl)
                                            default: 'any'
            - num_frames_min (str or int) : minimum of num_frames range of 
                                            interest
                                            default: 'any'
            - num_frames_max (str or int) : maximum of num_frames range of 
                                            interest (excl)
                                            default: 'any'
            - gabk (int or list)          : if not None, will overwrite 
                                            stimPar2 (4, 16, or 'any')
                                            default: None
            - gab_ori (int or list)       : if not None, will overwrite 
                                            stimPar1 (0, 45, 90, 135, or 'any')
                                            default: None
            - bri_size (int or list)      : if not None, will overwrite 
                                            stimPar1 (128, 256, or 'any')
                                            default: None
            - bri_dir (str or list)       : if not None, will overwrite 
                                            stimPar2 ('right', 'left' or 'any')
                                            default: None
            - remconsec (bool)              if True, consecutive segments are 
                                            removed within a block
                                            default: False
            - by (str)                    : determines whether segment numbers
                                            are returned in a flat list ('seg'),
                                            grouped by block ('block'), or 
                                            further grouped by display  
                                            sequence ('disp')
                                            default: 'block'
        
        Returns:
            - segs (list): list of seg numbers that obey the criteria
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
                pars[i] = gen_util.get_df_vals(self.sess.align_df, 'stimType', 
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
        stimPar1 = gen_util.get_df_label_vals(self.sess.align_df, 
                                              'stimPar1', stimPar1)
        stimPar2 = gen_util.get_df_label_vals(self.sess.align_df, 
                                              'stimPar2', stimPar2)
        surp     = gen_util.get_df_label_vals(self.sess.align_df, 
                                              'surp', surp)
        stimSeg  = gen_util.get_df_label_vals(self.sess.align_df, 
                                              'stimSeg', stimSeg)
        # here, ensure that the -1s are removed
        stimSeg = gen_util.remove_if(stimSeg, -1)
        gabfr   = gen_util.get_df_label_vals(self.sess.align_df, 
                                             'GABORFRAME', gabfr)
        
        if start_frame in ['any', None]:
            start_frame_min = int(self.sess.align_df['start_frame'].min())
            start_frame_max = int(self.sess.align_df['start_frame'].max()+1)
        if end_frame in ['any', None]:
            end_frame_min = int(self.sess.align_df['end_frame'].min())
            end_frame_max = int(self.sess.align_df['end_frame'].max()+1)
        if num_frames in ['any', None]:
            num_frames_min = int(self.sess.align_df['num_frames'].min())
            num_frames_max = int(self.sess.align_df['num_frames'].max()+1)
        
        segs = []
        for i in self.block_ran_seg:
            temp = []
            for j in i:
                idxs = self.sess.align_df.loc[(self.sess.align_df['stimType']==self.stimtype[0])     & 
                                              (self.sess.align_df['stimPar1'].isin(stimPar1))        &
                                              (self.sess.align_df['stimPar2'].isin(stimPar2))        &
                                              (self.sess.align_df['surp'].isin(surp))                &
                                              (self.sess.align_df['stimSeg'].isin(stimSeg))          &
                                              (self.sess.align_df['GABORFRAME'].isin(gabfr))         &
                                              (self.sess.align_df['start_frame'] >= start_frame_min) &
                                              (self.sess.align_df['start_frame'] < start_frame_max)  &
                                              (self.sess.align_df['end_frame'] >= end_frame_min)     &
                                              (self.sess.align_df['end_frame'] < end_frame_max)      &
                                              (self.sess.align_df['num_frames'] >= num_frames_min)   &
                                              (self.sess.align_df['num_frames'] < num_frames_max)    &
                                              (self.sess.align_df['stimSeg'] >= j[0])                &
                                              (self.sess.align_df['stimSeg'] < j[1])]['stimSeg'].tolist()
                
                # if removing consecutive values
                if remconsec: 
                    idxs_new = []
                    for k, val in enumerate(idxs):
                        if k == 0 or val != idxs[k-1]+1:
                            idxs_new.extend([val])
                    idxs = idxs_new
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
                                start_frame='any', end_frame='any', 
                                num_frames='any', gabk=None, gab_ori=None, 
                                bri_size=None, bri_dir=None, first_fr=True, 
                                remconsec=False, by='block'):
        """
        self.get_stim_fr_by_criteria()

        Returns a list of stimulus frames numbers that have the specified 
        values in specified columns in the alignment dataframe. 
        
        Will return frame numbers only for the current stim object.

        Note: grayscreen frames are NOT returned

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
            - gabfr (str, int or list)    : gaborframe value(s) of interest 
                                             (0, 1, 2, 3)
                                             default: 'any'
            - start_frame_min (str or int): minimum of 2p start_frame range of 
                                            interest 
                                            default: 'any'
            - start_frame_max (str or int): maximum of 2p start_frame range of 
                                            interest (excl)
                                            default: 'any'
            - end_frame_min (str or int)  : minimum of 2p end_frame range of 
                                            interest
                                            default: 'any'
            - end_frame_max (str or int)  : maximum of 2p end_frame range of 
                                            interest (excl)
                                            default: 'any'
            - num_frames_min (str or int) : minimum of num_frames range of 
                                            interest
                                            default: 'any'
            - num_frames_max (str or int) : maximum of num_frames range of 
                                            interest (excl)
                                            default: 'any'
            - gabk (int or list)          : if not None, will overwrite 
                                            stimPar2 (4, 16, or 'any')
                                            default: None
            - gab_ori (int or list)       : if not None, will overwrite 
                                            stimPar1 (0, 45, 90, 135, or 'any')
                                            default: None
            - bri_size (int or list)      : if not None, will overwrite 
                                            stimPar1 (128, 256, or 'any')
                                            default: None
            - bri_dir (str or list)       : if not None, will overwrite 
                                            stimPar2 ('right', 'left' or 'any')
                                            default: None
            - remconsec (bool)              if True, consecutive segments are 
                                            removed within a block
                                            default: False
            - by (str)                    : determines whether frame numbers 
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
                                         gabfr, start_frame, end_frame, 
                                         num_frames, gabk, gab_ori, bri_size, 
                                         bri_dir, remconsec, by='disp')

        frames = []
        for i in segs:
            temp = []
            for idxs in i:
                temp2 = []
                if first_fr: # if getting only first frame for each segment
                    for val in idxs:
                        temp2.extend([self.stim_seg_list.index(val)])
                else: # if getting all frames for each segment
                    stim_seg_list_array = np.asarray(self.stim_seg_list)
                    for val in idxs:
                        all_fr = np.where(stim_seg_list_array == val)[0]
                        temp2.extend(all_fr.tolist())
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
    def get_run_array(self, stim_ref_fr, pre, post, integ=False):
        """
        self.get_run_array(stim_ref_fr, pre, post)

        Returns array of run data around specific stimulus frame numbers. 

        Required args:
            - stim_ref_fr (list): 1D list of reference stimulus frame numbers
                                  around which to retrieve running data 
                                  (e.g., all 1st Gabor A frames)
            - pre (float)       : range of frames to include before each 
                                  reference frame number (in s)
            - post (float)      : range of frames to include after each 
                                  reference frame number (in s)
        
        Optional args:
            - integ (bool)  : if True, running is integrated over frames
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
        fr_idx = gen_util.num_ranges(stim_ref_fr, pre=-ran_fr[0], leng=len(xran))
                     
        # remove sequences with negatives or values above total number of stim 
        # frames
        neg_idx  = np.where(fr_idx[:,0] < 0)[0].tolist()
        over_idx = np.where(fr_idx[:,-1] >= self.sess.tot_stim_fr)[0].tolist()
        
        fr_idx = gen_util.remove_idx(fr_idx, neg_idx + over_idx, axis=0)

        data_array = self.sess.run[fr_idx]

        if integ:
            data_array = math_util.integ(data_array, 1./self.sess.stim_fps, 
                                         axis=1)

        return xran, data_array


    #############################################
    def get_run_array_stats(self, stim_ref_fr, pre, post, integ=False,
                            ret_arr=False, stats='mean', error='std'):
        """
        self.get_run_array_stats(stim_ref_fr, pre, post)

        Returns stats (mean and std or median and quartiles) for sequences of 
        running data around specific stimulus frames.

        Required args:
            - stim_ref_fr (list): 1D list of reference stimulus frames numbers
                                  around which to retrieve running data 
                                  (e.g., all 1st Gabor A frames)
            - pre (float)       : range of frames to include before each 
                                  reference frame number (in s)
            - post (float)      : range of frames to include after each 
                                  reference frame number (in s)

        Optional args:
            - integ (bool)  : if True, dF/F is integrated over sequences
                              default: False
            - ret_arr (bool): also return running data array, not just  
                              statistics
                              default: False 
            - stats (str)   : return mean ('mean') or median ('median')
                              default: 'mean'
            - error (str)   : return std dev/quartiles ('std') or SEM/MAD 
                              ('sem')
                              default: 'sem'
         
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

        xran, data_array = self.get_run_array(stim_ref_fr, pre, post, integ)

        all_data = self.get_array_stats(xran, data_array, ret_arr, axes=0, 
                                        stats=stats, error=error, integ=integ)

        if ret_arr:
            xran, data_stats, data_array = all_data
            return xran, data_stats, data_array
        
        else:
            xran, data_stats = all_data
            return xran, data_stats


    #############################################
    def get_roi_trace_array(self, twop_ref_fr, pre, post, fluor='dff', 
                            integ=False, remnans=True):
        """
        self.get_roi_trace_array(twop_ref_fr, pre, post)

        Returns an array of 2p trace data around specific 2p frame numbers. 

        Required args:
            - twop_ref_fr (list): 1D list of 2p frame numbers 
                                  (e.g., all 1st Gabor A frames)
            - pre (float)       : range of frames to include before each 
                                  reference frame number (in s)
            - post (float)      : range of frames to include after each 
                                  reference frame number (in s)

        Optional args:
            - fluor (str)   : if 'dff', dF/F is used, if 'raw', ROI traces
                              default: 'raw'
            - integ (bool)  : if True, dF/F is integrated over frames
                              default: False
            - remnans (bool): if True, removes ROIs with NaN/Inf values 
                              anywhere in session. If False, NaN values (but
                              not Inf values) are omitted in calculating the 
                              data statistics.
                              default: True
         
        Returns:
            - xran (1D array)           : time values for the 2p frames
            - data_array (2 or 3D array): roi trace data, structured as 
                                          ROI x sequences (x frames)
        """
        
        fr_idx, xran = self.sess.get_twop_fr_ran(twop_ref_fr, pre, post)

        # get dF/F: ROI x seq x fr
        data_array = self.sess.get_roi_seqs(fr_idx, fluor=fluor, 
                                            remnans=remnans)

        if integ:
            if remnans:
                nanpol = None
            else:
                nanpol = 'omit'
            data_array = math_util.integ(data_array, 1./self.sess.twop_fps, 
                                         axis=2, nanpol=nanpol)
        
        return xran, data_array
    
    
    #############################################
    def get_roi_trace_stats(self, twop_ref_fr, pre, post, byroi=True, 
                            fluor='dff', integ=False, remnans=True, 
                            ret_arr=False, stats='mean', error='std'):
        """
        self.get_roi_trace_stats(twop_ref_fr, pre, post)

        Returns stats (mean and std or median and quartiles) for sequences of 
        roi traces centered around specific 2p frame numbers.

        Required args:
            - twop_ref_fr (list): 1D list of 2p frame numbers 
                                  (e.g., all 1st Gabor A frames)
            - pre (float)       : range of frames to include before each 
                                  reference frame number (in s)
            - post (float)      : range of frames to include after each  
                                  reference frame number (in s)

        Optional args:
            - byroi (bool)  : if True, returns statistics for each ROI. If 
                              False, returns statistics across ROIs
                              default: True 
            - fluor (str)   : if 'dff', dF/F is used, if 'raw', ROI traces
                              default: 'raw'
            - integ (bool)  : if True, dF/F is integrated over sequences
                              default: False
            - remnans (bool): if True, removes ROIs with NaN/Inf values 
                              anywhere in session. If False, NaN values (but
                              not Inf values) are omitted in calculating the 
                              data statistics.
                              default: True
            - ret_arr (bool): also return ROI trace data array, not just  
                              statistics.
            - stats (str)   : return mean ('mean') or median ('median')
                              default: 'mean'
            - error (str)   : return std dev/quartiles ('std') or SEM/MAD 
                              ('sem')
                              default: 'sem'

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
                                                    fluor, integ,
                                                    remnans=remnans)

        # order in which to take statistics on data
        axes = [1, 0]
        if byroi:
            axes = 1

        if not remnans:
            nanpol = 'omit'
        else:
            nanpol = None

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
    def get_run(self, by='block'):
        """
        self.get_run()

        Returns run values for each stimulus frame of each stimulus block.

        Optional args:
            - by (str): determines whether run values are returned in a flat 
                        list ('frame'), grouped by block ('block'), or further 
                        grouped by display sequence ('disp')
                        default: 'block'

        Returns:
            - run (list): list of running values for stimulus blocks
        """
        
        run = []
        for i in self.block_ran_fr:
            temp = []
            for j in i:
                temp.append(self.sess.run[j[0]: j[1]].tolist())
            run.append(temp)

        # if not returning by disp
        if by == 'block' or by == 'frame':
            run = [x for sub in run for x in sub]
            if by == 'frame':
                run = [x for sub in run for x in sub]
        elif by != 'disp':
            raise ValueError(('\'by\' can only take the values \'disp\', '
                             '\'block\' or \'frame\'.'))
    
        return run

    
    #############################################
    def get_segs_per_twopfr(self, twop_fr):
        """
        self.get_segs_per_twopfr(twop_fr)

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

        # perform linear interpolation on the running speed
        segs = np.interp(twop_fr, self.sess.stim_align, self.stim_seg_list)

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

            - deg_per_pix (float)     : degrees per pixels used in conversion
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
            - phase (float)           : phase of the gabors (0-1)
            - pos (3D array)          : gabor positions for each segment type
                                        (A, B, C, D, E), in pixels with window
                                        center being (0, 0), structured as:
                                            segment type x gabor x coord (x, y) 
            - post (float)            : number of seconds from frame A that are
                                        included in a set (gray, A, B, C, D/E)
            - pre (float)             : number of seconds before frame A that
                                        are included in a set 
                                        (gray, A, B, C, D/E)
            - set_len_s (float)       : length of a set in seconds
                                        (set: gray, A, B, C, D/E)
            - sf (float)              : spatial frequency of the gabors 
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

        Stim.__init__(self, sess, stim_n, stimtype='gabors')

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
        self.oris      = gabor_par['oris']
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
        seg_surps = np.asarray(self.sess.align_df.loc[(self.sess.align_df['stimType'] == 'g')]['surp'])
        seg_gabfr = np.asarray(self.sess.align_df.loc[(self.sess.align_df['stimType'] == 'g')]['GABORFRAME'])
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
        self.pre  = 1*self.seg_len_s # 0.3 s blank
        self.post = self.n_seg_per_set*self.seg_len_s # 1.2 ms gabors
        self.set_len_s = self.pre+self.post
        
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
                segs = self.sess.align_df.loc[(self.sess.align_df['stimType']==self.stimtype[0]) & 
                                              (self.sess.align_df['stimSeg'] >= block[0]) & 
                                              (self.sess.align_df['stimSeg'] < block[1])]
                # skipping stimPar1 which indicates gabor orientations which 
                # change at each gabor sequence presentation
                stimPar2 = segs['stimPar2'].unique().tolist()

                if len(stimPar2) > 1:
                    block_n = i*len(self.block_ran_seg)+j+1
                    raise ValueError(('Block {} of {} comprises segments with '
                                      'different stimPar2 '
                                      'values: {}').format(block_n, 
                                                           self.stimtype, 
                                                           stimPar2))
                block_par.append(stimPar2[0])
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
            raise NotImplementedError(('Function not properly implemented if '
                                       'the minimum segment is not 0.'))
        if max_seg != self.oris_pr.shape[0]:
            raise NotImplementedError(('Function not properly implemented if '
                                    'the maximum segment is not the same '
                                    'as the number of orientations recorded.'))

        # check that at least one parameter type is requested
        if not(pos or ori or size):
            raise ValueError(('At least one of the following must be True: '
                              'pos, ori, size.'))
        
        segs = np.asarray(segs)

        if pos or size:
            segs_max = np.max(segs)
            
            # values will be returned in ascending order of seg number
            gabfr = self.sess.align_df.loc[(self.sess.align_df['stimType'] == 'g') &
                                        (self.sess.align_df['stimSeg'] < segs_max)]['GABORFRAME'].tolist()
            surps = self.sess.align_df.loc[(self.sess.align_df['stimType'] == 'g') &
                                        (self.sess.align_df['stimSeg'] < segs_max)]['surp'].tolist()

            # reintroduce -1s
            gabfr.append(-1)
            surps.append(-1)

            if len(gabfr) != segs_max + 1 or len(surps) != segs_max + 1:
                raise ValueError(('Something went wrong.'))

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

        Note: A warning will be thrown if any of the 2p frame sequences occur
        during Bricks frames. 
        (-1 parameter values will be returned for these frames, as if they
        were grayscreen frames.)

        Required args:
            - twop_ref_fr (list): 1D list of 2p frame numbers 
                                  (e.g., all 1st Gabor A frames)
            - pre (float)       : range of frames to include before each 
                                  reference frame number (in s)
            - post (float)      : range of frames to include after each 
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
            bri_segs = self.sess.bricks.get_segs_per_twopfr(twopfr_seqs)
            if not (bri_segs == -1).all():
                print(('\nWARNING: some of the frames requested occur while '
                       'Bricks are presented.'))

        # get seg numbers for each twopfr in each sequence
        seq_segs = self.get_segs_per_twopfr(twopfr_seqs)

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

            - deg_per_pix (float)     : degrees per pixels used in conversion
                                        to generate stimuli

            - direcs (list)           : main brick direction for each block
            - flipfrac (float)        : fraction of bricks that flip direction 
                                        at each surprise
            - n_bricks (float or list): n_bricks for each brick block (only one
                                        value for production data)
            - sizes (int or list)     : brick size for each brick block (only
                                        one value for production data) (in pix)
            - speed (float)           : speed at which the bricks are moving 
                                        (in pix/sec)
            - units (str)             : units used to create stimuli in 
                                        PsychoPy (e.g., 'pix')
        
        Required args:
            - sess (Session)  : session to which the bricks belongs
            - stim_n (int)    : this stimulus' number (2 in the case of
                                production bricks): x in 
                                sess.stim_dict['stimuli'][x]
        """

        Stim.__init__(self, sess, stim_n, stimtype='bricks')
            
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
                segs = self.sess.align_df.loc[(self.sess.align_df['stimType']==self.stimtype[0]) & 
                                              (self.sess.align_df['stimSeg'] >= block[0]) & 
                                              (self.sess.align_df['stimSeg'] < block[1])]
                stimPars = []
                for par_name in ['stimPar2', 'stimPar1']:
                    stimPar = segs[par_name].unique().tolist()
                    if len(stimPar) > 1:
                        block_n = i*len(self.block_ran_seg)+j+1
                        raise ValueError('Block {} of {} comprises segments '
                                        'with different {} '
                                        'values: {}'.format(block_n, 
                                                            self.stimtype, 
                                                            par_name, stimPar))
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
    def get_dir_frames_reg(self, by='block'):
        """
        self.get_dir_frames_reg()

        Returns two lists of stimulus frame number, the first is a list of the 
        first frame for each right moving segment. The second is a list of the 
        first frame for each left moving segment. Both lists exclude surprise 
        segments.

        Optional args:
            - by (str): determines whether frame numbers are returned in a 
                        flat list ('frame'), grouped by block ('block'), or 
                        further grouped by display sequence ('disp')
                        default: 'block'

        Returns:
            - right_frames (list): list of first stimulus frame numbers for 
                                   each right moving segment, excluding 
                                   surprise segments.
            - left_frames (list) : list of first stimulus frame numbers for 
                                   each left moving segment, excluding surprise 
                                   segments.
        """

        right_frames = self.get_stim_fr_by_criteria(bri_dir='right', surp=0, 
                                                   by=by)
        left_frames  = self.get_stim_fr_by_criteria(bri_dir='left', surp=0, 
                                                   by=by)

        return right_frames, left_frames
        

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
            raise ValueError(('No frame lists were found for either stimulus '
                             ' types (gabors, bricks.'))
        elif length == 1:
            frames_sum = np.asarray(frames)
        else:
            frames_sum = np.sum(np.asarray(frames), axis=0)
        grays = np.where(frames_sum==length*-1)[0].tolist()

        if len(grays) == 0:
            raise ValueError(('No grayscreen frames were found outside of '
                             'gabor stimulus sequences.'))

        return grays


    #############################################
    def get_first_nongab_stim_fr(self):
        """
        self.get_first_nongab_stim_fr()

        Returns every first grayscreen stimulus frame number for every 
        grayscreen sequence occuring outside of gabor stimulus blocks, and 
        the number of consecutive grayscreen stimulus frames for each sequence. 
                
        Note that any grayscreen stimulus frames for sequences flanking gabor 
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
                raise ValueError(('\'by\' can only take the values \'disp\', '
                                 '\'block\' or \'frame\'.'))
            
            return gab_grays
        else:
            raise ValueError(('Session does not have a gabors attribute. Be '
                              'sure to extract stim info and check that '
                              'session contains a gabor stimulus.'))


    #############################################    
    def get_gab_gray_stim_fr(self, by='block'):
        """
        self.get_gab_gray_stim_fr()

        Returns every first grayscreen stimulus frame number for every 
        grayscreen sequence occuring during a gabor stimulus blocks, and the 
        number of consecutive grayscreen stimulus frames for each sequence. 
                
        Note that any grayscreen stimulus frames for sequences flanking gabor 
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
            raise ValueError(('\'by\' can only take the values \'disp\', '
                             '\'block\' or \'frame\'.'))

        return first_gab_grays, n_gab_grays


