"""
session.py

Classes to store, extract, and analyze an AIBS OpenScope session for
the Credit Assignment Project.

Authors: Colleen Gillon, Blake Richards

Date: August, 2018

Note: this code uses python 2.7.

"""
import os

import numpy as np
import pickle
import h5py
import pandas
import scipy.stats as st
import glob
import exceptions

import file_util
import sync_util

import pdb

class Session(object):
    """
    The Session object is the top-level object for analyzing a session from the AIBS
    OpenScope Credit Assignment Project. All that needs to be provided to create the
    object is the directory in which the session data directories are located and 
    the ID for the session to analyze/work with. The Session object that is created
    will contain all of the information relevant to a session, including stimulus
    information, behaviour information and pointers to the 2P data.
    """
    def __init__(self, datadir, sessionid, droptol=0.0003):
        """
        __init__(datadir, sessionid)

        Create the new Session object using the specified data directory and ID.

        Required arguments:
            - datadir (string)  : full path to the directory where session folders are stored.
            - sessionid (string): the ID for this session.

        Optional arguments:
                -droptol (float): the tolerance for percentage stimulus frames dropped, create a Warning
                                  if this condition isn't met.
                                  default = 0.0003 
        """
        self.home    = datadir
        self.session = sessionid
        self.droptol = droptol
        self._init_directory()
        
    #############################################
    def _init_directory(self):
        """
        _init_directory()

        Initialize the directory information for the session. This involves checking that the
        given directory obeys the appropriate organization scheme, determining the filenames
        for the stimulus dictionary pickle, h5 sync, stimulus alignment dataframe pickle, and
        others, and setting the experiment ID, mouse ID, and date. All of this info is stored
        in the session object.
        """

        # check that the high-level home directory exists
        if not os.path.isdir(self.home):
            raise exceptions.OSError('%s either does not exist or is not a directory', self.home)

        # set the session directory (full path)
        self.dir = os.path.join(self.home, 'ophys_session_' + self.session)

        # extract the mouse ID, and date from the stim pickle file
        pklglob    = glob.glob(os.path.join(self.dir,self.session + '*stim.pkl'))
        if len(pklglob) == 0:
            raise exceptions.OSError('Could not find stim pkl file in {}'.format(self.dir))
        else:
            pklinfo    = os.path.basename(pklglob[0]).split("_")
        self.mouse = pklinfo[1]
        self.date  = pklinfo[2]

        # extract the experiment ID from the experiment directory name
        expglob         = glob.glob(os.path.join(self.dir,'ophys_experiment*'))
        if len(expglob) == 0:
            raise exceptions.OSError('Could not find experiment directory in {}'.format(self.dir))
        else:
            expinfo    = os.path.basename(expglob[0]).split("_")
        self.experiment = expinfo[2]

        # create the filenames
        (self.expdir, self.procdir, self.stim_pkl, self.stim_sync, self.align_pkl, self.corrected, self.roi_traces, self.zstack) = \
                                        file_util.get_file_names(self.home, self.session, self.experiment, self.date, self.mouse)       
    
    #############################################
    def load_stim_dict(self):
        """
        load_stim_dict()

        Loads the stimulus dictionary from the stimulus pickle file and store a few variables
        for easy access.
        """
        
        # open the file
        try:
            with open(self.stim_pkl, 'rb') as f:
                    self.stim_dict = pickle.load(f)
        except:
            raise exceptions.IOError("Could not open {} for reading".format(self.stim_pkl))

        # store some variables for easy access
        self.stim_fps      = self.stim_dict['fps']
        self.tot_frames    = self.stim_dict['total_frames']
        self.pre_blank     = self.stim_dict['pre_blank_sec']  # seconds
        self.post_blank    = self.stim_dict['post_blank_sec'] # seconds
        self.drop_frames   = self.stim_dict['droppedframes']
        self.n_drop_frames = len(self.drop_frames[0])

        # check our drop tolerance
        if np.float(self.n_drop_frames)/self.tot_frames > self.droptol:
            raise exceptions.UserWarning('{} dropped stimulus frames out of {}.'.format(self.n_drop_frames, self.tot_frames))
        
    #############################################
    def load_sync_h5(self):
        raise NotImplementedError('Loading h5 sync file of a session has not been implemented yet.')
    
    #############################################
    def load_align_df(self):
        """
        load_align_df()

        Loads the alignment dataframe object and stores it in the Session. Note: this will
        also create a pickle file with the alignment data frame in the Session directory.
        The stimulus alignment array is also stored.
        """
        # create align_df if doesn't exist
        if not os.path.exists(self.align_pkl):
            sync_util.get_stim_frames(self.stim_pkl, self.stim_sync, self.align_pkl)
        else:
            print('NOTE: Stimulus alignment pickle already exists in {}'.format(self.dir))

        try:
            with open(self.align_pkl,"rb") as f:
                    align = pickle.load(f)
        except:
            raise exceptions.IOError("Could not read stimulus alignment pickle file {}".format(self.align_pkl))
        self.align_df   = align['stim_df']
        self.stim_align = align['stim_align']
    
    #############################################
    # load running speed array as an attribute
    def load_run(self):
        """
        load_run()

        Loads the running wheel data into the session object.
        """

        # running speed per stimulus frame in cm/s
        self.run_array = sync_util.get_run_speed(self.stim_pkl)
    
    #############################################
    # load additional info as attributes
    def extract_stim_info(self):
        """
        extract_stim_info()

        
        """

        # load the simulus and running 
        if not hasattr(self, 'stim_dict'):
            self.load_stim_dict()
        if not hasattr(self, 'align_df'):
            self.load_align_df()
        if not hasattr(self, 'run_array'):
            self.load_run()
        
        self.stim_types = []
        self.n_stims = len(self.stim_dict['stimuli'])
        self.stims = []
        for i in range(self.n_stims):
            stim = self.stim_dict['stimuli'][i]
            stim_type = stim['stimParams']['elemParams']['name']
            self.stim_types.extend([stim_type])   
            if stim_type == 'gabors':
                self.gabors = Gabors(self, i)
            elif stim_type == 'bricks':
                self.bricks = Bricks(self, i)
        self.grayscr = Grayscr(self)

    #############################################
    def load_roi_traces(self):
        """
        load_roi_traces()

        Loads some basic information about ROI dF/F traces. This includes
        the number of ROIs, their names, and the number of data points in the traces.
        """
        
        try:
            # open the roi file and get the info
            with h5py.File(self.roi_traces,'r') as f:
                
                # get the names of the rois
                self.roi_names = f['roi_names'].value.tolist()

                # get the number of rois
                self.nroi = len(self.roi_names)

                # get the number of data points in the traces
                self.nframes = f['data'].shape[1]
        except:
            raise exceptions.IOError("Could not open {} for reading".format(self.roi_traces))

    #############################################
    def get_run_speed(self, frames):
        """
        get_run_speed(frames)

        Returns the running speed for the given two-photon imaging
        frames using linear interpolation.

        Required arguments:
            - frames (int array): set of 2p imaging frames to give speed for
        
        Returns:
            - speed (float array): running speed (in cm/s) - CHECK THIS
        """

        # make sure the frames are all legit
        if any(frames >= self.nframes) or any(frames < 0):
            raise UserWarning("Some of the specified frames are out of range")

        # perform linear interpolation on the running speed
        speed = np.interp(frames, self.stim_align, self.run_array)

        return speed

    #############################################
    def get_roi_traces(self, frames, rois='all'):
        """
        get_roi_traces(frames)

        Returns the processed ROI dF/F traces for the given two-photon imaging
        frames and specified ROIs.

        Required arguments:
            - frames (int array): set of 2p imaging frames to give ROI dF/F for, if
                                  any frames are out of range then NaNs returned

        Optional arguments:
            - rois (int array): set of ROIs to return traces for, if string 'all'
                                is provided then all ROIs are returned, if an ROI
                                that doesn't exist is requested then NaNs returned
                                for that ROI
                                default = 'all'
        Returns:
            - traces (float array): array of dF/F for the specified frames/ROIs
        """
        
        # make sure the frames are all legit
        if any(frames >= self.nframes) or any(frames < 0):
            raise UserWarning("Some of the specified frames are out of range")

        # make sure the rois are all legit
        if rois is 'all':
            rois = np.arange(self.nroi)
        else:
            if any(rois >= self.nroi) or any(rois < 0):
                rois[rois >= self.nframes] = -1
                rois[rois < 0] = -1
                raise UserWarning("Some of the specified ROIs do not exist, NaNs will be returned")

        # initialize the return array
        traces = np.zeros((len(rois),len(frames))) + np.nan

        # read the data points into the return array
        with h5py.File(self.roi_traces,'r') as f:
            for roi in rois:
                if roi >= 0:
                    try:
                        traces[roi,:] = f['data'].value[roi,frames]
                    except:
                        raise exceptions.IOError("Could not read ROI number {}".format(roi))

        return traces
 
class Stim(object):
    """
    The Stim object is a higher level class for describing stimulus properties. It should
    be extended with other classes containing stimulus specific information. The only
    core structure to all stimuli is blocks and segments. A block is a sequence of
    stimulus presentations of the same stimulus type, and there can be multiple
    blocks in one experiment. Segments refer to an individual stimulus "sweeps" per
    the AIBS stimulus. 
    """
    def __init__(self, sess, stim_n):
        self.sess = sess
        self._init_attribs(stim_n)
        
    def _init_attribs(self, stim_n):
        if self.stim_type == 'grayscr':
            self._get_stim_disps()
        else:
            self.stim_n = stim_n
            self.stim = self.sess.stim_dict['stimuli'][self.stim_n]
            self.blank_seg = self.sess.stim_dict['stimuli'][self.stim_n]['blank_sweeps']
            self._get_stim_disps()
            self._get_frames()
            self._get_running()
    
    def _get_stim_disps(self):
        """Get display sequences for each stimulus.
        Note: These display sequences take into account the pre and post blank periods. (So they are shifted.)
        """
        disps = []
        for i in range(self.sess.n_stims):
            # add pre-blank sec to all display seq
            disps.append(self.sess.stim_dict['stimuli'][i]['display_sequence']+self.sess.pre_blank) 
        if self.stim_type == 'gabors' or self.stim_type == 'bricks': 
            self.disp_seq = disps[self.stim_n] # stimulus display seq (in sec)
            self.block_len = np.diff(self.disp_seq) # length in sec of each display period
        elif self.stim_type == 'grayscr':
            # display seq times are calculated as periods before, after and between stimulus block
            stim_disps_2D = []
            for x in disps:
                stim_disps_2D.append(x[0])
            stim_disps_2D = np.asarray(stim_disps_2D)
            stim_disps_2D = stim_disps_2D[stim_disps_2D[:,1].argsort()] # sort from low to high start time
            self.disp_seq = []
            if stim_disps_2D[0][0] != 0: # checking if grayscr occurs before first stim
                self.disp_seq.append([0, stim_disps_2D[0][0]])
            for i in range(len(stim_disps_2D)-1):
                # add intervening grayscreen display times
                self.disp_seq.append([stim_disps_2D[i][1], stim_disps_2D[i+1][0]])
            if self.sess.post_blank != 0: # checking if grayscr occurs after last stim
                self.disp_seq.append([stim_disps_2D[i+1][0], stim_disps_2D[i+1][0]+self.sess.post_blank])
            self.block_len = np.diff(self.disp_seq)
        else:
            raise ValueError('Stimulus type \'{}\' not recognized.'.format(self.stim_type))
        self.n_blocks = len(self.disp_seq)
        
    def _get_frames(self):
        # fill out the frame_list to be the same length as running array
        self.frame_list = int(self.sess.pre_blank*self.sess.stim_fps)*[-1] + self.stim['frame_list'].tolist() + \
                          int((self.sess.tot_frames - len(self.stim['frame_list']) + \
                          self.sess.post_blank)*self.sess.stim_fps)*[-1] 
        self.seg_range = [] # range of segments in each block
        self.frame_n = [] # frame numbers for each block
        self.n_frames = [] # number of frames in each block
        min_seg = 0
        up_lim = 0
        for i in range(self.n_blocks):
            # get segment ranges
            up_lim += int(np.diff(self.disp_seq[i])/self.seg_len_s)
            max_seg = np.max(self.sess.stim_dict['stimuli'][self.stim_n]['sweep_order'][:up_lim]) # max segment
            self.seg_range.append([min_seg, max_seg])
            # get frame ranges
            min_ind = self.frame_list.index(min_seg)    
            max_ind = len(self.frame_list)-1 - self.frame_list[::-1].index(max_seg)
            self.frame_n.append([min_ind, max_ind])
            # get number of frames in block
            length = max_ind-min_ind+1
            sess_length = self.block_len[i][0]*self.sess.stim_fps
            self.n_frames.extend([length])
            min_seg = max_seg + 1
    
    def _get_running(self):
        """
        """
        self.run = []
        for i in range(self.n_blocks):
            self.run.extend([self.sess.run_array[self.frame_n[i][0]:self.frame_n[i][1]]])
    
    
class Gabors(Stim):
    """Inherits from Stim class.
    Deals with information related to sequences where the gabors are being displayed, including the 300 ms segments
    where the screen is gray.
    """
    def __init__(self, sess, stim_n):
        self.stim_type = 'gabors'
        self.seg_ps = 1/1.5*4 # 4 gabor segments (and 1 blank segment) per 1.5s
        self.seg_len_s = 0.3 # in sec
        self.surp_min_s = 3
        self.surp_max_s = 6
        self.reg_min_s = 30
        self.reg_max_s = 90
        self.pre = 1*self.seg_len_s # 300 ms blank
        self.post = 4*self.seg_len_s # 1200 ms gabors
        
        Stim.__init__(self, sess, stim_n)
        self._get_A_frames()
        self._get_surp_frames()
    
    def _get_A_frames(self):
        seg_ind = self.sess.align_df.loc[(self.sess.align_df['stimType'] == 1) & (self.sess.align_df['GABORFRAME'] == 0)]['stimSeg'].tolist()
        self.A_frame_n = []
        for i in seg_ind:
            self.A_frame_n.extend([self.frame_list.index(i)])
    
    def _get_surp_frames(self):
        seg_ind_all_surp = self.sess.align_df.loc[(self.sess.align_df['stimType'] == 1) & (self.sess.align_df['surp'] == 1)]['stimSeg'].tolist()
        seg_ind_all_nosurp = self.sess.align_df.loc[(self.sess.align_df['stimType'] == 1) & (self.sess.align_df['surp'] == 0)]['stimSeg'].tolist()
        self.surp_frame_n = []
        self.first_surp_frame_n = []
        self.nosurp_frame_n = []
        self.first_nosurp_frame_n = []
        for i, j in enumerate(seg_ind_all_surp):
            self.surp_frame_n.extend([self.frame_list.index(j)])
            # get only non-consecutive seg to get only first surprise
            if i == 0 or j != seg_ind_all_surp[i-1]+1:
                self.first_surp_frame_n.extend([self.frame_list.index(j)])
        for i, j in enumerate(seg_ind_all_nosurp):
            self.nosurp_frame_n.extend([self.frame_list.index(j)])
            if i == 0 or j != seg_ind_all_nosurp[i-1]+1:
                self.first_nosurp_frame_n.extend([self.frame_list.index(j)])
    
class Bricks(Stim):
    """Inherits from Stim class.
    Deals with information related to sequences where the bricks are being displayed.
    """
    def __init__(self, sess, stim_n):
        self.stim_type = 'bricks'
        self.seg_ps = 1 # 1 segment per second
        self.seg_len_s = 1 # 1 segment is 1 second
        self.surp_min_s = 2
        self.surp_max_s = 4
        self.reg_min_s = 30
        self.reg_max_s = 90
        
        Stim.__init__(self, sess, stim_n)
        self._get_surp_frames()
        self._get_dir_frames()
        

    def _get_dir_frames(self):
        # does not include surprise frames
        for i in ['left', 'right']:
            seg_ind = self.sess.align_df.loc[(self.sess.align_df['stimType'] == 0) & (self.sess.align_df['surp'] == 0) &
                                           (self.sess.align_df['stimPar2'] == i)]['stimSeg'].tolist()
            frame_n = []
            for j in seg_ind:
                frame_n.extend([self.frame_list.index(j)])
            if i == 'left':
                self.left_frame_n = frame_n[:]
            elif i == 'right':
                self.right_frame_n = frame_n[:]
    
    def _get_surp_frames(self):
        seg_ind_all_surp = self.sess.align_df.loc[(self.sess.align_df['stimType'] == 0) & (self.sess.align_df['surp'] == 1)]['stimSeg'].tolist()
        seg_ind_all_nosurp = self.sess.align_df.loc[(self.sess.align_df['stimType'] == 0) & (self.sess.align_df['surp'] == 0)]['stimSeg'].tolist()
        self.surp_frame_n = []
        self.first_surp_frame_n = []
        self.nosurp_frame_n = []
        self.first_nosurp_frame_n = []
        for i, j in enumerate(seg_ind_all_surp):
            self.surp_frame_n.extend([self.frame_list.index(j)])
            # get only non-consecutive seg to get only first surprise
            if i == 0 or j != seg_ind_all_surp[i-1]+1:
                self.first_surp_frame_n.extend([self.frame_list.index(j)])
        for i, j in enumerate(seg_ind_all_nosurp):
            self.nosurp_frame_n.extend([self.frame_list.index(j)])
            if i == 0 or j != seg_ind_all_nosurp[i-1]+1:
                self.first_nosurp_frame_n.extend([self.frame_list.index(j)])
        

class Grayscr(Stim):
    """Inherits from Stim class.
    Deals with information related to sequences where the screen is gray.
    """
    
    def __init__(self, sess):
        self.stim_type = 'grayscr'
        
        Stim.__init__(self, sess, None)
        self.min_s = 60 # hard coding a minimum secto allow short grayscr to be excluded
        self._get_frames()
        self._get_running()
        
    def _get_frames(self):
        all_stim_frames = []
        for i in range(self.sess.n_stims): 
            stim_frames = self.sess.stim_dict['stimuli'][i]['frame_list'].tolist()
            stim_frames = int(self.sess.pre_blank*self.sess.stim_fps)*[-1] + stim_frames + int((self.sess.tot_frames - \
                          len(stim_frames) + self.sess.post_blank*self.sess.stim_fps))*[-1]   
            all_stim_frames.append(np.asarray(stim_frames))
        all_stim_frames = np.asarray(all_stim_frames)
        all_frames_sum = np.sum(all_stim_frames, axis=0, dtype=int)
        
        # get the start-end of grayscr
        pos = 0
        self.frame_n_all = []
        self.n_frames_all = []
        self.frame_n_excl = []
        self.n_frames_excl = []
        for i in range(len(all_frames_sum)):
            if pos == 0 and all_frames_sum[i] == -1*self.sess.n_stims:
                start = i
                pos = 1
            elif pos == 1 and all_frames_sum[i] != -1*self.sess.n_stims:
                pos = 0
                self.frame_n_all.append([start, i])
                self.n_frames_all.extend([i-start])
                if (i-start) > self.min_s:
                    self.frame_n_excl.append([start, i])
                    self.n_frames_excl.extend([i-start])
        if pos == 0:
            self.frame_n_all.append([start, i+1])
            self.n_frames_all.extend([i+1-start])
            if (i+1-start) > self.min_s:
                    self.frame_n_excl.append([start, i+1])
                    self.n_frames_excl.extend([i+1-start])
    
    def _get_running(self):
        self.run_all = [] # includes segments within stimuli (e.g., gabors)
        self.run_excl = []
        for i in range(len(self.frame_n_all)):
            self.run_all.extend([self.sess.run_array[self.frame_n_all[i][0]:self.frame_n_all[i][1]]])
            if (self.n_frames_all[i]) > self.min_s:
                self.run_excl.extend([self.sess.run_array[self.frame_n_all[i][0]:self.frame_n_all[i][1]]])
