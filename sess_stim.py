"""
Classes to store and extract session and stimulus information.
"""
import os

import numpy as np
import pickle
import h5py
import pandas
import scipy.stats

import sync_util


class Session(object):
    """
    Session object contains information relevant to a session, particularly needed
    analyse running and stimulus. Two-photon component not currently implemented.
    """
    def __init__(self, filename):
        self.filename = filename
        self._init_filenames()
        
    def _init_filenames(self):
        """
        Creates file name attributes for the stimulus dictionary pickle, h5 sync and 
        stimulus alignment dataframe pickle from the filename attribute
        """
        # file names
        self.stim_pkl_name = self.filename + '_stim.pkl'
        self.sync_h5_name = self.filename + '_sync.h5'
        self.align_pkl_name = self.filename + '_df.pkl'
    
    # load stim dictionary as an attribute (and any other general stim info, like fps)
    def load_stim_dict(self):
        with open(self.stim_pkl_name, 'rb') as f:
            self.stim_dict = pickle.load(f)
        self.stim_fps = self.stim_dict['fps']
        self.tot_frames = self.stim_dict['total_frames']
        self.pre_blank = self.stim_dict['pre_blank_sec'] # seconds
        self.post_blank = self.stim_dict['post_blank_sec'] # seconds
        self.drop_frames = self.stim_dict['droppedframes']
        self.n_drop_frames = len(self.drop_frames[0])
        if np.float(self.n_drop_frames)/self.tot_frames > 0.0003:
            print('WARNING: {} dropped stimulus frames out of {}.'.format(self.n_drop_frames, self.tot_frames))
        
    def load_sync_h5(self):
        raise NotImplementedError('Loading h5 sync file of a session has not been implemented yet.')
    
    # load alignment pickle as an attribute
    def load_align_df(self):
        # create align_df if doesn't exist
        if not os.path.exists(self.align_pkl_name):
            stimulus_alignment = sync_util.get_stim_frames(self.stim_pkl_name, self.sync_h5_name, self.align_pkl_name)
        else:
            print('stimulus alignment pickle already exists.')

        self.align_df = pandas.read_pickle(self.align_pkl_name)       
    
    # load running speed array as an attribute
    def load_run(self):
        # running speed per stimulus frame in cm/s
        self.run_array = sync_util.get_run_speed(self.stim_pkl_name)
    
    # load additional info as attributes
    def extract_stim_run_info(self): 
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


class Stim(object):
    """Deals with the different stimuli that may appear on the screen during a session.
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
