"""
session.py

Classes to store, extract, and analyze an AIBS OpenScope session for
the Credit Assignment Project.

Authors: Colleen Gillon, Blake Richards

Date: August, 2018

Note: this code uses python 2.7.

"""
import os
import glob
import exceptions
import warnings

import numpy as np
import pickle
import h5py
import pandas
import scipy.stats as st

import file_util
import sync_util

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
            raise exceptions.OSError('{} either does not exist or is not a directory'.format(self.home))

        # set the session directory (full path)
        self.dir = os.path.join(self.home, 'ophys_session_{}'.format(self.session))

        # extract the mouse ID, and date from the stim pickle file
        pklglob    = glob.glob(os.path.join(self.dir, '{}*stim.pkl'.format(self.session)))
        if len(pklglob) == 0:
            raise exceptions.OSError('Could not find stim pkl file in {}'.format(self.dir))
        else:
            pklinfo    = os.path.basename(pklglob[0]).split('_')
        self.mouse = pklinfo[1]
        self.date  = pklinfo[2]

        # extract the experiment ID from the experiment directory name
        expglob         = glob.glob(os.path.join(self.dir,'ophys_experiment*'))
        if len(expglob) == 0:
            raise exceptions.OSError('Could not find experiment directory in {}'.format(self.dir))
        else:
            expinfo    = os.path.basename(expglob[0]).split('_')
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
            raise exceptions.IOError('Could not open {} for reading'.format(self.stim_pkl))

        # store some variables for easy access
        self.stim_fps      = self.stim_dict['fps']
        self.tot_frames    = self.stim_dict['total_frames']
        self.pre_blank     = self.stim_dict['pre_blank_sec']  # seconds
        self.post_blank    = self.stim_dict['post_blank_sec'] # seconds
        self.drop_frames   = self.stim_dict['droppedframes']
        self.n_drop_frames = len(self.drop_frames[0])

        # check our drop tolerance
        if np.float(self.n_drop_frames)/self.tot_frames > self.droptol:
            print('WARNING: {} dropped stimulus frames out of {}.'.format(self.n_drop_frames, self.tot_frames))
        
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
            with open(self.align_pkl,'rb') as f:
                    align = pickle.load(f)
        except:
            raise exceptions.IOError('Could not read stimulus alignment pickle file {}'.format(self.align_pkl))
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
        self.run = sync_util.get_run_speed(self.stim_pkl)
    
    #############################################
    # load additional info as attributes
    def extract_stim_info(self):
        """
        extract_stim_info()

        Runs load_align_df(), load_stim_dict() if this has not been done yet.
        Then, initializes Stim objects (Gabors, Bricks, Grayscr) from the stimulus dictionary. 
        """

        # load the stimulus, running, alignment information 
        if not hasattr(self, 'stim_dict'):
            self.load_stim_dict()
        if not hasattr(self, 'align_df'):
            self.load_align_df()
        
        self.stim_types = []
        self.n_stims    = len(self.stim_dict['stimuli'])
        self.stims      = []
        for i in range(self.n_stims):
            stim      = self.stim_dict['stimuli'][i]
            stim_type = stim['stimParams']['elemParams']['name']
            self.stim_types.extend([stim_type])
            # initialize a Gabors object
            if stim_type == 'gabors':
                self.gabors = Gabors(self, i)
            # initialize a Bricks object
            elif stim_type == 'bricks':
                self.bricks = Bricks(self, i)
            else:
                print('{} stimulus type not recognized. No Stim object created for this stimulus. \n'.format(stim_type))
        # initialize a Grayscr object
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
        speed = np.interp(frames, self.stim_align, self.run)

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
    be extended with other classes containing stimulus specific information. Here are the
    most relevant attributes:
    
    self.stim_type (str)                       : 'gabors' or 'bricks'
    self.stim_fps (int)                        : fps of the stimulus
    self.act_n_blocks (int)                    : nbr of blocks (where an overarching parameter is held constant)
    self.surp_min_s (int)                      : minimum duration of a surprise sequence
    self.surp_max_s (int)                      : maximum duration of a surprise sequence
    self.reg_min_s (int)                       : minimum duration of a regular sequence
    self.reg_max_s (int)                       : maximum duration of a regular sequence
    self.blank_per (int)                       : period at which a blank segment occurs
    self.block_ran_seg (list of list of tuples): segment tuples (start, end) for each block (end is EXCLUDED)
                                                 each sublist contains tuples for a display sequence
                                                 e.g., for 2 sequences with 2 blocks each:
                                                 [[[start, end], [start, end]], [[start, end], [start, end]]] 
    self.block_len_seg (list of list)          : len of blocks in segments
                                                 each sublist contains the len of each block for a display sequence
                                                 e.g., for 2 sequences with 2 blocks each:
                                                 [[len, len], [len, len]]
    self.frame_list (list)                     : list of segment numbers for each frame (-1 for grayscreen)
    self.block_ran_fr (list of list of tuples) : same as self.block_ran_seg but in stimulus frame numbers instead
    self.block_len_fr (list of list of tuples) : same as self.block_ran_len but in stimulus frame numbers instead

    core structure to all stimuli is fps, type, blocks and segments. A block is a sequence of
    stimulus presentations of the same stimulus type, and there can be multiple
    blocks in one experiment. Segments refer to an individual stimulus 'sweeps' per
    the AIBS stimulus. 
    """
    def __init__(self, sess, stim_n, stim_type):
        """

        """
        self.stim_type = stim_type
        self.sess = sess
        self.stim_fps = self.sess.stim_fps
        self.stim_n = stim_n
        

        # get segment parameters
        # seg is equivalent to a sweep, as defined in camstim 
        if self.stim_type == 'gabors':
            params = 'gabor_params'
            self.seg_len_s     = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams'][params]['im_len'] # segment length (sec) (0.3 sec)
            self.n_seg_per_set = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams'][params]['n_im'] # num seg per set (4: A, B, C D/E)
            self.exp_n_blocks  = 2 # HARD-CODED, 2 blocks (1 per kappa) should be shown.
        elif self.stim_type == 'bricks':
            params = 'square_params'
            self.seg_len_s     = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams'][params]['seg_len'] # segment length (sec) (1 sec)
            self.exp_n_blocks  = 4 # HARD-CODED, 4 blocks (1 per direction/size) should be shown.
        else:
            raise ValueError('{} stim type not recognized. Stim object cannot be initialized.'.format(self.stim_type))
        
        self.blank_per     = self.sess.stim_dict['stimuli'][self.stim_n]['blank_sweeps'] # blank period (i.e., 1 blank every _ segs)
        self.seg_ps_wibl   = 1/self.seg_len_s # num seg per sec (blank segs count) 
        self.seg_ps_nobl   = self.seg_ps_wibl*self.blank_per/(1.+self.blank_per) # num seg per sec (blank segs do not count)
        
        # sequence parameters
        self.surp_min_s  = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams'][params]['surp_len'][0]   # min duration of each surprise sequence (sec)
        self.surp_max_s  = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams'][params]['surp_len'][1]   # max duration of each surprise sequence (sec)
        self.reg_min_s   = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams'][params]['reg_len'][0]    # min duration of each regular sequence (sec)
        self.reg_max_s   = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams'][params]['reg_len'][1]    # max duration of each regular sequence (sec)
        self.exp_block_len_s = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams'][params]['block_len'] # expected length of a block (sec) where an overarching parameter
                                                                                                    # is held constant
        self._get_blocks()
        self._get_frames()


    #################################
    # calculates block lengths
    def _get_blocks(self):

        self.disp_seq    = self.sess.stim_dict['stimuli'][self.stim_n]['display_sequence']
        self.n_segs_nobl = np.empty([len(self.disp_seq)])
        tot_disp         = int(sum(np.diff(self.disp_seq)))

        if self.stim_type == 'gabors':
            # block length is correct, as it was set to include blanks
            block_len = self.exp_block_len_s
        elif self.stim_type == 'bricks':
            # block length was not set to include blanks, so must be adjusted
            block_len = self.exp_block_len_s*self.seg_ps_wibl/self.seg_ps_nobl

        # calculate number of blocks that started and checking whether it is as expected
        self.act_n_blocks = int(np.ceil(float(tot_disp)/block_len))
        if self.act_n_blocks != self.exp_n_blocks:
            print('WARNING: {} {} blocks started instead of the expected {}. \n'.format(self.act_n_blocks, self.stim_type, self.exp_n_blocks))            
            if self.act_n_blocks > self.exp_n_blocks:
                self.extra_segs = (float(tot_disp) - self.exp_n_blocks*block_len)*self.seg_ps_wibl 
                print('WARNING: In total, {} extra segments were shown, including blanks. \n'.format(self.extra_segs))
    
        # calculate uninterrupted segment ranges for each block and check for incomplete or split blocks
        rem_sec_all         = 0
        self.block_ran_seg  = []
        start               = 0
        for i in range(len(self.disp_seq)):
            # useable length is reduced if previous block was incomplete
            length = np.diff(self.disp_seq)[i]-rem_sec_all
            n_bl = int(np.ceil(float(length)/block_len))
            rem_sec_all += float(n_bl)*block_len - length
            rem_seg = int(np.around((float(n_bl)*block_len - length)*self.seg_ps_wibl))
            
            # collect block starts and ends (in segment numbers)
            temp = []
            for _ in range(n_bl-1):
                end = start + int(np.around(block_len*self.seg_ps_nobl))
                temp.append([start, end])
                start = end
            end = start + int(np.around(block_len*self.seg_ps_nobl))-np.max([0, rem_seg-1]) # 1 removed because last segment is a blank
            temp.append([start, end])
            self.block_ran_seg.append(temp)
            start = end + np.max([0, rem_seg-1])
            
            if rem_seg == 1:
                if i == len(self.disp_seq)-1:
                    print('WARNING: During last sequence of {}, the last blank segment of the {}. block'
                        'was omitted. \n'.format(self.stim_type, n_bl))
                else:
                    print('WARNING: During {}. sequence of {}, the last blank segment of the {}. block '
                        'was pushed to the start of the next sequence. \n'.format(i+1, self.stim_type, n_bl))
            elif rem_seg > 1:

                if i == len(self.disp_seq)-1:
                    print('WARNING: During last sequence of {}, {} segments (incl. blanks) '
                        'from the {}. block were omitted. \n'.format(self.stim_type, rem_seg, n_bl))
                else:
                    print('WARNING: During {}. sequence of {}, {} segments (incl. blanks) '
                        'from the {}. block were pushed to the next sequence.'
                        'These segments will be omitted from analysis. \n'.format(i+1, self.stim_type, rem_seg, n_bl))
            # get the actual length in segments of each block
            self.block_len_seg = np.diff(self.block_ran_seg).squeeze(2).tolist()


    #####################################
    # calculates behavioural (not 2P) frame range for each block
    def _get_frames(self):
        # fill out the stimulus frame_list to be the same length as running array

        self.frame_list = int(self.sess.pre_blank*self.stim_fps)*[-1] + \
                          self.sess.stim_dict['stimuli'][self.stim_n]['frame_list'].tolist() + \
                          int(self.sess.tot_frames - len(self.sess.stim_dict['stimuli'][self.stim_n]['frame_list']))*[-1] + \
                          int(self.sess.post_blank*self.stim_fps)*[-1] 
        
        # cutting off first frame as done elsewhere (NOTE: should be last frame?)
        self.frame_list = self.frame_list[1:]
        self.block_ran_fr = []
        for i in self.block_ran_seg:
            temp = []
            for j in i:
                # get first occurrence of first segment
                min_ind = self.frame_list.index(j[0])
                max_ind = len(self.frame_list)-1 - self.frame_list[::-1].index(j[1]-1)+1 # 1 added as range end is excluded
                temp.append([min_ind, max_ind])
            self.block_ran_fr.append(temp)
        
        # get the length in frames of each block (flanking grayscreens are omitted in these numbers)
        self.block_len_fr = np.diff(self.block_ran_fr).squeeze(2).tolist()

    ####################################
    def get_n_frames_per_seg(self, segs):
        """
        get_n_frames_per_seg()

        Returns a list with the number of frames for each seg passed.    

        Argument:
            segs (list): list of segments

        Output:
            n_frames (list): list of number of frames in each segment
        """

        if not isinstance(segs , list):
            segs = [segs]
        
        # segs are in increasing order in dataframe and n_frames will be returned in that order
        # so get indices for sorting in this order, to resort at the end


        n_frames = self.sess.align_df.loc[(self.sess.align_df['stimType']==self.stim_type[0]) &
                                          (self.sess.align_df['stimSeg'].isin(segs))]['num_frames'].tolist()
        
        # resort based on segs, as n_frames will be ordered in increasing segments
        return [x for _, x in sorted(zip(segs, n_frames))]

    ####################################
    def get_segs_by_criteria(self, stimPar1='any', stimPar2='any', surp='any', 
                               stimSeg='any', gaborframe='any', start_frame='any', end_frame='any',
                               num_frames='any', remconsec=False, by='block'):
        """
        get_segs_by_criteria()

        Returns a list of stimulus segs that have the specified values in specified columns in 
        the alignment dataframe.    

        Optional arguments:
            stimPar1 (int or list)      : stimPar1 value(s) of interest (256, 128, 45, 90)
            stimPar2 (str, int or list) : stimPar2 value(s) of interest ('right', 'left', 4, 16)
            surp (int or list)          : surp value(s) of interest (0, 1)
            stimSeg (int or list)       : stimSeg value(s) of interest
            gaborframe (int or list)    : gaborframe value(s) of interest (0, 1, 2, 3)
            start_frame_min (int)       : minimum of start_frame range of interest 
            start_frame_max (int)       : maximum of start_frame range of interest (excl)
            end_frame_min (int)         : minimum of end_frame range of interest
            end_frame_max (int)         : maximum of end_frame range of interest (excl)
            num_frames_min (int)        : minimum of num_frames range of interest
            num_frames_max (int)        : maximum of num_frames range of interest (excl)
                -> defaults are all 'any'

            remconsec (default: False)  : if True, consecutive segments are removed within a block
            by (default: 'block')       : determines whether segments are returned in a flat list ('frame'),
                                        grouped by block ('block'), or further grouped by display sequence ('disp')
        """

        if stimPar1 == 'any':
            stimPar1 = self.sess.align_df['stimPar1'].unique().tolist()
        elif not isinstance(stimPar1, list):
            stimPar1 = [stimPar1]
        if stimPar2 == 'any':
            stimPar2 = self.sess.align_df['stimPar2'].unique().tolist()
        elif not isinstance(stimPar2, list):
            stimPar2 = [stimPar2]
        if surp == 'any':
            surp = self.sess.align_df['surp'].unique().tolist()
        elif not isinstance(surp, list):
            surp = [surp]
        if stimSeg == 'any':
            stimSeg = self.sess.align_df['stimSeg'].unique().tolist()
            # here, ensure that non seg is removed
            if -1 in stimSeg:
                stimSeg.remove(-1)
        elif not isinstance(stimSeg, list):
            stimSeg = [stimSeg]
        if gaborframe == 'any':
            gaborframe = self.sess.align_df['GABORFRAME'].unique().tolist()
        elif not isinstance(gaborframe, list):
            gaborframe = [gaborframe]
        if start_frame == 'any':
            start_frame_min = int(self.sess.align_df['start_frame'].min())
            start_frame_max = int(self.sess.align_df['start_frame'].max())
        if end_frame == 'any':
            end_frame_min = int(self.sess.align_df['end_frame'].min())
            end_frame_max = int(self.sess.align_df['end_frame'].max())
        if num_frames == 'any':
            num_frames_min = int(self.sess.align_df['num_frames'].min())
            num_frames_max = int(self.sess.align_df['num_frames'].max())
        
        segs = []
        for i in self.block_ran_seg:
            temp = []
            for j in i:
                inds = self.sess.align_df.loc[(self.sess.align_df['stimType']==self.stim_type[0])    & 
                                              (self.sess.align_df['stimPar1'].isin(stimPar1))        &
                                              (self.sess.align_df['stimPar2'].isin(stimPar2))        &
                                              (self.sess.align_df['surp'].isin(surp))                &
                                              (self.sess.align_df['stimSeg'].isin(stimSeg))          &
                                              (self.sess.align_df['GABORFRAME'].isin(gaborframe))    &
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
                    temp2 = []
                    for k, val in enumerate(inds):
                        if k == 0 or val != inds[k-1]+1:
                            temp2.extend([val])
                temp2 = inds
                # check for empty
                if len(temp2) != 0:
                    temp.append(temp2)
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
        
        return segs

    ####################################
    def get_frames_by_criteria(self, stimPar1='any', stimPar2='any', surp='any', 
                               stimSeg='any', gaborframe='any', start_frame='any', end_frame='any',
                               num_frames='any', first_fr=True, remconsec=False, by='block'):
        """
        get_frames_by_criteria()

        Returns a list of stimulus frames that have the specified values in specified columns in 
        the alignment dataframe.    
        Note: grayscreen frames are NOT returned

        Optional arguments:
            stimPar1 (int or list)      : stimPar1 value(s) of interest (256, 128, 45, 90)
            stimPar2 (str, int or list) : stimPar2 value(s) of interest ('right', 'left', 4, 16)
            surp (int or list)          : surp value(s) of interest (0, 1)
            stimSeg (int or list)       : stimSeg value(s) of interest
            gaborframe (int or list)    : gaborframe value(s) of interest (0, 1, 2, 3)
            start_frame_min (int)       : minimum of start_frame range of interest 
            start_frame_max (int)       : maximum of start_frame range of interest (excl)
            end_frame_min (int)         : minimum of end_frame range of interest
            end_frame_max (int)         : maximum of end_frame range of interest (excl)
            num_frames_min (int)        : minimum of num_frames range of interest
            num_frames_max (int)        : maximum of num_frames range of interest (excl)
                -> defaults are all 'any'

            first_fr (default: True)    : if True, only returns the first frame of each segment
            remconsec (default: False)  : if True, consecutive segments are removed within a block
            by (default: 'block')       : determines whether segments are returned in a flat list ('frame'),
                                          grouped by block ('block'), or further grouped by display sequence ('disp')
        """


        if stimPar1 == 'any':
            stimPar1 = self.sess.align_df['stimPar1'].unique().tolist()
        elif not isinstance(stimPar1, list):
            stimPar1 = [stimPar1]
        if stimPar2 == 'any':
            stimPar2 = self.sess.align_df['stimPar2'].unique().tolist()
        elif not isinstance(stimPar2, list):
            stimPar2 = [stimPar2]
        if surp == 'any':
            surp = self.sess.align_df['surp'].unique().tolist()
        elif not isinstance(surp, list):
            surp = [surp]
        if stimSeg == 'any':
            stimSeg = self.sess.align_df['stimSeg'].unique().tolist()
            # here, ensure that non seg is removed
            if -1 in stimSeg:
                stimSeg.remove(-1)
        elif not isinstance(stimSeg, list):
            stimSeg = [stimSeg]
        if gaborframe == 'any':
            gaborframe = self.sess.align_df['GABORFRAME'].unique().tolist()
        elif not isinstance(gaborframe, list):
            gaborframe = [gaborframe]
        if start_frame == 'any':
            start_frame_min = int(self.sess.align_df['start_frame'].min())
            start_frame_max = int(self.sess.align_df['start_frame'].max())
        if end_frame == 'any':
            end_frame_min = int(self.sess.align_df['end_frame'].min())
            end_frame_max = int(self.sess.align_df['end_frame'].max())
        if num_frames == 'any':
            num_frames_min = int(self.sess.align_df['num_frames'].min())
            num_frames_max = int(self.sess.align_df['num_frames'].max())
        
        frames = []
        for i in self.block_ran_seg:
            temp = []
            for j in i:
                temp2 = []
                inds = self.sess.align_df.loc[(self.sess.align_df['stimType']==self.stim_type[0])    & 
                                              (self.sess.align_df['stimPar1'].isin(stimPar1))        &
                                              (self.sess.align_df['stimPar2'].isin(stimPar2))        &
                                              (self.sess.align_df['surp'].isin(surp))                &
                                              (self.sess.align_df['stimSeg'].isin(stimSeg))          &
                                              (self.sess.align_df['GABORFRAME'].isin(gaborframe))    &
                                              (self.sess.align_df['start_frame'] >= start_frame_min) &
                                              (self.sess.align_df['start_frame'] < start_frame_max)  &
                                              (self.sess.align_df['end_frame'] >= end_frame_min)     &
                                              (self.sess.align_df['end_frame'] < end_frame_max)      &
                                              (self.sess.align_df['num_frames'] >= num_frames_min)   &
                                              (self.sess.align_df['num_frames'] < num_frames_max)    &
                                              (self.sess.align_df['stimSeg'] >= j[0])                &
                                              (self.sess.align_df['stimSeg'] < j[1])]['stimSeg'].tolist()
                
                # get the frames for each index
                if remconsec: # if removing consecutive values
                    for k, val in enumerate(inds):
                        if k == 0 or val != inds[k-1]+1:
                            temp2.extend([self.frame_list.index(val)])
                elif first_fr: # if getting only first frame for each segment
                    for val in inds:
                        temp2.extend([self.frame_list.index(val)])
                else: # if getting all frames
                    for val in inds:
                        temp2.extend([k for k, x in enumerate(self.frame_list) if x == val])
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
        
        return frames

    def get_first_surp_segs(self, by='block'):
        """
        Returns two lists of stimulus segments, the first is a list of all the first surprise segments for the 
        stimulus type at transitions from regular to surprise sequences. The second is a list of all the first
        regular segements for the stimulus type at transitions from surprise to regular sequences.

        Optional argument:
            by (default: 'block'): determines whether segments are returned in a flat list ('seg'),
                                  grouped by block ('block'), or further grouped by display sequence ('disp')
        
        Outputs:
            surp_segs (list)   : list of first surprise segments at regular to surprise transitions for stimulus type
            no_surp_segs (list): list of first regular segments at surprise to regular transitions for stimulus type
        """

        surp_segs   = self.get_segs_by_criteria(surp=1, remconsec=True, by=by)
        nosurp_segs = self.get_segs_by_criteria(surp=0, remconsec=True, by=by)

        return surp_segs, nosurp_segs

    def get_all_surp_segs(self, by='block'):
        """
        Returns two lists of stimulus segments, the first is a list of all the surprise segments for the 
        stimulus type. The second is a list of all the regular segments for the stimulus type.

        Optional argument:
            by (default: 'block'): determines whether segments are returned in a flat list ('seg'),
                                  grouped by block ('block'), or further grouped by display sequence ('disp')
        
        Outputs:
            surp_segs (list)   : list of surprise segments for stimulus type
            no_surp_segs (list): list of regular segments for stimulus type
        """

        surp_segs   = self.get_segs_by_criteria(surp=1, by=by)
        nosurp_segs = self.get_segs_by_criteria(surp=0, by=by)

        return surp_segs, nosurp_segs
    
    def get_first_surp_frame_1s(self, by='block'):
        """
        Returns two lists of stimulus frames, the first is a list of all the first surprise frames for the 
        stimulus type at transitions from regular to surprise sequences. The second is a list of all the first
        regular frames for the stimulus type at transitions from surprise to regular sequences.

        Optional argument:
            by (default: 'block'): determines whether frames are returned in a flat list ('frame'),
                                    grouped by block ('block'), or further grouped by display sequence ('disp')
        
        Outputs:
            surp_frames (list)   : list of first surprise frames at regular to surprise transitions for stimulus type
            no_surp_frames (list): list of first regular frames at surprise to regular transitions for stimulus type
        """
    
        surp_frames   = self.get_frames_by_criteria(surp=1, remconsec=True, by=by)
        nosurp_frames = self.get_frames_by_criteria(surp=0, remconsec=True, by=by)

        return surp_frames, nosurp_frames

    def get_all_surp_frames(self, by='block'):
        """
        Returns two lists of stimulus frames, the first is a list of all surprise frames for the 
        stimulus type. The second is a list of all regular frames for the stimulus type.

        Optional argument:
            by (default: 'block'): determines whether frames are returned in a flat list ('frame'),
                                   grouped by block ('block'), or further grouped by display sequence ('disp')
        
        Outputs:
            surp_frames (list)   : list of all surprise frames for stimulus type
            no_surp_frames (list): list of all regular frames for stimulus type
        """

        surp_frames   = self.get_frames_by_criteria(surp=1, first_fr=False, by=by)
        nosurp_frames = self.get_frames_by_criteria(surp=0, first_fr=False, by=by)

        return surp_frames, nosurp_frames
    
    def get_run(self, by='block'):
        """
        Returns run values for stimulus blocks.

        Optional argument:
            by (default: 'block'): determines whether segments are returned in a flat list ('seg'),
                                  grouped by block ('block'), or further grouped by display sequence ('disp')
        
        Output:
            run (list): list of running values for stimulus blocks
        """
        
        if not hasattr(self.sess, 'run'):
            self.sess.load_run()
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
    
        return run
    
class Gabors(Stim):
    """
    The Gabors object inherits from the Stim object and describes gabor specific properties.
    """

    def __init__(self, sess, stim_n):
        """
        __init__(sess, stim_n)

        Create the new Gabors object using the Session it belongs to and stimulus number the object
        corresponds to.

        Required arguments:
            - sess (Session)  : full path to the directory where session folders are stored.
            - stim_n (int)    : this stimulus' number, x in sess.stim_dict['stimuli'][x]
        """

        Stim.__init__(self, sess, stim_n, stim_type='gabors')

        # gabor specific parameters
        self.units     = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams']['gabor_params']['units']
        self.phase     = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams']['gabor_params']['phase']
        self.sf        = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams']['gabor_params']['sf']
        self.n_patches = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams']['gabor_params']['n_gabors']
        self.oris      = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams']['gabor_params']['oris']
        self.ori_std   = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams']['gabor_params']['ori_std']
        # kappas calculated as 1/std**2
        self.ori_kaps = [1/x**2 for x in self.ori_std] 

        # seg sets (hard-coded, based on the repeating structure  we are interested in, namely: blank, A, B, C, D/E)
        self.pre  = 1*self.seg_len_s # 300 ms blank
        self.post = self.n_seg_per_set*self.seg_len_s # 1200 ms gabors


    def get_A_segs(self, by='block'):
        """
        Returns lists of A gabor segments.

        Optional argument:
            by (default: 'block'): determines whether frames are returned in a flat list ('frame'),
                                    grouped by block ('block'), or further grouped by display sequence ('disp')
        
        Outputs:
            A_segs (list) : list of A gabor segments.
        """
        A_segs = self.get_segs_by_criteria(gaborframe=0, by=by)

        return A_segs

    def get_A_frame_1s(self, by='block'):
        """
        Returns list of first frame for each A gabor segment.

        Optional argument:
            by (default: 'block'): determines whether frames are returned in a flat list ('frame'),
                                    grouped by block ('block'), or further grouped by display sequence ('disp')
        
        Outputs:
            A_segs (list) : lists of first frame for each A gabor segment
        """
        A_frames = self.get_frames_by_criteria(gaborframe=0, by=by)

        return A_frames
    

    
class Bricks(Stim):
    """
    The Bricks object inherits from the Stim object and describes bricks specific properties.
    """

    def __init__(self, sess, stim_n):
        """
        __init__(sess, stim_n)

        Create the new Bricks object using the Session it belongs to and stimulus number the object
        corresponds to.

        Required arguments:
            - sess (Session)  : full path to the directory where session folders are stored.
            - stim_n (int)    : this stimulus' number, x in sess.stim_dict['stimuli'][x]

        """

        Stim.__init__(self, sess, stim_n, stim_type='bricks')

        # initialize brick specific parameters
        self.units    = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams']['square_params']['units']
        self.flipfrac = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams']['square_params']['flipfrac']
        self.speed = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams']['square_params']['speed']
        self.sizes = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams']['square_params']['sizes']
        # number of bricks not recorded in stim parameters, so extracting from dataframe 
        self.n_bricks = pandas.unique(sess.align_df.loc[(self.sess.align_df['stimType'] == 0)]['stimPar1']).tolist() # preserves order
        self.direcs = self.sess.stim_dict['stimuli'][self.stim_n]['stimParams']['square_params']['direcs']


    def get_dir_segs_no_surp(self, by='block'):
        """
        Returns two lists of stimulus segments, the first is a list of the right moving segments.
        The second is a list of left moving segments. Both lists exclude surprise segments.

        Optional argument:
            by (default: 'block'): determines whether frames are returned in a flat list ('frame'),
                                    grouped by block ('block'), or further grouped by display sequence ('disp')
        
        Outputs:
            right_frames (list) : list of right moving segments, excluding surprise segments.
            left_frames (list)  : list of left moving segments, excluding surprise segments.
        """

        right_segs = self.get_segs_by_criteria(stimPar2='right', surp=0, by=by)
        left_segs  = self.get_segs_by_criteria(stimPar2='left', surp=0, by=by)

        return right_segs, left_segs

    def get_dir_frames_no_surp(self, by='block'):
        """
        Returns two lists of stimulus frames, the first is a list of the first frame for each right moving segment.
        The second is a list of the first frame for each left moving segment. Both lists exclude surprise segments.

        Optional argument:
            by (default: 'block'): determines whether frames are returned in a flat list ('frame'),
                                    grouped by block ('block'), or further grouped by display sequence ('disp')
        
        Outputs:
            right_frames (list) : list of first frames for each right moving segment, excluding surprise segments.
            left_frames (list)  : list of first frames for each left moving segment, excluding surprise segments.
        """

        right_frames = self.get_frames_by_criteria(stimPar2='right', surp=0, by=by)
        left_frames  = self.get_frames_by_criteria(stimPar2='left', surp=0, by=by)

        return right_frames, left_frames
        

class Grayscr():
    """
    Class to retrieve frame number information for grayscreen frames within Gabor stimuli
    or outside of Gabor stimuli.
    """

    
    def __init__(self, sess):
        
        self.sess = sess
        if hasattr(self.sess, 'gabors'):
            self.gabors = True
        
    def get_all_nongab_frames(self):
        """
        Returns a lists of grayscreen frames, excluding grayscreen frames occurring duriing gabor 
        stimulus blocks. Note that any grayscreen frames flanking gabor stimulus blocks are included in the
        returned list.
        
        Outputs:
            grays (list) : list of grayscreen frames.
        """

        frames = []
        if self.gabors:
            frames_gab = np.asarray(self.sess.gabors.frame_list)
            gab_blocks = self.sess.gabors.block_ran_fr
            for i in gab_blocks:
                for j in i:
                    frames_gab[j[0]:j[1]] = 0
            frames.append(frames_gab)
        if hasattr(self.sess, 'bricks'):
            frames.append(np.asarray(self.sess.bricks.frame_list))
        length = len(frames)
        if length == 0:
            raise ValueError('No frame lists were found for either stimulus types (gabors, bricks.')
        elif length == 1:
            frames_sum = np.asarray(frames)
        else:
            frames_sum = np.sum(np.asarray(frames), axis=0)
        grays = np.where(frames_sum==length*-1)[0].tolist()

        if length(grays) == 0:
            raise ValueError('No grayscreen frames were found outside of gabor stimulus sequences.')

        return grays

    def get_first_nongab_frames(self):
        """
        Returns two lists of equal length:
        - First grayscreen frames for every grayscreen sequence, excluding those 
          occurring during gabor stimulus blocks. Note that any first grayscreen frames flanking gabor stimulus 
          blocks are included in the returned list.
        - Number of consecutive grayscreen frames for each sequence.
        
        Outputs:
            first_grays (list) : list of first grayscreen frames for every grayscreen sequence
            n_grays (list)     : list of number of grayscreen frames for every grayscreen sequence
        """

        grays_all = self.get_all_nongab_frames()
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

    def get_all_gab_frames(self, by='block'):
        """
        Returns a list of grayscreen frames for every grayscreen sequence during a gabor block, 
          excluding flanking grayscreen sequences.
    
        Optional argument:
            by (default: 'block'): determines whether frames are returned in a flat list ('frame'),
                                   grouped by block ('block'), or further grouped by display sequence ('disp')
        
        Outputs:
            gab_grays (list) : list of grayscreen frames for every grayscreen sequence during gabors
        """
        
        if self.gabors:
            frames_gab = np.asarray(self.sess.gabors.frame_list) # make copy!!!
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
                if by == 'seg':
                    gab_grays = [x for sub in gab_grays for x in sub]
            
            return gab_grays
        else:
            raise IOError('Session does not have a gabors attribute. Be sure to extract stim info \
                           and check that session contains a gabor stimulus.')
    
    def get_first_gab_frames(self, by='block'):
        """
        Returns two lists of equal length:
        - First grayscreen frames for every grayscreen sequence during a gabor block, 
          excluding flanking grayscreen sequences.
        - Number of consecutive grayscreen frames for each sequence.

        Optional argument:
            by (default: 'block'): determines whether frames numbers and number of frames are returned in a flat list ('frame'),
                                   grouped by block ('block'), or further grouped by display sequence ('disp')
        
        Outputs:
            first_gab_grays (list) : list of first grayscreen frames for every grayscreen sequence during gabors
            n_gab_grays (list)     : list of number of grayscreen frames for every grayscreen sequence during gabors
        """

        grays_gab = self.get_all_gab_frames(by='disp')
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

        return first_gab_grays, n_gab_grays