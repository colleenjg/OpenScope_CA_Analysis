"""
sync_util.py

This module contains functions for synchronizing the different data files 
generated by the AIBS experiments for the Credit Assignment Project.

Authors: Allen Brain Institute, Joel Zylberberg, Blake Richards, Colleen Gillon

Date: August, 2018

Note: this code uses python 3.7.

"""

import pdb
import os
import warnings

import h5py
import json
import numpy as np
import pandas as pd
import pickle

from util import file_util
from sess_util import dataset, Dataset2p


# set a few basic parameters
ASSUMED_DELAY = 0.0351
DELAY_THRESHOLD = 0.001
FIRST_ELEMENT_INDEX = 0
SECOND_ELEMENT_INDEX = 1
SKIP_FIRST_ELEMENT = 1
SKIP_LAST_ELEMENT = -1
ROUND_PRECISION = 4
ZERO = 0
ONE = 1
TWO = 2
MIN_BOUND = .03
MAX_BOUND = .04


#############################################
def check_drop_tolerance(n_drop_stim_fr, tot_stim_fr, droptol=0.0003, 
                         raise_exc=False):
    """
    check_drop_tolerance(n_drop_stim_fr, tot_stim_fr)

    Prints a warning or raises an exception if dropped stimulus frames 
    tolerance is passed.

    Required args:
        - n_drop_stim_fr (int): number of dropped stimulus frames
        - tot_stim_fr (int)   : total number of stimulus frames

    Optional args:
        - droptol (float) : threshold proportion of dropped stimulus frames at 
                            which to print warning or raise an exception. 
        - raise_exc (bool): if True, an exception is raised if threshold is 
                            passed. Otherwise, a warning is printed.
    """

    if np.float(n_drop_stim_fr)/tot_stim_fr > droptol:
        warn_str = (f'{n_drop_stim_fr} dropped stimulus '
                    f'frames out of {tot_stim_fr}.')
        if raise_exc:
            raise OSError(warn_str)
        else:    
            print(f'    WARNING: {warn_str}')


#############################################
def calculate_stimulus_alignment(stim_time, valid_twop_vsync_fall):
    """
    calculate_stimulus_alignment(stim_time, valid_twop_vsync_fall)

    """

    print('Calculating stimulus alignment.')

    # convert stimulus frames into twop frames
    stimulus_alignment = np.empty(len(stim_time))

    for index in range(len(stim_time)):
        crossings = np.nonzero(np.ediff1d(np.sign(valid_twop_vsync_fall - \
                                                  stim_time[index])) > ZERO)
        try:
            stimulus_alignment[index] = int(crossings[FIRST_ELEMENT_INDEX]
                                                     [FIRST_ELEMENT_INDEX])
        except:
            stimulus_alignment[index] = np.NaN

    return stimulus_alignment


#############################################
def calculate_valid_twop_vsync_fall(sync_data, sample_frequency):
    """
    calculate_valid_twop_vsync_fall(sync_data, sample_frequency)

    """

    ####microscope acquisition frames####
    # get the falling edges of 2p
    twop_vsync_fall = sync_data.get_falling_edges('2p_vsync') / sample_frequency

    if len(twop_vsync_fall) == 0:
        raise ValueError('Error: twop_vsync_fall length is 0, possible '
                         'invalid, missing, and/or bad data')

    ophys_start = twop_vsync_fall[0]

    # only get data that is beyond the start of the experiment
    valid_twop_vsync_fall = twop_vsync_fall[np.where(twop_vsync_fall > 
                                            ophys_start)[FIRST_ELEMENT_INDEX]]

    # skip the first element to eliminate the DAQ pulse
    return valid_twop_vsync_fall


#############################################
def calculate_stim_vsync_fall(sync_data, sample_frequency):
    """
    calculate_stim_vsync_fall(sync_data, sample_frequency)

    """

    ####stimulus frames####
    # skip the first element to eliminate the DAQ pulse
    stim_vsync = sync_data.get_falling_edges('stim_vsync')[SKIP_FIRST_ELEMENT:]
    stim_vsync_fall = stim_vsync / sample_frequency

    return stim_vsync_fall


#############################################
def calculate_delay(sync_data, stim_vsync_fall, sample_frequency):
    """
    calculate_delay(sync_data, stim_vsync_fall, sample_frequency)

    """

    print('calculating delay')

    try:
        # photodiode transitions
        photodiode_rise = (sync_data.get_rising_edges('stim_photodiode') / sample_frequency)

        ####Find start and stop of stimulus####
        # test and correct for photodiode transition errors
        photodiode_rise_diff = np.ediff1d(photodiode_rise)
        min_short_photodiode_rise = 0.1
        max_short_photodiode_rise = 0.3
        min_medium_photodiode_rise = 0.5
        max_medium_photodiode_rise = 1.5

        # find the short and medium length photodiode rises
        short_rise_indexes = np.where(np.logical_and(photodiode_rise_diff > min_short_photodiode_rise, \
                                                     photodiode_rise_diff < max_short_photodiode_rise))[FIRST_ELEMENT_INDEX]
        medium_rise_indexes = np.where(np.logical_and(photodiode_rise_diff > min_medium_photodiode_rise, \
                                                      photodiode_rise_diff < max_medium_photodiode_rise))[
            FIRST_ELEMENT_INDEX]

        short_set = set(short_rise_indexes)

        # iterate through the medium photodiode rise indexes to find the start 
        # and stop indices looking for three rise pattern
        next_frame = ONE
        start_pattern_index = 2
        end_pattern_index = 3
        ptd_start = None
        ptd_end = None

        for medium_rise_index in medium_rise_indexes:
            if set(range(medium_rise_index - start_pattern_index, medium_rise_index)) <= short_set:
                ptd_start = medium_rise_index + next_frame
            elif set(range(medium_rise_index + next_frame, medium_rise_index + end_pattern_index)) <= short_set:
                ptd_end = medium_rise_index

        # if the photodiode signal exists
        if ptd_start != None and ptd_end != None:
            # check to make sure there are no there are no photodiode errors
            # sometimes two consecutive photodiode events take place close to 
            # each other correct this case if it happens
            photodiode_rise_error_threshold = 1.8
            last_frame_index = -1

            # iterate until all of the errors have been corrected
            while any(photodiode_rise_diff[ptd_start:ptd_end] < photodiode_rise_error_threshold):
                error_frames = np.where(photodiode_rise_diff[ptd_start:ptd_end] < \
                                        photodiode_rise_error_threshold)[FIRST_ELEMENT_INDEX] + ptd_start
                # remove the bad photodiode event
                photodiode_rise = np.delete(photodiode_rise, error_frames[last_frame_index])
                ptd_end -= 1
                photodiode_rise_diff = np.ediff1d(photodiode_rise)

            ####Find the delay####
            # calculate monitor delay
            first_pulse = ptd_start
            number_of_photodiode_rises = ptd_end - ptd_start
            half_vsync_fall_events_per_photodiode_rise = 60
            vsync_fall_events_per_photodiode_rise = half_vsync_fall_events_per_photodiode_rise * 2

            delay_rise = np.empty(number_of_photodiode_rises)
            for photodiode_rise_index in range(number_of_photodiode_rises):
                delay_rise[photodiode_rise_index] = photodiode_rise[photodiode_rise_index + first_pulse] - \
                                                    stim_vsync_fall[(photodiode_rise_index * vsync_fall_events_per_photodiode_rise) + \
                                                                    half_vsync_fall_events_per_photodiode_rise]

            # get a single delay value by finding the mean of all of the delays 
            # - skip the element in the array (the end of the experiment)
            delay = np.mean(delay_rise[:last_frame_index])

            if (delay > DELAY_THRESHOLD or np.isnan(delay)):
                print('Sync error needs to be fixed')
                delay = ASSUMED_DELAY
                print('Using assumed delay:', round(delay, ROUND_PRECISION))

        # assume delay
        else:
            delay = ASSUMED_DELAY
    except Exception as e:
        print(e)
        print('Process without photodiode signal')
        delay = ASSUMED_DELAY
        print('Assumed delay:', round(delay, ROUND_PRECISION))

    return delay


#############################################
def get_frame_rate(syn_file_name):
    """
    get_frame_rate(stim_sync_file)

    Pulls out the ophys frame times stimulus sync file and returns stats for
    ophys frame rates.

    Required args:
        - stim_sync_file (str): full path name of the experiment sync hdf5 
                                file

    Returns:
        - twop_rate_mean (num)  : mean ophys frame rate
        - twop_rate_med (num)   : median ophys frame rate
        - twop_rate_std (num)   : standard deviation of ophys frame rate
    """

    # create a Dataset object with the sync file
    sync_data = dataset.Dataset(syn_file_name)
   
    sample_frequency = sync_data.meta_data['ni_daq']['counter_output_freq']
    
    # calculate the valid twop_vsync fall
    valid_twop_vsync_fall = calculate_valid_twop_vsync_fall(sync_data, 
                                                            sample_frequency)
    twop_diff = np.diff(valid_twop_vsync_fall)
    
    twop_rate_mean = np.mean(1./twop_diff)
    twop_rate_med = np.median(1./twop_diff)
    twop_rate_std = np.std(1./twop_diff)
    
    return twop_rate_mean, twop_rate_med, twop_rate_std


#############################################
def get_stim_frames(pkl_file_name, syn_file_name, df_pkl_name, runtype='prod'):
    """
    get_stim_frames(pkl_file_name, syn_file_name, df_pkl_name)

    Pulls out the stimulus frame information from the stimulus pickle file, as
    well as synchronization information from the stimulus sync file, and stores
    synchronized stimulus frame information in the output pickle file along 
	with the stimulus alignment array.

    Required args:
        - pkl_file_name (str): full path name of the experiment stim pickle 
                               file
        - syn_file_name (str): full path name of the experiment sync hdf5 file
        - df_pkl_name (str)  : full path name of the output pickle file to 
                               create
    
    Optional argument:
        - runtype (str): the type of run, either 'pilot' or 'prod'
                         default: 'prod'
    """

    # check that the input files exist
    file_util.checkfile(pkl_file_name)
    file_util.checkfile(syn_file_name)
    
    num_stimtypes = 2 #bricks and Gabors

    # read the pickle file and call it 'pkl'
    pkl = file_util.loadfile(pkl_file_name, filetype='pickle')

    if runtype == 'pilot':
        num_stimtypes = 2 # bricks and Gabors
    elif runtype == 'prod':
        num_stimtypes = 3 # 2 bricks and 1 set of Gabors
    if len(pkl['stimuli']) != num_stimtypes:
        raise ValueError(f'{num_stimtypes} stimuli types expected, but '
                         '{} found.'.format(len(pkl['stimuli'])))
        
    # create a Dataset object with the sync file
    sync_data = dataset.Dataset(syn_file_name)

    # create Dataset2p object which will be used for the delay
    dset = Dataset2p.Dataset2p(syn_file_name)

    sample_frequency = sync_data.meta_data['ni_daq']['counter_output_freq']

    # calculate the valid twop_vsync fall
    valid_twop_vsync_fall = calculate_valid_twop_vsync_fall(sync_data, 
                                                            sample_frequency)

    # get the stim_vsync_fall
    stim_vsync_fall = calculate_stim_vsync_fall(sync_data, sample_frequency)

    # find the delay
    # delay = calculate_delay(sync_data, stim_vsync_fall, sample_frequency)
    delay = dset.display_lag

    # adjust stimulus time with monitor delay
    stim_time = stim_vsync_fall + delay

    # find the alignment
    stimulus_alignment = calculate_stimulus_alignment(
        stim_time, valid_twop_vsync_fall)
    offset = int(pkl['pre_blank_sec'] * pkl['fps'])
    
    print('Creating the stim_df:')
    
    # get number of segments expected and actually recorded for each stimulus
    segs = []
    segs_exp = []
    frames_per_seg = []
    stim_types = []
    
    for i in range(num_stimtypes):
        # records the max num of segs in the frame list for each stimulus
        segs.extend([np.max(pkl['stimuli'][i]['frame_list'])+1])
        
        # calculates the expected number of segs based on fps, 
        # display duration (s) and seg length
        fps = pkl['stimuli'][i]['fps']
        
        if runtype == 'pilot':
            name = pkl['stimuli'][i]['stimParams']['elemParams']['name']
        elif runtype == 'prod':
            name = pkl['stimuli'][i]['stim_params']['elemParams']['name']

        if name == 'bricks':
            stim_types.extend(['b'])
            frames_per_seg.extend([fps])
            segs_exp.extend([int(60.*np.sum(np.diff(
                pkl['stimuli'][i]['display_sequence']))/frames_per_seg[i])])
        elif name == 'gabors':
            stim_types.extend(['g'])
            frames_per_seg.extend([fps/1000.*300])
            # to exclude grey seg
            segs_exp.extend([int(60.*np.sum(np.diff(
                pkl['stimuli'][i]['display_sequence'])
                )/frames_per_seg[i]*4./5)]) 
        else:
            raise ValueError(f'{name} stimulus type not recognized.')
        
        
        # check whether the actual number of frames is within a small range of 
        # expected about two frames per sequence?
        n_seq = pkl['stimuli'][0]['display_sequence'].shape[0] * 2
        if np.abs(segs[i] - segs_exp[i]) > n_seq:
            raise ValueError(f'Expected {segs_exp[i]} frames for stimulus {i}, '
                             f'but found {segs[i]}.')
    
    total_stimsegs = np.sum(segs)
    
    stim_df = pd.DataFrame(index=list(range(np.sum(total_stimsegs))), 
                           columns=['stimType', 'stimPar1', 'stimPar2', 'surp', 
                                    'stimSeg', 'GABORFRAME', 'start_frame', 
                                    'end_frame', 'num_frames'])
    
    zz = 0
    # For gray-screen pre_blank
    stim_df.ix[zz, 'stimType'] = -1
    stim_df.ix[zz, 'stimPar1'] = -1
    stim_df.ix[zz, 'stimPar2'] = -1
    stim_df.ix[zz, 'surp'] = -1
    stim_df.ix[zz, 'stimSeg'] = -1
    stim_df.ix[zz, 'GABORFRAME'] = -1
    stim_df.ix[zz, 'start_frame'] = stimulus_alignment[0] # 2p start frame
    stim_df.ix[zz, 'end_frame'] = stimulus_alignment[offset] # 2p end frame
    stim_df.ix[zz, 'num_frames'] = (stimulus_alignment[offset] - \
                                    stimulus_alignment[0])
    zz += 1

    for stype_n in range(num_stimtypes):
        print(f'    stimtype: {stim_types[stype_n]}')
        movie_segs = pkl['stimuli'][stype_n]['frame_list']

        for segment in range(segs[stype_n]):
            seg_inds = np.where(movie_segs == segment)[0]
            tup = (segment, int(stimulus_alignment[seg_inds[0] + offset]), \
                   int(stimulus_alignment[seg_inds[-1] + 1 + offset]))

            stim_df.ix[zz, 'stimType'] = stim_types[stype_n]
            stim_df.ix[zz, 'stimSeg'] = segment
            stim_df.ix[zz, 'start_frame'] = tup[1]
            stim_df.ix[zz, 'end_frame'] = tup[2]
            stim_df.ix[zz, 'num_frames'] = tup[2] - tup[1]

            get_seg_params(stim_types, stype_n, stim_df, zz, pkl, segment, 
                           runtype)

            zz += 1
            
    # check whether any 2P frames are in associated to 2 stimuli
    overlap = np.any((np.sort(stim_df['start_frame'])[1:] - 
                     np.sort(stim_df['end_frame'])[:-1]) < 0)
    if overlap:
        raise ValueError('Some 2P frames associated with two stimulus \
                         segments.')
	
    # create a dictionary for pickling
    stim_dict = {'stim_df': stim_df, 'stim_align': stimulus_alignment}   
 
    # store in the pickle file
    try:
        file_util.saveinfo(stim_dict, df_pkl_name, overwrite=True)
    except:
        raise OSError(f'Could not save stimulus pickle file {df_pkl_name}')  


#############################################
def get_seg_params(stim_types, stype_n, stim_df, zz, pkl, segment, 
                   runtype='prod'):
    """
    get_seg_params(stim_types, stype_n, stim_df, zz, pkl, segment)

    Populates the parameter columns for a segment in stim_df depending on 
    whether the segment is from a bricks or gabors stimulus block and whether 
    it is a pilot or production session.

    Required args:
        - stim_types (list): list of stimulus types for each stimulus, 
                             e.g., ['b', 'g']
        - stype_n (int)    : stimulus number
        - stim_df (pd df)  : dataframe 
        - zz (int)         : dataframe index
        - pkl (dict)       : experiment stim dictionary
        - segment (int)    : segment number
    
    Optional argument:
        - runtype (str): run type, i.e., 'pilot' or 'prod'
                         default: 'prod'
    """

    if stim_types[stype_n] == 'b':
        if runtype == 'pilot':
            stim_df.ix[zz, 'stimPar1'] = pkl['stimuli'][stype_n]['stimParams']['subj_params']['flipdirecarray'][segment][1] #big or small
            stim_df.ix[zz, 'stimPar2'] = pkl['stimuli'][stype_n]['stimParams']['subj_params']['flipdirecarray'][segment][3] #left or right
            stim_df.ix[zz, 'surp'] = pkl['stimuli'][stype_n]['stimParams']['subj_params']['flipdirecarray'][segment][0] #SURP
        elif runtype == 'prod':
            stim_df.ix[zz, 'stimPar1'] = pkl['stimuli'][stype_n]['stim_params']['elemParams']['sizes'] # small
            stim_df.ix[zz, 'stimPar2'] = pkl['stimuli'][stype_n]['stim_params']['direc'] #L or R
            stim_df.ix[zz, 'surp'] = pkl['stimuli'][stype_n]['sweep_params']['Flip'][0][segment] #SURP
        stim_df.ix[zz, 'GABORFRAME'] = -1
    elif stim_types[stype_n] == 'g':
        if runtype == 'pilot':
            stim_df.ix[zz, 'stimPar1'] = pkl['stimuli'][stype_n]['stimParams']['subj_params']['oriparsurps'][int(np.floor(segment/4.))][0] #angle
            stim_df.ix[zz, 'stimPar2'] = pkl['stimuli'][stype_n]['stimParams']['subj_params']['oriparsurps'][int(np.floor(segment/4.))][1] #angular disp (kappa)
            stim_df.ix[zz, 'surp'] = pkl['stimuli'][stype_n]['stimParams']['subj_params']['oriparsurps'][int(np.floor(segment/4.))][2] #SURP
        elif runtype == 'prod':
            stim_df.ix[zz, 'stimPar1'] = pkl['stimuli'][stype_n]['sweep_params']['OriSurp'][0][int(np.floor(segment/4.))][0] #angle
            stim_df.ix[zz, 'stimPar2'] = (1./(pkl['stimuli'][stype_n]['stim_params']['gabor_params']['ori_std']))**2 #angular disp (kappa)
            stim_df.ix[zz, 'surp'] = pkl['stimuli'][stype_n]['sweep_params']['OriSurp'][0][int(np.floor(segment/4.))][1] #SURP
        stim_df.ix[zz, 'GABORFRAME'] = np.mod(segment,4)


#############################################
def get_run_velocity(pkl_file_name='', stim_dict=None):
    """
    get_run_velocity(pkl_file_name)

    Returns the running velocity information as a numpy array. Takes as input 
    the stim pickle file containing the information (provided either as a path 
    or the actual dictionary). 
    
    NOTE: the length of the array is equivalent to the array returned by 
    get_stimulus_frames. The running velocity provided corresponds to the 
    velocity at each stimulus frame. Thus, aligning to the 2p data can be done 
    using the stimulus_alignment array.

    Optional args:
        - pkl_file_name (str): full path name of the experiment stim 
                               pickle file
                               default: ''
        - stim_dict (dict)   : stimulus dictionary, with keys 'fps' and 
                               'items', from which running velocity is 
                               extracted.
                               If not None, overrides pkl_file_name.
                               default: None

    Returns:
        - running_velocity (array): array of length equal to the number of 
                                    stimulus frames, each element indicates 
                                    running velocity for that stimulus frame
    """

    if pkl_file_name == '' and stim_dict is None:
        raise ValueError('Must provide either the pickle file name or the '
                         'stimulus dictionary.')

    if stim_dict is None:
        # check that the input file exists
        file_util.checkfile(pkl_file_name)

        # read the input pickle file and call it 'pkl'
        stim_dict = file_util.loadfile(pkl_file_name)
        
    # Info from Allen
    wheel_radius = 5.5036

    # determine the frames per second of the running wheel recordings
    fps = stim_dict['fps']

    # determine the change in angle during each frame
    dtheta = stim_dict['items']['foraging']['encoders'][0]['dx']

    # wheel circumference in cm/degree
    cm_deg = 2.0 * np.pi * wheel_radius / 360.0

    # calculate the running velocity 
    # (skip last element, since it is ignored in stimulus frames as well)
    run_velocity = dtheta[:SKIP_LAST_ELEMENT] * fps * cm_deg

    return run_velocity


#############################################
def get_twop2stimfr(stim2twopfr, n_twop_fr):
    """
    get_twop2stimfr(stim2twopfr, n_twop_fr)

    Returns the stimulus frame alignment for each 2p frame.
        
    Required args:
        - stim2twopfr (1D array): 2p frame numbers for each stimulus frame, 
                                    as well as the flanking
                                    blank screen frames 
        - n_twop_fr (int)       : total number of 2p frames

    Returns:
        - twop2stimfr (1D array): Stimulus frame numbers for the beginning
                                  of each 2p frame (np.nan when no stimulus
                                  appears)
    """

    stim2twopfr_diff = np.append(1, np.diff(stim2twopfr))
    stim_idx = np.where(stim2twopfr_diff)[0]

    dropped = np.where(stim2twopfr_diff > 1)[0]
    if len(dropped) > 0:
        print(f'    WARNING: {len(dropped)} dropped stimulus frames '
            'sequences (2nd align).')
        # repeat stim idx when frame is dropped
        for drop in dropped[-1:]:
            loc = np.where(stim_idx == drop)[0][0]
            add = [stim_idx[loc-1]] * (stim2twopfr_diff[drop] - 1)
            stim_idx = np.insert(stim_idx, loc, add)
    
    twop2stimfr = np.full(n_twop_fr, np.nan) 
    start = int(stim2twopfr[0])
    end = int(stim2twopfr[-1]) + 1
    try:
        twop2stimfr[start:end] = stim_idx
    except:
        warnings.warn(message='get_twop2stimfr() not working for this '
            'session. twop2stimfr set to all NaNs.')

    return twop2stimfr

