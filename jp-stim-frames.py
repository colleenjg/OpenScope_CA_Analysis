#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 13:43:34 2019

@author: jay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:49:56 2019

@author: jay
"""

import h5py
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.decomposition import pca#, randomizedpca
from sess_util import sess_sync_util
from util import file_util
import h5py as h5
import time
import copy

#pkl_file_name = '../jay/712483302_389778_20180621_stim.pkl'
#syn_file_name = '../jay/712483302_389778_20180621_sync.h5'
#df_pkl_name = '../jay/712483302_389778_20180621_df.pkl'
#pkl_file_name = '../jay/761269197_411771_20181008_stim.pkl'
#syn_file_name = '../jay/761269197_411771_20181008_sync.h5'
#df_pkl_name = '../jay/761269197_411771_20181008_df.pkl'
#time_synch_file_name = '../jay/761605196_time_synchronization.h5'

#pkl_file_name = '../jay/761624763_411424_20181009_stim.pkl'
#syn_file_name = '../jay/761624763_411424_20181009_sync.h5'
#df_pkl_name = '../jay/761624763_411424_20181009_df.pkl'
#time_synch_file_name = '../jay/761865843_time_synchronization.h5'
#video_0_frames_name = '../jay/761624763_411424_20181009_video-0.h5'
#video_1_frames_name = '../jay/761624763_411424_20181009_video-1.h5'
#eye_data_name = '../../jay-sample-tracking/'\
#                '761624763_411424_20181009_video-1DeepCut_resnet50_'\
#                'pupil-and-eye-trackingJun12shuffle1_250000.csv'

pkl_file_name = '../jay/758519303_408021_20180926_stim.pkl'
syn_file_name = '../jay/758519303_408021_20180926_sync.h5'
df_pkl_name = '../jay/758519303_408021_20180926_df.pkl'
time_synch_file_name = '../jay/759038671_time_synchronization.h5'
video_0_frames_name = '../jay/758519303_408021_20180926_video-0.h5'
video_1_frames_name = '../jay/758519303_408021_20180926_video-1.h5'
eye_data_name = '../../jay-sample-tracking/758519303_408021_20180926_'\
                'video-1DeepCut_resnet50_'\
                'pupil-and-eye-trackingJun12shuffle1_250000.csv'

#file_names = {'pkl_file_name' : '../jay/758519303_408021_20180926_stim.pkl',
#              'syn_file_name' : '../jay/758519303_408021_20180926_sync.h5',
#              'df_pkl_name' : '../jay/758519303_408021_20180926_df.pkl',
#              'time_synch_file_name' : '../jay/759038671_time_'\
#                                       'synchronization.h5',
#              'video_0_frames_name' : '../jay/758519303_408021_20180926_'\
#                                      'video-0.h5',
#              'video_1_frames_name' : '../jay/758519303_408021_20180926_'\
#                                      'video-1.h5',
#              'eye_data_name' : '../../jay-sample-tracking/'\
#                                '758519303_408021_20180926_video-1DeepCut_'\
#                                'resnet50_pupil-and-eye-trackingJun12shuffle1'\
#                                '_250000.csv'}

file_names = {'pkl_file_name' : pkl_file_name,
              'syn_file_name' : syn_file_name,
              'df_pkl_name' : df_pkl_name,
              'time_synch_file_name' : time_synch_file_name,
              'video_0_frames_name' : video_0_frames_name,
              'video_1_frames_name' : video_1_frames_name,
              'eye_data_name' : eye_data_name}

#sess_sync_util.get_stim_frames(pkl_file_name, syn_file_name, df_pkl_name, runtype='prod')

#############################################
def behavioral_stim_synch(file_names):
    '''
    Returns:
        - Eye-tracking frames and run-data indices corresponding to 
          surprise frames.  In the case of consecutive surprise sequences, 
          only the frames corresponding to the first surprises in the sequence 
          are returned.
        - Times corresponding to each frame for eye-tracking and running data.
          0 corresponds to the beginning of the eye-tracking video.
    '''

    file = open(file_names['df_pkl_name'], 'rb')
    stim_panda = pickle.load(file)
    file.close()
        
#    eye_alignment: 
#        Indices are 2-photon frame numbers, values are eye-tracking
#        video (video-1) frames. Corroboration through bc_alignment
#        verification, below, and comparing, e.g., large movements
#        between eye-tracking and body-camera videos (should only be
#        at most a couple of frames off from each other, as indicated by
#        the two alignment arrays)
#    bc_alignment:
#        Indices are 2-photon frame numbers, values are body-camera video
#        (video-0) frames.  Can verify by comparing predicted surprise and
#        gray screens against the body camera (video-0).
#    stim_alignment:
#        Indices are stimulus frame numbers, values are 2-photon frame numbers.
#        NOTE: compared to stim_panda['stim_align'], may be slightly different
#        E.g., for 761624763_411424, the sizes are one off, as are the first 
#        and last values
    file = h5.File(file_names['time_synch_file_name'], 'r')
    eye_alignment = file['eye_tracking_alignment'].value
    stim_alignment = file['stimulus_alignment'].value
    bc_alignment=file['body_camera_alignment'].value
#    twop_fall = file['twop_vsync_fall'].value
    file.close()
    
    file = h5.File(file_names['video_1_frames_name'], 'r')
    frame_eye_int = file['frame_intervals'].value
    file.close()
        
    stim_df = stim_panda['stim_df']
    run = sess_sync_util.get_run_speed(file_names['pkl_file_name'])
    start_frame = np.array( stim_df['start_frame'][1:-1] ).astype(
            'int')

# stim_df = pd.DataFrame(index=list(range(np.sum(total_stimsegs))), 
#                           columns=['stimType', 'stimPar1', 'stimPar2', 'surp', 
#                                    'stimSeg', 'GABORFRAME', 'start_frame', 
#                                    'end_frame', 'num_frames'])

#    align_df = pd.DataFrame(index=list(stim_alignment.size),
#                            columns=['stim_frame', '2_photon_frame',
#                                     'eye_tracking_frame', 
#                                     'body_camera_frame'])


    
#    two_photon_frames = np.arange(0, eye_alignment.size)

#   index: two-photon frames.  values: stimulus frames    
    stim_frames = np.zeros(eye_alignment.shape)
    stim_frames[ 0 : stim_frames.size ] = np.nan
    stim_alignment_diff = np.append( 1, np.diff(stim_alignment) )
    start = stim_alignment[0].astype('int')
    stop = stim_alignment[-1].astype('int')
    stim_frames[ start : stop+1 ] = np.where( stim_alignment_diff )[0]
    
    
            

    surp_b_f = np.where(( stim_df['surp'][1:-1]==1) & 
                        (stim_df['stimType'][1:-1]=='b'))[0].squeeze()
    #we just want the first frames of any given sequence of bricks:
    surp_b_f_diff = np.append( 100, np.diff( surp_b_f ) )  
    surp_b_f = surp_b_f[ np.where( surp_b_f_diff > 1 ) ]
    
    tpsf = start_frame[ surp_b_f ]  #2-photon surprise frames
    surp_b_eye_f = eye_alignment[tpsf] + 3  #delay of ~0.1s to display on scrn
    
    
    
    surp_g_f = np.where(( stim_df['surp'][1:-1]==1) & 
                             (stim_df['stimType'][1:-1]=='g'))[0].squeeze()
    # we want frames for d/e stims.  4th in each seq
    surp_g_f_de = np.arange(1, surp_g_f.size + 1 )
    surp_g_f_de = np.where( np.mod(surp_g_f_de,4)==0 ) 
    
    surp_g_f = surp_g_f[ surp_g_f_de ]
    tpsf = start_frame[ surp_g_f ]  #2-phtn surprise frames
    surp_g_eye_f = eye_alignment[tpsf] + 3  #delay of ~0.1sec to display on scrn
    # we only want the first presentation of the surprise stimulus sequences:
    surp_g_eye_f_diff =  np.append(1000, np.diff(surp_g_eye_f) )
    surp_g_eye_f = surp_g_eye_f[ np.where( surp_g_eye_f_diff > 500 ) ]
    


    surp_b_run_f = np.where( np.isin( eye_alignment, surp_b_eye_f ) )
    surp_b_run_f = np.where( np.isin( stim_alignment, surp_b_run_f ) )[0]
    surp_g_run_f = np.where( np.isin( eye_alignment, surp_g_eye_f ) )
    surp_g_run_f = np.where( np.isin( stim_alignment, surp_g_run_f ) )[0]
    
    
    run_eye_f = stim_alignment[ 0:run.size ].astype('int')
    run_eye_f = eye_alignment[ run_eye_f ].astype('int')
    eye_time_eye = np.append( 0, np.cumsum(frame_eye_int) )
    eye_time_run = eye_time_eye[ run_eye_f ]

    bhv_stim_synch = dict( surp_b_eye_f = surp_b_eye_f,
                           surp_g_eye_f = surp_g_eye_f,
                           surp_b_run_f = surp_b_run_f,
                           surp_g_run_f = surp_g_run_f,
                           run = run,
                           run_eye_f = run_eye_f,
                           eye_time_eye = eye_time_eye,
                           eye_time_run = eye_time_run )
    alignment = dict( stim_panda = stim_panda,
                      eye_alignment = eye_alignment,
                      stim_alignment = stim_alignment,
                      stim_frames = stim_frames,
                      frame_eye_int = frame_eye_int )
    
    return bhv_stim_synch, alignment


#############################################
def eye_diam_center(eye_data_name):
    '''
    Returns the approximated pupil diameter, center, and frame-by-frame
    center differences (approximate derivative).  All in pixels.
    '''

    M = np.array( pd.read_csv(eye_data_name)[2:-1][:] ).astype('float64')
    
    # ordering of data (from config.yaml)is (w = whatever): 
    # w, left x2, w, right x2, w, top x2, w, bottom x2, w, 
    # lower left x2, w, upper left x2, w, upper right x2, w, lower right x2, w
    x = M[ :, [1,4,7,10,13,16,19,22] ]
    y = M[ :, [2,5,8,11,14,17,20,23] ]
    
    # dx and dy are pairwise distances of points furthest apart; ordering:
    # left -- right, top -- bottom, lower left -- upper right, upper left --
    # lower right
    dx = np.zeros( [x.shape[0], 4] )
    dy = np.zeros( [x.shape[0], 4] )
    dx[:,0] = np.abs( x[:,0]-x[:,1] )
    dx[:,1] = np.abs( x[:,2]-x[:,3] )
    dx[:,2] = np.abs( x[:,4]-x[:,6] )
    dx[:,3] = np.abs( x[:,5]-x[:,7] )
    dy[:,0] = np.abs( y[:,0]-y[:,1] )
    dy[:,1] = np.abs( y[:,2]-y[:,3] )
    dy[:,2] = np.abs( y[:,4]-y[:,6] )
    dy[:,3] = np.abs( y[:,5]-y[:,7] )
    
    
    # find diameters
    diams = np.sqrt( dx**2 + dy**2 )
    #max_diam = np.max( diams )
    #mean_diam = np.mean( diams )
    median_diam = np.median( diams, axis=1 )
    #min_diam = np.min( diams )
    
    # find centers and frame-to-frame differences
    center = np.transpose( [np.mean( x,axis=1 ), np.mean( y,axis=1 )] )
    center_diff = np.diff( center, axis=0 )
    center_dist_diff = np.sqrt( center_diff[:,0]**2 + center_diff[:,1]**2 )
    
    return median_diam, center, center_dist_diff


#############################################
def diam_no_blink(diam, thr=5):
    '''
    Returns the diameter without large deviations likely caused by blinks
    '''

    nan_diam = diam

    #Find aberrant blocks:
    diam_diff = np.append( 0, np.diff(diam) )
    diam_thr = np.where( np.abs(diam_diff) > thr )[0]
    diam_thr_diff = np.append( 1, np.diff(diam_thr) ) 
    
    diff_thr = 10 #how many consecutive frames should be non-aberrant
    searching = 1
    i = 0
    while( searching ):
        left = diam_thr[i]
        w =  np.where( diam_thr_diff[ i+1 : diam_thr_diff.size+1 
                                                  ] > diff_thr )[0]
        if w.size: #i.e., non-empty array
            right_i = np.min( w+i+1 ) - 1
            right = diam_thr[right_i]
        else:
            right = diam_thr[-1]
            searching = 0
        i = right_i + 1
        nan_diam[left:right+1] = np.nan
        
    return nan_diam
        

#############################################
def peristimulus_behavior(bhv_stm_synch, nan_diam, eyesec=3.5, runsec=2):
    '''
    Returns activity and average differences of eye and run data around 
    surprise frames
    '''
    surp_b_eye_f = bhv_stm_synch['surp_b_eye_f'].astype('int')    
    surp_g_eye_f = bhv_stm_synch['surp_g_eye_f'].astype('int')    
    surp_b_run_f = bhv_stm_synch['surp_b_run_f'].astype('int')    
    surp_g_run_f = bhv_stm_synch['surp_g_run_f'].astype('int')    
    run = bhv_stm_synch['run']

    preposteye = np.round( eyesec*30 ).astype('int')
    prepostrun = np.round( runsec*60 ).astype('int')
    size = np.round( np.max( [surp_b_eye_f.size, surp_g_eye_f.size, 
                    surp_b_run_f.size, surp_g_run_f.size] ) )

    prepost_b_eye = np.zeros( [surp_b_eye_f.size, 2*preposteye+1] )
    avgdiff_b_eye = np.zeros( [surp_b_eye_f.size, 1] )
    prepost_g_eye = np.zeros( [surp_g_eye_f.size, 2*preposteye+1] )
    avgdiff_g_eye = np.zeros( [surp_g_eye_f.size, 1] )

    prepost_b_run = np.zeros( [surp_b_run_f.size, 2*prepostrun+1] )
    avgdiff_b_run = np.zeros( [surp_b_run_f.size, 1] )
    prepost_g_run = np.zeros( [surp_g_run_f.size, 2*prepostrun+1] )
    avgdiff_g_run = np.zeros( [surp_g_run_f.size, 1] )

    for i in range(0, size):
        if i < surp_b_eye_f.size:
            t0 = surp_b_eye_f[i]
            prepost_b_eye[i, :] = nan_diam[ t0-preposteye : t0+preposteye+1 ]
            avgdiff_b_eye[i] = np.nanmean( nan_diam[ t0+1 : t0+preposteye+1 ] )-\
                               np.nanmean( nan_diam[ t0-preposteye : t0 ] )

        if i < surp_g_eye_f.size:
            t0 = surp_g_eye_f[i]
            prepost_g_eye[i, :] = nan_diam[ t0-preposteye : t0+preposteye+1 ]
            avgdiff_g_eye[i] = np.nanmean( nan_diam[ t0+1 : t0+preposteye+1 ] )-\
                               np.nanmean( nan_diam[ t0-preposteye : t0 ] )

        if i < surp_b_run_f.size:
            t0 = surp_b_run_f[i]
            prepost_b_run[i, :] = run[ t0-prepostrun : t0+prepostrun+1 ]
            avgdiff_b_run[i] = np.nanmean( run[ t0+1 : t0+prepostrun+1 ] )-\
                               np.nanmean( run[ t0-prepostrun : t0 ] )

        if i < surp_g_run_f.size:
            t0 = surp_g_run_f[i]
            prepost_g_run[i, :] = run[ t0-prepostrun : t0+prepostrun+1 ]
            avgdiff_g_run[i] = np.nanmean( run[ t0+1 : t0+prepostrun+1 ] )-\
                               np.nanmean( run[ t0-prepostrun : t0 ] )
                            
                            
    prepost = dict( prepost_b_eye = prepost_b_eye, 
                    prepost_g_eye = prepost_g_eye,
                    prepost_b_run = prepost_b_run, 
                    prepost_g_run = prepost_g_run )
    
    avgdiff = dict( avgdiff_b_eye = avgdiff_b_eye, 
                    avgdiff_g_eye = avgdiff_g_eye,
                    avgdiff_b_run = avgdiff_b_run, 
                    avgdiff_g_run = avgdiff_g_run )
    
    return prepost, avgdiff
    
    

#############################################
#############################################
t = time.time()

bhv_stm_synch, alignment = behavioral_stim_synch( file_names )
median_diam, center, center_dist_diff = eye_diam_center( file_names
                                                        ['eye_data_name'] )
nan_diam = diam_no_blink( copy.deepcopy(median_diam) )
prepost, avgdiff = peristimulus_behavior( bhv_stm_synch, nan_diam )

# do stuff
elapsed = time.time() - t
print(elapsed)
