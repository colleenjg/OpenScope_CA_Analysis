"""
mean_dff_over_days_analysis.py

This module run various anlyses on the tracked ROI USIs (Unexpected event
Selectivity Indices).

Authors: Jason E. Pina

Last modified: 29 May 2021
"""

import numpy as np
import pandas as pd
import scipy.stats as scist
import sys, copy, time
import itertools as it

sys.path.extend(['../..'])
from analysis import session
from util import gen_util, math_util


#############################################

def set_up_mouse_info(mouse_df_fnm):
    """
    Returns mouse dataframe and mouse numbers.
    
    Parameters
    ----------
    mouse_df_fnm : string
        File name for the dataframe that contains experiment information
        
    Returns
    -------
    mouse_df : Pandas DataFrame
        Dataframe with mouse/experiment information
    mouse_ns_full : 1-D array of numbers
        All mouse numbers
    mouse_ns_sess_123 : 1-D array of numbers
        Mouse numbers for mice with data from all of sessions 1-3
    """

    # declarations/initializations
    mouse_df = pd.read_csv(mouse_df_fnm) 
    # Restrict to production mice that passed and have all files
    mouse_df = mouse_df[(mouse_df.runtype=='prod') & (mouse_df.pass_fail=='P')
                        & (mouse_df.all_files)]
    # Obtain all mouse numbers
    mouse_ns_full = mouse_df['mouse_n'].unique()
    mouse_ns_sess_123 = []

    print('All mice: ', mouse_ns_full)

    for mouse_n in mouse_ns_full:
        # Obtain session numbers for each mouse
        sess_ns = mouse_df[(mouse_df.mouse_n==mouse_n)]['sess_n'].values
        # Add mouse to mouse numbers with data from sess. 1-3
        if np.sum(np.isin(sess_ns, np.asarray([1,2,3])))>=3:
            mouse_ns_sess_123.append(mouse_n)
        # Want mouse 8 as well for running.  will update pupil diameter later
        elif mouse_n==8:
            mouse_ns_sess_123.append(mouse_n)

    mouse_ns_sess_123 = np.asarray(mouse_ns_sess_123)
    print('Mice with all of sessions 1-3 :', mouse_ns_sess_123)
    
    return mouse_df, mouse_ns_full, mouse_ns_sess_123

#############################################

def make_session_averaged_df(mouse_df, mouse_df_fnm, datadir, mouse_ns, sess_ns,
                             stimtype_list, only_matched_rois):
    """
    Returns dataframes (1 each for Gabor and visual flow stimuli), organized by
    df/f session averages for each ROI for expected and unexpected events for 
    gabor and visual flow stimuli. Averages for Gabor stimuli are further 
    separated by Gabor frames (A, B, C, D, U, G).
    
    Parameters
    ----------
    dff_df : Pandas DataFrame
        Empty dataframe with column info filled out; this function fills in the 
        column data
    mouse_df : Pandas DataFrame
        Dataframe with mouse information
    mouse_df_fnm : string
        File name of mouse_df
    mouse_n : number
        Mouse number
    sess_n : number
        Session number
    stimtype : string
        Either 'gabors' or 'bricks'
    only_matched_rois : boolean
        If 'True', obtain df/f data only for ROIs that have been matched 
        across sessions
        
    Returns
    -------
    gab_dff_df, brk_dff_df : Pandas DataFrames
        Contains, for each stimulus type/mouse/session/ROI, df/f values averaged 
        across all sequence presentations in the session. For visual flow 
        (bricks), the data are segregated into expected and unexpected, while 
        for Gabors, the data are further grouped into Gabor frames (A, B, C, D,
        U, G)

        Columns:
            'mouse_n', 'sess_n', layer', 'compartment', 'stimtype', 'num_rois' :
                mouse number, session number, layer, and compartment, stimulus 
                type, and number of ROIs, respectively
            For Gabors:
                'expec_<frm>' for <frm> in {a,b,c,d,g} :
                    1-D array of df/f values for each ROI for specified expected 
                    Gabor frame, averaged over all presentations in the session
                'unexp_<frm>' for <frm> in {a,b,c,u,g} :
                    1-D array of df/f values for each ROI for specified 
                    unexpected (surprise) Gabor frame, averaged over all 
                    presentations in the session
            For visual flow (bricks):
                'brk_expec' :
                    1-D array of df/f values for each ROI for expected visual 
                    flow sequences, averaged over all sequences in the session
                'brk_unexp' :
                    1-D array of df/f values for each ROI for unexpected 
                    (surprise)  visual flow sequences, averaged over all 
                    sequences in the session
    """

    # declarations/initializations
    gab_dff_df = pd.DataFrame(columns = ['mouse_n', 'sess_n', 'layer', 
                                         'compartment', 'stimtype', 'num_rois',
                                         'expec_a', 'expec_b', 'expec_c', 
                                         'expec_d', 'expec_g',
                                         'unexp_a', 'unexp_b', 'unexp_c', 
                                         'unexp_u', 'unexp_g'])

    brk_dff_df = pd.DataFrame(columns = ['mouse_n', 'sess_n', 'layer', 
                                         'compartment', 'stimtype', 'num_rois',
                                         'brk_expec', 'brk_unexp'])
    layers_dict = {'L23-Cux2': 'L2/3', 'L5-Rbp4': 'L5'}
    gabfr_dict = {'A':0, 'B':1, 'C':2, 'D/U':3, 'G':3}
    t = time.time()

    # Loops to obtain data for each stimulus types, mouse, and session 
    for stimtype in stimtype_list:
        print('\nStimulus:', stimtype)
        for mouse_n in mouse_ns:
            if mouse_n > 1:
                print('\n\n')
            for sess_n in sess_ns:
                # Account for different mice's missing sessions
                if not np.isin(sess_n, mouse_df[mouse_df['mouse_n']==mouse_n]
                               ['sess_n'].values):
                    print('nope for mouse ', mouse_n, ', sess ', sess_n)
                    continue
                print('\nMouse ', mouse_n, '  Session', sess_n)
                
                # Finally, actually obtain new df/f data by adding line to 
                #  dataframe
                if stimtype=='gabors':
                    gab_dff_df = \
                        get_session_averaged_df_row(gab_dff_df, mouse_df, 
                                                    mouse_df_fnm, datadir, 
                                                    mouse_n, sess_n, stimtype, 
                                                    only_matched_rois)
                elif stimtype=='bricks':
                    brk_dff_df = \
                        get_session_averaged_df_row(brk_dff_df, mouse_df, 
                                                    mouse_df_fnm, datadir, 
                                                    mouse_n, sess_n, stimtype, 
                                                    only_matched_rois)    
    
        print( '{:.3f} sec\n'.format(time.time() - t) )        
            

    return gab_dff_df, brk_dff_df
    
#############################################

def get_session_averaged_df_row(dff_df, mouse_df, mouse_df_fnm, datadir, 
                                mouse_n, sess_n, stimtype, only_matched_rois):
    """
    Adds next line (for given mouse_n, sess_n) to dataframe (dff_df), organized 
    by df/f session averages for each ROI for expected and unexpected events for
    gabor and visual flow stimuli. Averages for Gabor stimuli are further 
    separated by Gabor frames (A, B, C, D, U, G).
    
    Parameters
    ----------
    dff_df : Pandas DataFrame
        Empty dataframe with column info filled out; this function fills in the 
        column data
    mouse_df : Pandas DataFrame
        Dataframe with mouse information
    mouse_df_fnm : string
        File name of mouse_df
    mouse_n : number
        Mouse number
    sess_n : number
        Session number
    stimtype : string
        Either 'gabors' or 'bricks'
    only_matched_rois : boolean
        If 'True', obtain df/f data only for ROIs that have been matched 
        across sessions
        
    Returns
    -------
    dff_df : Pandas DataFrame
        Contains, for each mouse/session/ROI, df/f values averaged across all 
        sequence presentations in the session. For visual flow (bricks), the 
        data are segregated into expected and unexpected, while for Gabors, the 
        data are further grouped into Gabor frames (A, B, C, D, U, G)

        Columns:
            'mouse_n', 'sess_n', layer', 'compartment', 'stimtype', 'num_rois' :
                mouse number, session number, layer, and compartment, stimulus 
                type, and number of ROIs, respectively
            For Gabors:
                'expec_<frm>' for <frm> in {a,b,c,d,g} :
                    1-D array of df/f values for each ROI for specified expected 
                    Gabor frame, averaged over all presentations in the session
                'unexp_<frm>' for <frm> in {a,b,c,u,g} :
                    1-D array of df/f values for each ROI for specified 
                    unexpected (surprise) Gabor frame, averaged over all 
                    presentations in the session
            For visual flow (bricks):
                'brk_expec' :
                    1-D array of df/f values for each ROI for expected visual 
                    flow sequences, averaged over all sequences in the session
                'brk_unexp' :
                    1-D array of df/f values for each ROI for unexpected 
                    (surprise) visual flow sequences, averaged over all 
                    sequences in the session
    """
    
    # declarations/initializations
    layers_dict = {'L23-Cux2': 'L2/3', 'L5-Rbp4': 'L5'}
    gabfr_dict = {'A':0, 'B':1, 'C':2, 'D/U':3, 'G':3}
    idx = dff_df.shape[0]
    layer = layers_dict[mouse_df[(mouse_df['mouse_n']==mouse_n) & 
                                 (mouse_df['sess_n']==sess_n)]['line'].
                                 values[0]]
    compartment = mouse_df[(mouse_df['mouse_n']==mouse_n) & 
                           (mouse_df['sess_n']==sess_n)]['plane'].values[0]

    sessid = mouse_df[(mouse_df['mouse_n']==mouse_n) & 
                      (mouse_df['sess_n']==sess_n)]['sessid'].values[0]
    expec_dff = {}
    unexp_dff = {}

    # Obtain session objects
    sess = session.Session(datadir, sessid, only_matched_rois=only_matched_rois)
    sess.extract_sess_attribs(mouse_df=mouse_df_fnm)
    sess.extract_info(fulldict=False, roi=True, run=False, pupil=False) 
    stim = sess.get_stim(stimtype)

    # Obtain df/f for gabors or visual flow, averaged over trials / repeat 
    #  presentations
    print('Getting data for {}'.format(stimtype))
    if stimtype=='gabors':
        for fr in gabfr_dict.keys():
            expec_dff[fr] = get_dff_tensor(stim, stimtype, fr, gabfr_dict, 
                                           surp=0)
            unexp_dff[fr]  = get_dff_tensor(stim, stimtype, fr, gabfr_dict, 
                                            surp=1)
            n_rois = expec_dff[fr].shape[0]
        dff_df.loc[idx] = [mouse_n, sess_n, layer, compartment, stimtype, 
                           n_rois,
                           expec_dff['A'], expec_dff['B'], expec_dff['C'],
                           expec_dff['D/U'], expec_dff['G'],
                           unexp_dff['A'], unexp_dff['B'], unexp_dff['C'],
                           unexp_dff['D/U'], unexp_dff['G']]
    elif stimtype=='bricks':
        # Frame and gabfr_dict are dummies here
        expec_dff = get_dff_tensor(stim, stimtype, 'A', gabfr_dict, surp=0)
        unexp_dff = get_dff_tensor(stim, stimtype, 'A', gabfr_dict, surp=1)
        n_rois = expec_dff.shape[0]
        dff_df.loc[idx] = [mouse_n, sess_n, layer, compartment, stimtype, 
                           n_rois,
                           expec_dff, unexp_dff]


    return dff_df

#############################################

def get_dff_tensor(stim, stimtype, fr, gabfr_dict, surp):
    """
    Provides df/f data integrated in time and averaged over all sequences in the
    session in a 1-D array of length = number of ROIs
    
    Parameters
    ----------
    stim : stimulus object
        Object used to access df/f data for each mouse/session 
    stimtype : string
        Either 'gabors' or 'bricks'
    fr : String
        Gabor frame ('A', 'B', 'C', 'D/U', 'G')
    gabfr_dict : dictionary
        Matches the Gabor frame ('fr', above) with the frame number used in the 
        codebase to access that particular frame
    surp : number
        Either 0 (for expected sequences) or 1 (for unexpected/surprise 
        sequences) 

    Returns
    -------
        dff_data : 1-D array
            Df/f data, averaged over all sequences in the session for each ROI
    """
    
    # declarations/initializations
    prepost_dict = {'gabors':{'A':{'pre':0, 'post':0.3},
                              'B':{'pre':0, 'post':0.3},
                              'C':{'pre':0, 'post':0.3},
                              'D/U':{'pre':0, 'post':0.3},
                              'G':{'pre':-0.3, 'post':0.6}},
                    'bricks':{'pre':0,'post':1}}
    if stimtype=='gabors':
        pre = prepost_dict[stimtype][fr]['pre']
        post = prepost_dict[stimtype][fr]['post']
    elif stimtype=='bricks':
        pre = prepost_dict[stimtype]['pre']
        post = prepost_dict[stimtype]['post']
        print('pre, post = ', pre, post)

    # Obtain df/f values for given session (provided by 'stim' object).  
    #  'remconsec' set to false because we need all frame presentations
    seg_ns = stim.get_segs_by_criteria(surp=surp, gabfr=gabfr_dict[fr], 
                                       remconsec=False, by='seg')
    twop_fr_ns = stim.get_twop_fr_by_seg(seg_ns, first=True, 
                                         ch_fl=[pre, post])['first_twop_fr']
    # Note: since we're interested in comparing the mean dff across days, 
    #  we set scale to false
    roi_data_df = stim.get_roi_data(twop_fr_ns, pre, post, integ=True, 
                                    scale=False)
    # roi x seq
    dff_data = gen_util.reshape_df_data(roi_data_df, squeeze_cols=True)
    # Average over all sequences in session
    dff_data = np.nanmean(dff_data, axis=1)
    if fr=='A' and surp==0:
        print('Number of ROIs:', dff_data.shape[0])
    
    return dff_data

#############################################

def make_normed_summary_dfs(gab_mn_dff_df, brk_mn_dff_df, sess_ns, 
                            n_perm=1e5, bonf_n=24):
    """
    Returns normalized dataframes, organized by layer/compartment.
    
    Parameters
    ----------
    gab_mn_dff_df, brk_mn_dff_df : Pandas DataFrames
        Unnormalized dataframe w/ session-averaged df/f values for each 
        mouse/sess/ROI.
    sess_ns : 1-D arraylike of numbers
        Sessions for which we have df/f data in the dataframes.
    n_perm : number
        Number of permutations to use in permutation tests for p-values.
    bonf_n : number
        Number of multiple comparisons to correct for via Bonferroni correction.
    
    Returns
    -------
    gab_abc_lyr_cmpt_dff_normed_df, gab_dug_lyr_cmpt_dff_normed_df,
     brk_lyr_cmpt_dff_normed_df : Pandas DataFrames
        Contains, for each mouse/session/ROI, df/f values averaged across all 
        sequence presentations in the session. For visual flow (bricks), the 
        data are segregated into expected and unexpected, while for Gabors, the
        data are further grouped into Gabor frames (A, B, C, D, U, G)

        Columns:
            'mouse_n', 'sess_n', layer', 'compartment', 'stimtype', 'num_rois' :
                mouse number, session number, layer, and compartment, stimulus
                type, and number of ROIs, respectively
            For Gabors:
                'expec_<frm>' for <frm> in {a,b,c,d,g} :
                    1-D array of df/f values for each ROI for specified expected 
                    Gabor frame, averaged over all presentations in the session
                    and divided by the mean 
                'unexp_<frm>' for <frm> in {a,b,c,u,g} :
                    1-D array of df/f values for each ROI for specified 
                    unexpected (surprise) Gabor frame, averaged over all 
                    presentations in the session
            For visual flow (bricks):
                'brk_expec' :
                    1-D array of df/f values for each ROI for expected visual 
                    flow sequences, averaged over all sequences in the session
                'brk_unexp' :
                    1-D array of df/f values for each ROI for unexpected 
                    (surprise) visual flow sequences, averaged over all 
                    sequences in the session

    """

    t = time.time()
    layers = ['L2/3', 'L5']
    compartments = ['dend', 'soma']
    
    print('Making normalized dataframes\n')
    gab_mn_dff_normed_df = make_normed_df(gab_mn_dff_df, 'gabors')
    brk_mn_dff_normed_df = make_normed_df(brk_mn_dff_df, 'bricks')
    
    print('\n\nMake summarized dataframe for frames A, B, C\n')
    frames = ['A', 'B', 'C']
    gab_abc_lyr_cmpt_dff_normed_df = \
        make_summary_df(gab_mn_dff_normed_df, sess_ns, frames, 
                        compartments, layers)
    frames = ['D/U', 'G']
    gab_dug_lyr_cmpt_dff_normed_df = \
        make_summary_df(gab_mn_dff_normed_df, sess_ns, frames, 
                        compartments, layers)

    frames = ['brk']
    brk_lyr_cmpt_dff_normed_df = make_summary_df(brk_mn_dff_normed_df, sess_ns, 
                                                 frames, compartments, layers)

    for expec_str in ['expec', 'unexp']:
        print(expec_str, 'sequences')
        print('\n\nCompute intersession p-values for Gabor frames A, B, C\n')
        compute_intersession_pvals(gab_abc_lyr_cmpt_dff_normed_df, expec_str, 
                                   compartments, layers, n_perm, bonf_n)

        print('\n\nCompute intersession p-values for Gabor frames D/U, G\n')
        compute_intersession_pvals(gab_dug_lyr_cmpt_dff_normed_df, expec_str, 
                                   compartments, layers, n_perm, bonf_n)
        
        print('\n\nCompute intersession p-values for bricks\n')
        compute_intersession_pvals(brk_lyr_cmpt_dff_normed_df, expec_str, 
                                   compartments, layers, n_perm, bonf_n)
        print('{:.01f} sec'.format(time.time()-t))
        
    return gab_abc_lyr_cmpt_dff_normed_df, gab_dug_lyr_cmpt_dff_normed_df, \
           brk_lyr_cmpt_dff_normed_df

#############################################

def make_normed_df(dff_df, stimtype):
    """"
    Makes normalized df/f dataframe based on EXPECTED (/REGULAR) responses from 
    sess 1 for each mouse to allow to compare trends across mice (else one 
    mouse's df/f values could overwhelm those of others for some sessions).
    
    Parameters
    ----------
    dff_df : Pandas DataFrame
        Unnormalized dataframe w/ session-averaged df/f values for each 
        mouse/sess/ROI
    stimtype : string
        Either 'gabors' or 'bricks'
    
    Returns
    -------
    dff_normed_df : Pandas DataFrame
        Contains, for each mouse/session/ROI, df/f values averaged across all 
        sequence presentations in the session. For visual flow (bricks), the 
        data are segregated into expected and unexpected, while for Gabors, the
        data are further grouped into Gabor frames (A, B, C, D, U, G)

        Columns:
            'mouse_n', 'sess_n', layer', 'compartment', 'stimtype', 'num_rois' :
                mouse number, session number, layer, and compartment, stimulus 
                type, and number of ROIs, respectively
            For Gabors:
                'expec_<frm>' for <frm> in {a,b,c,d,g} :
                    1-D array of df/f values for each ROI for specified expected 
                    Gabor frame, averaged over all presentations in the session
                    and divided by the mean 
                'unexp_<frm>' for <frm> in {a,b,c,u,g} :
                    1-D array of df/f values for each ROI for specified 
                    unexpected (surprise) Gabor frame, averaged over all 
                    presentations in the session
            For visual flow (bricks):
                'brk_expec' :
                    1-D array of df/f values for each ROI for expected visual 
                    flow sequences, averaged over all sequences in the session
                'brk_unexp' :
                    1-D array of df/f values for each ROI for unexpected 
                    (surprise) visual flow sequences, averaged over all 
                    sequences in the session
    """

    # declarations/initializations
    if stimtype=='gabors':
        dff_normed_df = pd.DataFrame( columns = ['mouse_n', 'sess_n', 'layer', 
                                                 'compartment', 
                                                 'stimtype', 'num_rois',
                                                 'expec_a', 'expec_b', 
                                                 'expec_c', 'expec_d', 
                                                 'expec_g',
                                                 'unexp_a', 'unexp_b', 
                                                 'unexp_c', 'unexp_u', 
                                                 'unexp_g'] )
    elif stimtype=='bricks':
        dff_normed_df = pd.DataFrame( columns = ['mouse_n', 'sess_n', 'layer', 
                                                 'compartment', 
                                                 'stimtype', 'num_rois',
                                                 'brk_expec', 'brk_unexp'] )
    shared_keys = ['mouse_n', 'sess_n', 'layer', 'compartment',
                   'stimtype', 'num_rois']
    mn_dict = {'gabors':
               {'expec_a':[], 'expec_b':[], 'expec_c':[], 
                'expec_d':[], 'expec_g':[],
                'unexp_a':[], 'unexp_b':[],  'unexp_c':[],  
                'unexp_u':[], 'unexp_g':[]},
               'bricks':
               {'brk_expec':[], 'brk_unexp':[]}}
    mask_sess_1 = dff_df['sess_n']==1
    mouse_ns = dff_df['mouse_n'].unique()

    # Construct normalized dataframe
    for mouse_n in mouse_ns:
        print('Mouse ', mouse_n)
        mask_ms = dff_df['mouse_n']==mouse_n
        masks = mask_sess_1 & mask_ms
        # Obtain mean values from session 1 from original df. Will divide by 
        #  these below
        for key in mn_dict[stimtype].keys():
            mn_dict[stimtype][key] = np.nanmean(dff_df[masks][key].values[0])
        # Set values for normed df from raw df.  This populates the normed df 
        #  rows
        for key in shared_keys:
            dff_normed_df[key] = dff_df[key]
        # Note: extra "_n" at end of masks indicates they are for normed df. 
        #  Else orig
        mask_ms_n = dff_normed_df['mouse_n']==mouse_n
        # Loop over sessions to get data for each sess
        for sess_n in dff_df[mask_ms]['sess_n'].unique():
            # Set masks in orig and normed dfs to work on same items
            mask_sess = dff_df['sess_n']==sess_n
            mask_sess_n = dff_normed_df['sess_n']==sess_n
            masks = mask_ms & mask_sess
            masks_n = mask_ms_n & mask_sess_n
            # Compute normed_df values by dividing all values by 
            #  expected/regular means from session 1
            for key in mn_dict[stimtype].keys():
                # Get index for normed df reference
                idx = dff_df[masks].index[0]
                divisor_key = None
                # Set the key to use for determining divisor
                if stimtype=='gabors':
                    frame = key.split('_')[1]
                    # Match (un)expected frames of all sessions with expected 
                    #  frames from session 1. If unexpected 'u' frame, match 
                    #  with expected 'd' frame
                    frame = 'd' if frame.find('u')>=0 else frame
                    divisor_key = 'expec_' + frame
                else:
                    divisor_key = 'brk_expec'
                # Finally!  compute the normalized values
                dff_normed_df.loc[idx, key] = \
                    dff_df.loc[idx, key] / mn_dict[stimtype][divisor_key]
    return dff_normed_df

#############################################

def make_summary_df(dff_df, which_sessns, frames, compartments, layers):

    '''
    Makes dataframe, with the session-averaged df/f values for all ROIs for each 
    layer and compartment and the associated mean and standard errors over ROIs.
    
    Parameters
    ----------
    dff_df : Pandas DataFrame
        Contains, for each mouse/session/ROI, df/f values averaged across all 
        sequence
        presentations in the session. For visual flow (bricks), the data are 
        segregated into expected and unexpected, while for Gabors, the data are 
        further grouped into Gabor frames (A, B, C, D, U, G).  Note, can be 
        normalized or unnormalized data.
    which_sessns : 1-D arraylike of numbers
        Sessions for which we have df/f data in dff_df.
    frames : 1-D arraylike of chars
        Gabor frames for which to take df/f data in dff_df. This is a dummy 
        variable for visual flow (bricks).
    compartments : 1-D arraylike of strings
        Compartments ('dend', 'soma') for which to obtain data.
    layers : 1-D arraylike of strings
        Layers ('L2/3', 'L5') for which to obtain data.
    
    
    Returns
    -------
    summary_df : Pandas DataFrame
        Dataframe, grouped by layer/compartment, with session-averaged 
        (un)expected df/f values for each ROI, and the associated means and 
        standard errors
        
        Columns:
            'layer', 'compartment', 'mouse_ns' :
                layer and compartment (L2/3 or L5 soma or dendrites) and the 
                associated mice numbers
            'sess_123_num_rois' :
                1-D array with the total number of ROIs for sessions 1, 2, and 3
                in order. 
            'expec_dff__all_rois' :
                1-D array with the session-averaged df/f values for expected 
                sequences. The first sess_123_num_rois[0] belong to session 1, 
                the next sess_123_num_rois[1] belong to session 2, and similarly
                for session 3 (thus there are a total of sum(sess_123_num_rois)
                entries).
            'expec_dff__mn__sess_123' :
                1-D array with the ROI-averaged df/f values in 
                expec_dff__all_rois for each session
            'expec_dff__se__sess_123' :
                1-D array with the standard error over ROIs of the df/f values 
                in expec_dff__all_rois for each session
            'unexpec_dff__all_rois' :
                1-D array with the session-averaged df/f values for unexpected 
                sequences. The first sess_123_num_rois[0] belong to session 1, 
                the next sess_123_num_rois[1] belong to session 2, and similarly
                for session 3 (thus there are a total of sum(sess_123_num_rois)
                entries).
            'unexpec_dff__mn__sess_123' :
                1-D array with the ROI-averaged df/f values in 
                unexpec_dff__all_rois for each session
            'unexpec_dff__se__sess_123' :
                1-D array with the standard error over ROIs of the df/f values
                in unexpec_dff__all_rois for each session
    '''

    # declarations/initializations
    summary_df = \
        pd.DataFrame(columns = ['layer', 'compartment', 'mouse_ns', 
                                'sess_123_num_rois',
                                'expec_dff__all_rois', 
                                'expec_dff__mn__sess_123', 
                                'expec_dff__se__sess_123',
                                'unexp_dff__all_rois', 
                                'unexp_dff__mn__sess_123', 
                                'unexp_dff__se__sess_123'])

    
    # Compute (un)expected df/f values and stats for each compartment/layer
    for compartment, layer in it.product(compartments, layers):
        # declare arrays/lists
        mouse_ns = np.array([])
        n_rois = np.empty((len(which_sessns,))).astype('int')
        expec_dff_list = []
        expec_dff_mn = np.empty((len(which_sessns,)))
        expec_dff_se = np.empty((len(which_sessns,)))
        unexp_dff_list = []
        unexp_dff_mn = np.empty((len(which_sessns,)))
        unexp_dff_se = np.empty((len(which_sessns,)))
        
        # Compute for each session
        for i_sess, which_sess in enumerate(which_sessns):
#             mouse_ns0, n_rois[i_sess], \
            # Obtain mouse numbers, number of ROIs, and expected-sequence data
            mouse_ns, n_rois[i_sess], \
             expec_dff, expec_dff_mn[i_sess], expec_dff_se[i_sess] = \
                compute_sess_dff_stats(dff_df, which_sess, frames, 
                                       compartment, layer, expec_str='expec', 
                                       lcm='lc')
            expec_dff_list.append(expec_dff)
            
            # Obtain unexpected-sequence data
            _, _, unexp_dff, unexp_dff_mn[i_sess], unexp_dff_se[i_sess] = \
                compute_sess_dff_stats(dff_df, which_sess, frames, 
                                       compartment, layer, expec_str='unexp', 
                                       lcm='lc')
            # mouse_ns includes all mice that have at least one session for this 
            #  layer/compartment
#             if len(mouse_ns0) > len(mouse_ns):
#                 mouse_ns = mouse_ns0
            unexp_dff_list.append(unexp_dff)
        
            
        #######################

        # Record data to summary_df
        idx = summary_df.shape[0]
        summary_df.loc[idx] = \
            [layer, compartment, mouse_ns,
             np.array([n_rois[0], n_rois[1], n_rois[2]]),
             np.hstack([expec_dff_list[0], expec_dff_list[1], 
                        expec_dff_list[2]]),
             np.array([expec_dff_mn[0], expec_dff_mn[1], expec_dff_mn[2]]),
             np.array([expec_dff_se[0], expec_dff_se[1], expec_dff_se[2]]),
             np.hstack([unexp_dff_list[0], unexp_dff_list[1], 
                        unexp_dff_list[2]]),
             np.array([unexp_dff_mn[0], unexp_dff_mn[1], unexp_dff_mn[2]]),
             np.array([unexp_dff_se[0], unexp_dff_se[1], unexp_dff_se[2]])]

    return summary_df

#############################################

def compute_sess_dff_stats(dff_df, which_sess_n, frames, compartment='blank', 
                           layer='blank', mouse_n=-1, expec_str='expec', 
                           lcm='lc'):
    """
    Compute df/f values/statistics for a given session, surprise, and possibly
    Gabor frames.
    
    Parameters
    ----------
    dff_df : Pandas DataFrame
        Contains, for each mouse/session/ROI, df/f values averaged across all 
        sequence presentations in the session. For visual flow (bricks), the 
        data are segregated into expected and unexpected, while for Gabors, the
        data are further grouped into Gabor frames (A, B, C, D, U, G).  Note, 
        can be normalized or unnormalized data.
    which_sess_n : number
        Session number for which to obtain data.
    frames : 1-D arraylike of chars
        Gabor frames for which to obtain data; dummy variable for visual flow 
        (bricks).
    compartment : string; optional, default = 'blank'
        Compartment ('dend', 'soma') for which to obtain data.
    layer : string; optional, default = 'blank'
        Layer ('L2/3', 'L5') for which to obtain data.
    mouse_n : number; optional, default = -1
        Mouse number for which to obtain data. Default is -1 to ensure that if 
        data is desired for all mice in layer/compartment, than a mask based off
        of mouse_n will not erroneously obtain undesired data.
    expec_str : string; optional, default = 'expec'
        Obtain non-surprise ('expec') or surprise ('unexp') data.
    lcm : string; optional, default = 'lc'
        Obtain data for all mice for a given layer/compartment ('lc') or for a 
        specified mouse number ('m').
    
    Returns
    -------
    mouse_ns : 1-D array of numbers
        Mouse numbers associated with the given layer/compartment or mouse_n/
    n_rois : number
        Total number of ROIs for the given session and either mouse_n or 
        layer/compartment/
    dff : 1-D array of numbers
        Df/f data for each ROI, averaged over 'frames'.
    dff_mn : number
        Average over ROIs of dff.
    dff_se : number
        Standard error over ROIs of dff.
    """
    
    # declarations/initializations
    frms_dict = {'expec':
                 {'A':'expec_a', 'B':'expec_b', 'C':'expec_c', 'D/U':'expec_d', 
                  'G':'expec_g', 
                  'brk':'brk_expec'},
                 'unexp':
                 {'A':'unexp_a', 'B':'unexp_b', 'C':'unexp_c', 'D/U':'unexp_u', 
                  'G':'unexp_g',
                  'brk':'brk_unexp'}}
    # Get masks for all mice in the given layer/compartment
    if lcm=='lc':
        mask0 = dff_df.compartment==compartment
        mask1 = dff_df.layer==layer
    # Get masks for the specified mouse
    elif lcm=='m':
        mask0 = dff_df.mouse_n==mouse_n
        mask1 = mask0
    mask_sess = dff_df['sess_n']==which_sess_n
    masks = mask0 & mask1 & mask_sess

    # Obtain metadata
    mouse_ns = np.hstack(dff_df[masks]['mouse_n'].values)
    n_rois = np.sum(dff_df[masks]['num_rois'].values)
    dff_frms = []
    # Compute df/f for each frame in the list of frames and amalgamate over 
    #  frames
    for frame in frames:
        dff = np.hstack(dff_df[masks][frms_dict[expec_str][frame]].values)
        dff_frms.append(dff)
    # frames x roi
    dff = np.array(dff_frms)
    # Average over frames
    dff = np.nanmean(dff, axis=0)
    # Compute dff stats over ROIs
    dff_mn = np.nanmean(dff)
    dff_se = np.nanstd(dff) / np.sqrt(dff.size)
    
    return mouse_ns, n_rois, dff, dff_mn, dff_se

#############################################

def compute_intersession_pvals(df, expec_str, compartments, layers, n_perm, 
                               bonf_n):
    """
    Compute *Bonferroni-corrected, 2-tail* p-values of differences between 
    sessions 
    from normalized or unnormalized summary dataframe. Adds these columns and 
    values to input argument 'df'. Note, to determine raw 2-tailed p-values, 
    simply divide recorded values by 'bonf_n' (this is why corrected p-values 
    are output, rather than max(p_val, 1))
    
    Parameters
    ----------
    df : Pandas DataFrame
        Contains, for each mouse/session/ROI, df/f values averaged across all 
        sequence presentations in the session. For visual flow (bricks), the 
        data are segregated into expected and unexpected, while for Gabors, the 
        data are further grouped into Gabor frames (A, B, C, D, U, G).  Note, 
        can be normalized or unnormalized data.
    expec_str : string
        Specifies whether to compute values for expected ('expec') or unexpected
        ('unexpec') sequences.
    compartments : 1-D arraylike of strings
        Compartments ('dend', 'soma') for which to obtain data.
    layers : 1-D arraylike of strings
        Layers ('L2/3', 'L5') for which to obtain data.
    n_perm : number
        Number of permutations to use to calculate p-values.
    """
    
    # declarations/initializations
    dff_dict = {'expec':{'dff':'expec_dff__all_rois', 
                         'dff_mn':'expec_dff__mn__sess_123',
                         'dff_se':'expec_dff__se__sess_123'},
                'unexp':{'dff':'unexp_dff__all_rois', 
                        'dff_mn':'unexp_dff__mn__sess_123',
                        'dff_se':'unexp_dff__se__sess_123'}}
    pval_1_2 = []
    pval_2_3 = []
    pval_1_3 = []
    n_perm = int(n_perm)

    # Main loop over compartments/layers
    for compartment, layer in it.product(compartments, layers):
        print( 'layer', layer, 'compartment', compartment )
        # declare and initialize distribution arrays 
        distro__1_2 = np.empty((n_perm,))
        distro__2_3 = np.empty((n_perm,))
        distro__1_3 = np.empty((n_perm,))
        distro__1_2[:] = np.nan
        distro__2_3[:] = np.nan
        distro__1_3[:] = np.nan

        # Set layer and compartment masks for dataframe and obtain mouse
        mask0 = df.compartment==compartment
        mask1 = df.layer==layer
        masks = mask0 & mask1
#         lc_mouse_ns = df[masks]['mouse_ns'].values[0]

        # Obtain df/f data and statistics
        dff = df[masks][dff_dict[expec_str]['dff']].values[0]
        dff_mn = df[masks][dff_dict[expec_str]['dff_mn']].values[0]
        dff_se = df[masks][dff_dict[expec_str]['dff_se']].values[0]
        n_rois = df[masks]['sess_123_num_rois'].values[0]

        # Determine mean differences between sessions
        dff_diff__1_2 = dff_mn[0] - dff_mn[1]
        dff_diff__2_3 = dff_mn[1] - dff_mn[2]
        dff_diff__1_3 = dff_mn[0] - dff_mn[2]

        # Permutation loop; this can be vectorized if desired; will depend on 
        #  n_perm and system memory
        for i in range(n_perm):
            # Shuffle session labels by choosing a random ROI index permutation
            random_rois = \
                np.random.choice(np.arange(0,dff.size), size=dff.size, 
                                 replace=False)
            # Compute permuation means, using the above ROI permutation
            dff_random_sess_1 = np.nanmean(dff[random_rois[:n_rois[0]]])
            dff_random_sess_2 = np.nanmean(dff[random_rois[n_rois[0]:
                                                           n_rois[0]+
                                                           n_rois[1]]])
            dff_random_sess_3 = np.nanmean(dff[random_rois[n_rois[0]+
                                                           n_rois[1]:]])
            # Compute shuffled mean differences and add to distributions
            distro__1_2[i] = dff_random_sess_1 - dff_random_sess_2
            distro__2_3[i] = dff_random_sess_2 - dff_random_sess_3
            distro__1_3[i] = dff_random_sess_1 - dff_random_sess_3   
                
        # Compute p-values for each session comparison
        pval_1_2.append(compute_2_tailed_pval(dff_diff__1_2, distro__1_2))
        pval_2_3.append(compute_2_tailed_pval(dff_diff__2_3, distro__2_3))
        pval_1_3.append(compute_2_tailed_pval(dff_diff__1_3, distro__1_3))
    
    # Add p-value data to dataframe
    df['pval_1_2__{}'.format(expec_str)] = np.asarray(pval_1_2) * bonf_n
    df['pval_2_3__{}'.format(expec_str)] = np.asarray(pval_2_3) * bonf_n
    df['pval_1_3__{}'.format(expec_str)] = np.asarray(pval_1_3) * bonf_n

#############################################

def compute_2_tailed_pval(value, distro):
    """
    Compute 2-tailed p-value 
    
    Parameters
    ----------
    value : number
        value for which to ascertain the p-value
    distro : 1-D array of numbers
        computed distribution against which to compare 'value' to ascertain the 
        p-value
    
    Returns
    -------
    pval : number
        computed 2-tailed p-value
    """
    distro = np.asarray(distro)
    n_perm_idcs = distro.size
    # Form array of indices where 'value' is no greater than the distribution 
    #  values
    perm_idcs_larger = np.where(distro >= value)[0]
    # The probability is then the raio of the length of this array to the 
    #  distribution size
    pval = len(perm_idcs_larger) / n_perm_idcs
    # 2-tailed correction
    if pval > 0.5:
        pval = 1-pval
    pval *= 2
    if np.isnan(value):
        pval = np.nan
    
    return pval

#############################################