"""
trkd_roi_analysis.py

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
from pathlib import Path

sys.path.extend([str(Path('..', '..'))])
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

def make_usi_df(stimtype, mouse_df, mouse_df_fnm, mouse_ns, datadir, 
                brk_dir = 'any', 
                op = 'd-prime', 
                only_matched_rois = True, 
                sess_ns = [1,2,3],
                scale = True, 
                remnans = False):

    """
    Returns a dataframe that contains, for each tracked ROI,
    USI, mean and std of the df/f for (un)expected events over each session.
    
    Parameters
    ----------
    stimtype : string
        Either 'gabors' or 'bricks'.
    mouse_df : Pandas DataFrame
        Dataframe with mouse information
    mouse_df_fnm : string
        File name for 'mouse_df'
    mouse_ns : 1-D arraylike of numbers
        Contains all mouse numbers for which to obtain data
    datadir : string
        Data directory
    brk_dir : string;  optional, default = 'any'
        Options: 'right', 'left', 'temp', 'nasal' or 'any' (default).  Specify 
        brick direction if desired.
    op : string; optional, default = 'd-prime'
        Options: 'diff', 'discr', 'd-prime' (default).  Operation to use when 
        computing USIs.
    only_matched_rois : boolean; optional, default = True
        If 'True' (default), only USIs for ROIs that have been matched across 
        sessions (and contain no NaNs in any recorded session) are included.  
        Else, USIs for all ROIs are included (excluding ROIs with NaN df/f 
        values for any particular session if 'remnans' argument is set to True).
    sess_ns : 1-D arraylike; optional, default = [1,2,3]
        Which sessions to include for selected mice.
    scale : boolean; optional, default = True
        Option to scale each ROI by its statistics for each session.  Rescales 
        through an analog to a z-score, where the mean is replaced by the 
        median, and the standard deviation is replaced by the 5% to 95% 
        interval.  This allows ROIs to be treated more equally to others in a 
        given session, compensating for calcium signal issues, e.g.
    remnans : boolean; optional, default = False
        Option to discount ROIs that have NaN df/f values.  Note, this is 
        redundant if 'only_matched_rois' is set to True.
        
    Returns
    -------
    usi_df (pd DataFrame) : Pandas dataframe 
        Contains, for each tracked ROI, USI, mean and std of the df/f for 
        (un)expected events over each session
        
        Columns:
            'mouse_n', 'layer', 'compartment' :
                mouse number, layer, and compartment, respectively
            'sess_<session>_usi' for <session> in [1,2,3] : 
                USI values for given sessions
            'sess_<session>_expec_evnt_<statistic>' for <session> in [1,2,3], 
            <statistic> in ['mn', 'std'] : 
                mean and std of integrated df/f for each ROI over all 
                expected events in sess. (D/G frames or expected flow)
            'sess_<session>_unexp_evnt_<statistic>' for <session> in [1,2,3], 
            <statistic> in ['mn', 'std'] : 
                mean and std of integrated df/f for each ROI over all expected 
                events in sess. (U/G frames or unexpected flow)
    """
    
    # declarations/initializations
    t = time.time()
    layers_dict = {'L23-Cux2': 'L2/3', 'L5-Rbp4': 'L5'}

    tracked_roi_usi_df = \
        pd.DataFrame(columns=['mouse_n', 'layer', 'compartment',
            'sess_1_usi', 'sess_2_usi', 'sess_3_usi',
            'sess_1_expec_evnt_mn', 'sess_2_expec_evnt_mn', 
            'sess_3_expec_evnt_mn',
            'sess_1_expec_evnt_std', 'sess_2_expec_evnt_std', 
            'sess_3_expec_evnt_std',
            'sess_1_unexp_evnt_mn', 'sess_2_unexp_evnt_mn',
            'sess_3_unexp_evnt_mn',
            'sess_1_unexp_evnt_std', 'sess_2_unexp_evnt_std', 
            'sess_3_unexp_evnt_std'])

    # Loop over all given mouse numbers
    for idx, mouse_n in enumerate(mouse_ns):
        # declarations/initializations
        unexp_mn = []
        unexp_std = []
        expec_mn = []
        expec_std = []
        usi = []
        # Determine layer and compartment
        layer = layers_dict[mouse_df[mouse_df['mouse_n']==mouse_n]['line'].
            values[0]]
        compartment = mouse_df[mouse_df['mouse_n']==mouse_n]['plane'].values[0]
        for sess_n in sess_ns:
            print('Mouse ', mouse_n, ', sess ', sess_n)
            # Obtain session object
            sessid = mouse_df[(mouse_df['mouse_n']==mouse_n) & 
                              (mouse_df['sess_n']==sess_n)]['sessid'].values[0]
            sess = session.Session(datadir, sessid, mouse_df=mouse_df_fnm,
                only_matched_rois=only_matched_rois)
            sess.extract_info(fulldict=False, roi=True, run=False, pupil=False) 
            roi_data = []
            # Get expected event data
            expec_data = get_roi_data(sess, stimtype, surp=0, brk_dir=brk_dir,
                scale=scale, remnans=remnans)
            expec_data_mn = np.nanmean(expec_data, axis=1)
            expec_data_std = np.nanstd(expec_data, axis=1)
            expec_mn.append(expec_data_mn)
            expec_std.append(expec_data_std)
            roi_data.append(expec_data)
            # Get unexpected event data
            unexp_data = get_roi_data(sess, stimtype, surp=1, brk_dir=brk_dir,
                scale=scale, remnans=remnans)
            unexp_data_mn = np.nanmean(unexp_data, axis=1)
            unexp_data_std = np.nanstd(unexp_data, axis=1)
            unexp_mn.append(unexp_data_mn)
            unexp_std.append(unexp_data_std)
            roi_data.append(unexp_data)
            print('unexp_data.shape = ', unexp_data.shape)
            # Compute USI
            # Direct computation
            usi.append((unexp_data_mn - expec_data_mn) / 
                        np.sqrt(0.5*(unexp_data_std**2 + expec_data_std**2)))

            # Below is equivalent to above directly computed line (no 
            # permutations needed here). Note: no need for distribution
#             unexp_idxs, _, _ = surp_idx_by_sess(
#                 roi_data, n_perms=0, datatype='roi', op=op, nanpol=None)
#             usi.append(unexp_idxs)

            print('{:.2f} sec'.format(time.time()-t))

        tracked_roi_usi_df.loc[idx] = [mouse_n, layer, compartment,
                                       usi[0], usi[1], usi[2],
                                       expec_mn[0], expec_mn[1], expec_mn[2],
                                       expec_std[0], expec_std[1], expec_std[2],
                                       unexp_mn[0], unexp_mn[1], unexp_mn[2],
                                       unexp_std[0], unexp_std[1], unexp_std[2]]

    return tracked_roi_usi_df

#############################################

def get_roi_data(sess, stimtype, surp, brk_dir, scale, remnans):
    """
    Returns 2-D array of integrated df/f for each ROI and segment.  Segments are 
    frames D-G or U-G (both comprise 0.6 sec each) for Gabors and the 2 sec 
    immediately pre- or proceeding the onset of unexpected flow.
    
    Parameters
    ----------
    sess : session object
        Session object for particular mouse/session
    stimtype : string
        Options: 'gabors', 'bricks'
    brk_dir : string
        Options: 'right', 'left', 'temp', 'nasal' or 'any' (default).  Specify 
        brick direction if desired
    scale : boolean; optional, default = True
        Option to scale each ROI by its statistics for each session.  Rescales 
        through an analog to a z-score, where the mean is replaced by the 
        median, and the standard deviation is replaced by the 5% to 95% 
        interval.  This allows ROIs to be treated more equally to others in a 
        given session, compensating for calcium signal issues, e.g.
    remnans : boolean; optional, default = False
        Option to discount ROIs that have NaN df/f values.  Note, this is 
        redundant if 'only_matched_rois' is set to True.
    
    Returns
    -------
    roi_data : 2-D array of numbers
        Integated df/f for each ROI for each segment (frames D/U-G or 2s 
        immediately before or after onset of unexpected flow). 
        Rows: ROIs.  Columns: Segments (/sequences)
    """
    
    # Warameters:
    # Width for brick stim-locked traces = 2*half_width
    half_width = 2
    # Pre / post times, organized by stimulus and surprise
    pre_dict = {'gabors':[0,0], 'bricks':[half_width,0]}
    post_dict = {'gabors':[0.6,0.6], 'bricks':[0,half_width]}
    pre = pre_dict[stimtype][surp]
    post = post_dict[stimtype][surp]
    # Only get segements around counterflow onset
    surp = 1 if stimtype=='bricks' else surp 
    
    # Obtain ROI data
    stim = sess.get_stim(stimtype)
    seg_ns = \
        stim.get_segs_by_criteria(surp=surp, gabk=16, gabfr=3, 
                                  bri_size=128, bri_dir=brk_dir, by="seg", 
                                  remconsec=(stimtype == "bricks"))
    twop_fr_ns = \
        stim.get_twop_fr_by_seg(seg_ns, first=True, 
                                ch_fl=[pre, post])['first_twop_fr']
    roi_data_df = \
        stim.get_roi_data(twop_fr_ns, pre=pre, post=post, integ=True, 
                          scale=scale, remnans=remnans)
    # Roi x seq (time is already integrated out):
    roi_data = gen_util.reshape_df_data(roi_data_df, squeeze_rows=False, 
        squeeze_cols=True)
    
    return roi_data

#############################################

def surp_idx_by_sess(data, n_perms=1000, datatype='roi', op='diff', 
                     stats='mean', nanpol=None):
    """
    surp_idx_by_sess(data)
    
    Returns session item (ROIs or 1 for running) indices for difference between 
    surprise and regular sequences, as well as their percentiles based on 
    random permutations for each item.

    Required args:
        - data (3D array): data array, structured as 
                               reg, surp [x ROIs] x sequences 

    Optional args:
        - n_perms (int)         : number of permutations for CI estimation
                                  default: 1000
        - datatype (str)        : type of data (e.g., 'roi', 'run')
                                  default: 'roi'
        - op (str)              : operation to use in measuring surprise 
                                  indices ('diff', 'rel_diff', 'discr')
                                  default: 'diff'
        - stats (str)           : statistic used across sequences
                                  default: 'mean'
        - nanpol (str)          : NaN policy ('omit' or None)
                                  default: None

    Returns:
        - item_idxs (1-D array) : item (ROIs or 1 for running) surprise indices 
                                 for the session
        - item_percs (1-D array): item (ROIs or 1 for running) surprise index 
                                 percentiles for the session, based on 
                                 each item's random permutations
        - all_rand (2-D array)  : item (ROIs or 1 for running) indices 
                                 calculated through randomized permutation, 
                                    structured as item x n_perms
    """
    
    # take statistic across sequences, unless the index id discr (D')
    if op != 'discr' and op != 'd-prime':
        seq_mes = np.stack([math_util.mean_med(
            subdata, stats=stats, axis=-1, nanpol=nanpol) 
            for subdata in data])
        axis = None
    else:
        seq_mes = data
        axis = -1

    # calculate index
    item_idxs = math_util.calc_op(seq_mes, op=op, nanpol=nanpol)

    # reshape to add an item/channel dimension if datatype isn't ROIs
    last_dim = np.sum([sub.shape[-1] for sub in data])
    if datatype != 'roi':
        item_idxs = np.asarray(item_idxs).reshape(-1)
        targ = (1, last_dim)
    else:
        targ = (-1, last_dim)

    # get CI
    div = data[0].shape[-1] # length of reg
    # perms (items x perms)
    all_rand = math_util.permute_diff_ratio(
        np.concatenate(data, axis=-1).reshape(targ), div=div, 
        n_perms=n_perms, stats=stats, nanpol=nanpol, op=op)

    item_percs = np.empty(len(item_idxs))
    for r, (item_idx, item_rand) in enumerate(zip(item_idxs, all_rand)):
        item_percs[r] = scist.percentileofscore(
            item_rand, item_idx, kind='mean')
    
    return item_idxs, item_percs, all_rand

#############################################

def make_usi_abs_mean_df(gab_df, brk_df, stimtype_list, n_perm):
    """
    Returns dataframe of the mean |USI| values over ROIs for each session, the 
    p-values for each session pair comparison, and the Bonferroni-corrected 
    significance level of each.
    
    Parameters
    ----------
    gab_df : Pandas DataFrame
        Dataframe of USIs for Gabor stimulus
    brk_df : Pandas DataFrame
        Dataframe of USIs for visual flow (bricks) stimulus
    stimtype_list : list of strings
        Contains stimuli over which to compute values
    n_perm : number
        Number of permutations to use to compute p-values for intersession 
        comparisons
    
    Returns
    -------
    df : Pandas DataFrame
        Dataframe with mean |USI| session values and comparisons
        
        Columns:
            'layer', 'compartment', 'stimtype' :
                layer, compartment, and stimulus type, respectively
            'usi_abs_mn_<session n>' :  
                <|USI|>, where the average is taken over ROIs
            'usi_abs_mn_raw_p__<sess m>_<sess n>' :  
                raw p-value of <|USI|> between sessions m and n
            'usi_sig_for_abs_mn__<sess m>_<sess n>' :  
                (Bonferroni-) corrected significance level for <|USI|> between 
                sessions m and n
    """
    
    # declarations/initializations
    sess_ns = [1,2,3]
    bonf_n = 12
    alpha_001 = 0.001/bonf_n
    alpha_01  = 0.01/bonf_n
    alpha_05  = 0.05/bonf_n
    df = pd.DataFrame(
        columns = ['layer', 'compartment', 'stimtype',
                   'usi_abs_mn_1', 'usi_abs_mn_2', 'usi_abs_mn_3',
                   'usi_abs_mn_raw_p__1_2', 'usi_abs_mn_raw_p__2_3', 
                   'usi_abs_mn_raw_p__1_3',
                   'usi_sig_for_abs_mn__1_2', 'usi_sig_for_abs_mn__2_3',
                   'usi_sig_for_abs_mn__1_3'])
    df_dict = {'gabors':gab_df, 'bricks':brk_df}
    idx = -1

    # Loop through stimulus types, compartments/layers and sessions
    for stimtype in stimtype_list:
        for compartment in ['dend', 'soma']:
            for layer in ['L2/3', 'L5']:
                # declarations/initializations
                usi_df = df_dict[stimtype]
                idx+=1
                print(layer, compartment)
                # Set masks to desired layer/compartment
                mask0 = usi_df['layer']==layer
                mask1 = usi_df['compartment']==compartment
                usis = []
                usi_abs_mn = []
                usi_abs_mn_pval = []
                usi_sig_abs_mn_pval = []
                # Obtain USIs and calculate the mean over ROIs of their absolute 
                #  values for each session
                for sess_n in sess_ns:
                    usis.append(np.hstack(usi_df[(mask0 & mask1)]
                                          ['sess_{}_usi'.format(sess_n)].
                                          values))
                    usi_abs_mn.append(np.nanmean(np.abs(usis[-1])))
                # Compute the p-value for the mean of the absolute values by 
                #  shuffling all session pairs
                usi_abs_mn_pval = compute_usi_abs_mn_pval(usis, usi_abs_mn, 
                    n_perm, metric='abs_mean')
                # Determine the significance levels
                for pval in usi_abs_mn_pval:
                    if pval <= alpha_001:
                        pval_str = '<= 0.001'
                    elif pval <= alpha_01:
                        pval_str = '<= 0.01'
                    elif pval <= alpha_05:
                        pval_str = '<= 0.05'
                    else:
                        pval_str = 'False'
                    usi_sig_abs_mn_pval.append(pval_str)
                df.loc[idx] = [layer, compartment, stimtype,
                               usi_abs_mn[0], usi_abs_mn[1], usi_abs_mn[2],
                               usi_abs_mn_pval[0], usi_abs_mn_pval[1], 
                               usi_abs_mn_pval[2],
                               usi_sig_abs_mn_pval[0], usi_sig_abs_mn_pval[1], 
                               usi_sig_abs_mn_pval[2]]

            
    
    return df

#############################################

def compute_usi_abs_mn_pval(usis, usi_scalar, n_perm, metric='abs_mean'):
    """
    Compute p-values of <|USI|> across sessions by shuffling pairs
    
    Parameters
    ----------
    usis : list of 1-D arrays of numbers
        usis[i] = 1-D array of USIs for session i for all mice in a particular 
        layer/compartment/stimulus type
    usi_scalar : list of numbers
        Specifies the scalar metric for a given session.  Can be, e.g., 
        variance.  See options below in 'metric'.  E.g., usi_scalar[i] can be 
        mean of USIs for session i
    n_perm : number 
        Number of permutations
    metric : string; optional, default = 'abs_mean'
        Options: 'var', 'mean', 'abs_mean'. Scalar metric of USIs used in 
        'usi_scalar', above. Used to determine what scalar metric to use with 
        shuffled USIs.
    
    Returns
    -------
    usi_scalar_pval : list of numbers
        P-values for chosen scalar (var, mean, |mean|).  In order, compares 
        sessions 1&2, 2&3, and 1&3
    """
    
    # declarations/initializations
    diff_distro = []; 
    usi_scalar_pval = []; 
    # Compute differences:
    usi_diff = [usi_scalar[0]-usi_scalar[1],
                usi_scalar[1]-usi_scalar[2],
                usi_scalar[0]-usi_scalar[2]]
    for _ in range(int(n_perm)):
        usi_perm = []
        # Make into array, with each column corresponding to a different session
        usi_arr = np.asarray(usis).transpose() 
        # Vector of which session (column) to take from usi_arr:
        rand_sess = \
            np.random.choice(range(usi_arr.shape[1]), 
                             size=usi_arr.shape[0], replace=True)
        roi_idx = range(usi_arr.shape[0])
        # Use above to select 2 of the 3 random sessions from usi_arr
        usi_perm.append(usi_arr[roi_idx, rand_sess])
        direc = np.random.choice([-1,1], size=usi_arr.shape[0], replace=True)
        usi_perm.append(usi_arr[roi_idx, np.mod(rand_sess + direc, 3)])
        # Compute a differences of metrics from the distributions
        if metric=='var':
            diff_distro.append(np.nanvar(usi_perm[0]) - np.nanvar(usi_perm[1]))
        elif metric=='mean':
            diff_distro.append(np.nanmean(usi_perm[0]) - 
                               np.nanmean(usi_perm[1]))
        elif metric=='abs_mean':
            diff_distro.append(np.nanmean(np.abs(usi_perm[0])) - 
                               np.nanmean(np.abs(usi_perm[1])))
    diff_distro = np.asarray(diff_distro)
    
    for i in range(len(usi_scalar)):
        usi_scalar_pval.append(compute_2_tailed_pval(usi_diff[i], diff_distro))
    
    return usi_scalar_pval

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

def make_usi_abs_frac_chng_df(gab__tracked_roi_usi_df, brk__tracked_roi_usi_df, 
                              n_perm, n_bstrap):
    """
    Returns dataframe containing fractional chng from session 1 to 3 of the 
    <|USIs|> over ROIs for Gabors and visual flow, along with their 
    uncertainties (bootstrapped stdev) and p-value of the difference between the 
    Gabor and visual flow chng for each layer/compartment
    
    Parameters
    ----------
    gab__tracked_roi_usi_df : Pandas DataFrame
        Dataframe containing tracked usi for each 
        mouse/layer/compartment/session for the Gabor stimulus
    brk__tracked_roi_usi_df : Pandas DataFrame
        Dataframe containing tracked usi for each 
        mouse/layer/compartment/session for the bricks stimulus
    n_perm : number
        Number of permutations to perform to compare Gabors against visual flow
    n_bstrap : number
        Number of resamplings with replacement to do
    
    Returns
    -------
    usi_chng_df : Pandas DataFrame
        Dataframe with <|USIs|> (over ROIs) fractional chng from sess 1 to 3, 
        and comparisons across stimuli
        
        Columns:
            'layer', compartment, 'sess_compare' : 
                layer, compartment, and sessions compared, respectively
            'gab_mn_abs_frac_chng' : 
                Gabor fractional chng in <|USIs|> from sess 1 to 3
            'brk_mn_abs_frac_chng' : 
                Bricks fractional chng in <|USIs|> from sess 1 to 3
            'pval_raw' : 
                Raw p-value comparing Gabor and brick fractional changes
            'gab_bstrap_std' : 
                Bootstrapped stdev for 'gab_mn_abs_frac_chng'
            'brk_bstrap_std' : 
                Bootstrapped stdev for 'brk_mn_abs_frac_chng'
    """
    
    # declarations/initializations
    sess_compare = [1,3]
    compartment_list = ['dend', 'soma', 'all']
    layer_list = ['L2/3', 'L5', 'all']
    usi_chng_df = pd.DataFrame(columns=['layer', 'compartment', 'sess_compare', 
                                        'gab_mn_abs_frac_chng', 
                                        'brk_mn_abs_frac_chng',
                                        'pval_raw', 'gab_bstrap_std', 
                                        'brk_bstrap_std'])
    idx = -1
    # Loop through compartments and layers to get data for each
    for layer, compartment in it.product(layer_list, compartment_list):
        if layer == 'all' and compartment != 'all':
            continue
        if layer != 'all' and compartment == 'all':
            continue
        print(layer, compartment)
        idx+=1
        # Obtain fractional changes and ROI pairs across sessions and 
        #  layers/compartments if layer is 'all' for statistical testing
        gab_frac_chng, brk_frac_chng, gab_usi_lay_comp, brk_usi_lay_comp = \
            get_usi_abs_frac_chng(gab__tracked_roi_usi_df, 
                                  brk__tracked_roi_usi_df, 
                                  layer, compartment, sess_compare)
        # Compute p-value
        pval = \
            compute_usi_abs_frac_chng_pval(gab_usi_lay_comp, brk_usi_lay_comp, 
                                           gab_frac_chng, brk_frac_chng, 
                                           n_perm=n_perm)
        # Compute uncertainty
        gab_std, brk_std = \
            compute_usi_abs_frac_chng_err(gab_usi_lay_comp, brk_usi_lay_comp, 
                                          n_bstrap=n_bstrap)        
        # Add data to dataframe
        usi_chng_df.loc[idx] = [layer, compartment, sess_compare,
                                gab_frac_chng, brk_frac_chng, 
                                pval, gab_std, brk_std]    
    
    return usi_chng_df

#############################################

def get_usi_abs_frac_chng(gab__tracked_roi_usi_df, brk__tracked_roi_usi_df,
                          layer, compartment, sess_compare):
    '''
    Get absolute fractional changes of <|USIs|> (over ROIs) from sess 
    sess_compare[0] (usually sess 1) to sess_compare[1] (usually sess 3) for 
    Gabors and visual flow, along with the associated USI pairs for 
    Gabors/bricks for each session in order to shuffle stimulus labels to later 
    obtain p-values when comparing Gabors and visual flow.

    Parameters
    ----------
    gab__tracked_roi_usi_df : Pandas DataFrame
        Dataframe containing tracked usi for each 
        mouse/layer/compartment/session for the Gabor stimulus
    brk__tracked_roi_usi_df : Pandas DataFrame
        Dataframe containing tracked usi for each 
        mouse/layer/compartment/session for the bricks stimulus
    layer : string
        Layer for which to obtain data
    compartment : string
        Compartment for which to obtain data
    sess_compare : 1-D arraylike of numbers
        Sessions for which to obtain data
        
    Returns
    -------
    gab_abs_usi_frac_chng : 1-D list of numbers
        Absolute fractional change from sess_compare[0] to sess_compare[1] in 
        <|USIs|> over ROIs for Gabors. Note, only 1 number in list unless 
        looking at all layers/compartments
    brk_usi_abs_frac_chng : 1-D list of numbers
        Absolute fractional change from sess_compare[0] to sess_compare[1] in 
        <|USIs|> over ROIs for visual flow. Note, only 1 number in list unless 
        looking at all layers/compartments
    gab_usi_lay_comp : 3-D list of numbers
        Outer: Layers/compartments (just 1 entry if layer not 'all').  
        Next: Sessions. Inner: USIs for tracked ROIs
    brk_usi_lay_comp : 3-D list of numbers
        Outer: Layers/compartments (just 1 entry if layer not 'all'). 
        Next: Sessions. Inner: USIs for tracked ROIs
    '''

    # declarations/initializations
    layer_list = ['L2/3', 'L5']
    compartment_list = ['dend', 'soma']
    gab_abs_usi_frac_chng = []
    brk_usi_abs_frac_chng = []
    gab_usi_lay_comp = []
    brk_usi_lay_comp = []

    # Unless we're processing all of the layers/compartments together, no reason 
    #  to loop
    if layer != 'all':
        layer_list = [layer]
        compartment_list = [compartment]
    # Loop over chosen layers/compartments (trivial loops unless looking at all 
    #  layers/compartments)    
    for layer, compartment in it.product(layer_list, compartment_list):
        # declarations/initializations
        gab_usi_pairs = []
        brk_usi_pairs = []
        # Obtain data for each session that we want to compare
        for sess_n in sess_compare:
            # Obtain sub-dataframes limited to chosen layer/compartment
            mask0 = gab__tracked_roi_usi_df['layer']==layer
            mask1 = gab__tracked_roi_usi_df['compartment']==compartment
            gab_df = gab__tracked_roi_usi_df[mask0 & mask1]
            mask0 = brk__tracked_roi_usi_df['layer']==layer
            mask1 = brk__tracked_roi_usi_df['compartment']==compartment
            brk_df = brk__tracked_roi_usi_df[mask0 & mask1]
            # Obtain the USI values for Gabor / bricks for each session
            gab_usi_pairs.append(np.hstack(gab_df['sess_{}_usi'.format(sess_n)].
                                           values))
            brk_usi_pairs.append(np.hstack(brk_df['sess_{}_usi'.format(sess_n)].
                                           values))
        # Amalgamate USIs across layers/compartments (for 'all')
        gab_usi_lay_comp.append(gab_usi_pairs)
        brk_usi_lay_comp.append(brk_usi_pairs)    
        # compute absolute fractional change across sessions for <|USI|> over 
        #  ROIs for Gabors
        mean0 = np.nanmean(np.abs(gab_usi_pairs[0]))
        mean1 = np.nanmean(np.abs(gab_usi_pairs[1]))
        gab_abs_usi_frac_chng.append(np.abs((mean1-mean0)/mean0))
        # compute absolute fractional change across sessions for <|USI|> over 
        #  ROIs for visual flow
        mean0 = np.nanmean(np.abs(brk_usi_pairs[0]))
        mean1 = np.nanmean(np.abs(brk_usi_pairs[1]))
        brk_usi_abs_frac_chng.append(np.abs((mean1-mean0)/mean0))

    return gab_abs_usi_frac_chng, brk_usi_abs_frac_chng, gab_usi_lay_comp, \
           brk_usi_lay_comp

#############################################

def compute_usi_abs_frac_chng_pval(gab_usi_lay_comp, brk_usi_lay_comp,
                                   gab_frac_chng, brk_frac_chng, n_perm):
    """
    Compute p-values for <|USI|> (over ROIs) |fractional changes| across 
    sessions by shuffling stimulus labels
    
    Parameters
    ----------
    gab_usi_lay_comp : 3-D list of numbers
        Outer: Layers/compartments (just 1 entry if layer not 'all').  
        Next: Sessions. Inner: USIs for tracked ROIs
    brk_usi_lay_comp : 3-D list of numbers
        Outer: Layers/compartments (just 1 entry if layer not 'all').  
        Next: Sessions. Inner: USIs for tracked ROIs
    gab_abs_usi_frac_chng : 1-D list of numbers
        Absolute fractional change from sess_compare[0] to sess_compare[1] in 
        <|USIs|> over ROIs for Gabors. Note, only 1 number in list unless 
        looking at all layers/compartments
    brk_usi_abs_frac_chng : 1-D list of numbers
        Absolute fractional change from sess_compare[0] to sess_compare[1] in 
        <|USIs|> over ROIs for visual flow. Note, only 1 number in list unless 
        looking at all layers/compartments
    n_perm : number
        Number of shuffles to perform

    Returns
    -------
    pval : number
        P-value of difference between |fractional changes| across sessions of 
        <|USI|> over ROIs for Gabors vs. bricks
    """
    
    # declarations/initializations
    n_perm = int(n_perm)
    gab_brk_distro = []
    # Permutations loop
    for _ in range(n_perm):
        # declarations/initializations
        gab_perm_pair_mn = []
        gab_frac_chng_perm = []
        brk_perm_pair_mn = []
        brk_frac_chng_perm = []
        # Get USIs for each layer/compartment (just one iteration if layer not 
        #  'all')
        for i in range(len(gab_usi_lay_comp)):
            gab_usi_pair = gab_usi_lay_comp[i]
            brk_usi_pair = brk_usi_lay_comp[i]
            # Shuffle stimulus labels for each session
            for j in range(len(gab_usi_pair)):
                # Arrange USIs as columns, one column per stimulus type
                arr = np.vstack((gab_usi_pair[j],brk_usi_pair[j])).transpose()
                # Index ROIs
                roi_idx = range(arr.shape[0])
                # Randomly pick 1st or 2nd column of array
                col = np.random.choice(range(arr.shape[1]), size=len(roi_idx), 
                                       replace=True)
                # Obtain <|USI|> (over ROIs) with shuffled stimulus labels
                gab_perm_pair_mn.append(np.nanmean(np.abs(arr[roi_idx, col])))
                brk_perm_pair_mn.append(np.nanmean(np.abs(arr[roi_idx, 
                                                          np.mod(col+1,2)])))
            # Compute |fractional change| across session for shuffled <|USI|>
            gab_frac_chng_perm.append(
                np.abs((gab_perm_pair_mn[1]-gab_perm_pair_mn[0]) / 
                       gab_perm_pair_mn[0]))
            brk_frac_chng_perm.append(
                np.abs((brk_perm_pair_mn[1]-brk_perm_pair_mn[0]) / 
                       brk_perm_pair_mn[0]))
        # Add difference of distributions (averaged over layers/compartments if 
        #  layer is 'all') to shuffled distribution
        gab_brk_distro.append(
            np.mean(gab_frac_chng_perm) - np.mean(brk_frac_chng_perm))

    gab_brk_distro = np.asarray(gab_brk_distro)
    # Compute difference and p-value
    diff = np.mean(gab_frac_chng) - np.mean(brk_frac_chng)
    pval = compute_2_tailed_pval(diff, gab_brk_distro)

    return pval

#############################################

def compute_usi_abs_frac_chng_err(gab_usi_lay_comp, brk_usi_lay_comp, 
                              n_bstrap=1e3):

    """
    Compute uncertainty for <|USI|> (over ROIs) |fractional change| 
    
    Parameters
    ----------
    gab_usi_lay_comp : 3-D list of numbers
        Outer: Layers/compartments (just 1 entry if layer not 'all'). 
        Next: Sessions. Inner: USIs for tracked ROIs
    brk_usi_lay_comp : 3-D list of numbers
        Outer: Layers/compartments (just 1 entry if layer not 'all').  
        Next: Sessions. Inner: USIs for tracked ROIs
    n_bstrap : number; optional, default = 1e3
        Number of resamplings with replacement to do
    
    Returns
    -------
    gab_bstrap_std : number
        Standard deviation of distribution of <|USI|> (over ROIs) 
        |fractional change| obtained via resampling with replacement from each 
        session's USIs for Gabors.
    brk_bstrap_std : 
        Standard deviation of distribution of <|USI|> (over ROIs) 
        |fractional change| obtained via resampling with replacement from each 
        session's USIs for Gabors.
    """
    
    # declarations/initializations
    n_bstrap = int(n_bstrap)
    gab_brk_distro = []
    gab_bstrap_pair_mn = []
    brk_bstrap_pair_mn = []
    # Loop over layers/compartments (1 iteration if layer not 'all')
    for i in range(len(gab_usi_lay_comp)):
        # Obtain USI pairs (across sessions) for each stimulus for 
        #  layer/compartment
        gab_usi_pairs = gab_usi_lay_comp[i]
        brk_usi_pairs = brk_usi_lay_comp[i]
        # Loop over sessions
        for j in range(len(gab_usi_pairs)):
            # Sample with replacement
            gab_bstrap = \
                np.random.choice(gab_usi_pairs[j], size=(len(gab_usi_pairs[j]), 
                                 n_bstrap), replace=True)
            brk_bstrap = \
                np.random.choice(brk_usi_pairs[j], size=(len(brk_usi_pairs[j]), 
                                 n_bstrap), replace=True)
            # Compute average over ROIs of |USIs|
            gab_bstrap_pair_mn.append(np.mean(np.abs(gab_bstrap), axis=0))
            brk_bstrap_pair_mn.append(np.mean(np.abs(brk_bstrap), axis=0))
        # Compute |fractional change| of <|USI|>
        gab_frac_chng = \
            np.abs((gab_bstrap_pair_mn[1]-gab_bstrap_pair_mn[0]) / 
                   gab_bstrap_pair_mn[0])
        brk_frac_chng = \
            np.abs((brk_bstrap_pair_mn[1]-brk_bstrap_pair_mn[0]) / 
                   brk_bstrap_pair_mn[0])
    # Uncertainty
    gab_bstrap_std = np.std(gab_frac_chng)
    brk_bstrap_std = np.std(brk_frac_chng)
    
    return gab_bstrap_std, brk_bstrap_std

#############################################

def make_usi_corr_df(tracked_roi_usi_df, usi_corr_df, stimtype, usi_base_sess=1, 
                     n_perm=1e4, n_bstrap=1e4):
    """
    Returns a dataframe that contains, for each compartment and layer, the raw 
    and residual USI correlations (i.e., correlations between usi_{day m} vs. 
    Delta(USI) = usi_{day m+1} - usi_{day m} for m = 1 or 2), the bounds of the 
    standard deviation for the residual correlations and 95% confidence interval 
    for the null distribution of the residual correlations, the raw p-value 
    (multiply by 8 = Bonferroni N for corrected p-value), and the corrected 
    significance level.  
    
    For inspection convenience, the variables used to determine the 
    correlations, the raw shuffled correlation distribution, and the raw 
    bootstrapped distribution are included as well.
    
    Parameters
    ----------
    tracked_roi_usi_df : Pandas DataFrame
        Dataframe with tracked ROI USIs and statistics
    usi_corr_df : Pandas DataFrame
        Dataframe with correlation information of USIs across sessions.  See 
        below under 'Returns' 
    stimtype : string
        Stimulus type ('gabors' or 'bricks')
    usi_base_sess : number; optional, default = 1
        Session to obtain USIs and against which to compare delta(USI) with 
        usi_base_sess+1
    n_perm : number; optional, default = 1e4
        Number of permutations to perform to compare Gabors against visual flow
    n_bstrap : number; optional, default =1e4
        Number of resamplings with replacement to do

    Returns
    -------
    usi_corr_df (pd DataFrame) : dataframe that contains, for each 
        layer/compartment, data on the correlations between the USI in 1 session 
        and change in USI by the next session
        
        Columns:
            stimtype, layer, compartment : 
                stimulus type (gabor / bricks), layer, and compartment, 
                respectively
            usi_base_sess : 
                session USI obtained from difference in USIs taken from this and 
                subsequent sessions
            corr_raw : 
                raw correlation between USI and Delta(USI)
            corr_resid : 
                normalized residual correlation
            corr_resid_low_sd, corr_resid_high_sd : 
                interval of bootstrapped correlation std dev (/err)
            corr_resid_null_low_ci, corr_resid_null_high_ci : 
                interval of 95% CI of null resid corr
            pval_raw : 
                raw p-value of correlation.  Corrected p-value = 8*pval_raw, 
                where 8 = the Bonferroni N
            sig_correc : 
                (bonferroni-) corrected significance level
            usi, delta_usi : 
                USI and Delta(USI).  Correlation of these variables = 
                'corr_raw'.  For inspection if desired
            corr_raw_distro : 
                raw correlation distribution obtained by shuffling session 
                labels.  Used to obtain p-value. For inspection if desired
            bstrap_raw_distro : 
                raw correlation bootstrapped distribution, used to obtain 
                standard error. For inspection if desired
    """

    # declarations/initializations
    bonf_n = 8
    alpha_001 = 0.001/bonf_n
    alpha_01  = 0.01/bonf_n
    alpha_05  = 0.05/bonf_n
    
    idx = usi_corr_df.shape[0]
    layers = {'L23-Cux2':'L2/3', 'L5-Rbp4':'L5'}
    lc = list(it.product(layers.values(), ['dend', 'soma']))

    # Loop through layers and compartments
    for i, (l,c) in enumerate(lc):
        print(c, l)

        # declarations/initializations
        usi                   = []
        corr_raw_distro       = []
        error_corr_raw_distro = []
        mask0 = tracked_roi_usi_df['compartment']==c
        mask1 = tracked_roi_usi_df['layer']==l
        usi.append( np.hstack(tracked_roi_usi_df[mask0 & mask1]['sess_1_usi']) )
        usi.append( np.hstack(tracked_roi_usi_df[mask0 & mask1]['sess_2_usi']) )
        usi.append( np.hstack(tracked_roi_usi_df[mask0 & mask1]['sess_3_usi']) )

        # Find correlation between USI and delta(USI)
        x = usi[usi_base_sess-1]
        x2 = usi[usi_base_sess]
        y = x2 - x
        corr_raw = scist.pearsonr(x, y)[0]

        # Permuation and bootstrap loop
        roi_idx = range(x.size)
        loop_length = int(max(n_perm, n_bstrap))
        for i_perm in range(loop_length):
            # Shuffle session labels for USIs
            if i_perm < n_perm:
                # ROIs x sessions
                arr = np.asarray(usi).transpose()
                # Row 1: roi indexes; row 2: randomly chosen session
                sess_col = np.random.choice([0,1,2], x.size)
                # USIs from randomly chosen session
                usi1_perm = arr[roi_idx, sess_col]
                # USIs from randomly chosen remaining session (periodic boundary 
                #  conditions)
                direction  = np.random.choice([-1,1], x.size)
                usi2_perm = arr[roi_idx,np.mod(sess_col+direction,3)]
                # compute pearson from shuffled USIs and add to shuffled distro
                x_perm = usi1_perm
                y_perm = usi2_perm - usi1_perm
                corr_raw_distro.append(scist.pearsonr(x_perm, y_perm)[0])
            # Bootstrapped standard deviation (~ standard error of correlation)
            if i_perm < n_bstrap:
                # sample from ROI indices with replacement
                samp_idxs = np.random.choice(roi_idx, x.size, replace=True)
                x_bstrap = x[samp_idxs]
                y_bstrap = y[samp_idxs]
                error_corr_raw_distro.append(scist.pearsonr(x_bstrap, y_bstrap)
                                             [0])            
    
        # Compute correlation p-value, normalized value, and standard deviation
        corr_raw_distro = np.asarray(corr_raw_distro)
        error_corr_raw_distro = np.asarray(error_corr_raw_distro)
        pval_raw = compute_2_tailed_pval(corr_raw, corr_raw_distro)
        if pval_raw <= alpha_001:
            pval_correc_str = '<= 0.001'
        elif pval_raw <= alpha_01:
            pval_correc_str = '<= 0.01'
        elif pval_raw <= alpha_05:
            pval_correc_str = '<= 0.05'
        else:
            pval_correc_str = 'False'
        corr_med = np.median(corr_raw_distro)
        corr_resid = corr_raw - corr_med
        sigma = np.nanstd(error_corr_raw_distro)
        # Normalize to be in [-1, 1] by dividing difference by distance between 
        #  median and left (if corr_raw < corr_med) or right 
        #  (if corr_raw > corr_med) bound
        if corr_resid < 0:
            corr_resid /= (corr_med+1)
            sigma     /= (corr_med+1)
        elif corr_resid > 0:
            corr_resid /= (1-corr_med)
            sigma     /= (1-corr_med)
        corr_resid_low_sd  = corr_resid - sigma
        corr_resid_high_sd = corr_resid + sigma
            
        # Compute normalized correlation CI
        corr_resid_low_arg = int(np.round(alpha_05*corr_raw_distro.size))
        corr_resid_low_nom = np.sort(corr_raw_distro)[corr_resid_low_arg]
        corr_resid_high_arg = int(np.round((1-alpha_05)*corr_raw_distro.size))
        corr_resid_high_nom = np.sort(corr_raw_distro)[corr_resid_high_arg]
        corr_resid_null_low_ci = (corr_resid_low_nom - corr_med) / (corr_med+1)
        corr_resid_null_high_ci = (corr_resid_high_nom - corr_med) / \
                                  (1-corr_med)
            
        # Add row to dataframe
        usi_corr_df.loc[idx] = \
            [stimtype, l, c, usi_base_sess, 
            corr_raw, corr_resid, corr_resid_low_sd, corr_resid_high_sd,
            corr_resid_null_low_ci, corr_resid_null_high_ci, 
            pval_raw, pval_correc_str,
            x, y, corr_raw_distro, error_corr_raw_distro]
        idx+=1
        

    return usi_corr_df

#############################################

def usi_corr_permutation(tracked_roi_usi_df, stimtype, l, c, usi_base_sess=1):
    """
    Returns the median of the shuffled distribution of correlations between
    USI and Delta(USI)
    
    Parameters
    ----------
    tracked_roi_usi_df : Pandas DataFrame
        Dataframe with tracked ROI USIs and statistics
    stimtype : string
        Stimulus type ('gabors' or 'bricks')
    l : string
        Layer from which to get data
    c : string
        Compartment from which to get data
    usi_base_sess : number; optional, default = 1
        Session to obtain USIs and against which to compare delta(USI) with 
        usi_base_sess+1

    Returns
    -------
    median_array : 2-D array of numbers
        Array with data with a correlation closest to median of a shuffled
        distro.  Size: ROIs x 2.  1st column: USIs; 2nd column: Delta(USI)
    """

    print(l, c)

    # declarations/initializations
    usi                   = []
    corr_raw_distro       = []
    mask0 = tracked_roi_usi_df['compartment']==c
    mask1 = tracked_roi_usi_df['layer']==l
    usi.append( np.hstack(tracked_roi_usi_df[mask0 & mask1]['sess_1_usi']) )
    usi.append( np.hstack(tracked_roi_usi_df[mask0 & mask1]['sess_2_usi']) )
    usi.append( np.hstack(tracked_roi_usi_df[mask0 & mask1]['sess_3_usi']) )

    # Find correlation between USI and delta(USI)
    x = usi[usi_base_sess-1]
    x2 = usi[usi_base_sess]
    y = x2 - x
    corr_raw = scist.pearsonr(x, y)[0]

    # Permuation loop
    roi_idx = range(x.size)
    for i_perm in range(int(1e4)):
        # Shuffle session labels for USIs
        # ROIs x sessions
        arr = np.asarray(usi).transpose()
        # Row 1: roi indexes; row 2: randomly chosen session
        sess_col = np.random.choice([0,1,2], x.size)
        # USIs from randomly chosen session
        usi1_perm = arr[roi_idx, sess_col]
        # USIs from randomly chosen remaining session (periodic boundary 
        #  conditions)
        direction  = np.random.choice([-1,1], x.size)
        usi2_perm = arr[roi_idx,np.mod(sess_col+direction,3)]
        # compute pearson from shuffled USIs and add to shuffled distro
        x_perm = usi1_perm
        y_perm = usi2_perm - usi1_perm
        corr_raw_distro.append(scist.pearsonr(x_perm, y_perm)[0])

    corr_med = np.median(corr_raw_distro)
    corr_resid = corr_raw - corr_med

    print('Corr raw, median = ', corr_raw, corr_med)

    ###############
    
    # Second run to grab the permutation data with correlation
    #  closest to median from above

    # declarations/initializations
    corr_raw_distro       = []
    # Just make sure it's farther away from median corr than anything possible
    prev_pearson = -2 

    for i_perm in range(int(1e4)):
        # Shuffle session labels for USIs
        # ROIs x sessions
        arr = np.asarray(usi).transpose()
        # Row 1: roi indexes; row 2: randomly chosen session
        sess_col = np.random.choice([0,1,2], x.size)
        # USIs from randomly chosen session
        usi1_perm = arr[roi_idx, sess_col]
        # USIs from randomly chosen remaining session (periodic boundary 
        #  conditions)
        direction  = np.random.choice([-1,1], x.size)
        usi2_perm = arr[roi_idx,np.mod(sess_col+direction,3)]
        # Compute pearson from shuffled USIs and add to shuffled distro
        x_perm = usi1_perm
        y_perm = usi2_perm - usi1_perm
        corr_raw_distro.append(scist.pearsonr(x_perm, y_perm)[0])
        # Find data for median of distro
        if np.abs(corr_med - corr_raw_distro[-1]) < np.abs(corr_med - prev_pearson):
            x_keep_perm = x_perm
            y_keep_perm = y_perm
            prev_pearson = corr_raw_distro[-1]

    # Median data
    median_array = np.stack((x_keep_perm, y_keep_perm), axis=1)

    return median_array