import os

import numpy as np
import pickle
import scipy.stats
import pandas as pd
from matplotlib import pyplot as plt
import pdb

from analysis import session


#############################################
def get_sess_per_mouse(mouse_df, sessid='any', depth='any', 
                       pass_fail='any', all_files='any', any_files='any',
                       overall_sess=1, omit_sess=[], omit_mice=[]):
    """
    Returns list of session IDs and (up to 1 per mouse) of included mice 
    based on parameters.

    Required arguments:
        - mouse_df (pandas df): dataframe containing parameters for each session.
        
    Optional arguments:
        - sessid (int or list)     : session id value(s) of interest
                                     (default: 'any')
        - depth (int or list)      : depths value(s) of interest
                                     (20, 75, 175, 375)
                                     (default: 'any')
        - pass_fail (str or list)  : pass/fail values to pick from 
                                     ('P', 'F')
                                     (default: 'any')
        - all_files (int or list)  : all_files values to pick from (0, 1)
                                     (default: 'any')
        - any_files (int or list)  : any_files values to pick from (0, 1)
                                     (default: 'any')
        - overall_sess (int or str): overall_sess value to aim for
                                     (1, 2, 3, ... or 'last')
                                     (default: 1)
        - sess_omit (list)         : sessions to omit
                                     (default: [])
        - mice_omit (list)         : mice to omit
                                     (default: [])
    
    Returns:
        - sesses_n (list): sessions to analyse (1 per mouse)
        - mice_n (list)  : mice included
    """
    
    sesses_n = []
    mice_n = []

    if sessid == 'any':
        sessid = mouse_df['sessionid'].unique().tolist()
    elif not isinstance(sessid, list):
        sessid = [sessid]
    elif not isinstance(omit_sess, list):
        omit_sess = [omit_sess]
    for i in omit_sess:
        if i in sessid:
            sessid.remove(i)
    if depth == 'any':
        depth = mouse_df['depth'].unique().tolist()
    elif not isinstance(depth, list):
        depth = [depth]
    if pass_fail == 'any':
        pass_fail = mouse_df['pass_fail'].unique().tolist()
    elif not isinstance(pass_fail, list):
        pass_fail = [pass_fail]
    if all_files == 'any':
        all_files = mouse_df['all_files'].unique().tolist()
    elif not isinstance(all_files, list):
        all_files = [all_files]
    if any_files == 'any':
        any_files = mouse_df['any_files'].unique().tolist()
    elif not isinstance(any_files, list):
        any_files = [any_files]

    # get session numbers for each mouse based on criteria 
    for i in range(n_mice):
        if i+1 in omit_mice:
            continue
        sessions = mouse_df.loc[(mouse_df['mouseid'] == i+1) & 
                                (mouse_df['sessionid'].isin(sessid)) &
                                (mouse_df['depth'].isin(depth)) &
                                (mouse_df['pass_fail'].isin(pass_fail)) &
                                (mouse_df['all_files'].isin(all_files)) &
                                (mouse_df['any_files'].isin(any_files))]['overall_sess_n'].tolist()

        if len(sessions) == 0:
            continue
        elif overall_sess == 'last':
            sess_n = sessions[-1]
        else:
            if len(sessions) < overall_sess:
                continue
            else:
                sess_n = sessions[overall_sess-1]
        sess = mouse_df.loc[(mouse_df['mouseid'] == i+1) & 
                            (mouse_df['sessionid'].isin(sessid)) &
                            (mouse_df['depth'].isin(depth)) &
                            (mouse_df['pass_fail'].isin(pass_fail)) &
                            (mouse_df['all_files'].isin(all_files)) &
                            (mouse_df['any_files'].isin(any_files)) &
                            (mouse_df['overall_sess_n'] == sess_n)]['sessionid'].tolist()[0]
        sesses_n.append(sess)
        mice_n.append(i+1)
    
    return sesses_n, mice_n


#############################################
def plot_chunks(ax, chunk_val, stats, title='', lw=1.5, hbars=None, bars=None, 
                labels=None, xpos=None, t_hei=0.65, col=['k', 'gray']):
    ax.plot(chunk_val[0], chunk_val[1], lw=lw, color=col[0])
    if stats == 'mean':
        ax.fill_between(chunk_val[0], chunk_val[1] - chunk_val[2], 
                        chunk_val[1] + chunk_val[2], 
                        facecolor=col[1], alpha=0.5)
    elif stats == 'median':
        ax.fill_between(chunk_val[0], chunk_val[2][0], chunk_val[2][1], 
                        facecolor=col[1], alpha=0.5)
    else:
        raise ValueError(('\'plot_stat\' value \'{}\' not '
                          'recognized.'.format(plot_stat)))

    thickness = [2, 1]
    torem = []
    for i, j in enumerate([hbars, bars]):
        if j is not None:
            if not isinstance(j, list):
                j = [j]
            if i == 0:
                torem = j
            if i == 1:
                for r in torem:
                    if r in j:
                        j.remove(r)
            for k in j:
                ax.axvline(x=k, ls='dashed', c='k', lw='{}'
                           .format(thickness[i]), alpha=0.5)  
    if labels is not None and xpos is not None:
        if len(labels) != len(xpos):
            raise IOError(('Arguments \'labels\' and \'xpos\' must be of the '
                           'same length.'))
        ymin, ymax = ax.get_ylim()
        ypos = (ymax-ymin)*t_hei+ymin
        for l, x in zip(labels, xpos):
            ax.text(x, ypos, l, horizontalalignment='center', fontsize=15, 
                    color=col[0])
    ax.set_title(title)
    ax.set_ylabel('dF/F')
    ax.set_xlabel('Time (s)')


if __name__ == "__main__":

    # set the main data directory (this needs to be changed by each user)
    maindir = '/media/colleen/LaCie/CredAssign/pilot_data'
    mouse_df_dir = 'mouse_df.pkl'
    figdir_roi = 'figures/prelim_roi'

    # specific parameters
    sess_order = 1 # 1 for first, etc. or 'last'
    depth = 'hi' # 'hi' for dendrites or 'lo' for somata
    gab_k = [16] # kappa value(s) to use (either [4], [16] or [4, 16])
    quintiles = 4 # number of quintiles to divide stimulus into
    n_perms = 1000 # n of permutations for permutation analysis
    threshold = 95 # threshold for permutation analysis

    # general parameters
    gab_fr    = 3 # gabor frame to retrieve
    pre       = 0 # sec before frame
    post      = 1.5 # sec before frame
    byroi     = True # run analysis by ROI (vs averaged across)
    remnans   = True # remove ROIs containing NaNs or Infs
    dfoverf   = True # use dfoverf instead of raw ROI traces
    plot_stat = 'mean' # plot mean or median
    
    # fairly fixed parameters
    n_mice = 4
    omit_sess = [721038464] # alignment didn't work

    # create directory if it doesn't exist
    if not os.path.exists(figdir_roi):
        os.makedirs(figdir_roi)

    omit_mice = []
    if 4 not in gab_k:
        omit_mice = [1] # mouse 1 only got K=4
    elif 16 not in gab_k:
        omit_mice = [3] # mouse 1 only got K=16

    try:
        mouse_df = pd.read_pickle(mouse_df_dir)
    except:
        raise IOError('{} not found.'.format(mouse_df_dir))
    
    if len(gab_k) > 1:
        gab_k_str = ''
    else:
        gab_k_str = gab_k[0]

    # get session numbers
    if depth == 'lo':
        sesses_n, mice_n = get_sess_per_mouse(mouse_df, depth=[175, 375], 
                                              pass_fail='P', all_files=1, 
                                              overall_sess=sess_order, 
                                              omit_sess=omit_sess, 
                                              omit_mice=omit_mice)
        cell_area = 'somata'
    elif depth == 'hi':
        sesses_n, mice_n = get_sess_per_mouse(mouse_df, depth=[20, 75], 
                                              pass_fail='P', all_files=1, 
                                              overall_sess=sess_order, 
                                              omit_sess=omit_sess, 
                                              omit_mice=omit_mice)
        cell_area = 'dendrites'
    else:
        raise ValueError('Value \'{}\' for depth not recognized.'
                        .format(depth))

    # create a dictionary with Session objects prepared for analysis
    sessions = []
    for sess_n in sesses_n:
        print('\nCreating session {}...'.format(sess_n))
        sess = session.Session(maindir, sess_n) # creates a session object to work with
        sess.extract_info()                     # extracts necessary info for analysis
        print('Finished session {}.'.format(sess_n))
        sessions.append(sess)

    # print session information
    print('\nAnalysing gab{} {} ROIs from session {}.'.format(gab_k_str, 
                                                          cell_area, sess_order))

    # get ROI traces
    gab_no_surp_chunks_all = [] # ROI stats for no surp (sess x quintile (x ROI, if byroi=True) x frame)
    if byroi:
        gab_no_surp_chunks_me = [] # ROI mean/medians for no surp (sess x quintile (x ROI, if byroi=True) x frame)
    n_gab_no_surp = [] # nbr of segs (no surp, gab_fr 3) (sess x quintile)

    gab_surp_chunks_all = [] # ROI stats for surp (sess x quintile (x ROI, if byroi=True) x frame)
    if byroi:
        gab_surp_chunks_me = [] # ROI mean/medians for surp (sess x quintile (x ROI, if byroi=True) x frame)
    n_gab_surp = [] # nbr of segs (surp, gab_fr 3) (sess x quintile)

    twop_fps = [] # 2p fps by session
    for i, sess in enumerate(sessions):
        print('\n{}'.format(sess.session))
        # get the min seg number for stimulus
        seg_min = min(sess.gabors.get_segs_by_criteria(stimPar2=gab_k, by='seg'))
        # get the max seg number for stimulus
        seg_max = max(sess.gabors.get_segs_by_criteria(stimPar2=gab_k, by='seg'))+1
        # calculate number of quintiles in each segment (ok if not round number)
        quints = (seg_max - seg_min)/quintiles
        
        # no surp
        print('Getting non surprise ROI traces.')
        # retrieve non surprise seg numbers
        no_surp_seg = sess.gabors.get_segs_by_criteria(surp=0, gaborframe=gab_fr, 
                                                       stimPar2=gab_k, by='seg')
        gab_no_surp_count = [] 
        no_surp_qu_all = [] 
        no_surp_qu_me = []
        for j in range(quintiles):
            # get no surp seg numbers for the current quintile
            quint_no_surp = [seg for seg in no_surp_seg if (seg >= j*quints+seg_min and 
                                                            seg < (j+1)*quints+seg_min)]
            # store number of no surp segs for each quintile
            gab_no_surp_count.extend([len(quint_no_surp)])
            # get the stats for ROI traces for these segs 
            # returns [x_ran, mean/median, std/quartiles] for each ROI or across ROIs
            chunk_stats = sess.gabors.get_roi_chunk_stats(sess.gabors
                            .get_2pframes_by_seg(quint_no_surp, first=True), 
                            pre, post, byroi=byroi, dfoverf=dfoverf, 
                            remnans=remnans, rand=False, stats=plot_stat)
            # store by quintile
            no_surp_qu_all.append(chunk_stats)
            # retrieve the mean/median for each ROI and store
            if byroi:
                chunk_stats_me = []
                for i in chunk_stats:
                    chunk_stats_me.append(i[1])
                # store by quintile
                no_surp_qu_me.append(chunk_stats_me)
        # store by session
        gab_no_surp_chunks_all.append(no_surp_qu_all)
        gab_no_surp_chunks_me.append(np.asarray(no_surp_qu_me))
        n_gab_no_surp.append(gab_no_surp_count)
        
        # start surp
        print('Getting surprise ROI traces.')
        # retrieve surprise seg numbers
        surp_seg = sess.gabors.get_segs_by_criteria(surp=1, gaborframe=gab_fr, 
                                                    stimPar2=gab_k, by='seg')
        gab_surp_count = []
        surp_qu_all = []
        surp_qu_me = []
        for j in range(quintiles):
            # get surp seg numbers for the current quintile
            quint_surp = [seg for seg in surp_seg if (seg >= j*quints+seg_min and 
                                                    seg < (j+1)*quints+seg_min)]
            # store number of surp segs for each quintile
            gab_surp_count.extend([len(quint_surp)])
            # get the stats for ROI traces for these segs 
            # returns [x_ran, mean/median, std/quartiles] for each ROI or across ROIs
            chunk_stats = sess.gabors.get_roi_chunk_stats(sess.gabors
                        .get_2pframes_by_seg(quint_surp, first=True), 
                        pre, post, byroi=byroi, dfoverf=dfoverf, 
                        remnans=remnans, rand=False, stats=plot_stat)
            # store by quintile
            surp_qu_all.append(chunk_stats)
            # retrieve the mean/median for each ROI and store
            chunk_stats_me = []
            if byroi:
                for i in chunk_stats:
                    chunk_stats_me.append(i[1])
                # store by quintile
                surp_qu_me.append(chunk_stats_me)
        # store by session
        gab_surp_chunks_all.append(surp_qu_all)
        gab_surp_chunks_me.append(np.asarray(surp_qu_me))
        n_gab_surp.append(gab_surp_count)
        
        # store the 2p fps by session
        twop_fps.extend([sess.twop_fps])

    # If analysing average ROI traces (byroi=False), plot average traces per 
    # quintile
    if not byroi:
        # draw lines
        # light at each segment (+1 for gray and +1 for end)
        seg_bars = [0.3, 0.6, 0.9, 1.2]
        xpos = [0.15, 0.45, 0.75, 1.05, 1.35]

        # Non surprise (blue)
        labels_no_surp = ['D', 'gray', 'A', 'B', 'C']
        col_no_surp = [['midnightblue', 'steelblue'], 
                       ['darkblue', 'dodgerblue'], ['mediumblue', 'skyblue'], 
                       ['blue', 'lightskyblue']]

        labels_surp = ['E', 'gray', 'A', 'B', 'C']

        # Surprise (orange)
        col_surp = [['chocolate', 'burlywood'], ['peru', 'wheat'], 
                    ['darkorange', 'navajowhite'], ['orange', 'papayawhip']]

        fig_gab_surp_nosurp, ax_gab_surp_nosurp = plt.subplots(ncols=2, nrows=2, 
                                                            figsize=(15, 15))
        for i in range(len(mice_n)):
            leg = []
            for j in range(quintiles):
                # non surprise
                if j != quintiles-1:
                    plot_chunks(ax_gab_surp_nosurp[i/2][i%2], 
                                gab_no_surp_chunks_all[i][j], stats=plot_stat,
                                col=col_no_surp[j])
                else:
                    plot_chunks(ax_gab_surp_nosurp[i/2][i%2], 
                                gab_no_surp_chunks_all[i][j], stats=plot_stat,
                                title=('Mouse {} - {} dF/F across gabor ' 
                                    'sequences \n(session {}, {})')
                                    .format(mice_n[i], plot_stat, sess_order, 
                                    cell_area), 
                                labels=labels_no_surp, xpos=xpos, t_hei=0.8, 
                                col=col_no_surp[j])
                leg.extend(['{}-nosurp'.format(j+1)])
            # surprise
            for j in range(quintiles):
                if j != quintiles-1:
                    plot_chunks(ax_gab_surp_nosurp[i/2][i%2], gab_surp_chunks_all[i][j], 
                                stats=plot_stat, col=col_surp[j])
                else:
                    plot_chunks(ax_gab_surp_nosurp[i/2][i%2], gab_surp_chunks_all[i][j], 
                                stats=plot_stat, title=('Mouse {} - {} dF/F '
                                    'across gabor sequences \n (session {}, '
                                    '{})')
                                .format(mice_n[i], plot_stat, sess_order, 
                                cell_area), bars=seg_bars, lw=0.8, 
                                labels=labels_surp, xpos=xpos, t_hei=0.95, 
                                col=col_surp[j])
                leg.extend(['{}-surp'.format(j+1)])
            ax_gab_surp_nosurp[i/2][i%2].legend(leg)
        fig_gab_surp_nosurp.savefig('{}/roi_session_{}_gab{}_{}_surp_nosurp_{}quint.png'
                                    .format(figdir_roi, sess_order, gab_k_str,
                                    cell_area, quintiles), bbox_inches='tight')

    # if analysing by roi (byroi=True), plot average ROI area under dF/F by
    # group and quintile
    else:
        print('\n')
        # remove ROIs with NaNs or Infs in average signal
        if remnans:
            rem_rois = [] # ROI numbers removed for each session (mouse)
            orig_roi_n = [] # original nbrs of ROIs retained for each session (mouse)
            for i in range(len(mice_n)):
                n_rois = len(gab_surp_chunks_me[i][0])
                temp = [] # ROIs with infs or NaNs
                temp2 = np.arange(0,n_rois) # original ROI numbers
                for j in range(n_rois):
                    # identify ROIs with NaNs or Infs in surprise or no surprise data
                    if (sum(sum(np.isnan(gab_surp_chunks_me[i][:, j]))) > 0 or 
                        sum(sum(np.isinf(gab_surp_chunks_me[i][:, j]))) > 0 or
                        sum(sum(np.isnan(gab_no_surp_chunks_me[i][:, j]))) > 0 or 
                        sum(sum(np.isinf(gab_no_surp_chunks_me[i][:, j]))) > 0):
                            temp.extend([j])
                gab_surp_chunks_me[i] = np.delete(gab_surp_chunks_me[i], temp, 1)
                gab_no_surp_chunks_me[i] = np.delete(gab_no_surp_chunks_me[i], temp, 1)
                temp2 = np.delete(temp2, temp) # remove nbrs of ROIs removed
                print('Mouse {}: Removing {}/{} ROIs: {}'
                    .format(mice_n[i], len(temp), n_rois, ', '.join(map(str, temp))))
                # store
                rem_rois.append(temp) # store
                orig_roi_n.append(temp2) # store

        # Integrate for each chunk per quintile per surp/non-surp (sum*fps)
        gab_diff_area = [] # difference in integrated dF/F (sess x quartile x ROI)

        for i in range(len(mice_n)):
            # get area under the curve
            temp_surp = np.sum(gab_surp_chunks_me[i], 2)*1./twop_fps[i]
            temp_no_surp = np.sum(gab_no_surp_chunks_me[i], 2)*1./twop_fps[i]
            diff_surp = temp_surp - temp_no_surp
            # print diff for each ROI
            # print('\nMouse {}, ROI diff 1st quint: {} \n({})'.format(i+1, 
            #       np.mean(diff_surp[0]), ', '.join('{:.3f}'.format(x) 
            #                                        for x in diff_surp[0])))
            gab_diff_area.append(diff_surp)

        # Run permutation test for first and last quintiles
        # (gab_diff_area: mouse x qu x ROI area diff)
        print('\nRunning permutation test')
        
        rois_sign_first = []
        rois_sign_last = []
        for i, sess in enumerate(sessions):
            # recalculates quintiles (like above) and gets all segs (surp or not)
            seg_min = min(sess.gabors.get_segs_by_criteria(stimPar2=gab_k, by='seg'))
            seg_max = max(sess.gabors.get_segs_by_criteria(stimPar2=gab_k, by='seg'))+1
            quints = (seg_max - seg_min)/quintiles
            # get all segs (surprise or not)
            all_seg = sess.gabors.get_segs_by_criteria(gaborframe=gab_fr, 
                                                       stimPar2=gab_k, by='seg')
            print('\nMouse {}'.format(mice_n[i]))
            for t, j in enumerate([0, quintiles-1]):
                # retrieve segs for the current quintile
                quint = [seg for seg in all_seg if (seg >= j*quints+seg_min and 
                                                    seg < (j+1)*quints+seg_min)]
                fr = sess.gabors.get_2pframes_by_seg(quint, first=True)
                ran_fr = [np.around(x*sess.twop_fps) for x in [-pre, post]]

                # get corresponding roi subblocks [[start:end]]
                fr_ind = ([range(x + int(ran_fr[0]), x + int(ran_fr[1])) 
                        for x in fr])

                # remove arrays with negatives or values above total number of 
                # stim frames
                neg_ind = np.where(np.asarray(zip(*fr_ind)[0])<0)[0].tolist()
                over_ind = (np.where(np.asarray(zip(*fr_ind)[-1]) >= 
                                                sess.tot_2p_frames)[0].tolist())
                k=0
                for r, ind in enumerate(neg_ind):
                    fr_ind.pop(ind-r) # compensates for prev popped indices
                    k=r+1
                for r, ind in enumerate(over_ind):
                    fr_ind.pop(ind-r-i) # compensates for prev popped indices

                # get dF/F for each segment and each ROI
                roi_data = sess.get_roi_segments(fr_ind, dfoverf=dfoverf)
                # get area under the curve
                roi_data_integ = np.sum(roi_data, axis=1)*1./sess.twop_fps
                # remove ROIs with nans or infs
                if remnans:
                    roi_data_integ = np.delete(roi_data_integ, rem_rois[i], 0)

                # create permutation indices
                perms_seg = np.argsort(np.random.rand(roi_data_integ.shape[1], 
                                                      n_perms), axis=0)[np.newaxis, :, :]
                dim_roi = np.arange(roi_data_integ.shape[0])[:, np.newaxis, np.newaxis]
               # generate permutation array
                permed_roi = np.stack(roi_data_integ[dim_roi, perms_seg])
                # calculate surp - no surp (roi x permutation)
                diffs = (np.mean(permed_roi[:, 0:n_gab_surp[i][j]], axis=1) - 
                         np.mean(permed_roi[:, n_gab_surp[i][j]:], axis=1))
                # calculate threshold difference for each ROI
                threshs = np.percentile(diffs, threshold, axis=1)
                # for first quartile, identify ROIs with diff > threshold
                if t == 0:
                    rois_sign_first.append(np.where(gab_diff_area[i][j] > 
                                                    threshs)[0])
                    print('first quintile: ROIs:{} \n\t\tdiffs: [{}]'.format(
                        rois_sign_first[i], (' '.join('{:.2f}'.format(x) 
                        for x in gab_diff_area[i][j, rois_sign_first[i]]))))

                elif t == 1:
                    rois_sign_last.append(np.where(gab_diff_area[i][j] > 
                                                  threshs)[0])
                    print('last quintile: ROIs:{} \n\t\tdiffs: [{}]'.format(
                        rois_sign_last[i], (' '.join('{:.2f}'.format(x) 
                        for x in gab_diff_area[i][j, rois_sign_last[i]]))))

        # get ROI numbers for each group
        surp_surp = []
        surp_nosurp = []
        nosurp_surp = []
        nosurp_nosurp = []
        for i in range(len(mice_n)):
            n_rois = gab_diff_area[i].shape[1]
            all_rois = range(n_rois)
            surp_surp.append(list(set(rois_sign_first[i]) & set(rois_sign_last[i])))
            surp_nosurp.append(list(set(rois_sign_first[i]) - set(rois_sign_last[i])))
            nosurp_surp.append(list(set(rois_sign_last[i]) - set(rois_sign_first[i])))
            nosurp_nosurp.append(list(set(all_rois) - set(surp_surp[i]) - 
                                    set(surp_nosurp[i]) - set(nosurp_surp[i])))

        # get stats for each group
        rois = [surp_surp, surp_nosurp, nosurp_surp, nosurp_nosurp]
        leg = ['surp_surp', 'surp_nosurp', 'nosurp_surp', 'nosurp_nosurp']
        roi_stats = []
        # roi_stats will have same structure as rois: 
        # groupe (e.g., surp_surp) x mouse/session x [mean/median, std/mad] x quartile
        for i in range(len(rois)):
            roi_stats.append([])
        for i in range(len(mice_n)):
            # by roi group (e.g., surp_surp)
            for j, roi in enumerate(rois):
                me = []
                dev = []
                for q in range(quintiles):
                    if plot_stat == 'mean':
                        me.append(np.mean(gab_diff_area[i][q, roi[i]]))
                        dev.append(scipy.stats.sem(gab_diff_area[i][q, roi[i]]))
                    elif plot_stat == 'median':
                        me.append(np.median(gab_diff_area[i][q, roi[i]]))
                        if len (roi[i]) != 0:
                            dev.append([np.percentile(gab_diff_area[i][q, roi[i]], 25),
                                    np.percentile(gab_diff_area[i][q, roi[i]], 75)])
                        else:
                            dev.append([np.nan, np.nan])
                    else:
                        raise ValueError(('\'plot_stat\' value \'{}\' not '
                                        'recognized.'.format(plot_stat)))
                roi_stats[j].append([me, dev])

        # plot 
        x_ran = [x+1 for x in range(quintiles)]
        fig_gab_surp_nosurp_qu, ax_gab_surp_nosurp_qu = plt.subplots(ncols=2, nrows=2, 
                                                                    figsize=(15, 15))
        for i in range(len(mice_n)):
            act_leg = []

            ax = ax_gab_surp_nosurp_qu[i/2][i%2]
            for j in range(len(roi_stats)):
                if plot_stat == 'mean':
                    ax.errorbar(x_ran, roi_stats[j][i][0], yerr=roi_stats[j][i][1], 
                                fmt='-o', capsize=4, capthick=2)
                elif plot_stat == 'median':
                    medians = np.asarray(roi_stats[j][i][0])
                    errs = np.asarray(roi_stats[j][i][1])
                    yerr1 = medians - errs[:,0]
                    yerr2 = errs[:,1] - medians
                    ax.errorbar(x_ran, medians, yerr=[yerr1, yerr2],
                                fmt='-o', capsize=4, capthick=2)
                n_rois = len(rois[j][i])
                act_leg.append('{} ({})'.format(leg[j], n_rois))

            ax.set_title(('Mouse {} - {} difference in dF/F \n for surprise vs '
                        'non surprise gab{} sequences \n (session {}, {})')
                        .format(mice_n[i], plot_stat, gab_k_str, sess_order,
                        cell_area))
            ax.set_xticks(x_ran)
            ax.set_ylabel('dF/F')
            ax.set_xlabel('Quintiles')
            ax.legend(act_leg)
        
        fig_gab_surp_nosurp_qu.savefig('{}/roi_session_{}_gab{}_{}_diff_surp_nosurp_{}quint.png'
                                    .format(figdir_roi, sess_order, gab_k_str,
                                    cell_area, quintiles), bbox_inches='tight')