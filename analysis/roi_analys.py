"""
roi_analys.py

This script runs ROI trace analyses using a Session object with data generated 
by the AIBS experiments for the Credit Assignment Project.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import copy
import multiprocessing

from joblib import Parallel, delayed
import numpy as np

from analysis import ori_analys, quint_analys, signif_grps
from util import file_util, gen_util, math_util
from sess_util import sess_gen_util, sess_ntuple_util, sess_str_util
from plot_fcts import roi_analysis_plots as roi_plots


#############################################
def run_roi_areas_by_grp_qu(sessions, analyspar, sesspar, stimpar, extrapar,
                            permpar, quintpar, roigrppar, roi_grps, figpar, 
                            savedict=True):

    """
    run_roi_areas_by_grp_qu(sessions, analysis, analyspar, sesspar, stimpar, 
                            permpar, quintpar, roigrppar, roi_grps, figpar)

    Plots average integrated surprise, no surprise or difference between 
    surprise and no surprise activity across ROIs per group for each quintiles
    with each session in a separate subplot.

    Returns save directory path and results in roi_grps dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - extrapar (dict)      : dictionary containing additional analysis 
                                 parameters
            ['datatype'] (str): datatype (e.g., 'roi')
        - permpar (PermPar)    : named tuple containing permutation parameters
        - quintpar (QuintPar)  : named tuple containing quintile analysis 
                                 parameters
        - roigrppar (RoiGrpPar): named tuple containing ROI grouping parameters
        - roi_grps (dict)      : dictionary containing ROI grps information:
            ['grp_names'] (list)   : list of names of the ROI groups in roi grp 
                                     lists (order preserved)
            ['all_roi_grps'] (list): nested lists containing ROI numbers 
                                     included in each group, structured as 
                                     follows:
                                         if sets of groups are passed: 
                                             session x set x roi_grp
                                         if one group is passed: 
                                             session x roi_grp
        - figpar (dict)        : dictionary containing figure parameters
    
    Optional args:
        - savedict (bool): if True, dictionaries containing parameters used
                           for analysis are saved

    Returns:
        - fulldir (str)  : final name of the directory in which the figures are 
                           saved 
        - roi_grps (dict): dictionary containing ROI grps information:
            ['grp_names'] (list)   : see above
            ['all_roi_grps'] (list): see above
            ['grp_st'] (array-like): nested list or array of group stats 
                                     (mean/median, error) across ROIs, 
                                     structured as:
                                         session x quintile x grp x stat
            ['grp_ns'] (array-like): nested list of group ns, structured as: 
                                         session x grp
    """

    opstr_pr = sess_str_util.op_par_str(roigrppar.plot_vals, roigrppar.op, 
                                        str_type='print')
    sessstr_pr = sess_str_util.sess_par_str(sesspar.sess_n, stimpar.stimtype,
                                    sesspar.layer, stimpar.bri_dir, 
                                    stimpar.bri_size, stimpar.gabk, 'print')
    datastr = sess_str_util.datatype_par_str(extrapar['datatype'])
    if extrapar['datatype'] != 'roi':
        raise ValueError('Analysis only implemented for roi datatype.')

    print(('\nAnalysing and plotting {} {} average responses '
           'by quintile ({}). \n{}.').format(opstr_pr, datastr, 
                                             quintpar.n_quints, sessstr_pr))
    
    # get full data for qu of interest: session x surp x [seq x ROI]
    integ_info = quint_analys.trace_stats_by_qu_sess(sessions, analyspar, 
                                                  stimpar, quintpar.n_quints, 
                                                  'all', bysurp=True, 
                                                  integ=True)     

    # retrieve only mean/medians per ROI
    all_me = [sess_stats[:, :, 0] for sess_stats in integ_info[1]]

    # get statistics per group and number of ROIs per group
    grp_st, grp_ns = signif_grps.grp_stats(all_me, roi_grps['all_roi_grps'], 
                                           roigrppar.plot_vals, roigrppar.op, 
                                           analyspar.stats, analyspar.error)

    roi_grps = copy.deepcopy(roi_grps)
    roi_grps['grp_st'] = grp_st.tolist()
    roi_grps['grp_ns'] = grp_ns.tolist()

    sess_info = sess_gen_util.get_sess_info(sessions, analyspar.fluor)
    
    info = {'analyspar': analyspar._asdict(),
            'sesspar'  : sesspar._asdict(),
            'stimpar'  : stimpar._asdict(),
            'extrapar' : extrapar,
            'quintpar' : quintpar._asdict(),
            'permpar'  : permpar._asdict(),
            'roigrppar': roigrppar._asdict(),
            'sess_info': sess_info,
            'roi_grps' : roi_grps
            }
    
    # plot
    fulldir, savename = roi_plots.plot_roi_areas_by_grp_qu(figpar=figpar, 
                                                           **info)
    
    if savedict:
        file_util.saveinfo(info, savename, fulldir, 'json')

    return fulldir, roi_grps


#############################################
def run_roi_traces_by_grp(sessions, analyspar, sesspar, stimpar, extrapar, 
                          permpar, quintpar, roigrppar, roi_grps, figpar, 
                          savedict=True):
                           
    """
    run_roi_traces_by_grp(sessions, analysis, sesspar, stimpar, extrapar, 
                          permpar, quintpar, roigrppar, roi_grps, figpar)

    Calculates and plots ROI traces across ROIs by group for surprise, no 
    surprise or difference between surprise and no surprise activity per 
    quintile (first/last) with each group in a separate subplot and each 
    session in a different figure.

    Optionally saves results and parameters relevant to analysis in a 
    dictionary.

    Returns save directory path and results in roi_grps dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - extrapar (dict)      : dictionary containing additional analysis 
                                 parameters
            ['datatype'] (str): datatype (e.g., 'roi')
        - permpar (PermPar)    : named tuple containing permutation parameters
        - quintpar (QuintPar)  : named tuple containing quintile analysis 
                                parameters
        - roigrppar (RoiGrpPar): named tuple containing ROI grouping parameters
        - roi_grps (dict) : dictionary containing ROI grps information:
            ['grp_names'] (list)   : list of names of the ROI groups in roi grp 
                                     lists (order preserved)
            ['all_roi_grps'] (list): nested lists containing ROI numbers 
                                     included in each group, structured as 
                                     follows:
                                         if sets of groups are passed: 
                                             session x set x roi_grp
                                         if one group is passed: 
                                             session x roi_grp
        - figpar (dict)        : dictionary containing figure parameters
        
    Optional args:
        - savedict (bool): if True, dictionaries containing parameters used
                           for analysis are saved

    Returns:
        - fulldir (str)  : final name of the directory in which the figures are 
                           saved 
        - roi_grps (dict): dictionary containing ROI grps information:
            ['grp_names'] (list)        : see above
            ['all_roi_grps'] (list)     : see above
            ['xran'] (array-like)       : array or list of time values for the
                                          frame chunks
            ['trace_stats'] (array-like): array or nested list of statistics of
                                          ROI groups for quintiles of interest
                                          structured as:
                                              sess x qu x ROI grp x stats 
                                              x frame
    """

    opstr_pr = sess_str_util.op_par_str(roigrppar.plot_vals, 
                                        roigrppar.op, str_type='print')
    sessstr_pr = sess_str_util.sess_par_str(sesspar.sess_n, stimpar.stimtype,
                                       sesspar.layer, stimpar.bri_dir,
                                       stimpar.bri_size, stimpar.gabk, 'print')

    opstr = sess_str_util.op_par_str(roigrppar.plot_vals, roigrppar.op)
    sessstr = sess_str_util.sess_par_str(sesspar.sess_n, stimpar.stimtype,
                                         sesspar.layer, stimpar.bri_dir,
                                         stimpar.bri_size, stimpar.gabk, 'file')
    datastr = sess_str_util.datatype_par_str(extrapar['datatype'])
    if extrapar['datatype'] != 'roi':
        raise ValueError('Analysis only implemented for roi datatype.')

    print(('\nAnalysing and plotting {} {} surp vs reg traces by '
           'quintile ({}). \n{}.').format(opstr_pr, datastr, quintpar.n_quints, 
                                          sessstr_pr))

    # get sess x surp x quint x stats x ROIs x frames
    trace_info = quint_analys.trace_stats_by_qu_sess(sessions, analyspar, 
                                           stimpar, n_quints=quintpar.n_quints, 
                                           qu_idx=quintpar.qu_idx, byroi=True, 
                                           bysurp=True)
    xran = trace_info[0]

    # retain mean/median from trace stats
    trace_me = [sessst[:, :, 0] for sessst in trace_info[1]]

    grp_stats = signif_grps.grp_traces_by_qu_surp_sess(trace_me, analyspar, 
                                           roigrppar, roi_grps['all_roi_grps'])

    roi_grps = copy.deepcopy(roi_grps)
    roi_grps['xran'] = xran.tolist()
    roi_grps['trace_stats'] = grp_stats.tolist()

    sess_info = sess_gen_util.get_sess_info(sessions, analyspar.fluor)

    info = {'analyspar'  : analyspar._asdict(),
            'sesspar'    : sesspar._asdict(),
            'stimpar'    : stimpar._asdict(),
            'extrapar'   : extrapar,
            'permpar'    : permpar._asdict(),
            'quintpar'   : quintpar._asdict(),
            'roigrppar'  : roigrppar._asdict(),
            'sess_info'  : sess_info,
            'roi_grps'   : roi_grps
            }

    fulldir = roi_plots.plot_roi_traces_by_grp(figpar=figpar, **info)

    if savedict:
        infoname = ('roi_tr_{}_grps_{}_{}quint_'
                    '{}tail').format(sessstr, opstr, quintpar.n_quints, 
                                     permpar.tails)

        file_util.saveinfo(info, infoname, fulldir, 'json')

    return fulldir, roi_grps


#############################################
def run_roi_areas_by_grp(sessions, analyspar, sesspar, stimpar, extrapar,  
                         permpar, quintpar, roigrppar, roi_grps, figpar, 
                         savedict=False):
    """
    run_roi_areas_by_grp(sessions, analyspar, sesspar, stimpar, extrapar, 
                         permpar, quintpar, roigrppar, roi_grps, fig_par)

    Calculates and plots ROI traces across ROIs by group for surprise, no 
    surprise or difference between surprise and no surprise activity per 
    quintile (first/last) with each group in a separate subplot and each 
    session in a different figure. 

    Optionally saves results and parameters relevant to analysis in a 
    dictionary.

    Returns save directory path and results in roi_grps dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - extrapar (dict)      : dictionary containing additional analysis 
                                 parameters
            ['datatype'] (str): datatype (e.g., 'roi')
        - permpar (PermPar)    : named tuple containing permutation parameters
        - quintpar (QuintPar)  : named tuple containing quintile analysis 
                                 parameters
        - roigrppar (RoiGrpPar): named tuple containing ROI grouping parameters
        - roi_grps (dict)      : dictionary containing ROI grps information:
            ['all_roi_grps'] (list): nested lists containing ROI numbers 
                                     included in each group, structured as 
                                     follows:
                                         if sets of groups are passed: 
                                             session x set x roi_grp
                                         if one group is passed: 
                                             session x roi_grp
            ['grp_names'] (list)   : list of names of the ROI groups in ROI 
                                     grp lists (order preserved)
        - figpar (dict)        : dictionary containing figure parameters
        
    Optional args:
        - savedict (bool): if True, dictionaries containing parameters used
                           for analysis are saved

    Returns:
        - fulldir (str)  : final name of the directory in which the figures are 
                           saved 
        - roi_grps (dict): dictionary containing ROI groups:
            ['all_roi_grps'] (list)          : see above
            ['grp_names'] (list)             : see above
            ['area_stats'] (array-like)      : ROI group stats (mean/median,      
                                               error) for quintiles of interest,
                                               structured as:
                                                 session x quintile x grp x 
                                                 stat
            ['area_stats_scale'] (array-like): same as 'area_stats', but with 
                                              last quintile scaled relative to 
                                              first
    """
    
    opstr_pr = sess_str_util.op_par_str(roigrppar.plot_vals, 
                                        roigrppar.op, str_type='print')
    sessstr_pr = sess_str_util.sess_par_str(sesspar.sess_n, stimpar.stimtype,
                                       sesspar.layer, stimpar.bri_dir,
                                       stimpar.bri_size, stimpar.gabk, 'print')

    opstr = sess_str_util.op_par_str(roigrppar.plot_vals, roigrppar.op)
    sessstr = sess_str_util.sess_par_str(sesspar.sess_n, stimpar.stimtype,
                                         sesspar.layer, stimpar.bri_dir,
                                         stimpar.bri_size, stimpar.gabk)
    datastr = sess_str_util.datatype_par_str(extrapar['datatype'])
    if extrapar['datatype'] != 'roi':
        raise ValueError('Analysis only implemented for roi datatype.')

    print(('\nAnalysing and plotting {} {} surp vs reg average responses '
           'by quintile ({}). \n{}.').format(opstr_pr, datastr, 
                                             quintpar.n_quints, sessstr_pr))

    # get full data for qu of interest: session x surp x [seq x ROI]
    integ_info = quint_analys.trace_stats_by_qu_sess(sessions, analyspar, 
                                                  stimpar, quintpar.n_quints, 
                                                  quintpar.qu_idx, bysurp=True, 
                                                  integ=True)     

    # retrieve only mean/medians per ROI
    all_me = [sess_stats[:, :, 0] for sess_stats in integ_info[1]]

    roi_grps = copy.deepcopy(roi_grps)
    # get statistics per group and number of ROIs per group
    for scale in [False, True]:
        scale_str = sess_str_util.scale_par_str(scale)
        # sess x quint x grp x stat
        grp_st, _ = signif_grps.grp_stats(all_me, roi_grps['all_roi_grps'], 
                                        roigrppar.plot_vals, roigrppar.op, 
                                        analyspar.stats, analyspar.error, scale)
        roi_grps['area_stats{}'.format(scale_str)] = grp_st.tolist()

    sess_info = sess_gen_util.get_sess_info(sessions, analyspar.fluor)

    info = {'analyspar': analyspar._asdict(),
            'sesspar'  : sesspar._asdict(),
            'stimpar'  : stimpar._asdict(),
            'extrapar' : extrapar,
            'permpar'  : permpar._asdict(),
            'quintpar' : quintpar._asdict(),
            'roigrppar': roigrppar._asdict(),
            'sess_info': sess_info,
            'roi_grps' : roi_grps
            }
        
    fulldir = roi_plots.plot_roi_areas_by_grp(figpar=figpar, **info)

    if savedict:
        infoname = ('roi_area_{}_grps_{}_{}quint_'
                       '{}tail').format(sessstr, opstr, quintpar.n_quints, 
                                        permpar.tails)
        file_util.saveinfo(info, infoname, fulldir, 'json')
    
    return fulldir, roi_grps


#############################################
def run_rois_by_grp(sessions, analysis, seed, analyspar, sesspar, stimpar, 
                    permpar, quintpar, roigrppar, figpar):
    """
    run_rois_by_grp(sessions, analysis, seed, analyspar, sesspar, stimpar, 
                    permpar, quintpar, roigrppar, figpar)

    Identifies ROIs showing significant surprise in first and/or last quintile,
    group accordingly and plots traces and areas across ROIs for surprise, 
    no surprise or difference between surprise and no surprise activity per 
    quintile (first/last) with each group in a separate subplot and each 
    session in a different figure. 
    
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., 'ch')
        - seed (int)           : seed value to use. (-1 treated as None) 
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - permpar (PermPar)    : named tuple containing permutation parameters
        - quintpar (QuintPar)  : named tuple containing quintile analysis 
                                 parameters
        - roigrppar (RoiGrpPar): named tuple containing ROI grouping parameters
        - figpar (dict)        : dictionary containing figure parameters
    """

    datatype = 'roi'

    opstr = sess_str_util.op_par_str(roigrppar.plot_vals, roigrppar.op)
    sessstr = sess_str_util.sess_par_str(sesspar.sess_n, stimpar.stimtype,
                                         sesspar.layer, stimpar.bri_dir,
                                         stimpar.bri_size, stimpar.gabk)

    sessids = [sess.sessid for sess in sessions]

    # get full data for qu of interest: session x surp x [seq x ROI]
    integ_info = quint_analys.trace_stats_by_qu_sess(sessions, analyspar, 
                                                  stimpar, quintpar.n_quints, 
                                                  quintpar.qu_idx, bysurp=True, 
                                                  integ=True, ret_arr=True)
    _, _, _, qu_data = integ_info     

    if analyspar.remnans:
        nanpol = None
    else:
        nanpol = 'omit'

    seed = gen_util.seed_all(seed, 'cpu', print_seed=False)

    # identify significant ROIs
    [all_roi_grps, grp_names, 
        permpar_mult] = signif_grps.signif_rois_by_grp_sess(sessids, 
                                          qu_data, permpar, roigrppar, 
                                          quintpar.qu_lab,
                                          stats=analyspar.stats, nanpol=nanpol)
    
    roi_grps = {'grp_names'   : grp_names,
                'all_roi_grps': all_roi_grps,
                'permpar_mult': permpar_mult._asdict(),
               }
    
    extrapar  = {'analysis': analysis,
                 'datatype': datatype,
                 'seed'    : seed
                 }

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    _, roi_grps_q = run_roi_areas_by_grp_qu(sessions, analyspar, sesspar, 
                                            stimpar, extrapar, permpar, 
                                            quintpar, roigrppar, roi_grps, 
                                            figpar, savedict=False)    

    _, roi_grps_t = run_roi_traces_by_grp(sessions, analyspar, sesspar, 
                                   stimpar, extrapar, permpar, quintpar, 
                                   roigrppar, roi_grps, figpar, savedict=False)

    fulldir, roi_grps_a = run_roi_areas_by_grp(sessions, analyspar, sesspar, 
                                               stimpar, extrapar, permpar, 
                                               quintpar, roigrppar, roi_grps,
                                               figpar, savedict=False)

    # add roi_grps_t and roi_grps_a keys to roi_grps dictionary
    for roi_grps_dict in [roi_grps_q, roi_grps_t, roi_grps_a]:
        for key in roi_grps_dict.keys():
            if key not in roi_grps:
                roi_grps[key] = roi_grps_dict[key]

    sess_info = sess_gen_util.get_sess_info(sessions, analyspar.fluor)

    info = {'analyspar': analyspar._asdict(),
            'sesspar'  : sesspar._asdict(),
            'stimpar'  : stimpar._asdict(),
            'extrapar' : extrapar,
            'permpar'  : permpar._asdict(),
            'quintpar' : quintpar._asdict(),
            'roigrppar': roigrppar._asdict(),
            'sess_info': sess_info,
            'roi_grps' : roi_grps
            }
     
    infoname = ('roi_{}_grps_{}_{}q_'
                    '{}tail').format(sessstr, opstr, quintpar.n_quints, 
                                     permpar.tails)

    file_util.saveinfo(info, infoname, fulldir, 'json')


#############################################
def run_oridirs_by_qu_sess(se, sess, oridirs, surps, xran, mes, counts, 
                           analyspar, sesspar, stimpar, extrapar, quintpar, 
                           figpar, parallel=False):
    """

    run_oridirs_by_qu_sess(se, sess, oridirs, surps, xran, mes, counts, 
                           analyspar, sesspar, stimpar, extrapar, quintpar, 
                           figpar)

    Plots average activity across gabor orientations or brick directions 
    per ROI as colormaps, and across ROIs as traces for a single session and
    specified quintile.
    Saves results and parameters relevant to analysis in a dictionary. 

    Required args:
        - se (int)             : session index in roi_me
        - sess (Session)       : Session object
        - oridirs (list)       : list of orientations/directions
        - surps (list)         : list of surprise values
        - xran (1D array)      : time values for the 2p frames
        - mes (nested list)    : ROI mean/median data, structured as:
                                    oridirs x session x surp x ROI x frames
        - counts (nested list) : number of sequences, structured as:
                                    oridirs x session x surp
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - extrapar (dict)      : dictionary containing additional analysis 
                                 parameters
            ['analysis'] (str): analysis type (e.g., 'o')
            ['datatype'] (str): datatype (e.g., 'roi')
        - quintpar (QuintPar)  : named tuple containing quintile analysis 
                                 parameters
        - figpar (dict)        : dictionary containing figure parameters

    Optional args:
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
    """

    stimstr    = sess_str_util.stim_par_str(stimpar.stimtype, stimpar.bri_dir, 
                                            stimpar.bri_size, stimpar.gabk)
    if extrapar['datatype'] != 'roi':
        raise ValueError('Analysis only implemented for roi datatype.')

    qu_str, qu_str_pr = quintpar.qu_lab[0], quintpar.qu_lab[0]
    if qu_str != '':
        qu_str_pr = ', {}'.format(qu_str_pr.capitalize())
        qu_str    = '_{}'.format(qu_str)

    if not analyspar.remnans:
        nanpol = 'omit'
    else:
        nanpol = None

    print(('    Mouse {}, sess {}, {}, {}{}').format(sess.mouse_n, sess.sess_n, 
                                    sess.line, sess.layer, qu_str_pr))
    [n_seqs, roi_me, stats, 
        scale_vals, roi_sort] = [dict(), dict(), dict(), dict(), dict()]
    for o, od in enumerate(oridirs):
        for s, surp in enumerate(surps):
            key = '{}_{}'.format(surp, od)
            me = mes[o][se][s] # me per ROI
            n_seqs[key] = counts[o][se][s]
            # sorting idx
            roi_sort[key] = np.argsort(np.argmax(me, axis=1)).tolist()
            scale_vals['{}_max'.format(key)] = np.max(me, 
                                                        axis=1).tolist()
            scale_vals['{}_min'.format(key)] = np.min(me, 
                                                        axis=1).tolist()
            roi_me[key] = me.tolist()
            # stats across ROIs
            stats[key]  = math_util.get_stats(me, analyspar.stats, 
                                analyspar.error, 0, nanpol).tolist()

    tr_data = {'xran'      : xran.tolist(),
               'n_seqs'    : n_seqs,
               'roi_me'    : roi_me,
               'stats'     : stats,
               'scale_vals': scale_vals,
               'roi_sort'  : roi_sort
               }

    sess_info = sess_gen_util.get_sess_info(sess, analyspar.fluor)

    info = {'analyspar': analyspar._asdict(),
            'sesspar'  : sesspar._asdict(),
            'stimpar'  : stimpar._asdict(),
            'extrapar' : extrapar,
            'quintpar' : quintpar._asdict(),
            'tr_data'  : tr_data,
            'sess_info': sess_info
            }
    
    roi_plots.plot_oridir_colormaps(figpar=figpar, parallel=parallel, 
                                    **info)

    fulldir = roi_plots.plot_oridir_traces(figpar=figpar, **info)

    savename = ('roi_cm_tr_m{}_'
                'sess{}{}_{}_{}').format(sess.mouse_n, sess.sess_n, 
                                            qu_str, stimstr, sess.layer)

    file_util.saveinfo(info, savename, fulldir, 'json')


#############################################
def run_oridirs_by_qu(sessions, oridirs, surps, analyspar, sesspar, stimpar, 
                      extrapar, quintpar, figpar, parallel=False):
    """
    run_oridirs_by_qu(sessions, oridirs, surps, analyspar, sesspar, stimpar,
                      extrapar, quintpar, figpar)

    Plots average activity across gabor orientations or brick directions 
    per ROI as colormaps, and across ROIs as traces for a specified quintile.
    Saves results and parameters relevant to analysis in a dictionary. 

    Required args:
        - sessions (list)      : list of Session objects
        - oridirs (list)       : list of orientations/directions
        - surps (list)         : list of surprise values
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - extrapar (dict)      : dictionary containing additional analysis 
                                 parameters
            ['analysis'] (str): analysis type (e.g., 'o')
            ['datatype'] (str): datatype (e.g., 'roi')
        - quintpar (QuintPar)  : named tuple containing quintile analysis 
                                 parameters
        - figpar (dict)        : dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
    """

    # for each orientation/direction
    xran, mes, counts = [], [], []
    for od in oridirs:
        # create a specific stimpar for each direction or orientation
        stimpar_dict = stimpar._asdict()
        if stimpar.stimtype == 'bricks':
            stimpar_dict['bri_dir'] = od
        elif stimpar.stimtype == 'gabors':
            stimpar_dict['gab_ori'] = od
        stimpar_od = sess_ntuple_util.init_stimpar(**stimpar_dict)
        # NaN stats if no segments fit criteria
        nan_empty = True
        trace_info = quint_analys.trace_stats_by_qu_sess(sessions, 
                                        analyspar, stimpar_od, 
                                        quintpar.n_quints, quintpar.qu_idx,
                                        byroi=True, bysurp=True,
                                        nan_empty=nan_empty)
        xran = trace_info[0]
        # retrieve mean/medians and single quintile data:
        # sess x [surp x ROIs x frames]
        mes.append([sess_stats[:, 0, 0] for sess_stats in trace_info[1]]) 
        # retrieve single quintile counts: sess x surp
        counts.append([[surp_c[0] for surp_c in sess_c] 
                                    for sess_c in trace_info[2]])
    
    if parallel:
        n_jobs = min(multiprocessing.cpu_count(), len(sessions))
        Parallel(n_jobs=n_jobs)(delayed(run_oridirs_by_qu_sess)
                (se, sess, oridirs, surps, xran, mes, counts, analyspar, 
                 sesspar, stimpar, extrapar, quintpar, figpar, False) 
                 for se, sess in enumerate(sessions))
    else:
        for se, sess in enumerate(sessions):
            run_oridirs_by_qu_sess(se, sess, oridirs, surps, xran, mes, counts, 
                                   analyspar, sesspar, stimpar, extrapar, 
                                   quintpar, figpar, parallel)


#############################################
def run_oridirs(sessions, analysis, analyspar, sesspar, stimpar, quintpar, 
                figpar, parallel=False):
    """
    run_oridirs(sessions, analysis, analyspar, sesspar, stimpar, quintpar, 
                figpar)

    Plots average activity across gabor orientations or brick directions 
    per ROI as colormaps, and across ROIs as traces. 
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., 'a')
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - quintpar (QuintPar)  : named tuple containing quintile analysis 
                                 parameters
        - figpar (dict)        : dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
    """
   
    datatype = 'roi'

    # update stim parameters parameters
    stimpar_dict = stimpar._asdict()
    if stimpar.stimtype == 'bricks':
        # update stimpar with both brick directions
        # replace quintpar with quintpar split in 2 (for each direction)
        stimpar_dict['bri_dir'] = ['right', 'left']
        oridirs = stimpar_dict['bri_dir']
    elif stimpar.stimtype == 'gabors':
        # update stimpar with gab_fr = 0 and all gabor orientations
        stimpar_dict['gabfr'] = 0
        stimpar_dict['gab_ori'] = [0, 45, 90, 135]
        oridirs = stimpar_dict['gab_ori'] 
    stimpar = sess_ntuple_util.init_stimpar(**stimpar_dict)

    sessstr_pr = sess_str_util.sess_par_str(sesspar.sess_n, stimpar.stimtype,
                                       sesspar.layer, stimpar.bri_dir, 
                                       stimpar.bri_size, stimpar.gabk, 'print')

    # split quintiles apart and add a quint=1
    quintpar_one  = sess_ntuple_util.init_quintpar(1, 0, '', '')
    quintpars = [quintpar_one]
    for qu_idx, qu_lab, qu_lab_pr in zip(quintpar.qu_idx, quintpar.qu_lab, 
                                         quintpar.qu_lab_pr):
        qp = sess_ntuple_util.init_quintpar(quintpar.n_quints, qu_idx, qu_lab, 
                                            qu_lab_pr)
        quintpars.append(qp)

    print(('\nAnalysing and plotting colormaps and '
           'traces ({}).').format(sessstr_pr))

    extrapar = {'analysis': analysis,
                'datatype': datatype,
                }

    surps = ['reg', 'surp']  
    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    if parallel and len(quintpars) > len(sessions):
        n_jobs = min(multiprocessing.cpu_count(), len(quintpars))
        Parallel(n_jobs=n_jobs)(delayed(run_oridirs_by_qu)
                (sessions, oridirs, surps, analyspar, sesspar, stimpar, 
                 extrapar, quintpar, figpar, False) 
                 for quintpar in quintpars)
    else:
        for quintpar in quintpars:
            run_oridirs_by_qu(sessions, oridirs, surps, analyspar, sesspar, 
                              stimpar, extrapar, quintpar, figpar, parallel)


#############################################
def run_tune_curves(sessions, analysis, seed, analyspar, sesspar, stimpar, 
                    tcurvpar, figpar, parallel=False, plot_tc=True):
    """
    run_tune_curves(sessions, analysis, seed, analyspar, sesspar, stimpar, 
                    tcurvpar, figpar)

    Calculates and plots ROI orientation tuning curves, as well as a 
    correlation plot for regular vs surprise orientation preferences. 

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., 'c')
        - seed (int)           : seed to use
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - tcurvpar (TCurvPar)  : named tuple containing tuning curve 
                                 parameters
        - figpar (dict)        : dictionary containing figure parameters

    Optional args:
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores 
        - plot_tc (bool) : if True, tuning curves are plotted for each ROI 
                           (causes errors on the clusters...)  
    """

    datatype = 'roi'

    if stimpar.stimtype == 'bricks':
        print('Tuning curve analysis not implemented for bricks.')
        return
    
    sessstr_pr = sess_str_util.sess_par_str(sesspar.sess_n, stimpar.stimtype,
                                       sesspar.layer, stimpar.bri_dir, 
                                       stimpar.bri_size, stimpar.gabk, 'print')
    
    print(('\nAnalysing and plotting ROI tuning curves for orientations '
           '({}).').format(sessstr_pr))

    nrois = 'all'
    ngabs = 'all'
    comb_gabs_all = [True, False]
    # small values for testing
    if tcurvpar.test:
        nrois = 8
        ngabs = 'all'
        comb_gabs_all = [True]
    
    print('Number ROIs: {}\nNumber of gabors: {}'.format(nrois, ngabs))

    # modify parameters
    stimpar_tc_dict = stimpar._asdict()
    stimpar_tc_dict['gabfr'] = tcurvpar.gabfr
    stimpar_tc_dict['pre'] = tcurvpar.pre
    stimpar_tc_dict['post'] = tcurvpar.post
    stimpar_tc = sess_ntuple_util.init_stimpar(**stimpar_tc_dict)

    seed = gen_util.seed_all(seed, 'cpu', print_seed=False)

    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    for comb_gabs in comb_gabs_all:
        for sess in sessions:
            returns = ori_analys.calc_tune_curvs(sess, analyspar, stimpar_tc, 
                                nrois, ngabs, tcurvpar.grp2, comb_gabs, 
                                tcurvpar.prev, collapse=True, parallel=parallel)
            if tcurvpar.prev:
                [tc_oris, tc_data, tc_nseqs, tc_vm_pars, 
                                tc_vm_mean, tc_hist_pars] = returns
            else:
                tc_oris, tc_data, tc_nseqs = returns

            tcurv_data = {'oris'     : tc_oris,
                          'data'     : [list(data) for data in zip(*tc_data)],
                          'nseqs'    : tc_nseqs,
                          }

            if tcurvpar.prev: # PREVIOUS ESTIMATION METHOD
                tcurv_data['vm_pars'] = np.transpose(np.asarray(tc_vm_pars), 
                                                     [1, 0, 2, 3]).tolist()
                tcurv_data['vm_mean'] = np.transpose(np.asarray(tc_vm_mean), 
                                                     [1, 0, 2]).tolist()
                tcurv_data['hist_pars'] = np.transpose(np.asarray(tc_hist_pars), 
                                                       [1, 0, 2, 3]).tolist()
                tcurv_data['vm_regr'] = ori_analys.ori_pref_regr(
                                                tcurv_data['vm_mean']).tolist()

            extrapar = {'analysis': analysis,
                        'datatype': datatype,
                        'seed': seed,
                        'comb_gabs': comb_gabs,
                        }

            sess_info = sess_gen_util.get_sess_info(sess, analyspar.fluor)

            info = {'analyspar' : analyspar._asdict(),
                    'sesspar'   : sesspar._asdict(),
                    'stimpar'   : stimpar_tc._asdict(),
                    'extrapar'  : extrapar,
                    'tcurvpar'  : tcurvpar._asdict(),
                    'tcurv_data': tcurv_data,
                    'sess_info' : sess_info
                    }

            fulldir, savename = roi_plots.plot_tune_curves(figpar=figpar, 
                                   parallel=parallel, plot_tc=plot_tc, **info)

            file_util.saveinfo(info, savename, fulldir, 'json')


#############################################
def posori_resp(sess, analyspar, stimpar, nrois='all'):
    """
    posori_resp(sess, analyspar, stimpar)

    Calculates integrated fluorescence levels for ROI locations and 
    orientations.

    Required args:
        - sess (Session): Session object
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters

    Optional args:
        - nrois (int): number of ROIs to include in analysis
                       default: 'all'

    Returns:
        - oris (list)     : stimulus mean orientations
        - roi_stats (list): ROI statistics, structured as 
                                mean orientation x gaborframe x stats x ROI
        - nseqs (list)    : number of sequences structured as 
                                mean orientation x gaborframe
    """

    stim = sess.get_stim(stimpar.stimtype)
    oris = stim.oris
    nrois_tot = sess_gen_util.get_nrois(sess.nrois, len(sess.nanrois), 
                                        len(sess.nanrois_dff), 
                                        analyspar.remnans, analyspar.fluor)
    if nrois == 'all':
        sess_nrois = nrois_tot
    else:
        sess_nrois = np.min([nrois_tot, nrois])

    roi_stats = []
    nseqs = []
    
    for ori in oris:
        ori_stats = []
        ori_nseqs = []
        for gf in range(5):
            if gf == 3:
                s = 0
            elif gf == 4:
                s = 1
                gf = 3
            else:
                s = 'any'
            # get segments
            segs = stim.get_segs_by_criteria(gabfr=gf, 
                                             bri_dir=stimpar.bri_dir, 
                                             bri_size=stimpar.bri_size, 
                                             gab_ori=ori, gabk=stimpar.gabk, 
                                             surp=s, by='seg')
            ori_nseqs.append(len(segs))
            twopfr = stim.get_twop_fr_by_seg(segs, first=True)
            # stats x ROI
            gf_stats = stim.get_roi_trace_stats(twopfr, stimpar.pre, 
                            stimpar.post, byroi=True, 
                            fluor=analyspar.fluor, integ=True, 
                            remnans=analyspar.remnans, 
                            stats=analyspar.stats, 
                            error=analyspar.error)[1][:, :sess_nrois]
            ori_stats.append(gf_stats.tolist())
        roi_stats.append(ori_stats)
        nseqs.append(ori_nseqs)
    
    return oris, roi_stats, nseqs 


#############################################
def run_posori_resp(sessions, analysis, analyspar, sesspar, stimpar, figpar, 
                    parallel=False):
    """
    run_posori_resp(sessions, analysis, analyspar, sesspar, stimpar, figpar)

    Calculates and plots integrated fluorescence levels for ROI positions and 
    mean orientations.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., 'c')
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - figpar (dict)        : dictionary containing figure parameters

    Optional args:
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores 
    """

    datatype = 'roi'
    
    if stimpar.stimtype == 'bricks':
        print('Location preference analysis not implemented for bricks.')
        return
    
    sessstr_pr = sess_str_util.sess_par_str(sesspar.sess_n, stimpar.stimtype,
                                       sesspar.layer, stimpar.bri_dir, 
                                       stimpar.bri_size, stimpar.gabk, 'print')
    datastr = sess_str_util.datatype_par_str(datatype)
    
    print(('\nAnalysing and plotting {} location preferences '
           '({}).').format(datastr, sessstr_pr))

    nrois = 'all'
    nrois = 8
    
    print('Number ROIs: {}'.format(nrois))

    # modify parameters
    stimpar_loc_dict = stimpar._asdict()
    stimpar_loc_dict['pre'] = 0
    stimpar_loc_dict['post'] = 0.45
    stimpar_loc = sess_ntuple_util.init_stimpar(**stimpar_loc_dict)

    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    for sess in sessions:
        oris, roi_stats, nseqs = posori_resp(sess, analyspar, stimpar_loc, 
                                             nrois)
        posori_data = {'oris'     : oris,
                       'roi_stats': roi_stats,
                       'nseqs'    : nseqs
                      }

        extrapar = {'analysis': analysis,
                   'datatype' : datatype,
                   }

        sess_info = sess_gen_util.get_sess_info(sess, analyspar.fluor)

        info = {'analyspar'   : analyspar._asdict(),
                'sesspar'     : sesspar._asdict(),
                'stimpar'     : stimpar_loc._asdict(),
                'extrapar'    : extrapar,
                'posori_data': posori_data,
                'sess_info'   : sess_info
                }

        fulldir, savename = roi_plots.plot_posori_resp(figpar=figpar, **info)

        file_util.saveinfo(info, savename, fulldir, 'json')

