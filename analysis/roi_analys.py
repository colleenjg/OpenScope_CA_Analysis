"""
roi_analysis.py

This script runs ROI trace analyses using a Session object with data generated 
by the AIBS experiments for the Credit Assignment Project.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 2.7.

"""

import copy
import multiprocessing

from joblib import Parallel, delayed
import numpy as np

import ori_analys, quint_analys, signif_grps
from util import file_util, gen_util, math_util
from sess_util import sess_gen_util, sess_ntuple_util, sess_str_util
from plot_fcts import roi_analysis_plots as roi_plots


#############################################
def run_traces_by_qu_surp_sess(sessions, analysis, analyspar, sesspar, 
                               stimpar, quintpar, figpar):
    """
    run_traces_by_qu_surp_sess(sessions, analysis, analyspar, sesspar, 
                               stimpar, quintpar, figpar)

    Retrieves trace statistics by session x surp val x quintile and
    plots traces across ROIs by quintile/surprise with each session in a 
    separate subplot.
    
    Also runs analysis for one quintile (full data).
    
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., 't')
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - quintpar (QuintPar)  : named tuple containing quintile analysis 
                                 parameters
        - figpar (dict)        : dictionary containing figure parameters
    """

    sessstr_pr = sess_str_util.sess_par_str(sesspar.sess_n, stimpar.stimtype, 
                                            sesspar.layer, stimpar.bri_dir, 
                                            stimpar.bri_size, stimpar.gabk,
                                            'print')

    print(('\nAnalysing and plotting surprise vs non surprise ROI traces '
           'by quintile ({}) \n({}).').format(quintpar.n_quints, sessstr_pr))

    # modify quintpar to retain all quintiles
    quintpar_one  = sess_ntuple_util.init_quintpar(1, 0, '', '')
    n_quints      = quintpar.n_quints
    quintpar_mult = sess_ntuple_util.init_quintpar(n_quints, 'all')

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    for quintpar in [quintpar_one, quintpar_mult]:
        print('\n{} quint'.format(quintpar.n_quints))
        # get the stats (all) separating by session, surprise and quintiles    
        trace_info = quint_analys.trace_stats_by_qu_sess(sessions, analyspar, 
                                                  stimpar, quintpar.n_quints, 
                                                  quintpar.qu_idx, 
                                                  byroi=False, bysurp=True)
        extrapar = {'analysis': analysis}

        all_stats = [sessst.tolist() for sessst in trace_info[1]]
        trace_stats = {'x_ran'     : trace_info[0].tolist(),
                       'all_stats' : all_stats,
                       'all_counts': trace_info[2]
                      }

        sess_info = sess_gen_util.get_sess_info(sessions, analyspar.fluor)

        info = {'analyspar'  : analyspar._asdict(),
                'sesspar'    : sesspar._asdict(),
                'stimpar'    : stimpar._asdict(),
                'quintpar'   : quintpar._asdict(),
                'extrapar'   : extrapar,
                'sess_info'  : sess_info,
                'trace_stats': trace_stats
                }

        fulldir, savename = roi_plots.plot_traces_by_qu_surp_sess(figpar=figpar, 
                                                                  **info)
        file_util.saveinfo(info, savename, fulldir, 'json')

      
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
                                        area=True, str_type='print')
    sessstr_pr = sess_str_util.sess_par_str(sesspar.sess_n, stimpar.stimtype,
                                    sesspar.layer, stimpar.bri_dir, 
                                    stimpar.bri_size, stimpar.gabk, 'print')

    print(('\nAnalysing and plotting {} ROI average responses '
           'by quintile ({}). \n{}.').format(opstr_pr, quintpar.n_quints, 
                                             sessstr_pr))
    
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

    print(('\nAnalysing and plotting {} ROI surp vs reg traces by '
           'quintile ({}). \n{}.').format(opstr_pr, quintpar.n_quints, 
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

    print(('\nAnalysing and plotting {} ROI surp vs reg average responses '
           'by quintile ({}). \n{}.').format(opstr_pr, quintpar.n_quints,
                                             sessstr_pr))

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
def run_mag_change(sessions, analysis, seed, analyspar, sesspar, stimpar, 
                   permpar, quintpar, figpar):
    """
    run_mag_change(sessions, analysis, seed, analyspar, sesspar, stimpar, 
                   permpar, quintpar, figpar)

    Calculates and plots the magnitude of change in activity of ROIs between 
    the first and last quintile for non surprise vs surprise sequences.
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)      : list of Session objects
        - analysis (str)       : analysis type (e.g., 'm')
        - seed (int)           : seed value to use. (-1 treated as None) 
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - sesspar (SessPar)    : named tuple containing session parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters
        - permpar (PermPar)    : named tuple containing permutation parameters
        - quintpar (QuintPar)  : named tuple containing quintile analysis 
                                 parameters
        - figpar (dict)        : dictionary containing figure parameters    
    """

    sessstr_pr = sess_str_util.sess_par_str(sesspar.sess_n, stimpar.stimtype,
                                       sesspar.layer, stimpar.bri_dir,
                                       stimpar.bri_size, stimpar.gabk, 'print')

    print(('\nCalculating and plotting the magnitude changes in ROI activity '
           'across quintiles \n({})').format(sessstr_pr))

    
    # get full data: session x surp x quints of interest x [ROI x seq]
    integ_info = quint_analys.trace_stats_by_qu_sess(sessions, analyspar, 
                                stimpar, quintpar.n_quints, quintpar.qu_idx, 
                                bysurp=True, integ=True, ret_arr=True)
    all_counts = integ_info[-2]
    qu_data = integ_info[-1]

    # extract session info
    mouse_ns = [sess.mouse_n for sess in sessions]
    lines    = [sess.line for sess in sessions]

    if analyspar.remnans:
        nanpol = None
    else:
        nanpol = 'omit'

    seed = gen_util.seed_all(seed, 'cpu', print_seed=False)

    mags = quint_analys.qu_mags(qu_data, permpar, mouse_ns, lines, 
                                analyspar.stats, analyspar.error, 
                                nanpol=nanpol, op_qu='diff', op_surp='diff')

    # convert mags items to list
    mags = copy.deepcopy(mags)
    mags['all_counts'] = all_counts
    for key in ['mag_st', 'L2', 'mag_rel_th', 'L2_rel_th']:
        mags[key] = mags[key].tolist()

    sess_info = sess_gen_util.get_sess_info(sessions, analyspar.fluor)
    extrapar  = {'analysis': analysis,
                 'seed'    : seed
                 }

    info = {'analyspar': analyspar._asdict(),
            'sesspar': sesspar._asdict(),
            'stimpar': stimpar._asdict(),
            'extrapar': extrapar,
            'permpar': permpar._asdict(),
            'quintpar': quintpar._asdict(),
            'mags': mags,
            'sess_info': sess_info
            }
    
    fulldir, savename = roi_plots.plot_mag_change(figpar=figpar, **info)

    file_util.saveinfo(info, savename, fulldir, 'json')


#############################################
def run_autocorr(sessions, analysis, analyspar, sesspar, stimpar, autocorrpar, 
                 figpar):
    """
    run_autocorr(sessions, analysis, analyspar, sesspar, stimpar, autocorrpar, 
                 figpar)

    Calculates and plots autocorrelation during stimulus blocks.
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list)          : list of Session objects
        - analysis (str)           : analysis type (e.g., 'a')
        - analyspar (AnalysPar)    : named tuple containing analysis parameters
        - sesspar (SessPar)        : named tuple containing session parameters
        - stimpar (StimPar)        : named tuple containing stimulus parameters
        - autocorrpar (AutocorrPar): named tuple containing autocorrelation 
                                     analysis parameters
        - figpar (dict)            : dictionary containing figure parameters
    """

    sessstr_pr = sess_str_util.sess_par_str(sesspar.sess_n, stimpar.stimtype,
                                       sesspar.layer, stimpar.bri_dir,
                                       stimpar.bri_size, stimpar.gabk, 'print')

    print(('\nAnalysing and plotting ROI autocorrelations ' 
           '({}).').format(sessstr_pr))

    xrans = []
    stats = []
    for sess in sessions:
        stim = sess.get_stim(stimpar.stimtype)
        all_segs = stim.get_segs_by_criteria(bri_dir=stimpar.bri_dir, 
                                             bri_size=stimpar.bri_size, 
                                             gabk=stimpar.gabk, by='block')
        sess_traces = []
        for segs in all_segs:
            if len(segs) == 0:
                continue
            segs = sorted(segs)
            # check that segs are contiguous
            if max(np.diff(segs)) > 1:
                raise NotImplementedError(('Segments used for autocorrelation '
                                           'must be contiguous within blocks.'))
            frame_edges = stim.get_twop_fr_per_seg([min(segs), max(segs)])
            frames = range(min(frame_edges[0]), max(frame_edges[1])+1)
            traces = sess.get_roi_traces(frames, fluor=analyspar.fluor, 
                                         remnans=analyspar.remnans)
            sess_traces.append(traces)
        xran, ac_st = math_util.autocorr_stats(sess_traces, autocorrpar.lag_s, 
                                  sess.twop_fps, byitem=autocorrpar.byitem, 
                                  stats=analyspar.stats, error=analyspar.error)
        xrans.append(xran)
        stats.append(ac_st)

    autocorr_data = {'xrans': [xran.tolist() for xran in xrans],
                     'stats': [stat.tolist() for stat in stats]
                     }

    sess_info = sess_gen_util.get_sess_info(sessions, analyspar.fluor)
    extrapar  = {'analysis': analysis}

    info = {'analyspar'     : analyspar._asdict(),
            'sesspar'       : sesspar._asdict(),
            'stimpar'       : stimpar._asdict(),
            'extrapar'      : extrapar,
            'autocorrpar'   : autocorrpar._asdict(),
            'autocorr_data': autocorr_data,
            'sess_info'     : sess_info
            }

    fulldir, savename = roi_plots.plot_autocorr(figpar=figpar, **info)

    file_util.saveinfo(info, savename, fulldir, 'json')


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
        - quintpar (QuintPar)  : named tuple containing quintile analysis 
                                 parameters
        - figpar (dict)        : dictionary containing figure parameters

    Optional args:
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
    """

    stimstr    = sess_str_util.stim_par_str(stimpar.stimtype, stimpar.bri_dir, 
                                            stimpar.bri_size, stimpar.gabk)
    
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
    [n_segs, roi_me, stats, 
        scale_vals, roi_sort] = [dict(), dict(), dict(), dict(), dict()]
    for o, od in enumerate(oridirs):
        for s, surp in enumerate(surps):
            key = '{}_{}'.format(surp, od)
            me = mes[o][se][s] # me per ROI
            n_segs[key] = counts[o][se][s]
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
               'n_segs'    : n_segs,
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

    extrapar = {'analysis': analysis}

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
    run_tune_curves(sessions, analysis, analyspar, sesspar, stimpar, figpar)

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

    gabfrs = gen_util.list_if_not(stimpar_tc.gabfr)
    if len(gabfrs) == 1:
        gabfrs = gabfrs * 2
    if tcurvpar.grp2 == 'surp':
        surps = [0, 1]
    elif tcurvpar.grp2 in ['reg', 'rand']:
        surps = [0, 0]
    else:
        gen_util.accepted_values_error('tcurvpar.grp2', tcurvpar.grp2, 
                                       ['surp', 'reg', 'rand'])

    seed = gen_util.seed_all(seed, 'cpu', print_seed=False)

    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    for comb_gabs in comb_gabs_all:
        for sess in sessions:
            stim = sess.get_stim(stimpar_tc.stimtype)
            nrois_tot = sess_gen_util.get_nrois(sess.nrois, len(sess.nanrois), 
                                            len(sess.nanrois_dff), 
                                            analyspar.remnans, analyspar.fluor)
            ngabs_tot = stim.n_patches
            if nrois == 'all':
                sess_nrois = nrois_tot
            else:
                sess_nrois = np.min([nrois_tot, nrois])
            
            if ngabs == 'all':
                sess_ngabs = stim.n_patches
            else:
                sess_ngabs = np.min([stim.n_patches, ngabs])                

            tc_data, tc_oris, tc_nseqs = [], [], []
            if tcurvpar.prev: # PREVIOUS ESTIMATION METHOD
                tc_vm_pars, tc_vm_mean, tc_hist_pars = [], [], []
    
            for i, (gf, s) in enumerate(zip(gabfrs, surps)):
                # get segments
                segs = stim.get_segs_by_criteria(gabfr=gf, 
                                                 bri_dir=stimpar_tc.bri_dir, 
                                                 bri_size=stimpar_tc.bri_size, 
                                                 gabk=stimpar_tc.gabk, 
                                                 surp=s, by='seg')
                
                if tcurvpar.grp2 == 'rand' and i == 1:
                    n_segs = len(stim.get_segs_by_criteria(gabfr=gf, 
                                                 bri_dir=stimpar_tc.bri_dir, 
                                                 bri_size=stimpar_tc.bri_size, 
                                                 gabk=stimpar_tc.gabk, 
                                                 surp=1, by='seg'))
                    np.random.shuffle(segs)
                    segs = sorted(segs[: n_segs])
                tc_nseqs.append(len(segs))
                twopfr = stim.get_twop_fr_per_seg(segs, first=True)
                # ROI x seq
                roi_data = stim.get_roi_trace_array(twopfr, stimpar_tc.pre, 
                                  stimpar_tc.post, analyspar.fluor, integ=True, 
                                  remnans=analyspar.remnans)[1][:sess_nrois]
                # gab x seq 
                gab_oris = stim.get_stim_par_by_seg(segs, pos=False, ori=True, 
                                                    size=False)[0].T
                # gab_oris = ori_analys.collapse_dir(gab_oris)

                if comb_gabs:
                    ngabs = 1
                    gab_oris = gab_oris.reshape([1, -1])
                    roi_data = np.tile(roi_data, [1, ngabs_tot])
                    
                tc_oris.append(gab_oris.tolist())
                tc_data.append(roi_data.tolist())

                if tcurvpar.prev: # PREVIOUS ESTIMATION METHOD
                    [gab_tc_oris, gab_tc_data, gab_vm_pars, 
                    gab_vm_mean, gab_hist_pars] = ori_analys.tune_curv_estims(
                                     gab_oris, roi_data, ngabs_tot, sess_nrois, 
                                     sess_ngabs, comb_gabs, parallel=parallel)
                    tc_oris[i] = gab_tc_oris
                    tc_data[i] = gab_tc_data
                    tc_vm_pars.append(gab_vm_pars)
                    tc_vm_mean.append(gab_vm_mean)
                    tc_hist_pars.append(gab_hist_pars)

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
