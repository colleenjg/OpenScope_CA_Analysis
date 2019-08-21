"""
roi_analysis_plots.py

This script contains functions to plot results of ROI analyses on specific
sessions (run_roi_analysis.py) from dictionaries.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import copy
import multiprocessing
import os

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st

from sess_util import sess_gen_util, sess_plot_util, sess_str_util
from util import file_util, gen_util, plot_util
from plot_fcts import gen_analysis_plots as gen_plots


#############################################
def plot_from_dict(dict_path, parallel=False, plt_bkend=None, fontdir=None,
                   plot_tc=True):
    """
    plot_from_dict(info_path, args)

    Plots data from dictionaries containing analysis parameters and results.

    Required args:
        - dict_path (str): path to dictionary to plot data from
    
    Optional_args:
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
                           default: False
        - plt_bkend (str): mpl backend to use for plotting (e.g., 'agg')
                           default: None
        - plot_tc (bool) : if True, tuning curves are plotted for each ROI 
    """

    figpar = sess_plot_util.init_figpar(plt_bkend=plt_bkend, fontdir=fontdir)
    plot_util.manage_mpl(cmap=False, **figpar['mng'])

    info = file_util.loadfile(dict_path)
    savedir = os.path.dirname(dict_path)

    analysis = info['extrapar']['analysis']

    # 0. Plots the full traces for each session
    if analysis == 'f': # full traces
        gen_plots.plot_full_traces(figpar=figpar, savedir=savedir, **info)

    # 1. Plot average traces by quintile x surprise for each session 
    if analysis == 't': # traces
        gen_plots.plot_traces_by_qu_surp_sess(figpar=figpar, savedir=savedir, 
                                              **info)

    # 2. Plot average traces by quintile, locked to surprise for each session 
    elif analysis == 'l': # surprise locked traces
        gen_plots.plot_traces_by_qu_lock_sess(figpar=figpar, savedir=savedir, 
                                              **info)

    # 3. Plot magnitude of change in dF/F area from first to last quintile of 
    # surprise vs no surprise sequences, for each session
    elif analysis == 'm': # mag
        gen_plots.plot_mag_change(figpar=figpar, savedir=savedir, **info)

    # 4. Plot autocorrelations
    elif analysis == 'a': # autocorr
        gen_plots.plot_autocorr(figpar=figpar, savedir=savedir, **info)
    
    # 5. Plots: a) trace areas by quintile, b) average traces, c) trace areas 
    # by suprise for first vs last quintile, for each ROI group, for each 
    # session
    elif analysis == 'g': # roi_grps_ch
        plot_rois_by_grp(figpar=figpar, savedir=savedir, **info)

    # 6. Plot colormaps and traces for orientations/directions
    elif analysis == 'o': # colormaps
        plot_oridirs(figpar=figpar, savedir=savedir, parallel=parallel, **info)

    # 7. Plot orientation tuning curves for ROIs
    elif analysis == 'c': # tuning curves
        plot_tune_curves(figpar=figpar, savedir=savedir, parallel=parallel, 
                         plot_tc=plot_tc, **info)

    # 8. Plots ROI responses for positions and mean gabor orientations
    elif analysis == 'p': # position orientation resp
        plot_posori_resp(figpar=figpar, savedir=savedir, **info)

    else:
        print('No plotting function for analysis {}'.format(analysis))


#############################################
def plot_roi_areas_by_grp_qu(analyspar, sesspar, stimpar, extrapar, permpar, 
                             quintpar, roigrppar, sess_info, roi_grps, 
                             figpar=None, savedir=None):
    """
    plot_roi_areas_by_grp_qu(analyspar, sesspar, stimpar, extrapar, permpar, 
                             quintpar, roigrppar, sess_info, roi_grps)

    From dictionaries, plots average integrated surprise, no surprise or 
    difference between surprise and no surprise activity per group of ROIs 
    showing significant surprise in first and/or last quintile. Each session is 
    in a different plot.

    Returns figure name and save directory path.

    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ['analysis'] (str): analysis type (e.g., 'g')
            ['datatype'] (str): datatype (e.g., 'roi')
            ['seed']     (int): seed value used
        - permpar (dict)  : dictionary with keys of PermPar namedtuple 
        - quintpar (dict) : dictionary with keys of QuintPar namedtuple
        - roigrppar (dict): dictionary with keys of RoiGrpPar namedtuple 
        - sess_info (dict): dictionary containing information from each
                            session 
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - roi_grps (dict) : dictionary containing ROI grps information
            ['all_roi_grps'] (list): nested lists containing ROI numbers 
                                     included in each group, structured as 
                                     follows:
                                         if sets of groups are passed: 
                                             session x set x roi_grp
                                         if one group is passed: 
                                             session x roi_grp
            ['grp_names'] (list)   : list of names of the ROI groups in roi grp 
                                     lists (order preserved)
            ['grp_st'] (array-like): nested list or array of group stats 
                                     (mean/median, error) across ROIs, 
                                     structured as:
                                         session x quintile x grp x stat
            ['grp_ns'] (array-like): nested list of group ns, structured as: 
                                         session x grp

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    

    Returns:
        - fulldir (str) : final name of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
        - savename (str): name under which the figure is saved
    """


    opstr_pr = sess_str_util.op_par_str(roigrppar['plot_vals'], 
                                        roigrppar['op'], str_type='print')
    stimstr_pr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                       stimpar['bri_dir'], stimpar['bri_size'], 
                                       stimpar['gabk'], 'print')
    statstr_pr = sess_str_util.stat_par_str(analyspar['stats'], 
                                            analyspar['error'], 'print')
    dendstr_pr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                              extrapar['datatype'], 'print')

    opstr = sess_str_util.op_par_str(roigrppar['plot_vals'], roigrppar['op'])
    sessstr = sess_str_util.sess_par_str(sesspar['sess_n'], stimpar['stimtype'], 
                                         sesspar['layer'], stimpar['bri_dir'],
                                         stimpar['bri_size'], stimpar['gabk']) 
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                         extrapar['datatype'])

    datatype = extrapar['datatype']
    if datatype != 'roi':
        raise ValueError('Function only implemented for roi datatype.')
    dimstr = sess_str_util.datatype_dim_str(datatype)

    # extract some info from sess_info
    keys = ['mouse_ns', 'sess_ns', 'lines', 'layers', 'nrois']
    [mouse_ns, sess_ns, lines, layers, nrois] = [sess_info[key] for key in keys]
    
    n_sess = len(mouse_ns)
    nanroi_vals = [sess_info['nanrois'], sess_info['nanrois_dff']]
    [n_nan, n_nan_dff] = [[len(val[i]) for i in range(n_sess)] 
                                       for val in nanroi_vals]

    grp_st = np.asarray(roi_grps['grp_st'])
    grp_ns = np.asarray(roi_grps['grp_ns'])

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    fig, ax = plot_util.init_fig(n_sess, **figpar['init'])
    for i, sess_st in enumerate(grp_st):
        sub_ax = plot_util.get_subax(ax, i)
        sess_nrois = sess_gen_util.get_nrois(nrois[i], n_nan[i], n_nan_dff[i], 
                                   analyspar['remnans'], analyspar['fluor'])
        for g, g_n in enumerate(grp_ns[i]):
            leg = '{} ({})'.format(roi_grps['grp_names'][g], g_n)
            plot_util.plot_errorbars(sub_ax, y=sess_st[:, g, 0], 
                                     err=sess_st[:, g, 1:], label=leg)

        title=(u'Mouse {} - {} {} across {}\nfor {} seqs \n(sess {}, {} '
                '{}{}, {} tail (n={}))').format(mouse_ns[i], stimstr_pr, 
                                 statstr_pr, dimstr, opstr_pr, sess_ns[i], 
                                 lines[i], layers[i], dendstr_pr, 
                                 permpar['tails'], sess_nrois)
        
        sess_plot_util.add_axislabels(sub_ax, fluor=analyspar['fluor'], 
                                      area=True, x_ax='Quintiles', 
                                      datatype=datatype)
        sub_ax.set_title(title)

    if savedir is None:
        savedir = os.path.join(figpar['dirs'][datatype], 
                               figpar['dirs']['surp_qu'], 
                               figpar['dirs']['grped'])
    
    savename = '{}_{}{}_grps_{}_{}q_{}tail'.format(datatype, sessstr, dendstr,
                    opstr, quintpar['n_quints'], permpar['tails'])

    fulldir = plot_util.savefig(fig, savename, savedir, **figpar['save'])

    return fulldir, savename


#############################################
def plot_roi_traces_by_grp(analyspar, sesspar, stimpar, extrapar, permpar, 
                           quintpar, roigrppar, sess_info, roi_grps, 
                           figpar=None, savedir=None):
    """
    plot_roi_traces_by_grp(analyspar, sesspar, stimpar, extrapar, permpar, 
                           quintpar, roigrppar, sess_info, roi_grps)

    From dictionaries, plots ROI traces by group across surprise, no surprise or 
    difference between surprise and no surprise activity per quintile 
    (first/last) with each group in a separate subplot and each session in a 
    different figure.

    Returns save directory path.

    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ['analysis'] (str): analysis type (e.g., 'g')
            ['datatype'] (str): datatype (e.g., 'roi')
            ['seed']     (int): seed value used
        - permpar (dict)  : dictionary with keys of PermPar namedtuple 
        - quintpar (dict) : dictionary with keys of QuintPar namedtuple
        - roigrppar (dict): dictionary with keys of RoiGrpPar namedtuple 
        - sess_info (dict): dictionary containing information from each
                            session 
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - roi_grps (dict) : dictionary containing ROI groups:
            ['all_roi_grps'] (list)     : nested lists containing ROI numbers 
                                          included in each group, structured 
                                          as follows:
                                              if sets of groups are passed: 
                                                  session x set x roi_grp
                                              if one group is passed: 
                                                  session x roi_grp
            ['grp_names'] (list)        : list of names of the ROI groups in 
                                          ROI grp lists (order preserved)
            ['x_ran'] (array-like)      : list of time values for the traces
            ['trace_stats'] (array-like): nested lists or array of statistics 
                                          across ROIs for ROI groups 
                                          structured as:
                                              sess x qu x ROI grp x 
                                              stats x frame
    
    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
    
    Returns:
        - fulldir (str) : final name of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
        - savename (str): name under which the figure is saved
    """

    opstr_pr = sess_str_util.op_par_str(roigrppar['plot_vals'], roigrppar['op'], 
                                        'print')
    stimstr_pr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                    stimpar['bri_dir'], stimpar['bri_size'], 
                                    stimpar['gabk'], 'print')
    statstr_pr = sess_str_util.stat_par_str(analyspar['stats'], 
                                            analyspar['error'], 'print')
    dendstr_pr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                            extrapar['datatype'], 'print')

    opstr = sess_str_util.op_par_str(roigrppar['plot_vals'], roigrppar['op'])
    sessstr = sess_str_util.sess_par_str(sesspar['sess_n'], stimpar['stimtype'], 
                                         sesspar['layer'], stimpar['bri_dir'],
                                         stimpar['bri_size'], stimpar['gabk']) 
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                         extrapar['datatype'])
    datatype = extrapar['datatype']
    if datatype != 'roi':
        raise ValueError('Function only implemented for roi datatype.')
    dimstr = sess_str_util.datatype_dim_str(datatype)

    # extract some info from sess_info
    keys = ['mouse_ns', 'sess_ns', 'lines', 'layers', 'nrois']
    [mouse_ns, sess_ns, lines, layers, nrois] = [sess_info[key] for key in keys]
    
    n_sess = len(mouse_ns)
    nanroi_vals = [sess_info['nanrois'], sess_info['nanrois_dff']]
    [n_nan, n_nan_dff] = [[len(val[i]) for i in range(n_sess)] 
                                       for val in nanroi_vals]

    xran        = np.asarray(roi_grps['xran'])
    trace_stats = np.asarray(roi_grps['trace_stats'])
    
    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    # figure directories
    if savedir is None:
        savedir = os.path.join(figpar['dirs'][datatype], 
                               figpar['dirs']['surp_qu'], 
                               figpar['dirs']['grped'])

    if figpar['save']['use_dt'] is None:
        figpar = copy.deepcopy(figpar)
        figpar['save']['use_dt'] = gen_util.create_time_str()

    print_dir = False
    for i in range(n_sess):
        if i == n_sess - 1:
            print_dir = True
        n_grps = len(roi_grps['all_roi_grps'][i])
        fig, ax = plot_util.init_fig(n_grps, **figpar['init'])
        sess_nrois = sess_gen_util.get_nrois(nrois[i], n_nan[i], n_nan_dff[i], 
                                    analyspar['remnans'], analyspar['fluor'])
        for g, [grp_nam, grp_rois] in enumerate(zip(roi_grps['grp_names'], 
                                                 roi_grps['all_roi_grps'][i])):
            title = '{} group (n={})'.format(grp_nam, len(grp_rois))
            sub_ax = plot_util.get_subax(ax, g)
            sess_plot_util.add_axislabels(sub_ax, fluor=analyspar['fluor'], 
                                          datatype=datatype)
            for q, qu_lab in enumerate(quintpar['qu_lab']):
                plot_util.plot_traces(sub_ax, xran, trace_stats[i, q, g, 0], 
                                      trace_stats[i, q, g, 1:], title=title,
                                      alpha=0.8/len(quintpar['qu_lab']), 
                                      label=qu_lab.capitalize())

        if stimpar['stimtype'] == 'gabors': 
            sess_plot_util.plot_labels(ax, stimpar['gabfr'], 
                                    roigrppar['plot_vals'], 
                                    pre=stimpar['pre'], post=stimpar['post'], 
                                    sharey=figpar['init']['sharey'])

        fig.suptitle((u'Mouse {} - {} {} across {} for {} seqs '
                       '\n(sess {}, {} {}{}, {} tail (n={}))')
                        .format(mouse_ns[i], stimstr_pr, statstr_pr, dimstr, 
                                opstr_pr, sess_ns[i], lines[i], layers[i], 
                                dendstr_pr, permpar['tails'], sess_nrois))

        savename = ('{}_tr_m{}_{}{}_grps_{}_{}q_'
                    '{}tail').format(datatype, mouse_ns[i], sessstr, dendstr,
                                opstr, quintpar['n_quints'], permpar['tails'])
        
        fulldir = plot_util.savefig(fig, savename, savedir, 
                                    print_dir=print_dir, **figpar['save'])

    return fulldir


#############################################
def plot_roi_areas_by_grp(analyspar, sesspar, stimpar, extrapar, permpar, 
                          quintpar, roigrppar, sess_info, roi_grps, 
                          figpar=None, savedir=None):
    """
    plot_roi_areas_by_grp(analyspar, sesspar, stimpar, extrapar, permpar, 
                          quintpar, roigrppar, sess_info, roi_grps)

    From dictionaries, plots ROI traces by group across surprise, no surprise 
    or difference between surprise and no surprise activity per quintile 
    (first/last) with each group in a separate subplot and each session in a 
    different figure.

    Returns save directory path.

    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ['analysis'] (str): analysis type (e.g., 'g')
            ['datatype'] (str): datatype (e.g., 'roi')
            ['seed']     (int): seed value used
        - permpar (dict)  : dictionary with keys of PermPar namedtuple 
        - quintpar (dict) : dictionary with keys of QuintPar namedtuple
        - roigrppar (dict): dictionary with keys of RoiGrpPar namedtuple 
        - sess_info (dict): dictionary containing information from each
                            session 
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - roi_grps (dict)  : dictionary containing ROI groups:
            ['all_roi_grps'] (list)           : nested lists containing ROI 
                                                numbers included in each group, 
                                                structured as follows:
                                                  if sets of groups are passed: 
                                                      session x set x roi_grp
                                                  if one group is passed: 
                                                      session x roi_grp
            ['grp_names'] (list)              : list of names of the ROI groups  
                                                in ROI grp lists (order 
                                                preserved)
            ['area_stats'] (array-like)       : ROI group stats (mean/median,  
                                                error) across ROIs, structured
                                                as:
                                                  session x quintile x 
                                                  grp x stat
            ['area_stats_scaled'] (array-like): same as 'area_stats', but 
                                                with last quintile scaled 
                                                relative to first

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None    
    
    Returns:
        - fulldir (str) : final name of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
        - savename (str): name under which the figure is saved
    """
   
    opstr_pr = sess_str_util.op_par_str(roigrppar['plot_vals'], roigrppar['op'], 
                                        'print')
    stimstr_pr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                    stimpar['bri_dir'], stimpar['bri_size'], 
                                    stimpar['gabk'], 'print')
    statstr_pr = sess_str_util.stat_par_str(analyspar['stats'], 
                                            analyspar['error'], 'print')
    dendstr_pr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                            extrapar['datatype'], 'print')

    opstr = sess_str_util.op_par_str(roigrppar['plot_vals'], roigrppar['op'])
    sessstr = sess_str_util.sess_par_str(sesspar['sess_n'], stimpar['stimtype'], 
                                         sesspar['layer'], stimpar['bri_dir'],
                                         stimpar['bri_size'], stimpar['gabk']) 
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                              extrapar['datatype'])

    datatype = extrapar['datatype']
    if datatype != 'roi':
        raise ValueError('Function only implemented for roi datatype.')
    dimstr = sess_str_util.datatype_dim_str(datatype)

    # extract some info from sess_info
    keys = ['mouse_ns', 'sess_ns', 'lines', 'layers', 'nrois']
    [mouse_ns, sess_ns, lines, layers, nrois] = [sess_info[key] for key in keys]
    
    n_sess = len(mouse_ns)
    nanroi_vals = [sess_info['nanrois'], sess_info['nanrois_dff']]
    [n_nan, n_nan_dff] = [[len(val[i]) for i in range(n_sess)] 
                                       for val in nanroi_vals]

    # scaling strings for printing and filenames
    scales = [False, True]
    
    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    figpar['init']['subplot_wid'] /= 2.0
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    # for spacing the bars on the graph
    barw = 0.75
    _, bar_pos, xlims = plot_util.get_barplot_xpos(1, len(quintpar['qu_idx']), 
                                                   barw, btw_grps=1.5)
    bar_pos = bar_pos[0] # only one grp

    print_dir = False
    for i in range(n_sess):
        if i == n_sess - 1:
            print_dir = True
        n_grps = len(roi_grps['all_roi_grps'][i])
        sess_nrois = sess_gen_util.get_nrois(nrois[i], n_nan[i], n_nan_dff[i], 
                                    analyspar['remnans'], analyspar['fluor'])
        figs = []
        for scale in scales:
            sc_str    = sess_str_util.scale_par_str(scale)
            fig, ax = plot_util.init_fig(n_grps, **figpar['init'])
            figs.append(fig)
            for g, [grp_nam, grp_rois] in enumerate(zip(roi_grps['grp_names'], 
                                                roi_grps['all_roi_grps'][i])):
                title = '{} group (n={})'.format(grp_nam, len(grp_rois))
                sub_ax = plot_util.get_subax(ax, g)
                sub_ax.tick_params(labelbottom=False)
                sess_plot_util.add_axislabels(sub_ax, area=True, scale=scale, 
                                              x_ax='', datatype=datatype)
                for q, qu_lab in enumerate(quintpar['qu_lab']):
                    vals = roi_grps['area_stats{}'.format(sc_str)]
                    vals = np.asarray(vals)[i, q, g]
                    plot_util.plot_bars(sub_ax, bar_pos[q], vals[0], vals[1:], 
                                        title, alpha=0.5, xticks='None', 
                                        xlims=xlims, label=qu_lab.capitalize(), 
                                        hline=0, width=barw)


        suptitle = (u'Mouse {} - {} {} across {} for {} seqs\n(sess {},'
                    ' {} {}{}, {} tail (n={}))').format(mouse_ns[i], stimstr_pr, 
                                    statstr_pr, dimstr, opstr_pr, sess_ns[i], 
                                    lines[i], layers[i], dendstr_pr, 
                                    permpar['tails'], sess_nrois)

        savename = ('{}_area_m{}_{}{}_grps_{}_{}q_'
                    '{}tail').format(datatype, mouse_ns[i], sessstr, dendstr, 
                                opstr, quintpar['n_quints'], permpar['tails'])
        
        # figure directories
        if savedir is None:
            savedir = os.path.join(figpar['dirs'][datatype], 
                                   figpar['dirs']['surp_qu'], 
                                   figpar['dirs']['grped'])

        for i, (fig, scale) in enumerate(zip(figs, scales)):
            scale_str    = sess_str_util.scale_par_str(scale)
            scale_str_pr = sess_str_util.scale_par_str(scale, 'print')
            fig.suptitle(u'{}{}'.format(suptitle, scale_str_pr))
            full_savename = '{}{}'.format(savename, scale_str)
            fulldir = plot_util.savefig(fig, full_savename, savedir, 
                                        print_dir=print_dir, 
                                        **figpar['save'])

    return fulldir


#############################################
def plot_rois_by_grp(analyspar, sesspar, stimpar, extrapar, permpar, quintpar, 
                     roigrppar, sess_info, roi_grps, figpar=None, savedir=None):
    """
    plot_rois_by_grp(analyspar, sesspar, stimpar, extrapar, permpar, quintpar, 
                     roigrppar, sess_info, roi_grps)

    From dictionaries, plots ROI data by group across surprise, no surprise 
    or difference between surprise and no surprise activity per quintile 
    (first/last) with each group in a separate subplot and each session in a 
    different figure.

    Two types of ROI data are plotted:
        1. ROI traces, if 'trace_stats' is in roi_grps
        2. ROI areas, if 'area_stats' is in roi_grps 

    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ['analysis'] (str): analysis type (e.g., 'g')
            ['datatype'] (str): datatype (e.g., 'roi')
            ['seed']     (int): seed value used
        - permpar (dict)  : dictionary with keys of PermPar namedtuple 
        - quintpar (dict) : dictionary with keys of QuintPar namedtuple
        - roigrppar (dict): dictionary with keys of RoiGrpPar namedtuple
        - sess_info (dict): dictionary containing information from each
                            session 
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - roi_grps (dict) : dictionary containing ROI groups:
            ['all_roi_grps'] (list)           : nested lists containing ROI  
                                                numbers included in each group, 
                                                structured as follows:
                                                  if sets of groups are passed: 
                                                      session x set x roi_grp
                                                  if one group is passed: 
                                                      session x roi_grp
            ['grp_names'] (list)              : list of names of the ROI groups 
                                                in ROI grp lists (order 
                                                preserved)
            ['grp_st'] (array-like)           : nested list or array of group 
                                                stats (mean/median, error) 
                                                across ROIs, structured as:
                                                    session x quintile x grp x 
                                                    stat
            ['grp_ns'] (array-like)           : nested list of group ns, 
                                                structured as: 
                                                    session x grp
            ['xran'] (array-like)             : array or list of time values 
                                                for the frame chunks
            ['trace_stats'] (array-like)      : array or nested list of 
                                                statistics across ROIs, for ROI 
                                                groups structured as:
                                                    sess x qu x ROI grp x stats 
                                                    x frame
            ['area_stats'] (array-like)       : ROI group stats (mean/median, 
                                                error) across ROIs, structured 
                                                as:
                                                    session x quintile x grp x 
                                                    stat
            ['area_stats_scaled'] (array-like): same as 'area_stats', but 
                                                with last quintile scaled
                                                relative to first
    
    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None
    """

    datatype = extrapar['datatype']
    if datatype != 'roi':
        raise ValueError('Function only implemented for roi datatype.')

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    comm_info = {'analyspar': analyspar,
                 'sesspar'  : sesspar,
                 'stimpar'  : stimpar,
                 'extrapar' : extrapar,
                 'permpar'  : permpar,
                 'quintpar' : quintpar,
                 'roigrppar': roigrppar,
                 'sess_info': sess_info,
                 'roi_grps' : roi_grps,
                 'figpar'   : figpar,
                 }

    if 'grp_st' in roi_grps.keys():
        plot_roi_areas_by_grp_qu(savedir=savedir, **comm_info)

    if 'trace_stats' in roi_grps.keys():
        plot_roi_traces_by_grp(savedir=savedir, **comm_info)

    if 'area_stats' in roi_grps.keys():
        plot_roi_areas_by_grp(savedir=savedir, **comm_info)


#############################################
def plot_oridir_traces(analyspar, sesspar, stimpar, extrapar, quintpar, 
                        tr_data, sess_info, figpar=None, savedir=None):
    """
    plot_oridir_traces(analyspar, sesspar, stimpar, extrapar, quintpar, 
                       tr_data, sess_info)

    From dictionaries, plots average activity across gabor orientations or 
    brick directions per ROI as colormaps for a single session and optionally
    a single quintile. 

    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ['analysis'] (str): analysis type (e.g., 'o')
            ['datatype'] (str): datatype (e.g., 'roi')
        - quintpar (dict) : dictionary with keys of QuintPar namedtuple
        - sess_info (dict): dictionary containing information from each
                            session (only first session used)
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - tr_data (dict)   : dictionary containing information to plot colormap.
                             Surprise x ori/dir keys are formatted as 
                             [{}_{}.format(s, od)] for surp in ['reg', 'surp']
                                                  and od in [0, 45, 90, 135] or 
                                                            ['right', 'left']
            ['n_seqs'] (dict): dictionary containing number of segs for each
                               surprise x ori/dir combination under a 
                               separate key
            ['stats'] (dict) : dictionary containing trace mean/medians across
                               ROIs in 2D arrays or nested lists, 
                               structured as:
                                   stats (me, err) x frames
                               with each surprise x ori/dir combination under a 
                               separate key
                               (NaN arrays for combinations with 0 seqs.)
    
    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None
    
    Returns:
        - fulldir (str) : final name of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
        - savename (str): name under which the figure is saved
    """

    statstr_pr = sess_str_util.stat_par_str(analyspar['stats'], 
                                            analyspar['error'], 'print')
    stimstr_pr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                    stimpar['bri_dir'], stimpar['bri_size'], 
                                    stimpar['gabk'], 'print')
    dendstr_pr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                    extrapar['datatype'], 'print')

    stimstr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                       stimpar['bri_dir'], stimpar['bri_size'], 
                                       stimpar['gabk'])
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                         extrapar['datatype'])
    datatype = extrapar['datatype']
    if datatype != 'roi':
        raise ValueError('Function only implemented for roi datatype.')
    dimstr = sess_str_util.datatype_dim_str(datatype)

    if savedir is None:
        savedir = os.path.join(figpar['dirs'][datatype], figpar['dirs']['oridir'])

    # extract some info from dictionaries
    keys = ['mouse_ns', 'sess_ns', 'lines', 'layers', 'nrois']
    [mouse_n, sess_n, line, layer, nrois] = [sess_info[key][0] for key in keys]

    nanroi_vals = [sess_info['nanrois'], sess_info['nanrois_dff']]
    [n_nan, n_nan_dff] = [len(val[0]) for val in nanroi_vals]
    sess_nrois = sess_gen_util.get_nrois(nrois, n_nan, n_nan_dff, 
                                    analyspar['remnans'], analyspar['fluor'])

    xran = tr_data['xran']

    surps = ['reg', 'surp']
    if stimpar['stimtype'] == 'gabors':
        deg = u'\u00B0'
        oridirs = stimpar['gab_ori']
    elif stimpar['stimtype'] == 'bricks':
        deg = ''
        oridirs = stimpar['bri_dir']

    qu_str, qu_str_pr = quintpar['qu_lab'][0], quintpar['qu_lab_pr'][0]
    if qu_str != '':
        qu_str    = '_{}'.format(qu_str)      
    if qu_str_pr != '':
        qu_str_pr = ' - {}'.format(qu_str_pr.capitalize())

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    figpar['init']['ncols'] = len(oridirs) 
    
    suptitle = (u'Mouse {} - {} {} across {}{}\n(sess {}, {} {}{}, '
                 'n={})').format(mouse_n, stimstr_pr, statstr_pr, dimstr,
                 qu_str_pr, sess_n, line, layer, dendstr_pr, sess_nrois)
    savename = '{}_tr_m{}_sess{}{}_{}_{}{}'.format(datatype, mouse_n, sess_n, 
                                               qu_str, stimstr, layer, dendstr)
    
    fig, ax = plot_util.init_fig(len(oridirs), **figpar['init'])
    for o, od in enumerate(oridirs):
        cols = []
        for surp in surps: 
            sub_ax = plot_util.get_subax(ax, o)
            key = '{}_{}'.format(surp, od)
            stimtype_str_pr = stimpar['stimtype'][:-1].capitalize()
            title_tr = u'{} traces ({}{})'.format(stimtype_str_pr, od, deg)
            lab = '{} (n={})'.format(surp, tr_data['n_seqs'][key])
            sess_plot_util.add_axislabels(sub_ax, datatype=datatype)
            me  = np.asarray(tr_data['stats'][key][0])
            err = np.asarray(tr_data['stats'][key][1:])
            plot_util.plot_traces(sub_ax, xran, me, err, title_tr, label=lab)
            cols.append(sub_ax.lines[-1].get_color())
    
    if stimpar['stimtype'] == 'gabors':
        sess_plot_util.plot_labels(ax, stimpar['gabfr'], cols=cols,
                                   pre=stimpar['pre'], post=stimpar['post'], 
                                   sharey=figpar['init']['sharey'])

    fig.suptitle(suptitle)
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar['save'])

    return fulldir


#############################################
def scale_sort_trace_data(tr_data, fig_type='byplot', surps=['reg', 'surp'], 
                          oridirs=[0, 45, 90, 135]):
    """
    scale_sort_trace_data(tr_data)

    Returns a dictionary containing ROI traces scaled and sorted as 
    specified.

    Required args:        
        - tr_data (dict): dictionary containing information to plot colormap.
                            Surprise x ori/dir keys are formatted as 
                            [{}_{}.format(s, od)] for surp in ['reg', 'surp']
                                                and od in [0, 45, 90, 135] or 
                                                        ['right', 'left']
            ['n_seqs'] (dict)    : dictionary containing number of segs for 
                                   each surprise x ori/dir combination under a 
                                   separate key
            ['scale_vals'] (dict): dictionary containing 1D array or list of 
                                   scaling values for each surprise x ori/dir 
                                   combination under a separate key.
                                   (NaN arrays for combinations with 0 seqs.)
                ['{}_min'] (num): minimum value from corresponding tr_stats 
                                  mean/medians
                ['{}_max'] (num): maximum value from corresponding tr_stats 
                                  mean/medians
            ['roi_sort'] (dict)  : dictionary containing 1D arrays or list 
                                   of peak sorting order for each 
                                   surprise x ori/dir combination under
                                   a separate key.
                                   (NaN arrays for combinations with 0 seqs.)
            ['roi_me'] (dict)    : dictionary containing trace mean/medians
                                   for each ROI as 2D arrays or nested lists, 
                                   structured as:
                                       ROIs x frames, 
                                   with each surprise x ori/dir combination 
                                   under a separate key. 
                                   (NaN arrays for combinations with 0 seqs.)

    Optional args:
        - fig_type (str) : how to scale and sort ROIs, 
                               i.e. each plot separately ('byplot'), 
                                   each orientation/direction by its regular 
                                       plot ('byreg'), 
                                   each reg/surp by the first 
                                       orientation/direction 
                                       ('by0deg' or 'byright')
                                   each plot by the first plot ('byfirst')
                           default: 'byplot'
        - surps (list)   : surprise value names used in keys, ordered
                           default: ['reg', 'surp']
        - oridirs (list) : orientation/direction value names used in keys,
                           ordered
                           default: [0, 45, 90, 135]
    
    Returns:
        - scaled_sort_data_me (dict): dictionary containing scaled and 
                                      sorted trace mean/medians for each ROI as 
                                      2D arrays, structured as:
                                          ROIs x frames, 
                                      with each surprise x ori/dir combination 
                                      under a separate key, as above
    """

    scaled_sort_data_me = dict()
    scale_vals  = tr_data['scale_vals']
    roi_sort   = tr_data['roi_sort']
    for od in oridirs:
        for s in surps:
            key = '{}_{}'.format(s, od)
            me = np.asarray(tr_data['roi_me'][key])
            # mean/median organized as ROI x fr
            if tr_data['n_seqs'][key] == 0: # no data under these criteria
                scaled_sort_data_me[key] = me.T
                continue
            if fig_type == 'byplot':
                min_v = np.asarray(scale_vals['{}_min'.format(key)])
                max_v = np.asarray(scale_vals['{}_max'.format(key)])
                sort_arg = roi_sort[key]

            elif fig_type == 'byreg':
                mins = [scale_vals['{}_{}_min'.format(sv, od)] for sv in surps]
                maxs = [scale_vals['{}_{}_max'.format(sv, od)] for sv in surps]
                min_v = np.nanmin(np.asarray(mins), axis=0)
                max_v = np.nanmax(np.asarray(maxs), axis=0)
                idx = 0
                # find first reg/surp plot with data
                while tr_data['n_seqs']['{}_{}'.format(surps[idx], od)] == 0:
                    idx += 1
                sort_arg = roi_sort['{}_{}'.format(surps[idx], od)]
            
            elif fig_type in ['by0deg', 'byright']:
                mins = [scale_vals['{}_{}_min'.format(s, odv)] 
                                           for odv in oridirs]
                maxs = [scale_vals['{}_{}_max'.format(s, odv)] 
                                           for odv in oridirs]
                min_v = np.nanmin(np.asarray(mins), axis=0)
                max_v = np.nanmax(np.asarray(maxs), axis=0)
                idx = 0
                # find first oridir plot with data
                while tr_data['n_seqs']['{}_{}'.format(s, oridirs[idx])] == 0:
                    idx += 1
                sort_arg = roi_sort['{}_{}'.format(s, oridirs[idx])]
                
            elif fig_type == 'byfir':

                mins = [scale_vals['{}_{}_min'.format(sv, odv)] 
                                        for sv in surps for odv in oridirs]
                maxs = [scale_vals['{}_{}_max'.format(sv, odv)]
                                        for sv in surps for odv in oridirs]
                min_v = np.nanmin(np.asarray(mins), axis=0)
                max_v = np.nanmax(np.asarray(maxs), axis=0)
                idx_s, idx_od, count = 0, 0, 0
                # find first plot with data (by oridirs, then surps)
                while tr_data['n_seqs']['{}_{}'.format(surps[idx_s], 
                                                       oridirs[idx_od])] == 0:
                    count += 1
                    idx_od = count % len(oridirs)
                    idx_s  = count // len(oridirs) % len(surps)
                sort_arg = roi_sort['{}_{}'.format(surps[idx_s], 
                                                   oridirs[idx_od])]

            me_scaled = ((me.T - min_v)/(max_v - min_v))
            scaled_sort_data_me[key] = me_scaled[:, sort_arg]
    
    return scaled_sort_data_me


#############################################
def plot_oridir_colormap(fig_type, analyspar, sesspar, stimpar, quintpar, 
                         tr_data, sess_info, figpar=None, savedir=None, 
                         print_dir=True):
    """
    plot_oridir_colormap(fig_type, analyspar, sesspar, stimpar, quintpar,  
                         tr_data, sess_info)

    From dictionaries, plots average activity across gabor orientations or 
    brick directions per ROI for a single session and optionally a single 
    quintile. (Single figure type) 

    Required args:
        - fig_type (str)  : type of figure to plot, i.e., 'byplot', 'byreg', 
                            'byfir' or 'by{}{}' (ori/dir, deg)
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - quintpar (dict) : dictionary with keys of QuintPar namedtuple
        - sess_info (dict): dictionary containing information from each
                            session (only first session used)
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - tr_data (dict)   : dictionary containing information to plot colormap.
                             Surprise x ori/dir keys are formatted as 
                             [{}_{}.format(s, od)] for surp in ['reg', 'surp']
                                                  and od in [0, 45, 90, 135] or 
                                                            ['right', 'left']
            ['n_seqs'] (dict)    : dictionary containing number of segs for 
                                   each surprise x ori/dir combination under a 
                                   separate key
            ['scale_vals'] (dict): dictionary containing 1D array or list of 
                                   scaling values for each surprise x ori/dir 
                                   combination under a separate key.
                                   (NaN arrays for combinations with 0 seqs.)
                ['{}_min'] (num): minimum value from corresponding tr_stats 
                                  mean/medians
                ['{}_max'] (num): maximum value from corresponding tr_stats 
                                  mean/medians
            ['roi_sort'] (dict)  : dictionary containing 1D arrays or list 
                                   of peak sorting order for each 
                                   surprise x ori/dir combination under a 
                                   separate key.
                                   (NaN arrays for combinations with 0 seqs.)
            ['roi_me'] (dict)    : dictionary containing trace mean/medians for 
                                   each ROI as 2D arrays or nested lists, 
                                   structured as:
                                       ROIs x frames, 
                                   with each surprise x ori/dir combination 
                                   under a separate key
                                   (NaN arrays for combinations with 0 seqs.)
    
    Optional args:
        - figpar (dict)  : dictionary containing the following figure parameter 
                           dictionaries
                           default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
            ['mng']  (dict): dictionary with parameters to manage matplotlib
        - savedir (str)  : path of directory in which to save plots.
                           default: None
    
    Returns:
        - fulldir (str) : final name of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
    """

    statstr_pr = sess_str_util.stat_par_str(analyspar['stats'], 
                                            analyspar['error'], 'print')
    stimstr_pr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                    stimpar['bri_dir'], stimpar['bri_size'], 
                                    stimpar['gabk'], 'print')
    dendstr_pr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                           'roi', 'print')

    stimstr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                       stimpar['bri_dir'], stimpar['bri_size'], 
                                       stimpar['gabk'])
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                         'roi')

    if savedir is None:
        savedir = os.path.join(figpar['dirs']['roi'], 
                               figpar['dirs']['oridir'])

    cmap = plot_util.manage_mpl(cmap=True, nbins=100, **figpar['mng'])

    # extract some info from sess_info (only one session)
    keys = ['mouse_ns', 'sess_ns', 'lines', 'layers']
    [mouse_n, sess_n, line, layer] = [sess_info[key][0] for key in keys]

    surps = ['reg', 'surp']
    if stimpar['stimtype'] == 'gabors':
        var_name = 'orientation'
        deg  = 'deg'
        deg_pr = u'\u00B0'
        oridirs = stimpar['gab_ori']
    elif stimpar['stimtype'] == 'bricks':
        var_name = 'direction'
        deg  = ''
        deg_pr = ''
        oridirs = stimpar['bri_dir']
    
    qu_str, qu_str_pr = quintpar['qu_lab'][0], quintpar['qu_lab_pr'][0]
    if qu_str != '':
        qu_str    = '_{}'.format(qu_str)      
    if qu_str_pr != '':
        qu_str_pr = ' - {}'.format(qu_str_pr.capitalize())

    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)
    figpar['init']['ncols'] = len(oridirs)
    figpar['init']['sharex'] = True
    
    # plot colormaps
    gentitle = (u'Mouse {} - {} {} across seqs colormaps{}\n(sess {}, '
                 '{} {}{})').format(mouse_n, stimstr_pr, statstr_pr,  
                                  qu_str_pr, sess_n, line, layer, dendstr_pr)
    gen_savename = 'roi_cm_m{}_sess{}{}_{}_{}{}'.format(mouse_n, sess_n, 
                                               qu_str, stimstr, layer, dendstr)

    if fig_type == 'byplot':
        scale_type = 'per plot'
        peak_sort  = ''
        figpar['init']['sharey'] = False
    elif fig_type == 'byreg':
        scale_type = 'within {}'.format(var_name)
        peak_sort  = ' of {}'.format(surps[0])
        figpar['init']['sharey'] = False
    elif fig_type == 'by{}{}'.format(oridirs[0], deg):
        scale_type = 'within surp/reg'
        peak_sort  = ' of first {}'.format(var_name)
        figpar['init']['sharey'] = True
    elif fig_type == 'byfir':
        scale_type = 'across plots'
        peak_sort  = ' of first plot'
        figpar['init']['sharey'] = True
    else:
        gen_util.accepted_values_error('fig_type', fig_type, 
                                   ['byplot', 'byreg', 
                                    'by{}{}'.format(oridirs[0], deg), 'byfir'])

    subtitle = (u'ROIs sorted by peak activity{} and scaled '
                 '{}').format(peak_sort, scale_type)
    print(u'    - {}'.format(subtitle))
    suptitle = u'{}\n({})'.format(gentitle, subtitle)
    
    # get scaled and sorted ROI mean/medians (ROI x frame)
    scaled_sort_me = scale_sort_trace_data(tr_data, fig_type, surps, oridirs)
    fig, ax = plot_util.init_fig(len(oridirs) * len(surps), **figpar['init'])

    nrois = len(scaled_sort_me['roi_sort']['{}_{}'.format(surps[0], 
                                                          oridirs[0])])
    yticks_ev = int(10 * np.max([1, np.ceil(nrois/100)])) # avoid > 10 ticks
    for o, od in enumerate(oridirs):
        for s, surp in enumerate(surps):    
            sub_ax = ax[s][o]
            key = '{}_{}'.format(surp, od)
            title = u'{} seqs ({}{}) (n={})'.format(surp.capitalize(), od, 
                                                deg_pr, tr_data['n_seqs'][key])
            x_ax = None
            if s == 0:
                x_ax = ''
            
            sess_plot_util.add_axislabels(sub_ax, fluor=analyspar['fluor'], 
                                       x_ax=x_ax, y_ax='ROIs', datatype='roi')
            im = plot_util.plot_colormap(sub_ax, scaled_sort_me[key], 
                                    title=title, cmap=cmap,
                                    xran=[stimpar['pre'], stimpar['post']], 
                                    yticks_ev=yticks_ev)
    
    for s, surp in enumerate(surps):
        sub_ax = ax[s:s+1]
        if stimpar['stimtype'] == 'gabors':
            sess_plot_util.plot_labels(sub_ax, stimpar['gabfr'], surp, 
                            pre=stimpar['pre'], post=stimpar['post'], 
                            sharey=figpar['init']['sharey'], t_heis=-0.05)
    
    plot_util.add_colorbar(fig, im, len(oridirs))
    fig.suptitle(suptitle)
    savename = '{}_{}'.format(gen_savename, fig_type)
    fulldir = plot_util.savefig(fig, savename, savedir, print_dir=print_dir, 
                                **figpar['save'])
    
    plt.close(fig)
    
    return fulldir


#############################################
def plot_oridir_colormaps(analyspar, sesspar, stimpar, extrapar, quintpar, 
                          tr_data, sess_info, figpar=None, savedir=None, 
                          parallel=False):
    """
    plot_oridir_colormaps(analyspar, sesspar, stimpar, extrapar, quintpar, 
                          tr_data, sess_info)

    From dictionaries, plots average activity across gabor orientations or 
    brick directions per ROI as colormaps for a single session and optionally
    a single quintile. 

    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ['analysis'] (str): analysis type (e.g., 'o')
            ['datatype'] (str): datatype (e.g., 'roi')
        - quintpar (dict) : dictionary with keys of QuintPar namedtuple
        - sess_info (dict): dictionary containing information from each
                            session (only first session used)
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - tr_data (dict)   : dictionary containing information to plot colormap.
                             Surprise x ori/dir keys are formatted as 
                             [{}_{}.format(s, od)] for surp in ['reg', 'surp']
                                                  and od in [0, 45, 90, 135] or 
                                                            ['right', 'left']
            ['n_seqs'] (dict)    : dictionary containing number of seqs for 
                                   each surprise x ori/dir combination under a 
                                   separate key
            ['scale_vals'] (dict): dictionary containing 1D array or list of 
                                   scaling values for each surprise x ori/dir 
                                   combination under a separate key.
                                   (NaN arrays for combinations with 0 seqs.)
                ['{}_min'] (num): minimum value from corresponding tr_stats 
                                  mean/medians
                ['{}_max'] (num): maximum value from corresponding tr_stats 
                                  mean/medians
            ['roi_sort'] (dict)  : dictionary containing 1D arrays or list 
                                   of peak sorting order for each 
                                   surprise x ori/dir combination under a 
                                   separate key.
                                   (NaN arrays for combinations with 0 seqs.)
            ['roi_me'] (dict)    : dictionary containing trace mean/medians for 
                                   each ROI as 2D arrays or nested lists, 
                                   structured as:
                                       ROIs x frames, 
                                   with each surprise x ori/dir combination 
                                   under a separate key
                                   (NaN arrays for combinations with 0 seqs.)
    
    Optional args:
        - figpar (dict)  : dictionary containing the following figure parameter 
                           dictionaries
                           default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
            ['mng']  (dict): dictionary with parameters to manage matplotlib
        - savedir (str)  : path of directory in which to save plots.
                           default: None
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
                           default: False
    
    Returns:
        - fulldir (str) : final name of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
    """

    datatype = extrapar['datatype']
    if datatype != 'roi':
        raise ValueError('Function only implemented for roi datatype.')

    if stimpar['stimtype'] == 'gabors':
        oridirs = stimpar['gab_ori']
        deg  = 'deg'
    elif stimpar['stimtype'] == 'bricks':
        oridirs = stimpar['bri_dir']
        deg = ''
    
    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()
    figpar['save']['fig_ext'] = 'png' # svg too big

    fig_types  = ['byplot', 'byreg', 'by{}{}'.format(oridirs[0], deg), 'byfir']
    fig_last = len(fig_types) - 1
    
    if parallel:
        n_jobs = min(multiprocessing.cpu_count(), len(fig_types))
        fulldirs = Parallel(n_jobs=n_jobs)(delayed(plot_oridir_colormap)
                         (fig_type, analyspar, sesspar, stimpar, quintpar, 
                          tr_data, sess_info, figpar, savedir, (f == fig_last)) 
                          for f, fig_type in enumerate(fig_types)) 
        fulldir = fulldirs[-1]
    else:
        for f, fig_type in enumerate(fig_types):
            print_dir = (f == fig_last)
            fulldir = plot_oridir_colormap(fig_type, analyspar, sesspar, 
                                    stimpar, quintpar, tr_data, sess_info, 
                                    figpar, savedir, print_dir)

    return fulldir


#############################################
def plot_oridirs(analyspar, sesspar, stimpar, extrapar, quintpar, 
                 tr_data, sess_info, figpar=None, savedir=None, 
                 parallel=False):
    """
    plot_oridirs(analyspar, sesspar, stimpar, extrapar, quintpar, 
                 tr_data, sess_info)

    From dictionaries, plots average activity across gabor orientations or 
    brick directions per ROI as colormaps, as well as traces across ROIs for a 
    single session and optionally a single quintile. 

    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ['analysis'] (str): analysis type (e.g., 'o')
            ['datatype'] (str): datatype (e.g., 'roi')
        - quintpar (dict) : dictionary with keys of QuintPar namedtuple
        - sess_info (dict): dictionary containing information from each
                            session (one first session used) 
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - tr_data (dict)   : dictionary containing information to plot colormap.
                             Surprise x ori/dir keys are formatted as 
                             [{}_{}.format(s, od)] for surp in ['reg', 'surp']
                                                  and od in [0, 45, 90, 135] or 
                                                            ['right', 'left']
            ['n_seqs'] (dict)    : dictionary containing number of segs for each
                                   surprise x ori/dir combination under a 
                                   separate key
            ['scale_vals'] (dict): dictionary containing 1D array or list of 
                                   scaling values for each surprise x ori/dir 
                                   combination under a separate key.
                                   (NaN arrays for combinations with 0 seqs.)
                ['{}_min'] (num): minimum value from corresponding tr_stats 
                                  mean/medians
                ['{}_max'] (num): maximum value from corresponding tr_stats 
                                  mean/medians
            ['roi_sort'] (dict) : dictionary containing 1D arrays or list of 
                                  peak sorting order for each 
                                  surprise x ori/dir combination under a 
                                  separate key.
                                  (NaN arrays for combinations with 0 seqs.)
            ['roi_me'] (dict)   : dictionary containing trace mean/medians
                                  for each ROI as 2D arrays or nested lists, 
                                  structured as:
                                      ROIs x frames, 
                                  with each surprise x ori/dir combination 
                                  under a separate key.
                                  (NaN arrays for combinations with 0 seqs.)
            ['stats'] (dict)    : dictionary containing trace mean/medians 
                                  across ROIs in 2D arrays or nested lists, 
                                  structured as: 
                                      stats (me, err) x frames
                                  with each surprise x ori/dir combination 
                                  under a separate key
                                  (NaN arrays for combinations with 0 seqs.)

    Optional args:
        - figpar (dict)  : dictionary containing the following figure parameter 
                           dictionaries
                           default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
            ['mng']  (dict): dictionary with parameters to manage matplotlib
        - savedir (str)  : path of directory in which to save plots.
                           default: None
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
                           default: False
    """

    datatype = extrapar['datatype']
    if datatype != 'roi':
        raise ValueError('Function only implemented for roi datatype.')

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    comm_info = {'analyspar': analyspar,
                 'sesspar'  : sesspar,
                 'stimpar'  : stimpar,
                 'extrapar' : extrapar,
                 'quintpar' : quintpar,
                 'sess_info': sess_info,
                 'tr_data' : tr_data,
                 'figpar'   : figpar,
                 }

    if 'roi_me' in tr_data.keys():
        plot_oridir_colormaps(savedir=savedir, parallel=parallel, 
                              **comm_info)

    if 'stats' in tr_data.keys():
        plot_oridir_traces(savedir=savedir, **comm_info)


#############################################
def plot_prev_analysis(subax_col, xran, gab_oris, gab_data, gab_vm_pars, 
                       gab_hist_pars, title_str='', fluor='dff'):
    """
    plot_prev_analysis(subax_col, xran, gab_oris, gab_data, gab_vm_pars, 
                       gab_hist_pars)

    Plots orientation fluorescence data for a specific ROI, as well as ROI 
    orientation tuning curves. 

    Required args:
        - subax_col (plt Axis)    : axis column
        - xran (array-like)       : x values
        - gab_oris (list)         : list of orientation values corresponding to 
                                    the gab_data
        - gab_data (list)         : list of mean integrated fluorescence data 
                                    per orientation
        - gab_vm_pars (array-like): array of Von Mises parameters 
                                    (kappa, mean, scale)
        - gab_hist_pars (list)  : parameters used to convert gab_data to 
                                  histogram values (sub, mult) used in Von 
                                  Mises parameter estimation (sub, mult)
    
    Optional args:
        - title_str (str)   : main title
                              default: ''
        - fluor (str)       : fluorescence type
                              default: 'dff'
    """

    deg = u'\u00B0'

    # Top: Von Mises fits
    vm_fit = st.vonmises.pdf(np.radians(xran), *gab_vm_pars)
    subax_col[0].plot(xran, vm_fit)
    subax_col[0].set_title('Von Mises fits{}'.format(title_str))
    subax_col[0].set_ylabel('Probability density')
    col = subax_col[0].lines[-1].get_color()
    
    # Mid: actual data
    subax_col[1].plot(gab_oris, gab_data, marker='.', lw=0, alpha=0.3, 
                       color=col)
    subax_col[1].set_title('Mean AUC per orientation')
    y_str = sess_str_util.fluor_par_str(fluor, str_type='print')
    subax_col[1].set_ylabel(u'{} area (mean)'.format(y_str))

    # Bottom: data as histogram for fitting
    counts = np.around((np.asarray(gab_data) - gab_hist_pars[0]) * \
                        gab_hist_pars[1]).astype(int)
    freq_data = np.repeat(np.asarray(gab_oris), counts)                
    subax_col[2].hist(freq_data, 360, color=col)
    subax_col[2].set_title('Orientation histogram')
    subax_col[2].set_xlabel(u'Orientations ({})'.format(deg))
    subax_col[2].set_ylabel('Artificial counts')
    plot_util.set_ticks(subax_col[2], 'x', np.min(xran), np.max(xran), 10)


#############################################
def plot_roi_tune_curves(tc_oris, roi_data, n, nrois, seq_info, 
                         roi_vm_pars=None, roi_hist_pars=None, 
                         fluor='dff', comb_gabs=False, gentitle='', 
                         gen_savename='', figpar=None, savedir=None):
    """
    plot_roi_tune_curves(tc_oris, roi_data, n, nrois, seq_info)
    
    Plots orientation fluorescence data for a specific ROI, as well as ROI 
    orientation tuning curves, if provided. 


    Required args:
        - tc_oris (list)        : list of orientation values corresponding to 
                                  the tc_data
                                    surp x gabor (1 if comb_gabs) x oris
        - roi_data (list)       : list of mean integrated fluorescence data per 
                                  orientation, structured as 
                                    surp (x gabor (1 if comb_gabs)) x oris
        - n (int)               : ROI number
        - nrois (int)           : total number of ROIs
        - seq_info (list)       : list of strings with info on each group
                                  plotted (2)
    
    Optional args:
        - roi_vm_pars (3D array): array of Von Mises parameters: 
                                    surp x gabor (1 if comb_gabs) 
                                        x par (kappa, mean, scale)
                                  default: None
        - roi_hist_pars (list)  : parameters used to convert tc_data to 
                                  histogram values (sub, mult) used in Von 
                                  Mises parameter estimation, structured as:
                                    surp x gabor (1 if comb_gabs) x 
                                    param (sub, mult)
                                  default: None
        - fluor (str)           : fluorescence type
                                  default: 'dff'
        - comb_gabs (bool)      : if True, data from all gabors was combined
                                  default: False
        - gentitle (str)        : general title for the plot
                                  default: ''
        - gen_savename (str)    : general title for the plot
                                  default: ''
        - figpar (dict)         : dictionary containing the following figure 
                                  parameter dictionaries
                                  default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
            ['mng']  (dict): dictionary with parameters to manage matplotlib
        - savedir (str)         : path of directory in which to save plots.
                                  default: None
    Returns:
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
    """

    flat_oris = [o for gos in tc_oris for oris in gos for o in oris]
    max_val = 90
    if np.max(np.absolute(flat_oris)) > max_val:
        max_val = 180
        if np.max(np.absolute(flat_oris)) > max_val:
            raise ValueError(('Orientations expected to be at most between '
                              '-180 and 180.'))
    xran = np.linspace(-max_val, max_val, 360)

    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()
    figpar['init']['ncols'] = 2
    figpar['init']['sharex'] = True
    figpar['init']['sharey'] = False
    figpar['save']['fig_ext'] = 'png' # svg too big

    plot_util.manage_mpl(**figpar['mng'])

    if savedir is None:
        savedir = os.path.join(figpar['dirs']['roi'], 
                               figpar['dirs']['tune_curv'])
    if roi_vm_pars is not None:
        savedir = os.path.join(savedir, 'prev')

    print_dir = False
    if n == 0:
        print_dir = True
    if (n + 1) % 15 == 0 or (n + 1) == nrois:
        print('ROI {}/{}'.format(n + 1, nrois))

    n_subplots = figpar['init']['ncols']
    if roi_vm_pars is not None:
        n_subplots *= 3
    fig, ax = plot_util.init_fig(n_subplots, **figpar['init'])
    fig.suptitle('{} - ROI {} ({} total)'.format(gentitle, n + 1, nrois))
    
    deg = u'\u00B0'
    for s, surp_oris in enumerate(tc_oris):
        if comb_gabs:
            gab_str = '(gabors combined)'
            for subax in ax[0, s+1:]: # to advance the color cycle
                subax.plot([], [])
        else:
            gab_str = '({} gabors)'.format(len(surp_oris))
        title_str = ' for {} {}'.format(seq_info[s], gab_str)
        for g, gab_oris in enumerate(surp_oris):
            if roi_vm_pars is not None:
                plot_prev_analysis(ax[:, s], xran, gab_oris, roi_data[s][g], 
                                roi_vm_pars[s][g], roi_hist_pars[s][g], 
                                title_str, fluor)
            else:
                # Just plot activations by orientation
                ax[0, s].plot(gab_oris, roi_data[s], marker='.', lw=0, 
                              alpha=0.3)
                ax[0, s].set_title('AUC per orientation{}'.format(title_str), 
                                   fontsize='large')
                xlab = u'Orientations ({})'.format(deg)
                sess_plot_util.add_axislabels(ax[0, s], fluor=fluor, area=True, 
                                              x_ax=xlab, datatype='roi')
                plot_util.set_ticks(ax[0, s], 'x', -max_val, max_val, 5)

    # share y axis ranges within rows
    plot_util.share_lims(ax, 'row')
    savename = '{}_roi{}'.format(gen_savename, n + 1)
    fulldir = plot_util.savefig(fig, savename, savedir, 
                                print_dir=print_dir, **figpar['save'])

    plt.close(fig)

    return fulldir


#############################################
def plot_tune_curve_regr(vm_means, vm_regr, seq_info, gentitle='', 
                         gen_savename='', figpar=None, savedir=None):
    """
    plot_tune_curve_regr(vm_means, vm_regr, seq_info)
    
    Plots correlation for regular vs surprise orientation preferences. 

    Required args:
        - vm_mean (3D array): array of mean Von Mises means for each ROI, not 
                              weighted by kappa value or weighted (in rad): 
                                ROI x surp x kappa weighted (False, (True)
        - vm_regr (2D array): array of regression results correlating surprise 
                              and non surprise means across ROIs, not weighted 
                              by kappa value or weighted (in rad): 
                                  regr_val (score, slope, intercept)
                                     x kappa weighted (False, (True)
        - seq_info (list)   : list of strings with info on each group
                              plotted (2)
    
    Optional args:
        - gentitle (str)        : general title for the plot
                                  default: ''
        - gen_savename (str)    : general title for the plot
                                  default: ''
        - figpar (dict)         : dictionary containing the following figure 
                                  parameter dictionaries
                                  default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
            ['mng']  (dict): dictionary with parameters to manage matplotlib
        - savedir (str)         : path of directory in which to save plots.
                                  default: None
    
    Returns:
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
    """
    
    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)

    if savedir is None:
        savedir = os.path.join(figpar['dirs']['roi'], 
                               figpar['dirs']['tune_curv'])
    savedir = os.path.join(savedir, 'prev')

    vm_means = np.asarray(vm_means)
    vm_regr = np.asarray(vm_regr)

    max_val = 90
    if np.max(np.absolute(vm_means)) > max_val:
        max_val = 180
        if np.max(np.absolute(vm_means)) > max_val:
            raise ValueError(('Orientations expected to be at most between '
                              '-180 and 180.'))
    xvals = [-max_val, max_val]

    deg = u'\u00B0'
    
    kapw = [0, 1]
    if vm_regr.shape[1] == 1:
        kapw = [0]

    for i in kapw:
        if i == 0:
            kap_str, kap_str_pr = '', ''
        if i == 1:
            kap_str = '_kapw'
            kap_str_pr = ' (kappa weighted)'
        data = np.rad2deg(vm_means[:, :, i])
        r_sqr, slope, interc = vm_regr[:, i]
        interc = np.rad2deg(interc)
        figpar['init']['ncols'] = 1
        fig, ax = plt.subplots(1)
        ax.plot(data[:, 0], data[:, 1], marker='.', lw=0, alpha=0.8)
        col = ax.lines[-1].get_color()
        yvals = [x * slope + interc for x in xvals]
        lab = u'R{} = {:.4f}'.format(u'\u00b2', r_sqr) # R2 = ##
        ax.plot(xvals, yvals, marker='', label=lab, color=col)
        for ax_let in ['x', 'y']:
            plot_util.set_ticks(ax, ax_let, xvals[0], xvals[1], 5)
        ax.set_xlabel((u'Mean orientation preference\nfrom {} '
                        '({})'.format(seq_info[0], deg)))
        ax.set_ylabel((u'Mean orientation preference\nfrom {} '
                        '({})'.format(seq_info[1], deg)))
        ax.set_title(('{}\nCorrelation btw orientation '
                      'pref{}'.format(gentitle, kap_str_pr)))
        ax.legend()
        savename = '{}_regr{}'.format(gen_savename, kap_str)
        fulldir = plot_util.savefig(fig, savename, savedir, print_dir=False, 
                                    **figpar['save'])
        plt.close(fig)

        return fulldir


#############################################
def plot_tune_curves(analyspar, sesspar, stimpar, extrapar, tcurvpar, 
                     tcurv_data, sess_info, figpar=None, savedir=None, 
                     parallel=False, plot_tc=True):
    """
    plot_tune_curves(analyspar, sesspar, stimpar, extrapar, tcurv_data, 
                     sess_info)

    Plots orientation fluorescence data, as well as ROI orientation tuning 
    curves, and a correlation plot for regular vs surprise orientation 
    preferences, if provided. 

    Required args:
        - analyspar (dict) : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)   : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)   : dictionary with keys of StimPar namedtuple
        - extrapar (dict)  : dictionary containing additional analysis 
                             parameters
            ['analysis'] (str)  : analysis type (e.g., 'o')
            ['comb_gabs'] (bool): if True, gabors were combined
            ['datatype'] (str): datatype (e.g., 'roi')
            ['seed'] (int)      : seed
        - tcurvpar (dict)  : dictionary with keys of TCurvPar namedtuple
        - sess_info (dict) : dictionary containing information from each
                             session (one first session used) 
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - tcurv_data (dict): tuning curve data 
            ['oris'] (list)         : list of orientation values 
                                      corresponding to the tc_data
                                      surp x gabor (1 if comb_gabs) x oris
            ['data'] (list)         : list of mean integrated fluorescence 
                                      data per orientation, for each ROI, 
                                      structured as 
                                      ROI x surp x gabor (1 if comb_gabs) 
                                          x oris
            ['nseqs'] (list)        : number of sequences per surp

            if tcurvpar['prev']:
            ['vm_pars'] (4D array)  : array of Von Mises parameters for each
                                      ROI: 
                                      ROI x surp x gabor (1 if comb_gabs) 
                                          x par
            ['vm_mean'] (3D array)  : array of mean Von Mises means for each
                                      ROI, not weighted by kappa value or 
                                      weighted (if not comb_gabs) (in rad): 
                                      ROI x surp 
                                          x kappa weighted (False, (True))
            ['vm_regr'] (2D array)  : array of regression results 
                                      correlating surprise and non surprise
                                      means across ROIs, not weighted by 
                                      kappa value or weighted (if not 
                                      comb_gabs) (in rad): 
                                      regr_val (score, slope, intercept)
                                          x kappa weighted (False, (True)) 
            ['hist_pars'] (list)    : parameters used to convert tc_data to 
                                      histogram values (sub, mult) used
                                      in Von Mises parameter estimation, 
                                      structured as:
                                      ROI x surp x gabor (1 if comb_gabs) x 
                                         param (sub, mult)

    Optional args:
        - figpar (dict)  : dictionary containing the following figure parameter 
                           dictionaries
                           default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
            ['mng']  (dict): dictionary with parameters to manage matplotlib
        - savedir (str)  : path of directory in which to save plots.
                           default: None
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
                           default: False
        - plot_tc (bool) : if True, tuning curves are plotted for each ROI 
                           (causes errors on the clusters...)

    Returns:
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): general name under which the figures are saved
    """
    
    stimstr_pr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                    stimpar['bri_dir'], stimpar['bri_size'], 
                                    stimpar['gabk'], 'print')
    dendstr_pr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                    extrapar['datatype'], 'print')

    stimstr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                       stimpar['bri_dir'], stimpar['bri_size'], 
                                       stimpar['gabk'])
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                       extrapar['datatype'])

    datatype = extrapar['datatype']
    if datatype != 'roi':
        raise ValueError('Function only implemented for roi datatype.')

    gabfrs = gen_util.list_if_not(stimpar['gabfr'])
    if len(gabfrs) == 1:
        gabfrs = gabfrs * 2
    
    rand_str, rand_str_pr = '', ['', '']
    if tcurvpar['grp2'] == 'surp':
        surps = [0, 1]
        seq_types = ['reg', 'surp']
    elif tcurvpar['grp2'] in ['reg', 'rand']:
        surps = [0, 0]
        seq_types = ['reg', 'reg']
        if tcurvpar['grp2'] == 'rand':
            rand_str = '_rand'
            rand_str_pr = ['', ' samp']
    gabfr_letts = sess_str_util.gabfr_letters(gabfrs, surps)

    if extrapar['comb_gabs']:
        comb_gab_str = '_comb'
    else:
        comb_gab_str = ''

    # extract some info from sess_info (only one session)
    keys = ['mouse_ns', 'sess_ns', 'lines', 'layers', 'nrois']
    [mouse_n, sess_n, line, layer, nrois] = [sess_info[key][0] for key in keys]
    
    n_nan, n_nan_dff = [len(sess_info[key][0]) for key in 
                                ['nanrois', 'nanrois_dff']]
    sess_nrois = sess_gen_util.get_nrois(nrois, n_nan, n_nan_dff, 
                                     analyspar['remnans'], analyspar['fluor'])
    
    seq_info = ['{}{} {} seqs ({})'.format(sqt, ra, gf, nseqs) 
                                    for sqt, ra, gf, nseqs in 
                                    zip(seq_types, rand_str_pr, gabfr_letts, 
                                        tcurv_data['nseqs'])]
    
    gentitle = (u'Mouse {} - {} orientation tuning\n(sess {}, {} '
                 '{}{})').format(mouse_n, stimstr_pr, sess_n, line, layer, 
                                 dendstr_pr)
    gen_savename = '{}_tc_m{}_sess{}_{}_{}{}{}'.format(datatype, mouse_n, 
                    sess_n, stimstr, layer, dendstr, comb_gab_str)
    if savedir is None:
        savedir = os.path.join(figpar['dirs'][datatype], 
                               figpar['dirs']['tune_curv'], 
                               '{}v{}{}_seqs'.format(gabfr_letts[0], rand_str,
                                                     gabfr_letts[1]))

    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    if tcurvpar['prev']: # plot regression
        plot_tune_curve_regr(tcurv_data['vm_mean'], tcurv_data['vm_regr'], 
                             seq_info, gentitle, gen_savename, figpar, savedir)
    else:
        fulldir = plot_util.savefig(None, None, fulldir=savedir, 
                                    print_dir=False, **figpar['save'])

    if plot_tc:
        if not tcurvpar['prev']: # send None values if not using previous analysis
            tcurv_data = copy.deepcopy(tcurv_data)
            tcurv_data ['vm_pars'] = [None] * len(tcurv_data['data'])
            tcurv_data ['hist_pars'] = [None] * len(tcurv_data['data'])
        if parallel:
            n_jobs = min(multiprocessing.cpu_count(), len(tcurv_data['data']))
            fulldir = Parallel(n_jobs=n_jobs)(delayed(plot_roi_tune_curves)
                    (tcurv_data['oris'], roi_data, n, sess_nrois, seq_info, 
                    tcurv_data['vm_pars'][n], tcurv_data['hist_pars'][n], 
                    analyspar['fluor'], extrapar['comb_gabs'], gentitle, 
                    gen_savename, figpar, savedir)
                    for n, roi_data in enumerate(tcurv_data['data']))[0]
        else:
            for n, roi_data in enumerate(tcurv_data['data']):
                fulldir = plot_roi_tune_curves(tcurv_data['oris'], roi_data, 
                                n, sess_nrois, seq_info, 
                                tcurv_data['vm_pars'][n], 
                                tcurv_data['hist_pars'][n],  
                                analyspar['fluor'], extrapar['comb_gabs'], 
                                gentitle, gen_savename, figpar, savedir)
    else:
        print('Not plotting tuning curves.')

    return fulldir, gen_savename


#############################################
def plot_posori_resp(analyspar, sesspar, stimpar, extrapar, sess_info, 
                     posori_data, figpar=None, savedir=None):
    """
    plot_posori_resp(analyspar, sesspar, stimpar, extrapar, sess_info, 
                     posori_data)

    Plots responses according to position and orientation for a specific 
    session.

    Required args:
        - analyspar (dict)  : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)    : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)    : dictionary with keys of StimPar namedtuple
        - extrapar (dict)   : dictionary containing additional analysis 
                              parameters
            ['analysis'] (str): analysis type (e.g., 'o')
            ['datatype'] (str): datatype (e.g., 'roi')
        - sess_info (dict) : dictionary containing information from each
                             session (one first session used) 
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - posori_data (dict): position and orientation responses
            ['oris'] (list)     : stimulus mean orientations
            ['roi_stats'] (list): ROI statistics, structured as 
                                    mean orientation x gaborframe x stats x ROI
            ['nseqs'] (list)    : number of sequences structured as 
                                    mean orientation x gaborframe

    Optional args:
        - figpar (dict)  : dictionary containing the following figure parameter 
                           dictionaries
                           default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
            ['mng']  (dict): dictionary with parameters to manage matplotlib
        - savedir (str)  : path of directory in which to save plots.
                           default: None

    Returns:
        - fulldir (str) : final name of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
        - savename (str): name under which the figure is saved
    """

    stimstr_pr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                    stimpar['bri_dir'], stimpar['bri_size'], 
                                    stimpar['gabk'], 'print')
    statstr_pr = sess_str_util.stat_par_str(analyspar['stats'], 
                                    analyspar['error'], 'print')
    dendstr_pr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                    extrapar['datatype'], 'print')
    
    stimstr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                       stimpar['bri_dir'], stimpar['bri_size'], 
                                       stimpar['gabk'])
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                       extrapar['datatype'])

    datatype = extrapar['datatype']
    if datatype != 'roi':
        raise ValueError('Function only implemented for roi datatype.')

    # extract some info from sess_info (only one session)
    keys = ['mouse_ns', 'sess_ns', 'lines', 'layers']
    [mouse_n, sess_n, line, layer] = [sess_info[key][0] for key in keys]
    
    # n_rois info not included
    # n_rois = sess_info['n_rois'][0]
    # n_nan, n_nan_dff = [len(sess_info[key][0]) for key in 
    #                             ['nanrois', 'nanrois_dff']]

    title = (u'Mouse {} - {} {} location and orientation responses across '
              'seqs\n(sess {}, {} {}{})').format(mouse_n, stimstr_pr, 
                                    statstr_pr, sess_n, line, layer, dendstr_pr)
    savename = '{}_posori_m{}_sess{}_{}_{}{}'.format(datatype, mouse_n, sess_n, 
                                                   stimstr, layer, dendstr)
    if savedir is None:
        savedir = os.path.join(figpar['dirs'][datatype], 
                               figpar['dirs']['posori'])

    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()
    figpar['init']['n_subplots'] = 1
    figpar['init']['subplot_wid'] *= 2 

    fig, ax = plot_util.init_fig(**figpar['init'])
    sub_ax = ax[0, 0]

    n_oris = len(posori_data['roi_stats'])
    n_gabfr = len(posori_data['roi_stats'][0])
    n_rois = len(posori_data['roi_stats'][0][0][0])

    if n_gabfr != 5:
        raise ValueError('Expected data for 5 gabor frames.')

    center_pos, dot_pos, _ = plot_util.get_barplot_xpos(n_oris, n_gabfr, 
                                                        in_grp=2.5, barw=0.3)

    all_labs = []
    tick_pos = []
    roi_cols = [None] * n_rois
    for o, ori_data in enumerate(posori_data['roi_stats']):
        # ROI x stats x gab
        ori_data = np.transpose(ori_data, (2, 1, 0))
        xpos = dot_pos[o] 
        for r, roi_data in enumerate(ori_data):
            plot_util.plot_errorbars(sub_ax, roi_data[0], roi_data[1:], 
                                     x=xpos, title=title, col = roi_cols[r], 
                                     xticks='None')
            roi_cols[r] = sub_ax.lines[-1].get_color()
        labels = ['A', 'B', 'C', 'D', 'E']
        labels = ['{}\n({})'.format(lab, nseqs) for lab, nseqs 
                        in zip(labels, posori_data['nseqs'][o])]
        all_labs.extend(labels)
        tick_pos.extend(xpos)

    deg = u'\u00B0'
    oris = [u'{} {}'.format(ori, deg) for ori in posori_data['oris']]
    plot_util.add_labels(sub_ax, oris, center_pos, t_hei=-0.18, col='k')

    sub_ax.set_xticks(tick_pos)
    sub_ax.set_xticklabels(all_labs)

    sess_plot_util.add_axislabels(sub_ax, fluor=analyspar['fluor'], area=True,
                                  x_ax='', datatype=datatype)

    fulldir = plot_util.savefig(fig, savename, savedir, print_dir=False, 
                                **figpar['save'])

    return fulldir, savename

