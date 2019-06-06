"""
modif_analysis_plots.py

This script is used to modify plots on the fly for ROI and running analyses 
from dictionaries, e.g. for presentations or papers.

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
from plot_fcts import roi_analysis_plots as roi_plots


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
                           (causes errors on the clusters...)
    """

    figpar = sess_plot_util.init_figpar(plt_bkend=plt_bkend, fontdir=fontdir)
    plot_util.manage_mpl(cmap=False, **figpar['mng'])

    plt.rcParams['figure.titlesize'] = 'xx-large'
    plt.rcParams['axes.titlesize'] = 'xx-large'

    info = file_util.loadfile(dict_path)
    savedir = os.path.dirname(dict_path)

    analysis = info['extrapar']['analysis']

    # 1. Plot average traces by quintile x surprise for each session 
    if analysis == 't': # traces
        plot_traces_by_qu_surp_sess(figpar=figpar, savedir=savedir, **info)

    # 2. Plot average traces by quintile, locked to surprise for each session 
    elif analysis == 'l': # surprise locked traces
        plot_traces_by_qu_lock_sess(figpar=figpar, savedir=savedir, **info)

    # 4. Plot autocorrelations
    elif analysis == 'a': # autocorr
        plot_autocorr(figpar=figpar, savedir=savedir, **info)

  # 6. Plot colormaps and traces for orientations/directions
    elif analysis == 'o': # colormaps
        plot_oridirs(figpar=figpar, savedir=savedir, parallel=parallel, **info)


    else:
        print('No plotting function for analysis {}'.format(analysis))


#############################################
def plot_traces_by_qu_surp_sess(analyspar, sesspar, stimpar, extrapar, 
                                quintpar, sess_info, trace_stats, figpar=None, 
                                savedir=None):
    """
    plot_traces_by_qu_surp_sess(analyspar, sesspar, stimpar, extrapar, 
                                quintpar, sess_info, trace_stats)

    From dictionaries, plots traces by quintile/surprise with each session in a 
    separate subplot.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict)  : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)    : dictionary with keys of SessPar namedtuple
        - stimpar (dict)    : dictionary with keys of StimPar namedtuple
        - extrapar (dict)   : dictionary containing additional analysis 
                              parameters
            ['analysis'] (str): analysis type (e.g., 't')
            ['datatype'] (str): datatype (e.g., 'run', 'roi')
        - quintpar (dict)   : dictionary with keys of QuintPar namedtuple
        - sess_info (dict)  : dictionary containing information from each
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
        - trace_stats (dict): dictionary containing trace stats information
            ['x_ran'] (array-like)     : time values for the 2p frames
            ['all_stats'] (list)       : list of 4D arrays or lists of trace 
                                         data statistics across ROIs, 
                                         structured as:
                                            surp x quintiles x
                                            stats (me, err) x frames
            ['all_counts'] (array-like): number of sequences, structured as:
                                                sess x surp x quintiles
                
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
 
    stimstr_pr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                    stimpar['bri_dir'], stimpar['bri_size'],
                                    stimpar['gabk'], 'print')
    statstr_pr = sess_str_util.stat_par_str(analyspar['stats'], 
                                            analyspar['error'], 'print')

    sessstr = sess_str_util.sess_par_str(sesspar['sess_n'], stimpar['stimtype'],
                                         sesspar['layer'], stimpar['bri_dir'], 
                                         stimpar['bri_size'], stimpar['gabk'])
    
    datatype = extrapar['datatype']
    dimstr = sess_str_util.datatype_dim_str(datatype)

    # extract some info from sess_info
    keys = ['mouse_ns', 'sess_ns', 'lines', 'layers', 'nrois']
    [mouse_ns, sess_ns, lines, layers, nrois] = [sess_info[key] for key in keys]
    
    n_sess = len(mouse_ns)
    nanroi_vals = [sess_info['nanrois'], sess_info['nanrois_dff']]
    [n_nan, n_nan_dff] = [[len(val[i]) for i in range(n_sess)] 
                                       for val in nanroi_vals]

    x_ran      = np.asarray(trace_stats['x_ran'])
    all_stats  = [np.asarray(sessst) for sessst in trace_stats['all_stats']]
    all_counts = trace_stats['all_counts']

    cols, lab_cols = sess_plot_util.get_quint_cols(quintpar['n_quints'])
    alpha = np.min([0.4, 0.8/quintpar['n_quints']])

    surps = ['reg', 'surp']
    ev = 6
    if stimpar['stimtype'] == 'bricks':
        ev = 7

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    fig, ax = plot_util.init_fig(n_sess, **figpar['init'])
    for i in range(n_sess):
        sess_nrois = sess_gen_util.get_nrois(nrois[i], n_nan[i], n_nan_dff[i],
                                    analyspar['remnans'], analyspar['fluor'])
        sub_ax = plot_util.get_subax(ax, i)
        for s, [col, leg_ext] in enumerate(zip(cols, surps)):
            for q, qu_lab in enumerate(quintpar['qu_lab']):
                if qu_lab != '':
                    qu_lab = '{} '.format(qu_lab.capitalize())
                line, layer = '5', 'dendrites'
                if '23' in lines[i]:
                    line = '2/3'
                if 'soma' in layers[i]:
                    layer = 'somata'
                title=('M{} - layer {} {}'.format(mouse_ns[i], line, layer))
                # title=(u'Mouse {} - {} {} across {}\n(sess {}, '
                #        '{} {}, n={})').format(mouse_ns[i], stimstr_pr, 
                #                               statstr_pr, dimstr, sess_ns[i], 
                #                               lines[i], layers[i], sess_nrois)
                leg = None
                y_ax = ''
                if i == 0:
                    leg = '{}{}'.format(qu_lab, leg_ext)
                    y_ax = None
                # leg = '{}{} ({})'.format(qu_lab, leg_ext, all_counts[i][s][q])
                plot_util.plot_traces(sub_ax, x_ran, all_stats[i][s, q, 0], 
                                      all_stats[i][s, q, 1:], title, 
                                      col=col[q], alpha=alpha, label=leg, 
                                      xticks_ev=ev)
                sess_plot_util.add_axislabels(sub_ax, fluor=analyspar['fluor'], 
                                              datatype=datatype, y_ax=y_ax)

    if stimpar['stimtype'] == 'gabors': 
        sess_plot_util.plot_labels(ax, stimpar['gabfr'], 'both', 
                            pre=stimpar['pre'], post=stimpar['post'], 
                            cols=lab_cols, sharey=figpar['init']['sharey'])
    
    if savedir is None:
        savedir = os.path.join(figpar['dirs'][datatype], 
                               figpar['dirs']['surp_qu'])

    qu_str = '_{}q'.format(quintpar['n_quints'])
    if quintpar['n_quints'] == 1:
        qu_str = ''

    savename = '{}_av_{}{}'.format(datatype, sessstr, qu_str)
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar['save'])

    return fulldir, savename


#############################################
def plot_traces_by_qu_lock_sess(analyspar, sesspar, stimpar, extrapar, 
                                quintpar, sess_info, trace_stats, 
                                figpar=None, savedir=None):
    """
    plot_traces_by_qu_lock_sess(analyspar, sesspar, stimpar, extrapar, 
                                quintpar, sess_info, trace_stats)

    From dictionaries, plots traces by quintile, locked to transitions from 
    surprise to regular or v.v. with each session in a separate subplot.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict)  : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)    : dictionary with keys of SessPar namedtuple
        - stimpar (dict)    : dictionary with keys of StimPar namedtuple
        - extrapar (dict)   : dictionary containing additional analysis 
                              parameters
            ['analysis'] (str): analysis type (e.g., 'l')
            ['datatype'] (str): datatype (e.g., 'run', 'roi')
        - quintpar (dict)   : dictionary with keys of QuintPar namedtuple
        - sess_info (dict)  : dictionary containing information from each
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
        - trace_stats (dict): dictionary containing trace stats information
            ['x_ran'] (array-like)     : time values for the 2p frames
            ['all_stats'] (list)       : list of 4D arrays or lists of trace 
                                         data statistics across ROIs, 
                                         structured as:
                                            (surp_len x) quintiles x
                                            stats (me, err) x frames
            ['all_counts'] (array-like): number of sequences, structured as:
                                                sess x (surp_len x) quintiles
            ['lock'] (str)             : value to which segments are locked:
                                         'surp', 'reg' or 'surp_split'
            ['baseline'] (num)         : number of seconds used for baseline
            ['reg_stats'] (list)       : list of 3D arrays or lists of trace 
                                         data statistics across ROIs for
                                         regular sampled sequences, 
                                         structured as:
                                            quintiles (1) x stats (me, err) 
                                            x frames
            ['reg_counts'] (array-like): number of sequences corresponding to
                                         reg_stats, structured as:
                                            sess x quintiles (1)
            
            if data is by surp_len:
            ['surp_lens'] (list)       : number of consecutive segments for
                                         each surp_len, structured by session
                
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

    stimstr_pr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                    stimpar['bri_dir'], stimpar['bri_size'],
                                    stimpar['gabk'], 'print')
    statstr_pr = sess_str_util.stat_par_str(analyspar['stats'], 
                                            analyspar['error'], 'print')

    sessstr = sess_str_util.sess_par_str(sesspar['sess_n'], stimpar['stimtype'],
                                         sesspar['layer'], stimpar['bri_dir'], 
                                         stimpar['bri_size'], stimpar['gabk'])
    basestr = sess_str_util.base_par_str(trace_stats['baseline'])
    basestr_pr = sess_str_util.base_par_str(trace_stats['baseline'], 'print')

    datatype = extrapar['datatype']
    dimstr = sess_str_util.datatype_dim_str(datatype)

    # extract some info from sess_info
    keys = ['mouse_ns', 'sess_ns', 'lines', 'layers', 'nrois']
    [mouse_ns, sess_ns, lines, layers, nrois] = [sess_info[key] for key in keys]
    
    n_sess = len(mouse_ns)
    nanroi_vals = [sess_info['nanrois'], sess_info['nanrois_dff']]
    [n_nan, n_nan_dff] = [[len(val[i]) for i in range(n_sess)] 
                                       for val in nanroi_vals]

    x_ran      = np.asarray(trace_stats['x_ran'])
    all_stats  = [np.asarray(sessst) for sessst in trace_stats['all_stats']]
    reg_stats  = [np.asarray(regst) for regst in trace_stats['reg_stats']]
    all_counts = trace_stats['all_counts']
    reg_counts = trace_stats['reg_counts']

    lock  = trace_stats['lock']
    col_idx = 0
    if 'surp' in lock:
        lock = 'surp'
        col_idx = 1
    
    surp_lab, len_ext = '', ''
    surp_lens = [[None]] * n_sess
    if 'surp_lens' in trace_stats.keys():
        surp_lens = trace_stats['surp_lens']
        len_ext = '_bylen'
        if stimpar['stimtype'] == 'gabors':
            surp_lens = [[sl * 1.5/4 for sl in sls] for sls in surp_lens]
    
    if figpar is None:
        figpar = sess_plot_util.init_figpar()
    figpar = copy.deepcopy(figpar)
    figpar['init']['subplot_wid'] = 6.5
    ev = 21

    # RANGE TO PLOT
    st  = int(len(x_ran)*2.0/5)
    end = int(len(x_ran)*4.0/5)
    ev = 9

    fig, ax = plot_util.init_fig(n_sess, **figpar['init'])
    for i, (stats, counts) in enumerate(zip(all_stats, all_counts)):
        sub_ax = plot_util.get_subax(ax, i)
        sess_nrois = sess_gen_util.get_nrois(nrois[i], n_nan[i], n_nan_dff[i],
                                     analyspar['remnans'], analyspar['fluor'])
        line, layer = '5', 'dendrites'
        if '23' in lines[i]:
            line = '2/3'
        if 'soma' in layers[i]:
            layer = 'somata'

        title=('M{} - layer {} {}'.format(mouse_ns[i], line, layer))
        # title=(u'Mouse {} - {} {} {} locked across {}{}\n(sess {}, {} {}, '
        #         'n={})').format(mouse_ns[i], stimstr_pr, statstr_pr, lock, 
        #                         dimstr, basestr_pr, sess_ns[i], lines[i], 
        #                         layers[i], sess_nrois)
        y_ax = ''
        if i == 0:
            y_ax = None

        sess_plot_util.add_axislabels(sub_ax, fluor=analyspar['fluor'], 
                                      datatype=datatype, y_ax=y_ax)
        plot_util.add_bars(sub_ax, hbars=0)
        n_lines = quintpar['n_quints'] * len(surp_lens[i])
        try: 
            cols = sess_plot_util.get_quint_cols(n_lines)[0][col_idx]
        except:
            cols = [None] * n_lines
        alpha      = np.min([0.4, 0.8/n_lines])
        if stimpar['stimtype'] == 'gabors':
            sess_plot_util.plot_gabfr_pattern(sub_ax, x_ran[st:end], 
                                              bars_omit=[0] + surp_lens[i])
        # plot regular data
        if reg_stats[i].shape[0] != 1:
            raise ValueError(('Expected only one quintile for reg_stats.'))
        
        leg = None
        if i == 0:
            leg = 'reg'

        # leg = 'reg (no lock) ({})'.format(reg_counts[i][0])
        plot_util.plot_traces(sub_ax, x_ran[st:end], reg_stats[i][0][0, st:end], 
                              reg_stats[i][0][1:, st:end], alpha=alpha, 
                              label=leg, alpha_line=0.8, col='darkgray')
        n = 0 # count lines plotted
        for s, surp_len in enumerate(surp_lens[i]):
            if surp_len is not None:
                counts, stats = all_counts[i][s], all_stats[i][s]                
                surp_lab = 'surp len {}'.format(surp_len)
            else:
                # surp_lab = 'surp lock'
                surp_lab = 'surp'
            for q, qu_lab in enumerate(quintpar['qu_lab']):
                if qu_lab != '':
                    qu_lab = '{} '.format(qu_lab.capitalize())
                lab = '{}{}'.format(qu_lab, surp_lab)
                if n == 2 and cols[n] is None:
                    sub_ax.plot([], []) # to advance the color cycle (past gray)
                #leg = '{} ({})'.format(lab, counts[q])
                leg = None
                if i == 0:
                    leg = lab
                plot_util.plot_traces(sub_ax, x_ran[st:end], 
                                      stats[q][0, st:end], stats[q][1:, st:end], 
                                      title, alpha=alpha, label=leg, 
                                      xticks_ev=ev, alpha_line=0.8, 
                                      col=cols[n])
                n += 1
            if surp_len is not None:
                plot_util.add_bars(sub_ax, hbars=surp_len, 
                                   col=sub_ax.lines[-1].get_color(), alpha=1)
    
    if savedir is None:
        savedir = os.path.join(figpar['dirs'][datatype], 
                               figpar['dirs']['surp_qu'], 
                               '{}_lock'.format(lock), basestr.replace('_', ''))

    qu_str = '_{}q'.format(quintpar['n_quints'])
    if quintpar['n_quints'] == 1:
        qu_str = ''
 
    savename = '{}_av_{}lock{}{}_{}{}'.format(datatype, lock, len_ext, basestr, 
                                              sessstr, qu_str)
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar['save'])

    return fulldir, savename


#############################################
def plot_autocorr(analyspar, sesspar, stimpar, extrapar, autocorrpar, 
                  sess_info, autocorr_data, figpar=None, savedir=None):
    """
    plot_autocorr(analyspar, sesspar, stimpar, extrapar, autocorrpar, 
                  sess_info, autocorr_data)

    From dictionaries, plots autocorrelation during stimulus blocks.

    Required args:
        - analyspar (dict)    : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)      : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)      : dictionary with keys of StimPar namedtuple
        - extrapar (dict)     : dictionary containing additional analysis 
                                parameters
            ['analysis'] (str): analysis type (e.g., 'a')
            ['datatype'] (str): datatype (e.g., 'run', 'roi')
        - autocorrpar (dict)  : dictionary with keys of AutocorrPar namedtuple
        - sess_info (dict)    : dictionary containing information from each
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
        - autocorr_data (dict): dictionary containing data to plot:
            ['xrans'] (list): list of lag values in seconds for each session
            ['stats'] (list): list of 3D arrays (or nested lists) of
                              autocorrelation statistics, structured as:
                                     sessions stats (me, err) 
                                     x ROI or 1x and 10x lag 
                                     x lag
    
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
    
    sessstr = sess_str_util.sess_par_str(sesspar['sess_n'], stimpar['stimtype'], 
                                         sesspar['layer'], stimpar['bri_dir'],
                                         stimpar['bri_size'], stimpar['gabk']) 
 
    datatype = extrapar['datatype']
    if datatype == 'roi':
        fluorstr_pr = sess_str_util.fluor_par_str(analyspar['fluor'], 
                                                 str_type='print')
        title_str = u'{} autocorrelation'.format(fluorstr_pr)
        if not autocorrpar['byitem']:
            title_str = '{} across ROIs'.format(title_str) 
    elif datatype == 'run':
        datastr = sess_str_util.datatype_par_str(datatype)
        title_str = u'{} autocorrelation'.format(datastr)

    if stimpar['stimtype'] == 'gabors':
        seq_bars = [-1.5, 1.5] # light lines
    else:
        seq_bars = [-1.0, 1.0] # light lines

    # extract some info from sess_info
    keys = ['mouse_ns', 'sess_ns', 'lines', 'layers', 'nrois']
    [mouse_ns, sess_ns, lines, layers, nrois] = [sess_info[key] for key in keys]
    
    n_sess = len(mouse_ns)
    nanroi_vals = [sess_info['nanrois'], sess_info['nanrois_dff']]
    [n_nan, n_nan_dff] = [[len(val[i]) for i in range(n_sess)] 
                                       for val in nanroi_vals]

    xrans = autocorr_data['xrans']
    stats = [np.asarray(stat) for stat in autocorr_data['stats']]

    lag_s = autocorrpar['lag_s']
    xticks = np.linspace(-lag_s, lag_s, lag_s*2+1)
    yticks = np.linspace(0, 1, 6)

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    byitemstr = ''
    if autocorrpar['byitem']:
        byitemstr = '_byroi'

    fig, ax = plot_util.init_fig(n_sess, **figpar['init'])
    for i in range(n_sess):
        sub_ax = plot_util.get_subax(ax, i)
        sess_nrois = sess_gen_util.get_nrois(nrois[i], n_nan[i], n_nan_dff[i], 
                                    analyspar['remnans'], analyspar['fluor'])
        line, layer = '5', 'dendrites'
        if '23' in lines[i]:
            line = '2/3'
        if 'soma' in layers[i]:
            layer = 'somata'
        title=('M{} - layer {} {}'.format(mouse_ns[i], line, layer))
        # title = (u'Mouse {} - {} {} {}\n(sess {}, {} {}, '
        #           '(n={}))').format(mouse_ns[i], statstr_pr, stimstr_pr, 
        #                             title_str, sess_ns[i], lines[i], layers[i], 
        #                             sess_nrois)
        # transpose to ROI/lag x stats x series
        sess_stats = stats[i].transpose(1, 0, 2) 
        for s, sub_stats in enumerate(sess_stats):
            lab = None
            if i == 0:
                if not autocorrpar['byitem']:
                    lab = ['actual lag', '10x lag'][s]
            plot_util.plot_traces(sub_ax, xrans[i], sub_stats[0], 
                                  sub_stats[1:], xticks=xticks, yticks=yticks, 
                                  alpha=0.2, label=lab)
        plot_util.add_bars(sub_ax, hbars=seq_bars)
        sub_ax.set_ylim([0, 1])
        sub_ax.set_title(title)
        sub_ax.set_xlabel('Lag (s)')

    if savedir is None:
        savedir = os.path.join(figpar['dirs'][datatype], 
                               figpar['dirs']['autocorr'])

    savename = ('{}_autocorr{}_{}').format(datatype, byitemstr, sessstr)

    fulldir = plot_util.savefig(fig, savename, savedir, **figpar['save'])

    return fulldir, savename


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
            ['n_segs'] (dict): dictionary containing number of segs for each
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

    stimstr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                       stimpar['bri_dir'], stimpar['bri_size'], 
                                       stimpar['gabk'])

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
    
    # suptitle = (u'Mouse {} - {} {} across {}{}\n(sess {}, {} {}, '
    #              'n={})').format(mouse_n, stimstr_pr, statstr_pr, dimstr,
    #                              qu_str_pr, sess_n, line, layer, sess_nrois)
    line_str, layer_str = '5', 'dendrites'
    if '23' in line:
        line_str = '2/3'
    if 'soma' in layer:
        layer_str = 'somata'
    suptitle=('M{} - layer {} {}'.format(mouse_n, line_str, layer_str))

    savename = '{}_tr_m{}_sess{}{}_{}_{}'.format(datatype, mouse_n, sess_n, 
                                                 qu_str, stimstr, layer)
    
    fig, ax = plot_util.init_fig(len(oridirs), **figpar['init'])
    for o, od in enumerate(oridirs):
        cols = []
        for surp in surps: 
            sub_ax = plot_util.get_subax(ax, o)
            key = '{}_{}'.format(surp, od)
            stimtype_str_pr = stimpar['stimtype'][:-1].capitalize()
            title_tr = u'{} traces ({}{})'.format(stimtype_str_pr, od, deg)
            lab = '{} (n={})'.format(surp, tr_data['n_segs'][key])
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
def plot_oridir_colormap(fig_type, analyspar, stimpar, quintpar, tr_data, 
                         sess_info, figpar=None, savedir=None, print_dir=True):
    """
    plot_oridir_colormap(fig_type, analyspar, stimpar, quintpar, tr_data, 
                         sess_info)

    From dictionaries, plots average activity across gabor orientations or 
    brick directions per ROI for a single session and optionally a single 
    quintile. (Single figure type) 

    Required args:
        - fig_type (str)  : type of figure to plot, i.e., 'byplot', 'byreg', 
                            'byfir' or 'by{}{}' (ori/dir, deg)
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
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
            ['n_segs'] (dict)    : dictionary containing number of segs for 
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

    stimstr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                       stimpar['bri_dir'], stimpar['bri_size'], 
                                       stimpar['gabk'])

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
    # gentitle = (u'Mouse {} - {} {} across seqs colormaps{}\n(sess {}, '
    #              '{} {})').format(mouse_n, stimstr_pr, statstr_pr,  
    #                               qu_str_pr, sess_n, line, layer)
    line_str, layer_str = '5', 'dendrites'
    if '23' in line:
        line_str = '2/3'
    if 'soma' in layer:
        layer_str = 'somata'
    gentitle=('M{} - layer {} {}'.format(mouse_n, line_str, layer_str))
    
    gen_savename = 'roi_cm_m{}_sess{}{}_{}_{}'.format(mouse_n, sess_n, 
                                                      qu_str, stimstr, layer)

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
    scaled_sort_me = roi_plots.scale_sort_trace_data(tr_data, fig_type, surps, 
                                                     oridirs)
    fig, ax = plot_util.init_fig(len(oridirs) * len(surps), **figpar['init'])

    for o, od in enumerate(oridirs):
        for s, surp in enumerate(surps):    
            sub_ax = ax[s][o]
            key = '{}_{}'.format(surp, od)
            title = u'{} segs ({}{}) (n={})'.format(surp.capitalize(), od, 
                                                deg_pr, tr_data['n_segs'][key])
            x_ax = None
            if s == 0:
                x_ax = ''
            sess_plot_util.add_axislabels(sub_ax, fluor=analyspar['fluor'], 
                                       x_ax=x_ax, y_ax='ROIs', datatype='roi')
            im = plot_util.plot_colormap(sub_ax, scaled_sort_me[key], 
                                    title=title, cmap=cmap,
                                    xran=[stimpar['pre'], stimpar['post']])
    
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
            ['n_segs'] (dict)    : dictionary containing number of segs for 
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
                         (fig_type, analyspar, stimpar, quintpar, tr_data,
                          sess_info, figpar, savedir, (f == fig_last)) 
                          for f, fig_type in enumerate(fig_types)) 
        fulldir = fulldirs[-1]
    else:
        for f, fig_type in enumerate(fig_types):
            print_dir = (f == fig_last)
            fulldir = plot_oridir_colormap(fig_type, analyspar, stimpar, 
                                           quintpar, tr_data, sess_info, 
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
            ['n_segs'] (dict)    : dictionary containing number of segs for each
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

