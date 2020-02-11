"""
pup_analysis_plots.py

This script contains functions to plot results ofpupil analyses on 
specific sessions (pup_analys.py) from dictionaries.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import copy
import os

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st

from sess_util import sess_gen_util, sess_plot_util, sess_str_util
from util import file_util, gen_util, math_util, plot_util


#############################################
def plot_from_dict(dict_path, plt_bkend=None, fontdir=None, parallel=False):
    """
    plot_from_dict(dict_path)

    Plots data from dictionaries containing analysis parameters and results.

    Required args:
        - dict_path (str): path to dictionary to plot data from
    
    Optional_args:
        - plt_bkend (str): mpl backend to use for plotting (e.g., 'agg')
                           default: None
        - fontdir (str)  : path to directory where additional fonts are stored
                           default: None
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
                           default: False
    """

    print('\nPlotting from dictionary: {}'.format(dict_path))
    
    figpar = sess_plot_util.init_figpar(plt_bkend=plt_bkend, fontdir=fontdir)
    plot_util.manage_mpl(cmap=False, **figpar['mng'])

    info = file_util.loadfile(dict_path)
    savedir = os.path.dirname(dict_path)

    analysis = info['extrapar']['analysis']

    # 0. Plots the correlation between pupil and roi/run changes for each 
    # session
    if analysis == 'c': # difference correlation
        plot_pup_diff_corr(figpar=figpar, savedir=savedir, **info)
    
    # difference correlation per ROI between stimuli
    elif analysis == 'r': 
        plot_pup_roi_stim_corr(figpar=figpar, savedir=savedir, **info)

    else:
        print('No plotting function for analysis {}'.format(analysis))


#############################################
def plot_pup_diff_corr(analyspar, sesspar, stimpar, extrapar, 
                       sess_info, corr_data, figpar=None, savedir=None):
    """
    plot_pup_diff_corr(analyspar, sesspar, stimpar, extrapar, 
                       sess_info, corr_data)

    From dictionaries, plots correlation between surprise-locked changes in 
    pupil diameter and running or ROI data for each session.

    Required args:
        - analyspar (dict)    : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)      : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)      : dictionary with keys of StimPar namedtuple
        - extrapar (dict)     : dictionary containing additional analysis 
                                parameters
            ['analysis'] (str): analysis type (e.g., 'c')
            ['datatype'] (str): datatype (e.g., 'run', 'roi')
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
        - corr_data (dict)    : dictionary containing data to plot:
            ['corrs'] (list): list of correlation values between pupil and 
                              running or ROI differences for each session
            ['diffs'] (list): list of differences for each session, structured
                                  as [pupil, ROI/run] x trials x frames
    
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

    sessstr = sess_str_util.sess_par_str(sesspar['sess_n'], stimpar['stimtype'], 
                                         sesspar['layer'], stimpar['bri_dir'],
                                         stimpar['bri_size'], stimpar['gabk']) 
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                         extrapar['datatype'])
    
    datatype = extrapar['datatype']
    datastr = sess_str_util.datatype_par_str(datatype)

    if datatype == 'roi':
        label_str = sess_str_util.fluor_par_str(analyspar['fluor'], 
                                                str_type='print')
        full_label_str = u'{}, {} across ROIs'.format(label_str, 
                                                      analyspar['stats'])
    elif datatype == 'run':
        label_str = datastr
        full_label_str = datastr
    
    lab_app = ' ({} over {}/{} sec)'.format(analyspar['stats'], stimpar['pre'], 
                                            stimpar['post'])

    print('Plotting pupil vs {} changes.'.format(datastr))
    
    delta = '\u0394'

    # extract some info from sess_info
    keys = ['mouse_ns', 'sess_ns', 'lines', 'layers', 'nrois']
    [mouse_ns, sess_ns, lines, layers, nrois] = [sess_info[key] for key in keys]

    n_sess = len(mouse_ns)
    nanroi_vals = [sess_info['nanrois'], sess_info['nanrois_dff']]
    [n_nan, n_nan_dff] = [[len(val[i]) for i in range(n_sess)] 
                                       for val in nanroi_vals]

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()
    figpar['init']['sharex'] = False
    figpar['init']['sharey'] = False
    figpar['init']['ncols'] = n_sess
    
    fig, ax = plot_util.init_fig(2 * n_sess, **figpar['init'])
    suptitle = (u'Relationship between pupil diam and {} changes locked '
                 'to surprise'.format(datastr))
    fig.suptitle(suptitle)
    for i, sess_diffs in enumerate(corr_data['diffs']):
        sub_axs = ax[:, i]
        remnans = analyspar['remnans'] * (datatype == 'roi')
        sess_nrois = sess_gen_util.get_nrois(nrois[i], n_nan[i], n_nan_dff[i], 
                                             remnans, analyspar['fluor'])
        title = (u'Mouse {} - {} {}\n(sess {}, {} {}{}, '
                  '(n={}))').format(mouse_ns[i], statstr_pr, stimstr_pr, 
                     sess_ns[i], lines[i], layers[i], dendstr_pr, sess_nrois)
        
        # top plot: correlations
        corr = 'Corr = {:.2f}'.format(corr_data['corrs'][i])
        sub_axs[0].plot(sess_diffs[0], sess_diffs[1], marker='.', 
                        linestyle='None', label=corr)
        sub_axs[0].set_title(title)
        sub_axs[0].set_xlabel(u'{} pupil diam{}'.format(delta, lab_app))
        if i == 0:
            sub_axs[0].set_ylabel(u'{} {}\n{}'.format(delta, full_label_str, 
                                                      lab_app))
        sub_axs[0].legend()
        
        # bottom plot: differences across occurrences
        data_lab = u'{} {}'.format(delta, label_str)   
        pup_lab = u'{} pupil diam'.format(delta)
        cols = []
        scaled = []
        for d, lab in enumerate([pup_lab, data_lab]):
            scaled.append(math_util.scale_data(np.asarray(sess_diffs[d]), 
                                               sc_type='min_max')[0])
            art, = sub_axs[1].plot(scaled[-1], marker='.')
            cols.append(sub_axs[-1].lines[-1].get_color())
            if i == n_sess - 1: # only for last graph
                art.set_label(lab)
                sub_axs[1].legend()
        sub_axs[1].set_xlabel('Surprise occurrence')
        if i == 0:
            sub_axs[1].set_ylabel(u'{} response locked\nto surprise '
                                   '(scaled)'.format(delta))
        # shade area between lines
        plot_util.plot_btw_traces(sub_axs[1], scaled[0], scaled[1], 
                                  col=cols, alpha=0.4)

    if savedir is None:
        savedir = os.path.join(figpar['dirs'][datatype],figpar['dirs']['pupil'])

    savename = ('{}_diff_corr_{}{}').format(datatype, sessstr, dendstr)

    fulldir = plot_util.savefig(fig, savename, savedir, **figpar['save'])

    return fulldir, savename                              


#############################################
def plot_pup_roi_stim_corr(analyspar, sesspar, stimpar, extrapar, 
                           sess_info, corr_data, figpar=None, savedir=None):
    """
    plot_pup_roi_stim_corr(analyspar, sesspar, stimpar, extrapar, 
                           sess_info, corr_data)

    From dictionaries, plots correlation between surprise-locked changes in 
    pupil diameter and each ROI, for gabors versus bricks responses for each 
    session.
    
    Required args:
        - analyspar (dict)    : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)      : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)      : dictionary with keys of StimPar namedtuple
        - extrapar (dict)     : dictionary containing additional analysis 
                                parameters
            ['analysis'] (str): analysis type (e.g., 'r')
            ['datatype'] (str): datatype (e.g., 'roi')
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
        - corr_data (dict)    : dictionary containing data to plot:
            ['stim_order'] (list): ordered list of stimtypes
            ['roi_corrs'] (list) : nested list of correlations between pupil 
                                   and ROI responses changes locked to 
                                   surprise, structured as 
                                       session x stimtype x ROI
            ['corrs'] (list)     : list of correlation between stimtype
                                   correlations for each session
    
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

    stimstr_prs = []
    for stimtype in corr_data['stim_order']:
        stimstr_prs.append(sess_str_util.stim_par_str(stimtype, 
                                    stimpar['bri_dir'], stimpar['bri_size'], 
                                    stimpar['gabk'], 'print'))
    dendstr_pr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                            extrapar['datatype'], 'print')

    sessstr = 'sess{}_{}'.format(sesspar['sess_n'], sesspar['layer']) 
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                         extrapar['datatype'])

    label_str = sess_str_util.fluor_par_str(analyspar['fluor'], 
                                            str_type='print')
    lab_app = ' ({} over {}/{} sec)'.format(analyspar['stats'], stimpar['pre'], 
                                            stimpar['post'])

    print('Plotting pupil-ROI difference correlations for '
          '{} vs {}.'.format(*corr_data['stim_order']))

    # extract some info from sess_info
    keys = ['mouse_ns', 'sess_ns', 'lines', 'layers', 'nrois']
    [mouse_ns, sess_ns, lines, layers, nrois] = [sess_info[key] for key in keys]

    n_sess = len(mouse_ns)
    nanroi_vals = [sess_info['nanrois'], sess_info['nanrois_dff']]
    [n_nan, n_nan_dff] = [[len(val[i]) for i in range(n_sess)] 
                                       for val in nanroi_vals]

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()
    figpar['init']['sharex'] = True
    figpar['init']['sharey'] = True
    
    fig, ax = plot_util.init_fig(n_sess, **figpar['init'])
    suptitle = (u'Relationship between pupil diam and {} changes locked to '
                 'surprise{} for each ROI for {} vs '
                 '{}'.format(label_str, lab_app, *corr_data['stim_order']))
    fig.suptitle(suptitle)
    for i, sess_roi_corrs in enumerate(corr_data['roi_corrs']):
        sub_ax = plot_util.get_subax(ax, i)
        sess_nrois = sess_gen_util.get_nrois(nrois[i], n_nan[i], n_nan_dff[i], 
                                   analyspar['remnans'], analyspar['fluor'])
        title = (u'Mouse {} (sess {}, {} {}{}, '
                  '(n={}))').format(mouse_ns[i], sess_ns[i], lines[i], 
                                    layers[i], dendstr_pr, sess_nrois)
        
        # top plot: correlations
        corr = 'Corr = {:.2f}'.format(corr_data['corrs'][i])
        sub_ax.plot(sess_roi_corrs[0], sess_roi_corrs[1], marker='.', 
                    linestyle='None', label=corr)
        sub_ax.set_title(title)
        sub_ax.set_xlabel('{} correlations'.format(stimstr_prs[0]))
        if i == 0:
            sub_ax.set_ylabel('{} correlations'.format(stimstr_prs[1]))
        sub_ax.legend()

    if savedir is None:
        savedir = os.path.join(figpar['dirs']['roi'],figpar['dirs']['pupil'])

    savename = ('roi_diff_corrbyroi_{}{}').format(sessstr, dendstr)

    fulldir = plot_util.savefig(fig, savename, savedir, **figpar['save'])

    return fulldir, savename                           

