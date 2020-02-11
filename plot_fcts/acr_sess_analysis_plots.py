"""
acr_sess_analysis_plots.py

This script contains functions to plot results of across sessions analyses 
(acr_sess_analys.py) from dictionaries.

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
        - parallel (bool): if True, some of the plotting is parallelized across 
                           CPU cores
                           default: False
    """

    print('\nPlotting from dictionary: {}'.format(dict_path))
    
    figpar = sess_plot_util.init_figpar(plt_bkend=plt_bkend, fontdir=fontdir)
    plot_util.manage_mpl(cmap=False, **figpar['mng'])

    info = file_util.loadfile(dict_path)
    savedir = os.path.dirname(dict_path)

    analysis = info['extrapar']['analysis']

    # 0. Plots the difference between surprise and regular across sessions
    if analysis == 's': 
        plot_surp_area_diff(figpar=figpar, savedir=savedir, **info)

    # 1. Plots the difference between surprise and regular locked to surprise
    # across sessions
    elif analysis == 'l':
        plot_lock_area_diff(figpar=figpar, savedir=savedir, **info)

    # 2. Plots the surprise and regular traces across sessions
    elif analysis == 't':
        plot_surp_traces(figpar=figpar, savedir=savedir, **info)

    # 3. Plots the surprise and regular locked to surprise traces
    # across sessions
    elif analysis == 'r':
        plot_lock_traces(figpar=figpar, savedir=savedir, **info)

    # 3. Plots the difference between surprise and regular locked to surprise
    # across sessions
    elif analysis == 'u':
        plot_surp_latency(figpar=figpar, savedir=savedir, **info)

    else:
        print('    No plotting function for analysis {}'.format(analysis))


#############################################
def get_linlay_idx(linlay_ord, line='L2/3', layer='soma', verbose=False, 
                   newline=False):
    """
    get_linlay_idx(linlay_ord)


    Required args:
        - linlay_ord (list): ordered list of line/layer combinations formatted 

    Optional args:
        - line (str)    : line (e.g., L2/3, L5)
                          default: 'L2/3'
        - layer (str)   : layer (e.g., soma, dend)
                          default: 'soma'
        - verbose (bool): if True and no data is found, this is printed to the 
                          console
                          default: False
        - newline (bool): if True, text is printed on a new line
                          default: False

    Returns:
        - l_idx (int or None): line/layer combination index in linlay_ord or 
                               None if not found
    """

    data_name = '{} {}'.format(line, layer[:4]) # get data name
    if data_name not in linlay_ord:
        add = ''
        if newline:
            add = '\n'
        if verbose:
            print('{}No data for {}.'.format(add, data_name))
        l_idx = None
    else:
        l_idx = linlay_ord.index(data_name)

    return l_idx


#############################################
def plot_area_diff_per_mouse(sub_ax, mouse_diff_st, sess_info, sess_ns=None, 
                             col=None, use_lab=True):
    """
    plot_area_diff_per_mouse(sub_ax, mouse_diff_st, sess_info)

    Plots area difference statistics for each mouse.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - mouse_diff_st (3D array) : difference statistics across ROIs or seqs, 
                                     structured as mouse x session x stats
        - sess_info (list)         : list of dictionaries for each mouse 
                                     containing information from each session, 
                                     with None for missing sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists

    Optional args:
        - sess_ns (array-like): session numbers for each session
                                default: None
        - col (str)           : plotting color
                                default: None
        - use_lab (bool)      : if True, label with mouse numbers is added
                                default: True
    """

    mouse_diff_st = np.asarray(mouse_diff_st)
    
    if sess_ns is None:
        sess_ns = range(mouse_diff_st.shape[0])
    sess_ns = np.asarray(sess_ns)

    lab = 'M'
    for m, m_info in enumerate(sess_info):
        lab_use = None
        
        # get this mouse's ns
        mouse_ns = set([n for n in m_info['mouse_ns'] if n is not None])
        if len(mouse_ns) != 1:
            raise ValueError('Should not be more than 1 mouse.')
        mouse_n = list(mouse_ns)[0]

        # extend label or create final version
        if use_lab:
            if m != len(sess_info) - 1:
                lab = '{}{}, '.format(lab, mouse_n)
            else: # if label to be included
                lab_use = '{}{}'.format(lab, mouse_n)

        # get non NaNs
        keep_idx = np.where(np.isfinite(mouse_diff_st[m, :, 0]))[0]
        
        if sess_ns is None:
            sess_ns = np.asarray(range(len(mouse_diff_st[m])))
        elif len(sess_ns) != len(mouse_diff_st[m]):
            raise ValueError('Not as many session numbers as sessions.')

        plot_util.plot_errorbars(sub_ax, mouse_diff_st[m, keep_idx, 0], 
                  mouse_diff_st[m, keep_idx, 1:].T, sess_ns[keep_idx], col=col, 
                  label=lab_use, alpha=0.6)


#############################################
def plot_area_diff_stats(sub_ax, all_diff_st, sess_ns=None, mouse_mes=None, 
                         col=None):
    """
    plot_area_diff_stats(sub_ax, all_diff_st, sess_info)

    Plots area differences statistics across mice.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - all_diff_st (2D array)   : difference statistics across mice, 
                                     structured as session x stats

    Optional args:
        - sess_ns (array-like)  : session numbers for each session
                                  default: None
        - mouse_mes (array-like): mouse mean/median data, structured as 
                                  mouse x session
                                  default: None
        - col (str)             : plotting color
                                  default: None
    """
    
    all_diff_st = np.asarray(all_diff_st)

    if sess_ns is None:
        sess_ns = range(all_diff_st.shape[0])
    sess_ns = np.asarray(sess_ns)

    diff_st_nan = np.isfinite(all_diff_st[:, 0])
    keep_idx = np.where(diff_st_nan)[0]
    
    lab = None
    # calculate number of mice for legend
    if mouse_mes is not None:
        if mouse_mes.shape[1] != len(sess_ns):
            raise ValueError(('`mouse_mes` second dimension should be '
                              'length {}.').format(len(sess_ns)))
        n_mice_per = np.sum(np.isfinite(mouse_mes[:, keep_idx]), axis=0)
        min_mice, max_mice = np.min(n_mice_per), np.max(n_mice_per)
        lab = 'n={}'.format(min_mice)
        if min_mice != max_mice:
            lab = '{}-{}'.format(lab, max_mice)

    plot_util.plot_errorbars(sub_ax, all_diff_st[keep_idx, 0], 
              all_diff_st[keep_idx, 1:].T, sess_ns[keep_idx], col=col, 
              label=lab, alpha=0.8)


#############################################
def plot_signif_from_mouse_diffs(sub_ax, signif_idx, st_data, signs, 
                                 sess_ns=None, col=None):
    """
    plot_signif_from_mouse_diffs(sub_ax, signif_idx, st_data, signs)

    Plots and positions significance markers based on the mouse statistics.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - signif_idx (list)        : list of significant sessions, possibly by 
                                     tail
        - st_data (3D array)       : statistics for each mouse, structured as
                                     mouse x sess x stats
        - signs (array-like)       : sign of difference between actual and 
                                     shuffled mean

    Optional args:
        - sess_ns (array-like): session numbers for each session
                                default: None
        - col (str)           : plotting color
                                default: None
    """

    st_data = np.asarray(st_data)

    if sess_ns is None:
        sess_ns = range(st_data.shape[1])
    sess_ns = np.asarray(sess_ns)

    if len(signif_idx) == 2: # flatten if 2 tailed                
        signif_idx = [i for sub in signif_idx for i in sub]
    
    for idx in signif_idx: # each star separately
        x_val = sess_ns[idx].reshape([1])
        # get high value or low depending on tail
        if signs[idx] == 1:
            ys    = np.nansum([st_data[:, idx, 0], st_data[:, idx, -1]], axis=0)
            y_val = np.nanmax(ys, axis=0).reshape([1])
            rel_y = 0.01
        elif signs[idx] == -1:
            ys    = np.nansum([st_data[:, idx, 0], -st_data[:, idx, 1]], axis=0)
            y_val = np.nanmin(ys, axis=0).reshape([1])
            rel_y = -0.07
        plot_util.add_signif_mark(sub_ax, x_val, y_val, rel_y=rel_y, col=col)


#############################################
def plot_area_diff_per_linlay(sub_ax, sess_ns, mouse_diff_st, diff_st, CI_vals, 
                              sign_sess, sess_info, plot='tog', d='data', 
                              col='k', zero_line=False):
    """
    plot_area_diff_per_linlay(sub_ax, sess_ns, mouse_diff_st, diff_st, CI_vals,
                              sign_sess, sess_info)

    Plots data or CIs for a specific layer/line combination.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - sess_ns (array-like)     : session numbers
        - mouse_diff_st (3D array) : difference statistics across ROIs or 
                                     seqs, structured as 
                                         mouse x session x stats
        - diff_st (2D array)       : difference stats across mice, structured 
                                         as session x stats
        - CI_vals (2D array)       : CIs values, structured as 
                                         session x perc (med, lo, high)
        - sign_sess (list)         : list of significant sessions, possibly by 
                                     tail
        - sess_info (list)         : list of dictionaries for each mouse 
                                     containing information from each session, 
                                     with None for missing sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists

    Optional args:
        - plot (str)      : type of plot ('sep' for mice separated or 
                                          'tog' for all mice grouped)
                            default: 'tog'
        - d (str)         : type of data to plot ('data' or 'CIs')
                            default: 'data'
        - col (str)       : color to use for data
                            default: 'k'
        - zero_line (bool): if True, a horizontal line is plotted at 0 for data
                            default: False
    """

    sess_ns = np.asarray(sess_ns)
    diff_st = np.asarray(diff_st)

    # plot the mouse lines
    if d == 'data':
        mouse_st = np.asarray(mouse_diff_st)
        if plot == 'sep':
            plot_area_diff_per_mouse(sub_ax, mouse_st, sess_info, sess_ns, 
                                     col, use_lab=True)
        elif plot == 'tog':
            plot_area_diff_stats(sub_ax, diff_st, sess_ns, mouse_st[:, :, 0], 
                                 col)
        else:
            gen_util.accepted_values_error('plot', plot, ['sep', 'tog'])
    
    elif d == 'CIs':
        if zero_line:
            sub_ax.axhline(y=0, ls='dashed', c='k', lw=2.0, alpha=0.5, 
                            zorder=-13)
        CI_vals  = np.asarray(CI_vals)
        keep_idx = np.where(np.isfinite(CI_vals[:, 0]))[0] # get mask
        
        # Add CIs
        plot_util.plot_CI(sub_ax, CI_vals[keep_idx, 1:].T, CI_vals[keep_idx, 0], 
                          sess_ns[keep_idx], width=0.3, zorder=-12)
        if plot == 'sep': # Add mean/median lines
            plot_util.plot_lines(sub_ax, diff_st[keep_idx, 0], 
                      sess_ns[keep_idx], y_rat=0.015, col=col, width=0.3)
            ypos_data = np.asarray(mouse_diff_st)
        else:
            ypos_data = np.expand_dims(diff_st, axis=0) 

        # plot significance markers
        signs = np.sign(diff_st[:, 0] - CI_vals[:, 0]).astype(int)
        plot_signif_from_mouse_diffs(sub_ax, sign_sess, ypos_data, signs, 
                                     sess_ns, col)

    else:
        gen_util.accepted_values_error('d', d, ['data', 'CIs'])


#############################################
def plot_area_diff_acr_sess(analyspar, sesspar, stimpar, extrapar, sess_info, 
                            diff_info, figpar=None, lock=False, plot='tog'):
    """
    plot_area_diff_acr_sess(analyspar, sesspar, stimpar, extrapar, sess_info, 
                            diff_info)

    From dictionaries, plots statistics across ROIs or mice of difference 
    between regular and surprise.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                              parameters
            ['analysis'] (str): analysis type (e.g., 't')
            ['datatype'] (str): datatype (e.g., 'run', 'roi')
        - diff_info (dict)       : dictionary with difference info
            ['all_diff_stats'] (list)  : difference stats across mice, 
                                         structured as layer/line x session 
                                                                  x stats
            ['mouse_diff_stats'] (list): difference statistics across ROIs or 
                                         seqs, structured as 
                                             layer/line x mouse x session 
                                                        x stats
            ['CI_vals'] (list)         : CIs values, structured as
                                             layer/line x session 
                                                        x perc (med, lo, high)
            ['sign_sess'] (list)       : significant session indices, 
                                         structured as layer/line (x tails)
            ['linlay_ord'] (list)      : order list of layers/lines
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/layer x mouse containing information 
                                   from each session, with None for missing 
                                   sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
                
    Optional args:
        - figpar (dict)     : dictionary containing the following figure 
                              parameter dictionaries
                              default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
        - lock (bool or str): if 'surplock' or 'reglock', differences being
                              plotted are surp or reg-locked, correspondingly. 
                              default: False    
        - plot (str)        : if 'tog', average is taken across mice, otherwise, 
                              if 'sep', each mouse is plotted separately
                              default: 'tog'

    Returns:
        - fig (plt Figure) : pyplot figure
    """
 
    datatype = extrapar['datatype']
    error = analyspar['error']
    if datatype == 'run' and plot == 'sep':
        error = 'None'
    
    dimstr = ''
    if plot == 'tog':
        dimstr = ' across mice'
    elif datatype == 'roi':
        dimstr = ' across {}'.format(sess_str_util.datatype_dim_str(datatype))

    stimstr_pr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                    stimpar['bri_dir'], stimpar['bri_size'],
                                    stimpar['gabk'], 'print')
    statstr_pr = sess_str_util.stat_par_str(analyspar['stats'], error, 'print')
    dendstr_pr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                            extrapar['datatype'], 'print')
    
    sess_ns = np.asarray(sesspar['sess_n'])
    if sesspar['sess_n'] in ['any', 'all']:
        sess_ns = np.arange(len(sess_info[0][0]['sess_ns'])) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.tolist())

    [lines, layers, linlay_iter, 
            lay_cols, _, n_plots] = sess_plot_util.fig_linlay_pars( 
                                        n_grps=len(diff_info['linlay_ord']))
    linlay_iter = [[d, ll] for d in ['data', 'CIs'] for ll in linlay_iter]
    figpar = sess_plot_util.fig_init_linlay(figpar)

    subtitle = 'Surp - reg activity'
    if lock:
        prepost_str = '{}s pre v post'.format(stimpar['post'])
        if lock == 'surplock':
            subtitle += ' locked to surprise onset'
        elif lock == 'reglock':
            subtitle = 'Reg - surp activity locked to regular onset'
        else:
            raise ValueError(('If lock is not False, it must be `reglock` or \
                              `surplock`.'))
    else:
        prepost_str = sess_str_util.prepost_par_str(stimpar['pre'], 
                                    stimpar['post'], str_type='print')
    title = '{} ({} seqs)\nfor {} - {}{}\n(sess {}{})'.format(subtitle, 
            prepost_str, stimstr_pr, statstr_pr, dimstr, sess_ns_str, 
            dendstr_pr)

    fig, ax = plot_util.init_fig(n_plots, **figpar['init'])
    fig.suptitle(title)
    for i, (d, [line, lay]) in enumerate(linlay_iter):
        li = lines.index(line)
        la = layers.index(lay)
        l_idx = get_linlay_idx(diff_info['linlay_ord'], line, lay, 
                               verbose=(d==0), newline=(i==0))
        if l_idx is None:
            continue

        plot_area_diff_per_linlay(ax[la, li], sess_ns, 
                      diff_info['mouse_diff_stats'][l_idx], 
                      diff_info['all_diff_stats'][l_idx], 
                      diff_info['CI_vals'][l_idx],
                      diff_info['sign_sess'][l_idx], sess_info[l_idx],
                      plot=plot, d=d, col=lay_cols[la], zero_line=False)
 
    # Add layer, line info to plots
    sess_plot_util.format_linlay_subaxes(ax, fluor=analyspar['fluor'], 
                   area=True, datatype=datatype, lines=lines, layers=layers, 
                   xticks=sess_ns)

    return fig
   

#############################################
def plot_surp_area_diff(analyspar, sesspar, stimpar, basepar, permpar, extrapar, 
                        sess_info, diff_info, figpar=None, savedir=None):
    """
    plot_surp_area_diff(analyspar, sesspar, stimpar, basepar, permpar, extrapar, 
                        sess_info, diff_stats)

    From dictionaries, plots statistics across ROIs or mice of difference 
    between regular and surprise averaged across sequences.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - basepar (dict)  : dictionary with keys of BasePar namedtuple
        - permpar (dict)  : dictionary with keys of PermPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ['analysis'] (str): analysis type (e.g., 't')
            ['datatype'] (str): datatype (e.g., 'run', 'roi')
        - sess_info (list): list of dictionaries for each mouse containing 
                              information from each session 
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - diff_info (dict): dictionary containing difference information
            ['all_diff_stats'] (list)  : difference stats across mice, 
                                         structured as layer/line x session 
                                                                  x stats
            ['mouse_diff_stats'] (list): difference statistics across ROIs or 
                                         seqs, structured as 
                                             layer/line x mouse x session 
                                                        x stats
            ['CI_vals'] (list)         : CIs values, structured as
                                             layer/line x session 
                                                        x perc (med, lo, high)
            ['sign_sess'] (list)       : significant session indices, 
                                         structured as layer/line (x tails)
            ['linlay_ord'] (list)      : order list of layers/lines
                
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
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): name under which the figure is saved
    """
    
    sessstr = sess_str_util.sess_par_str(sesspar['sess_n'], stimpar['stimtype'],
                                         sesspar['layer'], stimpar['bri_dir'], 
                                         stimpar['bri_size'], stimpar['gabk'])
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                         extrapar['datatype'])
    datatype = extrapar['datatype']

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    base_str = sess_str_util.base_par_str(basepar['baseline'])[1:]
    if savedir is None:
        savedir = os.path.join(figpar['dirs'][datatype], 
                               figpar['dirs']['acr_sess'], base_str)

    gen_savename = '{}_surp_diff_{}{}'.format(datatype, sessstr, dendstr)
    part = 'surp_diff'
    add_idx = gen_savename.find(part) + len(part)

    for plot in ['sep', 'tog']:
        fig = plot_area_diff_acr_sess(analyspar, sesspar, stimpar, extrapar, 
                                    sess_info, diff_info, figpar=figpar, 
                                    lock=False, plot=plot)


        savename = '{}_{}{}'.format(gen_savename[:add_idx], plot, 
                                    gen_savename[add_idx:])
        fulldir = plot_util.savefig(fig, savename, savedir, print_dir=(plot==0), 
                                    **figpar['save'])

    return fulldir, gen_savename


#############################################
def plot_lock_area_diff(analyspar, sesspar, stimpar, basepar, permpar, extrapar, 
                        sess_info, diff_info, figpar=None, savedir=None):
    """
    plot_lock_area_diff(analyspar, sesspar, stimpar, permpar, extrapar, 
                        sess_info, diff_info)

    From dictionaries, plots statistics across ROIs or mice of difference 
    between regular and surprise averaged across sequences, locked to 
    transitions from regular to surprise, and vice versa.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - basepar (dict)  : dictionary with keys of BasePar namedtuple
        - permpar (dict)  : dictionary with keys of PermPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ['analysis'] (str): analysis type (e.g., 'l')
            ['datatype'] (str): datatype (e.g., 'run', 'roi')
        - sess_info (list): list of dictionaries containing information from 
                            each session, structured as 
                            [surp-locked, reg-locked]
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - diff_info (list): list of dictionaries containing difference 
                            information, structured as [surp-locked, reg-locked]
            ['all_diff_stats'] (list)  : difference stats across mice, 
                                         structured as layer/line x session 
                                                                  x stats
            ['mouse_diff_stats'] (list): difference statistics across ROIs or 
                                         seqs, structured as 
                                             layer/line x mouse x session 
                                                        x stats
            ['CI_vals'] (list)         : CIs values, structured as
                                             layer/line x session 
                                                        x perc (med, lo, high)
            ['sign_sess'] (list)       : significant session indices, 
                                         structured as layer/line (x tails)
            ['linlay_ord'] (list)      : order list of layers/lines
                
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
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): name under which the figure is saved
    """
 
    sessstr = sess_str_util.sess_par_str(sesspar['sess_n'], stimpar['stimtype'],
                                         sesspar['layer'], stimpar['bri_dir'], 
                                         stimpar['bri_size'], stimpar['gabk'])
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                         extrapar['datatype'])
    datatype = extrapar['datatype']

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    base_str = sess_str_util.base_par_str(basepar['baseline'])[1:]
    if savedir is None:
        savedir = os.path.join(figpar['dirs'][datatype], 
                               figpar['dirs']['acr_sess'], base_str)

    gen_savename = '{}_lock_diff_{}{}'.format(datatype, sessstr, dendstr)

    for l, lock in enumerate(['surplock', 'reglock']):
        part = 'lock_diff'
        lock_savename = gen_savename.replace('lock', lock)
        add_idx = lock_savename.find(part) + len(part)

        for plot in ['sep', 'tog']:
            fig = plot_area_diff_acr_sess(analyspar, sesspar, stimpar, extrapar, 
                                      sess_info[l], diff_info[l], figpar=figpar, 
                                      lock=lock, plot=plot)

            savename = '{}_{}{}'.format(lock_savename[:add_idx], plot, 
                                        lock_savename[add_idx:])
            fulldir = plot_util.savefig(fig, savename, savedir, 
                                print_dir=(plot==0), **figpar['save'])

    return fulldir, gen_savename


#############################################
def plot_mouse_traces(sub_ax, xran, trace_st, lock=False, col='k', lab=True, 
                      stimtype='gabors', gabfr=0):
    """
    plot_mouse_traces(sub_ax, xran, trace_st, cols, names)

    Plot regular and surprise data traces for a mouse.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - xran (array-like)        : second values for each frame
        - trace_st (4D array)      : trace statistics, structured as
                                         reg/surp x frame x stats

    Optional args:
        - lock (bool)   : if True, plotted data is locked 
                          default: False
        - col  (str)    : colour for surprise data
                          default: 'k'
        - lab (bool)    : if True, data label is included for legend
                          default: True
        - stimtype (str): stimtype ('gabors' or 'bricks')
                          default: 'gabors'
        - gabfr (int)   : gabor start frame number
                          default: 0
    """

    cols = ['gray', col]
    names = ['reg', 'surp']
    xran = np.asarray(xran)

    xticks = None
    if lock:
        xticks = np.linspace(-np.max(xran), np.max(xran), 5)

    # horizontal 0 line
    sub_ax.axhline(y=0, ls='dashed', c='k', lw=2.0, alpha=0.5, zorder=-13)

    # add vertical lines       
    plot_util.add_bars(sub_ax, hbars=0)
    if lock:
        rev_xran = xran[::-1] * -1
        full_xran = np.concatenate([rev_xran, xran])
    else:
        full_xran = xran

    if stimtype == 'gabors':
        sess_plot_util.plot_gabfr_pattern(sub_ax, full_xran, offset=gabfr, 
                                          bars_omit=[0])

    trace_st = np.asarray(trace_st)
    for i, (col, name) in enumerate(zip(cols, names)):
        label = name if lab else None
        if lock and name not in lock:
            xran_use = rev_xran
        else:
            xran_use = xran
        if lock == 'reglock':
            i = 1 - i # data ordered as [surp, reg] instead of vv
        plot_util.plot_traces(sub_ax, xran_use, trace_st[i, :, 0], 
                              trace_st[i, :, 1:], label=label, alpha_line=0.8, 
                              col=col, xticks=xticks)



#############################################
def plot_traces_acr_sess(analyspar, sesspar, stimpar, extrapar, sess_info, 
                         trace_info, figpar=None, lock=False):
    """
    plot_traces_acr_sess(analyspar, sesspar, stimpar, extrapar, sess_info, 
                         trace_info)

    From dictionaries, plots traces across ROIs or mice for regular and 
    surprise sequences.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ['analysis'] (str): analysis type (e.g., 't')
            ['datatype'] (str): datatype (e.g., 'run', 'roi')
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/layer x mouse containing information 
                                   from each session, with None for missing 
                                   sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - trace_info (dict)      : dictionary with difference info
            ['linlay_ord'] (list) : order list of layers/lines
            ['trace_stats'] (list): trace statistics, structured as
                                    layer/line x session x reg/surp x frame 
                                               x stats
            ['xran'] (list)       : second values for each frame

    Optional args:
        - figpar (dict)     : dictionary containing the following figure 
                              parameter dictionaries
                              default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
        - lock (bool or str): if 'surplock' or 'reglock', differences being
                              plotted are surp or reg-locked, correspondingly. 
                              default: False     

    Returns:
        - fig (plt Figure) : pyplot figure
    """
 
    datatype = extrapar['datatype']
    stimstr_pr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                    stimpar['bri_dir'], stimpar['bri_size'],
                                    stimpar['gabk'], 'print')
    statstr_pr = sess_str_util.stat_par_str(analyspar['stats'], 
                                            analyspar['error'], 'print')
    dendstr_pr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                            extrapar['datatype'], 'print')
    
    sess_ns = np.asarray(sesspar['sess_n'])
    if sesspar['sess_n'] in ['any', 'all']:
        sess_ns = np.asarray(range(len(sess_info[0][0]['sess_ns']))) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.tolist())
    n_sess = len(sess_ns)

    [lines, layers, linlay_iter, 
          lay_cols, _, n_plots] = sess_plot_util.fig_linlay_pars(traces=n_sess, 
                                        n_grps=len(trace_info['linlay_ord']))
    figpar = sess_plot_util.fig_init_linlay(figpar, traces=n_sess)

    subtitle = 'Surp v reg activity'
    if lock:
        prepost_str = '{}s pre v post'.format(stimpar['post'])
        if lock == 'surplock':
            subtitle += ' locked to surprise onset'
        elif lock == 'reglock':
            subtitle = 'Reg v surp activity locked to regular onset'
        else:
            raise ValueError(('If lock is not False, it must be `reglock` or \
                              `surplock`.'))
    else:
        prepost_str = sess_str_util.prepost_par_str(stimpar['pre'], 
                                    stimpar['post'], str_type='print')
    title = ('{} ({} seqs)\nfor {} - {} across mice\n'
            '(sess {}{})').format(subtitle, prepost_str, stimstr_pr, 
                                  statstr_pr, sess_ns_str, dendstr_pr)

    fig, ax = plot_util.init_fig(n_plots, **figpar['init'])
    fig.suptitle(title)
    for i, (line, lay) in enumerate(linlay_iter):
        li = lines.index(line)
        la = layers.index(lay)
        
        l_idx = get_linlay_idx(trace_info['linlay_ord'], line, lay, 
                               verbose=True, newline=(i==0))
        if l_idx is None:
            continue

        # plot the mouse traces
        for s in range(n_sess):
            sub_ax = ax[s + la * n_sess, li]
            lab = (li == 1 and s == 0)
            plot_mouse_traces(sub_ax, trace_info['xran'], 
                              trace_info['trace_stats'][l_idx][s], lock, 
                              lay_cols[la], lab, stimpar['stimtype'], 
                              stimpar['gabfr'])
        
    # Add layer, line info to plots
    sess_plot_util.format_linlay_subaxes(ax, fluor=analyspar['fluor'], 
                   area=False, datatype=datatype, lines=lines, layers=layers, 
                   sess_ns=sess_ns)

    return fig
   

#############################################
def plot_surp_traces(analyspar, sesspar, stimpar, basepar, extrapar, sess_info, 
                     trace_info, figpar=None, savedir=None):
    """
    plot_surp_traces(analyspar, sesspar, stimpar, basepar, extrapar, sess_info, 
                     trace_info)

    From dictionaries, plots traces across ROIs or mice of difference 
    for regular and surprise averaged across sequences.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict)       : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)         : dictionary with keys of SessPar namedtuple
        - stimpar (dict)         : dictionary with keys of StimPar namedtuple
        - basepar (dict)         : dictionary with keys of BasePar namedtuple
        - extrapar (dict)        : dictionary containing additional analysis 
                                   parameters
            ['analysis'] (str): analysis type (e.g., 't')
            ['datatype'] (str): datatype (e.g., 'run', 'roi')
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/layer x mouse containing information 
                                   from each session, with None for missing 
                                   sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - trace_info (dict)      : dictionary with difference info
            ['linlay_ord'] (list) : order list of layers/lines            
            ['trace_stats'] (list): trace statistics, structured as
                                    layer/line x session x reg/surp x frame 
                                               x stats
            ['xran'] (list)       : second values for each frame

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
        - fulldir (str) : final name of the directory in which the figure 
                          is saved (may differ from input savedir, if 
                          datetime subfolder is added.)
        - savename (str): name under which the figure is saved
    """
    
    sessstr = sess_str_util.sess_par_str(sesspar['sess_n'], stimpar['stimtype'],
                                         sesspar['layer'], stimpar['bri_dir'], 
                                         stimpar['bri_size'], stimpar['gabk'])
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                         extrapar['datatype'])
    datatype = extrapar['datatype']

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    fig = plot_traces_acr_sess(analyspar, sesspar, stimpar, extrapar, 
                                sess_info, trace_info, figpar=figpar, 
                                lock=False)

    base_str = sess_str_util.base_par_str(basepar['baseline'])[1:]
    if savedir is None:
        savedir = os.path.join(figpar['dirs'][datatype], 
                            figpar['dirs']['acr_sess'], base_str)

    savename = '{}_surp_tr_{}{}'.format(datatype, sessstr, dendstr)
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar['save'])

    return fulldir, savename


#############################################
def plot_lock_traces(analyspar, sesspar, stimpar, basepar, extrapar, sess_info, 
                     trace_info, figpar=None, savedir=None):
    """
    plot_lock_traces(analyspar, sesspar, stimpar, extrapar, sess_info,
                     trace_info)

    From dictionaries, plots traces across ROIs or mice for regular and 
    surprise sequences, locked to transitions from regular to surprise, then
    from surprise to regular.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict)       : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)         : dictionary with keys of SessPar namedtuple
        - stimpar (dict)         : dictionary with keys of StimPar namedtuple
        - stimpar (dict)         : dictionary with keys of BasePar namedtuple
        - extrapar (dict)        : dictionary containing additional analysis 
                                   parameters
            ['analysis'] (str)        : analysis type (e.g., 't')
            ['datatype'] (str)        : datatype (e.g., 'run', 'roi')
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/layer x mouse containing information 
                                   from each session, with None for missing 
                                   sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - trace_info (dict)      : dictionary with difference info
            ['linlay_ord'] (list) : order list of layers/lines            
            ['trace_stats'] (list): trace statistics, structured as
                                    layer/line x session x reg/surp x frame 
                                               x stats
            ['xran'] (list)       : second values for each frame

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
        - fulldir (str)     : final name of the directory in which the figure 
                              is saved (may differ from input savedir, if 
                              datetime subfolder is added.)
        - gen_savename (str): general name under which the figures are saved
    """
 
    sessstr = sess_str_util.sess_par_str(sesspar['sess_n'], stimpar['stimtype'],
                                         sesspar['layer'], stimpar['bri_dir'], 
                                         stimpar['bri_size'], stimpar['gabk'])
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                         extrapar['datatype'])
    datatype = extrapar['datatype']

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    gen_savename = '{}_lock_tr_{}{}'.format(datatype, sessstr, dendstr)
    
    for l, lock in enumerate(['surplock', 'reglock']):
        fig = plot_traces_acr_sess(analyspar, sesspar, stimpar, extrapar, 
                                   sess_info[l], trace_info[l], figpar=figpar, 
                                   lock=lock)

        base_str = sess_str_util.base_par_str(basepar['baseline'])[1:]
        if savedir is None:
            savedir = os.path.join(figpar['dirs'][datatype], 
                                figpar['dirs']['acr_sess'], base_str)

        savename = gen_savename.replace('lock', lock)
    
        fulldir = plot_util.savefig(fig, savename, savedir, **figpar['save'])

    return fulldir, gen_savename


#############################################
def plot_lat_clouds(sub_ax, sess_ns, lat_data, sess_info, datatype='roi', 
                    col='blue', alpha=0.2):
    """
    plot_lat_clouds(sub_ax, sess_ns, lat_data)

    Plots clouds of latency data in different shades for each mouse.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - sess_ns (array-like)     : session numbers
        - lat_data (2D array)      : latency values for each ROI, structured as
                                         session x mouse
        - sess_info (list)         : list of dictionaries for each mouse 
                                     containing information from each session, 
                                     with None for missing sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists

    Optional args:
        - prev_maxes (1D array): array of previous max values for each session
        - datatype (str): datatype (e.g., 'run', 'roi')
                          default: 'roi'
        - col (str)     : name of colour to base shades on
                          default: 'blue'
        - alpha (float) : pyplot transparency parameter
                          default: 0.3
    
    Returns:
        - maxes (1D array): max values from data clouds for each session
    """

    n_mice = [len(s_vals) for s_vals in lat_data if s_vals is not None]

    if len(set(n_mice)) != 1:
        raise ValueError('Should be the same number of mice.')
    n_mice = n_mice[0]
    m_cols = plot_util.get_color_range(n_mice, col=col)

    labels = ['' for _ in range(n_mice)] # to collect labels
    clouds = [None for _ in range(n_mice)] # to collect artists
    maxes = np.full(len(sess_ns), np.nan)
    for s, sess_n in enumerate(sess_ns):
        for m in range(n_mice):
            if lat_data[s][m] is None:
                nrois = 0
            else:
                nrois = len(lat_data[s][m])
            if len(labels[m]) == 0:
                add = 'M#'
                if datatype == 'roi':
                    add = '{} ({}'.format(add, nrois)
            else:
                if datatype == 'roi':
                    add = '/{}'.format(nrois)
                else:
                    add = ''
            labels[m] = '{}{}'.format(labels[m], add)
            if nrois == 0:
                continue
            elif '#' in labels[m]:
                labels[m] = labels[m].replace('#', 
                                              str(sess_info[s]['mouse_ns'][m]))
            div_fact = np.max([1, len(lat_data[s][m])//50])
            alpha_spec = alpha/div_fact
            clouds[m] = plot_util.plot_data_cloud(sub_ax, sess_n, 
                                    lat_data[s][m], 0.15, label=None, 
                                    col=m_cols[m], alpha=alpha_spec, zorder=-11)
            maxes[s] = np.nanmax([maxes[s], np.nanmax(lat_data[s][m])])
    if datatype == 'roi':
        labels = ['{} ROIs)'.format(label) for label in labels]
    sub_ax.legend(clouds, labels, fontsize='large')

    return maxes


#############################################
def plot_lat_data_signif(ax, sess_ns, sig_comps, lin_p_vals, maxes, 
                         p_val_thr=0.05, n_comps=1):
    """
    plot_lat_data_signif(ax, sess_ns, sig_comps, lin_p_vals, maxes)

    Plot significance markers for significant session comparisons within and 
    across lines/layer combinations.

    Required args:
        - ax (plt Axis)         : axis
        - sess_ns (array-like)  : session numbers
        - sig_comps (array_like): list of session pair comparisons that are 
                                  significant (where the second session is 
                                  cycled in the inner loop, e.g., 0-1, 0-2, 
                                  1-2, including None sessions)
        - lin_p_vals            : p-values for each line comparison, 
                                  structured as line x session (np.nan for 
                                  sessions  missing in either layer)
        - maxes                 : max values used to adjust ylims, structured 
                                  as layer/line x session

    Optional args:
        - p_val_thr (float): p value threshold
                             default: 0.05
        - n_comps (int)    : total number of comparisons (used to modify p 
                             value threshold using Bonferroni correction)
                             default: 1
    """

    lines, layers, linlay_iter, _, _ , _ = sess_plot_util.fig_linlay_pars()

    p_val_thr = 0.05
    if n_comps == 0:
        return
    else:
        p_val_thr_corr = p_val_thr/n_comps

    all_max = np.nanmax(np.asarray(maxes))
    if not np.isnan(all_max):
        ylims = ax[0, 0].get_ylim()
        ax[0, 0].set_ylim([ylims[0], np.max([ylims[1], all_max * 1.20])])

    n_sess = len(sess_ns)
    n = 0
    # comparison number: first session start pts
    st_s1 = [sum(list(reversed(range(n_sess)))[:v]) for v in range(n_sess-1)]  
    for i, (line, lay) in enumerate(linlay_iter):
        li = lines.index(line)
        la = layers.index(lay)
        sub_ax = ax[la, li]
        if not(sig_comps[i] is None or len(sig_comps[i]) == 0):
            n = 0
            for p in sig_comps[i]:
                n += 1
                # get corresponding session numbers
                s1 = np.where(np.asarray(st_s1) <= p)[0][-1] 
                s2 = s1 + p - st_s1[s1] + 1
                y_pos = np.nanmax([maxes[i]]) + n * 0.03
                plot_util.plot_barplot_signif(sub_ax, [sess_ns[s1], 
                          sess_ns[s2]], [y_pos], rel_y=0.02)
        if la == 1:
            for s, p in enumerate(lin_p_vals[li]):
                if not np.isnan(p) and p < p_val_thr_corr:
                    # between subplots
                    plot_util.add_signif_mark(sub_ax, sess_ns[s], 0, rel_y=1.1)


#############################################
def plot_surp_latency(analyspar, sesspar, stimpar, latpar, extrapar, sess_info, 
                      lat_data, figpar=None, savedir=None):
    """
    plot_surp_latency(analyspar, sesspar, stimpar, extrapar, sess_info,
                      lat_data)

    From dictionaries, plots surprise latency across mice, as well as for all 
    ROIs.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict)       : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)         : dictionary with keys of SessPar namedtuple
        - stimpar (dict)         : dictionary with keys of StimPar namedtuple
        - latpar (LatPar)        : named tuple of latency parameters
        - extrapar (dict)        : dictionary containing additional analysis 
                                   parameters
            ['analysis'] (str): analysis type (e.g., 't')
            ['datatype'] (str): datatype (e.g., 'run', 'roi')
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/layer x session containing information 
                                   from each mouse, with None for missing 
                                   sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - lat_data (dict)        : dictionary with latency info
            ['linlay_ord'] (list): ordered list of layers/lines            
            ['lat_stats'] (list) : latency statistics, structured as
                                       layer/line x session x stats
            ['lat_vals'] (list)  : latency values for each ROI, structured as
                                       layer/line x session x mouse
            ['lat_p_vals'] (list): p-values for each latency comparison within 
                                   session pairs, (where the second session is 
                                   cycled in the inner loop, e.g., 0-1, 0-2, 
                                   1-2, including None sessions)
                                   structured as layer/line x comp
            ['lin_p_vals'] (list): p-values for each line comparison, 
                                   structured as line x session (np.nan for 
                                   sessions  missing in either layer)
            ['n_comps'] (int)    : number of comparisons

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
        - fulldir (str) : final name of the directory in which the figure 
                          is saved (may differ from input savedir, if 
                          datetime subfolder is added.)
        - savename (str): name under which the figure is saved
    """
 
    sessstr = sess_str_util.sess_par_str(sesspar['sess_n'], stimpar['stimtype'],
                                         sesspar['layer'], stimpar['bri_dir'], 
                                         stimpar['bri_size'], stimpar['gabk'])
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                         extrapar['datatype'])
    latstr = sess_str_util.lat_par_str(latpar['method'], latpar['p_val_thr'], 
                                       latpar['rel_std'])

    stimstr_pr = sess_str_util.stim_par_str(stimpar['stimtype'], 
                                    stimpar['bri_dir'], stimpar['bri_size'],
                                    stimpar['gabk'], 'print')
    statstr_pr = sess_str_util.stat_par_str(analyspar['stats'], 
                                            analyspar['error'], 'print')
    dendstr_pr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                            extrapar['datatype'], 'print')
    latstr_pr = sess_str_util.lat_par_str(latpar['method'], latpar['p_val_thr'], 
                                          latpar['rel_std'], 'print')

    datatype = extrapar['datatype']

    sess_ns = np.asarray(sesspar['sess_n'])
    if sesspar['sess_n'] in ['any', 'all']:
        sess_ns = np.asarray(range(len(sess_info[0]))) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.tolist())
    n_sess = len(sess_ns)

    # correct p-value (Bonferroni)
    p_val_thr = 0.05
    if lat_data['n_comps'] != 0:
        p_val_thr_corr = p_val_thr/lat_data['n_comps']

    [lines, layers, linlay_iter, 
     lay_cols, lay_col_names, n_plots] = sess_plot_util.fig_linlay_pars( 
                                        n_grps=len(lat_data['linlay_ord']))
    figpar = sess_plot_util.fig_init_linlay(figpar)

    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    prepost_str = sess_str_util.prepost_par_str(stimpar['pre'], 
                                stimpar['post'], str_type='print')
    title = ('Surprise latencies ({} seqs, {})\nfor {} - {} pooled across mice'
            '\n ROIs (sess {}{})').format(prepost_str, latstr_pr, stimstr_pr, 
                                   statstr_pr, sess_ns_str, dendstr_pr)

    fig, ax = plot_util.init_fig(n_plots, **figpar['init'])
    fig.suptitle(title)
    maxes = np.full([len(linlay_iter), n_sess], np.nan)
    sig_comps = [[] for _ in range(len(linlay_iter))]
    for i, (line, lay) in enumerate(linlay_iter):
        li = lines.index(line)
        la = layers.index(lay)
        sub_ax = ax[la, li]

        l_idx = get_linlay_idx(lat_data['linlay_ord'], line, lay, verbose=True, 
                               newline=(i==0))
        if l_idx is None:
            continue

        lat_st = np.asarray(lat_data['lat_stats'][l_idx])

        plot_util.plot_errorbars(sub_ax, lat_st[0], lat_st[1:], sess_ns, 
                                 col=lay_cols[la])
        # plot ROI cloud
        maxes[i] = plot_lat_clouds(sub_ax, sess_ns, lat_data['lat_vals'][l_idx], 
                                   sess_info[l_idx], datatype=datatype, 
                                   col=lay_col_names[la])

        # check p_val signif
        all_p_vals = lat_data['lat_p_vals'][l_idx]
        for p, p_val in enumerate(all_p_vals):
            if not np.isnan(p_val) and p_val < p_val_thr_corr:
                sig_comps[i].append(p)


    plot_lat_data_signif(ax, sess_ns, sig_comps, lat_data['lin_p_vals'], 
                         maxes, p_val_thr=0.05, n_comps=lat_data['n_comps'])

    # Add layer, line info to plots
    sess_plot_util.format_linlay_subaxes(ax, fluor=analyspar['fluor'], 
                   datatype=datatype, lines=lines, layers=layers, 
                   xticks=sess_ns, y_ax='Latency (s)')


    if savedir is None:
        savedir = os.path.join(figpar['dirs'][datatype], 
                               figpar['dirs']['acr_sess'], 
                               figpar['dirs']['lat'], latpar['method'])

    savename = '{}_surp_lat_{}{}_{}'.format(datatype, sessstr, dendstr, latstr)
    
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar['save'])

    return fulldir, savename


#############################################
def plot_resp_prop(analyspar, sesspar, stimpar, latpar, extrapar, sess_info, 
                   prop_data, figpar=None, savedir=None):
    """
    plot_resp_prop(analyspar, sesspar, stimpar, extrapar, sess_info,
                   prop_data)

    From dictionaries, plots surprise latency across mice, as well as for all 
    ROIs.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict)       : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)         : dictionary with keys of SessPar namedtuple
        - stimpar (dict)         : dictionary with keys of StimPar namedtuple
        - latpar (LatPar)        : named tuple of latency parameters
        - extrapar (dict)        : dictionary containing additional analysis 
                                   parameters
            ['analysis'] (str): analysis type (e.g., 't')
            ['datatype'] (str): datatype (e.g., 'run', 'roi')
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/layer x session containing information 
                                   from each mouse, with None for missing 
                                   sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['layers'] (list)     : imaging layers
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois'] (list)    : list of ROIs with NaNs/Infs in raw traces
            ['nanrois_dff'] (list): list of ROIs with NaNs/Infs in dF/F traces, 
                                    for sessions for which this attribute 
                                    exists
        - prop_data (dict)        : dictionary with responsive proportion info
            ['linlay_ord'] (list): ordered list of layers/lines            
            ['prop_stats'] (list): proportion statistics, structured as
                                       layer/line x session x comb x stats
            ['comb_names'] (int) : names of combinations for with proportions 
                                   were calculated

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
        - fulldir (str) : final name of the directory in which the figure 
                          is saved (may differ from input savedir, if 
                          datetime subfolder is added.)
        - savename (str): name under which the figure is saved
    """
 
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                         extrapar['datatype'])
    latstr = sess_str_util.lat_par_str(latpar['method'], latpar['p_val_thr'], 
                                       latpar['rel_std'])

    statstr_pr = sess_str_util.stat_par_str(analyspar['stats'], 
                                            analyspar['error'], 'print')
    dendstr_pr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                            extrapar['datatype'], 'print')
    latstr_pr = sess_str_util.lat_par_str(latpar['method'], latpar['p_val_thr'], 
                                          latpar['rel_std'], 'print')

    datatype = extrapar['datatype']

    sess_ns = np.asarray(sesspar['sess_n'])
    if sesspar['sess_n'] in ['any', 'all']:
        sess_ns = np.asarray(range(len(sess_info[0]))) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.tolist())

    # combinations: 'gabfrs', 'surps'
    ctrl_idx, surp_idx = [prop_data['comb_names'].index(comb) 
                          for comb in ['gabfrs', 'surps']]

    [lines, layers, linlay_iter, 
     lay_cols, _, n_plots] = sess_plot_util.fig_linlay_pars( 
                                  n_grps=len(prop_data['linlay_ord']))
    figpar = sess_plot_util.fig_init_linlay(figpar)

    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    prepost_str = sess_str_util.prepost_par_str(stimpar['pre'], 
                                stimpar['post'], str_type='print')
    title = ('Proportion surprise responsive ROIs ({} seqs, {})\n'
             '{} across mice\n (sess {}{})').format(prepost_str, latstr_pr, 
                                        statstr_pr, sess_ns_str, dendstr_pr)

    fig, ax = plot_util.init_fig(n_plots, **figpar['init'])
    fig.suptitle(title)
    for i, (line, lay) in enumerate(linlay_iter):
        li = lines.index(line)
        la = layers.index(lay)
        sub_ax = ax[la, li]

        l_idx = get_linlay_idx(prop_data['linlay_ord'], line, lay, verbose=True, 
                               newline=(i==0))
        if l_idx is None:
            continue
        
        for idx, col in zip([surp_idx, ctrl_idx], [lay_cols[la], 'gray']):
            # retrieve proportion (* 100)
            prop_st = np.asarray([sess_vals[idx] for sess_vals 
                                  in prop_data['prop_stats'][l_idx]]) * 100
            plot_util.plot_errorbars(sub_ax, prop_st[:, 0], prop_st[:, 1:], 
                                     sess_ns, col=col)

    # Add layer, line info to plots
    sess_plot_util.format_linlay_subaxes(ax, fluor=analyspar['fluor'], 
                   datatype=datatype, lines=lines, layers=layers, 
                   xticks=sess_ns, y_ax='Prop (%)')

    if savedir is None:
        savedir = os.path.join(figpar['dirs'][datatype], 
                               figpar['dirs']['acr_sess'], 
                               figpar['dirs']['prop'], latpar['method'])

    savename = '{}_prop_resp_sess{}{}_{}'.format(datatype, sess_ns_str, 
                                                 dendstr, latstr)
    
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar['save'])

    return fulldir, savename
