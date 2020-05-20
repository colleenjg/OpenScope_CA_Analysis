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
import warnings

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st

from sess_util import sess_gen_util, sess_plot_util, sess_str_util
from util import file_util, gen_util, math_util, plot_util

# skip tight layout warning
warnings.filterwarnings('ignore', message='This figure includes*')


#############################################
def plot_from_dict(dict_path, plt_bkend=None, fontdir=None, parallel=False, 
                   datetime=True):
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
        - datetime (bool): figpar['save'] datatime parameter (whether to 
                           place figures in a datetime folder)
                           default: True
    """

    print(f'\nPlotting from dictionary: {dict_path}')
    
    figpar = sess_plot_util.init_figpar(
        plt_bkend=plt_bkend, fontdir=fontdir, datetime=datetime)
    plot_util.manage_mpl(cmap=False, **figpar['mng'])

    sess_plot_util.update_plt_linpla()

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

    # 4. Plots the progression surprises/regular sequences across sessions
    elif analysis == 'g':
        plot_prog(figpar=figpar, savedir=savedir, **info)

    # 5. Plots the difference between surprise and regular locked to surprise
    # across sessions
    elif analysis == 'u':
        plot_surp_latency(figpar=figpar, savedir=savedir, **info)

    else:
        print(f'    No plotting function for analysis {analysis}')

    plt.close('all')


#############################################
def get_linpla_idx(linpla_ord, line='L2/3', plane='soma', verbose=False, 
                   newline=False):
    """
    get_linpla_idx(linpla_ord)


    Required args:
        - linpla_ord (list): ordered list of line/plane combinations formatted 

    Optional args:
        - line (str)    : line (e.g., L2/3, L5)
                          default: 'L2/3'
        - plane (str)   : plane (e.g., soma, dend)
                          default: 'soma'
        - verbose (bool): if True and no data is found, this is printed to the 
                          console
                          default: False
        - newline (bool): if True, text is printed on a new line
                          default: False

    Returns:
        - l_idx (int or None): line/plane combination index in linpla_ord or 
                               None if not found
    """

    data_name = f'{line} {plane[:4]}' # get data name
    if data_name not in linpla_ord:
        add = ''
        if newline:
            add = '\n'
        if verbose:
            print(f'{add}No data for {data_name}.')
        l_idx = None
    else:
        l_idx = linpla_ord.index(data_name)

    return l_idx


#############################################
def plot_data_signif(ax, sess_ns, sig_comps, lin_p_vals, maxes, 
                     p_val_thr=0.05, n_comps=1):
    """
    plot_data_signif(ax, sess_ns, sig_comps, lin_p_vals, maxes)

    Plot significance markers for significant session comparisons within and 
    across lines/plane combinations.

    Required args:
        - ax (plt Axis)         : axis
        - sess_ns (array-like)  : session numbers
        - sig_comps (array_like): list of session pair comparisons that are 
                                  significant (where the second session is 
                                  cycled in the inner loop, e.g., 0-1, 0-2, 
                                  1-2, including None sessions)
        - lin_p_vals            : p-values for each line comparison, 
                                  structured as line x session (np.nan for 
                                  sessions  missing in either plane)
        - maxes                 : max values used to adjust ylims, structured 
                                  as plane/line x session

    Optional args:
        - p_val_thr (float): p value threshold
                             default: 0.05
        - n_comps (int)    : total number of comparisons (used to modify p 
                             value threshold using Bonferroni correction)
                             default: 1
    """

    lines, planes, linpla_iter, pla_cols, _ , _ = \
        sess_plot_util.fig_linpla_pars()

    p_val_thr = 0.05
    if n_comps == 0:
        return
    else:
        p_val_thr_corr = p_val_thr/n_comps

    all_max = np.nanmax(np.asarray(maxes))
    if not np.isnan(all_max):
        ylims = ax[0, 0].get_ylim()
        ax[0, 0].set_ylim([ylims[0], np.max([ylims[1], all_max * 1.20])])

    ylims = ax[0, 0].get_ylim()
    prop = (ylims[1] - ylims[0])/10.0 # star position above data

    n_sess = len(sess_ns)
    n = 0
    # comparison number: first session start pts
    st_s1 = [sum(list(reversed(range(n_sess)))[:v]) for v in range(n_sess-1)] 
    highest = 0 
    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        sub_ax = ax[pl, li]
        if not(sig_comps[i] is None or len(sig_comps[i]) == 0):
            n = 0
            for p in sig_comps[i]:
                n += 1
                # get corresponding session numbers
                s1 = np.where(np.asarray(st_s1) <= p)[0][-1] 
                s2 = s1 + p - st_s1[s1] + 1
                y_pos = np.nanmax([maxes[i]]) + n * prop
                highest = np.max([highest, y_pos])
                plot_util.plot_barplot_signif(
                    sub_ax, [sess_ns[s1], sess_ns[s2]], [y_pos], rel_y=0.03, 
                    col=pla_cols[pl], lw=3, star_rel_y=0.09)

    # adjust for number of significance lines plotted
    ylims = ax[0, 0].get_ylim()
    if not np.isnan(highest):
        ax[0, 0].set_ylim(ylims[0], np.max([ylims[1], highest * 1.1]))

    ylims = ax[0, 0].get_ylim()
    pos = ylims[0] + (ylims[1] - ylims[0])/10.0 # star position (btw plots)
    
    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        if pl == 1:
            sub_ax = ax[pl, li]
            plot_util.set_ticks(sub_ax, min_tick=min(sess_ns), 
                max_tick=max(sess_ns), n=len(sess_ns), pad_p=1.0/(len(sess_ns)))
            for s, p in enumerate(lin_p_vals[li]):
                if not np.isnan(p) and p < p_val_thr_corr:
                    # between subplots
                    plot_util.add_signif_mark(sub_ax, sess_ns[s], pos, 
                        rel_y=1, col='k', fig_coord=True)


#############################################
def plot_per_mouse(sub_ax, mouse_st, sess_info, sess_ns=None, col=None, 
                   use_lab=True):
    """
    plot_per_mouse(sub_ax, mouse_st, sess_info)

    Plots statistics for each mouse.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - mouse_st (3D array)      : statistics across ROIs or seqs, 
                                     structured as mouse x session x stats
        - sess_info (list)         : list of dictionaries for each mouse 
                                     containing information from each session, 
                                     with None for missing sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['planes'] (list)     : imaging planes
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois_{}'] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ('raw', 'dff')

    Optional args:
        - sess_ns (array-like): session numbers for each session
                                default: None
        - col (str)           : plotting color
                                default: None
        - use_lab (bool)      : if True, label with mouse numbers is added
                                default: True
    """

    mouse_st = np.asarray(mouse_st)
    
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
                lab = f'{lab}{mouse_n}, '
            else: # if label to be included
                lab_use = f'{lab}{mouse_n}'

        # get non NaNs
        keep_idx = np.where(np.isfinite(mouse_st[m, :, 0]))[0]
        
        if sess_ns is None:
            sess_ns = np.asarray(range(len(mouse_st[m])))
        elif len(sess_ns) != len(mouse_st[m]):
            raise ValueError('Not as many session numbers as sessions.')

        plot_util.plot_errorbars(sub_ax, mouse_st[m, keep_idx, 0], 
            mouse_st[m, keep_idx, 1:].T, sess_ns[keep_idx], col=col, 
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
        - mouse_mes (array-like): mouse mean/median data (for legend), 
                                  structured as mouse x session
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
            raise ValueError('`mouse_mes` second dimension should be '
                f'length {len(sess_ns)}.')
        n_mice_per = np.sum(np.isfinite(mouse_mes[:, keep_idx]), axis=0)
        min_mice, max_mice = np.min(n_mice_per), np.max(n_mice_per)
        lab = f'n={min_mice}'
        if min_mice != max_mice:
            lab = f'{lab}-{max_mice}'

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
            rel_y = 0.09
        elif signs[idx] == -1:
            ys    = np.nansum([st_data[:, idx, 0], -st_data[:, idx, 1]], axis=0)
            y_val = np.nanmin(ys, axis=0).reshape([1])
            rel_y = -0.03
        plot_util.add_signif_mark(sub_ax, x_val, y_val, rel_y=rel_y, col=col)


#############################################
def plot_area_diff_per_linpla(sub_ax, sess_ns, mouse_diff_st, diff_st, CI_vals, 
                              sign_sess, sess_info, plot='tog', d='data', 
                              col='k', zero_line=False):
    """
    plot_area_diff_per_linpla(sub_ax, sess_ns, mouse_diff_st, diff_st, CI_vals,
                              sign_sess, sess_info)

    Plots data or CIs for a specific plane/line combination, and returns
    max y values for each session

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
            ['planes'] (list)     : imaging planes
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois_{}'] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ('raw', 'dff')

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

    Returns:
        if d == 'data':
        - maxes (1D array): max values y value for each session
    """

    sess_ns = np.asarray(sess_ns)
    diff_st = np.asarray(diff_st)

    # plot the mouse lines
    if d == 'data':
        mouse_st = np.asarray(mouse_diff_st)
        # get max y position
        maxes = np.nanmax(np.sum([diff_st[:, 0], diff_st[:, 1]], axis=0))
        if plot == 'sep':
            plot_per_mouse(
                sub_ax, mouse_st, sess_info, sess_ns, col, use_lab=True)
        elif plot in ['tog', 'grped']:
            plot_area_diff_stats(
                sub_ax, diff_st, sess_ns, mouse_st[:, :, 0], col)
        else:
            gen_util.accepted_values_error('plot', plot, ['sep', 'tog'])
        
        return maxes
    
    elif d == 'CIs':
        if zero_line:
            sub_ax.axhline(
                y=0, ls='dashed', c='k', lw=2.0, alpha=0.5, zorder=-13)
        CI_vals  = np.asarray(CI_vals)
        keep_idx = np.where(np.isfinite(CI_vals[:, 0]))[0] # get mask
        
        # Add CIs
        width = 0.45
        plot_util.plot_CI(sub_ax, CI_vals[keep_idx, 1:].T, CI_vals[keep_idx, 0], 
            sess_ns[keep_idx], med_rat=0.03, width=width, zorder=-12)
        if plot == 'sep': # Add mean/median lines
            plot_util.plot_lines(
                sub_ax, diff_st[keep_idx, 0], sess_ns[keep_idx], y_rat=0.025, 
                col=col, width=width)
            ypos_data = np.asarray(mouse_diff_st)
        else:
            ypos_data = np.expand_dims(diff_st, axis=0) 

        # plot significance markers
        signs = np.sign(diff_st[:, 0] - CI_vals[:, 0]).astype(int)
        plot_signif_from_mouse_diffs(
            sub_ax, sign_sess, ypos_data, signs, sess_ns, col)
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
                                         structured as plane/line x session 
                                                                  x stats
            ['mouse_diff_stats'] (list): difference statistics across ROIs or 
                                         seqs, structured as 
                                             plane/line x mouse x session 
                                                        x stats
            ['CI_vals'] (list)         : CIs values, structured as
                                             plane/line x session 
                                                        x perc (med, lo, high)
            ['sign_sess'] (list)       : significant session indices, 
                                         structured as plane/line (x tails)
            ['linpla_ord'] (list)      : order list of planes/lines
            if extrapar['datatype'] == 'roi':
                ['all_diff_st_grped'] (list): difference stats across ROIs 
                                              (grouped across mice), 
                                              structured as 
                                                plane/line x session x stats
                ['CI_vals_grped'] (list)    : CIs values across ROIs, 
                                              structured as 
                                                plane/line x session 
                                                    x perc (med, lo, high)
                ['lin_p_vals'] (list)       : p-values for each line comparison, 
                                              structured as line x session 
                                              (np.nan for sessions  missing in 
                                              either plane)
                ['max_comps_per'] (int)     : total number of comparisons
                ['p_vals_grped'] (list)     : p values for each comparison, 
                                              organized by session pairs (where 
                                              the second session is cycled in 
                                              the inner loop, e.g., 0-1, 0-2, 
                                              1-2, including empty groups)
                ['sign_sess_grped'] (list)  : significant session indices, 
                                              structured as plane/line (x tails)
                ['tot_n_comps'] (int)       : total number of comparisons

        - sess_info (nested list): nested list of dictionaries for each 
                                   line/plane x mouse containing information 
                                   from each session, with None for missing 
                                   sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['planes'] (list)     : imaging planes
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois_{}'] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ('raw', 'dff')
                
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
    if datatype == 'run':
        if plot == 'sep':
            error = 'None'
        elif plot == 'grped':
            raise ValueError('grped plot types only supported for ROI datatype.')
    
    dimstr = ''
    if plot == 'tog':
        dimstr = ' across mice'
    elif datatype == 'roi' and plot in ['sep', 'grped']:
        dimstr = f' across {sess_str_util.datatype_dim_str(datatype)}'
    
    grp_str, grp_str_pr = '', ''
    if plot == 'grped':
        grp_str = '_grped'
        grp_str_pr = ' (grouped)'
        
    stimstr_pr = sess_str_util.stim_par_str(
        stimpar['stimtype'], stimpar['bri_dir'], stimpar['bri_size'],
        stimpar['gabk'], 'print')
    statstr_pr = sess_str_util.stat_par_str(analyspar['stats'], error, 'print')
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar['dend'], sesspar['plane'], extrapar['datatype'], 'print')
    
    sess_ns = np.asarray(sesspar['sess_n'])
    if sesspar['sess_n'] in ['any', 'all']:
        sess_ns = np.arange(len(sess_info[0][0]['sess_ns'])) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.tolist())

    [lines, planes, linpla_iter, 
     pla_cols, _, n_plots] = sess_plot_util.fig_linpla_pars( 
        n_grps=len(diff_info['linpla_ord']))
    linpla_iter = [[d, ll] for d in ['data', 'CIs'] for ll in linpla_iter]
    figpar = sess_plot_util.fig_init_linpla(figpar)

    # correct p-value (Bonferroni)
    if plot == 'grped':
        p_val_thr = 0.05
        if diff_info['max_comps_per'] != 0:
            p_val_thr_corr = p_val_thr/diff_info['max_comps_per']
        else:
            p_val_thr_corr = p_val_thr
        sig_comps = [[] for _ in range(len(linpla_iter))]
        maxes = np.full([len(linpla_iter), len(sess_ns)], np.nan)

    subtitle = 'Surp - reg activity'
    if lock:
        prepost_str = '{}s pre v post'.format(stimpar['post'])
        if lock == 'surplock':
            subtitle += ' locked to surprise onset'
        elif lock == 'reglock':
            subtitle = 'Reg - surp activity locked to regular onset'
        else:
            raise ValueError('If lock is not False, it must be `reglock` or \
                `surplock`.')
    else:
        prepost_str = sess_str_util.prepost_par_str(
            stimpar['pre'], stimpar['post'], str_type='print')
    title = (f'{subtitle}\n({prepost_str} seqs) for {stimstr_pr}\n{statstr_pr}'
        f'{dimstr}{grp_str_pr} (sess {sess_ns_str}{dendstr_pr})')

    fig, ax = plot_util.init_fig(n_plots, **figpar['init'])
    
    for i, (d, [line, pla]) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        l_idx = get_linpla_idx(
            diff_info['linpla_ord'], line, pla, verbose=(d==0), newline=(i==0))
        if l_idx is None:
            continue

        ypos = plot_area_diff_per_linpla(ax[pl, li], sess_ns, 
            diff_info['mouse_diff_stats'][l_idx], 
            diff_info[f'all_diff_stats{grp_str}'][l_idx], 
            diff_info[f'CI_vals{grp_str}'][l_idx],
            diff_info[f'sign_sess{grp_str}'][l_idx], sess_info[l_idx],
            plot=plot, d=d, col=pla_cols[pl], zero_line=False)
 
        # check p_val signif
        if d == 'data' and plot == 'grped':
            maxes[i] = ypos
            all_p_vals = diff_info['p_vals_grped'][l_idx]
            for p, p_val in enumerate(all_p_vals):
                if not np.isnan(p_val) and p_val < p_val_thr_corr:
                    sig_comps[i].append(p)

    if plot == 'grped':
        plot_data_signif(ax, sess_ns, sig_comps, diff_info['lin_p_vals'], 
            maxes, p_val_thr=0.05, n_comps=diff_info['max_comps_per'])
    fig.suptitle(title, y=1.03, weight='bold')

    # Add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(ax, fluor=analyspar['fluor'], 
        area=True, datatype=datatype, lines=lines, planes=planes, 
        xticks=sess_ns, xlab='Sessions', adj_yticks=True, kind='reg')

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
            ['planes'] (list)     : imaging planes
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois_{}'] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ('raw', 'dff')
        - diff_info (dict): dictionary containing difference information
            ['all_diff_stats'] (list)  : difference stats across mice, 
                                         structured as plane/line x session 
                                                                  x stats
            ['mouse_diff_stats'] (list): difference statistics across ROIs or 
                                         seqs, structured as 
                                             plane/line x mouse x session 
                                                        x stats
            ['CI_vals'] (list)         : CIs values, structured as
                                             plane/line x session 
                                                        x perc (med, lo, high)
            ['sign_sess'] (list)       : significant session indices, 
                                         structured as plane/line (x tails)
            ['linpla_ord'] (list)      : order list of planes/lines
            if extrapar['datatype'] == 'roi':
                ['all_diff_st_grped'] (list): difference stats across ROIs 
                                              (grouped across mice), 
                                              structured as 
                                                plane/line x session x stats
                ['CI_vals_grped'] (list)    : CIs values across ROIs, 
                                              structured as 
                                                plane/line x session 
                                                    x perc (med, lo, high)
                ['lin_p_vals'] (list)       : p-values for each line comparison, 
                                              structured as line x session 
                                              (np.nan for sessions  missing in 
                                              either plane)
                ['max_comps_per'] (int)     : total number of comparisons
                ['p_vals_grped'] (list)     : p values for each comparison, 
                                              organized by session pairs (where 
                                              the second session is cycled in 
                                              the inner loop, e.g., 0-1, 0-2, 
                                              1-2, including empty groups)
                ['sign_sess_grped'] (list)  : significant session indices, 
                                              structured as plane/line (x tails)
                ['tot_n_comps'] (int)       : total number of comparisons

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
    
    sessstr = sess_str_util.sess_par_str(
        sesspar['sess_n'], stimpar['stimtype'], sesspar['plane'], 
        stimpar['bri_dir'], stimpar['bri_size'], stimpar['gabk'])
    dendstr = sess_str_util.dend_par_str(
        analyspar['dend'], sesspar['plane'], extrapar['datatype'])
    datatype = extrapar['datatype']

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    base_str = sess_str_util.base_par_str(basepar['baseline'])[1:]

    if savedir is None:
        savedir = os.path.join(
            figpar['dirs'][datatype], 
            figpar['dirs']['acr_sess'], 
            sess_str_util.get_stimdir(stimpar['stimtype'], stimpar['gabfr']), 
            base_str)

    gen_savename = f'{datatype}_surp_diff_{sessstr}{dendstr}'
    part = 'surp_diff'
    add_idx = gen_savename.find(part) + len(part)

    for p, plot in enumerate(['sep', 'tog', 'grped']):
        if plot == 'grped' and datatype == 'run':
            continue
        fig = plot_area_diff_acr_sess(
            analyspar, sesspar, stimpar, extrapar, sess_info, diff_info, 
            figpar=figpar, lock=False, plot=plot)


        savename = f'{gen_savename[:add_idx]}_{plot}{gen_savename[add_idx:]}'
        fulldir = plot_util.savefig(
            fig, savename, savedir, print_dir=(p==0), **figpar['save'])

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
            ['planes'] (list)     : imaging planes
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois_{}'] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ('raw', 'dff')
        - diff_info (list): list of dictionaries containing difference 
                            information, structured as [surp-locked, reg-locked]
            ['all_diff_stats'] (list)  : difference stats across mice, 
                                         structured as plane/line x session 
                                                                  x stats
            ['mouse_diff_stats'] (list): difference statistics across ROIs or 
                                         seqs, structured as 
                                             plane/line x mouse x session 
                                                        x stats
            ['CI_vals'] (list)         : CIs values, structured as
                                             plane/line x session 
                                                        x perc (med, lo, high)
            ['sign_sess'] (list)       : significant session indices, 
                                         structured as plane/line (x tails)
            ['linpla_ord'] (list)      : order list of planes/lines
            if extrapar['datatype'] == 'roi':
                ['all_diff_st_grped'] (list): difference stats across ROIs 
                                              (grouped across mice), 
                                              structured as 
                                                plane/line x session x stats
                ['CI_vals_grped'] (list)    : CIs values across ROIs, 
                                              structured as 
                                                plane/line x session 
                                                    x perc (med, lo, high)
                ['lin_p_vals'] (list)       : p-values for each line comparison, 
                                              structured as line x session 
                                              (np.nan for sessions  missing in 
                                              either plane)
                ['max_comps_per'] (int)     : total number of comparisons
                ['p_vals_grped'] (list)     : p values for each comparison, 
                                              organized by session pairs (where 
                                              the second session is cycled in 
                                              the inner loop, e.g., 0-1, 0-2, 
                                              1-2, including empty groups)
                ['sign_sess_grped'] (list)  : significant session indices, 
                                              structured as plane/line (x tails)
                ['tot_n_comps'] (int)       : total number of comparisons

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
 
    sessstr = sess_str_util.sess_par_str(
        sesspar['sess_n'], stimpar['stimtype'], sesspar['plane'], 
        stimpar['bri_dir'], stimpar['bri_size'], stimpar['gabk'])
    dendstr = sess_str_util.dend_par_str(
        analyspar['dend'], sesspar['plane'], extrapar['datatype'])
    datatype = extrapar['datatype']

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    base_str = sess_str_util.base_par_str(basepar['baseline'])[1:]

    if savedir is None:
        savedir = os.path.join(
            figpar['dirs'][datatype], 
            figpar['dirs']['acr_sess'], 
            sess_str_util.get_stimdir(stimpar['stimtype'], stimpar['gabfr']), 
            base_str)

    gen_savename = f'{datatype}_lock_diff_{sessstr}{dendstr}'

    for l, lock in enumerate(['surplock', 'reglock']):
        part = 'lock_diff'
        lock_savename = gen_savename.replace('lock', lock)
        add_idx = lock_savename.find(part) + len(part)

        for p, plot in enumerate(['sep', 'tog', 'grped']):
            if plot == 'grped' and datatype == 'run':
                continue
            fig = plot_area_diff_acr_sess(
                analyspar, sesspar, stimpar, extrapar, sess_info[l], 
                diff_info[l], figpar=figpar, lock=lock, plot=plot)

            savename = (f'{lock_savename[:add_idx]}_{plot}'
                f'{lock_savename[add_idx:]}')
            fulldir = plot_util.savefig(
                fig, savename, savedir, print_dir=(p==0), **figpar['save'])

    return fulldir, gen_savename


#############################################
def plot_traces(sub_ax, xran, trace_st, lock=False, col='k', lab=True, 
                stimtype='gabors', gabfr=0):
    """
    plot_traces(sub_ax, xran, trace_st)

    Plot regular and surprise data traces (single set).

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

    if lock:
        cols = [col, col]
    else:
        cols = ['gray', col]
    names = ['reg', 'surp']
    xran = np.asarray(xran)


    # horizontal and vertical 0 line
    if not sub_ax.is_last_row():
        alpha = 0.5
    else:
        # ensures that if no data is plotted in these subplots, there is at 
        # least a vertically finite horizontal object to prevent indefinite 
        # axis expansion. (Occurs when, for one axis, there are only indefinite
        # objects, e.g. shading or h/vline.) 
        alpha = 0
    sub_ax.axhline(y=0, ls='dashed', c='k', lw=3.0, alpha=alpha, zorder=-13)
    sub_ax.axvline(x=0, ls='dashed', c='k', lw=3.0, alpha=0.5, zorder=-13)
        

    xticks = None
    if lock:
        xticks = np.linspace(-np.max(xran), np.max(xran), 5)

    if lock:
        rev_xran = xran[::-1] * -1
        full_xran = np.concatenate([rev_xran, xran])
    else:
        full_xran = xran

    if stimtype == 'gabors':
        shade_col = '#871719' # dark red
        if lock == 'reglock':
            sh_xran = full_xran[np.where(full_xran <= 0)][1:]
            rest_xran = full_xran[np.where(full_xran > 0)][:-1]
        else:
            sh_xran = full_xran[np.where(full_xran >= 0)][:-1]
            rest_xran = full_xran[np.where(full_xran < 0)][1:]
        for sub_xran, sh_col in zip([sh_xran, rest_xran], [shade_col, 'none']):
            if len(sub_xran) == 0:
                continue
            sess_plot_util.plot_gabfr_pattern(sub_ax, sub_xran, offset=gabfr, 
                bars_omit=[0], shade_col=sh_col, alpha=0.2)

    trace_st = np.asarray(trace_st)
    for i, (col, name) in enumerate(zip(cols, names)):
        label = name if lab and not lock else None
        if lock and name not in lock:
                xran_use = rev_xran
        else:
            xran_use = xran
        if lock == 'reglock':
            i = 1 - i # data ordered as [surp, reg] instead of vv
        plot_util.plot_traces(sub_ax, xran_use, trace_st[i, :, 0], 
            trace_st[i, :, 1:], label=label, alpha_line=0.8, col=col, 
            xticks=xticks)


#############################################
def plot_single_prog(sub_ax, prog_st, col='k', label=None, alpha_line=0.8):
    """
    plot_single_prog(sub_ax, prog_st)

    Plot surprise or regular progression data (single set).

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - prog_st (4D array)       : progression statistics, structured as
                                         seqs x stats

    Optional args:
        - col  (str)      : colour for data
                            default: 'k'
        - alpha_line (num): plt alpha variable controlling line transparency                          transparency (from 0 to 1)
                            default: 0.5
        - label (str)     : label for legend
                            default: None

    """

    prog_st = np.asarray(prog_st)
    xran = np.arange(prog_st.shape[0])

    # ensures that if no data is plotted in these subplots, there is at 
    # least a vertically finite horizontal object to prevent indefinite 
    # axis expansion. (Occurs when, for some axes, there are only 
    # indefinite objects, e.g. shading or h/vline.) 
    sub_ax.axvline(x=0, ls='dashed', c='k', lw=3.0, alpha=0, zorder=-13)
     
    sub_ax.axhline(y=0, ls='dashed', c='k', lw=3.0, alpha=0.5, zorder=-13)
        
    plot_util.plot_traces(sub_ax, xran, prog_st[:, 0], prog_st[:, 1:], 
        alpha_line=alpha_line, col=col, label=label)


#############################################
def plot_traces_acr_sess(analyspar, sesspar, stimpar, extrapar, sess_info, 
                         trace_info, figpar=None, lock=False, grp='mice'):
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
                                   line/plane x mouse containing information 
                                   from each session, with None for missing 
                                   sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['planes'] (list)     : imaging planes
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois_{}'] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ('raw', 'dff')
        - trace_info (dict)      : dictionary with difference info
            ['linpla_ord'] (list) : order list of planes/lines
            ['trace_stats'] (list): trace statistics, structured as
                                    plane/line x session x reg/surp x frame 
                                               x stats
            ['xran'] (list)       : second values for each frame
            if datatype == 'roi':
                ['trace_st_acr_rois'] (list): trace statistics across ROIs, 
                                              grouped across mice, structured 
                                              as session x reg/surp 
                                                         x frame x stats
                                              (or surp/reg if surp == 'reglock')

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
        - grp (str)         : if 'mice', data is grouped per mouse, if 'roi', 
                              data is grouped across mice
                              default: 'mice'
    Returns:
        - fig (plt Figure) : pyplot figure
    """
 
    datatype = extrapar['datatype']
    stimstr_pr = sess_str_util.stim_par_str(
        stimpar['stimtype'], stimpar['bri_dir'], stimpar['bri_size'],
        stimpar['gabk'], 'print')
    statstr_pr = sess_str_util.stat_par_str(
        analyspar['stats'], analyspar['error'], 'print')
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar['dend'], sesspar['plane'], extrapar['datatype'], 'print')
    
    sess_ns = np.asarray(sesspar['sess_n'])
    if sesspar['sess_n'] in ['any', 'all']:
        sess_ns = np.asarray(range(len(sess_info[0][0]['sess_ns']))) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.tolist())
    n_sess = len(sess_ns)

    [lines, planes, linpla_iter, pla_cols, _, n_plots] = \
        sess_plot_util.fig_linpla_pars(
            n_sess=n_sess, n_grps=len(trace_info['linpla_ord']))
    figpar = sess_plot_util.fig_init_linpla(figpar, traces=n_sess)

    subtitle = 'Surp v reg activity'
    if lock:
        prepost_str = '{}s pre v post'.format(stimpar['post'])
        if lock == 'surplock':
            subtitle += ' locked to surprise onset'
        elif lock == 'reglock':
            subtitle = 'Reg v surp activity locked to regular onset'
        else:
            raise ValueError('If lock is not False, it must be `reglock` or \
                `surplock`.')
    else:
        prepost_str = sess_str_util.prepost_par_str(
            stimpar['pre'], stimpar['post'], str_type='print')
    
    dim_str = 'mice'
    grp_str = ''
    if grp == 'rois':
        if datatype == 'run':
            raise ValueError(f'Grouping (`grp`) only across ROIs for ROI '
                'datatype.')
        dim_str = 'ROIs (grouped)'
        grp_str = '_grped'
    title = (f'{subtitle}\n({prepost_str} seqs) for {stimstr_pr}\n'
        f'{statstr_pr} across {dim_str} (sess {sess_ns_str}{dendstr_pr})')

    fig, ax = plot_util.init_fig(n_plots, **figpar['init'])
    
    xran = trace_info['xran']
    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        l_idx = get_linpla_idx(
            trace_info['linpla_ord'], line, pla, verbose=True, newline=(i==0))
        if l_idx is None:
            continue

        # plot the mouse traces
        for s in range(n_sess):
            sub_ax = ax[s + pl * n_sess, li]
            lab = (li == 0 and s == 0)
            plot_traces(
                sub_ax, xran, trace_info[f'trace_stats{grp_str}'][l_idx][s], 
                lock, pla_cols[pl], lab, stimpar['stimtype'], stimpar['gabfr'])

    # Add plane, line info to plots
    if stimpar['stimtype'] == 'gabors':
        step = 0.3
        max_tick = np.max(xran)//step * step
        if lock:
            min_tick = - max_tick
        else:
            min_tick = np.min(xran)//step * step
        n_ticks = np.max([int((max_tick - min_tick)/step) % 5, 5]) + 1
        xticks = [np.around(v, 1) 
            for v in np.linspace(min_tick, max_tick, n_ticks)]
    else:
        xticks = [np.around(v, 1) for v in sub_ax.get_xticks()]

    sess_plot_util.format_linpla_subaxes(ax, fluor=analyspar['fluor'], 
        area=False, datatype=datatype, lines=lines, planes=planes, 
        sess_ns=sess_ns, xticks=xticks, kind='traces')

    fig.suptitle(title, y=1.1, weight='bold')

    return fig
   

#############################################
def plot_surp_traces(analyspar, sesspar, stimpar, basepar, extrapar, sess_info, 
                     trace_info, figpar=None, savedir=None):
    """
    plot_surp_traces(analyspar, sesspar, stimpar, basepar, extrapar, sess_info, 
                     trace_info)

    From dictionaries, plots traces across ROIs or mice of difference 
    for regular and surprise averaged across sequences.
    
    Returns general figure name and save directory path.
    
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
                                   line/plane x mouse containing information 
                                   from each session, with None for missing 
                                   sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['planes'] (list)     : imaging planes
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois_{}'] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ('raw', 'dff')
        - trace_info (dict)      : dictionary with difference info
            ['linpla_ord'] (list) : order list of planes/lines            
            ['trace_stats'] (list): trace statistics, structured as
                                    plane/line x session x reg/surp x frame 
                                               x stats
            if datatype == 'roi':
                ['trace_st_acr_rois'] (list): trace statistics across ROIs, 
                                              grouped across mice, structured 
                                              as session x reg/surp 
                                                         x frame x stats
                                              (or surp/reg if surp == 'reglock')
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
    
    sessstr = sess_str_util.sess_par_str(
        sesspar['sess_n'], stimpar['stimtype'], sesspar['plane'], 
        stimpar['bri_dir'], stimpar['bri_size'], stimpar['gabk'])
    dendstr = sess_str_util.dend_par_str(
        analyspar['dend'], sesspar['plane'], extrapar['datatype'])
    datatype = extrapar['datatype']

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    base_str = sess_str_util.base_par_str(basepar['baseline'])[1:]
    if savedir is None:
        savedir = os.path.join(
            figpar['dirs'][datatype], 
            figpar['dirs']['acr_sess'], 
            sess_str_util.get_stimdir(stimpar['stimtype'], stimpar['gabfr']), 
            base_str)

    gen_savename = f'{datatype}_surp_tr_{sessstr}{dendstr}'

    for grp in ['mice', 'rois']:
        if grp == 'rois' and datatype == 'run':
            continue
        fig = plot_traces_acr_sess(
            analyspar, sesspar, stimpar, extrapar, sess_info, trace_info, 
            figpar=figpar, lock=False, grp=grp)
        
        savename = gen_savename
        if grp == 'rois':
            savename = f'{gen_savename}_grped'
    
        fulldir = plot_util.savefig(fig, savename, savedir, **figpar['save'])

    return fulldir, gen_savename


#############################################
def plot_lock_traces(analyspar, sesspar, stimpar, basepar, extrapar, sess_info, 
                     trace_info, figpar=None, savedir=None):
    """
    plot_lock_traces(analyspar, sesspar, stimpar, extrapar, sess_info,
                     trace_info)

    From dictionaries, plots traces across ROIs or mice for regular and 
    surprise sequences, locked to transitions from regular to surprise, then
    from surprise to regular.
    
    Returns general figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - basepar (dict)  : dictionary with keys of BasePar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ['analysis'] (str)        : analysis type (e.g., 't')
            ['datatype'] (str)        : datatype (e.g., 'run', 'roi')
        - sess_info (list): list of dictionaries containing information from 
                            each session, structured as 
                            [surp-locked, reg-locked]
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['planes'] (list)     : imaging planes
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois_{}'] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ('raw', 'dff')
        - trace_info (list): list of dictionaries containing trace 
                             information, structured as 
                             [surp-locked, reg-locked]
            ['linpla_ord'] (list) : order list of planes/lines            
            ['trace_stats'] (list): trace statistics, structured as
                                    plane/line x session x reg/surp x frame 
                                               x stats
            if datatype == 'roi':
                ['trace_st_grped'] (list): trace statistics across ROIs, 
                                           grouped across mice, structured 
                                           as session x reg/surp 
                                                      x frame x stats
                                           (or surp/reg if surp == 'reglock')
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
 
    sessstr = sess_str_util.sess_par_str(
        sesspar['sess_n'], stimpar['stimtype'], sesspar['plane'], 
        stimpar['bri_dir'], stimpar['bri_size'], stimpar['gabk'])
    dendstr = sess_str_util.dend_par_str(
        analyspar['dend'], sesspar['plane'], extrapar['datatype'])
    datatype = extrapar['datatype']

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    gen_savename = f'{datatype}_lock_tr_{sessstr}{dendstr}'

    stimdir = sess_str_util.get_stimdir(stimpar['stimtype'], stimpar['gabfr'])

    for l, lock in enumerate(['surplock', 'reglock']):
        for grp in ['mice', 'rois']:
            if grp == 'rois' and datatype == 'run':
                continue
            fig = plot_traces_acr_sess(analyspar, sesspar, stimpar, extrapar, 
                sess_info[l], trace_info[l], figpar=figpar, lock=lock, grp=grp)

            base_str = sess_str_util.base_par_str(basepar['baseline'])[1:]
            if savedir is None:
                savedir = os.path.join(
                    figpar['dirs'][datatype], 
                    figpar['dirs']['acr_sess'], 
                    stimdir, 
                    base_str)

            savename = gen_savename.replace('lock', lock)
            if grp == 'rois':
                savename = f'{savename}_grped'
        
            fulldir = plot_util.savefig(
                fig, savename, savedir, **figpar['save'])

    return fulldir, gen_savename


#############################################
def plot_prog_per_mouse(ax, mouse_st, sess_info, sess_ns=None, col=None, 
                        use_lab=True):
    """
    plot_prog_per_mouse(ax, mouse_st, sess_info)

    Plots surprise/regular sequence progression statistics for each mouse.

    Required args:
        - ax (1D array of subplots): array of subplots
        - mouse_st (3D array)      : statistics across ROIs or seqs, 
                                     structured as mouse x session x stats
        - sess_info (list)         : list of dictionaries for each mouse 
                                     containing information from each session, 
                                     with None for missing sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['planes'] (list)     : imaging planes
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois_{}'] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ('raw', 'dff')

    Optional args:
        - sess_ns (array-like): session numbers for each session
                                default: None
        - col (str)           : plotting color
                                default: None
        - use_lab (bool)      : if True, label with mouse numbers is added
                                default: True
    """

    mouse_st = np.asarray(mouse_st)

    if sess_ns is None:
        sess_ns = np.asarray(range(len(mouse_st[0])))
    elif len(sess_ns) != len(mouse_st[0]):
        raise ValueError('Not as many session numbers as sessions.')

    labs = [None for _ in sess_ns]
    labs_used = [False for _ in sess_ns]
    for i, s in enumerate(sess_ns):
        mouse_ns = []
        for m, m_info in enumerate(sess_info):
            if np.isfinite(np.asarray(mouse_st[m][i])).any():
                mouse_ns.append(str(m_info['mouse_ns'][i]))
        labs[i] = 'M{}'.format(', '.join(mouse_ns))

    for m, m_info in enumerate(sess_info):
        for i, s in enumerate(sess_ns):
            lab = None
            if not labs_used[i]:
                lab = labs[i]
                labs_used[i] = True
            stats = np.asarray(mouse_st[m][i])
            if np.isfinite(np.asarray(mouse_st[m][i])).any():
                plot_single_prog(
                    ax[i], stats, col=col, label=lab, alpha_line=0.6)


#############################################
def plot_prog_acr_sess(analyspar, sesspar, stimpar, extrapar, sess_info, 
                       prog_info, figpar=None, prog='progsurp', 
                       plot='tog'):
    """
    plot_prog_acr_sess(analyspar, sesspar, stimpar, extrapar, sess_info, 
                       prog_info)

    From dictionaries, plots progression of surprise or regular sequences 
    within sessions across ROIs or mice.
    
    Returns figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ['analysis'] (str): analysis type (e.g., 't')
            ['datatype'] (str): datatype (e.g., 'run', 'roi')
            ['position'] (str): position plotted (e.g., 'first', 'second', etc.)
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/plane x mouse containing information 
                                   from each session, with None for missing 
                                   sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['planes'] (list)     : imaging planes
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois_{}'] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ('raw', 'dff')
        - prog_info (dict)       : dictionary with progression info
            ['linpla_ord'] (list)       : order list of planes/lines            
            ['prog_stats'] (list)  : surprise progression stats across mice, 
                                          structured as plane/line x session 
                                            x seq x stats
            ['mouse_prog_stats'] (list): surprise progression stats across 
                                          ROIs, structured as 
                                             plane/line x mouse x session 
                                                x seq x stats
            if extrapar['datatype'] == 'roi':
                ['prog_stats_grped'] (list): surprise progression stats across 
                                              ROIs (grouped across mice), 
                                              structured as 
                                                 plane/line x session x seqs x 
                                                 stats

    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
        - prog (str)   : if 'progsurp' or 'progreg', differences between
                         surprise and previous regular are plotted. If 
                         'progreg', v.v. 
                         default: 'progsurp'     
        - plot (str)   : if 'tog', data is grouped across mouse, if 'grped', 
                         data is grouped across ROIs, if 'sep', data is 
                         separated by mouse
                         default: 'tog'
    Returns:
        - fig (plt Figure) : pyplot figure
    """
 
    datatype = extrapar['datatype']
    stimstr_pr = sess_str_util.stim_par_str(
        stimpar['stimtype'], stimpar['bri_dir'], stimpar['bri_size'],
        stimpar['gabk'], 'print')
    statstr_pr = sess_str_util.stat_par_str(
        analyspar['stats'], analyspar['error'], 'print')
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar['dend'], sesspar['plane'], extrapar['datatype'], 'print')
    pos = extrapar['position']

    sess_ns = np.asarray(sesspar['sess_n'])
    if sesspar['sess_n'] in ['any', 'all']:
        sess_ns = np.asarray(range(len(sess_info[0][0]['sess_ns']))) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.tolist())
    n_sess = len(sess_ns)

    [lines, planes, linpla_iter, pla_cols, _, n_plots] = \
        sess_plot_util.fig_linpla_pars(n_sess=n_sess, 
            n_grps=len(prog_info['linpla_ord']))
    figpar = sess_plot_util.fig_init_linpla(figpar, prog=n_sess)

    prepost_str = sess_str_util.prepost_par_str(
            stimpar['pre'], stimpar['post'], str_type='print')

    if prog == 'progsurp':
        substrs = ['surprise', 'regular']
    elif prog == 'progreg':
        substrs = ['regular', 'surprise']
    else:
        raise ValueError('If prog is not False, it must be `progreg` or \
            `progsurp`.')
    subtitle = 'Progression of each {} {} vs preceeding {} sequence'.format(
        pos, *substrs)

    grp_str_pr = ''
    if plot == 'tog':
        key_str = 'prog_stats'
        dim_str = ' across mice'
    elif plot in ['sep', 'grped']:
        dim_str = f' across {sess_str_util.datatype_dim_str(datatype)}'
        key_str = 'mouse_prog_stats'
        if plot == 'grped':
            if datatype == 'run':
                raise ValueError(f'Grouping (`grp`) only across ROIs for ROI '
                    'datatype.')
            key_str = 'prog_stats_grped'
            grp_str_pr = ' (grouped)'
    else:
        gen_util.accepted_values_error('plot', plot, ['tog', 'sep', 'grped'])
    
    title = (f'{subtitle}\n({prepost_str} seqs) for {stimstr_pr} '
        f'{statstr_pr}{dim_str}{grp_str_pr} (sess {sess_ns_str}{dendstr_pr})')

    fig, ax = plot_util.init_fig(n_plots, **figpar['init'])
    
    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        l_idx = get_linpla_idx(
            prog_info['linpla_ord'], line, pla, verbose=True, newline=(i==0))
        if l_idx is None:
            continue

        if plot == 'sep':
            plot_prog_per_mouse(
                ax[pl, li * n_sess:(li + 1) * n_sess], 
                prog_info[key_str][l_idx], sess_info[l_idx], sess_ns, 
                pla_cols[pl], use_lab=True)
        else:
            for s in range(n_sess):
                sub_ax = ax[pl, s + li * n_sess]
                # lab = (pl == 0 and s == 0)
                plot_single_prog(
                    sub_ax, prog_info[key_str][l_idx][s], pla_cols[pl])

    if plot == 'sep':
        sub_ax = ax.reshape(-1)[-1]

    # Add plane, line info to plots
    xticks = [int(v) for v in sub_ax.get_xticks()]

    sess_plot_util.format_linpla_subaxes(ax, fluor=analyspar['fluor'], 
        area=False, datatype=datatype, lines=lines, planes=planes, 
        sess_ns=sess_ns, xticks=xticks, kind='prog')

    fig.suptitle(title, y=1.25, weight='bold')

    return fig


#############################################
def plot_prog(analyspar, sesspar, stimpar, basepar, extrapar, sess_info, 
              prog_info, figpar=None, savedir=None):
    """
    plot_prog(analyspar, sesspar, stimpar, basepar, extrapar, sess_info, 
              prog_info)

    From dictionaries, plots XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX










    
    Returns general figure name and save directory path.
    
    Required args:
        - analyspar (dict): dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)  : dictionary with keys of SessPar namedtuple
        - stimpar (dict)  : dictionary with keys of StimPar namedtuple
        - stimpar (dict)  : dictionary with keys of BasePar namedtuple
        - extrapar (dict) : dictionary containing additional analysis 
                            parameters
            ['analysis'] (str): analysis type (e.g., 'g')
            ['datatype'] (str): datatype (e.g., 'run', 'roi')
            ['position'] (str): position plotted (e.g., 'first', 'second', etc.)
        - sess_info (list): list of dictionaries containing information from 
                            each session, structured as 
                            [progsurp, progreg]
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['planes'] (list)     : imaging planes
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois_{}'] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ('raw', 'dff')
        - prog_info (list): list of dictionaries containing prog surprise or 
                             regular sequence information, structured as 
                             [progsurp, progreg]
            ['linpla_ord'] (list)       : order list of planes/lines            
            ['prog_stats'] (list)       : surprise progression stats across 
                                          mice, structured as 
                                            plane/line x session x seq x stats
            ['mouse_prog_stats'] (list): surprise progression stats across ROIs, 
                                          structured as 
                                             plane/line x mouse x session 
                                                x seq x stats
            if extrapar['datatype'] == 'roi':
                ['prog_stats_grped'] (list): surprise progression stats across 
                                             ROIs (grouped across mice), 
                                            structured as 
                                                 plane/line x session x seqs x 
                                                 stats

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

    sessstr = sess_str_util.sess_par_str(
        sesspar['sess_n'], stimpar['stimtype'], sesspar['plane'], 
        stimpar['bri_dir'], stimpar['bri_size'], stimpar['gabk'])
    dendstr = sess_str_util.dend_par_str(
        analyspar['dend'], sesspar['plane'], extrapar['datatype'])
    datatype = extrapar['datatype']
    pos = extrapar['position']

    if figpar is None:
        figpar = sess_plot_util.init_figpar()

    figpar = copy.deepcopy(figpar)
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    gen_savename = f'{datatype}_prog_{pos}_{sessstr}{dendstr}'
    part = f'prog_{pos}'
    add_idx = gen_savename.find(part) + len(part)

    stimdir = sess_str_util.get_stimdir(stimpar['stimtype'], stimpar['gabfr'])

    for l, prog in enumerate(['progsurp', 'progreg']):
        for plot in ['sep', 'tog', 'grped']:
            if plot == 'grped' and datatype == 'run':
                continue
            fig = plot_prog_acr_sess(analyspar, sesspar, stimpar, extrapar, 
                sess_info[l], prog_info[l], figpar=figpar, prog=prog, 
                plot=plot)

            base_str = sess_str_util.base_par_str(basepar['baseline'])[1:]
            if savedir is None:
                savedir = os.path.join(
                    figpar['dirs'][datatype], 
                    figpar['dirs']['acr_sess'], 
                    stimdir,
                    base_str,
                    figpar['dirs']['prog'])

            savename = \
                f'{gen_savename[:add_idx]}_{plot}{gen_savename[add_idx:]}'
            savename = savename.replace('prog', prog)

            fulldir = plot_util.savefig(
                fig, savename, savedir, **figpar['save'])

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
            ['planes'] (list)     : imaging planes
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois_{}'] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ('raw', 'dff')

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
                    add = f'{add} ({nrois}'
            else:
                if datatype == 'roi':
                    add = f'/{nrois}'
                else:
                    add = ''
            labels[m] = f'{labels[m]}{add}'
            if nrois == 0:
                continue
            elif '#' in labels[m]:
                labels[m] = labels[m].replace(
                    '#', str(sess_info[s]['mouse_ns'][m]))
            div_fact = np.max([1, len(lat_data[s][m])//50])
            alpha_spec = alpha/div_fact
            clouds[m] = plot_util.plot_data_cloud(
                sub_ax, sess_n, lat_data[s][m], 0.15, label=None, 
                col=m_cols[m], alpha=alpha_spec, zorder=-11)
            maxes[s] = np.nanmax([maxes[s], np.nanmax(lat_data[s][m])])
    if datatype == 'roi':
        labels = [f'{label} ROIs)' for label in labels]
    sub_ax.legend(clouds, labels, fontsize='small')

    return maxes


#############################################
def plot_surp_latency(analyspar, sesspar, stimpar, latpar, extrapar, sess_info, 
                      lat_data, permpar=None, figpar=None, savedir=None):
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
        - latpar (LatPar)        : dictionary with keys of LatPar namedtuple
        - extrapar (dict)        : dictionary containing additional analysis 
                                   parameters
            ['analysis'] (str): analysis type (e.g., 't')
            ['datatype'] (str): datatype (e.g., 'run', 'roi')
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/plane x session containing information 
                                   from each mouse, with None for missing 
                                   sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['planes'] (list)     : imaging planes
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois_{}'] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ('raw', 'dff')
        - lat_data (dict)        : dictionary with latency info
            ['linpla_ord'] (list)  : ordered list of planes/lines            
            ['lat_stats'] (list)   : latency statistics, structured as
                                         plane/line x session x stats
            ['lat_vals'] (list)    : latency values for each ROI, structured as
                                         plane/line x session x mouse
            ['lat_p_vals'] (list)  : p-values for each latency comparison within 
                                     session pairs, (where the second session is 
                                     cycled in the inner loop, e.g., 0-1, 0-2, 
                                     1-2, including None sessions)
                                     structured as plane/line x comp
            ['lin_p_vals'] (list)  : p-values for each line comparison, 
                                     structured as line x session (np.nan for 
                                     sessions  missing in either plane)
            ['max_comps_per'] (int): total number of comparisons
            ['n_sign_rois] (list)  : number of significant ROIs, structured as 
                                     plane/line x session
            ['tot_n_comps'] (int)  : total number of comparisons
            
    Optional args:
        - permpar (PermPar): dictionary with keys of PermPar namedtuple
                             default: None
        - figpar (dict)    : dictionary containing the following figure parameter 
                             dictionaries
                             default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
        - savedir (str)    : path of directory in which to save plots.
                             default: None    
    
    Returns:
        - fulldir (str) : final name of the directory in which the figure 
                          is saved (may differ from input savedir, if 
                          datetime subfolder is added.)
        - savename (str): name under which the figure is saved
    """
 
    sessstr = sess_str_util.sess_par_str(
        sesspar['sess_n'], stimpar['stimtype'], sesspar['plane'], 
        stimpar['bri_dir'], stimpar['bri_size'], stimpar['gabk'])
    dendstr = sess_str_util.dend_par_str(
        analyspar['dend'], sesspar['plane'], extrapar['datatype'])
    latstr = sess_str_util.lat_par_str(
        latpar['method'], latpar['p_val_thr'], latpar['rel_std'])

    stimstr_pr = sess_str_util.stim_par_str(
        stimpar['stimtype'], stimpar['bri_dir'], stimpar['bri_size'],
        stimpar['gabk'], 'print')
    statstr_pr = sess_str_util.stat_par_str(
        analyspar['stats'], analyspar['error'], 'print')
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar['dend'], sesspar['plane'], extrapar['datatype'], 'print')
    latstr_pr = sess_str_util.lat_par_str(
        latpar['method'], latpar['p_val_thr'], latpar['rel_std'], 'print')
    surp_resp_str = ''
    if latpar['surp_resp']:
        surp_resp_str = '_surp_resp'
    
    datatype = extrapar['datatype']

    sess_ns = np.asarray(sesspar['sess_n'])
    if sesspar['sess_n'] in ['any', 'all']:
        sess_ns = np.asarray(range(len(sess_info[0]))) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.tolist())
    n_sess = len(sess_ns)

    [lines, planes, linpla_iter, 
    pla_cols, pla_col_names, n_plots] = sess_plot_util.fig_linpla_pars( 
        n_grps=len(lat_data['linpla_ord']))

    # correct p-value (Bonferroni)
    p_val_thr = 0.05
    if lat_data['max_comps_per'] != 0:
        p_val_thr_corr = p_val_thr/lat_data['max_comps_per']
    else:
        p_val_thr_corr = p_val_thr
    sig_comps = [[] for _ in range(len(linpla_iter))]
    maxes = np.full([len(linpla_iter), n_sess], np.nan)

    figpar = sess_plot_util.fig_init_linpla(figpar)
    
    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    prepost_str = sess_str_util.prepost_par_str(
        stimpar['pre'], stimpar['post'], str_type='print')
    title = (f'Surprise latencies ({prepost_str} seqs, {latstr_pr})\nfor '
        f'{stimstr_pr} - {statstr_pr} pooled across '
        f'\nROIs (grouped) (sess {sess_ns_str}{dendstr_pr})')

    fig, ax = plot_util.init_fig(n_plots, **figpar['init'])
    fig.suptitle(title, y=1.03, weight='bold')
    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        sub_ax = ax[pl, li]

        l_idx = get_linpla_idx(
            lat_data['linpla_ord'], line, pla, verbose=True, newline=(i==0))
        if l_idx is None:
            continue

        lat_st = np.asarray(lat_data['lat_stats'][l_idx])
        plot_util.plot_errorbars(
            sub_ax, lat_st[0], lat_st[1:], sess_ns, col=pla_cols[pl])
        # plot ROI cloud
        if datatype == 'roi':
            maxes[i] = plot_lat_clouds(
                sub_ax, sess_ns, lat_data['lat_vals'][l_idx], sess_info[l_idx], 
                datatype=datatype, col=pla_col_names[pl])
        else:
            maxes[i] = np.max(lat_st[0])

        # check p_val signif
        all_p_vals = lat_data['lat_p_vals'][l_idx]
        for p, p_val in enumerate(all_p_vals):
            if not np.isnan(p_val) and p_val < p_val_thr_corr:
                sig_comps[i].append(p)

    plot_data_signif(
        ax, sess_ns, sig_comps, lat_data['lin_p_vals'], maxes, p_val_thr=0.05, 
        n_comps=lat_data['max_comps_per'])

    # Add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(
        ax, fluor=analyspar['fluor'], datatype=datatype, lines=lines, 
        planes=planes, xticks=sess_ns, ylab='Latency (s)', xlab='Sessions', 
        adj_yticks=True, kind='reg')

    if savedir is None:
        savedir = os.path.join(
            figpar['dirs'][datatype], 
            figpar['dirs']['acr_sess'], 
            sess_str_util.get_stimdir(stimpar['stimtype'], stimpar['gabfr']), 
            figpar['dirs']['lat'], 
            latpar['method'])

    savename = f'{datatype}_surp_lat_{sessstr}{dendstr}_{latstr}{surp_resp_str}'
    
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar['save'])

    return fulldir, savename


#############################################
def plot_resp_prop(analyspar, sesspar, stimpar, latpar, extrapar, sess_info, 
                   prop_data, permpar=None, figpar=None, savedir=None):
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
        - latpar (LatPar)        : dictionary with keys of LatPar parameters
        - extrapar (dict)        : dictionary containing additional analysis 
                                   parameters
            ['analysis'] (str): analysis type (e.g., 't')
            ['datatype'] (str): datatype (e.g., 'run', 'roi')
        - sess_info (nested list): nested list of dictionaries for each 
                                   line/plane x session containing information 
                                   from each mouse, with None for missing 
                                   sessions
            ['mouse_ns'] (list)   : mouse numbers
            ['sess_ns'] (list)    : session numbers  
            ['lines'] (list)      : mouse lines
            ['planes'] (list)     : imaging planes
            ['nrois'] (list)      : number of ROIs in session
            ['nanrois_{}'] (list) : list of ROIs with NaNs/Infs in raw or dF/F 
                                    traces ('raw', 'dff')
        - prop_data (dict)        : dictionary with responsive proportion info
            ['linpla_ord'] (list): ordered list of planes/lines            
            ['prop_stats'] (list): proportion statistics, structured as
                                       plane/line x session x comb x stats
            ['comb_names'] (int) : names of combinations for with proportions 
                                   were calculated
            ['n_sign_rois] (list): number of significant ROIs, structured as 
                                   plane/line x session

    Optional args:
        - permpar (PermPar): dictionary with keys of PermPar namedtuple
                             default: None
        - figpar (dict)    : dictionary containing the following figure 
                             parameter dictionaries
                             default: None
            ['init'] (dict): dictionary with figure initialization parameters
            ['save'] (dict): dictionary with figure saving parameters
            ['dirs'] (dict): dictionary with additional figure parameters
        - savedir (str)    : path of directory in which to save plots.
                             default: None    
    
    Returns:
        - fulldir (str) : final name of the directory in which the figure 
                          is saved (may differ from input savedir, if 
                          datetime subfolder is added.)
        - savename (str): name under which the figure is saved
    """
 
    dendstr = sess_str_util.dend_par_str(
        analyspar['dend'], sesspar['plane'], extrapar['datatype'])
    latstr = sess_str_util.lat_par_str(
        latpar['method'], latpar['p_val_thr'], latpar['rel_std'])

    statstr_pr = sess_str_util.stat_par_str(
        analyspar['stats'], analyspar['error'], 'print')
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar['dend'], sesspar['plane'], extrapar['datatype'], 'print')
    latstr_pr = sess_str_util.lat_par_str(
        latpar['method'], latpar['p_val_thr'], latpar['rel_std'], 'print')
    surp_resp_str = ''
    if latpar['surp_resp']:
        surp_resp_str = '_surp_resp'

    datatype = extrapar['datatype']

    sess_ns = np.asarray(sesspar['sess_n'])
    if sesspar['sess_n'] in ['any', 'all']:
        sess_ns = np.asarray(range(len(sess_info[0]))) + 1
    sess_ns_str = gen_util.intlist_to_str(sess_ns.tolist())

    # combinations: 'gabfrs', 'surps'
    ctrl_idx, surp_idx = [prop_data['comb_names'].index(comb) 
        for comb in ['gabfrs', 'surps']]

    [lines, planes, linpla_iter, pla_cols, _, n_plots] = \
        sess_plot_util.fig_linpla_pars(n_grps=len(prop_data['linpla_ord']))
    figpar = sess_plot_util.fig_init_linpla(figpar)

    if figpar['save']['use_dt'] is None:
        figpar['save']['use_dt'] = gen_util.create_time_str()

    prepost_str = sess_str_util.prepost_par_str(
        stimpar['pre'], stimpar['post'], str_type='print')

    title = (f'Proportion surprise responsive ROIs\n({prepost_str} seqs, '
        f'{latstr_pr})\n{statstr_pr} across mice (sess '
        f'{sess_ns_str}{dendstr_pr})')

    fig, ax = plot_util.init_fig(n_plots, **figpar['init'])
    fig.suptitle(title, y=1.03, weight='bold')
    for i, (line, pla) in enumerate(linpla_iter):
        li = lines.index(line)
        pl = planes.index(pla)
        sub_ax = ax[pl, li]

        l_idx = get_linpla_idx(
            prop_data['linpla_ord'], line, pla, verbose=True, newline=(i==0))
        if l_idx is None:
            continue
        
        for idx, col in zip([surp_idx, ctrl_idx], [pla_cols[pl], 'gray']):
            # retrieve proportion (* 100)
            prop_st = np.asarray([sess_vals[idx] for sess_vals 
                in prop_data['prop_stats'][l_idx]]) * 100
            plot_util.plot_errorbars(
                sub_ax, prop_st[:, 0], prop_st[:, 1:], sess_ns, col=col)
    # Add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(ax, fluor=analyspar['fluor'], 
        datatype=datatype, lines=lines, planes=planes, 
        xticks=sess_ns, ylab='Prop (%)', xlab='Sessions', adj_yticks=True, 
        kind='reg')

    if savedir is None:
        savedir = os.path.join(
            figpar['dirs'][datatype], 
            figpar['dirs']['acr_sess'], 
            sess_str_util.get_stimdir(stimpar['stimtype'], stimpar['gabfr']),
            figpar['dirs']['prop'], 
            latpar['method'])

    savename = (f'{datatype}_prop_resp_sess{sess_ns_str}{dendstr}_{latstr}'
        f'{surp_resp_str}')
    
    fulldir = plot_util.savefig(fig, savename, savedir, **figpar['save'])

    return fulldir, savename

