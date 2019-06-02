"""
logreg_plots.py

This script contains functions to create plots for logistic regression analyses
and results (logreg_plots.py) from dictionaries, dataframes and torch models.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import copy
import os

from matplotlib import pyplot as plt
import numpy as np

from sess_util import sess_gen_util, sess_plot_util, sess_str_util
from util import file_util, gen_util, logreg_util, math_util, plot_util


#############################################
def plot_from_dict(direc, plt_bkend=None, fontdir=None):
    """
    plot_from_dict(direc)

    Plots data from dictionaries containing analysis parameters and results, or 
    path to results.

    Required args:
        - direc (str): path to directory in which dictionaries to plot data 
                       from are located
    
    Optional_args:
        - plt_bkend (str): mpl backend to use for plotting (e.g., 'agg')
                           default: None
        - fontdir (str)  : directory in which additional fonts are located
                           default: None
    """
    plot_util.manage_mpl(plt_bkend, fontdir=fontdir)

    hyperpars = file_util.loadfile('hyperparameters.json', fulldir=direc)

    if 'logregpar' in hyperpars.keys():
        plot_traces_scores(hyperpars, savedir=direc)


#############################################
def plot_title(mouse_n, sess_n, line, layer, comp, stimtype, bri_dir='right',
               bri_size=128, gabk=16, q1v4=False):
    """
    plot_title(mouse_n, sess_n, line, layer)

    Creates plot title from session information.
    
    Required args:
        - mouse_n (int) : mouse number
        - sess_n (int)  : session number
        - line (str)    : transgenic line name
        - layer (str)   : layer name
        - comp (str)    : comparison name
        - stimtype (str): stimulus type
    
    Optional args:
        - bri_dir (str or list)      : brick direction
                                       default: 'right'
        - bri_size (int, str or list): brick size
                                       default: 128
        - gabk (int, str or list)    : gabor kappa parameter
                                       default: 16
        - q1v4 (bool)                : if True, analysis is separated across 
                                       first and last quintiles
                                       default: False
    
    Returns:
        - (str): plot title 
    """
    if comp == 'surp':
        comp_str = 'Surp v Reg'
    else:
        comp_str = comp
    
    stim_str = sess_str_util.stim_par_str(stimtype, bri_dir, bri_size, gabk, 
                                          'print')

    return 'Mouse {}, sess {}, {} {}\n{}, {}'.format(mouse_n, sess_n, line, 
                                                 layer, stim_str.capitalize(), 
                                                 comp_str)


#############################################
def plot_class_traces(analyspar, sesspar, stimpar, tr_stats, classes, 
                      comp='surp', q1v4=False, shuffle=False, plot_wei=True, 
                      modeldir='.', savedir='.'):
    """
    plot_class_traces(analyspar, sesspar, stimpar, tr_stats)

    Plots training traces by class, and optionally weights, and saves figure. 

    Required args:
        - analyspar (dict): dictionary with keys of analyspar named tuple
        - sesspar (dict)  : dictionary with keys of sesspar named tuple
        - stimpar (dict)  : dictionary with keys of stimpar named tuple
        - tr_stats (dict) : dictionary of trace stats data
            ['n_rois'] (int)                  : number of ROIs
            ['train_ns'] (list)               : number of sequences per class
            ['train_class_stats'] (3D array)  : training statistics, structured
                                                as class x stats (me, err) x 
                                                   frames
            ['xran'] (array-like)             : x values for frames
            
            optionally:
            ['test_Q4_ns'] (list)             : number of segments per class
            ['test_Q4_class_stats'] (3D array): Q4 test statistics, 
                                                  structured as 
                                                  class x stats (me, err) x 
                                                  frames
        - classes (list)  : class names
    
    Optional args:
        - comp (str)     : comparison type
                           default: 'surp'
        - q1v4 (bool)    : if True, regression was trained on Q1 and tested on 
                           Q4
                           default: False
        - shuffle (bool) : if True, data is shuffled
                           default: False
        - plot_wei (bool): if True, weights are plotted in a subplot
                           default: True
        - modeldir (str) : directory in which the model parameters are saved
                           default: '.'
        - savedir (str)  : directory in which to save figure
                           default: '.'        
    """

    fig, ax_tr, cols = logreg_util.plot_tr_data(tr_stats['xran'], 
                                   tr_stats['train_class_stats'], classes, 
                                   tr_stats['train_ns'], plot_wei=plot_wei, 
                                   modeldir=modeldir, stats=analyspar['stats'], 
                                   error=analyspar['error'], xlabel='Time (s)')
    
    qu_str = ''
    test_label = 'test_Q4'
    st_name = '{}_class_stats'.format(test_label)
    n_name  = '{}_ns'.format(test_label)
    if st_name in tr_stats.keys():
        qu_str = ' Q1 (only)'
        cols_Q4 = ['cornflowerblue', 'salmon']
        _ = logreg_util.plot_tr_data(tr_stats['xran'], tr_stats[st_name],  
                                     classes, tr_stats[n_name], fig, ax_tr, 
                                     False, cols=cols_Q4, 
                                     data_type=test_label.replace('_', ' '))

    # add plot details
    if stimpar['stimtype'] == 'gabors' and comp == 'surp':
        ax_arr = np.asarray(ax_tr).reshape(1, 1)
        sess_plot_util.plot_labels(ax_arr, stimpar['gabfr'], pre=stimpar['pre'], 
                                 post=stimpar['post'], cols=cols, sharey=False)
        
    fluor_str = sess_str_util.fluor_par_str(analyspar['fluor'], 'print')
    scale_str = sess_str_util.scale_par_str(analyspar['scale'], 'print')
    shuff_str = sess_str_util.shuff_par_str(shuffle, 'labels')
    stat_str  = sess_str_util.stat_par_str(analyspar['stats'], 
                                           analyspar['error'], 'print')
    
    ax_tr.set_ylabel(u'{}{}'.format(fluor_str, scale_str))

    fig_title = plot_title(sesspar['mouse_n'], sesspar['sess_n'], 
                           sesspar['line'], sesspar['layer'], comp, 
                           stimpar['stimtype'], stimpar['bri_dir'], 
                           stimpar['bri_size'], stimpar['gabk'], q1v4)

    ax_tr.set_title(u'{}{}, {} across ROIs (n={}){}'.format(fig_title, qu_str, 
                                      stat_str, tr_stats['n_rois'], shuff_str))
    ax_tr.legend()

    save_name = os.path.join(savedir, 'train_traces')
    fig.savefig(save_name)


#############################################
def plot_scores(analyspar, sesspar, stimpar, extrapar, scores, comp='surp', 
                q1v4=False, savedir='.'):
    """
    plot_scores(args, scores, classes)

    Plots each score type in a figure and saves each figure.
    
    Required args:
        - analyspar (dict)     : dictionary with keys of analyspar named tuple
        - sesspar (dict)       : dictionary with keys of sesspar named tuple
        - stimpar (dict)       : dictionary with keys of stimpar named tuple
        - extrapar (dict)      : dictionary with extra parameters
            ['classes'] (list) : class names
            ['loss_name'] (str): name of loss
            ['shuffle'] (bool) : if True, data was shuffled
        - scores (pd DataFrame): dataframe in which scores are recorded, for
                                 each epoch
    
    Optional args:
        - comp (str)   : comparison type
                         default: 'surp'
        - q1v4 (bool)  : if True, regression was trained on Q1 and tested on Q4
                         default: False
        - savedir (str): directory in which to save figure
                         default: '.'
    """

    fluor_str = sess_str_util.fluor_par_str(analyspar['fluor'], 'print')
    scale_str = sess_str_util.scale_par_str(analyspar['scale'], 'print')
    shuff_str = sess_str_util.shuff_par_str(extrapar['shuffle'], 'labels')
    fig_title = plot_title(sesspar['mouse_n'], sesspar['sess_n'], 
                           sesspar['line'], sesspar['layer'], comp, 
                           stimpar['stimtype'], stimpar['bri_dir'], 
                           stimpar['bri_size'], stimpar['gabk'], q1v4)

    if q1v4:
        qu_str = ' (trained on Q1 and tested on Q4)'
    else:
        qu_str = ''

    gen_title = u'{}{}, {}{}{}'.format(fig_title, qu_str, fluor_str, scale_str, 
                                       shuff_str)

    logreg_util.plot_scores(scores, extrapar['classes'], extrapar['loss_name'], 
                            savedir, gen_title=gen_title)


#############################################
def plot_traces_scores(hyperpars, tr_stats=None, full_scores=None, 
                       plot_wei=True, savedir=None):
    """
    plot_traces_scores(hyperpars)

    Plots training traces and scores for a logistic regression analysis run.
    
    Required args:
        - hyperpars (dict):
            ['analyspar'] (dict): dictionary with keys of analyspar named tuple
            ['extrapar'] (dict) : dictionary with extra parameters
                ['classes'] (list) : class names
                ['dirname'] (str)  : directory in which data and plots are saved
                ['loss_name'] (str): name of loss
                ['shuffle'] (bool) : if True, data was shuffled
            ['logregpar'] (dict): dictionary with keys of logregpar named tuple
            ['sesspar'] (dict)  : dictionary with keys of sesspar named tuple
            ['stimpar'] (dict)  : dictionary with keys of stimpar named tuple 

    Optional args:        
        - tr_stats (dict)           : dictionary of trace stats data
            ['n_rois'] (int)                  : number of ROIs
            ['train_ns'] (list)               : number of segments per class
            ['train_class_stats'] (3D array)  : training statistics, structured
                                                as class x stats (me, err) x 
                                                   frames
            ['xran'] (array-like)             : x values for frames
            
            optionally:
            ['test_Q4_ns'] (list)             : number of segments per class
            ['test_Q4_class_stats'] (3D array): Q4 test statistics, 
                                                  structured as 
                                                  class x stats (me, err) x 
                                                  frames
        - full_scores (pd DataFrame): dataframe in which scores are recorded, 
                                      for each epoch
        - plot_wei (bool)           : if True, weights are plotted in a subplot
                                      default: True
        - savedir (str)             : directory in which to save figure (used 
                                      instead of extrapar['dirname'], if 
                                      passed)
                                      default: None
    """

    analyspar = hyperpars['analyspar']
    sesspar   = hyperpars['sesspar']
    stimpar   = hyperpars['stimpar']
    logregpar = hyperpars['logregpar']
    extrapar  = hyperpars['extrapar']

    if savedir is None:
        savedir = extrapar['dirname']

    if tr_stats is None:
        tr_stats_path = os.path.join(savedir, 'tr_stats.json')
        if os.path.exists(tr_stats_path):
            tr_stats = file_util.loadfile(tr_stats_path)
        else:
            print('No trace statistics found.')
    
    if full_scores is None:
        full_scores_path = os.path.join(savedir, 'scores_df.csv')
        if os.path.exists(full_scores_path):
            full_scores = file_util.loadfile(full_scores_path)
        else:
            print('No scores dataframe found.')

    if tr_stats is not None:
        plot_class_traces(analyspar, sesspar, stimpar, tr_stats, 
                          extrapar['classes'], logregpar['comp'], 
                          logregpar['q1v4'], extrapar['shuffle'], 
                          plot_wei=plot_wei, modeldir=savedir, 
                          savedir=savedir)

    if full_scores is not None:
        plot_scores(analyspar, sesspar, stimpar, extrapar, full_scores, 
                    logregpar['comp'], logregpar['q1v4'], savedir=savedir)


#############################################    
def init_res_fig(n_subplots, max_sess=None):
    """
    init_res_fig(n_subplots)

    Initializes a figure in which to plot summary results.

    Required args:
        - n_subplots (int): number of subplots
        
    Optional args:
        - max_sess (int)  : maximum number of sessions plotted
                            default: None
    
    Returns:
        - fig (plt Fig): figure
        - ax (plt Axis): axis
    """

    subplot_hei = 7.5

    if max_sess is not None:
        subplot_hei *= 4/3.0

    fig, ax = plot_util.init_fig(n_subplots, 2, sharey=True, 
                                 subplot_hei=subplot_hei)

    return fig, ax


#############################################
def rois_x_label(sess_ns, arr):
    """
    rois_x_label(sess_ns, arr)

    Creates x axis labels with the number of ROIs per mouse for each session.
    
    For each session, formatted as: Session # (n/n rois)
    
    Required args:
        - sess_ns (list): list of session numbers
        - arr (3D array): array of number of ROIs, structured as 
                          mouse x session x shuffle

    Returns:
        - x_label (list): list of x_labels for each session.
    """

    arr = np.nan_to_num(arr) # convert NaNs to 0s
    
    # check that shuff and non shuff are the same
    if not (arr[:, :, 0] == arr[:, :, 1]).all():
        raise ValueError('Shuffle and non shuffle n_rois are not the same.')

    x_label = ['Session {}'.format(x+1) for x in range(arr.shape[1])]
    
    for sess_n in sess_ns:
        for m in range(arr.shape[0]):
            if m == 0:
                n_rois_str = '{}'.format(int(arr[m, int(sess_n-1), 0]))
            if m > 0:
                n_rois_str = '{}/{}'.format(n_rois_str, int(arr[m, int(sess_n-1), 0]))
        x_label[sess_n-1] = 'Session {}\n({} rois)'.format(int(sess_n), n_rois_str)
    return x_label


#############################################
def mouse_runs_leg(arr, mouse_n=None, shuffle=False, CI=0.95):
    """
    mouse_runs_leg(arr)

    Creates legend labels for a mouse or shuffle set.  
    
    For each mouse or shuffle set, formatted as: 
    Mouse # (n/n runs) or Shuff (n/n runs)

    Required args:
        - arr (2D array): array of number of ROIs, structured as 
                          mouse (or mice to sum) x session

    Optional args:
        - mouse_n (int) : mouse number (only needed if shuffle is False)
                          default: None
        - shuffle (bool): if True, shuffle legend is created. Otherwise, 
                          mouse legend is created.
                          default: False
        - CI (num)      : CI for shuffled data
                          default: 0.95 

    Returns:
        - leg (str): legend for the mouse or shuffle set
    """

    # create legend: Mouse # (n/n runs) or Shuff (n/n runs)
    if len(arr.shape) == 1:
        arr = arr[np.newaxis, :]
    arr = np.nan_to_num(arr) # convert NaNs to 0s

    for s in range(arr.shape[1]):
        if s == 0:
            n_runs_str = '{}'.format(int(np.sum(arr[:, s])))
        if s > 0:
            n_runs_str = '{}/{}'.format(n_runs_str, int(np.sum(arr[:, s])))
    
    if shuffle:
        if CI is not None:
            CI_pr = CI*100
            if CI_pr%1 == 0:
                CI_pr = int(CI_pr)
            leg = 'shuffled ({}% CI)\n({} runs)'.format(CI_pr, n_runs_str)
        else:
            leg = 'shuffled\n({} runs)'.format(n_runs_str)

    else:
        if mouse_n is None:
            raise IOError('If \'shuffle\' is False, Must specify \'mouse_n\'.')
        
        leg = 'mouse {}\n({} runs)'.format(int(mouse_n), n_runs_str)
    
    return leg


#############################################
def plot_CI(ax, x_label, arr, sess_ns, CI=0.95, q1v4=False):
    """
    plot_CI(ax, x_label, arr, sess_ns)

    Plots confidence intervals for each session.

    Required args:
        - ax (plt Axis subplot): subplot
        - x_label (list)       : list of x_labels for each session
        - arr (3D array)       : array of session information, structured as 
                                 mice x sessions x vals, where vals
                                 are: mean/med, sem/low_perc, sem/hi_perc, 
                                      (x2 if q1v4 and test accuracy)
                                      n_rois, n_runs
        - sess_ns (list)       : list of session numbers
    
    Optional args:
        - CI (num)   : CI for shuffled data
                       default: 0.95 
        - q1v4 (bool): if True, analysis is separated across first and 
                       last quintiles
    """

    # shuffle (combine across mice)
    # if q1v4: use Q4 data instead
    med  = np.nanmedian(arr[:, :, 0 + 3 * q1v4], axis=0) 
    p_lo = np.nanmedian(arr[:, :, 1 + 3 * q1v4], axis=0)
    p_hi = np.nanmedian(arr[:, :, 2 + 3 * q1v4], axis=0)

    leg = mouse_runs_leg(arr[:, :, -1], shuffle=True, CI=CI)

    # plot CI
    ax.bar(x_label, height=p_hi-p_lo, bottom=p_lo, color='lightgray', 
           width=0.4, label=leg)
    
    # plot median (with some thickness based on ylim)
    y_lim = ax.get_ylim()
    med_th = 0.015*(y_lim[1]-y_lim[0])
    ax.bar(x_label, height=med_th, bottom=med-med_th/2., color='grey', 
           width=.4)


#############################################
def summ_subplot(ax, arr, data_title, mouse_ns, sess_ns, line, layer, title,
                 stat='mean', CI=0.95, q1v4=False):
    """
    summ_subplot(ax, arr, data_title, mouse_ns, sess_ns, line, layer, title)

    Plots summary data in the specific subplot for a line and layer.

    Required args:
        - ax (plt Axis subplot): subplot
        - arr (4D array)       : array of session information, structured as 
                                 mice x sessions x shuffle x vals, where vals
                                 are: mean/med, sem/low_perc, sem/hi_perc, 
                                      (x2 if q1v4 and test accuracy)
                                      n_rois, n_runs
        - data_title (str)     : name of type of data plotted, 
                                 i.e. for epochs or test accuracy
        - mouse_ns (int)       : mouse numbers
        - sess_ns (int)        : session numbers
        - line (str)           : transgenic line name
        - layer (str)          : layer name
        - title (str)          : general plot titles (must contain 'data')
    
    Optional args:
        - stat (str) : stats to take for non shuffled data, 
                       i.e., 'mean' or 'median' 
                       default: 'mean'
        - CI (num)   : CI for shuffled data (e.g., 0.95)
        - q1v4 (bool): if True, analysis is separated across first and 
                       last quintiles

    """

    x_label = rois_x_label(sess_ns, arr[:, :, :, -2])

    if 'acc' in data_title:
        if q1v4:
            ax.set_ylabel('Accuracy in Q4 (%)')
        else:
            ax.set_ylabel('Accuracy (%)')
        plot_util.set_ticks(ax, 'y', 0, 100, 6)

    elif 'epoch' in data_title:
        q1v4 = False # treated as if no Q4
        ax.set_ylabel('Nbr epochs')
        plot_util.set_ticks(ax, 'y', 0, 1000, 6)

    if q1v4:
        mean_ids = [0, 3]
        alphas = [0.3, 0.8]
        add_leg = [' (Q1)', ' (Q4)']
    else:
        mean_ids = [0]
        alphas = [0.5]
        add_leg = ['']
    
    plot_CI(ax, x_label, arr[:, :, 1], sess_ns, CI, q1v4)
    
    # plot non shuffle data
    for m, mouse_n in enumerate(mouse_ns):
        leg = mouse_runs_leg(arr[m, :, 0, -1], mouse_n, False)
        col = None
        for i, m_i in enumerate(mean_ids):
            leg_i = leg.index('\n')
            leg_m = '{}{}{}'.format(leg[: leg_i], add_leg[i], leg[leg_i :])
            ax.errorbar(x_label, arr[m, :, 0, m_i], yerr=arr[m, :, 0, m_i + 1], 
                        fmt='-o', markersize=12, capthick=4, label=leg_m, 
                        alpha=alphas[i], lw=3, color=col)
            col = ax.lines[-1].get_color()

    # add a mean line
    for i in range(len(x_label)):
        for m, m_i in enumerate(mean_ids):
            if not np.isnan(arr[:, i, 0, m_i]).all():
                med = math_util.mean_med(arr[:, i, 0, m_i], axis=0, stats=stat, 
                                         nanpol='omit')
                y_lim = ax.get_ylim()
                med_th = 0.0075*(y_lim[1]-y_lim[0])

                ax.bar(x_label[i], height=med_th, bottom=med - med_th/2., 
                       color='black', width=0.5, alpha=alphas[m])

    if line == 'L23':
        line = 'L2/3'
    
    title = '{}{} {} {}'.format(title[: title.index('data')], line, layer, 
                                 title[title.index('data') :])

    ax.set_title(title)
    ax.legend()


#############################################    
def plot_data_summ(plot_lines, data, stats, shuff_stats, title, savename, 
                   CI=0.95, q1v4=False):
    """
    plot_data_summ(plot_lines, data, stats, shuff_stats, title, savename)

    Plots summary data for a specific comparison, for each line and layer and 
    saves figure.

    Required args:
        - plot_lines (pd DataFrame): DataFrame containing scores summary
                                     for specific comparison and criteria
        - data (str)               : label of type of data to plot,
                                     e.g., 'epoch_n' or 'test_acc_bal' 
        - stats (list)             : list of stats to use for non shuffled 
                                     data, e.g., ['mean', 'sem', 'sem']
        - shuff_stats (list)       : list of stats to use for shuffled 
                                     data, e.g., ['median', 'p2p5', 'p97p5']
        - title (str)              : general plot titles (must contain 'data')
        - savename (str)           : plot save path
        
    Optional args:
        - CI (num)   : CI for shuffled data (e.g., 0.95)
        - q1v4 (bool): if True, analysis is separated across first and 
                       last quintiles
    """
    
    celltypes = [[x, y] for x in ['L23', 'L5'] for y in ['soma', 'dend']]

    max_sess = max(plot_lines['sess_n'].tolist())

    fig, ax = init_res_fig(len(celltypes), max_sess)
    n_vals = 5 # (mean/med, sem/2.5p, sem/97.5p, n_rois, n_runs)
    
    if data == 'test_acc_bal':
        found = False
        for key in plot_lines.keys():
            if data in key:
                found = True
        if not found:
            print('test_acc_bal was not recorded')
            return
    
    data_types = gen_util.list_if_not(data)
    if q1v4 and 'test_acc' in data: 
        n_vals = 8 # (extra mean/med, sem/2.5p, sem/97.5p for Q4) 
        if data == 'test_acc': 
            data_types = ['test_acc', 'test_Q4_acc']
        elif data == 'test_acc_bal':   
            data_types = ['test_acc_bal', 'test_Q4_acc_bal']
        else:
            gen_util.accepted_values_error('data', data, 
                                           ['test_acc', 'test_acc_bal'])

    for i, [line, layer] in enumerate(celltypes):
        sub_ax = plot_util.get_subax(ax, i)
        sub_ax.set_xlim(-0.5, max_sess - 0.5)
        # get the right rows in dataframe
        cols       = ['layer']
        cri        = [layer]
        curr_lines = gen_util.get_df_vals(plot_lines.loc[plot_lines['line'].str.contains(line)], cols, cri)
        if len(curr_lines) == 0:
            cri_str = ['{}: {}'.format(col, crit) 
                       for col, crit in zip(cols, cri)]
            print('No data found for {} {}, {}'.format(line, layer,
                                                       ', '.join(cri_str)))
            continue
        sess_ns    = gen_util.get_df_vals(curr_lines, label='sess_n', 
                                          dtype=int)
        mouse_ns   = gen_util.get_df_vals(curr_lines, label='mouse_n', 
                                          dtype=int)
        # mouse x sess x shuffle x n_vals 
        data_arr = np.empty((len(mouse_ns), int(max_sess), 2, n_vals)) * np.nan
        
        for sess_n in sess_ns:
            sess_mice = gen_util.get_df_vals(curr_lines, 'sess_n', sess_n, 
                                             'mouse_n', dtype=int)
            for m, mouse_n in enumerate(mouse_ns):
                if mouse_n in sess_mice:
                    for sh, stat_types in enumerate([stats, shuff_stats]):
                        curr_line = gen_util.get_df_vals(curr_lines, 
                                            ['sess_n', 'mouse_n', 'shuffle'], 
                                            [sess_n, mouse_n, sh])
                        for st, stat in enumerate(stat_types):
                            for d, dat in enumerate(data_types):
                                i = d * 3 + st
                                data_arr[m, int(sess_n-1), sh, i] = curr_line['{}_{}'.format(dat, stat)]
                        data_arr[m, int(sess_n-1), sh, -2] = curr_line['n_rois']
                        data_arr[m, int(sess_n-1), sh, -1] = curr_line['runs_total'] - curr_line['runs_nan']
        
        summ_subplot(sub_ax, data_arr, title, mouse_ns, sess_ns, line, layer, 
                     title, stats[0], CI, q1v4)
    
    fig.savefig(savename)


#############################################    
def plot_summ(output, savename, stimtype='gabors', comp='surp', bri_dir='both', 
              fluor='dff', scale='roi', CI=0.95, plt_bkend=None, fontdir=None):
    """
    plot_summ(output)

    Plots summary data for a specific comparison, for each datatype in a 
    separate figure and saves figures. 

    Required args:
        - output (str)  : general directory in which summary dataframe 
                          is saved (runtype and q1v4 values are inferred from 
                          the directory name)
        - savename (str): name of the dataframe containing summary data to plot
            
    Optional args:
        - stimtype (str) : stimulus type
                           default: 'gabors'
        - comp (str)     : type of comparison
                           default: 'surp'
        - bri_dir (str)  : brick direction
                           default: 'both'
        - fluor (str)    : fluorescence trace type
                           default: 'dff'
        - scale (str)    : type of scaling
                           default: 'roi'
        - CI (num)       : CI for shuffled data
                           default: 0.95
        - plt_bkend (str): mpl backend to use for plotting (e.g., 'agg')
                           default: None
        - fontdir (str)  : directory in which additional fonts are located
                           default: None
    """
    
    plot_util.manage_mpl(plt_bkend, fontdir=fontdir)

    summ_scores_file = os.path.join(output, savename)
    
    if os.path.exists(summ_scores_file):
        summ_scores = file_util.loadfile(summ_scores_file)
    else:
        print('{} not found.'.format(summ_scores_file))
        return

    data_types  = ['epoch_n', 'test_acc', 'test_acc_bal']
    data_titles = ['epoch nbr', 'test accuracy', 'test accuracy (balanced)']

    stats = ['mean', 'sem', 'sem']
    shuff_stats = ['median'] + math_util.get_percentiles(CI)[1]

    q1v4 = False
    if 'q1v4' in output:
        q1v4 = True
    
    runtype = 'prod'
    if 'pilot' in output:
        runtype = 'pilot'

    if stimtype == 'gabors':
        bri_dir = 'none'
        stim_str = 'gab'
        stim_str_pr = 'gabors'

    else:
        bri_dir = sess_gen_util.get_params(stimtype, bri_dir)[0]
        stim_str = sess_str_util.dir_par_str(bri_dir, str_type='file')
        stim_str_pr = sess_str_util.dir_par_str(bri_dir, str_type='print')

    scale_str = sess_str_util.scale_par_str(scale, 'file')
    scale_str_pr = sess_str_util.scale_par_str(scale, 'file').replace('_', ' ')

    save_dir = os.path.join(output, 'figures_{}'.format(fluor))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    cols = ['scale', 'fluor', 'bri_dir', 'runtype']
    cri  = [scale, fluor, bri_dir, runtype]
    plot_lines = gen_util.get_df_vals(summ_scores, cols, cri)
    if len(plot_lines) == 0:
        cri_str = ['{}: {}'.format(col, crit) 
                        for col, crit in zip(cols, cri)]
        print('No data found for {}'.format(', '.join(cri_str)))
        return

    for data, data_title in zip(data_types, data_titles):
        title = ('{} {} - {} for log regr on'
                 '\n{} {} data ({})').format(stim_str_pr.capitalize(), comp, 
                                      data_title, scale_str_pr, fluor, runtype)
        savename = os.path.join(save_dir, '{}_{}_{}{}.svg'.format(data, 
                                stim_str, comp, scale_str))

        plot_data_summ(plot_lines, data, stats, shuff_stats, title, savename, 
                       CI, q1v4)

    plt.close('all')

