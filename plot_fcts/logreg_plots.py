"""
logreg_plots.py

This script contains functions to create plots for logistic regression analyses
and results (logreg.py) from dictionaries, dataframes and torch models.

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
        - fontdir (str)  : directory in which additional fonts are stored
                           default: None
    """
    
    plot_util.manage_mpl(plt_bkend, fontdir=fontdir)
    hyperpars = file_util.loadfile('hyperparameters.json', fulldir=direc)

    print(f'\nPlotting from hyperparameters in: {direc}')

    if 'logregpar' in hyperpars.keys():
        plot_traces_scores(hyperpars, savedir=direc)

    plt.close('all')

#############################################
def plot_title(mouse_n, sess_n, line, plane, comp, stimtype, bri_dir='right',
               bri_size=128, gabk=16):
    """
    plot_title(mouse_n, sess_n, line, plane)

    Creates plot title from session information.
    
    Required args:
        - mouse_n (int) : mouse number
        - sess_n (int)  : session number
        - line (str)    : transgenic line name
        - plane (str)   : plane name
        - comp (str)    : comparison name
        - stimtype (str): stimulus type
    
    Optional args:
        - bri_dir (str or list)      : brick direction
                                       default: 'right'
        - bri_size (int, str or list): brick size
                                       default: 128
        - gabk (int, str or list)    : gabor kappa parameter
                                       default: 16
    
    Returns:
        - (str): plot title 
    """
    if comp == 'surp':
        comp_str = 'Surp v Reg'
    elif comp == 'Direction':
        comp_str = 'Direction'
    elif comp == 'dir_reg':
        comp_str = 'Reg dir'
    elif comp == 'dir_surp':
        comp_str = 'Surp dir'
    elif comp == 'half_diff':
        comp_str = 'Halves (diff dir)'
    elif comp == 'half_right':
        comp_str = 'Halves (both right)'
    elif comp == 'half_left':
        comp_str = 'Halves (both left)'

    else:
        comp_str = comp
    
    stim_str = sess_str_util.stim_par_str(
        stimtype, bri_dir, bri_size, gabk, 'print')

    return (f'Mouse {mouse_n}, sess {sess_n}, {line} {plane}\n'
        f'{stim_str.capitalize()}, {comp_str}')


#############################################
def plot_class_traces(analyspar, sesspar, stimpar, logregpar, tr_stats, 
                      classes, shuffle=False, plot_wei=True, modeldir='', 
                      savedir=''):
    """
    plot_class_traces(analyspar, sesspar, stimpar, tr_stats)

    Plots training traces by class, and optionally weights, and saves figure. 

    Required args:
        - analyspar (dict): dictionary with keys of analyspar named tuple
        - sesspar (dict)  : dictionary with keys of sesspar named tuple
        - stimpar (dict)  : dictionary with keys of stimpar named tuple
        - logregpar (dict): dictionary with keys of logregpar named tuple
        - tr_stats (dict) : dictionary of trace stats data
            ['n_rois'] (int)                  : number of ROIs
            ['train_ns'] (list)               : number of segments per class
            ['train_class_stats'] (3D array)  : training statistics, structured
                                                as class x stats (me, err) x 
                                                   frames
            ['xran'] (array-like)             : x values for frames
            
            optionally, if an additional named set (e.g., 'test_Q4') is passed:
            ['set_ns'] (list)             : number of segments per class
            ['set_class_stats'] (3D array): trace statistics, 
                                                  structured as 
                                                  class x stats (me, err) x 
                                                  frames
        - classes (list)  : class names
    
    Optional args:
        - shuffle (bool)        : if True, data is shuffled
                                  default: False
        - plot_wei (bool or int): if True, weights are plotted in a subplot.
                                  Or if int, index of model to plot.
                                  default: True
        - modeldir (str)        : directory in which the model parameters are 
                                  saved
                                  default: ''
        - savedir (str)         : directory in which to save figure
                                  default: ''        
    """

    cols = None
    if len(classes) != 2:
        cols = plot_util.get_color_range(len(classes), col='blue')
    rois_collapsed = (tr_stats['n_rois'] in [1, 2])
    fig, ax_tr, cols = logreg_util.plot_tr_data(
        tr_stats['xran'], tr_stats['train_class_stats'], classes, 
        tr_stats['train_ns'], plot_wei=plot_wei, alg=logregpar['alg'], 
        modeldir=modeldir, stats=analyspar['stats'], error=analyspar['error'], 
        xlabel='Time (s)', cols=cols, rois_collapsed=rois_collapsed)

    ext_label =  [key for key in tr_stats.keys() 
        if ('_class_stats' in key and key != 'train_class_stats')]
    ext_str = ''        
    if len(ext_label) == 1:
        st_name = ext_label[0]
        test_lab = st_name.replace('_class_stats', '')
        n_name    = f'{test_lab}_ns'
        ext_str = sess_str_util.ext_test_str(
            logregpar['q1v4'], logregpar['regvsurp'], 'label')
        if len(classes) == 2:
            ext_cols = ['cornflowerblue', 'salmon']
        else:
            ext_cols = plot_util.get_color_range(len(classes), col='red')
        _ = logreg_util.plot_tr_data(
            tr_stats['xran'], tr_stats[st_name], classes, tr_stats[n_name], 
            fig, ax_tr, False, alg=logregpar['alg'], cols=ext_cols, 
            data_type=test_lab.replace('_', ' '))
    elif len(ext_label) > 1:
        raise ValueError('Did not expect more than 1 extra dataset to plot.')
    
    # add plot details
    if stimpar['stimtype'] == 'gabors' and logregpar['comp'] == 'surp':
        ax_arr = np.asarray(ax_tr).reshape(1, 1)
        sess_plot_util.plot_labels(ax_arr, stimpar['gabfr'], pre=stimpar['pre'], 
                       post=stimpar['post'], cols=cols, sharey=False)
    elif stimpar['stimtype'] == 'bricks':
        if stimpar['pre'] > 0 and stimpar['post'] > 0:
            ax_tr.axvline(x=0, ls='dashed', c='k', lw=1.5, alpha=0.5)

    fluor_str = sess_str_util.fluor_par_str(analyspar['fluor'], 'print')
    scale_str = sess_str_util.scale_par_str(analyspar['scale'], 'print')
    shuff_str = sess_str_util.shuff_par_str(shuffle, 'labels')
    stat_str  = sess_str_util.stat_par_str(analyspar['stats'], 
        analyspar['error'], 'print')
    if rois_collapsed:
        if tr_stats['n_rois'] == 1:
            stats = 'mean'
        else:
            stats = 'mean/std'
        stat_str = '{} stats ({}) across ROIs'.format(analyspar['stats'], stats)
    else:
        stat_str = u'{} across ROIs (n={})'.format(stat_str, tr_stats['n_rois'])
    
    ax_tr.set_ylabel(u'{}{}'.format(fluor_str, scale_str))

    fig_title = plot_title(
        sesspar['mouse_n'], sesspar['sess_n'], sesspar['line'], 
        sesspar['plane'], logregpar['comp'], stimpar['stimtype'], 
        stimpar['bri_dir'], stimpar['bri_size'], stimpar['gabk'])

    ax_tr.set_title(
        f'{fig_title}{ext_str}, ' + u'{} '.format(stat_str) + f'\n{shuff_str}')
    ax_tr.legend()

    save_name = os.path.join(savedir, 'train_traces')
    fig.savefig(save_name)


#############################################
def plot_scores(analyspar, sesspar, stimpar, logregpar, extrapar, scores, 
                savedir=''):
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
        - savedir (str): directory in which to save figure
                         default: ''
    """

    fluor_str = sess_str_util.fluor_par_str(analyspar['fluor'], 'print')
    scale_str = sess_str_util.scale_par_str(analyspar['scale'], 'print')
    shuff_str = sess_str_util.shuff_par_str(extrapar['shuffle'], 'labels')
    fig_title = plot_title(
        sesspar['mouse_n'], sesspar['sess_n'], sesspar['line'], 
        sesspar['plane'], logregpar['comp'], stimpar['stimtype'], 
        stimpar['bri_dir'], stimpar['bri_size'], stimpar['gabk'])

    ext_str = sess_str_util.ext_test_str(
        logregpar['q1v4'], logregpar['regvsurp'], 'print')

    gen_title = (f'{fig_title}{ext_str}' + u' {}'.format(fluor_str) +
        f'{scale_str}{shuff_str}')

    logreg_util.plot_scores(
        scores, extrapar['classes'], logregpar['alg'], extrapar['loss_name'], 
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
            
            optionally, if an additional named set (e.g., 'test_Q4') is passed:
            ['set_ns'] (list)             : number of segments per class
            ['set_class_stats'] (3D array): trace statistics, 
                                                  structured as 
                                                  class x stats (me, err) x 
                                                  frames
        - full_scores (pd DataFrame): dataframe in which scores are recorded, 
                                      for each epoch
        - plot_wei (bool or int)    : if True, weights are plotted in a subplot.
                                      Or if int, index of model to plot.
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
            if plot_wei and logregpar['alg'] == 'sklearn':
                saved = full_scores.loc[
                    full_scores['saved'] == 1]['run_n'].tolist()
                if len(saved) > 0:
                    plot_wei = saved[0]
        else:
            print('No scores dataframe found.')

    if tr_stats is not None:
        plot_class_traces(analyspar, sesspar, stimpar, logregpar, tr_stats, 
            extrapar['classes'], extrapar['shuffle'], plot_wei=plot_wei, 
            modeldir=savedir, savedir=savedir)

    if full_scores is not None:
        plot_scores(
            analyspar, sesspar, stimpar, logregpar, extrapar, full_scores, 
            savedir=savedir)


#############################################    
def init_res_fig(n_subplots, max_sess=None, modif=False):
    """
    init_res_fig(n_subplots)

    Initializes a figure in which to plot summary results.

    Required args:
        - n_subplots (int): number of subplots
        
    Optional args:
        - max_sess (int)  : maximum number of sessions plotted
                            default: None
        - modif (bool)    : if True, plots are made in a modified (simplified 
                            way)
                            default: False

    Returns:
        - fig (plt Fig): figure
        - ax (plt Axis): axis
    """

    subplot_hei = 14

    subplot_wid = 7.5
    if max_sess is not None:
        subplot_wid *= max_sess/4.0

    if modif:
        sess_plot_util.update_plt_linpla()
        figpar_init = sess_plot_util.fig_init_linpla(sharey=True)['init']
        fig, ax = plot_util.init_fig(n_subplots, **figpar_init)
    else:
        fig, ax = plot_util.init_fig(n_subplots, 2, sharey=True, 
            subplot_hei=subplot_hei, subplot_wid=subplot_wid)

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
                          mouse x session

    Returns:
        - x_label (list): list of x_labels for each session.
    """

    arr = np.nan_to_num(arr) # convert NaNs to 0s

    x_label = [f'Session {x+1}' for x in range(arr.shape[1])]
    
    for sess_n in sess_ns:
        for m in range(arr.shape[0]):
            if m == 0:
                n_rois_str = f'{int(arr[m, int(sess_n-1)])}'
            if m > 0:
                n_rois_str = f'{n_rois_str}/{int(arr[m, int(sess_n-1)])}'
        x_label[sess_n-1] = f'Session {int(sess_n)}\n({n_rois_str} rois)'

    # break up string
    for i in range(len(x_label)):
        max_leng = 10
        j = x_label[i].find('\n') + 2
        while len(x_label[i]) - j > max_leng:
            spt_pt = x_label[i][j : j + max_leng].rfind('/')
            if spt_pt == -1:
                break
            sub = spt_pt + j + 1
            x_label[i] = f'{x_label[i][:sub]}\n{x_label[i][sub:]}'
            j = sub + 1

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
            n_runs_str = f'{int(np.sum(arr[:, s]))}'
        if s > 0:
            n_runs_str = f'{n_runs_str}/{int(np.sum(arr[:, s]))}'
    
    if shuffle:
        if CI is not None:
            CI_pr = CI*100
            if CI_pr%1 == 0:
                CI_pr = int(CI_pr)
            leg = f'shuffled ({CI_pr}% CI)\n({n_runs_str} runs)'
        else:
            leg = f'shuffled\n({n_runs_str} runs)'

    else:
        if mouse_n is None:
            raise IOError('If `shuffle` is False, Must specify `mouse_n.')
        
        leg = f'mouse {int(mouse_n)}\n({n_runs_str} runs)'
    
    return leg


#############################################
def plot_CI(ax, x_label, arr, CI=0.95, ext_data=False, modif=False):
    """
    plot_CI(ax, x_label, arr)

    Plots confidence intervals for each session.

    Required args:
        - ax (plt Axis subplot): subplot
        - x_label (list)       : list of x_labels for each session
        - arr (3D array)       : array of session information, structured as 
                                 mice x sessions x vals, where vals
                                 are: mean/med, sem/low_perc, sem/hi_perc, 
                                      (x2 if q1v4 and test accuracy)
                                      n_runs
    
    Optional args:
        - CI (num)       : CI for shuffled data
                           default: 0.95 
        - ext_data (bool): if True, additional test data is included
                           default: False
        - modif (bool)   : if True, plots are made in a modified (simplified 
                           way)
                           default: False

    """

    # shuffle (combine across mice)
    # if q1v4: use Q4 data instead
    med  = np.nanmedian(arr[:, :, 0 + 3 * ext_data], axis=0) 
    p_lo = np.nanmedian(arr[:, :, 1 + 3 * ext_data], axis=0)
    p_hi = np.nanmedian(arr[:, :, 2 + 3 * ext_data], axis=0)

    leg = None
    if not modif:
        leg = mouse_runs_leg(arr[:, :, -1], shuffle=True, CI=CI)

    plot_util.plot_CI(
        ax, [p_lo, p_hi], med=med, x=x_label, width=0.4, label=leg)


#############################################
def summ_subplot(ax, arr, sh_arr, data_title, mouse_ns, sess_ns, line, plane, 
                 stat='mean', error='sem', CI=0.95, q1v4=False, rvs=False, 
                 modif=False):
    """
    summ_subplot(ax, arr, data_title, mouse_ns, sess_ns, line, plane, title)

    Plots summary data in the specific subplot for a line and plane.

    Required args:
        - ax (plt Axis subplot): subplot
        - arr (3D array)       : array of session information, structured as 
                                 mice x sessions x vals, where vals
                                 are: mean/med, sem/low_perc, sem/hi_perc, 
                                      (x2 if q1v4 and test accuracy)
                                      n_rois, n_runs
        - sh_arr (3D array)    : array of session information, structured as 
                                 mice (1) x sessions x vals, where vals
                                 are: mean/med, sem/low_perc, sem/hi_perc, 
                                      (x2 if q1v4 and test accuracy), n_runs
        - data_title (str)     : name of type of data plotted (must contain 
                                 'data'), i.e. for epochs or test accuracy
        - mouse_ns (int)       : mouse numbers (-1 for shuffled data)
        - sess_ns (int)        : session numbers
        - line (str)           : transgenic line name
        - plane (str)          : plane name
    
    Optional args:
        - stat (str)  : stats to take for non shuffled data, 
                        i.e., 'mean' or 'median' 
                        default: 'mean'
        - error (str) : error stats to take for non shuffled data, i.e., 'std', 
                        'sem'
                        default: 'sem'
        - CI (num)    : CI for shuffled data (e.g., 0.95)
                        default: 0.95
        - q1v4 (bool) : if True, analysis is separated across first and 
                        last quintiles
                        default: False
        - rvs (bool)  : if True, the first dataset will include regular 
                        sequences and the second will include surprise 
                        sequences
                        default: False
        - modif (bool): if True, plots are made in a modified (simplified 
                        way)
                        default: False

    """

    if modif:
        # only plot first few sessions
        limit = 3
        arr = arr[:, :limit]
        sh_arr = sh_arr[:, :limit]
        x_label = [x + 1 for x in range(arr.shape[1])]
    else:
        x_label = rois_x_label(sess_ns, arr[:, :, -2])

    if 'acc' in data_title.lower():
        if (not modif or ax.is_first_row()) and ax.is_first_col():
            if q1v4:
                ax.set_ylabel('Accuracy in Q4 (%)')
            elif rvs:
                ax.set_ylabel('Accuracy in surp (%)')
            else:
                ax.set_ylabel('Accuracy (%)')
            plot_util.set_ticks(ax, 'y', 0, 100, 6, pad_p=0)

    elif 'epoch' in data_title.lower():
        q1v4 = False # treated as if no Q4
        rvs = False # treated as if no reg v surp
        if (not modif or ax.is_first_row()) and ax.is_first_col():
            ax.set_ylabel('Nbr epochs')
        plot_util.set_ticks(ax, 'y', 0, 1000, 6, pad_p=0)

    if q1v4 or rvs:
        mean_ids = [0, 3]
        alphas = [0.3, 0.8]
        if modif:
            alphas = [0.5, 0.8]
        if q1v4:
            add_leg = [' (Q1)', ' (Q4)']
        else:
            add_leg = [' (reg)', ' (surp)']
    else:
        mean_ids = [0]
        alphas = [0.5]
        if modif:
            alphas = [0.8]
        add_leg = ['']
    
    plot_CI(ax, x_label, sh_arr, CI, q1v4 + rvs, modif)
    
    # plot non shuffle data
    main_col = 'blue'
    if plane == 'dend':
        main_col = 'green'

    if line == 'L23':
        line = 'L2/3'

    if not modif:
        cols = plot_util.get_color_range(len(mouse_ns), main_col)
        for m, mouse_n in enumerate(mouse_ns):
            leg = mouse_runs_leg(arr[m, :, -1], mouse_n, False)
            for i, m_i in enumerate(mean_ids):
                leg_i = leg.index('\n')
                leg_m = f'{leg[: leg_i]}{add_leg[i]}{leg[leg_i :]}'
                ax.errorbar(x_label, arr[m, :, m_i], yerr=arr[m, :, m_i + 1], 
                    fmt='-o', markersize=12, capthick=4, label=leg_m, 
                    alpha=alphas[i], lw=3, color=cols[m])
        for i in range(len(x_label)):
            for m, m_i in enumerate(mean_ids):
                if not np.isnan(arr[:, i, m_i]).all():
                    med = math_util.mean_med(
                        arr[:, i, m_i], axis=0, stats=stat, nanpol='omit')
                    y_lim = ax.get_ylim()
                    med_th = 0.0075*(y_lim[1]-y_lim[0])
                    ax.bar(x_label[i], height=med_th, bottom=med - med_th/2., 
                        color='black', width=0.5, alpha=alphas[m])
        title_pre = data_title[: data_title.index('data')]
        title_post = data_title[data_title.index('data') :] 
        title = f'{title_pre}{line} {plane} {title_post}'

        ax.set_title(title)

    # add a mean line
    else:
        col = plot_util.get_color(main_col, ret='single')
        for m, m_i in enumerate(mean_ids):
            if not np.isnan(arr[:, :, m_i]).all():
                meds = math_util.mean_med(arr[:, :, m_i], axis=0, stats=stat, 
                    nanpol='omit')
                errs = math_util.error_stat(arr[:, :, m_i], axis=0, 
                    stats=stat, error='sem', nanpol='omit')
                plot_util.plot_errorbars(ax, meds, err=errs, x=x_label, col=col, 
                    alpha=alphas[m])
    
    if not modif:
        ax.legend()


#############################################    
def plot_data_summ(plot_lines, data, stats, shuff_stats, title, savename, 
                   CI=0.95, q1v4=False, rvs=False, modif=False):
    """
    plot_data_summ(plot_lines, data, stats, shuff_stats, title, savename)

    Plots summary data for a specific comparison, for each line and plane and 
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
        - CI (num)    : CI for shuffled data (e.g., 0.95)
                        default: 0.95
        - q1v4 (bool) : if True, analysis is separated across first and 
                        last quintiles
                        default: False
        - rvs (bool)  : if True, the first dataset will include regular 
                        sequences and the second will include surprise 
                        sequences
                        default: False
        - modif (bool): if True, plots are made in a modified (simplified 
                        way)
                        default: False

    """
    
    celltypes = [[x, y] for y in ['dend', 'soma'] for x in ['L23', 'L5']]

    max_sess = max(plot_lines['sess_n'].tolist())

    fig, ax = init_res_fig(len(celltypes), max_sess, modif)
    
    if modif:
        fig.suptitle(title, y=1.0, weight='bold')

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
    if (q1v4 or rvs) and 'test_acc' in data:
        ext_test = sess_str_util.ext_test_str(q1v4, rvs)
        n_vals = 8 # (extra mean/med, sem/2.5p, sem/97.5p for Q4) 
        if data == 'test_acc': 
            data_types = ['test_acc', f'{ext_test}_acc']
        elif data == 'test_acc_bal':   
            data_types = ['test_acc_bal', f'{ext_test}_acc_bal']
        else:
            gen_util.accepted_values_error('data', data, 
                ['test_acc', 'test_acc_bal'])

    for i, [line, plane] in enumerate(celltypes):
        sub_ax = plot_util.get_subax(ax, i)
        if not modif:
            sub_ax.set_xlim(-0.5, max_sess - 0.5)
        # get the right rows in dataframe
        cols       = ['plane']
        cri        = [plane]
        curr_lines = gen_util.get_df_vals(plot_lines.loc[
            plot_lines['line'].str.contains(line)], cols, cri)
        cri_str = ', '.join([f'{col}: {crit}' for col, crit in zip(cols, cri)])
        if len(curr_lines) == 0: # no data
            print(f'No data found for {line} {plane}, {cri_str}')
            continue
        else: # shuffle or non shuffle missing
            skip = False
            for shuff in [False, True]:
                if shuff not in curr_lines['shuffle'].tolist():
                    print(f'No shuffle={shuff} data found for {line} {plane}, '
                        f'{cri_str}')
                    skip = True
            if skip:
                continue

        sess_ns = gen_util.get_df_vals(curr_lines, label='sess_n', dtype=int)
        mouse_ns = gen_util.get_df_vals(curr_lines, label='mouse_n', 
            dtype=int)
        # mouse x sess x n_vals 
        if -1 not in mouse_ns:
            raise ValueError('Shuffle data across mice is missing.')
        mouse_ns = gen_util.remove_if(mouse_ns, -1)
        data_arr = np.empty((len(mouse_ns), int(max_sess), n_vals)) * np.nan
        shuff_arr = np.empty((1, int(max_sess), n_vals - 1)) * np.nan

        for sess_n in sess_ns:
            sess_mice = gen_util.get_df_vals(
                curr_lines, 'sess_n', sess_n, 'mouse_n', dtype=int)
            for m, mouse_n in enumerate(mouse_ns + [-1]):
                if mouse_n not in sess_mice:
                    continue
                if mouse_n == -1:
                    stat_types = shuff_stats
                    arr = shuff_arr
                    m = 0
                else:
                    stat_types = stats
                    arr = data_arr
                curr_line = gen_util.get_df_vals(curr_lines, 
                    ['sess_n', 'mouse_n', 'shuffle'], 
                    [sess_n, mouse_n, mouse_n==-1])
                if len(curr_line) > 1:
                    raise ValueError('Several lines correspond to criteria.')
                elif len(curr_line) == 0:
                    continue
                for st, stat in enumerate(stat_types):
                    for d, dat in enumerate(data_types):
                        i = d * 3 + st
                        arr[m, int(sess_n-1), i] = curr_line[f'{dat}_{stat}']
                if mouse_n != -1:
                    arr[m, int(sess_n-1), -2] = curr_line['n_rois']
                arr[m, int(sess_n-1), -1] = curr_line['runs_total'] - \
                    curr_line['runs_nan']

        summ_subplot(sub_ax, data_arr, shuff_arr, title, mouse_ns, sess_ns, 
            line, plane, stats[0], stats[1], CI, q1v4, rvs, modif)
    
    if modif:
        n_sess_keep = 3
        ylab = ax[0, 0].get_ylabel()
        ax[0, 0].set_ylabel('')
        sess_plot_util.format_linpla_subaxes(ax, ylab=ylab, xlab='Sessions', 
            xticks=np.arange(1, n_sess_keep + 1))

        yticks = ax[0, 0].get_yticks()
        # always set ticks (even again) before setting labels
        ax[1, 0].set_yticks(yticks)
        ax[1, 0].set_yticklabels([int(v) for v in yticks], 
            fontdict={'weight': 'bold'})

    fig.savefig(savename)


#############################################    
def plot_summ(output, savename, stimtype='gabors', comp='surp', ctrl=False, 
              bri_dir='both', fluor='dff', scale=True, CI=0.95, plt_bkend=None, 
              fontdir=None, modif=False):
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
        - ctrl (bool)    : if True, control comparisons are analysed
                           default: False                           
        - bri_dir (str)  : brick direction
                           default: 'both'
        - fluor (str)    : fluorescence trace type
                           default: 'dff'
        - scale (bool)   : whether ROIs are scaled
                           default: True
        - CI (num)       : CI for shuffled data
                           default: 0.95
        - plt_bkend (str): mpl backend to use for plotting (e.g., 'agg')
                           default: None
        - fontdir (str)  : directory in which additional fonts are located
                           default: None
        - modif (bool)   : if True, plots are made in a modified (simplified 
                           way)
                           default: False

    """
    
    plot_util.manage_mpl(plt_bkend, fontdir=fontdir)

    summ_scores_file = os.path.join(output, savename)
    
    if os.path.exists(summ_scores_file):
        summ_scores = file_util.loadfile(summ_scores_file)
    else:
        print(f'{summ_scores_file} not found.')
        return

    if len(summ_scores) == 0:
        print(f'No data in {summ_scores_file}.')
        return

    # drop NaN lines
    summ_scores = summ_scores.loc[~summ_scores['epoch_n_mean'].isna()]

    data_types  = ['epoch_n', 'test_acc', 'test_acc_bal']
    data_titles = ['Epoch nbrs', 'Test accuracy', 'Test accuracy (balanced)']

    stats = ['mean', 'sem', 'sem']
    shuff_stats = ['median'] + math_util.get_percentiles(CI)[1]

    q1v4, rvs = False, False
    if 'q1v4' in output:
        q1v4 = True
    elif 'rvs' in output:
        rvs = True
    
    runtype = 'prod'
    if 'pilot' in output:
        runtype = 'pilot'

    if stimtype == 'gabors':
        bri_dir = 'none'
        stim_str = 'gab'
        stim_str_pr = 'gabors'

    else:
        bri_dir_vals = sess_gen_util.get_params(stimtype, bri_dir)[0]
        stim_str = sess_str_util.dir_par_str(bri_dir_vals, str_type='file')
        stim_str_pr = sess_str_util.dir_par_str(bri_dir_vals, str_type='print')

    scale_str = sess_str_util.scale_par_str(scale, 'file')
    scale_str_pr = sess_str_util.scale_par_str(scale, 'file').replace('_', ' ')
    ctrl_str = sess_str_util.ctrl_par_str(ctrl)
    ctrl_str_pr = sess_str_util.ctrl_par_str(ctrl, str_type='print')
    modif_str = '_modif' if modif else ''

    save_dir = os.path.join(output, f'figures_{fluor}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    cols = ['scale', 'fluor', 'bri_dir', 'runtype']
    cri  = [scale, fluor, bri_dir, runtype]
    plot_lines = gen_util.get_df_vals(summ_scores, cols, cri)
    cri_str = ', '.join([f'{col}: {crit}' for col, crit in zip(cols, cri)])
    if len(plot_lines) == 0: # no data
        print(f'No data found for {cri_str}')
        return
    else: # shuffle or non shuffle missing
        skip = False
        for shuff in [False, True]:
            if shuff not in plot_lines['shuffle'].tolist():
                print(f'No shuffle={shuff} data found for {cri_str}')
                skip = True
        if skip:
            return

    for data, data_title in zip(data_types, data_titles):
        if not modif:
            title = (f'{stim_str_pr.capitalize()} {comp}{ctrl_str_pr} - '
                f'{data_title} for log regr on\n' + 
                u'{} {} '.format(scale_str_pr, fluor) + 
                f'data ({runtype})')
        else:
            title = (f'{stim_str_pr.capitalize()} {comp}{ctrl_str_pr} - '
                f'{data_title}')
        
        if '_' in title:
            title = title.replace('_', ' ')

        savename = (f'{data}_{stim_str}_{comp}{ctrl_str}{scale_str}'
            f'{modif_str}.svg')
        full_savename = os.path.join(save_dir, savename)
        plot_data_summ(plot_lines, data, stats, shuff_stats, title, 
            full_savename, CI, q1v4, rvs, modif)

    plt.close('all')

