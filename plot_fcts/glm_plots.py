"""
glm_plots.py

This script contains functions to plot results of GLM analyses (glm.py) from 
dictionaries.

Authors: Colleen Gillon

Date: October, 2019

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
def plot_from_dict(dict_path, plt_bkend=None, fontdir=None):
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
    """

    print('\nPlotting from dictionary: {}'.format(dict_path))
    
    figpar = sess_plot_util.init_figpar(plt_bkend=plt_bkend, fontdir=fontdir)
    plot_util.manage_mpl(cmap=False, **figpar['mng'])

    info = file_util.loadfile(dict_path)
    savedir = os.path.dirname(dict_path)

    analysis = info['extrapar']['analysis']

    # 0. Plots the explained variance
    if analysis == 'v': # difference correlation
        plot_glm_expl_var(figpar=figpar, savedir=savedir, **info)


    else:
        print('No plotting function for analysis {}'.format(analysis))


#############################################
def plot_glm_expl_var(analyspar, sesspar, stimpar, extrapar, glmpar,
                      sess_info, all_expl_var, figpar=None, savedir=None):
    """
    plot_pup_diff_corr(analyspar, sesspar, stimpar, extrapar, 
                       sess_info, all_expl_var)

    From dictionaries, plots explained variance for different variables for 
    each ROI.

    Required args:
        - analyspar (dict)    : dictionary with keys of AnalysPar namedtuple
        - sesspar (dict)      : dictionary with keys of SessPar namedtuple 
        - stimpar (dict)      : dictionary with keys of StimPar namedtuple
        - glmpar (dict)       : dictionary with keys of GLMPar namedtuple
        - extrapar (dict)     : dictionary containing additional analysis 
                                parameters
            ['analysis'] (str): analysis type (e.g., 'c')
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
        - all_expl_var (list) : list of dictionaries with explained variance 
                                for each session set, with each glm 
                                coefficient as a key:
            ['full'] (list)    : list of full explained variance stats for 
                                 every ROI, structured as ROI x stats
            ['coef_all'] (dict): max explained variance for each ROI with each
                                 coefficient as a key, structured as ROI x stats
            ['coef_uni'] (dict): unique explained variance for each ROI with 
                                 each coefficient as a key, 
                                 structured as ROI x stats
            ['rois'] (list)    : ROI numbers (-1 for GLMs fit to 
                                 mean/median ROI activity)
    
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
    dendstr_pr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                            'roi', 'print')

    sessstr = sess_str_util.sess_par_str(sesspar['sess_n'], stimpar['stimtype'], 
                                         sesspar['layer'], stimpar['bri_dir'],
                                         stimpar['bri_size'], stimpar['gabk']) 
    dendstr = sess_str_util.dend_par_str(analyspar['dend'], sesspar['layer'], 
                                         'roi')

    # extract some info from sess_info
    keys = ['mouse_ns', 'sess_ns', 'lines', 'layers', 'nrois']
    [mouse_ns, sess_ns, lines, layers, nrois] = [sess_info[key] for key in keys]

    n_sess = len(mouse_ns)
    nanroi_vals = [sess_info['nanrois'], sess_info['nanrois_dff']]
    [n_nan, n_nan_dff] = [[len(val[i]) for i in range(n_sess)] 
                                       for val in nanroi_vals]

    plot_bools = [True for ev in all_expl_var if ev['rois'] != [-1]]
    n_sess = sum(plot_bools)

    if stimpar['stimtype'] == 'gabors':
        xyzc_dims = ['surp', 'gabfr', 'pup_diam_data', 'run_data']
        print_dims = xyzc_dims + ['gab_ori']
    elif stimpar['stimtype'] == 'bricks':
        xyzc_dims = ['surp', 'bri_dir', 'pup_diam_data', 'run_data']
        print_dims = xyzc_dims

    print(('Plotting GLM full and unique explained variance for '
           '{}.').format(', '.join(xyzc_dims)))

    if n_sess > 0:
        if figpar is None:
            figpar = sess_plot_util.init_figpar()

        figpar = copy.deepcopy(figpar)
        cmap = plot_util.manage_mpl(cmap=True, nbins=100, **figpar['mng'])

        if figpar['save']['use_dt'] is None:
            figpar['save']['use_dt'] = gen_util.create_time_str()
        figpar['init']['ncols'] = n_sess
        figpar['save']['fig_ext'] = 'png'
        
        fig, ax = plot_util.init_fig(2 * n_sess, **figpar['init'], proj='3d')

        fig.suptitle('Explained variance per ROI')
    else:
        print('No plots, as only results across ROIs are included')
        fig = None

    i = 0
    for e, expl_var in enumerate(all_expl_var):
        if expl_var['rois'] == ['all']:
            plot_bools[e] = False

        # collect info for plotting and print results across ROIs
        rs = np.where(np.asarray(expl_var['rois']) != -1)[0]
        all_rs = np.where(np.asarray(expl_var['rois']) == -1)[0]
        if len(all_rs) != 1:
            raise ValueError('Expected only one results for all ROIs.')
        else:
            all_rs = all_rs[0]
            full_ev = expl_var['full'][all_rs]


        sess_nrois = sess_gen_util.get_nrois(nrois[e], n_nan[e], n_nan_dff[e], 
                                   analyspar['remnans'], analyspar['fluor'])

        title = (u'Mouse {} - {}\n(sess {}, {} {}{}, (n={}))').format(
                  mouse_ns[i], stimstr_pr, sess_ns[i], lines[i], layers[i], 
                  dendstr_pr, sess_nrois)
        print(u'\n{}'.format(title))
        
        math_util.print_stats(full_ev, stat_str='\nFull explained variance')


        for v, var_type in enumerate(['coef_all', 'coef_uni']):
            if var_type == 'coef_all':
                sub_title = 'Explained variance per coefficient'
            elif var_type == 'coef_uni':
                sub_title = 'Unique explained variance per coefficient'
            print('\n{}'.format(sub_title))

            dims_all = []
            for key in print_dims:
                if key in xyzc_dims:
                    # get mean/med
                    dims_all.append(np.asarray(expl_var[var_type][key])[rs, 0])
                math_util.print_stats(expl_var[var_type][key][all_rs], 
                                      stat_str=key)
            if not plot_bools[-1]:
                continue

            [x, y, z, c] = dims_all
            
            if v == 0:
                subpl_title = u'{}\n{}'.format(title, sub_title)
            else:
                subpl_title = sub_title

            sub_ax = ax[v, i]
            im = sub_ax.scatter(x, y, z, c=c, cmap=cmap, vmin=0, vmax=1)
            sub_ax.set_title(subpl_title)
            # sub_ax.set_zlim3d(0, 1.0)
            sub_ax.set_xlabel(xyzc_dims[0])
            sub_ax.set_ylabel(xyzc_dims[1])
            sub_ax.set_zlabel(xyzc_dims[2])
            if v == 0:
                full_ev_lab = math_util.print_stats(full_ev, stat_str='Full EV')
                sub_ax.plot([], [], c='k', label=full_ev_lab)
                sub_ax.legend()

        i += 1

    if savedir is None:
        savedir = os.path.join(figpar['dirs']['roi'],figpar['dirs']['glm'])

    savename = ('{}_glm_ev_{}{}').format('roi', sessstr, dendstr)

    if n_sess > 0:
        plot_util.add_colorbar(fig, im, n_sess, label=xyzc_dims[3])

    fulldir = plot_util.savefig(fig, savename, savedir, **figpar['save'])

    return fulldir, savename                              

