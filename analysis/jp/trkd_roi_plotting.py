"""
trkd_roi_analysis.py

This module run various anlyses on the tracked ROI USIs (Unexpected event
Selectivity Indices).

Authors: Jason E. Pina

Last modified: 29 May 2021
"""

import numpy as np
import itertools as it

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colors as mcolors
import seaborn as sns





#############################################

def plot_usis_over_sessions(tracked_roi_usi_df, stimtype, alpha_plot=0.2, 
                            alpha_hline=0.3, alpha_mean=0.4):
    """
    Plot USIs of tracked ROIs across sessions
    
    Parameters
    ----------
    tracked_roi_usi_df : Pandas DataFrame
    
    stimtype : string
    
    alpha_plot : number; optional, default = 0.2
        Transparency level for plot
    alpha_hline : number; optional, default = 0.3
        Transparency level for y=0 line
    alpha_mean : number; optional, default = 0.4
        Transparency level for mean line
    """
    
    sess_ns = [1,2,3]
    
    fig, axs = plt.subplots(2,2, figsize=(10,14), constrained_layout=True)
    
    color_dict = {'dend' : mcolors.CSS4_COLORS['limegreen'], 
                  'soma' : mcolors.CSS4_COLORS['cornflowerblue'] }
    stimstr_dict = {'gabors':'Gabors', 'bricks':'Bricks'}

    print(stimstr_dict[stimtype])
    for i_compartment,compartment in enumerate(['dend', 'soma']):
        for i_layer,layer in enumerate(['L2/3', 'L5']):
            print(layer, compartment)
            mask0 = tracked_roi_usi_df['layer']==layer
            mask1 = tracked_roi_usi_df['compartment']==compartment
            high_low_neither = []
            sig_high_low_neither = []
            n_rois_sess = high_error_prob = []
            surp_sel = surp_sel_perms = []
            for sess_n in sess_ns:
                surp_sel.append(np.hstack(tracked_roi_usi_df[(mask0 & mask1)]
                                          ['sess_{}_usi'.format(sess_n)].
                                          values))
            surp_sel = np.asarray(surp_sel)
            plt.sca(axs[i_compartment, i_layer])
            plt.plot(sess_ns, surp_sel, linewidth=3, 
                     color=color_dict[compartment], alpha=alpha_plot)
            plt.plot(sess_ns, np.nanmean(surp_sel, axis=1), linewidth=5, 
                     color='black', alpha=alpha_mean)
            plt.axhline(y=0, linestyle='--', linewidth=5, color='black', 
                        alpha=alpha_hline)
            plt.xticks(sess_ns)
            axs[i_compartment, i_layer].spines['right'].set_visible(False)
            axs[i_compartment, i_layer].spines['top'].set_visible(False)

            plt.title(r'$\it {}\,{}$'.format(layer, compartment), fontsize=18)
            plt.xlabel(r'$\bf Session$'+'\n', fontsize=15)
            plt.ylabel(r'$\bf ROI\,USIs$', fontsize=15)
            
    opstr = 'd\''
    plt.suptitle('{}: ROI USIs across sessions\nMetric: {}\n\n'
                 .format(stimstr_dict[stimtype], opstr), fontsize=20)
    
    return fig, axs

#############################################

def plot_mean_abs_usis_over_sessions(tracked_roi_usi_df, stimtype, 
                                     alpha_plot=0.4, alpha_hline=0.3):
    """
    Plot <|USIs|> across sessions
    
    Parameters
    ----------
    tracked_roi_usi_df : Pandas DataFrame
        Dataframe of USIs and related statistics for tracked ROIs
    stimtype : string
        Stimulus type: 'Gabors' or 'bricks'
    alpha_plot : number; optional, default = 0.4
        Transparency level for plot
    alpha_hline : number; optional, default = 0.3
        Transparency level for y=0 line
    """
    
    sess_ns = [1,2,3]
    
    fig, axs = plt.subplots(2,2, figsize=(10,14), constrained_layout=True)
    
    color_dict = {'dend' : mcolors.CSS4_COLORS['limegreen'], 
                  'soma' : mcolors.CSS4_COLORS['cornflowerblue'] }
    stimstr_dict = {'gabors':'Gabors', 'bricks':'Bricks'}

    print(stimstr_dict[stimtype])
    for i_compartment,compartment in enumerate(['dend', 'soma']):
        for i_layer,layer in enumerate(['L2/3', 'L5']):
            print(layer, compartment)
            mask0 = tracked_roi_usi_df['layer']==layer
            mask1 = tracked_roi_usi_df['compartment']==compartment
            high_low_neither = []
            sig_high_low_neither = []
            n_rois_sess = []
            high_error_prob = []
            surp_sel = []
            surp_sel_perms = []
            for i_sess,sess_n in enumerate(sess_ns):
                surp_sel.append(np.hstack(tracked_roi_usi_df[(mask0 & mask1)]
                                          ['sess_{}_usi'.format(sess_n)].
                                          values))
                high_error_prob.append(np.std(surp_sel[i_sess]) / 
                                       np.sqrt(surp_sel[i_sess].shape[0]))
            surp_sel = np.asarray(surp_sel)
            surp_sel_mean = np.nanmean(surp_sel, axis=1)
            surp_sel_mean_abs = np.nanmean(np.abs(surp_sel), axis=1)
            linestyle = 'solid' if layer == 'L2/3' else (0,(5,10))
            
            plt.sca(axs[i_compartment, i_layer])
            plt.errorbar(sess_ns, surp_sel_mean_abs, yerr=high_error_prob,
                         color=color_dict[compartment], alpha=alpha_plot, 
                         linestyle=linestyle, marker='.', lw=3, ms=15, 
                         elinewidth=3)
            plt.axhline(y=0, linestyle='--', linewidth=5, color='black', 
                        alpha=alpha_hline)
            plt.xticks(sess_ns)
            axs[i_compartment, i_layer].spines['right'].set_visible(False)
            axs[i_compartment, i_layer].spines['top'].set_visible(False)

            newlines = '\n\n'
            plt.title(r'$\it {}\,{}${}'.format(layer, compartment, newlines), 
                      fontsize=18)
            plt.xlabel(r'$\bf Session$'+'\n', fontsize=15)
            plt.ylabel(r'$\bf ROI\,VSIs$', fontsize=15)
            
    opstr = 'd\''
    suptitle = '{}: Mean of abs. values of ROI selectivities'. \
               format(stimstr_dict[stimtype]) + \
               'across sessions\nMetric: {}\n\n'. \
               format(opstr)
    plt.suptitle(suptitle, fontsize=20)
    
    return fig, axs

#############################################

def plot_usi_abs_frac_chng(usi_abs_frac_chng_df, bonf_n):
    """
    Plot <|USI|> (over ROIs) fractional changes from sessions 1 to 3 for each 
    compartment/layer and all compartments/layers for each stimulus, and compare 
    these changes between stimuli
    
    Parameters
    ----------
    usi_abs_frac_chng_df : Pandas DataFrame
    usi_changes_df : Pandas DataFrame
        Dataframe with <|USI|> (over ROIs) fractional changes from sess 1 to 3, 
        uncertainty, and p-values of comparisons across stimuli
    bonf_n : number
        Number of multiple comparisons to correct for.  Threshold for p-value to 
        be considered significant = 0.05/bonf_n
    """
    
    colors = ['slateblue','firebrick']
    plot_rows = 3
    plot_cols = 2
    fig, ax = \
        plt.subplots(plot_rows, plot_cols, 
                     figsize=(plot_cols*2.5, plot_cols*6+1), 
                     constrained_layout=True)
    for plot_row, compartment in enumerate(usi_abs_frac_chng_df['compartment'].
                                           unique()):
        for plot_col, layer in enumerate(usi_abs_frac_chng_df['layer'].
                                         unique()[:-1]):
            if compartment == 'all':
                if plot_col==1:
                    ax[plot_row, plot_col].axis('off')
                    continue
                layer = 'all'
            print(layer, compartment)
            mask0 = usi_abs_frac_chng_df['layer']==layer
            mask1 = usi_abs_frac_chng_df['compartment']==compartment
            gab_frac_chng = usi_abs_frac_chng_df[mask0 & mask1] \
                            ['gab_mn_abs_frac_chng'].values[0]
            brk_frac_chng = usi_abs_frac_chng_df[mask0 & mask1] \
                            ['brk_mn_abs_frac_chng'].values[0]
            gab_bstrap_std = usi_abs_frac_chng_df[mask0 & mask1] \
                            ['gab_bstrap_std'].values[0]
            brk_bstrap_std = usi_abs_frac_chng_df[mask0 & mask1] \
                            ['brk_bstrap_std'].values[0]
            pval = usi_abs_frac_chng_df[mask0 & mask1]['pval_raw'].values[0]
            sess_compare = \
                usi_abs_frac_chng_df[mask0 & mask1]['sess_compare'].values[0]
            subplot_usi_abs_frac_chng(gab_frac_chng, brk_frac_chng, 
                                      gab_bstrap_std, brk_bstrap_std, pval,
                                      bonf_n, layer, compartment, sess_compare, 
                                      ax[plot_row, plot_col])    
    plt.suptitle('USI changes\n for unexpected events\nSession {} v {}\n'.
                 format(sess_compare[0], sess_compare[1]), fontsize=20)

    return fig

#############################################

def subplot_usi_abs_frac_chng(gab_frac_chng, brk_frac_chng, gab_bstrap_std, 
                              brk_bstrap_std, pval_raw, bonf_n, layer, 
                              compartment, sess_compare, ax):
    """
    Make each layer/compartment subplot for <|USI|> (over ROIs) fractional 
    changes 
    
    Parameters
    ----------
    gab_frac_chng : 1-D arraylike of numbers
        Gabor fractional change from session 1 to 3. 1 entry for each 
        layer/compartment
    brk_frac_chng : 1-D arraylike of numbers
        Visual flow fractional change from session 1 to 3. 1 entry for each 
        layer/compartment
    gab_bstrap_std : number
        Bootstrapped stdev for 'gab_frac_chng'
    brk_bstrap_std : number
        Bootstrapped stdev for 'brk_frac_chng'
    pval_raw : number
        Raw p-value for comparison between fractional changes for Gabor vs. 
        visual flow stimuli for each layer/compartment
    bonf_n : number
        Number of multiple comparisons to correct for
    layer : string
        Which layer to plot ('L2/3' or 'L5')
    compartment : string
        Which compartment to plot ('dend' or 'soma')
    sess_compare : arraylike of numbers
        Which sessions used to compute fractional changes
    ax : axis handle for subplot
    """

    alpha_001 = 0.001/bonf_n
    alpha_01  = 0.01/bonf_n
    alpha_05  = 0.05/bonf_n
    colors = ['slateblue','firebrick']
    compartment_dict = {'dend':'dendrites', 'soma':'somata'}
    if pval_raw <= alpha_001:
        sig_str = '***'
    elif pval_raw <= alpha_01:
        sig_str = '**'
    elif pval_raw <= alpha_05:
        sig_str = '*'
    
    plt.sca(ax)
    plt.bar(['Gabors', 'Bricks'], [np.mean(gab_frac_chng), 
            np.mean(brk_frac_chng)], color=colors, width=0.5, alpha=0.8)
    err_ylows  = [max(np.mean(gab_frac_chng)-gab_bstrap_std/2, 0), 
                  max(np.mean(brk_frac_chng)-brk_bstrap_std/2, 0)]
    err_yhighs = [np.mean(gab_frac_chng)+gab_bstrap_std/2, 
                  np.mean(brk_frac_chng)+brk_bstrap_std/2]
    plt.vlines(['Gabors', 'Bricks'], err_ylows, err_yhighs, color='black')

    if layer=='all':
        gab_arr = np.repeat(['Gabors'], len(gab_frac_chng))
        brk_arr = np.repeat(['Bricks'], len(brk_frac_chng))
        x0 = np.concatenate((gab_arr, brk_arr))
        y0 = np.concatenate((gab_frac_chng, brk_frac_chng))
        sns.set_palette('gray')
        sns.stripplot(x=x0, y=y0, jitter=0.2, size=10, alpha=0.6);
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if pval_raw <= alpha_05:
        ymin = plt.ylim()[0]
        ymax = plt.ylim()[1]
        yrange = ymax-ymin
        y_hline = ymax+0.05*yrange
        y_text = ymax+0.1*yrange
        plt.hlines(y_hline, 0, 1)
        plt.text(0.5, y_text, sig_str, fontsize=15, 
                 horizontalalignment='center')

    
    if layer!='all':
        plt.title('{} {}\nCorrected p-value =\n {:.3f}\n'.
                  format(layer, compartment_dict[compartment], 
                  min(bonf_n*pval_raw, 1)), fontsize=15);
    else:
        plt.title('All compartments\nCorrected p-value =\n {:.3f}\n'.
                  format(min(bonf_n*pval_raw, 1)), fontsize=15);        

#############################################

def plot_usi_corr(usi_corr_df, stim_sess):
    """
    Plot USI correlations
    
    Parameters
    ----------
    usi_corr_df : Pandas dataframe
        Dataframe with USI correlation data for each layer/compartment
    stim_sess : Iteration Tools cartesian product
        Contains all stimulus and start session combinations
    """
    
    bonf_n = 8
    stim_str = ['Gabor Sequences', 'Visual Flow']
    sess_str = ['Session 1 to Session 2', 'Session 2 to Session 3']
    fig, ax = plt.subplots(2,2, figsize=(10,10), constrained_layout=True)
    for i, (stimtype, sess) in enumerate(stim_sess):
        mask0 = usi_corr_df['stimtype']==stimtype
        mask1 = usi_corr_df['usi_base_sess']==sess
        df = usi_corr_df[mask0 & mask1]
        plot_row = int(np.floor(i/2))
        plot_col = int(np.mod(i,2))
        plt.sca(ax[plot_row, plot_col])
        
        layers = {'L23-Cux2':'L2/3', 'L5-Rbp4':'L5'}
        lc = list(it.product(layers.values(), ['dend', 'soma']))
        lc_str = [ls_s[0] + ' ' + ls_s[1] for ls_s in lc]

        color_dict = {'dend' : mcolors.CSS4_COLORS['limegreen'], 
                      'soma' : mcolors.CSS4_COLORS['cornflowerblue'] }
        colors = [color_dict[lc_spec[1]] for lc_spec in lc]
        corr_resid = df['corr_resid']

        x = range(len(lc_str))
        plt.bar(x, df['corr_resid'].values, width=0.6, alpha=0.5, color=colors)
        plt.vlines(x, df['corr_resid_low_sd'].values, df['corr_resid_high_sd'].
                   values)
        plt.bar(x, df['corr_resid_null_low_ci'].values, alpha=0.5, color='gray')
        plt.bar(x, df['corr_resid_null_high_ci'].values, alpha=0.5, 
                color='gray')
        plt.xticks(x, lc_str)
        plt.axhline(0, np.min(x)-1, np.max(x)+1, color='black', linestyle='--')
        for i,xval in enumerate(x):
            if df['sig_correc'].values[i] != 'False':
                text = '       *\n{:.2e}'.format(df['p_val_raw'].
                                                 values[i]*bonf_n)
                plt.text(xval-0.3, plt.ylim()[1], text)
        plt.title(stim_str[plot_row] + '\n' + sess_str[plot_col] + '\n\n')
        plt.ylim([-1,1])
        ax[plot_row, plot_col].spines['top'].set_visible(False)
        ax[plot_row, plot_col].spines['right'].set_visible(False)
        suptitle='USI Difference \nwith corrected p-values'
        plt.suptitle(suptitle)
    
    return fig

#############################################

def plot_corr_scatt_hist_perm(usi_corr_df, corr_perm, stimtype, layer, 
                              compartment, usi_base_sess):
    """
    Plot scatter and histogram plots for chosen compartment/layer, along with
    the scatter data from the median of the shuffled distribution

    Parameters
    ----------
    usi_corr_df : Pandas DataFrame
        Dataframe with USI correlation data
    corr_perm : 2-D array of numbers
        Median USI, Delta(USI) array of shuffled distribution
    stimtype : string
        Stimulus type ('gabors' or 'bricks')
    layer : string
        Layer for which to get data
    compartment : string
        Compartment for which to get data
    usi_base_sess : number
        Session for USIs against which to compare Delta(USI) (with the 
        following session)
    """

    stim_str = {'gabors':'Gabor Sequences', 'bricks':'Visual Flow'}
    fig, ax = plt.subplots(1,2, figsize=(12,6), constrained_layout=True)
    if compartment=='dend':
        color = mcolors.CSS4_COLORS['limegreen']
    elif compartment=='soma':
        color = mcolors.CSS4_COLORS['cornflowerblue']
    mask0 = usi_corr_df['stimtype']==stimtype
    mask1 = usi_corr_df['layer']==layer
    mask2 = usi_corr_df['compartment']==compartment
    mask3 = usi_corr_df['usi_base_sess']==usi_base_sess
    df = usi_corr_df[mask0 & mask1 & mask2 & mask3]
    for i,plot in enumerate(['scatt', 'hist']):
        corr_raw = df['corr_raw'].values[0]
        corr_resid = df['corr_resid'].values[0]
        corr_raw_distro = df['corr_raw_distro'].values[0]
        corr_med = np.median(corr_raw_distro)
        pval_raw = df['p_val_raw'].values[0]
        pvalchar = '+' if corr_resid > 0 else '-'
        plt.sca(ax[i])
        if plot == 'scatt':
            x = df['usi'].values[0]
            y = df['delta_usi'].values[0]
            x_perm = corr_perm[:,0]
            y_perm = corr_perm[:,1]
            plt.scatter(x, y, color=color, alpha=0.65)
            plt.scatter(x_perm, y_perm, color='gray', alpha=0.65)
            xlabel = 'Session {} selectivity'.format(usi_base_sess)
            ylabel = 'Sel {} and Sel {} diff'.format(usi_base_sess, 
                                                     usi_base_sess+1)
            plt.title('Scatter plot\n' + 'Color: Data\nGray: Median from ' +
                      'shuffled distribution')
        elif plot == 'hist':
            plt.hist(corr_raw_distro, bins='auto', color=color, alpha=0.65, 
                     density=True)
            plt.axvline(corr_raw, color='black')
            plt.axvline(corr_med, color='black', linestyle='--', alpha=0.5)
            ylabel = 'Density'
            xlabel = 'Pearson correlation values'
            plt.title('Histogram of Pearson correlation values\nfor shuffled ' +
                      'sessions\nSolid line: Raw correlation\nDashed line: ' +
                      'Median of shuffled distribution')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
    plt.suptitle(layer + ' ' + compartment + '\nCorrelations: ' +
                 'Corrected residual = {:.2f}, raw = {:.2f}, median = {:.2f}'.
                 format(corr_resid, corr_raw, corr_med) + 
                 '\n 2-tailed raw p-val = {:.2e} ({}) (shuffle days)'.
                 format(pval_raw, pvalchar) +
                 '\nNumber of ROIs = {}'.format(len(x)))
    return fig
