"""
mean_dff_over_days_plotting.py

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

def make_text_slide(text, figsize):
    """
    How to apply text between slides
    """
    
    fig = plt.figure(figsize=figsize)
    plt.text(0.5, 0.5, text,  horizontalalignment='center',
             verticalalignment='center', size=75, style='italic')
    plt.xticks([],[])
    plt.yticks([],[])
    plt.axis('off')
    return fig

#############################################

def plot_layer_compartment_df(df, mouse_df, frames, which_sessns, figsize,
                              layers = ['L2/3', 'L5'], 
                              compartments = ['dend', 'soma'],
                              expec_str_list=['expec', 'unexp'], 
                              title_frames=False):
    '''
    Plot normalized or unnormalized mean +/- SEM df/f across sessions for each 
    layer/compartment.
    
    Parameters
    ----------
    df : Pandas DataFrame
        Contains, for each mouse/session/ROI, df/f values averaged across all 
        sequence presentations in the session. For visual flow (bricks), the 
        data are segregated into expected and unexpected, while for Gabors, the 
        data are further grouped into Gabor frames (A, B, C, D, U, G).  Note, 
        can be normalized or unnormalized data.
    mouse_df : Pandas DataFrame
        Dataframe with mouse metadata
    frames : 1-D arraylike of strings
        Gabor frames being plotted. Note, for bricks there are no frames, and so
        this simply displays the stimulus type if title_frames=True, else this 
        is just a dummy variable.
    which_sessns : 1-D arraylike of numbers
        Sessions included in data
    figsize : Tuple
        Figure size.
    layers : 1-D arraylike of strings; optional, default = ['L2/3', 'L5']
        Layers for which to plot data.
    compartments : 1-D arraylike of strings; optional, 
     default = ['dend', 'soma']
        Compartments for which to plot data.
    expec_str_list : 1-D arraylike of strings; optional, 
     default = ['expec', 'unexp']
        Specifies which sequences are being plotted ('all', 'expec', 'unexpec').
    title_frames : boolean; optional, default = False
        Option to include Gabor frames in title (if visual flow and this is 
        True, the associated dummy variable will be displayed).

    
    Returns
    -------
    fig : Matplotlib figure handle
        Handle to the full layer/compartment figure with subplots
    '''
    
    # declarations/initializations
    dff_dict = {'expec':{'dff':'expec_dff__all_rois', 
                         'dff_mn':'expec_dff__mn__sess_123',
                         'dff_se':'expec_dff__se__sess_123'},
                'unexp':{'dff':'unexp_dff__all_rois', 
                        'dff_mn':'unexp_dff__mn__sess_123',
                        'dff_se':'unexp_dff__se__sess_123'}}
    n_rows = int(len(expec_str_list))
    fig, ax = plt.subplots(n_rows, 4, figsize=figsize, constrained_layout=True)

    # Loop over surprise and layers/compartments for subplots
    for plt_row,expec_str in enumerate(expec_str_list):
        plt_col=-1
        for compartment, layer in it.product(compartments, layers):
            plt_col+=1
            print('layer', layer, 'compartment', compartment)
            # Set masks
            mask0 = df.compartment==compartment
            mask1 = df.layer==layer
            masks = mask0 & mask1
            
            # Obtain data from dataframe
            lc_mouse_ns = df[masks]['mouse_ns'].values[0]
            dff = df[masks][dff_dict[expec_str]['dff']].values[0]
            dff_mn = df[masks][dff_dict[expec_str]['dff_mn']].values[0]
            dff_se = df[masks][dff_dict[expec_str]['dff_se']].values[0]
            num_rois = df[masks]['sess_123_num_rois'].values[0]
            pval__1_2 = df[masks]['pval_1_2__{}'.format(expec_str)].values[0]
            pval__2_3 = df[masks]['pval_2_3__{}'.format(expec_str)].values[0]
            pval__1_3 = df[masks]['pval_1_3__{}'.format(expec_str)].values[0]

            # Plot subplots
            ax_handle = ax[plt_row,plt_col] if n_rows > 1 else ax[plt_col]
            subplot_layer_compartment(ax_handle, layer, compartment, 
                                      which_sessns, lc_mouse_ns, frames,
                                      dff_mn, dff_se,
                                      pval__1_2, pval__2_3, pval__1_3, 
                                      expec_str, title_frames=title_frames)

    # Figure title
    if len(expec_str_list) > 1:
        top_text = 'Top row: Expected sequences\n' if len(expec_str_list)==2 \
            else 'Top row: All sequences\n'
        middle_text = '' if len(expec_str_list)==2 else \
            'Middle row: Expected sequences\n'

        plt.suptitle('Compartment amalgam\n' + 
                     'Frames {}\n\n'.format(frames) +
                     'p-values are 2-tailed\n\n' + 
                     top_text + middle_text + 
                     'Bottom row: Unexpected sequences\n' +
                     'p-values are 2-tailed')                     

    return fig

#############################################

def subplot_layer_compartment(ax, layer, compartment, which_sessns, mouse_ns, 
                              frames, dff_mn, dff_se, 
                              pval__1_2, pval__2_3, pval__1_3,
                              expec_str, with_pval=True, 
                              title_frames=False):
    """
    Plot df/f mean +/- SEM over sessions subplot for specified 
    layer/compartment.
    
    Parameters
    ----------
    ax : Matplotlib axis handle
    layer : string; optional, default = 'blank'
        Layer ('L2/3', 'L5') for which to obtain data.
    compartment : string; optional, default = 'blank'
        Compartment ('dend', 'soma') for which to obtain data.
    which_sessns : 1-D arraylike of numbers
        Sessions included in data.
    mouse_ns : 1-D arraylike of numbers
        Mice comprising given layer/compartment.
    frames : 1-D arraylike of strings
        Gabor frames being plotted. Note, for bricks there are no frames, and so
        this simply displays the stimulus type if title_frames=True, else this 
        is just a dummy variable.
    expec_str : string
        Specifies which sequences are being plotted ('all', 'expec', 'unexpec').
    with_pval : boolean; optional, default = True
        Option to plot Bonferroni-corrected p-values or not.
    title_frames : boolean; optional, default = False
        Option to include Gabor frames in title (if visual flow and this is 
        True, the associated dummy variable will be displayed).
    """
    
    # declarations/initializations
    color_dict = {'dend' : mcolors.CSS4_COLORS['limegreen'], 
                  'soma' : mcolors.CSS4_COLORS['cornflowerblue'] }
    expec_str_dict = {'expec':'Expected sequences', 
                      'unexp':'Unexpected sequences'}
    frms_title_dict = {'A':{'expec':'A, B, C', 'unexp':'A, B, C'},
                       'D/U':{'expec':'D, G', 'unexp':'U, G'}}
    alpha = 0.05 
    pval = [pval__1_2, pval__2_3, pval__1_3]


    # Plot the subplots
    plt.sca(ax)
    plt.errorbar(which_sessns, dff_mn, dff_se, color=color_dict[compartment], 
                 linewidth=5, marker='o', markersize=10, elinewidth=3)
    
    # Customize plotting parameters
    plt.xlim(which_sessns[0]-0.1, which_sessns[2]+0.1)
    plt.xticks(ticks=which_sessns)
    if title_frames:
        plt.title('{}\n'.format(expec_str_dict[expec_str]) +
                  '{} {}\nMice: {}\n'.format(layer, compartment, mouse_ns) +
                  'Frames {}\n\n\n'.format(frms_title_dict[frames[0]]
                  [expec_str]))
    else:
        plt.title('{}\n'.format(expec_str_dict[expec_str]) +
                  '{} {}\nMice: {}\n\n\n'.format(layer, compartment, mouse_ns))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Significance decorations
    line_xlims = [[1,2],[2,3],[1,3]]
    line_yvals = [plt.ylim()[1]+0.3*(plt.ylim()[1]-plt.ylim()[0]),
                  plt.ylim()[1]+0.15*(plt.ylim()[1]-plt.ylim()[0]),
                  plt.ylim()[1]+0.0*(plt.ylim()[1]-plt.ylim()[0])]

    text_xvals = [1.5, 2.5, 1.5]
    text_yvals = [line_yvals[0]-0.06*(plt.ylim()[1]-plt.ylim()[0]),
                  line_yvals[1]-0.06*(plt.ylim()[1]-plt.ylim()[0]),
                  line_yvals[2]-0.06*(plt.ylim()[1]-plt.ylim()[0])]
    text_yvals_ss = [line_yvals[0]-0.06*(plt.ylim()[1]-plt.ylim()[0]),
                     line_yvals[1]-0.06*(plt.ylim()[1]-plt.ylim()[0]),
                     line_yvals[2]-0.06*(plt.ylim()[1]-plt.ylim()[0])]

    if with_pval:
        for i_x,xval in enumerate(which_sessns):            
            if np.isnan(pval[i_x]):
                continue
            plt.hlines(line_yvals[i_x], line_xlims[i_x][0], line_xlims[i_x][1],
                       color=color_dict[compartment])
            if pval[i_x] <= alpha:
                text = '   *\np-val =\n {:.2f}\n'.format(pval[i_x])
                plt.text(text_xvals[i_x], text_yvals_ss[i_x], text)
            else:
                text = 'p-val =\n {:.2f}\n'.format(pval[i_x])                        
                plt.text(text_xvals[i_x], text_yvals[i_x], text)

#############################################

def plot_absolute_fractional_changes(df_full, alpha):
    '''
    Plot absolute fractional changes.
    
    Parameters
    ----------
    df_full : Pandas DataFrame
        Dataframe containing |fractional df/f| changes and associated
        p-values and standard deviations
    alpha : number
        Alpha is the p-value threshold, values below which 
        are considered statistically significant.

    Returns
    -------
    fig : Matplotlib figure handle
        Handle to the full layer/compartment figure with subplots
    '''
    
    # declarations/initializations
    plot_rows = 3
    plot_cols = 2
    fig, ax = plt.subplots(plot_rows, plot_cols, 
                           figsize=(plot_cols*2.5, plot_cols*6+1),
                           constrained_layout=True)
    # Plot rows / compartments
    for plot_row, compartment in enumerate(df_full['compartment'].unique()):
        # Plot columns / layers
        for plot_col, layer in enumerate(df_full['layer'].unique()[:-1]):
            if compartment == 'all':
                if plot_col==1:
                    ax[plot_row, plot_col].axis('off')
                    continue
                layer = 'all'
            mask0 = df_full['layer']==layer
            mask1 = df_full['compartment']==compartment
            df = df_full[mask0 & mask1]
            gab_frac = df['gab_frac_changes'].values[0]
            brk_frac = df['brk_frac_changes'].values[0]
            gab_std = df['gab_bstrap_std'].values[0]
            brk_std = df['brk_bstrap_std'].values[0]
            sess_compare = df['sess_compare'].values[0]
            pval = df['pval'].values[0]
            make_absolute_fractional_subplot(gab_frac, brk_frac, gab_std, 
                                             brk_std, pval, alpha, 
                                             layer, compartment, sess_compare, 
                                             ax[plot_row, plot_col])
    sess_compare_str = 'All session pairs' if sess_compare[0]=='all' else \
        'Session {} v {}'.format(sess_compare[0], sess_compare[1])
    plt.suptitle('|df/f fractional changes| \nfor unexpected events\n{}\n'.
                 format(sess_compare_str), fontsize=20)
    return fig

#############################################

def make_absolute_fractional_subplot(gab_frac, brk_frac, gab_frac_std, 
                                     brk_frac_std, pval, alpha, 
                                     layer, compartment, sess_compare, ax):

    '''
    Plot absolute fractional changes.
    
    Parameters
    ----------
    gab_frac : 1-d array of numbers
        Array of absolute fractional changes for Gabors.
    brk_frac : 1-d array of numbers
        Array of absolute fractional changes for visual flow.
    gab_frac_std : number
        Standard deviation of Gabor |fractional df/f| changes.
    brk_frac_std : number
        Standard deviation of brick |fractional df/f| changes.
    pval : number
        P-value of absolute fractional change.
    alpha : number
        Alpha is the p-value threshold, values below which 
        are considered statistically significant.
    layer : string
        Layer ('L2/3', 'L5', 'all') for which to obtain data.    
    compartment : string
        Compartment ('dend', 'soma') for which to obtain data.
    sess_compare : 1-D arraylike of numbers
        Array of sessions to compare (e.g., [1,3], ['all'])
    ax : Matplotlib axis handle
        Axis in which to make subplot
    '''
    
    # declarations/initializations
    colors = ['slateblue','firebrick']
    compartment_str_dict = {'dend':'dendrites', 'soma':'somata'}

    plt.sca(ax)
    plt.bar(['Gabors', 'Bricks'], [np.mean(gab_frac), np.mean(brk_frac)], 
            color=colors, width=0.5, alpha=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    err_ylows  = [max(np.mean(gab_frac)-gab_frac_std, 0), 
                  max(np.mean(brk_frac)-brk_frac_std, 0)]
    err_yhighs = \
        [np.mean(gab_frac)+gab_frac_std, np.mean(brk_frac)+brk_frac_std]
    plt.vlines(['Gabors', 'Bricks'], err_ylows, err_yhighs, color='black')

    # For all layers/compartments
    if layer=='all':
        gab_arr = np.repeat(['Gabors'], len(gab_frac))
        brk_arr = np.repeat(['Bricks'], len(brk_frac))
        x0 = np.concatenate((gab_arr, brk_arr))
        y0 = np.concatenate((gab_frac, brk_frac))
        sns.set_palette('gray')
        # sns.set_palette(colors)
        # sns.violinplot(x=x0, y=y0, inner=None, alpha=0.6);
        sns.stripplot(x=x0, y=y0, jitter=0.2, size=10, alpha=0.6);
    
    # Significant p-values
    if pval <= alpha:
        ymin = plt.ylim()[0]
        ymax = plt.ylim()[1]
        yrange = ymax-ymin
        y_hline = ymax+0.05*yrange
        y_text = ymax+0.1*yrange
        plt.hlines(y_hline, 0, 1)
        plt.text(0.5, y_text, '*', fontsize=15)
    
    compartment_compare_str = 'All compartments' if layer=='all' else \
        '{} {}'.format(layer, compartment_str_dict[compartment])
    plt.title('{}\np-value = {:.5f}\n'.
              format(compartment_compare_str,pval), fontsize=15)

#############################################