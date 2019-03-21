'''
plot_util.py

This module contains basic functions for plotting with pyplot for data 
generated by the AIBS experiments for the Credit Assignment Project

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 2.7.

'''
import os

import numpy as np
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt

import gen_util, str_util


#############################################
def linclab_plt_defaults(font='Liberation Sans', font_dir=None, print_fonts=False, 
                         example=False, dir=''):
    """
    linclab_plt_defaults()

    Sets pyplot defaults to Linclab style.

    Optional arguments:
        - font (str or list): font to use, or list in order of preference
                              default: 'Liberation Sans'
        - font_dir (str)    : directory to where extra fonts (.ttf) are stored
                              default: None
        - print_fonts (bool): if True, an alphabetical list of available fonts 
                              is printed
                              default: False
        - example (bool)    : if True, an example plot is created and saved
                              default: False
        - dir (str)         : directory in which to save example if example is True 
                              default: ''
    """

    colors = ['#50a2d5', # Linclab blue
              '#eb3920', # Linclab red
              '#969696', # Linclab grey
              '#76bb4b', # Linclab green
              '#9370db', # purple
              '#ff8c00', # orange
              '#bb4b76', # pink
              '#e0b424', # yellow
              '#b04900', # brown
              ] 
    col_cyc = plt.cycler(color=colors)

    # set pyplot params
    params = {'axes.labelsize'       : 'x-large',  # large axis labels
              'axes.linewidth'       : 1.5,        # thicker axis lines
              'axes.prop_cycle'      : col_cyc,    # line color cycle
              'axes.spines.right'    : False,      # no axis spine on right
              'axes.spines.top'      : False,      # no axis spine at top
              'axes.titlesize'       : 'x-large',  # x-large axis title
              'errorbar.capsize'     : 8,          # errorbar cap length
              'figure.titlesize'     : 'x-large',  # x-large figure title
              'legend.fontsize'      : 'large',    # large legend text
              'lines.dashed_pattern' : [8.0, 4.0], # longer dashes
              'lines.linewidth'      : 2.5,        # thicker lines
              'lines.markeredgewidth': 2.5,        # thick marker edge widths 
                                                   # (e.g., cap thickness) 
              'lines.markersize'     : 10,         # bigger markers
              'patch.linewidth'      : 2.5,        # thicker lines for patches
              'savefig.format'       : 'svg',      # figure save format
              'savefig.bbox'         : 'tight',    # tight cropping of figure
              'xtick.labelsize'      : 'large',    # large x-tick labels
              'xtick.major.size'     : 8.0,        # longer x-ticks
              'xtick.major.width'    : 2.0,        # thicker x-ticks
              'ytick.labelsize'      : 'large',    # large y-tick labels
              'ytick.major.size'     : 8.0,        # longer y-ticks
              'ytick.major.width'    : 2.0,        # thicker y-ticks
              }

    # add new fonts to list if a font directory is provided
    if font_dir is not None and os.path.exists(font_dir):
        font_dirs = [font_dir, ]
        font_files = fm.findSystemFonts(fontpaths=font_dirs)
        font_list = fm.createFontList(font_files)
        fm.fontManager.ttflist.extend(font_list)
    
    # list of available fonts
    all_fonts = list(set([f.name for f in fm.fontManager.ttflist]))

    # print list of fonts, if requested
    if print_fonts:
        print('Available fonts:')
        sorted_fonts = sorted(all_fonts)
        for font in sorted_fonts:
            print('    {}'.format(font))
    
    # check whether requested font is available, otherwise warn that
    # default will be used.
    font = gen_util.list_if_not(font)
    set_font = True
    f = 0
    while set_font and f < len(font):
        if font[f] in all_fonts:
            params['font.family'] = font
            set_font = False
        elif f == len(font) - 1:
            font_fam = plt.rcParams['font.family'][0]
            def_font = plt.rcParams['font.{}'.format(font_fam)][0]
            print('Warning: Desired font ({}) not found, so default ({}) will be '
                'used instead.\n'.format(font, def_font))
        f = f+1

    # update pyplot parameters
    plt.rcParams.update(params)

    # create and save an example plot, if requested
    if example:
        fig, ax = plt.subplots()
        
        n_col = len(colors)
        x = np.asarray(range(10))[:, np.newaxis]
        y = np.repeat(x/2.0, n_col, axis=1) - \
            np.asarray(range(-n_col, 0))[np.newaxis, :]
        ax.plot(x, y)
        ax.legend(colors)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_title('Example plot')
        ax.axvline(x=1, ls='dashed', c='k')
        
        fig.savefig('example_plot')


#############################################
def plot_seg_comp(analys_par, plot_vals='diff', op='diff'):
    """
    plot_seg_comp(analys_par)

    Creates lists with different components needed when plotting segments, 
    namely positions of labels, ordered labels, positions of heavy bars and 
    position of regular bars.

    Arguments:
        - analys_par (dict): dictionary containing relevant parameters
                             to what order the gabor frames will be plotted in
                ['gab_fr'] (int or list): gabor frame values to include
                                         (e.g., 0, 1, 2, 3)
                ['pre'] (float)         : range of frames to include before each 
                                          frame reference (in s)
                ['post'] (float)        : range of frames to include after each 
                                          frame reference (in s)

    Optional arguments:
        - plot_vals (str): 'surp', 'nosurp' or 'diff'
                           default: 'diff'
        - op (str)       : 'surp', 'nosurp' or 'diff'
                           default: 'diff'
    
    Returns:
        - xpos (list)          : list of x coordinates at which to add labels
                                 (same length as labels)
        - labels (list)         : ordered list of labels for gabor frames
        - hbars (list or float): list of x coordinates at which to add 
                                 heavy dashed vertical bars
                                 default: None
        - bars (list or float) : list of x coordinates at which to add 
                                 dashed vertical bars
                                 default: None
    """

    if analys_par['pre'] == 0 and analys_par['post'] == 1.5:
        xpos = [0.15, 0.45, 0.75, 1.05, 1.35]
        labels = plot_val_lab(plot_vals, op, analys_par['gab_fr'])
        bars = [0.3, 0.6, 0.9, 1.2]
        if analys_par['gab_fr'] == 3:
            hbars=None
        else:
            hbars = np.round(1.2 - 0.3*(analys_par['gab_fr']+1), 1)
            bars = gen_util.remove_if(bars, hbars)

    else:
        raise NotImplementedError('Figure parameters for x values and seg bars '
                                  'having only been implemented for pre=0, '
                                  'post=1.5.')
    return xpos, labels, hbars, bars


#############################################
def set_ticks(ax, axis='x', min_tick=0, max_tick=1.5, n=6, pad_p=0.05):
    """
    set_ticks(ax)

    Creates a list of labels for gabor frames based on values that are plotted,
    and operation on surprise v no surprise, starting with gray.

    Required arguments:
        - ax (plt Axis subplot): subplot

    Optional arguments:
        - axis (str)      : axis for which to set ticks, i.e., x, y or both
                            default: 'x'
        - min_tick (float): first tick value
                            default: 0
        - max_tick (float): last tick value
                            default: 1.5
        - n (int)         : number of ticks
                            default: 6
        - pad_p (float)   : percentage to pad axis length
    """

    pad = (max_tick - min_tick) * pad_p
    min_end = min_tick - pad
    max_end = max_tick + pad
    
    if axis == 'both':
        axis = ['x', 'y']
    elif axis in ['x', 'y']:
        axis = gen_util.list_if_not(axis)
    else:
        gen_util.accepted_values_error('axis', axis, ['x', 'y', 'both'])

    if 'x' in axis:
        ax.set_xlim(min_end, max_end)
        ax.set_xticks(np.linspace(min_tick, max_tick, n))
    elif 'y' in axis:
        ax.set_ylim(min_end, max_end)
        ax.set_yticks(np.linspace(min_tick, max_tick, n))


#############################################
def plot_val_lab(plot_vals='diff', op='diff', start_fr=-1):
    """
    plot_val_lab()

    Creates a list of labels for gabor frames based on values that are plotted,
    and operation on surprise v no surprise, starting with gray.

    Optional arguments:
        - plot_vals (str): 'surp', 'nosurp', 'reg' or 'diff'
                           default: 'diff'
        - op (str)       : 'surp', 'nosurp', 'reg' or 'diff'
                           default: 'diff'
        - start_fr (int) : starting gabor frame 
                           (-1: gray, 0: A, 1: B, 2:C, 3:D/E)
                           default: -1
    
    Returns:
        - labels (list)  : list of labels for gabor frames
    """
    labels = ['gray', 'A', 'B', 'C']

    if plot_vals == 'surp':
        labels.extend(['E'])
    elif plot_vals in ['nosurp', 'reg']:
        labels.extend(['D'])
    elif plot_vals == 'diff':
        if op == 'diff':
            labels.extend(['E-D'])      
        elif op == 'ratio':
            labels.extend(['E/D'])
        else:
            gen_util.accepted_values_error('op', op, ['diff', 'ratio'])
    else:
        gen_util.accepted_values_error('plot_vals', plot_vals, 
                                       ['diff', 'reg', 'surp', 'nosurp'])

    if start_fr != -1:
        labels = list(np.roll(labels, -(start_fr+1)))

    return labels


#############################################
def get_subax(ax, i):
    """
    get_subax(ax, i)

    Returns the correct sub_ax based on a 1D index. Indexing is by column, then 
    row.

    Required arguments:
        - ax (plt Axis): axis
        - i (int)      : 1D subaxis index

    Return:
        - sub_ax (plt Axis subplot): subplot
    """
    if len(ax.shape) == 1:
        n = ax.shape[0]
        sub_ax = ax[i%n]
    else:
        ncols = ax.shape[1]
        sub_ax = ax[i/ncols][i%ncols]

    return sub_ax


#############################################
def init_fig(n_subplots, fig_par, div=1.0):
    """
    init_fig(n_subplots, fig_par)

    Creates a figure with the correct number of rows and columns for the 
    number of subplots, following the figure parameters

    Required arguments:
        - n_subplots (int): number of subplots to accomodate in the figure
        - fig_par (dict)  : dictionary containing figure parameters:
                ['ncols'] (int)        : number of columns in the figure
                ['sharey'] (bool)      : if True, y axis lims are shared across 
                                         subplots
                ['subplot_wid'] (float): width of each subplot (inches)
                ['subplot_hei'] (float): height of each subplot (inches)
        - div (float)     : value by which to divide subplot_wid
                            default: 1.0

    Return:
        - fig (plt Fig): fig
        - ax (plt Axis): axis (even if just one subplot)
    """

    if n_subplots == 1:
        ncols = 1
    elif n_subplots < fig_par['ncols']:
        ncols = n_subplots
    else:
        ncols = fig_par['ncols']
    nrows = int(np.ceil(n_subplots/float(fig_par['ncols'])))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, 
                           figsize=(ncols*fig_par['subplot_wid']/div, 
                                    nrows*fig_par['subplot_hei']), 
                           sharey=fig_par['sharey'], squeeze=False)
    return fig, ax


#############################################
def save_fig(fig, save_dir, save_name, fig_par, print_dir=True):
    """
    save_fig(fig, save_dir, save_name, fig_par)

    Saves a figure under a specific directory and name, following figure
    parameters and returns final directory name.

    Required arguments:
        - fig (plt Fig)  : figure
        - save_dir (str) : directory in which to save figure
        - save_name (str): name under which to save figure (WITHOUT extension)
        - fig_par (dict) : dictionary containing figure parameters:
                ['bbox'] (str)      : bbox parameter for plt.savefig(), 
                                      e.g., 'tight'
                ['datetime'] (bool) : if True, figures are saved in a subfolder
                                      named based on the date and time.
                ['fig_ext'] (str)   : extension (without '.') with which to save
                                      figure
                ['mult'] (bool)     : if True, prev_dt is created or used.
                ['overwrite'] (bool): if False, overwriting existing figures is 
                                      prevented by adding suffix numbers.
                ['prev_dt'] (str)   : datetime folder to use
    
    Optional arguments:
        - print_dir (bool): if True, the save directory is printed 
                            default: True
    
    Returns:
        - save_dir (str): final name of the directory in which the figure is 
                          saved 
                          (may be different from input save_dir, as a datetime 
                          subfolder, or a suffix to prevent overwriting may have 
                          been added depending on the parameters in fig_par.)
    """

    # add subfolder with date and time
    if fig_par['datetime']:
        if fig_par['mult'] and fig_par['prev_dt'] is not None:
            save_dir = os.path.join(save_dir, fig_par['prev_dt'])
        else:
            datetime = str_util.create_time_str()
            save_dir = os.path.join(save_dir, datetime)
            fig_par['prev_dt'] = datetime
    
    # create directory if doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    full_name = '{}.{}'.format(save_name, fig_par['fig_ext'])

    # check if file aready exists, and if so, add number at end
    if not fig_par['overwrite']:
        if os.path.exists(os.path.join(save_dir, full_name)):     
            count = 1
            full_name = '{}_{}.{}'.format(save_name, count, fig_par['fig_ext']) 
            while os.path.exists(os.path.join(save_dir, full_name)):
                count += 1 
                full_name = '{}_{}.{}'.format(save_name, count, 
                                              fig_par['fig_ext'])

    if print_dir:
        print('\nFigures saved under {}.'.format(save_dir))

    fig.savefig(os.path.join(save_dir, full_name), bbox=fig_par['bbox'])
    
    return save_dir


#############################################
def add_labels(ax, labels, xpos, t_hei=0.9, col='k'):
    """
    add_labels(ax, labels, xpos)

    Adds labels to a subplot.

    Required arguments:
        - ax (plt Axis subplot): subplot
        - labels (list or str) : list of labels to add to axis
        - xpos (list or float) : list of x coordinates at which to add labels
                                 (same length as labels)
      

    Optional arguments:
        - t_hei (float): relative height between 0 and 1 at which to place 
                         labels, with respect to y limits. 
                         default: 0.9
        - col (str)    : color to use
                         default: 'k'
    """


    labels = gen_util.list_if_not(labels)
    xpos = gen_util.list_if_not(xpos)

    if len(labels) != len(xpos):
        raise IOError(('Arguments \'labels\' and \'xpos\' must be of '
                        'the same length.'))

    if t_hei > 1.0 or t_hei < 0.0:
        raise IOError('Must pass a t_hei between 0.0 and 1.0.')
    ymin, ymax = ax.get_ylim()
    ypos = (ymax-ymin)*t_hei+ymin
    for l, x in zip(labels, xpos):
        ax.text(x, ypos, l, ha='center', fontsize=15, color=col)


#############################################
def add_bars(ax, hbars=None, bars=None, col='k'):
    """
    add_bars(ax)

    Adds dashed vertical bars to a subplot.

    Required arguments:
        - ax (plt Axis subplot): subplot

    Optional arguments:
        - hbars (list or float): list of x coordinates at which to add 
                                 heavy dashed vertical bars
                                 default: None
        - bars (list or float) : list of x coordinates at which to add 
                                 dashed vertical bars
                                 default: None
        - col (str)            : color to use
                                 default: 'k'
    """

    torem = []
    if hbars is not None:
        hbars = gen_util.list_if_not(hbars)
        torem = hbars
        for b in hbars:
            ax.axvline(x=b, ls='dashed', c='k', lw=2, alpha=0.5)
    if bars is not None:
        bars = gen_util.remove_if(bars, torem)
        for b in bars:
            ax.axvline(x=b, ls='dashed', c='k', lw=1, alpha=0.5)


#############################################
def incr_ymax(ax, incr=1.1, sharey=False):
    """
    incr_ymax(ax)

    Increase heights of axis subplots.

    Required arguments:
        - ax (plt Axis object): axis (not a subplot)

    Optional arguments:
        - incr (float) : relative amount to increase subplot height
        - sharey (bool): if True, only the first subplot ymax is modified, as it 
                         will affect all. Otherwise, all subplot ymax are. 
                         default: False
    """

    if incr is not None:
        if sharey:
            change_ax = [get_subax(ax, 0)]
        else:
            n_ax = np.prod(ax.shape)
            change_ax = [get_subax(ax, i) for i in range(n_ax)]
        for sub_ax in change_ax:
            ymin, ymax = sub_ax.get_ylim()
            ymax = (ymax-ymin)*incr + ymin
            sub_ax.set_ylim(ymin, ymax) 


#############################################
def add_plot_comp(ax, title='', xticks=None, yticks=None, fluor='dff', dff=None):

    ax.set_title(title)

    if dff is not None:
        if dff:
            fluor = 'dff'
        else:
            fluor = 'raw'
    
    if fluor is not None:
        fluor_str = str_util.fluor_par_str(fluor, type_str='print')
        ax.set_ylabel(fluor_str)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)


#############################################
def plot_traces(ax, chunk_val, stats='mean', error='std', title='', lw=2.5, 
                col=None, alpha=0.5, plot_err=True, xticks=None, yticks=None,
                label=None, fluor='dff', dff=None):
    """
    plot_traces(ax, chunk_val)

    Plot traces (mean/median with shaded error bars) on axis (ax).

    Required arguments:
        - ax (plt Axis subplot): subplot
        - chunk_val (2D array) : array of chunk statistics, where the first
                                 dimension corresponds to the statistics 
                                 (x_ran [0], mean/median [1], deviation [2] 
                                 or [2:3] if quartiles)
    Optional arguments:
        - stats (str)          : statistic parameter, i.e. 'mean' or 'median'
                                 default: 'mean'
        - error (str)          : error statistic parameter, i.e. 'std' or 'sem'
                                 default: 'std'
        - title (str)          : axis title
                                 default: ''
        - lw (float)           : plt line weight variable
                                 default: 2.5
        - col (str)            : color to use
                                 default: None
        - alpha (float)        : plt alpha variable controlling shading 
                                 transparency (from 0 to 1)
                                 default: 0.5
        - plot_err (bool)      : if True, error is plotted
                                 default: True
        - xticks (str)         : xtick labels
                                 default: None
        - yticks (str)         : ytick labels
                                 default: None
        - label (str)          : label for legend
        - fluor (str)          : if 'raw', plot is labeled as raw fluorescence. 
                                 if 'dff, plot is labels with 'dF/F'.
                                 if None, no y label is added.
                                 default: 'dff'
        - dff (bool)           : can be used instead of fluor, and if so
                                 (not None), will supercede fluor. 
                                 if True, fluor is set to 'dff', if False, 
                                 fluor is set to 'raw'. If None, no effect.
                                 default: None  
    """
    
    ax.plot(chunk_val[0], chunk_val[1], lw=lw, color=col, label=label)
    col = ax.lines[-1].get_color()
    if plot_err:
        # only condition where pos and neg error are different
        if stats == 'median' and error == 'std': 
            ax.fill_between(chunk_val[0], chunk_val[2], chunk_val[3], 
                            facecolor=col, alpha=alpha)
        else:
            ax.fill_between(chunk_val[0], chunk_val[1] - chunk_val[2], 
                            chunk_val[1] + chunk_val[2], 
                            facecolor=col, alpha=alpha)

    # add x ticks
    if xticks is None:
        min_val = np.min(chunk_val[0])
        max_val = np.around(np.max(chunk_val[0]), 2)
        if min_val == 0:
            if max_val == 1.5:
                ax.set_xticks(np.linspace(min_val, max_val, 6))
            elif max_val == 0.45:
                ax.set_xticks(np.linspace(min_val, max_val, 4))
    
    ax.set_xlabel('Time (s)')
    if label is not None:
        ax.legend()

    add_plot_comp(ax, title, xticks, yticks, fluor, dff)


#############################################
def plot_bars(ax, x, y, err=None, title='', width=0.75, lw=2.5, col=None,  
              alpha=0.5, xticks=None, yticks=None, xlims=None, label=None, 
              hline=None, capsize=8, fluor='dff', dff=None):
    """
    plot_traces(ax, chunk_val)

    Plot traces (mean/median with shaded error bars) on axis (ax).

    Required arguments:
        - ax (plt Axis subplot): subplot
        - x (list) : list of x values

    Optional arguments:
        - stats (str)          : statistic parameter, i.e. 'mean' or 'median'
                                 default: 'mean'
        - error (str)          : error statistic parameter, i.e. 'std' or 'sem'
                                 default: 'std'
        - title (str)          : axis title
                                 default: ''
        - lw (float)           : plt line weight variable
                                 default: 2.5
        - col (str)            : color to use
                                 default: None
        - alpha (float)        : plt alpha variable controlling shading 
                                 transparency (from 0 to 1)
                                 default: 0.5
        - plot_err (bool)      : if True, error is plotted
                                 default: True
        - xticks (str)         : xtick labels ('None' to omit ticks entirely)
                                 default: None
        - yticks (str)         : ytick labels
                                 default: None
        - xlims (list)         : xlims
                                 default: None
        - label (str)          : label for legend
                                 default: None
        - hline (list or float): list of y coordinates at which to add 
                                 horizontal bars
                                 default: None
        - capsize (float)      : length of errorbar caps
                                 default: 8
        - fluor (str)          : if 'raw', plot is labeled as raw fluorescence. 
                                 if 'dff, plot is labels with 'dF/F'.
                                 if None, no y label is added.
                                 default: 'dff'
        - dff (bool)           : can be used instead of fluor, and if so
                                 (not None), will supercede fluor. 
                                 if True, fluor is set to 'dff', if False, 
                                 fluor is set to 'raw'. If None, no effect.
                                 default: None  
    """
    
    patches = ax.bar(x, y, width=width, lw=lw, label=label, color=col)
    
    # get color
    fc = patches[0].get_fc()


    # add errorbars
    if err is not None:
        ax.errorbar(x, y, np.squeeze(err), fmt='None', elinewidth=lw, 
                    capsize=capsize, capthick=lw, ecolor=fc)

    # set edge color to match patch face color
    [patch.set_ec(fc) for patch in patches]

    # set face color to transparency
    [patch.set_fc(list(fc[0:3])+[alpha]) for patch in patches]

    if label is not None:
        ax.legend()

    if xlims is not None:
        ax.set_xlim(xlims)
    
    if hline is not None:
        ax.axhline(y=hline, c='k', lw=1.5)
    
    if xticks in ['None', 'none']:
        ax.tick_params(axis='x', which='both', bottom=False) 
        xticks = None
    
    ax.set_title(title)

    add_plot_comp(ax, title, xticks, yticks, fluor, dff)

