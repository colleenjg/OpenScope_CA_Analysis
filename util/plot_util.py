"""
plot_util.py

This module contains basic functions for plotting with pyplot.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import os

import matplotlib as mpl
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from util import file_util, gen_util


#############################################
def linclab_plt_defaults(font='Liberation Sans', fontdir=None, 
                         print_fonts=False, example=False, dirname='.'):
    """
    linclab_plt_defaults()

    Sets pyplot defaults to Linclab style.

    Optional args:
        - font (str or list): font to use, or list in order of preference
                              default: 'Liberation Sans'
        - fontdir (str)     : directory to where extra fonts (.ttf) are stored
                              default: None
        - print_fonts (bool): if True, an alphabetical list of available fonts 
                              is printed
                              default: False
        - example (bool)    : if True, an example plot is created and saved
                              default: False
        - dirname (str)     : directory in which to save example if example is True 
                              default: '.'
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
    if fontdir and os.path.exists(fontdir):
        fontdirs = [fontdir, ]
        font_files = fm.findSystemFonts(fontpaths=fontdirs)
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
            print('Warning: Desired font ({}) not found, so default ({}) will '
                  'be used instead.\n'.format(font, def_font))
        f = f+1

    # update pyplot parameters
    plt.rcParams.update(params)

    # create and save an example plot, if requested
    if example:
        fig, ax = plt.subplots()
        
        n_col = len(colors)
        x = np.asarray(list(range(10)))[:, np.newaxis]
        y = np.repeat(x/2., n_col, axis=1) - \
            np.asarray(list(range(-n_col, 0)))[np.newaxis, :]
        ax.plot(x, y)
        ax.legend(colors)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_title('Example plot')
        ax.axvline(x=1, ls='dashed', c='k')
        
        fig.savefig(os.path.join(dirname, 'example_plot'))


#############################################
def linclab_colormap(nbins=100):
    """
    linclab_colormap()

    Returns a matplotlib colorplot using the linclab blue, white and linclab 
    red.

    Optional args:
        - nbins (int): number of bins to use to create colormap
                       default: 100

    Returns:
        - cmap (colormap): a matplotlib colormap
    """

    colors = ["#50a2d5", "#ffffff", "#eb3920"]

    # convert to RGB
    rgb_col = [[] for _ in range(len(colors))]
    for c, col in enumerate(colors):
        ch_vals = mpl.colors.to_rgb(col)
        for ch_val in ch_vals:
            rgb_col[c].append(ch_val)

    cmap = mpl.colors.LinearSegmentedColormap.from_list('linclab_byr', rgb_col, 
                                                        N=nbins)

    return cmap


#############################################
def manage_mpl(plt_bkend=None, linclab=True, fontdir=None, cmap=False, 
               nbins=100):
    """
    manage_mpl()

    Makes changes to the matplotlib backend used as well as matplotlib plotting
    defaults. If cmap is True, a colormap is returned.

    Optional args:
        - plt_bkend (str): matplotlib backend to use
                           default: None
        - linclab (bool) : if True, the Linclab default are set
                           default: True
        - fontdir (str ) : directory to where extra fonts (.ttf) are stored
                           default: None
        - cmap (bool)    : if True, a colormap is returned. If linclab is True,
                           the Linclab colormap is returned, otherwise the 
                           'jet' colormap
                           default: False
        - nbins (int)    : number of bins to use to create colormap
                           default: 100

    Returns:
        if cmap:
        - cmap (colormap): a matplotlib colormap
    """

    if plt_bkend is not None:
        plt.switch_backend(plt_bkend)
    
    if linclab:
        linclab_plt_defaults(font=['Arial', 'Liberation Sans'], 
                             fontdir=fontdir)

    if cmap:
        if linclab:
            cmap = linclab_colormap(nbins)
        else:
            cmap = 'jet'
        return cmap


#############################################
def set_ticks(sub_ax, axis='x', min_tick=0, max_tick=1.5, n=6, pad_p=0.05):
    """
    set_ticks(sub_ax)

    Sets ticks on specified axis and axis limits around ticks using specified 
    padding. 

    Required args:
        - sub_ax (plt Axis subplot): subplot

    Optional args:
        - axis (str)    : axis for which to set ticks, i.e., x, y or both
                          default: 'x'
        - min_tick (num): first tick value
                          default: 0
        - max_tick (num): last tick value
                          default: 1.5
        - n (int)       : number of ticks
                          default: 6
        - pad_p (num)   : percentage to pad axis length
                          default: 0.05
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
        sub_ax.set_xlim(min_end, max_end)
        sub_ax.set_xticks(np.linspace(min_tick, max_tick, n))
    elif 'y' in axis:
        sub_ax.set_ylim(min_end, max_end)
        sub_ax.set_yticks(np.linspace(min_tick, max_tick, n))


#############################################
def get_subax(ax, i):
    """
    get_subax(ax, i)

    Returns the correct sub_ax from a 1D or 2D axis array based on a 1D index. 
    Indexing is by column, then row.

    Required args:
        - ax (plt Axis): axis
        - i (int)      : 1D subaxis index

    Returns:
        - sub_ax (plt Axis subplot): subplot
    """

    if len(ax.shape) == 1:
        n = ax.shape[0]
        sub_ax = ax[i%n]
    else:
        ncols = ax.shape[1]
        sub_ax = ax[i//ncols][i%ncols]

    return sub_ax


#############################################
def share_lims(ax, dim='row'):
    """
    share_lims(ax)

    Adjusts limits within rows or columns for a 2D axis array. 

    Required args:
        - ax (plt Axis): axis (2D array)

    Optional args:
        - dim (str): which dimension to match limits along
                     default: 'row'
    """

    if len(ax.shape) != 2:
        raise NotImplementedError(('Function only implemented for 2D axis '
                                   'arrays.'))
    
    if dim == 'row':
        for r in range(ax.shape[0]):
            ylims = [np.inf, -np.inf]
            for task in ['get', 'set']:    
                for c in range(ax.shape[1]):
                    if task == 'get':
                        lim = ax[r, c].get_ylim()
                        if lim[0] < ylims[0]:
                            ylims[0] = lim[0]
                        if lim[1] > ylims[1]:
                            ylims[1] = lim[1]
                    elif task == 'set':
                        ax[r, c].set_ylim(ylims)
    
    if dim == 'col':
        for r in range(ax.shape[1]):
            xlims = [np.inf, -np.inf]
            for task in ['get', 'set']:   
                for c in range(ax.shape[0]):
                    if task == 'get':
                        lim = ax[r, c].get_xlim()
                        if lim[0] < xlims[0]:
                            xlims[0] = lim[0]
                        if lim[1] > xlims[1]:
                            xlims[1] = lim[1]
                    elif task == 'set':
                        ax[r, c].set_xlim(xlims)


#############################################
def set_axis_digits(sub_ax, xaxis=None, yaxis=None):

    if xaxis is not None:
        n_dig_str = '%.{}f'.format(int(xaxis))
        sub_ax.xaxis.set_major_formatter(FormatStrFormatter(n_dig_str))

    if yaxis is not None:
        n_dig_str = '%.{}f'.format(int(yaxis))
        sub_ax.yaxis.set_major_formatter(FormatStrFormatter(n_dig_str))


#############################################
def init_fig(n_subplots, ncols=3, sharex=False, sharey=True, subplot_hei=7.5, 
             subplot_wid=7.5):
    """
    init_fig(n_subplots, fig_par)

    Returns a figure and axes with the correct number of rows and columns for 
    the number of subplots, following the figure parameters.

    Required args:
        - n_subplots (int) : number of subplots to accomodate in the figure
        
    Optional args:
        - ncols (int)      : number of columns in the figure
                             default: 3
        - sharex (bool)    : if True, x axis lims are shared across subplots
                             default: False
        - sharey (bool)    : if True, y axis lims are shared across subplots
                             default: True
        - subplot_hei (num): height of each subplot (inches)
                             default: 7.5
        - subplot_wid (num): width of each subplot (inches)
                             default: 7.5

    Returns:
        - fig (plt Fig): fig
        - ax (plt Axis): axis (even if for just one subplot)
    """

    if n_subplots == 1:
        ncols = 1
    elif n_subplots < ncols:
        ncols = n_subplots

    nrows = int(np.ceil(n_subplots/float(ncols)))
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, 
                           figsize=(ncols*subplot_wid, nrows*subplot_hei), 
                           sharex=sharex, sharey=sharey, squeeze=False)
    return fig, ax


#############################################
def savefig(fig, savename, fulldir='.', datetime=True, use_dt=None, 
            fig_ext='svg', overwrite=False, print_dir=True):
    """
    savefig(fig, savename)

    Saves a figure under a specific directory and name, following figure
    parameters and returns final directory name.

    Required args:
        - fig (plt Fig) : figure (if None, no figure is saved, but fulldir is 
                          created and name is returned)
        - savename (str): name under which to save figure
    
    Optional args:
        - fulldir (str)   : directory in which to save figure
                            default: '.'
        - datetime (bool) : if True, figures are saved in a subfolder named 
                            based on the date and time.
                            default: True
        - use_dt (str)    : datetime folder to use
                            default: None
        - fig_ext (str)   : figure extension
                            default: 'svg'
        - overwrite (bool): if False, overwriting existing figures is prevented 
                            by adding suffix numbers.
                            default: False        
        - print_dir (bool): if True, the save directory is printed 
                            default: True
    
    Returns:
        - fulldir (str): final name of the directory in which the figure is 
                         saved (may differ from input fulldir, if datetime 
                         subfolder is added.)
    """

    # add subfolder with date and time
    if datetime:
        if use_dt is not None:
            fulldir = os.path.join(fulldir, use_dt)
        else:
            datetime = gen_util.create_time_str()
            fulldir = os.path.join(fulldir, datetime)

    # create directory if doesn't exist
    file_util.createdir(fulldir, print_dir=False)

    if fig is not None:
        # get extension and savename
        fullname, ext = file_util.add_ext(savename, fig_ext) 

        # check if file aready exists, and if so, add number at end
        if not overwrite:
            if os.path.exists(os.path.join(fulldir, fullname)):     
                savename, _ = os.path.splitext(fullname) # get only savename
                count = 1
                fullname = '{}_{}{}'.format(savename, count, ext) 
                while os.path.exists(os.path.join(fulldir, fullname)):
                    count += 1 
                    fullname = '{}_{}{}'.format(savename, count, ext)
        
        if print_dir:
            print('\nFigures saved under {}.'.format(fulldir))

        fig.savefig(os.path.join(fulldir, fullname))

    return fulldir


#############################################
def add_labels(sub_ax, labels, xpos, t_hei=0.9, col='k'):
    """
    add_labels(sub_ax, labels, xpos)

    Adds labels to a subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - labels (list or str)     : list of labels to add to axis
        - xpos (list or num)       : list of x coordinates at which to add 
                                     labels (same length as labels)
      
    Optional args:
        - t_hei (num): height relative to y limits at which to place labels. 
                       default: 0.9
        - col (str)  : color to use
                       default: 'k'
    """

    labels = gen_util.list_if_not(labels)
    xpos = gen_util.list_if_not(xpos)

    if len(labels) != len(xpos):
        raise ValueError(('Arguments \'labels\' and \'xpos\' must be of '
                          'the same length.'))

    ymin, ymax = sub_ax.get_ylim()
    ypos = (ymax-ymin)*t_hei+ymin
    for l, x in zip(labels, xpos):
        sub_ax.text(x, ypos, l, ha='center', fontsize=15, color=col)


#############################################
def add_bars(sub_ax, hbars=None, bars=None, col='k'):
    """
    add_bars(sub_ax)

    Adds dashed vertical bars to a subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot

    Optional args:
        - hbars (list or num): list of x coordinates at which to add 
                               heavy dashed vertical bars
                               default: None
        - bars (list or num) : list of x coordinates at which to add 
                               dashed vertical bars
                               default: None
        - col (str)          : color to use
                               default: 'k'
    """

    torem = []
    if hbars is not None:
        hbars = gen_util.list_if_not(hbars)
        torem = hbars
        for b in hbars:
            sub_ax.axvline(x=b, ls='dashed', c='k', lw=2.5, alpha=0.5)
    if bars is not None:
        bars = gen_util.remove_if(bars, torem)
        for b in bars:
            sub_ax.axvline(x=b, ls='dashed', c='k', lw=1.5, alpha=0.5)


#############################################
def incr_ymax(ax, incr=1.1, sharey=False):
    """
    incr_ymax(ax)

    Increases heights of axis subplots.

    Required args:
        - ax (plt Axis): axis

    Optional args:
        - incr (num)   : relative amount to increase subplot height
                         default: 1.1
        - sharey (bool): if True, only the first subplot ymax is modified, as  
                         it will affect all. Otherwise, all subplot ymax are. 
                         default: False
    """

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
def plot_traces(sub_ax, x, y, err=None, title='', lw=None, col=None, 
                alpha=0.5, xticks_ev=6, xticks=None, yticks=None, label=None):
    """
    plot_traces(sub_ax, x, y)

    Plots traces (e.g., mean/median with shaded error bars) on subplot (ax).

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - x (array-like)           : array of x values
        - y (array-like)           : array of y values
        
    Optional args:
        - err (1 or 2D array): either std, SEM or MAD, or quintiles. If 
                               quintiles, 2D array structured as stat x vals
                               default: None
        - title (str)        : subplot title
                               default: ''
        - lw (num)           : plt line weight variable
                               default: None
        - col (str)          : color to use
                               default: None
        - alpha (num)        : plt alpha variable controlling shading 
                               transparency (from 0 to 1)
                               default: 0.5
        - xticks_ev (int)    : frequency of xtick labels
                               default: 6
        - xticks (str)       : xtick labels (overrides xticks_ev)
                               default: None
        - yticks (str)       : ytick labels
                               default: None
        - label (str)        : label for legend
                               default: None
    """
    
    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()
    
    sub_ax.plot(x, y, lw=lw, color=col, label=label)
    col = sub_ax.lines[-1].get_color()
    
    if err is not None:
        err = np.asarray(err).squeeze()
        # only condition where pos and neg error are different
        if len(err.shape) == 2: 
            sub_ax.fill_between(x, err[0], err[1], facecolor=col, alpha=alpha)
        else:
            sub_ax.fill_between(x, y - err, y + err, facecolor=col, alpha=alpha)

    if xticks is None:
        set_ticks(sub_ax, 'x', np.min(x), np.max(x), xticks_ev)
    else:
        sub_ax.set_xticks(xticks)
    
    if yticks is not None:
        sub_ax.set_yticks(yticks)

    if label is not None:
        sub_ax.legend()

    sub_ax.set_title(title)


#############################################
def plot_errorbars(sub_ax, y, err, x=None, title='', lw=None, col=None, 
                   alpha=0.8, xticks=None, yticks=None, label=None):
    """
    plot_errorbars(sub_ax, y, err)

    Plots points with errorbars on subplot (ax).

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - y (array-like)           : array of y values
        - err (1 or 2D array)      : either std, SEM or MAD, or quintiles. If 
                                     quintiles, 2D array structured as stat x 
                                     vals

    Optional args:
        - x (array-like)     : array of x values. 
                               default: None
        - title (str)        : subplot title
                               default: ''
        - col (str)          : color to use
                               default: None
        - lw (num)           : plt line weight variable
                               default: None
        - alpha (num)        : plt alpha variable controlling shading 
                               transparency (from 0 to 1)
                               default: 0.5
        - xticks (str)       : xtick labels ('None' to omit ticks entirely)
                               default: None
        - yticks (str)       : ytick labels
                               default: None
        - label (str)        : label for legend
                               default: None
    """
    
    y = np.asarray(y).squeeze()
    
    if x is None:
        x = list(range(1, len(y) + 1))
    
    if xticks is None:
        sub_ax.set_xticks(x)
    else:
        sub_ax.set_xticks(xticks)

    if yticks is not None:
        sub_ax.set_yticks(yticks)

    x = np.asarray(x).squeeze()

    err = np.asarray(err).squeeze()

    # If err is 1D, provide errorbar length, if err is 2D, provide errorbar 
    # endpoints
    if len(err.shape) == 2: 
        err = [y - err[0], err[1] - y]
    sub_ax.errorbar(x, y, err, fmt='-o', label=label, alpha=alpha, color=col)

    if label is not None:
        sub_ax.legend()

    sub_ax.set_title(title)


#############################################
def get_barplot_xpos(n_grps, n_bars_per, barw, in_grp=1.5, btw_grps=4.0):
    """
    Returns center positions, bar positions and x limits to position bars in a 
    barplot in dense groups along the axis. 

    Required args:
        - n_grps (int)    : number of groups along the x axis
        - n_bars_per (int): number of bars within each group
        - barw (num)      : width of each bar

    Optional args:
        - in_grp (float)  : space between bars in a group, relative to bar 
                            default: 1.5
        - btw_grps (float): space between groups, relative to bar
                            (also determines space on each end, which is half)
                            default: 4.0

    Returns:
        - center_pos (list)    : central position of each group
        - bar_pos (nested list): position of each bar, structured as:
                                    grp x bar
        - xlims (list)         : x axis limit range
    """

    in_grp   = float(in_grp)
    btw_grps = float(btw_grps)
    barw     = float(barw)
    
    # space for each group, relative to barw
    per_grp = n_bars_per + in_grp * (n_bars_per - 1)
    
    # center position of each group
    center_pos = [barw * (x + .5) * (per_grp + btw_grps) 
                                  for x in range(n_grps)]

    # bar positions
    center_idx = (n_bars_per - 1)/2.
    btw_bars = (1 + in_grp) * barw # between bar centers
    bar_pos = [[pos + (i - center_idx) * btw_bars for i in range(n_bars_per)] 
                                                  for pos in center_pos]

    xlims = [0, n_grps * barw * (per_grp + btw_grps)]

    return center_pos, bar_pos, xlims


#############################################
def plot_barplot_signif(sub_ax, xpos, yval, yerr=None, rel_y=0.01):
    """
    Plots significance markers (line and star) above bars showing a significant
    difference. 
    Best to ensure that y axis limits are set before calling this function as
    line and star position are set relative to these limits.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - xpos (array-like)        : list of x positions for line to span
        - ypos (array-like)        : list of y values above which to place line
    
    Optional args:
        - yerr (array-like): list of errors to add to ypos when placing line
                             default: None
        - rel_y (float)    : relative position above ypos at which to place
                             line and star.
                             default: 0.01
    """

    rel_y = float(rel_y)

    # x positions
    if len(xpos) < 2:
        raise ValueError('xpos must be at least of length 2.')
    xpos = [np.min(xpos), np.max(xpos)]
    xmid = np.mean(xpos)

    # y positions
    if yerr is None:
        yerr = np.zeros_like(yval)

    yval = np.asarray(yval)    
    yerr = np.asarray(yerr)
    if len(yval) != len(yerr):
        raise ValueError('If provided, yerr must have the same length as yval.')

    # if quintiles are provided, the second (high) one is retained
    if yerr.shape == 2:
        yerr = yerr[:, 1]

    ymax = np.max(yval + yerr)
    ylims = sub_ax.get_ylim()

    # y line position
    yline = ymax + 2*rel_y*(ylims[1] - ylims[0])
    
    # y text position (will appear higher than line)
    ytext = ymax + rel_y*(ylims[1] - ylims[0])

    sub_ax.text(xmid, ytext, "*", color='k', fontsize='xx-large', 
                fontweight='bold', ha='center', va='bottom')
    sub_ax.plot(xpos, [yline, yline], linewidth=2, color='k')


#############################################
def plot_bars(sub_ax, x, y, err=None, title='', width=0.75, lw=None, col=None,  
              alpha=0.5, xticks=None, yticks=None, xlims=None, label=None, 
              hline=None, capsize=8):
    """
    plot_bars(sub_ax, chunk_val)

    Plots bars (e.g., mean/median with shaded error bars) on subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - x (array-like)           : list of x values

    Optional args:
        - err (1 or 2D array): either std, SEM or MAD, or quintiles. If 
                               quintiles, 2D array structured as stat x vals
                               default: None
        - title (str)        : subplot title
                               default: ''
        - lw (num)           : plt line weight variable
                               default: None
        - col (str)          : color to use
                               default: None
        - alpha (num)        : plt alpha variable controlling shading 
                               transparency (from 0 to 1)
                               default: 0.5
        - xticks (str)       : xtick labels ('None' to omit ticks entirely) 
                               (overrides xticks_ev)
                               default: None
        - yticks (str)       : ytick labels
                               default: None
        - xlims (list)       : xlims
                               default: None
        - label (str)        : label for legend
                               default: None
        - hline (list or num): list of y coordinates at which to add 
                               horizontal bars
                               default: None
        - capsize (num)      : length of errorbar caps
                               default: 8
    """
    
    x = np.asarray(x).squeeze()
    y = y.squeeze()

    patches = sub_ax.bar(x, y, width=width, lw=lw, label=label, color=col)
    
    # get color
    fc = patches[0].get_fc()

    # add errorbars
    if err is not None:
        sub_ax.errorbar(x, y, np.squeeze(err), fmt='None', elinewidth=lw, 
                        capsize=capsize, capthick=lw, ecolor=fc)

    # set edge color to match patch face color
    [patch.set_ec(fc) for patch in patches]

    # set face color to transparency
    [patch.set_fc(list(fc[0:3])+[alpha]) for patch in patches]

    if label is not None:
        sub_ax.legend()

    if xlims is not None:
        sub_ax.set_xlim(xlims)
    
    if hline is not None:
        sub_ax.axhline(y=hline, c='k', lw=1.5)
    
    if xticks in ['None', 'none']:
        sub_ax.tick_params(axis='x', which='both', bottom=False) 
    elif xticks is None:
        sub_ax.set_xticks(xticks)

    if yticks is not None:
        sub_ax.set_yticks(yticks)
    
    sub_ax.set_title(title)


#############################################
def add_colorbar(fig, im, n_cols):
    """
    add_colorbar(fig, im, n_cols)

    Adds a slim colorbar to the right side of a figure.

    Required args:
        - fig (plt Fig)     : figure
        - n_cols (int)      : number of columns in figure
        - im (plt Colormesh): colormesh
    """

    cm_w = 0.03/n_cols
    fig.subplots_adjust(right=1-cm_w*2)
    cbar_ax = fig.add_axes([1-cm_w*1.2, 0.15, cm_w, 0.7])
    fig.colorbar(im, cax=cbar_ax)


#############################################
def plot_colormap(sub_ax, data, xran=None, yran=None, title='', cmap=None, 
                  xticks_ev=6, xticks=None, yticks_ev=10, xlims=None, 
                  ylims=None):
    """
    plot_colormap(sub_ax, data)

    Plots colormap on subplot and returns colormesh image.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - data (2D array)          : data array (X x Y)

    Optional args:
        - x ran (list)   : first and last values along the x axis. If None,
                           will be inferred from the data.
                           default: None
        - yran (list)    : first and last values along the y axis. If None,
                           will be inferred from the data.
                           default: None
        - title (str)    : subplot title
                           default: ''
        - cmap (colormap): a matplotlib colormap
                           default: None
        - xticks_ev (int): frequency of xtick labels
                           default: 6
        - xticks (str)   : xtick labels (overrides xticks_ev)
                           default: None
        - yticks_ev (str): frequency at which to set ytick labels
                           default: None
        - xlims (list)   : xlims
                           default: None
        - ylims (list)   : ylims
                          default: None
    
    Returns:
        - im (plt Colormesh): colormesh image
    """
    
    if xran is None:
        xran = [data.shape[0]+0.5, 0.5, data.shape[0]+1]
    else:
        xran = np.linspace(xran[0], xran[1], data.shape[0]+1)

    if yran is None:
        yran = np.linspace(data.shape[1]+0.5, 0.5, data.shape[1]+1)
    else:
        yran = np.linspace(yran[0], yran[1], data.shape[1]+1)

    if yticks_ev is not None:
        yticks = list(range(0, data.shape[1], yticks_ev))
        sub_ax.set_yticks(yticks)
    
    if xticks is None:
        xticks = np.linspace(np.min(xran), np.max(xran), xticks_ev)
    sub_ax.set_xticks(xticks)

    im = sub_ax.pcolormesh(xran, yran, data.T, cmap=cmap)
    
    if xlims is not None:
        sub_ax.set_xlim(xlims)

    if ylims is not None:
        sub_ax.set_ylim(ylims)

    sub_ax.set_title(title)

    return im


