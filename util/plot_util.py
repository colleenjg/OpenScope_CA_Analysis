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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from util import file_util, gen_util, math_util


LINCLAB_COLS={'blue'  : '#50a2d5', # Linclab blue
              'red'   : '#eb3920', # Linclab red
              'grey'  : '#969696', # Linclab grey
              'green' : '#76bb4b', # Linclab green
              'purple': '#9370db',
              'orange': '#ff8c00',
              'pink'  : '#bb4b76',
              'yellow': '#e0b424',
              'brown' : '#b04900',
              }


#############################################
def linclab_plt_defaults(font='Liberation Sans', fontdir=None, 
                         print_fonts=False, example=False, dirname=''):
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
        - dirname (str)     : directory in which to save example if example is 
                              True 
                              default: ''
    """

    col_order = ['blue', 'red', 'grey', 'green', 'purple', 'orange', 'pink', 
                 'yellow', 'brown']
    colors = [get_color(key) for key in col_order] 
    col_cyc = plt.cycler(color=colors)

    # set pyplot params
    params = {'axes.labelsize'       : 'xx-large', # xx-large axis labels
              'axes.linewidth'       : 1.5,        # thicker axis lines
              'axes.prop_cycle'      : col_cyc,    # line color cycle
              'axes.spines.right'    : False,      # no axis spine on right
              'axes.spines.top'      : False,      # no axis spine at top
              'axes.titlesize'       : 'x-large',  # x-large axis title
              'errorbar.capsize'     : 4,          # errorbar cap length
              'figure.titlesize'     : 'x-large',  # x-large figure title
              'font.size'            : 12,         # basic font size value
              'legend.fontsize'      : 'x-large',  # x-large legend text
              'lines.dashed_pattern' : [8.0, 4.0], # longer dashes
              'lines.linewidth'      : 2.5,        # thicker lines
              'lines.markeredgewidth': 2.5,        # thick marker edge widths 
                                                   # (e.g., cap thickness) 
              'lines.markersize'     : 10,         # bigger markers
              'patch.linewidth'      : 2.5,        # thicker lines for patches
              'savefig.format'       : 'svg',      # figure save format
              'savefig.bbox'         : 'tight',    # tight cropping of figure
              'xtick.labelsize'      : 'x-large',  # x-large x-tick labels
              'xtick.major.size'     : 8.0,        # longer x-ticks
              'xtick.major.width'    : 2.0,        # thicker x-ticks
              'ytick.labelsize'      : 'x-large',  # x-large y-tick labels
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
            print(f'    {font}')
    
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
            def_font = plt.rcParams[f'font.{font_fam}'][0]
            print(f'Warning: Desired font ({font}) not found, so default '
                  f'({def_font}) will be used instead.\n')
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

    colors = [get_color('blue'), "#ffffff", get_color('red')]

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
def get_color(col='red', ret='single'):
    """
    get_color()

    Returns requested info for the specified color.

    Optional args:
        - col (str): color for which to return info
                     default: 'red'
        - ret (str): type of information to return for color
                     default: 'single'

    Returns:
        if ret == 'single' or 'both':
        - single (str)   : single hex code corresponding to requested color
        if ret == 'col_ends' or 'both':
        - col_ends (list): hex codes for each end of a gradient corresponding to 
                           requested color
    """
    
    # list of defined colors
    curr_cols = ['blue', 'red', 'grey', 'green', 'purple', 'orange', 'pink', 
                 'yellow', 'brown']
    
    if col == 'blue':
        # cols  = ['#7cc7f9', '#50a2d5', '#2e78a9', '#16547d']
        col_ends = ['#8DCCF6', '#07395B']
        single   = LINCLAB_COLS['blue']
    elif col == 'red':
        # cols = ['#f36d58', '#eb3920', '#c12a12', '#971a07']
        col_ends = ['#EF6F5C', '#7D1606']
        single   = LINCLAB_COLS['red']
    elif col == 'grey':
        col_ends = ['#969696', '#060707']
        single   = LINCLAB_COLS['grey']
    elif col == 'green':
        col_ends = ['#B3F38E', '#2D7006']
        single   = LINCLAB_COLS['green']
    elif col == 'purple':
        col_ends = ['#B391F6', '#372165']
        single   = LINCLAB_COLS['purple']
    elif col == 'orange':
        col_ends = ['#F6B156', '#CD7707']
        single   = LINCLAB_COLS['orange']
    elif col == 'pink':
        col_ends = ['#F285AD', '#790B33']
        single   = LINCLAB_COLS['pink']
    elif col == 'yellow':
        col_ends = ['#F6D25D', '#B38B08']
        single   = LINCLAB_COLS['yellow']
    elif col == 'brown':
        col_ends = ['#F7AD75', '#7F3904']
        single   = LINCLAB_COLS['brown']
    else:
        gen_util.accepted_values_error('col', col, curr_cols)

    if ret == 'single':
        return single
    elif ret == 'col_ends':
        return col_ends
    elif ret == 'both':
        return single, col_ends
    else:
        gen_util.accepted_values_error('ret', ret, ['single', 'col_ends', 
                                       'both'])


#############################################
def get_color_range(n=4, col='red'):
    """
    get_color_range()

    Returns a list of color values around the specified general color requested.

    Optional args:
        - n (int)          : number of colors required
                             default: 4
        - col (str or list): general color or two colors (see get_color() for 
                             accepted colors)
                             default: 'red'

    Returns:
        - cols (list): list of colors
    """

    
    cols = gen_util.list_if_not(col)
    if len(cols) not in [1, 2]:
        raise ValueError('`col` must be of length one or two')
    if len(cols) == 2 and n == 1:
        cols = cols[0:1] # retain only first colour

    ends = []
    for col in cols:
        single, col_ends = get_color(col, ret='both')
        if len(cols) == 2:
            ends.append(single)
        else:
            ends = col_ends

    if n == 1:
        cols = [single]
    else:
        cols = get_col_series(ends, n)

    return cols


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
        raise NotImplementedError('Function only implemented for 2D axis '
                                  'arrays.')
    
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
    """
    set_axis_digits(sub_ax)

    Sets the number of digits in the axis tick labels.

    Required args:
        - sub_ax (plt Axis subplot): subplot
    
    Optional args:
        - xaxis (int): number of digits for the x axis 
                       default: None
        - yaxis (int): number of digits for the y axis
                       default: None
    """

    if xaxis is not None:
        n_dig_str = f'%.{int(xaxis)}f'
        sub_ax.xaxis.set_major_formatter(FormatStrFormatter(n_dig_str))

    if yaxis is not None:
        n_dig_str = f'%.{int(yaxis)}f'
        sub_ax.yaxis.set_major_formatter(FormatStrFormatter(n_dig_str))


#############################################
def remove_ticks(sub_ax, xaxis=True, yaxis=True):
    """
    remove_ticks(sub_ax)

    Removes ticks and tick labels for the specified axes.

    Required args:
        - sub_ax (plt Axis subplot): subplot
    
    Optional args:
        - xaxis (bool): if True, applies to x axis 
                        default: None
        - yaxis (bool): if True, applies to y axis
                       default: None
    """

    if xaxis:
        sub_ax.tick_params(axis='x', which='both', bottom=False) 
        sub_ax.set_xticks([])
    if yaxis:
        sub_ax.tick_params(axis='y', which='both', bottom=False) 
        sub_ax.set_yticks([])


#############################################
def remove_graph_bars(sub_ax, bars='all'):
    """
    remove_graph_bars(sub_ax)

    Removes the framing bars around a subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot
    
    Optional args:
        - bars (str or list): bars to remove ('all', 'vert', 'horiz' or a list 
                              of bars (amongst 'top', 'bottom', 'left', 
                              'right'))
                              default: 'all'
    """

    if isinstance(bars, list):
        for bar in bars:
            sub_ax.spines[bar].set_visible(False)

    else: 
        if bars in ['top', 'bottom', 'right', 'left']:
            keys = [bars]
        if bars == 'all':
            keys = sub_ax.spines.keys()
        elif bars == 'vert':
            keys = ['left', 'right']
        elif bars == 'horiz':
            keys = ['top', 'bottom']    
        for key in keys:
            sub_ax.spines[key].set_visible(False)


#############################################
def init_fig(n_subplots, ncols=3, sharex=False, sharey=True, subplot_hei=7.5, 
             subplot_wid=7.5, gs=None, proj=None):
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
        - gs (dict)        : plt gridspec dictionary
                             default: None
        - proj (str)       : plt projection argument (e.g. '3d')
                             default: None

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
                           sharex=sharex, sharey=sharey, squeeze=False, 
                           gridspec_kw=gs, subplot_kw={'projection': proj})
    return fig, ax


#############################################
def savefig(fig, savename, fulldir='', datetime=True, use_dt=None, 
            fig_ext='svg', overwrite=False, print_dir=True, dpi=None):
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
                            default: ''
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
        - dpi (int)       : figure dpi
                            default: None
    
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
        if overwrite:
            fullname, _ = file_util.add_ext(savename, fig_ext) 
        else:
            fullname = file_util.get_unique_path(savename, ext=fig_ext)

        fig.savefig(os.path.join(fulldir, fullname), dpi=dpi)
        
        if print_dir:
            print(f'\nFigures saved under {fulldir}.')
            
    return fulldir


#############################################
def get_repeated_bars(xmin, xmax, cycle=1.0, offset=0):
    """
    get_repeated_bars(xmin, xmax)

    Returns lists of positions at which to place bars cyclicly.

    Required args:
        - xmin (num): minimum x value
        - xmax (num): maximum x value

    Optional args:
        - cycle (num) : distance between bars
                        default: 1.0
        - offset (num): offset from 0 
                        default: 0
    
    Returns:
        - bars (list) : list of x coordinates at which to add bars
                        
    """

    min_bar = int(np.absolute(xmin - offset)//1.5 * np.sign(xmin - offset))
    max_bar = int(np.absolute(xmax - offset)//1.5 * np.sign(xmax - offset))
    bars = [1.5 * b + offset for b in range(min_bar, max_bar + 1)]
 
    return bars


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
        raise ValueError('Arguments `labels` and `xpos` must be of '
                         'the same length.')

    ymin, ymax = sub_ax.get_ylim()
    ypos = (ymax-ymin)*t_hei+ymin
    for l, x in zip(labels, xpos):
        sub_ax.text(x, ypos, l, ha='center', fontsize=18, color=col)


#############################################
def add_bars(sub_ax, hbars=None, bars=None, col='k', alpha=0.5):
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
        - alpha (num)        : plt alpha variable controlling shading 
                               transparency (from 0 to 1)
                               default: 0.5
    """

    torem = []
    if hbars is not None:
        hbars = gen_util.list_if_not(hbars)
        torem = hbars
        for b in hbars:
            sub_ax.axvline(x=b, ls='dashed', c=col, lw=2.5, alpha=alpha)
    if bars is not None:
        bars = gen_util.remove_if(bars, torem)
        for b in bars:
            sub_ax.axvline(x=b, ls='dashed', c=col, lw=1.5, alpha=alpha)


#############################################
def hex_to_rgb(col):
    """
    hex_to_rgb(col)

    Returns hex color in RGB.

    Required args:
        - col (str): color in hex format

    Returns:
        - col_rgb (list): list of RGB values
    """
    n_comp = 3 # r, g, b
    pos = [1, 3, 5] # start of each component
    leng = 2 

    if '#' not in col:
            raise ValueError('All colors must be provided in hex format.')
    
    # get the int value for each color component
    col_rgb = [int(col[pos[i]:pos[i] + leng], 16) for i in range(n_comp)]

    return col_rgb


#############################################
def rgb_to_hex(col_rgb):
    """
    rgb_to_hex(col_rgb)

    Returns RGB in hex color.

    Required args:
        - col_rgb (list): list of RGB values

    Returns:
        - col (str): color in hex format
    """

    if len(col_rgb) != 3:
        raise ValueError('`col_rgb` must comprise 3 values.')
 
    col = '#{}{}{}'.format(*[hex(c)[2:] for c in col_rgb])

    return col


#############################################
def get_col_series(col_ends, n=3):
    """
    get_col_series(col_ends)

    Returns colors between two reference colors, including the two provided.

    Required args:
        - col_ends (list): list of colors in hex format (2)

    Optional args:
        - n (int): number of colors to return, including the 2 provided
                   default: 3

    Returns:
        - cols (list): list of colors between the two reference colors, 
                       including the two provided.
    """

    if len(col_ends) != 2:
        raise ValueError('Must provide exactly 2 reference colours as input.')

    if n < 2:
        raise ValueError('Must request at least 2 colors.')
    else:
        cols = col_ends[:]
        cols_rgb = [hex_to_rgb(col) for col in col_ends]
        div = n - 1
        for i in range(n-2): # for each increment
            vals = []
            for c in range(3): # for each component
                min_val = cols_rgb[0][c]
                max_val = cols_rgb[1][c]
                # get a weighted average for this value
                val = int(np.around((max_val - min_val) * (i + 1)/div + \
                          min_val))
                vals.append(val)
            hexval = rgb_to_hex(vals) # add as next to last
            cols.insert(-1, hexval)
    
    return cols


#############################################
def av_cols(cols):
    """
    av_cols(cols)

    Returns average across list of colors provided.

    Required args:
        - cols (list): list of colors in hex format

    Returns:
        - col (str): averaged color in hex format
    """

    cols = gen_util.list_if_not(cols)

    n_comp = 3 # r, g, b
    col_arr = np.empty([len(cols), n_comp])
    for c, col in enumerate(cols):
        col_arr[c] = hex_to_rgb(col)
    col_arr = np.mean(col_arr, axis=0) # average each component
    # extract hex string
    col = rgb_to_hex([int(np.round(c)) for c in col_arr])
    
    return col


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
def rel_confine_ylims(sub_ax, sub_ran, rel=5):
    """
    rel_confine_ylims(sub_ax, sub_ran)

    Adjusts the y limits of a sub axis to confine a specific range to a 
    relative middle range in the y axis. Will not reduce the y lims only
    increase them

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - sub_ran (list)           : range of values corresponding to subrange
                                     [min, max]

    Optional args:
        - rel (num): relative space to be occupied by the specified range, 
                     e.g. 5 for 1/5
                     default: 5
    """

    y_min, y_max = sub_ax.get_ylim()
    sub_min, sub_max = sub_ran 
    sub_cen = np.mean([sub_min, sub_max])

    y_min = np.min([y_min, sub_cen - rel/2 * (sub_cen - sub_min)])
    y_max = np.max([y_max, sub_cen + rel/2 * (sub_max - sub_cen)])

    sub_ax.set_ylim([y_min, y_max])


#############################################
def add_vshade(sub_ax, start, end=None, width=None, alpha=0.4, col='k'):
    """
    add_vshade(sub_ax, start)

    Plots shaded vertical areas on subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - start (list)             : list of start position arrays for each 
                                     shaded area (bottom)

    Optional args:
        - end (list)   : list of end position arrays for each shaded area 
                         (takes priority over width)
        - width (num)  : width of the shaded areas
        - alpha (num)  : plt alpha variable controlling shading 
                         transparency (from 0 to 1)
                         default: 0.5
        - col (str)    : color to use
                         default: None
    """

    start = gen_util.list_if_not(start)
    
    if end is None and width is None:
        raise ValueError('Must specify end or width.')
    elif end is not None:
        end = gen_util.list_if_not(end)
        if len(start) != len(end):
            raise ValueError('end and start must be of the same length.')
        for st, e in zip(start, end):
            sub_ax.axvspan(st, e, alpha=alpha, color=col)
        if width is not None:
            print('Cannot specify both end and width. Using end.')
    else:
        for st in start:
            sub_ax.axvspan(st, st + width, alpha=alpha, color=col)


#############################################
def plot_traces(sub_ax, x, y, err=None, title='', lw=None, col=None, 
                alpha=0.5, n_xticks=6, xticks=None, yticks=None, label=None, 
                alpha_line=1.0, zorder=None, errx=False):
    """
    plot_traces(sub_ax, x, y)

    Plots traces (e.g., mean/median with shaded error bars) on subplot (ax).

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - x (array-like)           : array of x values (inferred if None)
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
        - n_xticks (int)     : number of xticks
                               default: 6
        - xticks (str)       : xtick labels (overrides xticks_ev)
                               default: None
        - yticks (str)       : ytick labels
                               default: None
        - label (str)        : label for legend
                               default: None
        - alpha_line (num)   : plt alpha variable controlling line 
                               transparency (from 0 to 1)
                               default: 1.0
        - zorder (int)       : plt zorder variable controlling fore-background 
                               position of line
                               default: None
        - errx (bool)        : if True, error is on the x data, not y data
                               default: False
    """
        
    if x is None:
        x = range(len(y))

    x = np.asarray(x).squeeze()
    y = np.asarray(y).squeeze()

    sub_ax.plot(x, y, lw=lw, color=col, label=label, alpha=alpha_line, 
                zorder=zorder)
    col = sub_ax.lines[-1].get_color()
    
    if err is not None:
        err = np.asarray(err).squeeze()
        if not errx:
            # only condition where pos and neg error are different
            if len(err.shape) == 2: 
                sub_ax.fill_between(x, err[0], err[1], facecolor=col, 
                       alpha=alpha, zorder=zorder)
            else:
                sub_ax.fill_between(x, y - err, y + err, facecolor=col, 
                       alpha=alpha, zorder=zorder)
        else:
            # only condition where pos and neg error are different
            if len(err.shape) == 2: 
                sub_ax.fill_betweenx(y, err[0], err[1], facecolor=col, 
                       alpha=alpha, zorder=zorder)
            else:
                sub_ax.fill_betweenx(y, x - err, x + err, facecolor=col, 
                       alpha=alpha, zorder=zorder)


    if xticks is None:
        set_ticks(sub_ax, 'x', np.around(np.min(x), 1), 
                  np.around(np.max(x), 1), n_xticks)
    elif xticks in ['none', 'None']:
        sub_ax.tick_params(axis='x', which='both', bottom=False) 
    else:
        sub_ax.set_xticks(xticks)
    
    if yticks is not None:
        sub_ax.set_yticks(yticks)

    if label is not None:
        sub_ax.legend()

    sub_ax.set_title(title)


#############################################
def plot_btw_traces(sub_ax, y1, y2, x=None, col='k', alpha=0.5):
    """
    plot_btw_traces(sub_ax, y1, y2)

    Plots shaded area between x and y lines on subplot (ax).

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - y1 (array-like)          : first array of y values
        - y2 (array-like)          : first array of y values
        
    Optional args:
        - x (array-like)     : array of x values. If None, a range is used.
                               default: None
        - col (str)          : color to use. If a list is provided, the
                               average is used.
                               default: 'k'
        - alpha (num)        : plt alpha variable controlling shading 
                               transparency (from 0 to 1)
                               default: 0.5
    """

    y1 = np.asarray(y1).squeeze()
    y2 = np.asarray(y2).squeeze()

    if x is None:
        x = list(range(len(y1)))
    else:
        x = np.asarray(x).squeeze()

    if len(y1) != len(y2) or len(x) != len(y1):
        raise ValueError('y1 and y2, and x if provided, must have the same '
                         'length.')

    comp_arr = np.concatenate([y1[:, np.newaxis], y2[:, np.newaxis]], axis=1)
    maxes = np.max(comp_arr, axis=1)
    mins  = np.min(comp_arr, axis=1)

    if isinstance(col, list):
        col = av_cols(col)

    sub_ax.fill_between(x, mins, maxes, alpha=alpha, facecolor=col)


#############################################
def plot_errorbars(sub_ax, y, err, x=None, title='', lw=None, col=None, 
                   alpha=0.8, xticks=None, yticks=None, label=None, 
                   capsize=None, markersize=None):
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
        - capsize (num)      : capsize for errorbars
                               default: None
        - markersize(num)    : markersize
                               default: None
    """
    
    y = np.asarray(y).squeeze()
    
    if x is None:
        x = list(range(1, len(y) + 1))
    
    if not isinstance(xticks, list) and xticks in ['None', 'none']:
        sub_ax.tick_params(axis='x', which='both', bottom=False) 
    elif xticks is None:
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
    sub_ax.errorbar(x, y, err, fmt='-o', label=label, alpha=alpha, color=col, 
                    markersize=markersize, lw=lw)

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
        - in_grp (num)  : space between bars in a group, relative to bar 
                          default: 1.5
        - btw_grps (num): space between groups, relative to bar
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
def add_signif_mark(sub_ax, xpos, yval, yerr=None, rel_y=0.01, col='k'):
    """
    Plots significance markers (star) on subplot.

    Best to ensure that y axis limits are set before calling this function as
    star position are set relative to these limits.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - xpos (num)               : x positions for star
        - yval (num)               : y value above which to place line
    
    Optional args:
        - yerr (num) : errors to add to ypos when placing star
                       default: None
        - rel_y (num): relative position above ypos at which to place star.
                       default: 0.01
        - col (str)  : color for stars
                       default: 'k'
    """

    rel_y = float(rel_y)
    
    # y positions
    if yerr is not None:
        yval = yval + yerr

    ylims = sub_ax.get_ylim()

    # y text position (will appear higher than line)
    star_space = 0.02 # to lower star
    ytext = yval + (rel_y - star_space) * (ylims[1] - ylims[0])

    sub_ax.text(xpos, ytext, "*", color=col, fontsize='xx-large', 
                fontweight='bold', ha='center', va='bottom')


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
        - rel_y (num)      : relative position above ypos at which to place
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
    yval = np.asarray(yval)
    if yerr is None:
        yerr = np.zeros_like(yval)

    for y, err in enumerate(yerr): # in case of NaNs
        if np.isnan(err):
            yerr[y] = 0

    yerr = np.asarray(yerr)
    if len(yval) != len(yerr):
        raise ValueError('If provided, yerr must have the same length as yval.')

    # if quintiles are provided, the second (high) one is retained
    if yerr.shape == 2:
        yerr = yerr[:, 1]

    ymax = np.max(yval + yerr)
    ylims = sub_ax.get_ylim()

    # y line position
    yline = ymax + rel_y * (ylims[1] - ylims[0])
    
    # y text position will be higher than line
    add_signif_mark(sub_ax, xmid, ymax, rel_y=rel_y)

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
    
    if not isinstance(xticks, list) and xticks in ['None', 'none']:
        sub_ax.tick_params(axis='x', which='both', bottom=False) 
    elif xticks is None:
        sub_ax.set_xticks(xticks)

    if yticks is not None:
        sub_ax.set_yticks(yticks)
    
    sub_ax.set_title(title)


#############################################
def add_colorbar(fig, im, n_cols, label=None, cm_prop=0.03):
    """
    add_colorbar(fig, im, n_cols)

    Adds a slim colorbar to the right side of a figure.

    Required args:
        - fig (plt Fig)     : figure
        - n_cols (int)      : number of columns in figure
        - im (plt Colormesh): colormesh

    Optional args:
        - label (str)    : colormap label
                           default: None
        - cm_prop (float): colormap width wrt figure size, to be scaled by number 
                           of columns
                           default: 0.03
    """

    cm_w = cm_prop/n_cols
    fig.subplots_adjust(right=1-cm_w*2)
    cbar_ax = fig.add_axes([1-cm_w*1.2, 0.15, cm_w, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if label is not None:
        fig.set_label(label)


#############################################
def plot_colormap(sub_ax, data, xran=None, yran=None, title='', cmap=None, 
                  n_xticks=6, xticks=None, yticks_ev=10, xlims=None, 
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
        - n_xticks (int) : number of xtick labels
                           default: 6
        - xticks (str)   : xtick labels (overrides n_xticks)
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
        xticks = np.linspace(np.min(xran), np.max(xran), n_xticks)
    sub_ax.set_xticks(xticks)

    im = sub_ax.pcolormesh(xran, yran, data.T, cmap=cmap)
    
    if xlims is not None:
        sub_ax.set_xlim(xlims)

    if ylims is not None:
        sub_ax.set_ylim(ylims)

    sub_ax.set_title(title)

    return im


#############################################
def plot_sep_data(sub_ax, data, lw=0.1, no_edges=True):
    """
    plot_sep_data(sub_ax, data)

    Plots data separated along the first axis, so that each item is scaled to 
    within a unit range, and shifted by 1 from the previous item. Allows 
    items in the same range to be plotted in a stacked fashion. 

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - data (2D array)          : data array (items x values)

    Optional args:
        - lw (num)       : linewidth
        - no_edges (bool): if True, the edges and ticks are removed and the 
                           y limits are tightened
                           default: True
    """

    data_sc = math_util.scale_data(data, axis=1, sc_type='min_max')[0]
    add = np.linspace(0, data.shape[0] * 1.2 + 1, data.shape[0])[:, np.newaxis]
    data_sep = data_sc + add

    sub_ax.plot(data_sep.T, lw=lw)

    if no_edges:
        # removes ticks
        remove_ticks(sub_ax, True, True)
        # removes subplot edges
        remove_graph_bars(sub_ax, bars='all')
        # tighten y limits
        sub_ax.set_ylim(np.min(data_sep), np.max(data_sep))


#############################################
def plot_lines(sub_ax, y, x=None, y_rat=0.0075, col='black', width=0.4, 
              alpha=1.0, zorder=None):
    """
    plot_lines(sub_ax, y)

    Plots lines for each x value at specified height on a subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - y (array-like)           : array of y values for each line

    Optional args:
        - x (array-like): array of x values
                          default: None
        - y_rat (float) : med line thickness, relative to subplot height
                          default: 0.0075
        - col (str)     : bar color
                          default: 'black'
        - width (float) : bar thickness
                          default: 0.4
        - alpha (num)   : plt alpha variable controlling shading 
                          transparency (from 0 to 1)
                          default: 0.5
        - zorder (int)  : plt zorder variable controlling fore-background 
                          position of line
                          default: None
    """

    if x is None:
        x = range(len(y))
    if len(x) != len(y):
        raise ValueError('`x` and `y` must have the same last length.')

    y_lim = sub_ax.get_ylim()
    y_th = y_rat * (y_lim[1] - y_lim[0])
    bottom = y - y_th/2.

    sub_ax.bar(x, height=y_th, bottom=bottom, color=col, width=width, 
               alpha=alpha)


#############################################
def plot_CI(sub_ax, extr, med=None, x=None, width=0.4, label=None, 
            color='lightgrey', med_col='grey', med_rat=0.015, zorder=None):
    """
    plot_CI(sub_ax, extr)

    Plots confidence intervals on a subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - extr (2D array-like)     : array of CI extrema, structured 
                                     as perc [low, high] x bar
    Optional args:
        - med (array-like): array of median/mean values for each bar. If 
                            None, no median line is added
                            default: None
        - x (array-like)  : array of x values (if None, they are inferred)
                            default: None
        - width (float)   : bar thickness
                            default: 0.4
        - label (str)     : label for the bars
                            default: None
        - color (str)     : bar color
                            default: 'lightgrey'
        - med_col (str)   : med line color, if med is provided
                            default: 'grey'
        - med_rat (float) : med line thickness, relative to subplot height
                            default: 0.015
        - zorder (int)    : plt zorder variable controlling fore-background 
                            position of line/shading
                            default: None
    """

    extr = np.asarray(extr)
    if len(extr.shape) == 1:
        extr.reshape([-1, 1])
    if extr.shape[0] != 2:
        raise ValueError('Must provide exactly 2 extrema values for each bar.')

    if x is None:
        x = range(len(extr.shape[1]))
    if len(x) != extr.shape[1]:
        raise ValueError('`x` and `extr` must have the same last '
                         'dimension length.')

    # plot CI
    sub_ax.bar(x, height=extr[1]-extr[0], bottom=extr[0], color=color, 
               width=width, label=label, zorder=zorder)
    
    if label is not None:
        sub_ax.legend()

    # plot median (with some thickness based on ylim)
    if med is not None:
        med = np.asarray(med)
        if len(x) != len(med):
            raise ValueError('`x` and `med` must have the same last '
                             'dimension length.')
        
        plot_lines(sub_ax, med, x, med_rat, col=med_col, width=width, 
                  zorder=zorder)


#############################################
def plot_data_cloud(sub_ax, x_val, y_vals, disp_wid=0.3, label=None, 
                    col='k', alpha=0.5, zorder=None):
    """
    plot_data_cloud(sub_ax, extr)

    Plots y values as a data cloud around an x value

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - x_val (float)            : center of data
        - y_vals (array-like)      : array of y values for each marker.
                                     default: None

    Optional args:
        - disp_std (float)   : dispersion standard deviation 
                               (will clip at 2.5 * disp_std)
                               default: 0.4
        - label (str)        : label for the bars
                               default: None
        - col (str)          : marker color
                               default: 'k'
        - alpha (float)      : transparency
                               default: 0.5
        - zorder (int)       : plt zorder variable controlling fore-background 
                               position of line/shading
                               default: None
    
    Returns:
        - cloud (plt Line): pyplot Line object containing plotted dots
    """

    x_vals = np.random.normal(x_val, disp_wid, len(y_vals))

    # clip points outside 2.5 stdev
    min_val, max_val = [x_val + sign * 2.5 * disp_wid for sign in [-1, 1]]
    x_vals[np.where(x_vals < min_val)] = min_val
    x_vals[np.where(x_vals > max_val)] = max_val

    cloud = sub_ax.plot(x_vals, y_vals, marker='.', lw=0, color=col,
                        alpha=alpha, label=label, zorder=zorder)[0]

    if label is not None:
        sub_ax.legend()

    return cloud

    