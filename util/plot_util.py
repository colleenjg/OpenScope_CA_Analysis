"""
plot_util.py

This module contains basic plotting functions.

Authors: Colleen Gillon

Date: February 2023

Note: this code was aggregated from https://github.com/colleenjg/util.
"""

import copy
from pathlib import Path

from matplotlib import pyplot as plt
import matplotlib.cm as mpl_cm
import numpy as np

from util import gen_util, logger_util, math_util


logger = logger_util.get_module_logger(name=__name__)


#############################################
def init_figpar(ncols=4, sharex=False, sharey=True, subplot_hei=7, 
                subplot_wid=7, fig_ext="svg", overwrite=False, save_fig=True, 
                output=".", plt_bkend=None):
    
    """
    Returns a dictionary containing figure parameter dictionaries for 
    initializing a figure, saving a figure, and extra save directory 
    parameters.

    Optional args: 
        - ncols (int)      : number of columns in the figure
                             default: 4 
        - sharex (bool)    : if True, x axis lims are shared across subplots
                             default: False 
        - sharey (bool)    : if True, y axis lims are shared across subplots
                             default: True
        - subplot_hei (num): height of each subplot (inches)
                             default: 7
        - subplot_wid (num): width of each subplot (inches)
                             default: 7
        - fig_ext (str)    : figure extension
                             default: "svg"
        - overwrite (bool) : if False, overwriting existing figures is 
                             prevented by adding suffix numbers.
                             default: False
        - save_fig (bool)  : if True, figures are saved
                             default: True
        - output (Path)    : general directory in which to save output
                             default: "."
        - plt_bkend (str)  : mpl backend to use for plotting (e.g., "agg")
                             default: None
        - paper (bool)     : if True, figures are paper figures, and default 
                             output folder is modified to paper_figures
                             default: False 
                             

    Returns:
        - figpar (dict): dictionary containing figure parameters:
            ["init"] : dictionary containing the following inputs as
                       attributes:
                           ncols, sharex, sharey, subplot_hei, subplot_wid
            ["save"] : dictionary containing the following inputs as
                       attributes:
                           fig_ext, overwrite, save_fig
            ["dirs"]: dictionary containing the following attributes:
                ["figdir"] (Path)   : main folder in which to save figures                
            ["mng"]: dictionary containing the following attributes:
                ["plt_bkend"] (str): mpl backend to use
    """

    fig_init = {
                "ncols"      : ncols,
                "sharex"     : sharex,
                "sharey"     : sharey, 
                "subplot_hei": subplot_hei,
                "subplot_wid": subplot_wid,
                }

    fig_save = {
                "fig_ext"  : fig_ext,
                "overwrite": overwrite,
                "save_fig" : save_fig,
                }
    
    fig_dirs = {
                "figdir": Path(output, "paper_figures")
                }

    fig_mng = {
               "plt_bkend": plt_bkend,
                }


    figpar = {"init" : fig_init,
              "dirs" : fig_dirs,
              "save" : fig_save,
              "mng"  : fig_mng,
              }
    
    return figpar


#############################################
def cond_close_figs(close="all", nb_inline_bkend=False):
    """
    cond_close_figs()

    Checks the pyplot backend and closes all figures, unless 
    close_nb_inline_bkend is False, and the backend is a notebook or inline 
    backend.

    Optional args:
        - close (str or obj)    : what to close, e.g., "all" or a figure
                                  default: "all" 
        - nb_inline_bkend (bool): if True, figures are closed even if the 
                                  matplotlib backend is a notebook or an 
                                  inline backend
                                  default: False
    """

    close_figs = True 
    bkend = plt.get_backend()
    if not nb_inline_bkend and ("inline" in bkend or "nb" in bkend):
        close_figs = False

    if close_figs:
        plt.close(close)

    return


#############################################
def plt_defaults():
    """
    plt_defaults()

    Sets pyplot defaults.
    """


    # set pyplot params
    params = {
        "axes.labelsize"       : "x-large",  # x-large axis labels
        "axes.linewidth"       : 4.0,        # thicker axis lines
        "axes.spines.right"    : False,      # no axis spine on right
        "axes.spines.top"      : False,      # no axis spine at top
        "axes.titlesize"       : "x-large",  # x-large axis title
        "errorbar.capsize"     : 4,          # errorbar cap length
        "figure.titlesize"     : "x-large",  # x-large figure title
        "figure.autolayout"    : True,       # adjusts layout
        "figure.facecolor"     : "w",        # figure facecolor
        "font.size"            : 12,         # basic font size value
        "legend.fontsize"      : "large",    # large legend text
        "lines.dashed_pattern" : [8.0, 4.0], # longer dashes
        "lines.linewidth"      : 5.0,        # thicker lines
        "lines.markeredgewidth": 4.0,        # thick marker edge widths 
                                             # (e.g., cap thickness) 
        "lines.markersize"     : 10,         # bigger markers
        "patch.linewidth"      : 5.0,        # thicker lines for patches
        "savefig.format"       : "svg",      # figure save format
        "savefig.bbox"         : "tight",    # tight cropping of figure
        "savefig.transparent"  : False,      # background transparency
        "xtick.labelsize"      : "x-large",  # x-large x-tick labels
        "xtick.major.size"     : 8.0,        # longer x-ticks
        "xtick.major.width"    : 4.0,        # thicker x-ticks
        "ytick.labelsize"      : "x-large",  # x-large y-tick labels
        "ytick.major.size"     : 8.0,        # longer y-ticks
        "ytick.major.width"    : 4.0,        # thicker y-ticks
        }

    # update pyplot parameters
    plt.rcParams.update(params)


#############################################
def manage_mpl(plt_bkend=None, set_defaults=True):
    """
    manage_mpl()

    Makes changes to the matplotlib backend used as well as matplotlib plotting
    defaults. 

    Optional args:
        - plt_bkend (str)    : matplotlib backend to use
                               default: None
        - set_defaults (bool): if True, the pyplot defaults are updated
                               default: True
    """

    if plt_bkend is not None:
        plt.switch_backend(plt_bkend)
    
    if set_defaults:
        plt_defaults()


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

    if "#" not in col:
        raise ValueError("All colors must be provided in hex format.")
    
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
        raise ValueError("'col_rgb' must comprise 3 values.")
 
    col = "#{}{}{}".format(*[hex(c)[2:] for c in col_rgb])

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
        raise ValueError("Must provide exactly 2 reference colours as input.")

    if n < 2:
        raise ValueError("Must request at least 2 colors.")
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
                val = int(
                    np.around((max_val - min_val) * (i + 1)/div + min_val))
                vals.append(val)
            hexval = rgb_to_hex(vals) # add as next to last
            cols.insert(-1, hexval)
    
    return cols


#############################################
def get_cmap_colors(cmap_name="Blues", n_vals=5, min_n=5):
    """
    get_cmap_colors()

    Returns colors sampled evenly, with a dark bias, from the specified 
    colormap.

    Optional args:
        - cmap_name (str): colormap name
                           default: "Blues"
        - n_vals (int)   : number of values to sample from the colormap
                           default: 5
        - min_n (int)    : minimum number to initially sample (prevents colours 
                           from being too spread apart)
                           default: 5

    Returns:
        - colors (list): list of colors sampled from the colormap.
    """

    cmap = mpl_cm.get_cmap(cmap_name)
    samples = np.linspace(1.0, 0.3, max(min_n, n_vals))[: n_vals]
    colors = [cmap(s) for s in samples]

    return colors


#############################################
def savefig(fig, savename, fulldir=".", fig_ext="svg", overwrite=False, 
            save_fig=True, log_dir=True, **savefig_kw):
    """
    savefig(fig, savename)

    Saves a figure under a specific directory and name, following figure
    parameters and returns final directory name.

    Required args:
        - fig (plt Fig) : figure (if None, no figure is saved, but fulldir is 
                          created and name is returned)
        - savename (str): name under which to save figure
    
    Optional args:
        - fulldir (Path)  : directory in which to save figure
                            default: "."
        - fig_ext (str)   : figure extension
                            default: "svg"
        - overwrite (bool): if False, overwriting existing figures is prevented 
                            by adding suffix numbers.
                            default: False        
        - save_fig (bool) : if False, the figure saving step is skipped. If 
                            log_dir, figure directory will still be logged. 
                            default: True
        - log_dir (bool)  : if True, the save directory is logged 
                            default: True

    Keyword args:
        - savefig_kw (dict): keyword arguments for plt.savefig()

    Returns:
        - fulldir (Path): final name of the directory in which the figure is 
                          saved (may differ from input fulldir, if datetime 
                          subfolder is added.)
    """

    # add subfolder with date and time
    fulldir = Path(fulldir)

    # create directory if doesn't exist
    Path(fulldir).mkdir(parents=True, exist_ok=True)

    if fig is not None:
        # get extension and savename
        if overwrite:
            fullname, _ = gen_util.add_ext(savename, fig_ext) 
        else:
            fullname = gen_util.get_unique_path(
                savename, fulldir, ext=fig_ext
                ).parts[-1]
        if save_fig:
            fig.savefig(fulldir.joinpath(fullname), **savefig_kw)
            log_text = "Figures saved under"
        else:
            log_text = "Figure directory (figure not saved):"

        if log_dir:
            logger.info(f"{log_text} {fulldir}.", extra={"spacing": "\n"})
            
    return fulldir


#############################################
def init_fig(n_subplots, ncols=3, sharex=False, sharey=True, subplot_hei=7, 
             subplot_wid=7, gs=None, **fig_kw):
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
                             default: 7
        - subplot_wid (num): width of each subplot (inches)
                             default: 7
        - gs (dict)        : plt gridspec dictionary
                             default: None
 
    Keyword args:
        - fig_kw (dict): keyword arguments for plt.subplots()

    Returns:
        - fig (plt Fig): fig
        - ax (plt Axis): axis (even if for just one subplot)
    """
   
    nrows = 1
    if n_subplots == 1:
        ncols = 1
    elif n_subplots < ncols:
        ncols = n_subplots
    else:
        nrows = int(np.ceil(n_subplots/float(ncols)))
        # find minimum number of columns given number of rows
        ncols = int(np.ceil(n_subplots/float(nrows)))

    fig, ax = plt.subplots(
        ncols=ncols, nrows=nrows, 
        figsize=(ncols*subplot_wid, nrows*subplot_hei), sharex=sharex, 
        sharey=sharey, squeeze=False, gridspec_kw=gs, **fig_kw)

    return fig, ax
    
    
#############################################
def plot_traces(sub_ax, x, y, err=None, title=None, lw=None, color=None, 
                alpha=0.5, n_xticks=6, xticks=None, yticks=None, label=None, 
                alpha_line=1.0, zorder=None, errx=False, **plot_kw):
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
                               default: None
        - lw (num)           : plt line weight variable
                               default: None
        - color (str)        : color to use
                               default: None
        - alpha (num)        : plt alpha variable controlling shading 
                               transparency (from 0 to 1)
                               default: 0.5
        - n_xticks (int)     : number of xticks (used if xticks is "auto")
                               default: 6
        - xticks (str)       : xtick labels (overrides xticks_ev)
                               ("None" to remove ticks entirely, 
                               "auto" to set xticks automatically from n_xticks)
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

    Keyword args:
        - plot_kw (dict): keyword arguments for plt.plot()
    """
    
    if x is None:
        x = range(len(y))

    x = np.asarray(x).squeeze()
    x = x.reshape(1) if len(x.shape) == 0 else x

    y = np.asarray(y).squeeze()
    y = y.reshape(1) if len(y.shape) == 0 else y

    sub_ax.plot(
        x, y, lw=lw, color=color, label=label, alpha=alpha_line, zorder=zorder, 
        **plot_kw)
    color = sub_ax.lines[-1].get_color()
    
    if err is not None:
        err = np.asarray(err).squeeze()
        err = err.reshape(1) if len(err.shape) == 0 else err
        if not errx:
            # only condition where pos and neg error are different
            if len(err.shape) == 2: 
                sub_ax.fill_between(
                    x, err[0], err[1], facecolor=color, alpha=alpha, 
                    zorder=zorder)
            else:
                sub_ax.fill_between(
                    x, y - err, y + err, facecolor=color, alpha=alpha, 
                    zorder=zorder)
        else:
            # only condition where pos and neg error are different
            if len(err.shape) == 2: 
                sub_ax.fill_betweenx(
                    y, err[0], err[1], facecolor=color, alpha=alpha, 
                    zorder=zorder)
            else:
                sub_ax.fill_betweenx(
                    y, x - err, x + err, facecolor=color, alpha=alpha, 
                    zorder=zorder)

    if isinstance(xticks, str): 
        if xticks in ["none", "None"]:
            sub_ax.tick_params(axis="x", which="both", bottom=False) 
        elif xticks == "auto":
            set_ticks_from_vals(sub_ax, x, axis="x", n=n_xticks)
    elif xticks is not None:
        sub_ax.set_xticks(xticks)

    if yticks is not None:
        sub_ax.set_yticks(yticks)

    if label is not None:
        sub_ax.legend()

    if title is not None:
        sub_ax.set_title(title, y=1.02)


#############################################
def is_last_row(sub_ax):
    """
    is_last_row(sub_ax)

    Returns whether the subplot is in the last row of the axis grid. 
    Should enable compatibility with several matplotlib versions.

    Required args:
        - sub_ax (plt Axis subplot): subplot

    Returns:
        - (bool): whether the subplot is in the last row
    """

    if hasattr(sub_ax, "is_last_row"):
        return sub_ax.is_last_row

    else:
        return sub_ax.get_subplotspec().is_last_row()


#############################################
def set_minimal_ticks(sub_ax, axis="x", **font_kw):
    """
    set_minimal_ticks(sub_ax)

    Sets minimal ticks for a subplot.

    Required args:
        - sub_ax (plt Axis subplot): subplot

    Optional args:
        - axis (str): axes for which to set ticks ("x" or "y")
                      default: "x"

    Keyword args:
        - font_kw (dict): keyword arguments for plt.yticklabels() or 
                          plt.xticklabels() fontdict, e.g. weight
    """

    sub_ax.autoscale()

    if axis == "x":
        lims = sub_ax.get_xlim()
    elif axis == "y":
        lims = sub_ax.get_ylim()
    else:
        gen_util.accepted_values_error("axis", axis, ["x", "y"])

    ticks = rounded_lims(lims)

    if np.sign(ticks[0]) != np.sign(ticks[1]):
        if np.absolute(ticks[1]) > np.absolute(ticks[0]):
            ticks = [0, ticks[1]]
        elif np.absolute(ticks[1]) < np.absolute(ticks[0]):
            ticks = [ticks[0], 0]
        else:
            ticks = [ticks[0], 0, ticks[1]]

    if axis == "x":
        sub_ax.set_xticks(ticks)
        sub_ax.set_xticklabels(ticks, fontdict=font_kw)
    elif axis == "y":
        sub_ax.set_yticks(ticks)
        sub_ax.set_yticklabels(ticks, fontdict=font_kw)
    


#############################################
def rounded_lims(lims, out=False):
    """
    rounded_lims(lims)

    Returns axis limit values rounded to the nearest order of magnitude.

    Required args:
        - lims (iterable): axis limits (lower, upper)

    Optional args:
        - out (bool): if True, limits are only ever rounded out.
                      default: False

    Returns:
        - new_lims (list): rounded axis limits [lower, upper]
    """

    new_lims = list(lims)[:]
    lim_diff = lims[1] - lims[0]

    if lim_diff != 0:
        order = math_util.get_order_of_mag(lim_diff)
        o = -int(order) 

        new_lims = []
        for l, lim in enumerate(lims):
            round_fct = np.around
    
            if lim < 0:
                if out:
                    round_fct = np.ceil if l == 0 else np.floor
                new_lim = -round_fct(-lim * 10 ** o)
            else:
                if out:
                    round_fct = np.floor if l == 0 else np.ceil
                new_lim = round_fct(lim * 10 ** o)

            new_lim = new_lim / 10 ** o
            
            new_lims.append(new_lim)

    return new_lims


#############################################
def set_ticks_from_vals(sub_ax, vals, axis="x", n=6):
    """
    set_ticks_from_vals(sub_ax, vals)

    Sets ticks on specified axis and axis limits around ticks using specified 
    padding, based on the plotted axis values. 

    Required args:
        - sub_ax (plt Axis subplot): subplot
        - vals (array-like)        : axis values in the data

    Optional args:
        - axis (str )     : axis for which to set ticks, i.e., x, y or both
                            default: "x"
        - n (int)         : number of ticks
                            default: 6
        - ret_ticks (bool): if True, tick values are
    """

    n_ticks = np.min([n, len(vals)])
    diff = np.max(vals) - np.min(vals)
    if diff == 0:
        n_dig = 0
    else:
        n_dig = - np.floor(np.log10(np.absolute(diff))).astype(int) + 1
    set_ticks(sub_ax, axis, np.around(np.min(vals), n_dig), 
        np.around(np.max(vals), n_dig), n_ticks)


#############################################
def fig_init_linpla(figpar=None, kind="reg", n_sub=1, sharey=False, 
                    sharex=True):
    """
    fig_init_linpla()

    Returns figpar dictionary with initialization parameters modified for
    graphs across sessions divided by line/plane combinations.

    Optional args:
        - figpar (dict)       : dictionary containing figure parameters 
                                (initialized if None):
            ["init"] : dictionary containing the following inputs as
                       attributes:
                           ncols, sharex, sharey, subplot_hei, subplot_wid
                                default: None
        - kind (str)          : kind of plot 
                                "reg" for single plot per layer/line, 
                                "traces" for traces plot per session (rows), 
                                "prog" for progression plot per session (cols), 
                                "idx" for unexpected data index plot per 
                                    session (rows)
                                default: "reg"
        - n_sub (int)         : number of subplots per line/plane combination
                                default: 1
        - sharey (bool)       : y-axis sharing parameter
                                default: False
        - sharex (bool)       : x-axis sharing parameter
                                default: True

    Returns:
        - figpar (dict): dictionary containing figure parameters:
            ["init"] : dictionary containing the following inputs modified:
                           ncols, sharex, sharey, subplot_hei, subplot_wid
    """

    figpar = copy.deepcopy(figpar)

    if figpar is None:
        figpar = init_figpar()

    if "init" not in figpar.keys():
        raise KeyError("figpar should have 'init' subdictionary.")

    if sharey in [False, "rows"]:
        wspace = 0.5
    else:
        wspace = 0.2

    ncols = 2
    if kind == "traces":
        wid = 3.3
        hei = np.max([wid/n_sub * 1.15, 1.0])
    elif kind == "prog":
        ncols *= n_sub
        wid = np.max([9.0/n_sub, 3.0])
        hei = 2.3
    elif kind == "idx":
        wid = 5
        hei = np.max([wid * 1.5/n_sub, 1.0])
    else:
        wid = 2.5
        hei = 4.3
        figpar["init"]["gs"] = {"hspace": 0.15, "wspace": wspace}

    figpar["init"]["ncols"] = ncols
    figpar["init"]["subplot_hei"] = hei
    figpar["init"]["subplot_wid"] = wid
    figpar["init"]["sharex"] = sharex
    figpar["init"]["sharey"] = sharey

    return figpar


#############################################
def get_fig_rel_pos(ax, grp_len, axis="x"):
    """
    get_fig_rel_pos(ax, grp_len)

    Gets figure positions for middle of each subplot grouping in figure 
    coordinates.

    Required args:
        - ax (plt Axis): axis
        - grp_len (n)  : grouping

    Optional args:
        - axis (str): axis for which to get position ("x" or "y")
                      default: "x"
    Returns:
        - poses (list): positions for each group
    """


    if not isinstance(ax, np.ndarray) and len(ax.shape):
        raise ValueError("ax must be a 2D numpy array.")

    fig = ax.reshape(-1)[0].figure
    n_rows, n_cols = ax.shape
    poses = []
    if axis == "x":
        if n_cols % grp_len != 0:
            raise RuntimeError(f"Group length of {grp_len} does not fit with "
                f"{n_cols} columns.")
        n_grps = int(n_cols/grp_len)
        for n in range(n_grps):
            left_subax = ax[0, n * grp_len]
            left_pos = fig.transFigure.inverted().transform(
                left_subax.transAxes.transform([0, 0]))[0]

            right_subax = ax[0, (n + 1) * grp_len - 1]
            right_pos = fig.transFigure.inverted().transform(
                right_subax.transAxes.transform([1, 0]))[0]

            poses.append(np.mean([left_pos, right_pos]))
    elif axis == "y":
        if n_rows % grp_len != 0:
            raise RuntimeError(f"Group length of {grp_len} does not fit with "
                f"{n_rows} rows.")
        n_grps = int(n_rows/grp_len)
        for n in range(n_grps):
            top_subax = ax[n * grp_len, 0]
            top_pos = fig.transFigure.inverted().transform(
                top_subax.transAxes.transform([0, 1]))[1]

            bottom_subax = ax[(n + 1) * grp_len - 1, 0]
            bottom_pos = fig.transFigure.inverted().transform(
                bottom_subax.transAxes.transform([0, 0]))[1]

            poses.append(np.mean([top_pos, bottom_pos]))
    else:
        gen_util.accepted_values_error("axis", axis, ["x", "y"])

    return poses


#############################################
def set_interm_ticks(ax, n_ticks, axis="x", share=True, skip=True, 
                     update_ticks=False, **font_kw):
    """
    set_interm_ticks(ax, n_ticks)

    Sets axis tick values based on number of ticks, either all as major ticks, 
    or as major ticks with skipped, unlabelled ticks in between. When possible, 
    0 and top tick are set as major ticks.

    Required args:
        - ax (plt Axis): axis
        - n_ticks (n)  : max number of labelled ticks

    Optional args:
        - axis (str)         : axis for which to set ticks ("x" or "y")
                               default: "x"
        - share (bool)       : if True, all axes set the same, based on first 
                               axis.
                               default: True
        - skip (bool)        : if True, intermediate ticks are unlabelled. If 
                               False, all ticks are labelled
                               default: True
        - update_ticks (bool): if True, ticks are updated to axis limits first
                               default: False

    Keyword args:
        - font_kw (dict): keyword arguments for plt.yticklabels() or 
                          plt.xticklabels() fontdict, e.g. weight
    """

    if not isinstance(ax, np.ndarray):
        raise TypeError("Must pass an axis array.")
    
    if n_ticks < 2:
        raise ValueError("n_ticks must be at least 2.")

    for s, sub_ax in enumerate(ax.reshape(-1)):
        if s == 0 or not share:
            if axis == "x":
                if update_ticks:
                    sub_ax.set_xticks(sub_ax.get_xlim())
                ticks = sub_ax.get_xticks()
            elif axis == "y":
                if update_ticks:
                    sub_ax.set_yticks(sub_ax.get_ylim())
                ticks = sub_ax.get_yticks()
            else:
                gen_util.accepted_values_error("axis", axis, ["x", "y"])

            diff = np.mean(np.diff(ticks)) # get original tick steps
            if len(ticks) >= n_ticks:
                ratio = np.ceil(len(ticks) / n_ticks)
            else:
                ratio = 1 / np.ceil(n_ticks / len(ticks))
    
            step = diff * ratio / 2

              # 1 signif digit for differences
            if step == 0:
                o = 0
            else:
                o = -int(math_util.get_order_of_mag(step * 2))

            step = np.around(step * 2, o) / 2
            step = step * 2 if not skip else step

            min_tick_idx = np.round(np.min(ticks) / step).astype(int)
            max_tick_idx = np.round(np.max(ticks) / step).astype(int)

            tick_vals = np.linspace(
                min_tick_idx * step, 
                max_tick_idx * step, 
                max_tick_idx - min_tick_idx + 1
                )

            idx = np.where(tick_vals == 0)[0]
            if 0 not in tick_vals:
                idx = 0

            # adjust if only 1 tick is labelled
            if skip and (len(tick_vals) < (3 + idx % 2)):
                max_tick_idx += 1
                tick_vals = np.append(tick_vals, tick_vals[-1] + step)

            labels = []
            final_tick_vals = []
            for v, val in enumerate(tick_vals):
                val = np.around(val, o + 3) # to avoid floating point precision problems
                final_tick_vals.append(val)                    
                
                if (v % 2 == idx % 2) or not skip:
                    val = int(val) if int(val) == val else val
                    labels.append(val)
                else:
                    labels.append("")

        if axis == "x":
            sub_ax.set_xticks(final_tick_vals)
            # always set ticks (even again) before setting labels
            sub_ax.set_xticklabels(labels, fontdict=font_kw)
            # adjust limits if needed
            lims = list(sub_ax.get_xlim())
            if final_tick_vals[-1] > lims[1]:
                lims[1] = final_tick_vals[-1]
            if final_tick_vals[0] < lims[0]:
                lims[0] = final_tick_vals[0]
            sub_ax.set_xlim(lims)
    
        elif axis == "y":
            sub_ax.set_yticks(final_tick_vals)
            # always set ticks (even again) before setting labels
            sub_ax.set_yticklabels(labels, fontdict=font_kw)
            # adjust limits if needed
            lims = list(sub_ax.get_ylim())
            if final_tick_vals[-1] > lims[1]:
                lims[1] = final_tick_vals[-1]
            if final_tick_vals[0] < lims[0]:
                lims[0] = final_tick_vals[0]
            sub_ax.set_ylim(lims)


#############################################
def adjust_tick_labels_for_sharing(axis_set, axes="x"):
    """
    adjust_tick_labels_for_sharing(axis_set)

    Adjust presence of axis ticks labels for sharing. 

    Required args:
        - axis_set (list): axes to group

    Optional args:
        - axes (str or list): axes ("x", "y") to group
    """
    
    if not isinstance(axes, list):
        axes = [axes]
    for axis in axes:
        if axis == "x":
            row_ns = [subax.get_subplotspec().rowspan.start 
                for subax in axis_set]
            last_row_n = np.max(row_ns)

            for subax in axis_set:
                if subax.get_subplotspec().rowspan.start != last_row_n:
                    subax.tick_params(axis="x", labelbottom=False)

        elif axis == "y":
            col_ns = [subax.get_subplotspec().colspan.start 
                for subax in axis_set]
            first_col_n = np.min(col_ns)

            for subax in axis_set:
                if subax.get_subplotspec().colspan.start != first_col_n:
                    subax.tick_params(axis="y", labelleft=False)
        
        else:
            gen_util.accepted_values_error("axis", axis, ["x", "y"])


#############################################
def get_shared_axes(ax, axis="x"):
    """
    get_shared_axes(ax)

    Returns lists of subplots that share an axis, compensating for what appears 
    to be a bug in matplotlib where subplots from different figures accumulate 
    at each call.

    Required args:
        - ax (plt Axis): axis

    Optional args:
        - axis (str): axis for which to get grouping

    Returns:
        - fixed_grps (list): subplots, organized by group that share the axis
    """


    all_subplots = ax.reshape(-1).tolist()

    if axis == "x":
        grps = list(all_subplots[0].get_shared_x_axes())
    elif axis == "y":
        grps = list(all_subplots[0].get_shared_y_axes())
    else:
        gen_util.accepted_values_error("axis", axis, ["x", "y"])

    fixed_grps = []
    for grp in grps:
        fixed_grp = []
        for subplot in grp:
            if subplot in all_subplots:
                fixed_grp.append(subplot)
        if len(fixed_grp) != 0:
            fixed_grps.append(fixed_grp)

    return fixed_grps


#############################################
def set_shared_axes(axis_set, axes="x", adjust_tick_labels=False):
    """
    set_shared_axes(axis_set)

    Sets axis set passed to be shared. 
    
    Not sure how this interacts with the matplotlib bug described in 
    get_shared_axes() above. Alternative methods didn't work though. Bugs may 
    arise in the future when multiple figures are opened consecutively.

    Relevant matplotlib source code:
        - matplotlib.axes._base: where get_shared_x_axes() is defined
        - matplotlib.cbook.Grouper: where the grouper class is defined

    Required args:
        - axis_set (list): axes to group

    Optional args:
        - axes (str or list)       : axes ("x", "y") to group
                                     default: "x"
        - adjust_tick_labels (bool): if True, tick labels are adjusted for axis 
                                     sharing. (Otherwise, only the limits are 
                                     shared, but tick labels are repeated.)
                                     default: False
    """

    if not isinstance(axes, list):
        axes = [axes]
    for axis in axes:
        if axis == "x":
            grper = axis_set[0].get_shared_x_axes()
        elif axis == "y":
            grper = axis_set[0].get_shared_y_axes()
        else:
            gen_util.accepted_values_error("axis", axis, ["x", "y"])

        # this did not work as a work-around to using get_shared_x_axes()
        # grper = mpl.cbook.Grouper(init=axis_set)

        grper.join(*axis_set)

    if adjust_tick_labels:
        adjust_tick_labels_for_sharing(axis_set, axes)


#############################################
def remove_axis_marks(sub_ax):
    """
    remove_axis_marks(sub_ax)

    Removes all axis marks (ticks, tick labels, spines).

    Required args:
        - sub_ax (plt Axis subplot): subplot    
    """

    sub_ax.get_xaxis().set_visible(False)
    sub_ax.get_yaxis().set_visible(False)

    for spine in ["right", "left", "top", "bottom"]:
        sub_ax.spines[spine].set_visible(False)


#############################################
def set_ticks(sub_ax, axis="x", min_tick=0, max_tick=1.5, n=6, pad_p=0.05, 
              minor=False):
    """
    set_ticks(sub_ax)

    Sets ticks on specified axis and axis limits around ticks using specified 
    padding. 

    Required args:
        - sub_ax (plt Axis subplot): subplot

    Optional args:
        - axis (str)    : axis for which to set ticks, i.e., x, y or both
                          default: "x"
        - min_tick (num): first tick value
                          default: 0
        - max_tick (num): last tick value
                          default: 1.5
        - n (int)       : number of ticks
                          default: 6
        - pad_p (num)   : percentage to pad axis length
                          default: 0.05
        - minor (bool)  : if True, minor ticks are included
                          default: False
    """

    pad = (max_tick - min_tick) * pad_p
    min_end = min_tick - pad
    max_end = max_tick + pad

    if axis == "both":
        axis = ["x", "y"]
    elif axis in ["x", "y"]:
        if not isinstance(axis, list):
            axis = [axis]
    else:
        gen_util.accepted_values_error("axis", axis, ["x", "y", "both"])

    if "x" in axis:
        if min_end != max_end:
            sub_ax.set_xlim(min_end, max_end)
            sub_ax.set_xticks(np.linspace(min_tick, max_tick, n), minor=minor)
        else:
            sub_ax.set_xticks([min_end])
    elif "y" in axis:
        if min_end != max_end:
            sub_ax.set_ylim(min_end, max_end)
            sub_ax.set_yticks(np.linspace(min_tick, max_tick, n), minor=minor)
        else:
            sub_ax.set_yticks([min_end])


#############################################
def get_yticklabel_info(ax, kind="reg"):
    """
    get_yticklabel_info(ax)

    Returns information on how to label y axes.

    Required args:
        - ax (plt Axis): ax

    Optional args:
        - kind (str): kind of plot 
                      "reg" for single plot per layer/line, 
                      "traces" for traces plot per session (rows), 
                      "prog" for progression plot per session (cols), 
                      "idx" for unexpected data index plot per 
                           session (rows)
                      "map" for ROI maps

    Returns:
        - add_yticks (list) : list of subplots that should have ytick labels
    """

    if kind == "map":
        return []

    # establish which subplots should have y tick labels
    axgrps = get_shared_axes(ax, axis="y")
    if len(axgrps) == 4: # sharing by group
        add_idx = -1
        if kind == "prog":
            add_idx = 0
        add_yticks = [axg[add_idx] for axg in axgrps]

    elif len(axgrps) == 0: # no sharing
        add_yticks = ax.reshape(-1)
    elif len(axgrps) == 1: # all sharing
        add_yticks = ax[-1, 0:]
    elif len(axgrps) == ax.shape[0]: # sharing by row
        add_yticks = ax[:, 0].reshape(-1)
    else:
        raise NotImplementedError(f"Condition for {len(axgrps)} subplots in "
            "shared axis groups not implemented.")

    return add_yticks
    
    
#############################################
def get_axislabels(fluor="dff", area=False, scale=False, datatype="roi", 
                   x_ax=None, y_ax=None):
    """
    get_axislabels()

    Returns appropriate labels for x and y axes. 
    
    If y_ax is None, y axis is assumed to be fluorescence, and label is 
    inferred from fluor and dff parameters. If x_ax is None, x axis is assumed
    to be time in seconds.

    Optional args:
        - fluor (str)     : if y_ax is None, whether "raw" or processed 
                            fluorescence traces "dff" are plotted. 
                            default: "dff"
        - area (bool)     : if True, "area" is added after the y_ax label
                            default: False
        - scale (bool)    : if True, "(scaled)" is added after the y_ax label
                            default: False
        - datatype (str)  : type of data, either "run" or "roi"
                            default: "roi"
        - x_ax (str)      : label to use for x axis.
                            default: None
        - y_ax (str)      : label to use for y axis.
                            default: None

    Returns:
        - x_str (str): X axis label
        - y_str (str): Y axis label
    """

    area_str = ""
    if area:
        area_str = " area"
    
    scale_str = ""
    if scale:
        scale_str = " (scaled)"

    if x_ax is None:
        x_str = "Time (s)"
    else:
        x_str = x_ax

    if y_ax is None:
        if datatype == "roi":
            if fluor == "dff":
                delta = u"\u0394"
                y_str = u"{}F/F".format(delta)
            elif fluor == "raw":
                y_str = "raw fluorescence intensity"
            else:
                gen_util.accepted_values_error("fluor", fluor, ["raw", "dff"])
        elif datatype == "run":
            y_str = "Running velocity (cm/s)"
        else:
            gen_util.accepted_values_error("datatype", datatype, ["roi", "run"])
    else:
        y_str = y_ax

    y_str = u"{}{}{}".format(y_str, area_str, scale_str)

    return x_str, y_str
    
    
#############################################
def add_linpla_axislabels(ax, fluor="dff", area=False, scale=False, 
                          datatype="roi", x_ax=None, y_ax=None, 
                          single_lab=False, kind="reg"):
    """
    add_linpla_axislabels(ax)

    Adds the appropriate axis labels to the figure axes. 
    (See get_axislabel() for label content)

    Required args:
        - ax (plt Axis): ax

    Optional args:
        - fluor (str)      : if y_ax is None, whether "raw" or processed 
                             fluorescence traces "dff" are plotted. 
                             default: "dff"
        - area (bool)      : if True, "area" is added after the y_ax label
                             default: False
        - scale (bool)     : if True, "(scaled)" is added after the y_ax label
                             default: False
        - datatype (str)   : type of data, either "run" or "roi"
                             default: "roi"
        - x_ax (str)       : label to use for x axis.
                             default: None
        - y_ax (str)       : label to use for y axis.
                             default: None
        - single_lab (bool): if True, y label only added to top, left of each 
                             subplot group sharing y axis, and x label only 
                             added to bottom, middle, and tick labels only 
                             added to bottom left
                             default: False
        - kind (str)       : kind of plot 
                             "reg" for single plot per layer/line, 
                             "traces" for traces plot per session (rows),  
                             "prog" for progression plot per session (cols), 
                             "idx" for unexpected data index plot per 
                                session (rows)
                             "map" for ROI maps
    """

    if kind == "map":
        return

    add_yticks = get_yticklabel_info(ax, kind=kind)

    # get axis labels if not already provided
    x_str, y_str = get_axislabels(fluor, area, scale, datatype, x_ax, y_ax)

    fig = ax.reshape(-1)[0].figure
    n_rows, n_cols = ax.shape
    if n_rows % 2 != 0 or n_cols % 2 != 0:
        raise RuntimeError("Expected even number of rows and columns.")
    row_per_grp = int(n_rows / 2)
    col_per_grp = int(n_cols / 2)
    
    # add x label
    if single_lab:    
        if kind == "reg":
            fig_ypos = 0.03
        elif kind in ["traces", "idx"]:
            fig_ypos = -0.01
        else:
            fig_ypos = -0.02
        fig.text(0.5, fig_ypos, x_str, fontsize="x-large", 
            horizontalalignment="center", weight="bold")
    else:
        for sub_ax in ax.reshape(-1):
            if is_last_row(sub_ax):
                if kind == "prog":
                    x_pos = fig.transFigure.inverted().transform(
                        sub_ax.transAxes.transform([0.5, 0]))[0]
                    fig.text(x_pos, 0, x_str, fontsize="x-large", 
                        horizontalalignment="center", weight="bold")
                else:
                    sub_ax.set_xlabel(x_str, weight="bold")

    # y labels for each plane set (top and bottom)
    add_y_pos = get_fig_rel_pos(ax, row_per_grp, axis="y")
    if single_lab:
        add_y_pos = add_y_pos[:1] # top only

    ax[0, 0].set_ylabel(".", labelpad=5, color="white") # dummy label
    fig.canvas.draw() # must draw to get axis extent
    fig_transform = fig.transFigure.inverted().transform
    y_lab_xpos = fig_transform(ax[0, 0].yaxis.label.get_window_extent())[0, 0]
    
    for y_pos in add_y_pos:
        fig.text(y_lab_xpos, y_pos, y_str, rotation=90, fontsize="x-large", 
            verticalalignment="center", weight="bold")

    # remove tick labels for all but last row and first column
    label_cols = [0]
    skip_x = False
    if kind == "prog":
        label_cols = [0, col_per_grp]
    elif kind == "idx":
        skip_x = True
        if len(get_shared_axes(ax, axis="x")) != 1:
            skip_x = False
    if single_lab:
        for sub_ax in ax.reshape(-1):
            colNum = sub_ax.get_subplotspec().colspan.start
            if (not (is_last_row(sub_ax) and colNum in label_cols) and 
                skip_x):
                sub_ax.tick_params(labelbottom=False)
            if sub_ax not in add_yticks:
                sub_ax.tick_params(labelleft=False)


#############################################
def format_each_linpla_subaxis(ax, xticks=None, sess_ns=None, kind="reg", 
                               single_lab=True, sess_text=True):
    """
    format_each_linpla_subaxis(ax)

    Formats each subaxis separately, specifically:
    
    - Adds session numbers if provided
    - Removes bottom lines and ticks for top plots
    - Adds x tick labels to bottom plots
    - Adds y tick labels to correct plots

    Required args:
        - ax (plt Axis): plt axis

    Optional args:
        - xticks (list)    : x tick labels (if None, none are added)
                             default: None
        - sess_ns (list)   : list of session numbers
                             default: None 
        - kind (str)       : kind of plot 
                             "reg" for single plot per layer/line, 
                             "traces" for traces plot per session (rows), 
                             "prog" for progression plot per session (cols), 
                             "idx" for unexpected data index plot per 
                                session (rows)
                             default: "reg"
        - single_lab (bool): if True, only one set of session labels it added 
                             to the graph
                             default: True 
        - sess_text (bool) : if True, session numbers are included as text in 
                             the subplots
                             default: True
    """
    # make sure to autoscale subplots after this, otherwise bugs emerge
    for sub_ax in ax.reshape(-1):
        sub_ax.autoscale()

    # get information based on kind of graph
    n_rows, n_cols = ax.shape
    col_per_grp = 1
    pad_p = 0
    if kind == "reg":
        if xticks is not None:
            div = len(xticks)
            pad_p = 1.0 / div
        if n_rows != 2 or n_cols != 2:
            raise RuntimeError(
                "Regular plots should have 2 rows and 2 columns."
                )
    elif kind == "prog":
        if n_cols % 2 != 0:
            raise RuntimeError("Expected even number of columns")
        col_per_grp = int(n_cols/2)
    
    elif kind == "map":
        for sub_ax in ax.reshape(-1):
            remove_axis_marks(sub_ax)
            for spine in ["right", "left", "top", "bottom"]:
                sub_ax.spines[spine].set_visible(True)
                
    elif kind not in ["traces", "idx"]:
        gen_util.accepted_values_error(
            "kind", kind, ["reg", "traces", "prog", "idx", "map"]
            )

    if kind == "map":
        return

    for r in range(n_rows):
        for c in range(n_cols):
            sub_ax = ax[r, c]
            # set x ticks
            if xticks is not None:
                set_ticks(sub_ax, axis="x", min_tick=min(xticks), 
                    max_tick=max(xticks), n=len(xticks), pad_p=pad_p)
                # always set ticks (even again) before setting labels
                sub_ax.set_xticklabels(xticks, weight="bold")
                # to avoid very wide plot features
                if len(xticks) == 1:
                    sub_ax.set_xlim(xticks[0] - 1, xticks[0] + 1)
            # add session numbers
            if kind in ["traces", "idx", "prog"] and sess_ns is not None:
                if sess_text:
                    # place session labels in right/top subplots
                    if kind == "prog":
                        sess_idx = c % len(sess_ns)
                        if r != 0 or c < len(sess_ns):
                            sess_idx = None
                    else:
                        sess_idx = r
                        if c != 1 or r >= len(sess_ns):
                            sess_idx = None
                    if sess_idx is not None:
                        sess_lab = f"sess {sess_ns[sess_idx]}"
                        sub_ax.text(0.65, 0.75, sess_lab, fontsize="x-large", 
                            transform=sub_ax.transAxes, style="italic")
                elif kind == "prog": # alternative session labels for "prog"
                    if (is_last_row(sub_ax) and 
                        (c < len(sess_ns) or not(single_lab))): # BOTTOM
                        sub_ax.text(0.5, -0.5, sess_ns[c % len(sess_ns)], 
                            fontsize="x-large", transform=sub_ax.transAxes, 
                            weight="bold")
            
            # remove x ticks and spines from graphs
            if not is_last_row(sub_ax) and kind != "idx": # NOT BOTTOM
                sub_ax.tick_params(axis="x", which="both", bottom=False) 
                sub_ax.spines["bottom"].set_visible(False)

            # remove y ticks and spines from graphs
            colNum = sub_ax.get_subplotspec().colspan.start
            if kind == "prog" and not colNum in [0, col_per_grp]:
                sub_ax.tick_params(axis="y", which="both", left=False) 
                sub_ax.spines["left"].set_visible(False)

            yticks = [np.around(v, 10) for v in sub_ax.get_yticks()]
            if kind in ["traces", "idx"] and len(yticks) > 3:

                max_abs = np.max(np.absolute(yticks))
                new = [-max_abs, 0, max_abs]
                yticks = list(filter(lambda x: x == 0 or x in yticks, new))

            # always set ticks (even again) before setting labels
            sub_ax.set_yticks(yticks)
            sub_ax.set_yticklabels(yticks, weight="bold")            


#############################################
def adjust_linpla_y_axis_sharing(ax, kind="reg"):
    """
    adjust_linpla_y_axis_sharing(ax)

    If no y axes are shared, sets y-axes belonging to the same plane/line group 
    to be shared, and updates axis scaling.

    Required args:
        - ax (plt Axis): ax

    Optional args:
        - kind (str)       : kind of plot 
                             "reg" for single plot per layer/line, 
                             "traces" for traces plot per session (rows), 
                             "prog" for progression plot per session (cols), 
                             "idx" for unexpected data index plot per 
                                session (rows)
                             default: "reg"
    """

    # check whether any y axes are shared
    set_sharey = (len(get_shared_axes(ax, axis="y")) == 0)

    if kind in ["reg", "map"] or not set_sharey:
        return

    n_rows, n_cols = ax.shape
    to_share = []
    if kind in ["traces", "idx"]:
        if n_rows % 2 != 0:
            raise RuntimeError("Expected even number of rows")
        row_per_grp = int(n_rows/2)
        if row_per_grp > 1:
            to_share = [[ax[i * row_per_grp + r, c] 
                for r in range(row_per_grp)] 
                for i in range(2) for c in range(2)]
    elif kind == "prog":
        if n_cols % 2 != 0:
            raise RuntimeError("Expected even number of columns")
        col_per_grp = int(n_cols/2)
        if col_per_grp > 1:
            to_share = [[ax[r, i * col_per_grp + c] 
                for c in range(col_per_grp)] 
                for i in range(2) for r in range(2)]
    else:
        gen_util.accepted_values_error(
            "kind", kind, ["reg", "traces", "prog", "idx", "map"])

    for axis_set in to_share:
        set_shared_axes(axis_set, "y")
        if kind in ["traces", "idx"]:
            remove_labs = axis_set[:-1]
        elif kind == "prog":
            remove_labs = axis_set[1:]
        for subax in remove_labs:
            subax.tick_params(axis="y", labelleft=False)

    for sub_ax in ax.reshape(-1):
       sub_ax.autoscale()

    return


#############################################
def format_linpla_subaxes(ax, fluor="dff", area=False, datatype="roi", 
                          lines=None, planes=None, xlab=None, xticks=None, 
                          sess_ns=None, ylab=None, kind="reg", tight=True, 
                          modif_share=True, single_lab=True):
    """
    format_linpla_subaxes(ax)

    Formats axis labels and grids for a square of subplots, structured as 
    planes (2 or more rows) x lines (2 columns). 
    
    Specifically:
    - Adds line names to top plots
    - Adds plane information on right plots (midde of top and bottom half)

    Calls:
        - adjust_linpla_y_axis_sharing()
        - format_each_linpla_subaxis()
        - add_linpla_axislabels()

    Required args:
        - ax (plt Axis): plt axis

    Optional args:
        - fluor (str)       : if ylab is None, whether "raw" or processed 
                              fluorescence traces "dff" are plotted. 
                              default: "dff"
        - area (bool)       : if True, "area" is added after the ylab label
                              default: False
        - datatype (str)    : type of data, either "run" or "roi"
                              default: "roi"
        - lines (list)      : ordered lines (2)
                              default: None
        - planes (list)     : ordered planes (2)
                              default: None
        - xlab (str)        : x label
                              default: None
        - xticks (list)     : x tick labels (if None, none are added)
                              default: None
        - sess_ns (list)    : list of session numbers
                              default: None 
        - ylab (str)        : y axis label (overrides automatic one)
                              default: None
        - kind (str)        : kind of plot 
                              "reg" for single plot per layer/line, 
                              "traces" for traces plot per session (rows), 
                              "prog" for progression plot per session (cols), 
                              "idx" for unexpected data index plot per 
                                  session (rows)
                              default: "reg"
        - tight (bool)      : tight figure layout
                              default: True
        - modif_share (bool): if True, y axis sharing modifications are not made
                              default: True
        - single_lab (bool) : if True, where possible, duplicate labels 
                              (axes and ticks) are omitted.
                              default: True
    """
    
    if kind != "idx" and modif_share:
        adjust_linpla_y_axis_sharing(ax, kind=kind)

    sess_text = False if (kind == "prog" and xlab is None) else True
    format_each_linpla_subaxis(
        ax, xticks=xticks, sess_ns=sess_ns, kind=kind, sess_text=sess_text
        )

    # get information based on kind of graph
    n_rows, n_cols = ax.shape
    row_per_grp, col_per_grp = 1, 1
    if kind in ["reg", "map"]:
        fig_xpos = 0.93 # for plane names (x pos)
        fig_ypos = 1 if kind == "reg" else 1.02 # for line names (y pos)
        n = 4
        if n_rows != 2 or n_cols != 2:
            raise RuntimeError(
                "Regular or map plots should have 2 rows and 2 columns."
                )
    elif kind in ["traces", "idx"]:
        fig_xpos = 1.0 # for plane names (x pos)
        fig_ypos = 1.04 # for line names (y pos)
        if n_rows % 2 != 0:
            raise RuntimeError("Expected even number of rows")
        row_per_grp = int(n_rows/2)
        col_per_grp = int(n_cols/2)
        n = 4
    elif kind == "prog":
        n = 3
        fig_xpos = 1.0 # for plane names (x pos)
        fig_ypos = 1.02 # for line names (y pos)
        if n_cols % 2 != 0:
            raise RuntimeError("Expected even number of columns")
        col_per_grp = int(n_cols/2)
    else:
        gen_util.accepted_values_error(
            "kind", kind, ["reg", "traces", "prog", "idx"])

    if kind in ["reg", "prog", "idx"]:
        set_interm_ticks(ax, n, axis="y", weight="bold", share=False)

    # get x axis label and tick information
    if kind == "traces":
        xlab = "Time (s)" if xlab is None else xlab
    elif kind == "idx":
        xlab = "Index" if xlab is None else xlab
    elif kind != "map":
        xlab = "Session" if xlab is None else xlab

    # get and check lines and planes
    if lines is None:
        lines = ["L2/3", "L5"]
    if planes is None:
        planes = ["dendrites", "somata"]
    for l, name in zip([lines, planes], ["lines", "planes"]):
        if len(l) != 2:
            raise RuntimeError(f"2 {name} expected.")

    fig = ax[0, 0].figure

    if tight:
        # Calling tight layout here to ensure that labels are properly 
        # positioned with respect to final layout
        fig.tight_layout()

    # adds plane labels (vertical)
    plane_pos = get_fig_rel_pos(ax, row_per_grp, axis="y")
    for plane, pos in zip(planes, plane_pos):
        fig.text(fig_xpos, pos, plane, rotation=90, fontsize="x-large", 
            verticalalignment="center", weight="bold")

    # adds line names (horizontal)
    line_pos = get_fig_rel_pos(ax, col_per_grp, axis="x")
    for c, (line, pos) in enumerate(zip(lines, line_pos)):
        line_name = f"{line} Pyr" if len(line) and line[1].isdigit() else line
        if kind != "prog" and col_per_grp == 1:
            ax[0, c].set_title(line_name, weight="bold", y=fig_ypos) 
        else:
            # get ypos based on plane positions
            fact = 0.5 * fig_ypos
            ypos = np.max(plane_pos) + np.absolute(np.diff(plane_pos)) * fact
            fig.text(pos, ypos, line_name, fontsize="x-large", 
                horizontalalignment="center", weight="bold")

    # add axis labels
    add_linpla_axislabels(ax, fluor=fluor, area=area, datatype=datatype, 
        x_ax=xlab, y_ax=ylab, single_lab=single_lab, kind=kind)


