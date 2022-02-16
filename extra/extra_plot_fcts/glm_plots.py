"""
glm_plots.py

This script contains functions to plot results of GLM analyses (glm.py) from 
dictionaries.

Authors: Colleen Gillon

Date: October, 2019

Note: this code uses python 3.7.

"""

import copy
import warnings
from pathlib import Path

import numpy as np

from util import file_util, gen_util, logger_util, math_util, plot_util
from sess_util import sess_plot_util, sess_str_util


TAB = "    "


logger = logger_util.get_module_logger(name=__name__)


#############################################
def plot_from_dict(dict_path, plt_bkend=None, fontdir=None, datetime=True, 
                   overwrite=False):
    """
    plot_from_dict(dict_path)

    Plots data from dictionaries containing analysis parameters and results.

    Required args:
        - dict_path (Path): path to dictionary to plot data from
    
    Optional_args:
        - plt_bkend (str) : mpl backend to use for plotting (e.g., "agg")
                            default: None
        - fontdir (Path)  : path to directory where additional fonts are stored
                            default: None
        - datetime (bool) : figpar["save"] datatime parameter (whether to 
                            place figures in a datetime folder)
                            default: True
        - overwrite (bool): figpar["save"] overwrite parameter (whether to 
                            overwrite figures)
                            default: False
    """

    logger.info(f"Plotting from dictionary: {dict_path}", 
        extra={"spacing": "\n"})


    figpar = sess_plot_util.init_figpar(
        plt_bkend=plt_bkend, fontdir=fontdir, datetime=datetime, 
        overwrite=overwrite
        )
    plot_util.manage_mpl(cmap=False, **figpar["mng"])

    dict_path = Path(dict_path)

    info = file_util.loadfile(dict_path)
    savedir = dict_path.parent

    analysis = info["extrapar"]["analysis"]

    # 0. Plots the explained variance
    if analysis == "v": # difference correlation
        plot_glm_expl_var(figpar=figpar, savedir=savedir, **info)
    else:
        warnings.warn(f"No plotting function for analysis {analysis}", 
            category=UserWarning, stacklevel=1)

    plot_util.cond_close_figs()
    

#############################################
def plot_glm_expl_var(analyspar, sesspar, stimpar, extrapar, glmpar,
                      sess_info, all_expl_var, figpar=None, savedir=None):
    """
    plot_glm_expl_var(analyspar, sesspar, stimpar, extrapar, 
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
            ["analysis"] (str): analysis type (e.g., "v")
        - sess_info (dict)    : dictionary containing information from each
                                session 
            ["mouse_ns"] (list)   : mouse numbers
            ["sess_ns"] (list)    : session numbers  
            ["lines"] (list)      : mouse lines
            ["planes"] (list)     : imaging planes
            ["nrois"] (list)      : number of ROIs in session

        - all_expl_var (list) : list of dictionaries with explained variance 
                                for each session set, with each glm 
                                coefficient as a key:
            ["full"] (list)    : list of full explained variance stats for 
                                 every ROI, structured as ROI x stats
            ["coef_all"] (dict): max explained variance for each ROI with each
                                 coefficient as a key, structured as ROI x stats
            ["coef_uni"] (dict): unique explained variance for each ROI with 
                                 each coefficient as a key, 
                                 structured as ROI x stats
            ["rois"] (list)    : ROI numbers (-1 for GLMs fit to 
                                 mean/median ROI activity)
    
    Optional args:
        - figpar (dict): dictionary containing the following figure parameter 
                         dictionaries
                         default: None
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters
        - savedir (str): path of directory in which to save plots.
                         default: None
    
    Returns:
        - fulldir (str) : final path of the directory in which the figure is 
                          saved (may differ from input savedir, if datetime 
                          subfolder is added.)
        - savename (str): name under which the figure is saved
    """

    stimstr_pr = sess_str_util.stim_par_str(
        stimpar["stimtype"], stimpar["visflow_dir"], stimpar["visflow_size"], 
        stimpar["gabk"], "print")
    dendstr_pr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], "roi", "print")

    sessstr = sess_str_util.sess_par_str(
        sesspar["sess_n"], stimpar["stimtype"], sesspar["plane"], 
        stimpar["visflow_dir"], stimpar["visflow_size"], stimpar["gabk"]) 
    dendstr = sess_str_util.dend_par_str(
        analyspar["dend"], sesspar["plane"], "roi")

    # extract some info from sess_info
    keys = ["mouse_ns", "sess_ns", "lines", "planes"]
    [mouse_ns, sess_ns, lines, planes] = [sess_info[key] for key in keys]

    n_sess = len(mouse_ns)
    
    nroi_strs = sess_str_util.get_nroi_strs(sess_info, style="par")

    plot_bools = [ev["rois"] not in [[-1], "all"] for ev in all_expl_var]
    n_sess = sum(plot_bools)

    if stimpar["stimtype"] == "gabors":
        xyzc_dims = ["unexpected", "gabor_frame", "run_data", "pup_diam_data"]
        log_dims = xyzc_dims + ["gabor_mean_orientation"]
    elif stimpar["stimtype"] == "visflow":
        xyzc_dims = [
            "unexpected", "main_flow_direction", "run_data", "pup_diam_data"
            ]
        log_dims = xyzc_dims
    
    # start plotting
    logger.info("Plotting GLM full and unique explained variance for "
        f"{', '.join(xyzc_dims)}.", extra={"spacing": "\n"})

    if n_sess > 0:
        if figpar is None:
            figpar = sess_plot_util.init_figpar()

        figpar = copy.deepcopy(figpar)
        cmap = plot_util.linclab_colormap(nbins=100, no_white=True)

        if figpar["save"]["use_dt"] is None:
            figpar["save"]["use_dt"] = gen_util.create_time_str()
        figpar["init"]["ncols"] = n_sess
        figpar["init"]["sharex"] = False
        figpar["init"]["sharey"] = False
        figpar["init"]["gs"] = {"wspace": 0.2, "hspace": 0.35}
        figpar["save"]["fig_ext"] = "png"
        
        fig, ax = plot_util.init_fig(2 * n_sess, **figpar["init"], proj="3d")

        fig.suptitle("Explained variance per ROI", y=1)

        # get colormap range
        c_range = [np.inf, -np.inf]
        c_key = xyzc_dims[3]

        for expl_var in all_expl_var:
            for var_type in ["coef_all", "coef_uni"]:
                rs = np.where(np.asarray(expl_var["rois"]) != -1)[0]
                if c_key in expl_var[var_type].keys():
                    c_data = np.asarray(expl_var[var_type][c_key])[rs, 0]
                    # adjust colormap range
                    c_range[0] = np.min([c_range[0], min(c_data)])
                    c_range[1] = np.max([c_range[1], max(c_data)])
        
        if not np.isfinite(sum(c_range)):
            c_range = [-0.5, 0.5] # dummy range
        else:
            c_range = plot_util.rounded_lims(c_range, out=True)

    else:
        logger.info("No plots, as only results across ROIs are included")
        fig = None

    i = 0
    for expl_var in all_expl_var:
        # collect info for plotting and logging results across ROIs
        rs = np.where(np.asarray(expl_var["rois"]) != -1)[0]
        all_rs = np.where(np.asarray(expl_var["rois"]) == -1)[0]
        if len(all_rs) != 1:
            raise RuntimeError("Expected only one result for all ROIs.")
        else:
            all_rs = all_rs[0]
            full_ev = expl_var["full"][all_rs]

        title = (f"Mouse {mouse_ns[i]} - {stimstr_pr}\n(sess {sess_ns[i]}, "
                f"{lines[i]} {planes[i]}{dendstr_pr},{nroi_strs[i]})")
        logger.info(title, extra={"spacing": "\n"})

        math_util.log_stats(full_ev, stat_str="\nFull explained variance")

        dim_length = max([len(dim) for dim in log_dims])
        
        for v, var_type in enumerate(["coef_all", "coef_uni"]):
            if var_type == "coef_all":
                sub_title = "Explained variance per coefficient"
            elif var_type == "coef_uni":
                sub_title = "Unique explained variance\nper coefficient"
            logger.info(sub_title, extra={"spacing": "\n"})

            dims_all = []
            for key in log_dims:
                if key in xyzc_dims:
                    # get mean/med
                    if key not in expl_var[var_type].keys():
                        dims_all.append("dummy")
                        continue

                    dims_all.append(np.asarray(expl_var[var_type][key])[rs, 0])
                math_util.log_stats(
                    expl_var[var_type][key][all_rs], 
                    stat_str=key.ljust(dim_length), log_spacing=TAB
                    )

            if not plot_bools[-1]:
                continue

            if v == 0:
                y = 1.12
                subpl_title = f"{title}\n{sub_title}"
            else:
                y = 1.02
                subpl_title = sub_title

            # retrieve values and names for each dimension, including dummy 
            # dimensions
            use_xyzc_dims = []
            n_vals = None
            dummies = []
            pads = [16, 16, 20]
            for d, dim in enumerate(dims_all):
                dim_name = xyzc_dims[d].replace("_", " ")
                if " direction"  in dim_name:
                    dim_name = dim_name.replace(" direction", "\ndirection")
                    pads[d] = 24
                if isinstance(dim, str) and dim == "dummy":
                    dummies.append(d)
                    use_xyzc_dims.append(f"{dim_name} (dummy)")
                else:
                    n_vals = len(dim)
                    use_xyzc_dims.append(dim_name)
            
            for d in dummies:
                dims_all[d] = np.zeros(n_vals)

            [x_data, y_data, z_data, c_data] = dims_all

            sub_ax = ax[v, i]
            im = sub_ax.scatter(
                x_data, y_data, z_data, c=c_data, cmap=cmap, 
                vmin=c_range[0], vmax=c_range[1]
                )
            sub_ax.set_title(subpl_title, y=y)
            # sub_ax.set_zlim3d(0, 1.0)

            # adjust padding for z axis
            sub_ax.tick_params(axis='z', which='major', pad=10)

            # add labels
            sub_ax.set_xlabel(use_xyzc_dims[0], labelpad=pads[0])
            sub_ax.set_ylabel(use_xyzc_dims[1], labelpad=pads[1])
            sub_ax.set_zlabel(use_xyzc_dims[2], labelpad=pads[2])

            if v == 0:
                full_ev_lab = math_util.log_stats(
                    full_ev, stat_str="Full EV", ret_str_only=True
                    )
                sub_ax.plot([], [], c="k", label=full_ev_lab)
                sub_ax.legend()

        i += 1

    if fig is not None:
        plot_util.add_colorbar(
            fig, im, n_sess, label=use_xyzc_dims[3],
            space_fact=np.max([2, n_sess])
            )

        # plot 0 planes, and lines
        for sub_ax in ax.reshape(-1):
            sub_ax.autoscale(False)
            all_lims = [sub_ax.get_xlim(), sub_ax.get_ylim(), sub_ax.get_zlim()]
            xs, ys, zs = [
                [vs[0] - (vs[1] - vs[0]) * 0.02, vs[1] + (vs[1] - vs[0]) * 0.02]
                for vs in all_lims
                ]
            
            for plane in ["x", "y", "z"]:
                if plane == "x":
                    xx, yy = np.meshgrid(xs, ys)
                    zz = xx * 0
                    x_flat = xs
                    y_flat, z_flat = [0, 0], [0, 0]
                elif plane == "y":
                    yy, zz = np.meshgrid(ys, zs)
                    xx = yy * 0
                    y_flat = ys
                    z_flat, x_flat = [0, 0], [0, 0]
                elif plane == "z":
                    zz, xx = np.meshgrid(zs, xs)
                    yy = zz * 0
                    z_flat = zs
                    x_flat, y_flat = [0, 0], [0, 0]
                
                sub_ax.plot_surface(xx, yy, zz, alpha=0.05, color="k")
                sub_ax.plot(
                    x_flat, y_flat, z_flat, alpha=0.4, color="k", ls=(0, (2, 2))
                    )

    if savedir is None:
        savedir = Path(
            figpar["dirs"]["roi"],
            figpar["dirs"]["glm"])

    savename = (f"roi_glm_ev_{sessstr}{dendstr}")

    fulldir = plot_util.savefig(fig, savename, savedir, **figpar["save"])

    return fulldir, savename                              

