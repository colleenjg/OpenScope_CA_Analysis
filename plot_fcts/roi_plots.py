"""
roi_plots.py

This script contains functions for plotting ROI masks and projections.

Authors: Colleen Gillon

Date: February 2023

Note: this code was aggregated from https://github.com/colleenjg/OpenScope_CA_Analysis.
"""

import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches

from util import gen_util, logger_util, plot_util
from plot_fcts import plot_helper_fcts


TAB = "    "
ORDERED_COLORS = ["orange", "crimson", "mediumblue"]
N_LEVELS_MASKS = 2
N_LEVELS_PROJS = 256

PIX_PER_SIDE = 512
UM_PER_PIX = 400 / PIX_PER_SIDE


logger = logger_util.get_module_logger(name=__name__)


#############################################
def get_sess_cols(roi_mask_df):
    """
    get_sess_cols(roi_mask_df)

    Returns a dictionary mapping session numbers to colours, based on all 
    session numbers found in the dataframe.

    Required args:
        - roi_mask_df (pd.DataFrame):
            dataframe with session numbers under "sess_ns" column 
            (either as ints or lists)

    Returns:
        - sess_cols (dict):
            dictionary with session numbers (int) as keys, and colours as 
            values.
    """
    
    if "sess_ns" not in roi_mask_df.columns:
        raise KeyError(f"roi_mask_df should have a 'sess_ns' columns.")

    sess_ns = roi_mask_df["sess_ns"].tolist()
    if isinstance(sess_ns[0], list):
        sess_ns = np.concatenate(sess_ns)
    
    all_sess_ns = np.sort(np.unique(sess_ns).astype(int))
    
    # map session numbers to colors
    if len(all_sess_ns) > len(ORDERED_COLORS):
        raise RuntimeError(
            f"Expected sessions to go no higher than {len(ORDERED_COLORS)}."
            )
    sess_cols = {
        sess_n: col for sess_n, col in zip(all_sess_ns, ORDERED_COLORS)
        }

    return sess_cols


#############################################
def add_sess_col_leg(sub_ax, sess_cols, alpha=0.6, bbox_to_anchor=(0.5, 0.5), 
                     fontsize="medium"):
    """
    add_sess_col_leg(sub_ax, sess_cols)

    Adds "ghost" patches and a legend for session colours to a subplot.

    Required args:
        - sub_ax (plt Axis subplot): 
            subplot
        - sess_cols (dict):
            dictionary with session numbers (int) as keys, and colours as 
            values.
    
    Optional args:
        - alpha (num): 
            plt alpha variable controlling transparency of patches (from 0 to 1)
            default: 0.6
        - bbox_to_anchor (tuple):
            legend position on subplot (y, x)
            default: (0.5, 0.5)
        - fontsize (str):
            legend fontsize
            default: "medium"
    """

    leg_handles = []
    for sess_n, col in sess_cols.items():
        leg_handles.append(
            mpl.patches.Patch(color=col, label=f"sess {sess_n}", alpha=alpha)
        )

    sub_ax.legend(
        handles=leg_handles, 
        bbox_to_anchor=bbox_to_anchor, 
        fontsize=fontsize,
        borderpad=0.75
        )


#############################################
def crop_roi_image(df_row, roi_image, get_crop_area_only=False):
    """
    crop_roi_image(df_row, roi_image)

    Return ROI image cropped based on cropping and shift parameters.

    Required args:
        - df_row (pd Series):
            pandas series with the following keys:
            - "crop_fact" (num): factor by which to crop masks (> 1) 
            - "shift_prop_hei" (float): proportion by which to shift cropped 
                mask center vertically from left edge [0, 1]
            - "shift_prop_wid" (float): proportion by which to shift cropped 
                mask center horizontally from left edge [0, 1]

        - roi_image (2 or 3D array):
            ROI image

    Optional args:
        - get_crop_area_only (bool):
            if True, cropping information is retrieved for ROI masks, but no 
            cropping is done
            default: False

    Returns:
        - roi_image (2 or 3D array):
            cropped ROI 
        - crop_area (2D array):
            values indicating location of crop rectangle, 
            [[start_x, len_y], [start_y, len_y]]
            default = None
    """

    add_dim = (len(roi_image.shape) == 2)
    if add_dim:
        roi_image = roi_image[np.newaxis]

    dims = ["hei", "wid"]
    keys = ["crop_fact"] + [f"shift_prop_{dim}" for dim in dims]
    for key in keys:
        if key not in df_row.keys():
            raise KeyError(
                f"For cropping, df_row must include {key} key."
                )

    crop_fact = df_row["crop_fact"]
    if crop_fact < 1:
        raise ValueError("crop_fact must be at least 1")

    crop_area = []
    for d, dim in enumerate(dims):
        shift_prop = df_row[f"shift_prop_{dim}"]
        if shift_prop < 0 or shift_prop > 1:
            raise RuntimeError("shift_prop must be between 0 and 1.")
        orig_size = roi_image.shape[d + 1]
        new_size = int(np.around(orig_size / crop_fact))
        shift = int(shift_prop * (orig_size - new_size))
        crop_area.append([shift, new_size])
        if not get_crop_area_only:
            if d == 0:
                roi_image = roi_image[:, shift : shift + new_size]
            else:
                roi_image = roi_image[:, :, shift : shift + new_size]

    if add_dim:
        roi_image = roi_image[0]
    
    crop_area = np.asarray(crop_area)

    return roi_image, crop_area


#############################################
def create_roi_mask_contours(df_row, mask_key=None, sess_idx=0, cw=1, 
                             outer=False, crop=False):
    """
    create_roi_mask_contours(df_row)

    Returns ROI mask contour image.

    Required args:
        - df_row (pd Series):
            pandas series with the following keys:
            - {mask_key} (list): list of mask indices for each session, 
                and each ROI (sess x (ROI, hei, wid) x val) (not registered)
            - "nrois" (list): number of ROIs for each session
            - "roi_mask_shapes" (list): shape into which ROI mask indices index 
                (sess x hei x wid)
            if crop:
            - "crop_fact" (num): factor by which to crop masks (> 1) 
            - "shift_prop_hei" (float): proportion by which to shift cropped 
                mask center vertically from left edge [0, 1]
            - "shift_prop_wid" (float): proportion by which to shift cropped 
                mask center horizontally from left edge [0, 1]

    Optional args:
        - mask_key (str): 
            key under which ROI masks are stored. If None, default is used 
            "roi_mask_idxs"
            default: None
        - sess_idx (int):
            session index
            default: 0
        - cw (int):
            contour width (pixels)
            default: 1
        - outer (bool):
            if True, only pixels outside of ROI mask are used to create contour
            default: False
        - crop (bool):
            if True, ROI mask image is cropped, per specifications in df_row.
            default: False

    Returns:
        - roi_masks (2D array):
            ROI mask contour image (hei x wid), overlayed for all ROIs, 
            with 1s where mask contours are present, and 0s elsewhere.
    """

    mask_key = "roi_mask_idxs" if mask_key is None else mask_key

    idxs = [np.asarray(sub) for sub in df_row[mask_key][sess_idx]]
    nrois = df_row["nrois"][sess_idx]
    mask_shape = (nrois, ) + tuple(df_row["roi_mask_shapes"][1:])
    roi_masks = np.zeros(mask_shape).astype("int8")
    roi_masks[tuple(idxs)] = 1 # ROI x hei x wid

    pad_zhw = [0, 0], [cw, cw], [cw, cw]
    contour_mask = np.pad(roi_masks, pad_zhw, "constant", constant_values=0)
    shifts = range(-cw, cw + 1)
    _, h, w = roi_masks.shape
    for h_sh, w_sh in itertools.product(shifts, repeat=2):
        if h_sh == 0 and w_sh == 0:
            continue
        contour_mask[:, cw+h_sh: h+cw+h_sh, cw+w_sh: w+cw+w_sh] += roi_masks
    
    sub_mask = contour_mask[:, cw:h+cw, cw:w+cw]
    contour_mask = (sub_mask != len(shifts) ** 2) * (sub_mask != 0)
    if outer:
        restrict_masks = 1 - roi_masks
    else:
        restrict_masks = roi_masks
    
    # collapse mask contours
    roi_masks = np.max(contour_mask * restrict_masks, axis=0).astype(int)

    if crop:
        roi_masks, _ = crop_roi_image(df_row, roi_masks)

    return roi_masks


#############################################
def create_sess_roi_masks(df_row, mask_key=None, crop=False, 
                          get_crop_area_only=False):
    """
    create_sess_roi_masks(df_row)

    Returns overlayed ROI masks for sessions registered to one another.

    Required args:
        - df_row (pd Series):
            pandas series with the following keys:
            - {mask_key} (list): list of mask indices, registered across 
                sessions, for each session 
                (flattened across ROIs) ((sess, hei, wid) x val)
            - "roi_mask_shapes" (list): shape into which ROI mask indices index 
                (sess x hei x wid)
            if crop:
            - "crop_fact" (num): factor by which to crop masks (> 1) 
            - "shift_prop_hei" (float): proportion by which to shift cropped 
                mask center vertically from left edge [0, 1]
            - "shift_prop_wid" (float): proportion by which to shift cropped 
                mask center horizontally from left edge [0, 1]

    Optional args:
        - mask_key (str): 
            key under which ROI masks are stored. If None, default is used 
            "registered_roi_mask_idxs"
            default: None
        - crop (bool):
            if True, ROI mask image is cropped, per specifications in df_row.
            default: False
        - get_crop_area_only (bool):
            if True, cropping information is retrieved for ROI masks, but no 
            cropping is done
            default: False

    Returns:
        - roi_masks (2D array):
            ROI masks image (hei x wid), overlayed for all sessions, with 1s 
            where masks are present, and 0s elsewhere.
        if get_crop_area_only:
        - crop_area (2D array):
            values indicating where to crop roi masks, 
            [[start_x, len_y], [start_y, len_y]]
            default = None

    """

    mask_key = "registered_roi_mask_idxs" if mask_key is None else mask_key

    idxs = [np.asarray(sub) for sub in df_row[mask_key]]    
    roi_masks = np.zeros(df_row["roi_mask_shapes"]).astype(int)
    roi_masks[tuple(idxs)] = 1 # sess x hei x wid
        
    if crop or get_crop_area_only:
        roi_masks, crop_area = crop_roi_image(
            df_row, roi_masks, get_crop_area_only=get_crop_area_only
            )
    
        if get_crop_area_only:
            return roi_masks, crop_area


    return roi_masks


#############################################
def add_imaging_plane(sub_ax, imaging_plane, alpha=1.0, zorder=-13):
    """
    add_imaging_plane(sub_ax, imaging_plane)

    Adds imaging plane to subplot.

    Required args:
        - sub_ax (plt Axis subplot): 
            subplot
        - imaging_plane (2D array):
            imaging plane (hei x wid)
    
    Optional args:
        - alpha (num): 
            plt alpha variable controlling transparency of imaging plane 
            (from 0 to 1)
            default: 1.0
        - zorder (int):
            zorder for the imaging plane
            default: -13
    """

    alphas = np.ones(N_LEVELS_PROJS + 3) * alpha

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "mask_cmap", 
        ["black", "white"], 
        N=N_LEVELS_PROJS
        )

    cmap._init()
    cmap._lut[:, -1] = alphas

    sub_ax.imshow(imaging_plane, cmap=cmap, zorder=zorder)


#############################################
def add_roi_mask(sub_ax, roi_masks, col="orange", alpha=0.6, 
                 background="white", transparent=True, lighten=0, 
                 mark_crop_area=None):
    """
    add_roi_mask(sub_ax, roi_masks)

    Adds ROI masks to subplot.

    Required args:
        - sub_ax (plt Axis subplot): 
            subplot
        - roi_masks (2D array):
            ROI masks (hei x wid), with 1s where masks appear and 0s elsewhere
    
    Optional args:
        - col (str):
            ROI mask colour
            default: "orange"
        - alpha (num): 
            plt alpha variable controlling transparency of ROI masks 
            (from 0 to 1)
            default: 1.0
        - background (str):
            background colour for the masks
            default: "white"
        - transparent (bool):
            if True, background is made transparent. Note that some artifactual 
            dots can appear, so it is still good to choose the best background 
            colour
            default: True
        - lighten (num):
            plt alpha variable controlling transparency of white copy of ROI 
            masks (allows ROI masks to be lightened if they appear on a black 
            background)
            default: 0
        - mark_crop_area (2D array):
            if provided, values to use in marking crop area rectangle, 
            [[start_x, len_y], [start_y, len_y]]
            default = None
    """

    colors = [col]
    all_alphas = [alpha]
    if lighten != 0:
        colors = [col, "white"]
        all_alphas = [0.6, lighten]

    for col, alpha in zip(colors, all_alphas):

        alphas = np.ones(N_LEVELS_MASKS + 3) * alpha

        if transparent:
            alphas[0] = 0
        else:
            alphas[0] = 1

        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "mask_cmap", 
            [background, col], 
            N=N_LEVELS_MASKS
            )

        cmap._init()
        cmap._lut[:, -1] = alphas

        sub_ax.imshow(roi_masks, cmap=cmap)
    
    if mark_crop_area is not None:
        mark_crop_area = np.asarray(mark_crop_area)
        if mark_crop_area.shape != (2, 2):
            raise ValueError("'mark_crop_area' must have shape (2, 2).")
        
        len_x, len_y = mark_crop_area[:, 1][::-1]
        start_y, start_x = mark_crop_area[:, 0]
        rect = mpatches.Rectangle(
            (start_x, start_y), len_x, len_y, lw=2, ls=(3, (3, 3)), 
            edgecolor="k", facecolor="none"
            )
        sub_ax.add_patch(rect)


#############################################
def add_scale_marker(sub_ax, side_len=512, ori="horizontal", quadrant=1, 
                     fontsize=16):
    """
    add_scale_marker(sub_ax)

    Adds a scale marker and length in um to the subplot.

    Required args:
        - sub_ax (plt Axis subplot): 
            subplot

    Optional args:
        - side_len (int):
            length in pixels of the subplot side 
            (x axis if ori is "horizontal", and y axis if ori is "vertical")
            default: 512
        - ori (str):
            scale marker orientation ("horizontal" or "vertical")
            default: "horizontal"
        - quadrant (int):
            subplot quadrant in the corner of which to plot scale marker
            default: 1
        - fontsize (int):
            font size for scale length text
            default: 16
    """

    side_len_um = side_len * UM_PER_PIX
    half_len_um = side_len_um / 2
    
    if half_len_um >= 25:
        i = np.log(half_len_um / 25) // np.log(2)
        bar_len_um = int(25 * (2 ** i))
    else:
        i = np.log(half_len_um) / np.log(2)
        bar_len_um = 2 ** i
        if i >= 1:
            bar_len_um = int(bar_len_um)

    line_kwargs = {
        "color"         : "black",
        "lw"            : 4,
        "solid_capstyle": "butt",
    }

    if quadrant not in [1, 2, 3, 4]:
        gen_util.accepted_values_error("quadrant", quadrant, [1, 2, 3, 4])
    
    text_va = "center"
    if ori == "horizontal":
        axis_width_pts = sub_ax.get_window_extent().width

        sub_ax.set_xlim([0, side_len_um])
        sub_ax.set_ylim([0, 1])

        if quadrant in [1, 2]: # top
            y_coord = 0.95
            text_y = 0.8
        else:
            y_coord = 0.05
            text_y = 0.2
        
        if quadrant in [1, 4]: # right
            spine_width = sub_ax.spines["right"].get_linewidth()
            adj_um = spine_width / axis_width_pts * side_len_um
            xs = [side_len_um - bar_len_um - adj_um, side_len_um - adj_um]
            text_x = xs[-1]
            text_ha = "right"
        else:
            spine_width = sub_ax.spines["left"].get_linewidth()
            adj_um = spine_width / axis_width_pts * side_len_um
            xs = [adj_um, bar_len_um + adj_um]
            text_x = xs[0]
            text_ha = "left"

        sub_ax.plot(xs, [y_coord, y_coord], **line_kwargs)

    elif ori == "vertical":
        axis_height_pts = sub_ax.get_window_extent().height

        sub_ax.set_ylim([0, side_len_um])
        sub_ax.set_xlim([0, 1])

        if quadrant in [1, 2]: # top
            spine_height = sub_ax.spines["top"].get_linewidth()
            adj_um = spine_height / axis_height_pts * side_len_um
            ys = [side_len_um - bar_len_um - adj_um, side_len_um - adj_um]
            text_y = ys[-1]
            text_va = "top"
        else:
            spine_height = sub_ax.spines["bottom"].get_linewidth()
            adj_um = spine_height / axis_height_pts * side_len_um
            ys = [adj_um, bar_len_um + adj_um]
            text_y = ys[0]
            text_va = "bottom"
        
        if quadrant in [1, 4]: # right
            x_coord = 0.95
            text_x = 0.85
            text_ha = "right"
        else:
            x_coord = 0.05
            text_x = 0.1
            text_ha = "left"

        sub_ax.plot([x_coord, x_coord], ys, **line_kwargs)

    else:
        gen_util.accepted_values_error("ori", ori, ["horizontal", "vertical"])

    mu = u"\u03BC"
    sub_ax.text(
        text_x, text_y, r"{} {}m".format(bar_len_um, mu), 
        ha=text_ha, va=text_va, fontsize=fontsize, fontweight="bold",
        )

    plot_util.remove_axis_marks(sub_ax)

    return


#############################################
def plot_imaging_planes(imaging_plane_df, figpar, title=None):
    """
    plot_imaging_planes(imaging_plane_df, figpar)
    
    Plots imaging planes.

    Required args:
        - imaging_plane_df (pd.DataFrame in dict format):
            dataframe with a row for each mouse, and the following 
            columns, in addition to the basic sess_df columns: 
            - "max_projections" (list): pixel intensities of maximum projection 
                for the plane (hei x wid)
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  

    Optional args:
        - title (str):
            plot title
            default: None

    Returns:
        - ax (2D array): 
            array of subplots
    """

    figpar = plot_util.fig_init_linpla(figpar)

    figpar["init"]["sharex"] = False
    figpar["init"]["sharey"] = False
    figpar["init"]["subplot_hei"] = 3.4
    figpar["init"]["subplot_wid"] = 3.4
    figpar["init"]["gs"] = {"wspace": 0.2, "hspace": 0.15}

    # MUST ADJUST if anything above changes [right, bottom, width, height]
    new_axis_coords = [0.91, 0.115, 0.15, 0.34]

    fig, ax = plot_util.init_fig(plot_helper_fcts.N_LINPLA, **figpar["init"])

    sub_ax_scale = fig.add_axes(new_axis_coords)
    plot_util.remove_axis_marks(sub_ax_scale)
    sub_ax_scale.spines["left"].set_visible(True)

    if title is not None:
        fig.suptitle(title, y=1, weight="bold")

    hei_lens = []
    raster_zorder = -12
    for (line, plane), lp_mask_df in imaging_plane_df.groupby(["lines", "planes"]):
        li, pl, _, _ = plot_helper_fcts.get_line_plane_idxs(line, plane)
        sub_ax = ax[pl, li]

        if len(lp_mask_df) != 1:
            raise RuntimeError("Expected only one row per line/plane.")
        lp_row = lp_mask_df.loc[lp_mask_df.index[0]]
        
        # add projection
        imaging_plane = np.asarray(lp_row["max_projections"])
        hei_lens.append(imaging_plane.shape[0])
        add_imaging_plane(sub_ax, imaging_plane, alpha=0.98, 
            zorder=raster_zorder - 1
            )
        
    # add scale marker
    hei_lens = np.unique(hei_lens)
    if len(hei_lens) != 1:
        raise NotImplementedError(
            "Adding scale bar not implemented if ROI mask image heights are "
            "different for different planes."
            )
    add_scale_marker(
        sub_ax_scale, side_len=hei_lens[0], ori="vertical", quadrant=3, 
        fontsize=16
        )
 
    logger.info("Rasterizing imaging plane images...", extra={"spacing": TAB})
    for sub_ax in ax.reshape(-1):
        sub_ax.set_rasterization_zorder(raster_zorder)

    # Add plane, line info to plots
    plot_util.format_linpla_subaxes(ax, ylab="", kind="map")
    for sub_ax in ax.reshape(-1):
        plot_util.remove_axis_marks(sub_ax)

    return ax


#############################################
def add_proj_and_roi_masks(ax_grp, df_row, sess_cols, crop=False, alpha=0.6, 
                           proj_zorder=-13):
    """
    add_proj_and_roi_masks(ax_grp, df_row, sess_cols)

    Adds projections and ROI masks to a group of subplots.

    Required args:
        - ax_grp (2D array): group of axes (2 x n_sess)
        - df_row (pd Series):
            pandas series with the following keys:
            - "max_projections" (list): pixel intensities of maximum projection 
                for the plane (hei x wid)
            - "registered_roi_mask_idxs" (list): list of mask indices, 
                registered across sessions, for each session 
                (flattened across ROIs) ((sess, hei, wid) x val)
            - "roi_mask_idxs" (list): list of mask indices for each session, 
                and each ROI (sess x (ROI, hei, wid) x val) (not registered)
            - "roi_mask_shapes" (list): shape into which ROI mask indices index 
                (sess x hei x wid)

            - "crop_fact" (num): factor by which to crop masks (> 1) 
            - "shift_prop_hei" (float): proportion by which to shift cropped 
                mask center vertically from left edge [0, 1]
            - "shift_prop_wid" (float): proportion by which to shift cropped 
                mask center horizontally from left edge [0, 1]
        - sess_cols (dict):
            dictionary with session numbers (int) as keys, and colours as 
            values.
    
    Optional args:
        - crop (bool):
            if True, ROI mask image is cropped, per specifications in df_row.
            default: False
        - alpha (num): 
            plt alpha variable controlling transparency of patches (from 0 to 1)
            default: 0.6
        - proj_zorder (int):
            zorder for the imaging plane
            default: -12
    """

    n_sess = len(df_row["sess_ns"])

    reg_roi_masks = create_sess_roi_masks(
        df_row, mask_key="registered_roi_mask_idxs", crop=crop
        )

    imaging_planes = []
    for s, sess_n in enumerate(df_row["sess_ns"]):
        col = sess_cols[int(sess_n)]

        # individual subplot
        indiv_sub_ax = ax_grp[0, s]

        # add projection
        imaging_plane = np.asarray(df_row["max_projections"][s])
        if crop:
            imaging_plane, _ = crop_roi_image(df_row, imaging_plane)
        add_imaging_plane(
            indiv_sub_ax, imaging_plane, alpha=0.98, 
            zorder=proj_zorder
            )

        # add mask contours
        cw = np.max([1, int(np.ceil(imaging_plane.shape[0] / 170))])
        indiv_roi_masks = create_roi_mask_contours(
            df_row, mask_key="roi_mask_idxs", sess_idx=s, cw=cw, 
            outer=True, crop=crop, 
            )
        # black bckgrd to avoid artifacts
        add_roi_mask(
            indiv_sub_ax, indiv_roi_masks, col=col, alpha=1, 
            background="black", transparent=True, lighten=0.27
            )

        # add to shared subplot (center)
        shared_col = int((n_sess - 1) // 2)
        shared_sub_ax = ax_grp[1, shared_col]

        add_roi_mask(shared_sub_ax, reg_roi_masks[s], col=col, alpha=alpha)

        imaging_planes.append(imaging_plane)

    return imaging_planes


#############################################
def plot_roi_masks_overlayed_with_proj(roi_mask_df, figpar, title=None):
    """
    plot_roi_masks_overlayed_with_proj(roi_mask_df, figpar)

    Plots ROI mask contours overlayed over imaging planes, and ROI masks 
    overlayed over each other across sessions.

    Required args:
        - roi_mask_df (pd.DataFrame in dict format):
            dataframe with a row for each mouse, and the following 
            columns, in addition to the basic sess_df columns: 

            - "max_projections" (list): pixel intensities of maximum projection 
                for the plane (hei x wid)
            - "registered_roi_mask_idxs" (list): list of mask indices, 
                registered across sessions, for each session 
                (flattened across ROIs) ((sess, hei, wid) x val)
            - "roi_mask_idxs" (list): list of mask indices for each session, 
                and each ROI (sess x (ROI, hei, wid) x val) (not registered)
            - "roi_mask_shapes" (list): shape into which ROI mask indices index 
                (sess x hei x wid)

            - "crop_fact" (num): factor by which to crop masks (> 1) 
            - "shift_prop_hei" (float): proportion by which to shift cropped 
                mask center vertically from left edge [0, 1]
            - "shift_prop_wid" (float): proportion by which to shift cropped 
                mask center horizontally from left edge [0, 1]

        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  

    Optional args:
        - title (str):
            plot title
            default: None

    Returns:
        - ax (2D array): 
            array of subplots
    """

    n_lines = len(roi_mask_df["lines"].unique())
    n_planes = len(roi_mask_df["planes"].unique())

    sess_cols = get_sess_cols(roi_mask_df)
    n_sess = len(sess_cols)
    n_cols = n_sess * n_lines

    figpar = plot_util.fig_init_linpla(figpar)

    figpar["init"]["sharex"] = False
    figpar["init"]["sharey"] = False
    figpar["init"]["subplot_hei"] = 2.3
    figpar["init"]["subplot_wid"] = 2.3
    figpar["init"]["gs"] = {"wspace": 0.2, "hspace": 0.2}
    figpar["init"]["ncols"] = n_cols

    fig, ax = plot_util.init_fig(n_cols * n_planes * 2, **figpar["init"])

    if title is not None:
        fig.suptitle(title, y=0.93, weight="bold")

    crop = "crop_fact" in roi_mask_df.columns

    sess_cols = get_sess_cols(roi_mask_df)
    alpha = 0.6
    raster_zorder = -12

    for (line, plane), lp_mask_df in roi_mask_df.groupby(["lines", "planes"]):
        li, pl, _, _ = plot_helper_fcts.get_line_plane_idxs(line, plane)
        lp_col = plot_helper_fcts.get_line_plane_idxs(line, plane)[2]
        lp_name = plot_helper_fcts.get_line_plane_name(line, plane)

        if len(lp_mask_df) != 1:
            raise RuntimeError("Expected only one row per line/plane.")
        lp_row = lp_mask_df.loc[lp_mask_df.index[0]]

        # identify subplots
        base_row = (pl % n_planes) * n_planes
        base_col = (li % n_lines) * n_lines

        ax_grp = ax[base_row : base_row + 2, base_col : base_col + n_sess + 1]

        # add imaging planes and masks
        imaging_planes = add_proj_and_roi_masks(
            ax_grp, lp_row, sess_cols, crop=crop, alpha=alpha, 
            proj_zorder=raster_zorder - 1
            )

        # add markings
        shared_row = base_row + 1
        shared_col = base_col + int((n_sess - 1) // 2)
        shared_sub_ax = ax[shared_row, shared_col]

        if shared_col == 0:
            shared_sub_ax.set_ylabel(lp_name, fontweight="bold", color=lp_col)
        else:
            lp_sub_ax = ax[shared_row, 0]
            lp_sub_ax.set_xlim([0, 1])
            lp_sub_ax.set_ylim([0, 1])
            lp_sub_ax.text(
                0.5, 0.5, lp_name, fontweight="bold", color=lp_col, 
                ha="center", va="center", fontsize="x-large"
                )

        # add scale bar
        if n_sess < 2:
            raise NotImplementedError(
                "Scale bar placement not implemented for fewer than 2 "
                "sessions."
                )
        scale_ax = ax[shared_row, -1]
        wid_len = imaging_planes[0].shape[-1]
        add_scale_marker(
            scale_ax, side_len=wid_len, ori="horizontal", quadrant=1, 
            fontsize=16
            )

    logger.info("Rasterizing imaging plane images...", extra={"spacing": TAB})
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            sub_ax = ax[i, j]
            plot_util.remove_axis_marks(sub_ax)
            if not(i % 2):
                sub_ax.set_rasterization_zorder(raster_zorder)

    # add legend
    if n_sess < 2:
        raise NotImplementedError(
            "Legend placement not implemented for fewer than 2 sessions."
            )
    add_sess_col_leg(
        ax[-1, -1], sess_cols, bbox_to_anchor=(1, 0.6), alpha=alpha, 
        fontsize="small"
        )

    return ax


#############################################
def plot_roi_masks_overlayed(roi_mask_df, figpar, title=None, 
                             mark_crop_only=False):
    """
    plot_roi_masks_overlayed(roi_mask_df, figpar)

    Plots ROI masks overlayed across sessions, optionally cropped.

    Required args:
        - roi_mask_df (pd.DataFrame in dict format):
            dataframe with a row for each mouse, and the following 
            columns, in addition to the basic sess_df columns: 
            - "registered_roi_mask_idxs" (list): list of mask indices, 
                registered across sessions, for each session 
                (flattened across ROIs) ((sess, hei, wid) x val)
            - "roi_mask_shapes" (list): shape into which ROI mask indices index 
                (sess x hei x wid)
            
            and optionally, if cropping or marking cropping:
            - "crop_fact" (num): factor by which to crop masks (> 1) 
            - "shift_prop_hei" (float): proportion by which to shift cropped 
                mask center vertically from left edge [0, 1]
            - "shift_prop_wid" (float): proportion by which to shift cropped 
                mask center horizontally from left edge [0, 1]

        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  

    Optional args:
        - title (str):
            plot title
            default: None
        - mark_crop_only (bool):
            if True, cropping information is used to mark area, not to crop
            default: False

    Returns:
        - ax (2D array): 
            array of subplots
    """

    if "crop_fact" in roi_mask_df.columns:
        crop = not mark_crop_only
    else:
        crop, mark_crop_only = False, False

    figpar = plot_util.fig_init_linpla(figpar)

    figpar["init"]["sharex"] = False
    figpar["init"]["sharey"] = False
    figpar["init"]["subplot_hei"] = 5.2
    figpar["init"]["subplot_wid"] = 5.2
    figpar["init"]["gs"] = {"wspace": 0.03, "hspace": 0.32}

    # MUST ADJUST if anything above changes [right, bottom, width, height]
    new_axis_coords = [0.885, 0.11, 0.1, 0.33]
    if crop: # move to the left
        new_axis_coords[0] = 0.04

    fig, ax = plot_util.init_fig(plot_helper_fcts.N_LINPLA, **figpar["init"])

    sub_ax_scale = fig.add_axes(new_axis_coords)
    plot_util.remove_axis_marks(sub_ax_scale)
    sub_ax_scale.spines["left"].set_visible(True)

    if title is not None:
        fig.suptitle(title, y=0.95, weight="bold")

    sess_cols = get_sess_cols(roi_mask_df)
    alpha = 0.6
    hei_lens = []
    for (line, plane), lp_mask_df in roi_mask_df.groupby(["lines", "planes"]):
        li, pl, _, _ = plot_helper_fcts.get_line_plane_idxs(line, plane)
        sub_ax = ax[pl, li]

        if len(lp_mask_df) != 1:
            raise RuntimeError("Expected only one row per line/plane.")
        lp_row = lp_mask_df.loc[lp_mask_df.index[0]]
        
        outputs = create_sess_roi_masks(
            lp_row, crop=crop, get_crop_area_only=mark_crop_only
            )
 
        if mark_crop_only:
            roi_masks, crop_area = outputs
        else:
            roi_masks = outputs
            crop_area = None

        hei_lens.append(roi_masks.shape[1])

        for s, sess_n in enumerate(lp_row["sess_ns"]):
            col = sess_cols[int(sess_n)]
            add_roi_mask(
                sub_ax, roi_masks[s], col=col, alpha=alpha, 
                mark_crop_area=crop_area
                )
            

    # add legend
    add_sess_col_leg(
        ax[0, 1], sess_cols, bbox_to_anchor=(0.7, -0.01), alpha=alpha
        )

    # add scale marker
    hei_lens = np.unique(hei_lens)
    if len(hei_lens) != 1:
        raise NotImplementedError(
            "Adding scale bar not implemented if ROI mask image heights are "
            "different for different planes."
            )
    quadrant = 1 if crop else 3
    add_scale_marker(
        sub_ax_scale, side_len=hei_lens[0], ori="vertical", quadrant=quadrant, 
        fontsize=18
        )
 
    # Add plane, line info to plots
    plot_util.format_linpla_subaxes(ax, ylab="", kind="map")

    return ax


#############################################
def plot_roi_tracking(roi_mask_df, figpar, title=None):
    """
    plot_roi_tracking(roi_mask_df, figpar)
    
    Plots ROI tracking examples, for different session permutations, and union 
    across permutations.

    Required args:
        - roi_mask_df (pd.DataFrame in dict format):
            dataframe with a row for each mouse, and the following 
            columns, in addition to the basic sess_df columns: 
            - "roi_mask_shapes" (list): shape into which ROI mask indices index 
                (sess x hei x wid)
            - "union_n_conflicts" (int): number of conflicts after union
            for "union", "fewest" and "most" tracked ROIs:
            - "{}_registered_roi_mask_idxs" (list): list of mask indices, 
                registered across sessions, for each session 
                (flattened across ROIs) ((sess, hei, wid) x val),
                ordered by {}_sess_ns if "fewest" or "most"
            - "{}_n_tracked" (int): number of tracked ROIs
            for "fewest", "most" tracked ROIs:
            - "{}_sess_ns" (list): ordered session number 
        - figpar (dict): 
            dictionary containing the following figure parameter dictionaries
            ["init"] (dict): dictionary with figure initialization parameters
            ["save"] (dict): dictionary with figure saving parameters
            ["dirs"] (dict): dictionary with additional figure parameters  

    Optional args:
        - title (str):
            plot title
            default: None

    Returns:
        - ax (2D array): 
            array of subplots
    """

    if len(roi_mask_df) != 1:
        raise ValueError("Expected only one row in roi_mask_df")
    roi_mask_row = roi_mask_df.loc[roi_mask_df.index[0]]

    columns = ["fewest", "most", "", "union"]

    figpar["init"]["ncols"] = len(columns)
    figpar["init"]["sharex"] = False
    figpar["init"]["sharey"] = False
    figpar["init"]["subplot_hei"] = 5.05
    figpar["init"]["subplot_wid"] = 5.05
    figpar["init"]["gs"] = {"wspace": 0.06}

    # MUST ADJUST if anything above changes [right, bottom, width, height]
    new_axis_coords = [0.905, 0.125, 0.06, 0.74] 

    fig, ax = plot_util.init_fig(plot_helper_fcts.N_LINPLA, **figpar["init"])

    sub_ax_scale = fig.add_axes(new_axis_coords)
    plot_util.remove_axis_marks(sub_ax_scale)
    sub_ax_scale.spines["left"].set_visible(True)

    if title is not None:
        fig.suptitle(title, y=1.05, weight="bold")

    sess_cols = get_sess_cols(roi_mask_df)
    alpha = 0.6
    for c, column in enumerate(columns):
        sub_ax = ax[0, c]

        if c == 0:
            lp_col = plot_helper_fcts.get_line_plane_idxs(
                roi_mask_row["lines"], roi_mask_row["planes"]
                )[2]

            lp_name = plot_helper_fcts.get_line_plane_name(
                roi_mask_row["lines"], roi_mask_row["planes"]
                )
            sub_ax.set_ylabel(lp_name, fontweight="bold", color=lp_col)
            log_info = f"Conflicts and matches for a {lp_name} example:"

        if column == "":
            sub_ax.set_axis_off()
            subplot_title = \
                "     Union - conflicts\n...   =============>"
            sub_ax.set_title(subplot_title, fontweight="bold", y=0.5)
            continue
        else:
            plot_util.remove_axis_marks(sub_ax)
            for spine in ["right", "left", "top", "bottom"]:
                sub_ax.spines[spine].set_visible(True)

        if column in ["fewest", "most"]:
            y = 1.01
            ord_sess_ns = roi_mask_row[f"{column}_sess_ns"]
            ord_sess_ns_str = ", ".join([str(n) for n in ord_sess_ns])

            n_matches = int(roi_mask_row[f"{column}_n_tracked"])
            subplot_title = f"{n_matches} matches\n(sess {ord_sess_ns_str})"
            log_info = (f"{log_info}\n{TAB}"
                f"{column.capitalize()} matches (sess {ord_sess_ns_str}): "
                f"{n_matches}")
        
        elif column == "union":
            y = 1.04
            ord_sess_ns = roi_mask_row["sess_ns"]
            n_union = int(roi_mask_row[f"{column}_n_tracked"])
            n_conflicts = int(roi_mask_row[f"{column}_n_conflicts"])
            n_matches = n_union - n_conflicts

            subplot_title = f"{n_matches} matches"
            log_info = (f"{log_info}\n{TAB}"
                "Union - conflicts: "
                f"{n_union} - {n_conflicts} = {n_matches} matches"
                )

        sub_ax.set_title(subplot_title, fontweight="bold", y=y)

        roi_masks = create_sess_roi_masks(
            roi_mask_row, 
            mask_key=f"{column}_registered_roi_mask_idxs"
            )
        
        for sess_n in roi_mask_row["sess_ns"]:
            col = sess_cols[int(sess_n)]
            s = ord_sess_ns.index(sess_n)
            add_roi_mask(sub_ax, roi_masks[s], col=col, alpha=alpha)

    # add scale marker
    hei_len = roi_mask_row["roi_mask_shapes"][1]
    add_scale_marker(
        sub_ax_scale, side_len=hei_len, ori="vertical", quadrant=3, fontsize=12
        )

    logger.info(log_info, extra={"spacing": "\n"})

    # add legend
    add_sess_col_leg(
        ax[0, columns.index("")], 
        sess_cols, 
        bbox_to_anchor=(0.67, 0.3), 
        alpha=alpha
        )

    return ax


