"""
tracking_plots.py

This script contains functions for plotting tracked ROI masks.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

import numpy as np
import matplotlib as mpl

from util import logger_util, plot_util
from sess_util import sess_plot_util
from analysis import misc_analys
from plot_fcts import plot_helper_fcts

logger = logging.getLogger(__name__)

TAB = "    "
ORDERED_COLORS = ["orange", "crimson", "mediumblue"]
N_LEVELS = 2


#############################################
def plot_roi_masks_overlayed(roi_mask_df, figpar, title=None):
    """
    plot_roi_masks_overlayed(roi_mask_df, figpar)
    
    """

    figpar = sess_plot_util.fig_init_linpla(figpar)

    figpar["init"]["sharex"] = False
    figpar["init"]["sharey"] = False
    figpar["init"]["subplot_hei"] = 5.2
    figpar["init"]["subplot_wid"] = 5.2
    figpar["init"]["gs"] = {"wspace": 0.03, "hspace": 0.32}

    fig, ax = plot_util.init_fig(plot_helper_fcts.N_LINPLA, **figpar["init"])

    if title is not None:
        fig.suptitle(title, y=0.95, weight="bold")

    alphas = np.ones(N_LEVELS + 3) * 0.6
    alphas[0] = 0

    crop = "crop_fact" in roi_mask_df.columns

    leg_handles = [None for _ in range(len(ORDERED_COLORS))]
    for (line, plane), lp_mask_df in roi_mask_df.groupby(["lines", "planes"]):
        li, pl, _, _ = plot_helper_fcts.get_line_plane_idxs(line, plane)
        sub_ax = ax[pl, li]

        if len(lp_mask_df) != 1:
            raise RuntimeError("Expected only one row per line/plane.")
        lp_row = lp_mask_df.loc[lp_mask_df.index[0]]
        sess_ns = lp_row["sess_ns"]
        if np.max(sess_ns) > len(ORDERED_COLORS):
            raise RuntimeError(
                "Expected sessions to go no higher than "
                f"{len(ORDERED_COLORS)}."
                )

        roi_masks = np.zeros(lp_row["roi_mask_shapes"]).astype(int)
        idxs = [np.asarray(sub) for sub in lp_row["registered_roi_mask_idxs"]]
        roi_masks[tuple(idxs)] = 1 # sess x hei x wid

        if crop:
            crop_fact = lp_row["crop_fact"]
            if crop_fact < 1:
                raise ValueError("crop_fact must be at least 1")

            for d, dim in enumerate(["hei", "wid"]):
                shift_prop = lp_row[f"shift_prop_{dim}"]
                if shift_prop < 0 or shift_prop > 1:
                    raise RuntimeError("shift_prop must be between 0 and 1.")
                orig_size = roi_masks.shape[d + 1]
                new_size = int(np.around(orig_size / crop_fact))
                shift = int(shift_prop * (orig_size - new_size))
                if d == 0:
                    roi_masks = roi_masks[:, shift : shift + new_size]
                else:
                    roi_masks = roi_masks[:, :, shift : shift + new_size]
        
        for s, sess_n in enumerate(sess_ns):
            col_idx = sess_n - 1
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'mask_cmap',['white', ORDERED_COLORS[col_idx]], N=N_LEVELS)
            cmap._init()
            cmap._lut[:, -1] = alphas
            sub_ax.imshow(roi_masks[s, :], cmap=cmap)
            # to populate a single legend
            if leg_handles[col_idx] is None:
                leg_handles[col_idx] = mpl.patches.Patch(
                    color=ORDERED_COLORS[col_idx],
                    label=f"sess {sess_n}",
                    alpha=alphas[-1]
                    )

    ax[0, 1].legend(
        handles=leg_handles, 
        bbox_to_anchor=(0.7, -0.01), 
        fontsize="medium",
        borderpad=0.75
        )

    # Add plane, line info to plots
    sess_plot_util.format_linpla_subaxes(ax, ylab="", kind="map")

    return ax


#############################################
def plot_roi_tracking(roi_mask_df, figpar, title=None):
    """
    plot_roi_tracking(roi_mask_df, figpar)
    
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

    fig, ax = plot_util.init_fig(plot_helper_fcts.N_LINPLA, **figpar["init"])

    if title is not None:
        fig.suptitle(title, y=1.05, weight="bold")

    alphas = np.ones(N_LEVELS + 3) * 0.6
    alphas[0] = 0

    leg_handles = [None for _ in range(len(ORDERED_COLORS))]
    for c, col in enumerate(columns):
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

        if col == "":
            sub_ax.set_axis_off()
            subplot_title = "Union - conflicts\n==================>"
            sub_ax.set_title(subplot_title, fontweight="bold", y=0.5)
            continue
        else:
            plot_util.remove_axis_marks(sub_ax)
            for spine in ["right", "left", "top", "bottom"]:
                sub_ax.spines[spine].set_visible(True)

        n_matches = int(roi_mask_row[f"{col}_n_tracked"])
        subplot_title = f"{n_matches} matches"

        if col in ["fewest", "most"]:
            y = 1.01
            ord_sess_ns = roi_mask_row[f"{col}_sess_ns"]
            ord_sess_ns_str = ", ".join([str(n) for n in ord_sess_ns])
            subplot_title = f"{subplot_title}\n(sess {ord_sess_ns_str})"
            log_info = (f"{log_info}\n{TAB}"
                f"{col.capitalize()} matches (sess {ord_sess_ns_str}): "
                f"{n_matches}")
        elif col == "union":
            y = 1.04
            n_conflicts = int(roi_mask_row[f"{col}_n_conflicts"])
            log_info = (f"{log_info}\n{TAB}"
                "Union - conflicts: "
                f"{n_matches + n_conflicts} - {n_conflicts} = "
                f"{n_matches} matches")

        sub_ax.set_title(subplot_title, fontweight="bold", y=y)

        sess_ns = roi_mask_row["sess_ns"]
        if np.max(sess_ns) > len(ORDERED_COLORS):
            raise RuntimeError(
                "Expected sessions to go no higher than "
                f"{len(ORDERED_COLORS)}."
                )

        roi_masks = np.zeros(roi_mask_row["roi_mask_shapes"]).astype(int)
        idxs = roi_mask_row[f"{col}_registered_roi_mask_idxs"]
        roi_masks[tuple(idxs)] = 1 # sess x hei x wid
        
        for s, sess_n in enumerate(sess_ns):
            col_idx = sess_n - 1
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'mask_cmap',['white', ORDERED_COLORS[col_idx]], N=N_LEVELS)
            cmap._init()
            cmap._lut[:, -1] = alphas
            sub_ax.imshow(roi_masks[s, :], cmap=cmap)
            # to populate a single legend
            if leg_handles[col_idx] is None:
                leg_handles[col_idx] = mpl.patches.Patch(
                    color=ORDERED_COLORS[col_idx],
                    label=f"sess {sess_n}",
                    alpha=alphas[-1]
                    )

    logger.info(log_info, extra={"spacing": "\n"})

    ax[0, columns.index("")].legend(
        handles=leg_handles, 
        bbox_to_anchor=(0.67, 0.3), 
        fontsize="medium",
        borderpad=0.75
        )

    return ax

