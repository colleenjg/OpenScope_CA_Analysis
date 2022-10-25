#!/usr/bin/env python

from pathlib import Path

import numpy as np
import pandas as pd

from util import gen_util, logger_util, plot_util

gen_util.extend_sys_path(Path("").resolve(), parents=1)
from analysis import session

logger = logger_util.get_module_logger(name=__name__)


DFT_DATASET_PATH = Path("..", "..", "datasets", "osca")
TAB = "    "

CONDITIONS = ["A", "B", "C", "D/U", "G"]
CONDITION_MAPPING = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "U": 4,
    "G": 5
}


#############################################
def get_mouse_df():
    """
    get_mouse_df()

    Returns:
        mouse_df: dataframe specifying session information
    """

    mouse_df_path = Path("..", "mouse_df.csv")
    mouse_df = pd.read_csv(mouse_df_path)
    return mouse_df


#############################################
def sess_gabor_seg_info(sess):
    """
    sess_gabor_seg_info(sess)

    Args:
        sess: the session object

    Returns:
        seg_dict: dictionary with segment information for each trial
            'gabor_frame': an array specifying Gabor frames (n_trial x n_segs)
            'gabor_mean_orientation': an array specifying Gabor orientations    
                                      (n_trial x n_segs)
            'unexpected': an array specifying whether a segment is part of an 
                          expected or unexpected sequence (n_trial x n_segs)
    """

    logger.info(f"Loading data for Gabor stimulus...")

    sub_df = sess.stim_df.loc[sess.stim_df["stimulus_type"] == "gabors"]

    if len(sub_df) / 5 != len(sub_df) // 5:
        raise NotImplementedError(
            "Expected the Gabor sequence segments to be exactly divisible by 5."
            )
    num_segs = len(sub_df) // 5 * 5
    sub_df = sub_df[:int(num_segs)]


    seg_dict = {
        "segments": sub_df.index.to_numpy().reshape(-1, 5).astype(int)
    }

    keys = ["gabor_frame", "gabor_mean_orientation", "unexpected"]

    for key in keys:
        data = sub_df[key]
        if key == "gabor_frame":
            dtype = int
            data = data.map(CONDITION_MAPPING)
        elif key == "unexpected":
            dtype = bool
        else:
            dtype = float

        seg_dict[key] = data.to_numpy().reshape(-1, 5).astype(dtype)

    return seg_dict


#############################################
def get_calcium_traces(mouse_n=None, sess_n=None, sessid=None, scale=False, 
                       tracked=True, dataset_path=DFT_DATASET_PATH):
    """
    get_calcium_traces()

    Args:
        mouse_n: int specifying the mouse number (typically 1 to 13)
        sess_n: int specifying the session number (typically 1 to 3)
        sessid: ID of a specific session if not None
        scale: whether the calcium traces by normalized
        tracked: whether to include only tracked ROIs, in tracking order
        dataset_path: path to the dataset

    Returns:
        sess: the session object
        roi_indices: an array containing the indices of the different ROIs
        roi_timestamps: an array containing the relative times for each segment
        roi_data: the calcium traces as a tensor of dim 
                  (# trials x # segs x # frames x # ROIs)
        gabor_seg_dict: dictionary with segment information for each trial
            'gabor_frame': an array specifying Gabor frames (n_trial x n_segs)
            'gabor_mean_orientation': an array specifying Gabor orientations    
                                      (n_trial x n_segs)
            'unexpected': an array specifying whether a segment is part of an 
                          expected or unexpected sequence (n_trial x n_segs)

    """

    mouse_df = get_mouse_df()
    if sessid is not None:
        assert mouse_n is None and sess_n is None
        sess_kwargs = {"sessid": sessid}
    else:
        assert mouse_n is not None and sess_n is not None
        sess_kwargs = {
            "mouse_n": mouse_n,
            "sess_n" : sess_n
            }
    sess = session.Session(
        runtype="prod", datadir=dataset_path, mouse_df=mouse_df, 
        only_tracked_rois=tracked, **sess_kwargs)
    
    logger.info(
        f"Session: M{sess.mouse_n} S{sess.sess_n} ({sess.line} {sess.plane})", 
        extra={"spacing": f"\n{TAB}"}
        )

    sess.extract_info(full_table=False, roi=True, run=False, pupil=False)

    gabor_seg_dict = sess_gabor_seg_info(sess)

    pre, post = 0, 0.3
    twop_fr_ns = sess.gabors.get_fr_by_seg(
        gabor_seg_dict["segments"].reshape(-1), start=True, fr_type="twop"
        )["start_frame_twop"]
    roi_data_df = sess.gabors.get_roi_data(twop_fr_ns, pre, post, scale=scale)

    roi_indices = roi_data_df.index.unique("ROIs")
    roi_timestamps = roi_data_df.index.unique("time_values")
    roi_data = gen_util.reshape_df_data(roi_data_df, squeeze_cols=True)
    roi_data = np.transpose(roi_data,[1, 2, 0])

    _, n_per = gabor_seg_dict.pop("segments").shape
    n_segs, n_frames, n_rois = roi_data.shape

    n_segs_keep = int(n_segs // n_per * n_per)
    n_seqs_keep = int(n_segs_keep // n_per)

    roi_data = roi_data[:n_segs_keep].reshape(
        [n_seqs_keep, n_per, n_frames, n_rois]
        )
    for key in list(gabor_seg_dict.keys()):
        gabor_seg_dict[key] = gabor_seg_dict[key][:n_seqs_keep]

    return sess, roi_indices, roi_timestamps, roi_data, gabor_seg_dict


#############################################
if __name__ == "__main__":
    
    logger_util.format_all(level='info')
    plot_util.linclab_plt_defaults()
    import matplotlib.pyplot as plt

    sess, roi_indices, roi_time_values, roi_data, gabor_seg_dict = \
        get_calcium_traces(mouse_n=1, sess_n=1)

    I = 4
    i0 = 30
    J = roi_data.shape[1]
    K = 5
    k0 = 30
    incr = 0.6
    fig, ax = plt.subplots(I, J, sharey=True, sharex=True, figsize=[7.5, 7.5])
    for i in range(I):
        for j in range(J):
            for k in range(K):
                ax[i, j].plot(
                    roi_time_values, roi_data[i + i0, j, :, k + k0] + k * incr, 
                    alpha=0.5, 
                    )
            ax[i, j].set_xticks([])
            if j == 0: 
                ax[i, j].set_ylabel(f"ROI {i + i0}")
            if i == I - 1: 
                ax[i, j].set_xlabel(CONDITIONS[j])

    fig.suptitle(f"Sequences {k0} to {k0 + K}")

    plt.show()

