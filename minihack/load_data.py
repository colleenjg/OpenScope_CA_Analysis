
import numpy as np

from pathlib import Path
import pandas as pd
from util import gen_util
from analysis import session

gen_util.extend_sys_path(Path("").resolve(), parents=1)



def get_mouse_df():
    mouse_df_path = Path("..", "mouse_df.csv")
    mouse_df = pd.read_csv(mouse_df_path)
    return mouse_df


def sess_to_arrays(sess):
    a_indices = sess.stim_df['gabor_frame'] == 'A'
    all_indices = sess.stim_df['stimulus_type'] == "gabors"
    a_sess = sess.stim_df[a_indices]

    is_unexpected_list = []
    mean_orientations_list = []
    twop_frame_starts = []
    condition_id_list = []
    stim_frame_starts = []

    num_frames_twop = sess.stim_df[all_indices]['num_frames_twop'].min()
    num_frames_stim = sess.stim_df[all_indices]['num_frames_stim'].min()
    condition_letter_to_id = ['A', 'B', 'C', 'D', 'U', 'G']

    for index in a_sess.index:
        seg_indices = np.arange(5) + index
        condition_ids = []
        for i in range(5):
            seg = sess.stim_df.iloc[int(index+i)]
            condition_letter = seg['gabor_frame']
            condition_ids += [int(condition_letter_to_id.index(condition_letter))]

        assert len(condition_ids) == 5
        condition_id_list += [np.array(condition_ids, dtype=int)]
        mean_orientations_list += [sess.stim_df['gabor_mean_orientation'][seg_indices]]
        is_unexpected_list += [sess.stim_df["unexpected"][index]]
        twop_frame_starts += [sess.stim_df["start_frame_twop"][seg_indices]]
        stim_frame_starts += [sess.stim_df["start_frame_stim"][seg_indices]]

    stim_frame_starts = np.array(stim_frame_starts, dtype=int)
    twop_frame_starts = np.array(twop_frame_starts, dtype=int)
    condition_ids = np.stack(condition_id_list, 0)
    mean_orientations = np.array(mean_orientations_list, dtype=float)
    unexpected = np.array(is_unexpected_list, dtype=bool)

    return (stim_frame_starts, num_frames_stim), (twop_frame_starts, num_frames_twop), condition_ids, mean_orientations, unexpected


def get_calcium_traces(mouse_n=None, sess_n=None, sessid=None, scale=False, dataset_path ="../../datasets/osca/"):
    """

    Args:
        mouse_n: int containing the mouse number (typically 1 to 12)
        sess_n: int containing the session number (typically 1 to 3)
        sessid: direct access of a specific session if is it not None
        scale: Should the calcium traces by normalized?

    Returns:
        sess: the session object
        roi_indices: an array containing the indices of the different ROIs
        roi_time_line: an array containing the absolute times in the time series
        roi_data: the calcium traces as a tensor of dim (n_tridl x n_segments x n_time x n_rois)
        condition_ids:

    """
    print("Extracting calcium data...")
    mouse_df = get_mouse_df()
    if sessid is not None:
        assert mouse_n is None and sess_n is None
        sess = session.Session(sessid=sessid, runtype="prod", datadir=dataset_path, mouse_df=mouse_df)
    else:
        assert mouse_n is not None and sess_n is not None
        sess = session.Session(mouse_n=mouse_n, sess_n=sess_n, runtype="prod", datadir=dataset_path, mouse_df=mouse_df)
    sess.set_only_tracked_rois(True)
    sess.extract_info(full_table=False, roi=True, run=True, pupil=True)

    print(sess.gabors)

    _, (twop_frame_starts, num_frames_twop), condition_ids, mean_orientations, unexpected = sess_to_arrays(sess)

    pre = 0.
    post = num_frames_twop * 1/30.
    twop_fr_ns = twop_frame_starts.flatten()
    #twop_fr_ns = sess.gabors.get_fr_by_seg(gab_seg_ns, start=True, ch_fl=[pre, post], fr_type="twop")["start_frame_twop"]
    roi_data_df = sess.gabors.get_roi_data(twop_fr_ns, pre, post, scale=scale)

    roi_indices = roi_data_df.index.unique("ROIs")
    roi_time_line = roi_data_df.index.unique("time_values")
    roi_data = gen_util.reshape_df_data(roi_data_df, squeeze_cols=True)
    roi_data = np.transpose(roi_data,[1, 2, 0])

    n_segs, n_time, n_rois = roi_data.shape
    assert len(roi_indices) == n_rois
    assert len(roi_time_line) == n_time
    n_trials, n_seg_per_trial = twop_frame_starts.shape
    roi_data = roi_data.reshape([n_trials, n_seg_per_trial, n_time, n_rois])

    return sess, roi_indices, roi_time_line, roi_data, condition_ids, mean_orientations, unexpected


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sess_ns = [1, 2, 3]
    datadir = DATAPATH

    mouse_df_path = Path("..", "mouse_df.csv")
    mouse_df = pd.read_csv(mouse_df_path)

    PLANES = "all"

    # sess = session.Session(760260459, datadir=datadir, mouse_df=mouse_df)

    sess = session.Session(mouse_n=4, sess_n=1, runtype="prod", datadir=datadir, mouse_df=mouse_df)
    sess.set_only_tracked_rois(True)
    sess.extract_info(full_table=False, roi=True, run=True, pupil=True)

    print(sess.gabors)

    #gab_seg_ns = sess.gabors.get_segs_by_criteria(gabk=16, gabfr=3, unexp=1, by="seg")
    gab_seg_ns = sess.gabors.get_segs_by_criteria(gabk="any", gabfr="any", unexp="any", by="seg")
    print(gab_seg_ns)

    stim_params = sess.gabors.get_stim_par_by_seg(gab_seg_ns)
    print(stim_params.keys())

    pre = 1.0
    post = 1.0
    twop_fr_ns = sess.gabors.get_fr_by_seg(gab_seg_ns, start=True, ch_fl=[pre, post], fr_type="twop")[
        "start_frame_twop"]
    stim_fr_ns = sess.gabors.get_fr_by_seg(gab_seg_ns, start=True, ch_fl=[pre, post], fr_type="stim")[
        "start_frame_stim"]

    roi_data_df = sess.gabors.get_roi_data(twop_fr_ns, pre, post, scale=True)

    roi_indices = roi_data_df.index.unique("ROIs")
    print("ROI indices ", roi_indices)

    roi_time_line = roi_data_df.index.unique("time_values")
    roi_data = gen_util.reshape_df_data(roi_data_df, squeeze_cols=True)
    print("ROI data shape: {} ROIs x {} sequences x {} time values".format(*roi_data.shape))

    I = 4
    J = 4
    i0 = 30
    j0 = 30
    fig, ax_list = plt.subplots(I, J, sharey=True, sharex=True)
    for i in range(I):
        for j in range(J):
            ax_list[i, j].plot(roi_time_line, roi_data[i + i0, j + j0, :])
            ax_list[i, j].plot(roi_time_line, roi_data[i + i0, j + j0, :])
            if j ==0: ax_list[i, j].set_ylabel(f"ROI {i + i0}")
            if i==I-1: ax_list[i, j].set_xlabel(f"sequence {j + j0}")

    plt.show()

    # run_data_df = sess.gabors.get_run_data(stim_fr_ns, pre, post, scale=True)
    # pup_data_df = sess.gabors.get_pup_diam_data(twop_fr_ns, pre, post, scale=True)

    # run_time_line = run_data_df.index.unique("time_values")
    # pup_time_line = pup_data_df.index.unique("time_values")
    # run_data = gen_util.reshape_df_data(run_data_df, squeeze_cols=True)
    # pup_data = gen_util.reshape_df_data(pup_data_df, squeeze_cols=True)

    # print("Run data shape: {} sequences x {} time values".format(*run_data.shape))
    # print("Pup data shape: {} sequences x {} time values".format(*pup_data.shape))
    # fig, ax_list = plt.subplots(3, sharex=True)
    # ax_list[0].plot(roi_time_line, roi_data[1,0,:])
    # ax_list[1].plot(run_time_line, run_data[0,:])
    # ax_list[2].plot(pup_time_line, pup_data[0,:])
    # plt.show()

