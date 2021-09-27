"""
basic_analys.py

This script contains basic functions for data analysis.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging
from sess_util import sess_gen_util

import numpy as np

from util import logger_util, gen_util, math_util

logger = logging.getLogger(__name__)

TAB = "    "


#############################################
def get_frame_numbers(stim, refs, ch_fl=None, ref_type="segs", datatype="roi"):
    """
    get_frame_numbers(stim, refs)

    Returns frame numbers for the data type.

    Required args:
        - stim (Stim):
            Stimulus object
        - refs (1D array):
            Sequences references (either segments or frames, specified by 
            ref_type)

    Optional args:
        - ch_fl (list):
            flanks to check for discarding refs with insufficient flanks
            default: None
        - ref_type (str):
            type of references provided 
            ("segs", "twop_frs", "stim_frs", "pup_frs")
            default: "segs"
        - datatype (str):
            type of data to return ("roi", "run" or "pupil")
            default: "roi"

    Returns:
        - fr_ns (1D array):
            frame numbers
    """

    if datatype not in ["roi", "run", "pupil"]:
        gen_util.accepted_values_error(
            "datatype", datatype, ["roi", "run", "pupil"]
            )

    # convert frames to correct type for ROI or pupil data
    ch_fl = [0, 0] if ch_fl is None else ch_fl
    if ref_type == "segs":
        if datatype == "run":
            fr_ns = stim.get_stim_fr_by_seg(
                refs, first=True, ch_fl=ch_fl)["first_stim_fr"]
        elif datatype == "pupil":
            fr_ns = stim.get_twop_fr_by_seg(refs, first=True)["first_twop_fr"]
            fr_ns = stim.sess.get_pup_fr_by_twop_fr(fr_ns, ch_fl=ch_fl)
        elif datatype == "roi":
            fr_ns = stim.get_twop_fr_by_seg(
                refs, first=True, ch_fl=ch_fl)["first_twop_fr"]

        if len(fr_ns) == 0:
            raise RuntimeError("No frames found given flank requirements.")
    
    elif ref_type == "stim_frs":
        if np.max(refs) >= stim.sess.tot_stim_fr: 
            raise ValueError("Some refs values are out of bounds.")
        elif np.min(refs) < 0:
            raise ValueError("refs cannot include negative values.")
        if datatype == "run":
            fr_ns = stim.sess.check_flanks(refs, ch_fl, fr_type="stim")
        else:
            fr_ns = stim.sess.stim2twopfr[np.asarray(refs)]
            if datatype == "pupil":
                fr_ns = stim.sess.get_pup_fr_by_twop_fr(fr_ns, ch_fl=ch_fl)
            elif datatype == "roi":
                fr_ns = stim.sess.check_flanks(fr_ns, ch_fl, fr_type="twop")
    
    elif ref_type == "twop_frs":
        if datatype == "run":
            raise NotImplementedError(
                "Converting twop_frs to stim_frs for running data is not "
                "implemented."
                )
        elif datatype == "pupil":
            fr_ns = stim.sess.get_pup_fr_by_twop_fr(refs, ch_fl=ch_fl)
        elif datatype == "roi":
            fr_ns = stim.sess.check_flanks(refs, ch_fl, fr_type="twop")

    elif ref_type == "pup_frs":
        if datatype != "pupil":
            raise NotImplementedError(
                "'pup_frs' ref_type only implemented for 'pupil' datatype."
                )
        fr_ns = stim.sess.check_flanks(refs, ch_fl, fr_type="pup")

    else:
        gen_util.accepted_values_error(
            "ref_type", ref_type, ["segs", "twop_frs", "stim_frs", "pup_frs"]
            )

    fr_ns = np.asarray(fr_ns)

    return fr_ns


#############################################
def get_data(stim, refs, analyspar, pre=0, post=1, ch_fl=None, integ=False,
             ref_type="segs", datatype="roi"):
    """
    get_data(stim, refs, analyspar)

    Returns data for a specific stimulus around sequence references provided.

    Required args:
        - stim (Stim):
            Stimulus object
        - refs (1D array):
            Sequences references (either segments or frames, specified by 
            ref_type)
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters

    Optional args:
        - pre (num): 
            number of seconds to keep before refs
            default: 0
        - post (num): 
            number of seconds to keep after refs
            default: 1
        - ch_fl (list):
            flanks to check for discarding refs with insufficient flanks
            default: None
        - integ (bool):
            if True, sequence data is integrated
            default: False
        - ref_type (str):
            type of references provided 
            ("segs", "twop_frs", "stim_frs", "pup_frs")
            default: "segs"
        - datatype (str):
            type of data to return ("roi", "run" or "pupil")
            default: "roi"

    Returns:
        - data_arr (1-3D array):
            sequence data array
            dims: (ROIs x) seq (x frames)
        - time_values (1D array):
            values for each frame, in seconds
    """
    
    if stim.sess.only_matched_rois != analyspar.tracked:
        raise RuntimeError(
            "stim.sess.only_matched_rois should match analyspar.tracked."
            )

    fr_ns = get_frame_numbers(
        stim, 
        refs, 
        ch_fl=ch_fl, 
        ref_type=ref_type, 
        datatype=datatype
        )

    # obtain data
    if datatype == "roi":
        data_df = stim.get_roi_data(
            fr_ns, pre, post, remnans=analyspar.remnans, scale=analyspar.scale
            )
        col_name = "roi_traces"
        integ_dt = stim.sess.twop_fps
    elif datatype == "run":
        data_df = stim.get_run_data(
            fr_ns, pre, post, remnans=analyspar.remnans, scale=analyspar.scale
        )
        col_name = "run_velocity"
        integ_dt = stim.sess.stim_fps
    elif datatype == "pupil":
        data_df = stim.get_pup_diam_data(
            fr_ns, pre, post, remnans=analyspar.remnans, scale=analyspar.scale
        )
        col_name = "pup_diam"
        integ_dt = stim.sess.pup_fps
    else:
        gen_util.accepted_values_error(
            "datatype", datatype, ["roi", "run", "pupil"]
            )
    
    time_values = data_df.index.unique("time_values").to_numpy()

    data_arr = gen_util.reshape_df_data(data_df[col_name], squeeze_cols=True)

    if integ:
        nanpol = None if analyspar.remnans else "omit"
        data_arr = math_util.integ(
            data_arr, 1. / integ_dt, axis=-1, nanpol=nanpol
            )
    
    return data_arr, time_values
    

#############################################
def get_common_oris(stimpar, split="by_exp"):
    """
    get_common_oris(stimpar)

    Returns Gabor orientations for common orientations, and checks parameters. 

    Required args:
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - split (str): 
            how to split data:
            "by_exp" (all exp, all unexp)
            default: "by_exp"

    Returns:
        - gab_oris (list):
            Gabor orientations for [exp, unexp] sequences, respectively
    """

    if split != "by_exp":
        raise NotImplementedError("'common_oris' only implemented "
            "with 'split' set to 'by_exp'.")
    if stimpar.stimtype != "gabors":
        raise ValueError("Exp/unexp index analysis with common "
            "orientations can only be run on Gabors.")

    if (isinstance(stimpar.gab_ori, list) and (len(stimpar.gab_ori) == 2) 
        and isinstance(stimpar.gab_ori[0], list) 
        and isinstance(stimpar.gab_ori[1], list)):
        gab_oris = stimpar.gab_ori

    else:
        gab_oris = sess_gen_util.get_params(gab_ori=stimpar.gab_ori)
        gab_oris = sess_gen_util.gab_oris_common_U("D", "all")

    return gab_oris


#############################################
def get_by_exp_data(sess, analyspar, stimpar, integ=False, common_oris=False, 
                    datatype="roi"):
    """
    get_by_exp_data(sess, analyspar, stimpar)

    Returns data split into expected and unexpected sequences.

    Required args:
        - sess (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - integ (bool)
            if True, sequence data is integrated
            default: False
        - common_oris (bool): 
            if True, only Gabor stimulus orientations common to D and U frames 
            are included ("by_exp" split only)
            default: False
        - datatype (str):
            type of data to return ("roi", "run" or "pupil")
            default: "roi"

    Returns:
        - data_arr (nested list):
            sequence data array
            dims: split (x ROIs) x seq (x frames)
        - time_values (1D array):
            values for each frame, in seconds
    """

    stim = sess.get_stim(stimpar.stimtype)

    gab_oris = [stimpar.gab_ori] * 2
    if common_oris:
        gab_oris = get_common_oris(stimpar, split="by_exp")

    by_exp_data = []
    for e, exp in enumerate([0, 1]):
        
        segs = stim.get_segs_by_criteria(
            gabfr=stimpar.gabfr, gabk=stimpar.gabk, gab_ori=gab_oris[e],
            bri_dir=stimpar.bri_dir, bri_size=stimpar.bri_size, surp=exp, 
            remconsec=False, by="seg")

        data, time_values = get_data(
            stim, segs, analyspar, pre=stimpar.pre, post=stimpar.post, 
            integ=integ, datatype=datatype, ref_type="segs"
            )
        by_exp_data.append(data.tolist())

    return by_exp_data, time_values


#############################################
def get_locked_data(sess, analyspar, stimpar, split="unexp_lock", integ=False, 
                    datatype="roi"):
    """
    get_locked_data(sess, analyspar, stimpar)

    Returns data locked to unexpected sequence onset or expected sequence onset.

    Required args:
        - sess (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - split (str): 
            how to split data:
            "unexp_lock" (unexp, preceeding exp), 
            "exp_lock" (exp, preceeding unexp),
            default: "unexp_lock"
        - integ (bool)
            if True, sequence data is integrated
            default: False
        - datatype (str):
            type of data to return ("roi", "run" or "pupil")
            default: "roi"

    Returns:
        - data_arr (nested list):
            sequence data array
            dims: split (x ROIs) x seq (x frames)
        - time_values (1D array):
            values for each frame, in seconds (for 0 to stimpar.post)
    """

    if split not in ["unexp_lock", "exp_lock"]:
        gen_util.accepted_values_error(
            "split", split, ["unexp_lock", "exp_lock"])

    stim = sess.get_stim(stimpar.stimtype)

    exp = 1 if split == "unexp_lock" else 0

    locked_data = []
    for i in range(2):

        segs = stim.get_segs_by_criteria(
            gabfr=stimpar.gabfr, gabk=stimpar.gabk, gab_ori=stimpar.gab_ori,
            bri_dir=stimpar.bri_dir, bri_size=stimpar.bri_size, surp=exp, 
            remconsec=True, by="seg")

        if i == 0:
            pre, post = [stimpar.pre, 0]
        else:
            pre, post = [0, stimpar.post]

        data, time_values = get_data(
            stim, segs, analyspar, pre=pre, post=post, 
            ch_fl=[stimpar.pre, stimpar.post], integ=integ, datatype=datatype,
            ref_type="segs",
            )
        
        locked_data.append(data.tolist())

    return locked_data, time_values


#############################################
def get_stim_on_off_data(sess, analyspar, stimpar, split="stim_onset", 
                         integ=False, datatype="roi"):
    """
    get_stim_on_off_data(sess, analyspar, stimpar)

    Returns data locked to stimulus onset or stimulus offset.

    Required args:
        - sess (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - split (str): 
            how to split data:
            "stim_onset" (grayscr, stim on), 
            "stim_offset" (stim off, grayscr),
            default: "stim_onset"
        - integ (bool)
            if True, sequence data is integrated
            default: False
        - datatype (str):
            type of data to return ("roi", "run" or "pupil")
            default: "roi"

    Returns:
        - data_arr (nested list):
            sequence data array
            dims: split (x ROIs) x seq (x frames)
        - time_values (1D array):
            values for each frame, in seconds (for 0 to stimpar.post)
    """

    if split not in ["stim_onset", "stim_offset"]:
        gen_util.accepted_values_error(
            "split", split, ["stim_onset", "stim_offset"]
            )

    if stimpar.stimtype != "both":
        raise ValueError("stimpar.stimtype must be 'both', if analysing "
            "stimulus on/off data.")

    stim = None
    for stimtype in ["gabors", "bricks"]: # use any stimulus to retrieve data
        if hasattr(sess, stimtype):
            stim = sess.get_stim(stimtype)
            break
    
    if split == "stim_onset":
        stim_fr = sess.grayscr.get_last_nongab_stim_fr()["last_stim_fr"][:-1] + 1
    elif split == "stim_offset":
        stim_fr = sess.grayscr.get_first_nongab_stim_fr()["first_stim_fr"][1:]

    stim_on_off_data = []
    for i in range(2):

        if i == 0:
            pre, post = [stimpar.pre, 0]
        else:
            pre, post = [0, stimpar.post]

        # ROI x seq (x frames)
        data, time_values = get_data(
            stim, stim_fr, analyspar, pre=pre, post=post, 
            ch_fl=[stimpar.pre, stimpar.post], integ=integ, ref_type="stim_frs", 
            datatype=datatype,
            )
        
        # very few stim onset/offset sequences, so best to retain all
        axis = -1 if integ else -2
        if data.shape[axis] != len(stim_fr):
            raise RuntimeError("Not all sequences could be retained for "
                f"{split} with stimpar.pre={stimpar.pre} and "
                f"stimpar.post={stimpar.post}.")


        stim_on_off_data.append(data.tolist())
        
    return stim_on_off_data, time_values


#############################################
def get_split_data_by_sess(sess, analyspar, stimpar, split="by_exp", 
                           integ=False, baseline=0.0, common_oris=False, 
                           datatype="roi"):
    """
    get_split_data_by_sess(sess, analyspar, stimpar)

    Returns data for the session, split as requested.

    Required args:
        - sess (Session): 
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - split (str): 
            how to split data:
            "by_exp" (all exp, all unexp), 
            "unexp_lock" (unexp, preceeding exp), 
            "exp_lock" (exp, preceeding unexp),
            "stim_onset" (grayscr, stim on), 
            "stim_offset" (stim off, grayscr)
            default: "by_exp"
        - integ (bool)
            if True, sequence data is integrated
            default: False
        - baseline (bool or num): 
            if not False, number of second to use for baseline 
            (not implemented)
            default: 0.0
        - common_oris (bool): 
            if True, only Gabor stimulus orientations common to D and U frames 
            are included ("by_exp" split only)
            default: False
        - datatype (str):
            type of data to return ("roi", "run" or "pupil")
            default: "roi"

    Returns:
        - data (nested list): 
            list of data arrays
            dims: split (x ROIs) x seq (x frames)
        - time_values (1D array):
            values for each frame, in seconds 
            (only 0 to stimpar.post, unless split is "by_exp")
    """
    

    locks = ["exp_lock", "unexp_lock"]
    stim_on_offs = ["stim_onset", "stim_offset"] 

    if baseline != 0:
        raise NotImplementedError("Baselining not implemented here.")

    if common_oris:
        get_common_oris(stimpar, split=split) # checks if permitted

    arg_dict = {
        "sess"      : sess,
        "analyspar" : analyspar,
        "stimpar"   : stimpar,
        "integ"     : integ,
        "datatype"  : datatype,
    }

    if split == "by_exp":
        data, time_values = get_by_exp_data(common_oris=common_oris, **arg_dict)
    elif split in locks:
        data, time_values = get_locked_data(split=split, **arg_dict)
    elif split in stim_on_offs:
        data, time_values = get_stim_on_off_data(split=split, **arg_dict)
    else:
        gen_util.accepted_values_error(
            "split", split, ["by_exp"] + locks + stim_on_offs
            )

    return data, time_values


#############################################
def get_sess_roi_trace_stats(sess, analyspar, stimpar, basepar, 
                             split="by_exp"):
    """
    get_sess_roi_trace_stats(sess, analyspar, stimpar, basepar)

    Returns ROI trace statistics for a specific session, split as requested.

    Required args:
        - sess (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - basepar (BasePar): 
            named tuple containing baseline parameters

    Optional args:
        - split (str): 
            how to split data:
            "by_exp" (all exp, all unexp), 
            "unexp_lock" (unexp, preceeding exp), 
            "exp_lock" (exp, preceeding unexp),
            "stim_onset" (grayscr, stim on), 
            "stim_offset" (stim off, grayscr)
            default: "by_exp"

    Returns:
        - stats (4D array): 
            ROI trace statistics for a sessions
            dims: exp, unexp x ROIs x frames x stats
        - time_values (1D array):
            values for each frame, in seconds 
            (only 0 to stimpar.post, unless split is "by_exp")
    """
    
    nanpol = None if analyspar.remnans else "omit"

    split_data, time_values = get_split_data_by_sess(
        sess, analyspar, stimpar, split=split, baseline=basepar.baseline, 
        datatype="roi"
        )
    
    stats = []
    # split x ROIs x frames x stats
    for data in split_data:
        stats.append(
            np.transpose(
                math_util.get_stats(
                    data, stats=analyspar.stats, error=analyspar.error, 
                    axes=1, nanpol=nanpol
                    ), 
                [1, 2, 0])
            )
    stats = np.asarray(stats)

    return stats, time_values


#############################################
def split_seqs_by_block(by_exp_fr_ns): 
    """
    split_seqs_by_block(by_exp_fr_ns)

    Returns indices for sequences split by blocks alternating between expected 
    and unexpected sequences, based on the starting frame number for each 
    sequence.

    Required args:
        - by_exp_fr_ns (list):
            list [exp, unexp] of arrays containing the starting frame number
            for each sequence

    Returns:
        - exp_seq_block_idxs (list): 
            list of arrays of expected sequence indices split by block
        - unexp_seq_block_idxs (list):
            list of arrays of unexpected sequence indices split by block

    """

    # run checks
    if len(by_exp_fr_ns) != 2:
        raise ValueError("by_exp_fr_ns should comprise 2 lists or arrays.")

    if len(set(by_exp_fr_ns[0]).intersection(by_exp_fr_ns[1])):
        raise ValueError("")

    for fr_ns in by_exp_fr_ns:
        if (np.sort(fr_ns) != np.asarray(fr_ns)).any():
            raise ValueError(
                "All frames within each by_exp_fr_ns list or array should "
                "be sorted."
                )

    # split into blocks
    by_exp_ns = [len(fr_ns) for fr_ns in by_exp_fr_ns]
    sort_all = np.argsort(np.argsort(np.concatenate(by_exp_fr_ns)))
    sort_exp = sort_all[: by_exp_ns[0]]
    sort_unexp = sort_all[by_exp_ns[0] :]

    new_exp_blocks = np.insert(np.where(np.diff(sort_exp) > 1)[0] + 1, 0, 0)
    new_unexp_blocks = np.insert(np.where(np.diff(sort_unexp) > 1)[0] + 1, 0, 0)

    if sort_exp[0] != 0: # adjust if unexp frames preceed exp frames
        new_exp_blocks = new_exp_blocks[:-1]
        new_unexp_blocks = new_unexp_blocks[1:]
    
    if len(new_exp_blocks) != len(new_unexp_blocks):
        raise RuntimeError(
            "Implementation error. Arrays should have same length."
            )

    exp_seq_block_idxs = []
    unexp_seq_block_idxs = []
    for i in range(len(new_exp_blocks) - 1):
        exp_seq_block_idxs.append(
            np.arange(new_exp_blocks[i], new_exp_blocks[i + 1])
            )
        unexp_seq_block_idxs.append(
            np.arange(new_unexp_blocks[i], new_unexp_blocks[i + 1])
            )
    
    return exp_seq_block_idxs, unexp_seq_block_idxs


#############################################
def get_block_data(sess, analyspar, stimpar, datatype="roi", integ=False):
    """
    get_block_data(sess, analyspar, stimpar)

    Returns data statistics split by expected/unexpected sequences, and by 
    blocks, where one block is defined as consecutive expected sequences, and 
    the subsequent consecutive unexpected sequences.

    Required args:
        - sess (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters

    Optional args:
        - datatype (str):
            type of data to return ("roi", "run" or "pupil")
            default: "roi"
        - integ (bool):
            if True, data is integrated across frames, instead of a statistic 
            being taken
            default: False

    Returns:
        - block_data (3 or 4D array):
            data statistics across sequences per block
            dims: split x block (x ROIs) x stats (me, err)
    """

    stim = sess.get_stim(stimpar.stimtype)

    nanpol = None if analyspar.remnans else "omit"

    ch_fl = [stimpar.pre, stimpar.post]

    by_exp_fr_ns = []
    by_exp_data = []
    for exp in [0, 1]:
        segs = stim.get_segs_by_criteria(
            gabfr=stimpar.gabfr, gabk=stimpar.gabk, gab_ori=stimpar.gab_ori,
            bri_dir=stimpar.bri_dir, bri_size=stimpar.bri_size, surp=exp, 
            remconsec=False, by="seg")

        # MUST obtain frame numbers and check flanks for 
        # to ensure later data indexing is correct
        if datatype == "run":
            frame_type = "stim_frs"
            fr_ns = stim.get_stim_fr_by_seg(
                segs, first=True, ch_fl=ch_fl
                )["first_stim_fr"]
        
        elif datatype == "pupil":
            frame_type = "pup_frs"
            fr_ns = stim.get_twop_fr_by_seg(segs, first=True)["first_twop_fr"]
            fr_ns = stim.sess.get_pup_fr_by_twop_fr(fr_ns, ch_fl=ch_fl)
        
        elif datatype == "roi":
            frame_type = "twop_frs"
            fr_ns = stim.get_twop_fr_by_seg(
                segs, first=True, ch_fl=ch_fl
                )["first_twop_fr"]
        else:
            gen_util.accepted_values_error(
                
                "datatype", datatype, ["roi", "run", "pupil"]
                )

        by_exp_fr_ns.append(np.asarray(fr_ns))

        data, _ = get_data(
            stim, fr_ns, analyspar, pre=stimpar.pre, post=stimpar.post, 
            integ=integ, datatype=datatype, ref_type=frame_type
            )
        
        if not integ: # take statistic across frames
            with gen_util.TempWarningFilter("Mean of empty", RuntimeWarning):
                data = math_util.mean_med(
                    data, stats=analyspar.stats, axis=-1, nanpol=nanpol
                )

        by_exp_data.append(data)
    
    # take means per block
    block_idxs = split_seqs_by_block(by_exp_fr_ns)

    n_splits = len(by_exp_data)
    n_blocks = len(block_idxs[0])
    n_stats = 2 
    if analyspar.stats == "median" and analyspar.error == "std":
        n_stats = 3
    
    targ_shape = (n_splits, n_blocks, n_stats)
    if datatype == "roi":
        n_rois = sess.get_nrois(analyspar.remnans, analyspar.fluor)
        targ_shape = (n_splits, n_blocks, n_rois, n_stats)

    block_data = np.full(targ_shape, np.nan)
    for b, seq_idxs in enumerate(zip(*block_idxs)):
        for d, data_seq_idxs in enumerate(seq_idxs): 
            # take stats across sequences within each split/block
            block_data[d, b] = math_util.get_stats(
                by_exp_data[d][..., data_seq_idxs],
                stats=analyspar.stats,
                error=analyspar.error,
                nanpol=nanpol, 
                axes=-1 # sequences within 
            ).T


    return block_data


