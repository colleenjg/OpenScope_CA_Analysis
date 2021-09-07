"""
decoding_analys.py

This script contains functions for decoding analysis.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging
from sess_util import sess_ntuple_util

import numpy as np
import pandas as pd

from util import logger_util, gen_util, logreg_util
from sess_util import sess_gen_util
from analysis import misc_analys

logger = logging.getLogger(__name__)


#############################################
def get_decoding_data(sess, analyspar, stimpar, comp="Dori", ctrl=True, 
                      randst=None):

    if randst is None:
        randst = np.random
    elif isinstance(randst, int):
        randst = np.random.RandomState(randst)

    if stimpar.stimtype != "gabors":
        raise ValueError("Expected gabors stimtype.")

    if comp == "Dori":
        exp = 0
    elif comp == "Uori":
        exp = 1
        ctrl = False
    else:
        gen_util.accepted_values_error("comp", comp, ["Dori", "Uori"])

    gab_oris = sess_gen_util.get_params(gab_ori=stimpar.gab_ori)[-1]

    stim = sess.get_stim(stimpar.stimtype)

    all_input_data = []
    all_output_data = []
    n_ctrls = []
    for g, gab_ori in enumerate(gab_oris):
        segs = stim.get_segs_by_criteria(
            gabfr=stimpar.gabfr, gabk=stimpar.gabk, gab_ori=gab_ori, surp=exp, 
            remconsec=False, by="seg")
        fr_ns = stim.get_twop_fr_by_seg(segs, first=True)["first_twop_fr"]

        # sample as many sequences as are usable for unexpected data
        if ctrl:
            segs_ctrl = stim.get_segs_by_criteria(
                gabfr=stimpar.gabfr, gabk=stimpar.gabk, gab_ori=gab_ori, 
                surp=1, remconsec=False, by="seg")
            fr_ns_ctrl = stim.get_twop_fr_by_seg(
                segs_ctrl, first=True, ch_fl=[stimpar.pre, stimpar.post]
                )["first_twop_fr"]
            n_ctrls.append(len(fr_ns_ctrl))

        ori_data_df = stim.get_roi_data(
            fr_ns, stimpar.pre, stimpar.post, remnans=analyspar.remnans, 
            scale=analyspar.scale
            )
        # seq x frames x ROIs
        ori_data = np.transpose(
            gen_util.reshape_df_data(ori_data_df, squeeze_cols=True),
            [1, 2, 0]
        )

        all_input_data.append(ori_data)
        all_output_data.append(np.full(len(ori_data), g))

    all_input_data = np.concatenate(all_input_data, axis=0)
    all_output_data = np.concatenate(all_output_data)

    n_ctrls = False if not ctrl else n_ctrls

    return all_input_data, all_output_data, n_ctrls
        

#############################################
def run_log_reg(sess, analyspar, stimpar, logregpar, randst=None, parallel=False):


    input_data, output_data, n_ctrls = get_decoding_data(
        sess, analyspar, stimpar, comp=logregpar.comp, ctrl=logregpar.ctrl, 
        randst=randst)

    extrapar = {
        "n_runs": 100,
        "shuffle": False
    }

    mod_cvs, _, extrapar = logreg_util.run_logreg_cv_sk(
        input_data, output_data, logregpar._asdict(), extrapar, 
        analyspar.scale, n_ctrls, seed=randst, parallel=parallel, 
        catch_set_prob=False)


