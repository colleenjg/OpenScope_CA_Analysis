"""
misc_figs.py

This script contains functions defining decoding figure panel analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging
from sess_util import sess_ntuple_util

from util import logger_util
from analysis import misc_analys
from analysis import decoding_analys
from paper_fig_util import helper_fcts

logger = logging.getLogger(__name__)



#############################################
def gabor_decoding_sess123(sessions, analyspar, sesspar, stimpar, permpar, 
                           logregpar, figpar, seed=None, parallel=False):
    """
    gabor_decoding_sess123(sessions, analyspar, sesspar, stimpar, permpar, 
                           logregpar, figpar)

    Runs decoding analyses (D and U orientations).
        
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list): 
            Session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - stimpar (StimPar): 
            named tuple containing stimulus parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters
        - logregpar (LogRegPar): 
            named tuple containing logistic regression parameters
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """


    if not analyspar.scale:
        raise ValueError("analyspar.scale should be True.")

    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes", "sess_ns"])
    permpar = misc_analys.set_multcomp(
        permpar, sess_df=dummy_df, pairs=False, factor=2
        )

    scores_dfs = dict()
    for comp in logregpar.comp:
        logregpar_use = sess_ntuple_util.get_modif_ntuple(
            logregpar, "comp", comp
            )

        scores_df = decoding_analys.run_sess_log_regs(
            sessions, 
            analyspar=analyspar, 
            stimpar=stimpar,
            logregpar=logregpar_use, 
            permpar=permpar, 
            n_splits=100,
            seed=seed, 
            parallel=parallel,
            )
        
        scores_dfs[comp] = scores_df.to_dict()

    extrapar = {"seed": seed}

    info = {"analyspar" : analyspar._asdict(),
            "sesspar"   : sesspar._asdict(),
            "stimpar"   : stimpar._asdict(),
            "permpar"   : permpar._asdict(),
            "logregpar" : logregpar._asdict(),
            "extrapar"  : extrapar,
            "scores_dfs": scores_dfs
            }

    helper_fcts.plot_save_all(info, figpar)

