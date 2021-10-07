"""
behav_figs.py

This script contains functions defining running and pupil figure panel analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging
from sess_util import sess_ntuple_util

from util import logger_util
from analysis import behav_analys
from paper_fig_util import helper_fcts

logger = logging.getLogger(__name__)



############################################
def pupil_run_responses(sessions, analyspar, sesspar, stimpar, basepar, figpar, 
                        parallel=False):
    """
    pupil_run_responses(sessions, analyspar, sesspar, stimpar, basepar, figpar)

    Retrieves pupil and running responses to Gabor sequences for session 1.
        
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
        - basepar (BasePar): 
            named tuple containing baseline parameters
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    logger.info("Compiling pupil and running sequences for session 1.", 
        extra={"spacing": "\n"})

    split = "by_exp"
    trace_df = behav_analys.get_pupil_run_trace_stats_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        split=split, 
        parallel=parallel
        )

    extrapar = {
        "split": split
    }

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "basepar"  : basepar._asdict(),
            "extrapar" : extrapar,
            "trace_df" : trace_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


############################################
def pupil_run_block_diffs(sessions, analyspar, sesspar, stimpar, permpar, 
                          figpar, seed=None, parallel=False):
    """
    pupil_run_block_diffs(sessions, analyspar, sesspar, stimpar, permpar, 
                          figpar)

    Retrieves pupil and running block differences for Gabor sequences for 
    session 1.
        
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

    logger.info("Compiling pupil and running block differences for session 1.", 
        extra={"spacing": "\n"})

    permpar = sess_ntuple_util.get_modif_ntuple(permpar, "multcomp", False)

    block_df = behav_analys.get_pupil_run_block_stats_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        permpar=permpar, 
        randst=seed,
        parallel=parallel
        )

    extrapar = {
        "seed": seed
    }

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "permpar"  : permpar._asdict(),
            "extrapar" : extrapar,
            "block_df" : block_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)

