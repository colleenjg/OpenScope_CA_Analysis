"""
corr_figs.py

This script contains functions defining correlation figure panel analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

from util import logger_util
from analysis import misc_analys, corr_analys
from paper_fig_util import helper_fcts

logger = logging.getLogger(__name__)

# whether to shuffle ROI tracking (True), or session pair order (False)
PERMUTE_TRACKING = False 


############################################
def gabor_norm_res_corr_example(sessions):
    # 2 n_perms (idx and corrs)

    # if not analyspar.tracked:
    #     raise ValueError("analyspar.tracked should be set to True.")

    # # remove incomplete session series and warn
    # sessions = misc_analys.check_sessions_complete(sessions)

    print("NOT YET IMPLEMENTED")

    return


############################################
def gabor_norm_res_corrs_sess123_comps(sessions, analyspar, sesspar, stimpar, 
                                       basepar, idxpar, permpar, figpar, 
                                       seed=None, parallel=False):
    """
    gabor_norm_res_corrs_sess123_comps(sessions, analyspar, sesspar, stimpar, 
                                       basepar, idxpar, permpar, figpar)

    Retrieves tracked ROI Gabor USI correlations for session 1 to 3.
        
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
        - idxpar (IdxPar): 
            named tuple containing index parameters
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


    logger.info(
        "Compiling tracked ROI Gabor USI correlations for sessions 1 to 3.", 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    consec_only = True
    permpar = corr_analys.set_multcomp(
        permpar, sessions, analyspar, consec_only=consec_only
        )

    permute_tracking = PERMUTE_TRACKING
    idx_corr_df = corr_analys.get_idx_corrs_df(
        sessions, 
        analyspar=analyspar,
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar, 
        consec_only=consec_only,
        permute_tracking=permute_tracking,
        seed=seed,
        parallel=parallel
        )
        
    extrapar = {
        "seed"            : seed,
        "consec_only"     : consec_only,
        "permute_tracking": permute_tracking,
    }

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "stimpar"    : stimpar._asdict(),
            "basepar"    : basepar._asdict(),
            "idxpar"     : idxpar._asdict(),
            "permpar"    : permpar._asdict(),
            "extrapar"   : extrapar,
            "idx_corr_df": idx_corr_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


############################################
def visual_flow_norm_res_corrs_sess123_comps(sessions, analyspar, sesspar, 
                                             stimpar, basepar, idxpar, permpar, 
                                             figpar, seed=None, parallel=False):
    """
    visual_flow_norm_res_corrs_sess123_comps(sessions, analyspar, sesspar, 
                                             stimpar, basepar, idxpar, permpar, 
                                             figpar)

    Retrieves tracked ROI visual flow USI correlations for session 1 to 3.
        
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
        - idxpar (IdxPar): 
            named tuple containing index parameters
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


    logger.info(
        ("Compiling tracked ROI visual flow USI correlations for "
        "sessions 1 to 3."), 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    consec_only = True
    permpar = corr_analys.set_multcomp(
        permpar, sessions, analyspar, consec_only=consec_only
        )

    permute_tracking = PERMUTE_TRACKING
    idx_corr_df = corr_analys.get_idx_corrs_df(
        sessions, 
        analyspar=analyspar,
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar, 
        consec_only=consec_only,
        permute_tracking=permute_tracking,
        seed=seed,
        parallel=parallel
        )
        
    extrapar = {
        "seed"            : seed,
        "consec_only"     : consec_only,
        "permute_tracking": permute_tracking,
    }

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "stimpar"    : stimpar._asdict(),
            "basepar"    : basepar._asdict(),
            "idxpar"     : idxpar._asdict(),
            "permpar"    : permpar._asdict(),
            "extrapar"   : extrapar,
            "idx_corr_df": idx_corr_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)

    