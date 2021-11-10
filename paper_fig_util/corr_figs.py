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
from sess_util import sess_ntuple_util

logger = logging.getLogger(__name__)


PERMUTE = "tracking" # sess, tracking or all
CORR_TYPE = "corr" # corr or R_sqr
SIG_ONLY = False # whether to include only ROIs with significant USIs 

############################################
def gabor_corr_norm_res_ex(sessions, analyspar, sesspar, stimpar, basepar, 
                           idxpar, permpar, figpar, seed=None, parallel=False):
    """
    gabor_corr_norm_res_ex(sessions, analyspar, sesspar, stimpar, basepar, 
                           idxpar, permpar, figpar

    Retrieves example for calculating normalized residual correlations for 
    tracked ROI Gabor USIs.
        
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
        "Compiling example data for calculating normalized residual "
        "correlations for ROI Gabor USIs.", 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    permute = PERMUTE
    sig_only = SIG_ONLY
    n_bins = 40
    idx_corr_norm_df = corr_analys.get_ex_idx_corr_norm_df(
        sessions, 
        analyspar=analyspar,
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar,
        permute=permute,
        sig_only=sig_only,
        n_bins=n_bins,
        randst=seed,
        parallel=parallel
        )
        
    extrapar = {
        "n_bins"   : n_bins,
        "permute"  : permute,
        "seed"     : seed,
        "sig_only" : sig_only,
    }

    info = {"analyspar"       : analyspar._asdict(),
            "sesspar"         : sesspar._asdict(),
            "stimpar"         : stimpar._asdict(),
            "basepar"         : basepar._asdict(),
            "idxpar"          : idxpar._asdict(),
            "permpar"         : permpar._asdict(),
            "extrapar"        : extrapar,
            "idx_corr_norm_df": idx_corr_norm_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)

    return


############################################
def gabor_corrs_sess123_comps(sessions, analyspar, sesspar, stimpar, basepar, 
                              idxpar, permpar, figpar, seed=None, 
                              parallel=False):
    """
    gabor_corrs_sess123_comps(sessions, analyspar, sesspar, stimpar, basepar, 
                              idxpar, permpar, figpar)

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

    permute = PERMUTE
    corr_type = CORR_TYPE
    sig_only = SIG_ONLY

    if "R_sqr" in corr_type:
        permpar = sess_ntuple_util.get_modif_ntuple(permpar, "tails", "hi")

    idx_corr_df = corr_analys.get_idx_corrs_df(
        sessions, 
        analyspar=analyspar,
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar, 
        consec_only=consec_only,
        permute=permute,
        corr_type=corr_type,
        sig_only=sig_only,
        randst=seed,
        parallel=parallel
        )
        
    extrapar = {
        "consec_only": consec_only,
        "corr_type"  : corr_type,
        "permute"    : permute,
        "seed"       : seed,
        "sig_only"   : sig_only,
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
def visual_flow_corrs_sess123_comps(sessions, analyspar, sesspar, stimpar, 
                                    basepar, idxpar, permpar, figpar, 
                                    seed=None, parallel=False):
    """
    visual_flow_corrs_sess123_comps(sessions, analyspar, sesspar, stimpar, 
                                    basepar, idxpar, permpar, figpar)

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

    permute = PERMUTE
    corr_type = CORR_TYPE
    sig_only = SIG_ONLY

    if "R_sqr" in corr_type:
        permpar = sess_ntuple_util.get_modif_ntuple(permpar, "tails", "hi")

    idx_corr_df = corr_analys.get_idx_corrs_df(
        sessions, 
        analyspar=analyspar,
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar, 
        consec_only=consec_only,
        permute=permute,
        corr_type=corr_type,
        sig_only=sig_only,
        randst=seed,
        parallel=parallel
        )
        
    extrapar = {
        "consec_only": consec_only,
        "corr_type"  : corr_type,
        "permute"    : permute,
        "seed"       : seed,
        "sig_only"   : sig_only,
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

    