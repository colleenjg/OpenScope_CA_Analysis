"""
stim_figs.py

This script contains functions defining stimulus comparison figure panel 
analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging
from sess_util import sess_ntuple_util

from util import logger_util
from analysis import stim_analys, misc_analys
from paper_fig_util import helper_fcts

logger = logging.getLogger(__name__)


############################################
def unexp_resp_stimulus_comp_sess1v3(sessions, analyspar, sesspar, stimpar, 
                                     permpar, figpar, seed=None, 
                                     parallel=False):
    """
    unexp_resp_stimulus_comp_sess1v3(sessions, analyspar, sesspar, stimpar, 
                                     permpar, figpar)

    Retrieves changes in tracked ROI responses to unexpected sequences for 
    Gabors vs visual flow stimuli.
        
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

    logger.info(
        ("Compiling changes in unexpected responses to Gabors vs visual "
        "flow stimuli."), 
        extra={"spacing": "\n"})

    if analyspar.scale:
        raise ValueError("analyspar.scale should be set to False.")

    # calculate multiple comparisons
    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes"])
    multcomp = len(dummy_df) + 1
    permpar = sess_ntuple_util.get_modif_ntuple(permpar, "multcomp", multcomp)

    comp_sess = [1, 3]
    datatype = "unexp_rel_resp"
    rel_sess = 1
    pop_stats = True
    unexp_comp_df = stim_analys.get_stim_stats_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        permpar=permpar,
        comp_sess=comp_sess,
        datatype=datatype,
        rel_sess=rel_sess,
        pop_stats=pop_stats,
        seed=seed,
        parallel=parallel,
        )

    extrapar = {
        "comp_sess": comp_sess,
        "datatype" : datatype,
        "rel_sess" : rel_sess,
        "pop_stats": pop_stats,
        "seed"     : seed,
    }

    info = {"analyspar"    : analyspar._asdict(),
            "sesspar"      : sesspar._asdict(),
            "stimpar"      : stimpar._asdict(),
            "permpar"      : permpar._asdict(),
            "extrapar"     : extrapar,
            "unexp_comp_df": unexp_comp_df.to_dict(),
            }

    helper_fcts.plot_save_all(info, figpar)


############################################
def tracked_roi_usis_stimulus_comp_sess1v3(sessions, analyspar, sesspar, 
                                           stimpar, basepar, idxpar, permpar,
                                           figpar, seed=None, parallel=False):
    """
    tracked_roi_usis_stimulus_comp_sess1v3(sessions, analyspar, sesspar, 
                                           stimpar, basepar, idxpar, permpar,
                                           figpar)

    Retrieves changes in tracked ROI USIs for Gabors vs visual flow stimuli.
        
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
        ("Compiling changes in ROI USIs for Gabors vs visual flow stimuli."), 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    # calculate multiple comparisons
    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes"])
    multcomp = len(dummy_df) + 1
    permpar = sess_ntuple_util.get_modif_ntuple(permpar, "multcomp", multcomp)

    comp_sess = [1, 3]
    datatype = "usis"
    pop_stats = True
    usi_comp_df = stim_analys.get_stim_stats_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar,
        basepar=basepar,
        idxpar=idxpar, 
        permpar=permpar,
        comp_sess=comp_sess,
        datatype=datatype,
        pop_stats=pop_stats,
        seed=seed,
        parallel=parallel,
        )

    extrapar = {
        "comp_sess": comp_sess,
        "datatype" : datatype,
        "pop_stats": pop_stats,
        "seed"     : seed,
    }

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "stimpar"    : stimpar._asdict(),
            "basepar"    : basepar._asdict(),
            "idxpar"     : idxpar._asdict(),
            "permpar"    : permpar._asdict(),
            "extrapar"   : extrapar,
            "usi_comp_df": usi_comp_df.to_dict(),
            }

    helper_fcts.plot_save_all(info, figpar)

    