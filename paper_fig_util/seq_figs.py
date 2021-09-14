"""
corr_figs.py

This script contains functions defining sequence figure panel analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

from util import logger_util
from analysis import seq_analys, misc_analys
from paper_fig_util import helper_fcts

logger = logging.getLogger(__name__)


############################################
def gabor_sequences_sess123(sessions, analyspar, sesspar, stimpar, basepar, 
                            figpar, parallel=False):
    """
    gabor_sequences_sess123(sessions, analyspar, sesspar, stimpar, basepar, 
                            figpar)

    Retrieves ROI responses to Gabor sequences from sessions 1 to 3.
        
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

    logger.info("Compiling Gabor sequences from session 1 to 3.", 
        extra={"spacing": "\n"})

    trace_df = seq_analys.get_sess_grped_trace_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        split="by_exp", 
        parallel=parallel
        )

    extrapar = dict()

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "basepar"  : basepar._asdict(),
            "extrapar" : extrapar,
            "trace_df" : trace_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)
    

#############################################    
def gabor_sequence_diffs_sess123(sessions, analyspar, sesspar, stimpar, 
                                 basepar, permpar, figpar, seed=None, 
                                 parallel=False):
    """
    gabor_sequence_diffs_sess123(sessions, analyspar, sesspar, stimpar, 
                                 basepar, permpar, figpar)

    Retrieves differences in ROI responses to Gabor sequences from 
    sessions 1 to 3.
        
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

    logger.info("Compiling Gabor sequence differences from session 1 to 3.", 
        extra={"spacing": "\n"})

    # calculate multiple comparisons
    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes", "sess_ns"])
    permpar = misc_analys.set_multcomp(permpar, sess_df=dummy_df)

    diffs_df = seq_analys.get_sess_grped_diffs_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        permpar=permpar,
        split="by_exp", 
        parallel=parallel,
        seed=seed,
        )
    
    extrapar = {
        "seed": seed
    }

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "basepar"  : basepar._asdict(),
            "permpar"  : permpar._asdict(),
            "extrapar" : extrapar,
            "diffs_df" : diffs_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


#############################################
def gabor_rel_resp_sess123(sessions, analyspar, sesspar, stimpar, permpar, 
                           figpar, parallel=False, seed=None):
    """
    gabor_rel_resp_sess123(sessions, analyspar, sesspar, stimpar, permpar, 
                           figpar)

    Retrieves ROI responses to regular and unexpected Gabor frames, relative 
    to session 1.
        
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

    logger.info("Compiling ROI Gabor responses relative to session 1.", 
        extra={"spacing": "\n"})

    # calculate multiple comparisons
    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes", "sess_ns"])
    permpar = misc_analys.set_multcomp(
        permpar, sess_df=dummy_df, CIs=False, factor=2
        )

    rel_resp_df = seq_analys.get_relative_resp_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        permpar=permpar,
        rel_sess=1,
        parallel=parallel,
        seed=seed,
        )

    extrapar = {
        "seed": seed
    }

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "stimpar"    : stimpar._asdict(),
            "permpar"    : permpar._asdict(),
            "extrapar"   : extrapar,
            "rel_resp_df": rel_resp_df.to_dict(),
            }

    helper_fcts.plot_save_all(info, figpar)
    

#############################################
def stimulus_onset_sess123(sessions, analyspar, sesspar, stimpar, basepar, 
                           figpar, parallel=False):
    """
    stimulus_onset_sess123(sessions, analyspar, sesspar, stimpar, basepar, 
                           figpar)

    Retrieves ROI responses to stimulus onset from sessions 1 to 3.
        
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

    logger.info("Compiling stimulus onset sequences from session 1 to 3.", 
        extra={"spacing": "\n"})

    trace_df = seq_analys.get_sess_grped_trace_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        split="stim_onset", 
        parallel=parallel
        )

    extrapar = dict()

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "basepar"  : basepar._asdict(),
            "extrapar" : extrapar,
            "trace_df" : trace_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


#############################################
def gabor_ex_roi_responses_sess1(sessions, analyspar, sesspar, stimpar, 
                                 basepar, figpar, seed=None, parallel=False):
    """
    gabor_ex_roi_responses_sess1(sessions, analyspar, sesspar, stimpar, 
                                 basepar, figpar)

    Retrieves example ROI Gabor sequence responses from session 1.
        
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
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    logger.info(
        "Compiling example ROI Gabor sequence responses from session 1.", 
        extra={"spacing": "\n"})

    ex_traces_df = seq_analys.get_ex_traces_df(
        sessions, 
        analyspar, 
        stimpar, 
        basepar, 
        parallel=parallel,
        seed=seed,
        )

    extrapar = {
        "seed": seed
    }

    info = {"analyspar"   : analyspar._asdict(),
            "sesspar"     : sesspar._asdict(),
            "stimpar"     : stimpar._asdict(),
            "basepar"     : basepar._asdict(),
            "extrapar"    : extrapar,
            "ex_traces_df": ex_traces_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)
    

#############################################
def gabor_rel_resp_tracked_rois_sess123(sessions, analyspar, sesspar, stimpar, 
                                        permpar, figpar, parallel=False, 
                                        seed=None):
    """
    gabor_rel_resp_tracked_rois_sess123(sessions, analyspar, sesspar, stimpar, 
                                        permpar, figpar)
    
    Retrieves ROI responses to regular and unexpected Gabor frames, relative 
    to session 1, for tracked ROIs.
        
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

    logger.info("Compiling tracked ROI Gabor responses relative to session 1.", 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    # calculate multiple comparisons
    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes", "sess_ns"])

    permpar = misc_analys.set_multcomp(
        permpar, sess_df=dummy_df, CIs=False, factor=2
        )

    rel_resp_df = seq_analys.get_relative_resp_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        permpar=permpar,
        rel_sess=1,
        parallel=parallel,
        seed=seed,
        )

    extrapar = {
        "seed": seed
    }

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "stimpar"    : stimpar._asdict(),
            "permpar"    : permpar._asdict(),
            "extrapar"   : extrapar,
            "rel_resp_df": rel_resp_df.to_dict(),
            }

    helper_fcts.plot_save_all(info, figpar)
    

#############################################
def visual_flow_sequences_sess123(sessions, analyspar, sesspar, stimpar, 
                                  basepar, figpar, parallel=False):
    """
    visual_flow_sequences_sess123(sessions, analyspar, sesspar, stimpar, 
                                  basepar, figpar)

    Retrieves ROI responses to visual flow sequences from sessions 1 to 3.
        
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

    logger.info("Compiling visual flow sequences from session 1 to 3.", 
        extra={"spacing": "\n"})

    trace_df = seq_analys.get_sess_grped_trace_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        split="unexp_lock", 
        parallel=parallel
        )

    extrapar = dict()

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "basepar"  : basepar._asdict(),
            "extrapar" : extrapar,
            "trace_df" : trace_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)
    

#############################################
def visual_flow_diffs_sess123(sessions, analyspar, sesspar, stimpar, basepar, 
                              permpar, figpar, seed=None, parallel=False):
    """
    visual_flow_diffs_sess123(sessions, analyspar, sesspar, stimpar, basepar, 
                              permpar, figpar)

    Retrieves differences in ROI responses to visual flow sequences from 
    sessions 1 to 3.
        
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
        "Compiling visual flow sequence differences from session 1 to 3.", 
        extra={"spacing": "\n"})

    # calculate multiple comparisons
    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes", "sess_ns"])
    permpar = misc_analys.set_multcomp(permpar, sess_df=dummy_df)

    diffs_df = seq_analys.get_sess_grped_diffs_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        permpar=permpar,
        split="unexp_lock", 
        parallel=parallel,
        seed=seed,
        )
    
    extrapar = {
        "seed": seed
    }

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "basepar"  : basepar._asdict(),
            "permpar"  : permpar._asdict(),
            "extrapar" : extrapar,
            "diffs_df" : diffs_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


#############################################
def visual_flow_rel_resp_sess123(sessions, analyspar, sesspar, stimpar, 
                                 permpar, figpar, parallel=False, seed=None):
    """
    visual_flow_rel_resp_sess123(sessions, analyspar, sesspar, stimpar, 
                                 permpar, figpar

    Retrieves ROI responses to expected and unexpected visual flow, relative 
    to session 1.
        
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

    logger.info("Compiling ROI visual flow responses relative to session 1.", 
        extra={"spacing": "\n"})

    # calculate multiple comparisons
    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes", "sess_ns"])
    permpar = misc_analys.set_multcomp(
        permpar, sess_df=dummy_df, CIs=False, factor=2
        )

    rel_resp_df = seq_analys.get_relative_resp_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        permpar=permpar,
        rel_sess=1,
        parallel=parallel,
        seed=seed,
        )

    extrapar = {
        "seed": seed
    }

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "stimpar"    : stimpar._asdict(),
            "permpar"    : permpar._asdict(),
            "extrapar"   : extrapar,
            "rel_resp_df": rel_resp_df.to_dict(),
            }

    helper_fcts.plot_save_all(info, figpar)


############################################
def rel_resp_stimulus_comp_sess1v3(sessions):

    print("NOT YET IMPLEMENTED")
    return
    
