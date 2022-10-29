"""
seq_figs.py

This script contains functions defining sequence figure panel analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

from util import logger_util
from analysis import seq_analys, misc_analys
from paper_fig_util import helper_fcts


logger = logger_util.get_module_logger(name=__name__)


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

    split = "by_exp"
    trace_df = seq_analys.get_sess_grped_trace_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        split=split, 
        parallel=parallel
        )
   
    extrapar = {
        "split": split,
    }

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

    split = "by_exp"
    diffs_df = seq_analys.get_sess_grped_diffs_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        permpar=permpar,
        split=split, 
        randst=seed,
        parallel=parallel,
        )
    
    extrapar = {
        "split": split,
        "seed" : seed,
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
                           figpar, seed=None, parallel=False):
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

    if analyspar.scale:
        raise ValueError("analyspar.scale should be set to False.")

    # calculate multiple comparisons
    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes", "sess_ns"])
    permpar = misc_analys.set_multcomp(
        permpar, sess_df=dummy_df, CIs=False, factor=2
        )

    rel_sess = 1
    rel_resp_df = seq_analys.get_rel_resp_stats_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        permpar=permpar,
        rel_sess=rel_sess,
        randst=seed,
        parallel=parallel,
        )

    extrapar = {
        "rel_sess": rel_sess,
        "seed"    : seed,
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

    split = "stim_onset"
    trace_df = seq_analys.get_sess_grped_trace_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        split=split, 
        parallel=parallel
        )

    extrapar = {
        "split": split,
    }

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "basepar"  : basepar._asdict(),
            "extrapar" : extrapar,
            "trace_df" : trace_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


#############################################
def stimulus_offset_sess123(sessions, analyspar, sesspar, stimpar, basepar, 
                            figpar, parallel=False):
    """
    stimulus_offset_sess123(sessions, analyspar, sesspar, stimpar, basepar, 
                            figpar)

    Retrieves ROI responses to stimulus offset from sessions 1 to 3.
        
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

    logger.info("Compiling stimulus offset sequences from session 1 to 3.", 
        extra={"spacing": "\n"})

    split = "stim_offset"
    trace_df = seq_analys.get_sess_grped_trace_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        split=split, 
        parallel=parallel
        )

    extrapar = {
        "split": split,
    }

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "basepar"  : basepar._asdict(),
            "extrapar" : extrapar,
            "trace_df" : trace_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


#############################################
def gabor_ex_roi_exp_responses_sess1(sessions, analyspar, sesspar, stimpar, 
                                     basepar, figpar, seed=None, 
                                     parallel=False):
    """
    gabor_ex_roi_exp_responses_sess1(sessions, analyspar, sesspar, stimpar, 
                                     basepar, figpar)

    Retrieves example ROI responses to expected Gabor sequence from session 1.
        
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
        "Compiling example ROI responses to expected Gabor sequence from "
        "session 1.", 
        extra={"spacing": "\n"})

    n_ex = 6
    rolling_win = 4
    unexp = 0
    ex_traces_df = seq_analys.get_ex_traces_df(
        sessions, 
        analyspar, 
        stimpar, 
        basepar, 
        n_ex=n_ex,
        rolling_win=rolling_win,
        unexp=unexp,
        randst=seed,
        parallel=parallel,
        )

    extrapar = {
        "n_ex"       : n_ex,
        "rolling_win": rolling_win,
        "unexp"      : unexp,
        "seed"       : seed,
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
def gabor_ex_roi_unexp_responses_sess1(sessions, analyspar, sesspar, stimpar, 
                                       basepar, figpar, seed=None, 
                                       parallel=False):
    """
    gabor_ex_roi_unexp_responses_sess1(sessions, analyspar, sesspar, stimpar, 
                                       basepar, figpar)

    Retrieves example ROI responses to unexpected Gabor sequence from session 1.
        
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
        "Compiling example ROI responses to unexpected Gabor sequence from "
        "session 1.", 
        extra={"spacing": "\n"})

    n_ex = 6
    rolling_win = 4
    unexp = 1
    ex_traces_df = seq_analys.get_ex_traces_df(
        sessions, 
        analyspar, 
        stimpar, 
        basepar, 
        n_ex=n_ex,
        rolling_win=rolling_win,
        unexp=unexp,
        randst=seed,
        parallel=parallel,
        )

    extrapar = {
        "n_ex"       : n_ex,
        "rolling_win": rolling_win,
        "unexp"      : unexp,
        "seed"       : seed,
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
def visflow_ex_roi_nasal_responses_sess1(sessions, analyspar, sesspar, stimpar, 
                                         basepar, figpar, seed=None, 
                                         parallel=False):
    """
    visflow_ex_roi_nasal_responses_sess1(sessions, analyspar, sesspar, stimpar, 
                                         basepar, figpar)

    Retrieves example ROI responses to unexpected flow during nasal (leftward) 
    visual flow, from session 1.
        
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
        "Compiling example ROI responses to onset of unexpected flow during "
        "nasal (leftward) visual flow from session 1.", 
        extra={"spacing": "\n"})

    n_ex = 6
    rolling_win = 4
    unexp = 0
    ex_traces_df = seq_analys.get_ex_traces_df(
        sessions, 
        analyspar, 
        stimpar, 
        basepar, 
        n_ex=n_ex,
        rolling_win=rolling_win,
        unexp=unexp,
        randst=seed,
        parallel=parallel,
        )

    extrapar = {
        "n_ex"       : n_ex,
        "rolling_win": rolling_win,
        "seed"       : seed,
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
def visflow_ex_roi_temp_responses_sess1(sessions, analyspar, sesspar, stimpar, 
                                        basepar, figpar, seed=None, 
                                        parallel=False):
    """
    visflow_ex_roi_temp_responses_sess1(sessions, analyspar, sesspar, stimpar, 
                                        basepar, figpar)

    Retrieves example ROI responses to unexpected flow during temporal 
    (rightward) visual flow, from session 1.
        
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
        "Compiling example ROI responses to onset of unexpected flow during "
        "temporal (rightward) visual flow from session 1.", 
        extra={"spacing": "\n"})

    n_ex = 6
    rolling_win = 4
    unexp = 1
    ex_traces_df = seq_analys.get_ex_traces_df(
        sessions, 
        analyspar, 
        stimpar, 
        basepar, 
        n_ex=n_ex,
        rolling_win=rolling_win,
        unexp=unexp,
        randst=seed,
        parallel=parallel,
        )

    extrapar = {
        "n_ex"       : n_ex,
        "rolling_win": rolling_win,
        "seed"       : seed,
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
                                        permpar, figpar, seed=None, 
                                        parallel=False):
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

    if analyspar.scale:
        raise ValueError("analyspar.scale should be set to False.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    # calculate multiple comparisons
    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes", "sess_ns"])

    permpar = misc_analys.set_multcomp(
        permpar, sess_df=dummy_df, CIs=False, factor=2
        )

    rel_sess = 1
    rel_resp_df = seq_analys.get_rel_resp_stats_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        permpar=permpar,
        rel_sess=rel_sess,
        randst=seed,
        parallel=parallel,
        )

    extrapar = {
        "rel_sess": rel_sess,
        "seed"    : seed,
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

    split = "unexp_lock"
    trace_df = seq_analys.get_sess_grped_trace_df(
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

    split = "unexp_lock"
    diffs_df = seq_analys.get_sess_grped_diffs_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        permpar=permpar,
        split=split, 
        randst=seed,
        parallel=parallel,
        )
    
    extrapar = {
        "split": split,
        "seed" : seed,
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
                                 permpar, figpar, seed=None, parallel=False):
    """
    visual_flow_rel_resp_sess123(sessions, analyspar, sesspar, stimpar, 
                                 permpar, figpar)

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

    if analyspar.scale:
        raise ValueError("analyspar.scale should be set to False.")

    # calculate multiple comparisons
    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes", "sess_ns"])
    permpar = misc_analys.set_multcomp(
        permpar, sess_df=dummy_df, CIs=False, factor=2
        )

    rel_sess = 1
    rel_resp_df = seq_analys.get_rel_resp_stats_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        permpar=permpar,
        rel_sess=rel_sess,
        randst=seed,
        parallel=parallel,
        )

    extrapar = {
        "rel_sess": rel_sess,
        "seed"    : seed,
    }

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "stimpar"    : stimpar._asdict(),
            "permpar"    : permpar._asdict(),
            "extrapar"   : extrapar,
            "rel_resp_df": rel_resp_df.to_dict(),
            }

    helper_fcts.plot_save_all(info, figpar)

