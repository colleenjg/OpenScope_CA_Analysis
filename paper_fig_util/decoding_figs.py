"""
misc_figs.py

This script contains functions defining decoding figure panel analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import warnings

from util import logger_util
from sess_util import sess_ntuple_util, sess_gen_util
from analysis import misc_analys, decoding_analys
from paper_fig_util import helper_fcts


logger = logger_util.get_module_logger(name=__name__)


#############################################
def gabor_decoding_sess123(sessions, analyspar, sesspar, stimpar, logregpar, 
                           permpar, figpar, seed=None, parallel=False):
    """
    gabor_decoding_sess123(sessions, analyspar, sesspar, stimpar, logregpar, 
                           permpar, figpar)

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
        - logregpar (LogRegPar): 
            named tuple containing logistic regression parameters
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

    comp_str = logregpar.comp.replace("ori", " orientation")
    
    logger.info(
        f"Compiling Gabor {comp_str} decoder performances for sessions 1 to 3.", 
        extra={"spacing": "\n"})

    if not analyspar.scale:
        raise ValueError("analyspar.scale should be True.")

    # calculate multiple comparisons
    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes", "sess_ns"])
    permpar = misc_analys.set_multcomp(
        permpar, sess_df=dummy_df, pairs=False, factor=2
        )

    n_splits = 100
    score_df = decoding_analys.run_sess_logregs(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar,
        logregpar=logregpar, 
        permpar=permpar, 
        n_splits=n_splits,
        seed=seed, 
        parallel=parallel,
        )

    extrapar = {
        "n_splits": n_splits,
        "seed"    : seed,
        }

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "logregpar": logregpar._asdict(),
            "permpar"  : permpar._asdict(),
            "extrapar" : extrapar,
            "scores_df": score_df.to_dict(),
            }

    helper_fcts.plot_save_all(info, figpar)


#############################################
def gabor_Dori_decoding_sess123(sessions, analyspar, sesspar, stimpar, 
                                logregpar, permpar, figpar, seed=None, 
                                parallel=False):
    """
    gabor_Dori_decoding_sess123(sessions, analyspar, sesspar, stimpar, 
                                logregpar, permpar, figpar)

    Runs decoding analyses (D orientations).
        
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
        - logregpar (LogRegPar): 
            named tuple containing logistic regression parameters
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

    if logregpar.comp != "Dori":
        raise ValueError("logregpar.comp should be Dori.")

    gab_ori = sess_gen_util.filter_gab_oris("D", stimpar.gab_ori)
    stimpar = sess_ntuple_util.get_modif_ntuple(stimpar, "gab_ori", gab_ori)

    gabor_decoding_sess123(
        sessions, 
        analyspar=analyspar, 
        sesspar=sesspar, 
        stimpar=stimpar, 
        logregpar=logregpar, 
        permpar=permpar, 
        figpar=figpar, 
        seed=seed, 
        parallel=parallel
        )


#############################################
def gabor_Uori_decoding_sess123(sessions, analyspar, sesspar, stimpar, 
                                logregpar, permpar, figpar, seed=None, 
                                parallel=False):
    """
    gabor_Uori_decoding_sess123(sessions, analyspar, sesspar, stimpar, 
                                logregpar, permpar, figpar)

    Runs decoding analyses (U orientations).
        
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
        - logregpar (LogRegPar): 
            named tuple containing logistic regression parameters
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

    if logregpar.comp != "Uori":
        raise ValueError("logregpar.comp should be Uori.")

    # ctrl doesn't apply to U orientation decoding
    logregpar = sess_ntuple_util.get_modif_ntuple(logregpar, "ctrl", False)

    gab_ori = sess_gen_util.filter_gab_oris("U", stimpar.gab_ori)
    stimpar = sess_ntuple_util.get_modif_ntuple(stimpar, "gab_ori", gab_ori)

    gabor_decoding_sess123(
        sessions, 
        analyspar=analyspar, 
        sesspar=sesspar, 
        stimpar=stimpar, 
        logregpar=logregpar, 
        permpar=permpar, 
        figpar=figpar, 
        seed=seed, 
        parallel=parallel
        )


#############################################
def gabor_Aori_decoding_sess123(sessions, analyspar, sesspar, stimpar, 
                                logregpar, permpar, figpar, seed=None, 
                                parallel=False):
    """
    gabor_Aori_decoding_sess123(sessions, analyspar, sesspar, stimpar, 
                                logregpar, permpar, figpar)

    Runs decoding analyses (A orientations).
        
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
        - logregpar (LogRegPar): 
            named tuple containing logistic regression parameters
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

    if logregpar.comp != "Aori":
        raise ValueError("logregpar.comp should be Aori.")

    gab_ori = sess_gen_util.filter_gab_oris("A", stimpar.gab_ori)
    stimpar = sess_ntuple_util.get_modif_ntuple(stimpar, "gab_ori", gab_ori)

    gabor_decoding_sess123(
        sessions, 
        analyspar=analyspar, 
        sesspar=sesspar, 
        stimpar=stimpar, 
        logregpar=logregpar, 
        permpar=permpar, 
        figpar=figpar, 
        seed=seed, 
        parallel=parallel
        )


#############################################
def gabor_Bori_decoding_sess123(sessions, analyspar, sesspar, stimpar, 
                                logregpar, permpar, figpar, seed=None, 
                                parallel=False):
    """
    gabor_Bori_decoding_sess123(sessions, analyspar, sesspar, stimpar, 
                                logregpar, permpar, figpar)

    Runs decoding analyses (B orientations).
        
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
        - logregpar (LogRegPar): 
            named tuple containing logistic regression parameters
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

    if logregpar.comp != "Bori":
        raise ValueError("logregpar.comp should be Bori.")

    gab_ori = sess_gen_util.filter_gab_oris("B", stimpar.gab_ori)
    stimpar = sess_ntuple_util.get_modif_ntuple(stimpar, "gab_ori", gab_ori)

    gabor_decoding_sess123(
        sessions, 
        analyspar=analyspar, 
        sesspar=sesspar, 
        stimpar=stimpar, 
        logregpar=logregpar, 
        permpar=permpar, 
        figpar=figpar, 
        seed=seed, 
        parallel=parallel
        )


#############################################
def gabor_Cori_decoding_sess123(sessions, analyspar, sesspar, stimpar, 
                                logregpar, permpar, figpar, seed=None, 
                                parallel=False):
    """
    gabor_Cori_decoding_sess123(sessions, analyspar, sesspar, stimpar, 
                                logregpar, permpar, figpar)

    Runs decoding analyses (C orientations).
        
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
        - logregpar (LogRegPar): 
            named tuple containing logistic regression parameters
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

    if logregpar.comp != "Cori":
        raise ValueError("logregpar.comp should be Cori.")

    gab_ori = sess_gen_util.filter_gab_oris("C", stimpar.gab_ori)
    stimpar = sess_ntuple_util.get_modif_ntuple(stimpar, "gab_ori", gab_ori)

    gabor_decoding_sess123(
        sessions, 
        analyspar=analyspar, 
        sesspar=sesspar, 
        stimpar=stimpar, 
        logregpar=logregpar, 
        permpar=permpar, 
        figpar=figpar, 
        seed=seed, 
        parallel=parallel
        )


#############################################
def gabor_timecourse_decoding(sessions, analyspar, sesspar, stimpar, 
                              logregpar, permpar, figpar, seed=None, 
                              parallel=False):
    """
    gabor_timecourse_decoding(sessions, analyspar, sesspar, stimpar, 
                              logregpar, permpar, figpar)

    Runs timecourse decoding analyses (D vs U orientations).
        
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
        - logregpar (LogRegPar): 
            named tuple containing logistic regression parameters
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

    if not isinstance(logregpar.comp, list) or len(logregpar.comp) != 2:
        raise ValueError("logregpar.comp should be a list of length 2.")
    
    if not (stimpar.gab_ori in ["all", "any"] or 
        (isinstance(stimpar.gab_ori, list) and len(stimpar.gab_ori) == 6)
    ):
        warnings.warn(
            f"stimpar.gab_ori was {stimpar.gab_ori}, but will be set to "
            "'all' or 'any'."
            )
        gab_ori = sess_gen_util.filter_gab_oris("any", "any")
        stimpar = sess_ntuple_util.get_modif_ntuple(stimpar, "gab_ori", gab_ori)

    if not analyspar.scale:
        raise ValueError("analyspar.scale should be True.")

    # calculate multiple comparisons
    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes", "sess_ns"])
    permpar = misc_analys.set_multcomp(
        permpar, sess_df=dummy_df, pairs=False, factor=1
        )

    n_splits = 100
    score_df = decoding_analys.run_timecourse_logregs(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar,
        logregpar=logregpar, 
        permpar=permpar, 
        n_splits=n_splits,
        seed=seed, 
        parallel=parallel,
        )

    extrapar = {
        "n_splits": n_splits,
        "seed"    : seed,
        }

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "logregpar": logregpar._asdict(),
            "permpar"  : permpar._asdict(),
            "extrapar" : extrapar,
            "scores_df": score_df.to_dict(),
            }

    helper_fcts.plot_save_all(info, figpar)


#############################################
def gabor_timecourse_decoding_sess1(sessions, analyspar, sesspar, stimpar, 
                                    logregpar, permpar, figpar, seed=None, 
                                    parallel=False):
    """
    gabor_timecourse_decoding_sess1(sessions, analyspar, sesspar, stimpar, 
                                    logregpar, permpar, figpar)

    Runs timecourse decoding analyses (D vs U orientations).
        
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
        - logregpar (LogRegPar): 
            named tuple containing logistic regression parameters
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
        f"Compiling Gabor timecourse decoder performances for session 1.", 
        extra={"spacing": "\n"})


    gabor_timecourse_decoding(
        sessions, 
        analyspar=analyspar, 
        sesspar=sesspar, 
        stimpar=stimpar, 
        logregpar=logregpar, 
        permpar=permpar, 
        figpar=figpar, 
        seed=seed, 
        parallel=parallel,
        )
    

#############################################
def gabor_timecourse_decoding_sess2(sessions, analyspar, sesspar, stimpar, 
                                    logregpar, permpar, figpar, seed=None, 
                                    parallel=False):
    """
    gabor_timecourse_decoding_sess2(sessions, analyspar, sesspar, stimpar, 
                                    logregpar, permpar, figpar)

    Runs timecourse decoding analyses (D vs U orientations).
        
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
        - logregpar (LogRegPar): 
            named tuple containing logistic regression parameters
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
        f"Compiling Gabor timecourse decoder performances for session 2.", 
        extra={"spacing": "\n"})


    gabor_timecourse_decoding(
        sessions, 
        analyspar=analyspar, 
        sesspar=sesspar, 
        stimpar=stimpar, 
        logregpar=logregpar, 
        permpar=permpar, 
        figpar=figpar, 
        seed=seed, 
        parallel=parallel,
        )
    

#############################################
def gabor_timecourse_decoding_sess3(sessions, analyspar, sesspar, stimpar, 
                                    logregpar, permpar, figpar, seed=None, 
                                    parallel=False):
    """
    gabor_timecourse_decoding_sess3(sessions, analyspar, sesspar, stimpar, 
                                    logregpar, permpar, figpar)

    Runs timecourse decoding analyses (D vs U orientations).
        
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
        - logregpar (LogRegPar): 
            named tuple containing logistic regression parameters
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
        f"Compiling Gabor timecourse decoder performances for session 3.", 
        extra={"spacing": "\n"})


    gabor_timecourse_decoding(
        sessions, 
        analyspar=analyspar, 
        sesspar=sesspar, 
        stimpar=stimpar, 
        logregpar=logregpar, 
        permpar=permpar, 
        figpar=figpar, 
        seed=seed, 
        parallel=parallel,
        )
    