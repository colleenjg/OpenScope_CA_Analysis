"""
usi_figs.py

This script contains functions defining USI figure panel analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

from util import gen_util, logger_util
from sess_util import sess_gen_util, sess_ntuple_util
from analysis import usi_analys, misc_analys
from paper_fig_util import helper_fcts

logger = logging.getLogger(__name__)

TAB = "    "


############################################
def gabor_example_roi_usis(sessions, analyspar, sesspar, stimpar, basepar, 
                           permpar, idxpar, figpar, seed=None, parallel=False):
    """
    gabor_example_roi_usis(sessions, analyspar, sesspar, stimpar, basepar, 
                           permpar, idxpar, figpar)

    Retrieves example ROI Gabor USI traces.
        
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
        - idxpar (IdxPar): 
            named tuple containing index parameters
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

    logger.info("Compiling Gabor example ROI USI data.", 
        extra={"spacing": "\n"}
        )

    # check stimpar.pre
    if (not isinstance(stimpar.pre, list)) or (len(stimpar.pre) == 1):
        pre_list = gen_util.list_if_not(stimpar.pre)
        stimpar = sess_ntuple_util.get_modif_ntuple(stimpar, "pre", pre_list)
    
    elif len(stimpar.pre) != 2:
        raise ValueError("Expected 2 values for stimpar.pre: one for "
            "index calculation and one for traces.")

    # use first stimpar.pre for idx calculation
    stimpar_idx = sess_ntuple_util.get_modif_ntuple(
            stimpar, "pre", stimpar.pre[0]
         )

    chosen_rois_df = usi_analys.get_chosen_roi_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar_idx, 
        basepar=basepar, 
        permpar=permpar,
        idxpar=idxpar,
        target_idx_vals = [0.5, 0, -0.5],
        target_idx_sigs = ["sig", "not_sig", "sig"],
        seed=seed,
        parallel=parallel
        )

    # use second stimpar.pre for traces
    stimpar_tr = sess_ntuple_util.get_modif_ntuple(
        stimpar, "pre", stimpar.pre[1]
        )

    chosen_rois_df = usi_analys.add_chosen_roi_traces(
        sessions, 
        chosen_rois_df, 
        analyspar=analyspar, 
        stimpar=stimpar_tr, 
        basepar=basepar, 
        split=idxpar.feature, 
        parallel=parallel
        )
    
    extrapar = {"seed": seed}

    info = {
        "analyspar": analyspar._asdict(),
        "stimpar": stimpar._asdict(),
        "sesspar": sesspar._asdict(),
        "basepar": basepar._asdict(),
        "permpar": permpar._asdict(),
        "idxpar": idxpar._asdict(),
        "extrapar": extrapar,
        "chosen_rois_df": chosen_rois_df.to_dict()
    }

    helper_fcts.plot_save_all(info, figpar)
    

############################################
def gabor_example_roi_usi_sig(sessions, analyspar, sesspar, stimpar, basepar, 
                              permpar, idxpar, figpar, seed=None):
    """
    gabor_example_roi_usi_sig(sessions, analyspar, sesspar, stimpar, basepar, 
                              permpar, idxpar, figpar)

    Retrieves example ROI Gabor USI null distribution for significance 
    assessment.
        
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list): 
            Session objects (singleton)
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
        - idxpar (IdxPar): 
            named tuple containing index parameters
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None
    """

    target_roi_perc = 99.8
    logger.info(
        ("Calculating Gabor ROI USIs, and identifying an example at or near "
        f"the {target_roi_perc} percentile."), 
        extra={"spacing": "\n"}
        )

    if len(sessions) != 1:
        raise ValueError("Expected to receive only 1 session.")
    
    ex_idx_df = usi_analys.get_ex_idx_df(
        sessions[0], 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        permpar=permpar, 
        idxpar=idxpar, 
        seed=seed, 
        target_roi_perc=target_roi_perc
        )

    extrapar = {"seed": seed}
    
    info = {
        "analyspar": analyspar._asdict(),
        "stimpar"  : stimpar._asdict(),
        "sesspar"  : sesspar._asdict(),
        "basepar"  : basepar._asdict(),
        "permpar"  : permpar._asdict(),
        "idxpar"   : idxpar._asdict(),
        "extrapar" : extrapar,
        "ex_idx_df": ex_idx_df.to_dict()
    }
    
    helper_fcts.plot_save_all(info, figpar)
    

############################################
def gabor_roi_usi_distr(sessions, analyspar, sesspar, stimpar, basepar, 
                        permpar, idxpar, figpar, seed=None, parallel=False):
    """
    gabor_roi_usi_distr(sessions, analyspar, sesspar, stimpar, basepar, 
                        permpar, idxpar, figpar)

    Retrieves ROI Gabor USI percentile distributions.
        
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
        - idxpar (IdxPar): 
            named tuple containing index parameters
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

    logger.info("Calculating Gabor ROI USI distributions.", 
        extra={"spacing": "\n"}
        )

    n_bins = 40
    idx_df = usi_analys.get_idx_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        permpar=permpar, 
        idxpar=idxpar, 
        n_bins=n_bins,
        seed=seed, 
        parallel=parallel
        )

    extrapar = {"seed": seed}

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "basepar"  : basepar._asdict(),
            "permpar"  : permpar._asdict(),
            "idxpar"   : idxpar._asdict(),
            "extrapar" : extrapar,
            "idx_df"   : idx_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)
    

############################################
def gabor_roi_usi_sig(sessions, analyspar, sesspar, stimpar, basepar, 
                      permpar, idxpar, figpar, seed=None, parallel=False, 
                      common_oris=False):
    """
    gabor_roi_usi_sig(sessions, analyspar, sesspar, stimpar, basepar, 
                      permpar, idxpar, figpar)

    Retrieves percentage of signifiant ROI Gabor USIs.
        
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
        - idxpar (IdxPar): 
            named tuple containing index parameters
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
        - common_oris (bool): 
            if True, data is for common orientations
            default: False
    """

    common_str = ", with common orientations" if common_oris else ""
    logger.info(
        f"Calculating percentages of significant Gabor USIs{common_str}.", 
        extra={"spacing": "\n"}
        )

    if common_oris:
        gab_oris = sess_gen_util.gab_oris_common_U(["D", "U"], stimpar.gab_ori)
        stimpar = sess_ntuple_util.get_modif_ntuple(
            stimpar, "gab_ori", gab_oris
            )

    idx_df = usi_analys.get_idx_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        permpar=permpar, 
        idxpar=idxpar, 
        seed=seed, 
        common_oris=common_oris,
        parallel=parallel,
        )

    permpar = misc_analys.set_multcomp(permpar, sess_df=idx_df, factor=2)
    
    perc_sig_df = usi_analys.get_perc_sig_df(idx_df, permpar, seed)

    extrapar = {"seed": seed}

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "stimpar"    : stimpar._asdict(),
            "basepar"    : basepar._asdict(),
            "permpar"    : permpar._asdict(),
            "idxpar"     : idxpar._asdict(),
            "extrapar"   : extrapar,
            "perc_sig_df": perc_sig_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


############################################
def gabor_roi_usi_sig_common_oris(sessions, analyspar, sesspar, stimpar, 
                                  basepar, permpar, idxpar, figpar, seed=None, 
                                  parallel=False):
    """
    gabor_roi_usi_sig_common_oris(sessions, analyspar, sesspar, stimpar, 
                                  basepar, permpar, idxpar, figpar)

    Retrieves percentage of signifiant ROI Gabor USIs, measured using sequences 
    with mean orientation common to D and U sequences.
        
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
        - idxpar (IdxPar): 
            named tuple containing index parameters
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

    gabor_roi_usi_sig(
        sessions, 
        analyspar=analyspar, 
        sesspar=sesspar,
        stimpar=stimpar, 
        basepar=basepar, 
        permpar=permpar, 
        idxpar=idxpar, 
        figpar=figpar,
        seed=seed, 
        parallel=parallel,
        common_oris=True
        )


############################################
def gabor_tracked_roi_usis_sess123():
    return
    

############################################
def gabor_tracked_roi_usi_means_sess123():
    return


############################################
def gabor_roi_usi_sig_by_mouse(sessions, analyspar, sesspar, stimpar, basepar, 
                               permpar, idxpar, figpar, seed=None, 
                               parallel=False):
    """
    gabor_roi_usi_sig_by_mouse(sessions, analyspar, sesspar, stimpar, basepar, 
                               permpar, idxpar, figpar)

    Retrieves percentage of signifiant ROI Gabor USIs by mouse.
        
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
        - idxpar (IdxPar): 
            named tuple containing index parameters
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
        f"Calculating percentages of significant Gabor USIs by mouse.", 
        extra={"spacing": "\n"}
        )

    idx_df = usi_analys.get_idx_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        permpar=permpar, 
        idxpar=idxpar, 
        seed=seed, 
        by_mouse=True,
        parallel=parallel,
        )

    permpar = misc_analys.set_multcomp(
        permpar, sess_df=idx_df, pairs=False, factor=2
        )
    
    perc_sig_df = usi_analys.get_perc_sig_df(idx_df, permpar, seed)

    extrapar = {"seed": seed}

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "stimpar"    : stimpar._asdict(),
            "basepar"    : basepar._asdict(),
            "permpar"    : permpar._asdict(),
            "idxpar"     : idxpar._asdict(),
            "extrapar"   : extrapar,
            "perc_sig_df": perc_sig_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)

    
############################################
def gabor_tracked_roi_means_sess123_by_mouse():
    return
    

############################################
def visual_flow_tracked_roi_usis_sess123():
    return
    

############################################
def visual_flow_tracked_roi_usi_means_sess123_by_mouse():
    return


############################################
def tracked_roi_usis_stimulus_comp_sess1v3():
    return
    
