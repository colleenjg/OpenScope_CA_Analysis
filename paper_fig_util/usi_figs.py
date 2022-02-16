"""
usi_figs.py

This script contains functions defining USI figure panel analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

from util import gen_util, logger_util
from sess_util import sess_gen_util, sess_ntuple_util
from analysis import usi_analys, misc_analys
from paper_fig_util import helper_fcts


TAB = "    "
TARGET_ROI_PERC = 99.8


logger = logger_util.get_module_logger(name=__name__)


############################################
def gabor_example_roi_usis(sessions, analyspar, sesspar, stimpar, basepar, 
                           idxpar, permpar, figpar, seed=None, parallel=False):
    """
    gabor_example_roi_usis(sessions, analyspar, sesspar, stimpar, basepar, 
                           idxpar, permpar, figpar)

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
        idxpar=idxpar,
        permpar=permpar,
        target_idx_vals = [0.5, 0, -0.5],
        target_idx_sigs = ["sig", "not_sig", "sig"],
        randst=seed,
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
        "idxpar": idxpar._asdict(),
        "permpar": permpar._asdict(),
        "extrapar": extrapar,
        "chosen_rois_df": chosen_rois_df.to_dict()
    }

    helper_fcts.plot_save_all(info, figpar)
    

############################################
def gabor_example_roi_usi_sig(sessions, analyspar, sesspar, stimpar, basepar, 
                              idxpar, permpar, figpar, seed=None):
    """
    gabor_example_roi_usi_sig(sessions, analyspar, sesspar, stimpar, basepar, 
                              idxpar, permpar, figpar)

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
    """

    target_roi_perc = TARGET_ROI_PERC

    logger.info(
        ("Compiling Gabor ROI USIs, and identifying an example at or near "
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
        idxpar=idxpar, 
        permpar=permpar, 
        randst=seed, 
        target_roi_perc=target_roi_perc
        )

    extrapar = {"seed": seed}
    
    info = {
        "analyspar": analyspar._asdict(),
        "stimpar"  : stimpar._asdict(),
        "sesspar"  : sesspar._asdict(),
        "basepar"  : basepar._asdict(),
        "idxpar"   : idxpar._asdict(),
        "permpar"  : permpar._asdict(),
        "extrapar" : extrapar,
        "ex_idx_df": ex_idx_df.to_dict()
    }
    
    helper_fcts.plot_save_all(info, figpar)
    

############################################
def gabor_roi_usi_distr(sessions, analyspar, sesspar, stimpar, basepar, 
                        idxpar, permpar, figpar, seed=None, parallel=False):
    """
    gabor_roi_usi_distr(sessions, analyspar, sesspar, stimpar, basepar, 
                        idxpar, permpar, figpar)

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

    logger.info("Compiling Gabor ROI USI distributions.", 
        extra={"spacing": "\n"}
        )

    n_bins = 40
    idx_df = usi_analys.get_idx_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar, 
        n_bins=n_bins,
        randst=seed, 
        parallel=parallel
        )

    extrapar = {
        "n_bins"  : n_bins,
        "seed"    : seed,
        }

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "stimpar"  : stimpar._asdict(),
            "basepar"  : basepar._asdict(),
            "idxpar"   : idxpar._asdict(),
            "permpar"  : permpar._asdict(),
            "extrapar" : extrapar,
            "idx_df"   : idx_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)
    

############################################
def gabor_roi_usi_sig(sessions, analyspar, sesspar, stimpar, basepar, 
                      idxpar, permpar, figpar, common_oris=False, seed=None, 
                      parallel=False):
    """
    gabor_roi_usi_sig(sessions, analyspar, sesspar, stimpar, basepar, 
                      idxpar, permpar, figpar)

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
        - idxpar (IdxPar): 
            named tuple containing index parameters
        - permpar (PermPar): 
            named tuple containing permutation parameters
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - common_oris (bool): 
            if True, data is for common orientations
            default: False
        - seed (int): 
            seed value to use. (-1 treated as None)
            default: None
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    common_str = ", with common orientations" if common_oris else ""

    logger.info(
        f"Compiling percentages of significant Gabor USIs{common_str}.", 
        extra={"spacing": "\n"}
        )

    if common_oris:
        gab_ori = sess_gen_util.gab_oris_common_U(stimpar.gab_ori)
        stimpar = sess_ntuple_util.get_modif_ntuple(
            stimpar, "gab_ori", gab_ori
            )

    by_mouse = False
    idx_df = usi_analys.get_idx_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar, 
        common_oris=common_oris,
        by_mouse=by_mouse,
        randst=seed, 
        parallel=parallel,
        )

    permpar = misc_analys.set_multcomp(permpar, sess_df=idx_df, factor=2)
    
    perc_sig_df = usi_analys.get_perc_sig_df(idx_df, analyspar, permpar, seed)

    extrapar = {
        "common_oris": common_oris,
        "by_mouse"   : by_mouse,
        "seed"       : seed,
        }

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "stimpar"    : stimpar._asdict(),
            "basepar"    : basepar._asdict(),
            "idxpar"     : idxpar._asdict(),
            "permpar"    : permpar._asdict(),
            "extrapar"   : extrapar,
            "perc_sig_df": perc_sig_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


############################################
def gabor_roi_usi_sig_common_oris(sessions, analyspar, sesspar, stimpar, 
                                  basepar, idxpar, permpar, figpar, seed=None, 
                                  parallel=False):
    """
    gabor_roi_usi_sig_common_oris(sessions, analyspar, sesspar, stimpar, 
                                  basepar, idxpar, permpar, figpar)

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

    gabor_roi_usi_sig(
        sessions, 
        analyspar=analyspar, 
        sesspar=sesspar,
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar, 
        figpar=figpar,
        common_oris=True,
        seed=seed, 
        parallel=parallel,
        )


############################################
def gabor_tracked_roi_usis_sess123(sessions, analyspar, sesspar, stimpar, 
                                   basepar, idxpar, figpar, parallel=False):
    """
    gabor_tracked_roi_usis_sess123(sessions, analyspar, sesspar, stimpar, 
                                   basepar, idxpar, figpar)

    Retrieves tracked ROI Gabor USIs for session 1 to 3.
        
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
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    logger.info("Compiling tracked ROI Gabor USIs for sessions 1 to 3.", 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    idx_only_df = usi_analys.get_idx_only_df(
        sessions, 
        analyspar=analyspar,
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        parallel=parallel
        )
        
    extrapar = dict()

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "stimpar"    : stimpar._asdict(),
            "basepar"    : basepar._asdict(),
            "idxpar"     : idxpar._asdict(),
            "extrapar"   : extrapar,
            "idx_only_df": idx_only_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)
    

############################################
def gabor_tracked_roi_abs_usi_means_sess123(
        sessions, analyspar, sesspar, stimpar, basepar, idxpar, permpar, 
        figpar, seed=None, parallel=False):
    """
    gabor_tracked_roi_abs_usi_means_sess123(
        sessions, analyspar, sesspar, stimpar, basepar, idxpar, permpar, 
        figpar)

    Retrieves mean absolute for tracked ROI Gabor USIs for session 1 to 3.
        
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
        ("Compiling absolute means of tracked ROI Gabor USIs for "
        "sessions 1 to 3."), 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    # calculate multiple comparisons
    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes", "sess_ns"])

    permpar = misc_analys.set_multcomp(permpar, sess_df=dummy_df, CIs=False)

    absolute = True
    by_mouse = False
    idx_stats_df = usi_analys.get_idx_stats_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar, 
        absolute=absolute, 
        by_mouse=by_mouse, 
        randst=seed,
        parallel=parallel, 
        )

    extrapar = {
        "absolute": absolute,
        "by_mouse": by_mouse,
        "seed"    : seed,
        }

    info = {"analyspar"   : analyspar._asdict(),
            "sesspar"     : sesspar._asdict(),
            "stimpar"     : stimpar._asdict(),
            "basepar"     : basepar._asdict(),
            "idxpar"      : idxpar._asdict(),
            "permpar"     : permpar._asdict(),
            "extrapar"    : extrapar,
            "idx_stats_df": idx_stats_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


############################################
def gabor_tracked_roi_usi_variances_sess123(
        sessions, analyspar, sesspar, stimpar, basepar, idxpar, permpar, 
        figpar, seed=None, parallel=False):
    """
    gabor_tracked_roi_usi_variances_sess123(
        sessions, analyspar, sesspar, stimpar, basepar, idxpar, permpar, 
        figpar)

    Retrieves tracked ROI Gabor USI variances for session 1 to 3.
        
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
        "Compiling tracked ROI Gabor USI variances for sessions 1 to 3.", 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    # calculate multiple comparisons
    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes", "sess_ns"])

    permpar = misc_analys.set_multcomp(permpar, sess_df=dummy_df, CIs=False)

    absolute = False
    by_mouse = False
    stat = "var"
    idx_stats_df = usi_analys.get_idx_stats_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar, 
        absolute=absolute, 
        by_mouse=by_mouse, 
        stat=stat,
        randst=seed,
        parallel=parallel, 
        )

    extrapar = {
        "absolute": absolute,
        "by_mouse": by_mouse,
        "seed"    : seed,
        "stat"    : stat,
        }

    info = {"analyspar"   : analyspar._asdict(),
            "sesspar"     : sesspar._asdict(),
            "stimpar"     : stimpar._asdict(),
            "basepar"     : basepar._asdict(),
            "idxpar"      : idxpar._asdict(),
            "permpar"     : permpar._asdict(),
            "extrapar"    : extrapar,
            "idx_stats_df": idx_stats_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)

                
############################################
def gabor_roi_usi_sig_by_mouse(sessions, analyspar, sesspar, stimpar, basepar, 
                               idxpar, permpar, figpar, seed=None, 
                               parallel=False):
    """
    gabor_roi_usi_sig_by_mouse(sessions, analyspar, sesspar, stimpar, basepar, 
                               idxpar, permpar, figpar)

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
        f"Compiling percentages of significant Gabor USIs by mouse.", 
        extra={"spacing": "\n"}
        )

    by_mouse = True
    idx_df = usi_analys.get_idx_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar, 
        by_mouse=by_mouse,
        randst=seed, 
        parallel=parallel,
        )

    permpar = misc_analys.set_multcomp(
        permpar, sess_df=idx_df, pairs=False, factor=2
        )
    
    perc_sig_df = usi_analys.get_perc_sig_df(idx_df, analyspar, permpar, seed)

    extrapar = {
        "by_mouse": by_mouse,
        "seed"    : seed,
        }

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "stimpar"    : stimpar._asdict(),
            "basepar"    : basepar._asdict(),
            "idxpar"     : idxpar._asdict(),
            "permpar"    : permpar._asdict(),
            "extrapar"   : extrapar,
            "perc_sig_df": perc_sig_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)

    
############################################
def gabor_tracked_roi_abs_usi_means_sess123_by_mouse(
        sessions, analyspar, sesspar, stimpar, basepar, idxpar, figpar, 
        parallel=False):
    """
    gabor_tracked_roi_abs_usi_means_sess123_by_mouse(
        sessions, analyspar, sesspar, stimpar, basepar, idxpar, figpar)

    Retrieves mean absolute for tracked ROI Gabor USIs, for each mouse, for 
    session 1 to 3.
        
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
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    logger.info(
        ("Compiling absolute means per mouse of tracked ROI Gabor USIs for "
        "sessions 1 to 3."), 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    absolute = True
    by_mouse = True
    idx_stats_df = usi_analys.get_idx_stats_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        absolute=absolute, 
        by_mouse=by_mouse, 
        parallel=parallel, 
        )

    extrapar = {
        "absolute": absolute,
        "by_mouse": by_mouse,
        }

    info = {"analyspar"   : analyspar._asdict(),
            "sesspar"     : sesspar._asdict(),
            "stimpar"     : stimpar._asdict(),
            "basepar"     : basepar._asdict(),
            "idxpar"      : idxpar._asdict(),
            "extrapar"    : extrapar,
            "idx_stats_df": idx_stats_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


############################################
def visual_flow_tracked_roi_usis_sess123(sessions, analyspar, sesspar, stimpar, 
                                         basepar, idxpar, figpar, 
                                         parallel=False):
    """
    visual_flow_tracked_roi_usis_sess123(sessions, analyspar, sesspar, stimpar, 
                                         basepar, idxpar, figpar)

    Retrieves tracked ROI visual flow USIs for session 1 to 3.
        
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
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    logger.info("Compiling tracked ROI visual flow USIs for sessions 1 to 3.", 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    idx_only_df = usi_analys.get_idx_only_df(
        sessions, 
        analyspar=analyspar,
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        parallel=parallel
        )
        
    extrapar = dict()

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "stimpar"    : stimpar._asdict(),
            "basepar"    : basepar._asdict(),
            "idxpar"     : idxpar._asdict(),
            "extrapar"   : extrapar,
            "idx_only_df": idx_only_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)
    

############################################
def visual_flow_tracked_roi_abs_usi_means_sess123(
        sessions, analyspar, sesspar, stimpar, basepar, idxpar, permpar, 
        figpar, seed=None, parallel=False):
    """
    visual_flow_tracked_roi_abs_usi_means_sess123(
        sessions, analyspar, sesspar, stimpar, basepar, idxpar, permpar, 
        figpar)

    Retrieves mean absolute for tracked ROI visual flow USIs for session 1 to 3.
        
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
        ("Compiling absolute means of tracked ROI Gabor USIs for "
        "sessions 1 to 3."), 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    # calculate multiple comparisons
    dummy_df = misc_analys.get_check_sess_df(
        sessions, None, analyspar).drop_duplicates(
            subset=["lines", "planes", "sess_ns"])

    permpar = misc_analys.set_multcomp(permpar, sess_df=dummy_df, CIs=False)

    absolute = True
    by_mouse = False
    idx_stats_df = usi_analys.get_idx_stats_df(
        sessions, 
        analyspar=analyspar, 
        stimpar=stimpar, 
        basepar=basepar, 
        idxpar=idxpar, 
        permpar=permpar, 
        absolute=absolute, 
        by_mouse=by_mouse, 
        randst=seed,
        parallel=parallel, 
        )

    extrapar = {
        "absolute": absolute,
        "by_mouse": by_mouse,
        "seed": seed,
        }

    info = {"analyspar"   : analyspar._asdict(),
            "sesspar"     : sesspar._asdict(),
            "stimpar"     : stimpar._asdict(),
            "basepar"     : basepar._asdict(),
            "idxpar"      : idxpar._asdict(),
            "permpar"     : permpar._asdict(),
            "extrapar"    : extrapar,
            "idx_stats_df": idx_stats_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)

