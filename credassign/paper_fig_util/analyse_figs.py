"""
analyse_figs.py

This script contains functions defining figure panel analyses.

Authors: Colleen Gillon

Date: February 2023

Note: this code was aggregated from https://github.com/colleenjg/OpenScope_CA_Analysis.
"""

from credassign.util import gen_util, logger_util

from credassign.analysis import behav_analys, misc_analys, roi_analys, seq_analys
from credassign.paper_fig_util import helper_fcts


logger = logger_util.get_module_logger(name=__name__)

TAB = "    "


#############################################
def imaging_planes(sessions, sesspar, figpar, parallel=False):
    """
    imaging_planes(sessions, sesspar, figpar)

    Retrieves imaging plane image examples.
        
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - session (Session):
            Session object
        - sesspar (SessPar): 
            named tuple containing session parameters
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    logger.info("Compiling imaging plane projection examples.", 
        extra={"spacing": "\n"})

    imaging_plane_df = misc_analys.get_check_sess_df(sessions, roi=False)

    imaging_plane_df["max_projections"] = [
        sess.max_proj.tolist() for sess in sessions
    ]

    extrapar = dict()

    info = {"sesspar"         : sesspar._asdict(),
            "extrapar"        : extrapar,
            "imaging_plane_df": imaging_plane_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


############################################
def roi_tracking(sessions, analyspar, sesspar, figpar, parallel=False):
    """
    roi_tracking(sessions, analyspar, sesspar, figpar)

    Retrieves ROI mask tracking examples for sessions 1 to 3.
        
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - session (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    logger.info("Compiling ROI tracking examples.", 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    roi_mask_df = roi_analys.get_roi_tracking_df(
        sessions, 
        analyspar=analyspar, 
        reg_only=False,
        proj=True,
        crop_info="small",
        parallel=parallel,
        )
    roi_mask_df = roi_mask_df.drop(columns="registered_max_projections")

    extrapar = dict()

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "extrapar"   : extrapar,
            "roi_mask_df": roi_mask_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


############################################
def roi_overlays_sess123(sessions, analyspar, sesspar, figpar, parallel=False):
    """
    roi_overlays_sess123(sessions, analyspar, sesspar, figpar)

    Retrieves ROI mask overlay examples for sessions 1 to 3.
        
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - session (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    logger.info("Compiling ROI mask overlay examples.", 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    roi_mask_df = roi_analys.get_roi_tracking_df(
        sessions, 
        analyspar=analyspar, 
        reg_only=True,
        proj=False,
        crop_info="large",
        parallel=parallel,
        )

    extrapar = {
        "mark_crop_only": True
    }

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "extrapar"   : extrapar,
            "roi_mask_df": roi_mask_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


############################################
def roi_overlays_sess123_enlarged(sessions, analyspar, sesspar, figpar, 
                                  parallel=False):
    """
    roi_overlays_sess123_enlarged(sessions, analyspar, sesspar, figpar)

    Retrieves enlarged ROI mask overlay examples for sessions 1 to 3.
        
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - session (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    logger.info("Compiling enlarged ROI mask overlay examples.", 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    roi_mask_df = roi_analys.get_roi_tracking_df(
        sessions, 
        analyspar=analyspar, 
        reg_only=True,
        proj=False,
        crop_info="large",
        parallel=parallel,
        )

    extrapar = dict()


    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "extrapar"   : extrapar,
            "roi_mask_df": roi_mask_df.to_dict()
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

    # For dataset paper, so using "consistent" instead of "expected"
    logger.info(
        "Compiling example ROI responses to consistent Gabor sequence from "
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

    # For dataset paper, so using "inconsistent" instead of "unexpected"
    logger.info(
        "Compiling example ROI responses to inconsistent Gabor sequence from "
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

    # For dataset paper, so using "inconsistent" instead of "unexpected"
    logger.info(
        "Compiling example ROI responses to onset of inconsistent flow during "
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

    # For dataset paper, so using "inconsistent" instead of "unexpected"
    logger.info(
        "Compiling example ROI responses to onset of inconsistent flow during "
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
    

############################################
def pupil_run_full(sessions, analyspar, sesspar, figpar, parallel=False):
    """
    pupil_run_full(sessions, analyspar, sesspar, figpar)

    Retrieves pupil and running traces for an entire session.
        
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list): 
            Session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - figpar (dict): 
            dictionary containing figure parameters

    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    logger.info("Compiling pupil and running traces for a full session.", 
        extra={"spacing": "\n"})

    sess_df = behav_analys.get_pupil_run_full_df(
        sessions, 
        analyspar=analyspar, 
        parallel=parallel,
        )

    extrapar = dict()

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "extrapar" : extrapar,
            "sess_df"  : sess_df.to_dict()
    }

    helper_fcts.plot_save_all(info, figpar)


############################################
def pupil_run_histograms(sessions, analyspar, sesspar, figpar, parallel=False):
    """
    pupil_run_histograms(sessions, analyspar, sesspar, figpar)

    Retrieves pupil and running traces histograms across sessions.
        
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - sessions (list): 
            Session objects
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - figpar (dict): 
            dictionary containing figure parameters

    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """


    logger.info("Compiling pupil and running histograms across sessions.", 
        extra={"spacing": "\n"})

    hist_df = behav_analys.get_pupil_run_histograms(
        sessions, 
        analyspar=analyspar, 
        parallel=parallel,
        )

    extrapar = dict()

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "extrapar" : extrapar,
            "hist_df"  : hist_df.to_dict()
    }

    helper_fcts.plot_save_all(info, figpar)


#############################################
def snrs_sess123(sessions, analyspar, sesspar, figpar, parallel=False):
    """
    snrs_sess123(sessions, analyspar, sesspar, figpar)

    Retrieves ROI SNR values for sessions 1 to 3.
        
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - session (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    logger.info("Compiling ROI SNRs from session 1 to 3.", 
        extra={"spacing": "\n"})

    logger.info("Calculating ROI SNRs for each session...", 
        extra={"spacing": TAB})
    all_snrs = gen_util.parallel_wrap(
        misc_analys.get_snr, sessions, [analyspar, "snrs"], parallel=parallel
        )

    snr_df = misc_analys.get_check_sess_df(sessions)
    snr_df["snrs"] = [snr.tolist() for snr in all_snrs]


    extrapar = dict()

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "extrapar" : extrapar,
            "snr_df"   : snr_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


#############################################
def mean_signal_sess123(sessions, analyspar, sesspar, figpar, parallel=False):
    """
    mean_signal_sess123(sessions, analyspar, sesspar, figpar)

    Retrieves ROI mean signal values for sessions 1 to 3.
        
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - session (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    logger.info("Compiling ROI signal means from session 1 to 3.", 
        extra={"spacing": "\n"})

    logger.info("Calculating ROI signal means for each session...", 
        extra={"spacing": TAB})
    all_signal_means = gen_util.parallel_wrap(
        misc_analys.get_snr, sessions, [analyspar, "signal_means"], 
        parallel=parallel
        )

    sig_mean_df = misc_analys.get_check_sess_df(sessions)
    sig_mean_df["signal_means"] = [
        sig_mean.tolist() for sig_mean in all_signal_means
        ]

    extrapar = dict()

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "extrapar"   : extrapar,
            "sig_mean_df": sig_mean_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


############################################
def roi_corr_sess123(sessions, analyspar, sesspar, figpar, parallel=False):
    """
    roi_corr_sess123(sessions, analyspar, sesspar, figpar)

    Retrieves ROI correlation values for sessions 1 to 3.
        
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - session (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    logger.info("Compiling ROI correlations from session 1 to 3.", 
        extra={"spacing": "\n"})

    logger.info("Calculating ROI correlations for each session...", 
        extra={"spacing": TAB})

    rolling_win = 4
    corr_df = misc_analys.get_all_correlations(
        sessions, 
        analyspar=analyspar,
        rolling_win=rolling_win,
        parallel=parallel
        )

    extrapar = {
        "rolling_win": rolling_win,
    }

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "extrapar" : extrapar,
            "corr_df"  : corr_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)
                

############################################
def dendritic_roi_tracking_example(sessions, analyspar, sesspar, figpar, 
                                   parallel=False):
    """
    dendritic_roi_tracking_example(sessions, analyspar, sesspar, figpar)

    Retrieves dendritic tracking examples for sessions 1 to 3.
        
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - session (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    logger.info("Compiling dendritic tracking examples.", 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    roi_mask_df = roi_analys.get_roi_tracking_ex_df(
        sessions, 
        analyspar=analyspar, 
        parallel=parallel,
        )

    extrapar = dict()

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "extrapar"   : extrapar,
            "roi_mask_df": roi_mask_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)
    

############################################
def somatic_roi_tracking_example(sessions, analyspar, sesspar, figpar, 
                                 parallel=False):
    """
    somatic_roi_tracking_example(sessions, analyspar, sesspar, figpar)

    Retrieves somatic tracking examples for sessions 1 to 3.
        
    Saves results and parameters relevant to analysis in a dictionary.

    Required args:
        - session (Session):
            Session object
        - analyspar (AnalysPar): 
            named tuple containing analysis parameters
        - sesspar (SessPar): 
            named tuple containing session parameters
        - figpar (dict): 
            dictionary containing figure parameters
    
    Optional args:
        - parallel (bool): 
            if True, some of the analysis is run in parallel across CPU cores 
            default: False
    """

    logger.info("Compiling dendritic tracking examples.", 
        extra={"spacing": "\n"})

    if not analyspar.tracked:
        raise ValueError("analyspar.tracked should be set to True.")

    # remove incomplete session series and warn
    sessions = misc_analys.check_sessions_complete(sessions)

    roi_mask_df = roi_analys.get_roi_tracking_ex_df(
        sessions, 
        analyspar=analyspar, 
        parallel=parallel,
        )

    extrapar = dict()

    info = {"analyspar"  : analyspar._asdict(),
            "sesspar"    : sesspar._asdict(),
            "extrapar"   : extrapar,
            "roi_mask_df": roi_mask_df.to_dict()
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

