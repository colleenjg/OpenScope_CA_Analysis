"""
misc_figs.py

This script contains functions defining miscellaneous figure panel analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

from util import gen_util, logger_util
from analysis import misc_analys
from paper_fig_util import helper_fcts

logger = logging.getLogger(__name__)

TAB = "    "


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

    snr_df = misc_analys.get_check_sess_df(sessions, analyspar=analyspar)
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

    sig_mean_df = misc_analys.get_check_sess_df(sessions, analyspar=analyspar)
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


#############################################
def nrois_sess123(sessions, analyspar, sesspar, figpar): 
    """
    nrois_sess123(sessions, analyspar, sesspar, figpar)

    Retrieves number of ROIs for sessions 1 to 3.
        
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
    """

    logger.info("Compiling ROI numbers from session 1 to 3.", 
        extra={"spacing": "\n"})

    nrois_df = misc_analys.get_check_sess_df(sessions, analyspar=analyspar)

    extrapar = dict()

    info = {"analyspar": analyspar._asdict(),
            "sesspar"  : sesspar._asdict(),
            "extrapar" : extrapar,
            "nrois_df": nrois_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)


############################################
def roi_crosscorr_sess123(sessions, analyspar, sesspar, figpar, parallel=False):
    """
    roi_crosscorr_sess123(sessions, analyspar, sesspar, figpar)

    Retrieves ROI cross-correlation values for sessions 1 to 3.
        
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

    logger.info("Compiling ROI cross-correlations from session 1 to 3.", 
        extra={"spacing": "\n"})

    logger.info("Calculating ROI cross-correlations for each session...", 
        extra={"spacing": TAB})
    crosscorr_df = misc_analys.get_all_crosscorrelations(
        sessions, 
        analyspar=analyspar,  
        parallel=parallel
        )

    extrapar = dict()

    info = {"analyspar"   : analyspar._asdict(),
            "sesspar"     : sesspar._asdict(),
            "extrapar"    : extrapar,
            "crosscorr_df": crosscorr_df.to_dict()
            }

    helper_fcts.plot_save_all(info, figpar)
                
