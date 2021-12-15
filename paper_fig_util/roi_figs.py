"""
roi_figs.py

This script contains functions defining ROI mask and projection figure panel 
analyses.

Authors: Colleen Gillon

Date: January, 2021

Note: this code uses python 3.7.
"""

import logging

from util import logger_util, gen_util
from analysis import misc_analys, roi_analys
from paper_fig_util import helper_fcts

logger = logging.getLogger(__name__)


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
        crop_info=False,
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
    
