"""
sess_util.py

This module contains session utility functions.

Authors: Colleen Gillon

Date: February 2023

Note: this code was aggregated from https://github.com/colleenjg/util.

"""

import copy
from collections import namedtuple
from pathlib import Path

import pandas as pd


from util import gen_util, logger_util


logger = logger_util.get_module_logger(name=__name__)



#############################################
def get_visflow_screen_mouse_direc(direc="right"):
    """
    get_visflow_screen_mouse_direc()

    Returns direction for screen and mouse.

    Optional args:
        - direc (str): direction

    Returns:
        - direc (str): direction wrt screen and mouse
    """

    if "right" in direc or "temp" in direc:
        direc = "right (temp)"
    elif "left" in direc or "nasal" in direc:
        direc = "left (nasal)"
    elif "right" in direc and "nasal" in direc:
        raise ValueError(
            f"Invalid visual flow direction {direc}, "
            "as rightward motion is temp."
            )
    elif "left" in direc and "temp" in direc:
        raise ValueError(
            f"Invalid visual flow direction {direc}, "
            "as leftward motion is nasal."
            )
    else:
        raise ValueError(f"Visual flow direction {direc} not recognized.")

    return direc


#############################################
def get_param_vals(param="gabk", gabfr_lett=False):
    """
    get_param_vals()

    Returns all possible parameter values for the requested parameter.

    Optional args:
        - param (str)      : parameter name
                             default: "gabk"
    
        - gabfr_lett (bool): if True, and gabor frames are requested, 
                             Gabor frame letters (A, B, C, D, U, G) are 
                             returned instead of numbers (0, 1, 2, 3, 4)
                             default: False

    Returns:
        - param_vals (list): parameter values
    """
    
    if param in ["gabk", "gabor_kappa"]:
        param_vals = [4, 16]
    elif param in ["gab_ori", "gabor_orientation", "gabor_mean_orientation"]:
        param_vals = [0, 45, 90, 135, 180, 225]
    elif param in ["gabfr", "gabor_frame"]:
        if gabfr_lett:
            param_vals = ["A", "B", "C", "D", "U", "G"]
        else:
            param_vals = [0, 1, 2, 3, 4]
    elif param in ["gabor_number", "gab_nbr"]:
        param_vals = [30]
    elif param in ["visflow_size", "square_size"]:
        param_vals = [128, 256]
    elif param in ["visflow_number", "square_number"]:
        param_vals = [105, 26]
    elif param in ["visflow_dir", "main_flow_direction"]:
        param_vals = [
            get_visflow_screen_mouse_direc(direc) for direc in ["right", "left"]
            ]
    elif param in ["square_proportion_flipped"]:
        param_vals = [0, 0.25]
    else:
        raise ValueError(f"{param} not recognized as a parameter.")

    return param_vals


#############################################
def get_params(stimtype="both", visflow_dir="both", visflow_size=128, gabfr=0, 
               gabk=16, gab_ori="all"):
    """
    get_params()

    Gets and formats full parameters. For example, replaces "both" with a 
    list of parameter values, and sets parameters irrelevant to the stimulus
    of interest to "none".

    Required args:
        - stimtype  (str)            : stimulus to analyse 
                                       ("visflow", "gabors", "both")
        - visflow_dir (str or list)  : visual flow direction values 
                                       ("right", "left", "both")
        - visflow_size (int, str or list): visual flow square size values 
                                       (128, 256, "both")
        - gabfr (int, list or str)   : gabor frame value (0, 1, 2, 3, "0_3", 
                                                          [0, 3])
        - gabk (int, str or list)    : gabor kappa values (4, 16, "both")
        - gab_ori (int, str or list) : gabor orientation values 
                                       (0, 45, 90, 135, 180, 225 or "all")

    Returns:
        - visflow_dir (str or list) : visual flow direction values
        - visflow_size (int or list): visual flow square size values
        - gabfr (int or list)       : gabor frame values
        - gabk (int or list)        : gabor kappa values
        - gab_ori (int or list)     : gabor orientation values

    """

    # get all the parameters
    if gabk in ["both", "any", "all"]:
        gabk = get_param_vals("gabk")
    else:
        gabk = int(gabk)

    if gab_ori in ["any", "all"]:
        gab_ori = get_param_vals("gab_ori")
    elif not isinstance(gab_ori, list):
        gab_ori = int(gab_ori)

    if gabfr in ["any", "all"]:
        gabfr = get_param_vals("gabfr", gabfr_lett=False)
    elif not isinstance(gabfr, (list, int)) and gabfr.isdigit():
        gabfr = int(gabfr)

    if visflow_size in ["both", "any", "all"]:
        visflow_size = get_param_vals("visflow_size")
    else:
        visflow_size = int(visflow_size)

    if visflow_dir in ["both", "any", "all"]:
        visflow_dir = get_param_vals("visflow_dir")

    # set to "none" any parameters that are irrelevant
    if stimtype == "gabors":
        visflow_size = "none"
        visflow_dir = "none"
    elif stimtype == "visflow":
        gabfr = "none"
        gabk = "none"
        gab_ori = "none"
    elif stimtype != "both" and set(stimtype) != set(["gabors", "visflow"]):
        gen_util.accepted_values_error(
            "stim argument", stimtype, ["gabors", "visflow"])

    return visflow_dir, visflow_size, gabfr, gabk, gab_ori


#############################################
def gab_oris_common_U(gab_oris):
    """
    gab_oris_common_U(gab_oris)

    Returns Gabor orientations that are common to U frames and other frames.

    Required args:
        - gab_oris (list): Gabor orientations that can be included

    Returns:
        - new_oris (list): orientations common to U and other frames
    """

    gab_oris = get_params(gab_ori=gab_oris)[-1]

    common_oris = [90, 135] # oris common to ABCD and U frames

    new_oris = [ori for ori in common_oris if ori in gab_oris]

    return new_oris


#############################################
def depth_vals(plane, line):
    """
    depth_vals(plane, line)

    Returns depth values corresponding to a specified plane.

    Required args:
        - plane (str): plane (e.g., "dend", "soma", "any")
        - line (str) : line (e.g., "L23", "L5", "any")
    Returns:
        - depths (str or list): depths corresponding to plane and line or "any"
    """

    if plane in ["any", "all"] and line in ["any", "all"]:
        return "any"
    
    depth_dict = {"L23_dend": [50, 75],
                  "L23_soma": [175],
                  "L5_dend" : [20],
                  "L5_soma" : [375]
                 }

    all_planes = ["dend", "soma"]
    if plane in ["any", "all"]:
        planes = all_planes
    else:
        planes = plane
        if not isinstance(planes, list):
            planes = [plane]

    
    all_lines = ["L23", "L5"]
    if line in ["any", "all"]:
        lines = all_lines
    else:
        lines = line
        if not isinstance(lines, list):
            lines = [line]
    
    depths = []
    for plane in planes:
        if plane not in all_planes:
            allowed_planes = all_planes + ["any", "all"]
            gen_util.accepted_values_error("plane", plane, allowed_planes)
        for line in lines:
            if line not in all_lines:
                allowed_lines = all_lines + ["any", "all"]
                gen_util.accepted_values_error("line", line, allowed_lines)
            depths.extend(depth_dict[f"{line}_{plane}"])

    return depths


#############################################
def init_analyspar(fluor="dff", rem_bad=True, stats="mean", error="sem", 
                   scale=False, dend="extr", tracked=False):
    """
    init_analyspar()

    Returns a AnalysPar namedtuple with the inputs arguments as named 
    attributes.

    Optional args:
        - fluor (str)   : whether "raw" or processed fluorescence traces 
                          "dff" are used  
                          default: "dff"
        - rem_bad (str) : if True, NaN/Inf values are interpolated bad in
                          for the analyses.
                          default: True
        - stats (str)   : statistic parameter ("mean" or "median")
                          default: "mean"
        - error (str)   : error statistic parameter, ("std" or "sem")
                          default: "sem"
        - scale (bool)  : if True, data is scaled
                          default: False
        - dend (str)    : dendrites to use ("allen" or "extr")
                          default: "extr"
        - tracked (bool): whether to use only tracked ROIs
                          default: False

    Returns:
        - analyspar (AnalysPar namedtuple): AnalysPar with input arguments as 
                                            attributes
    """

    analys_pars = [fluor, rem_bad, stats, error, scale, dend, tracked]
    analys_keys = \
        ["fluor", "rem_bad", "stats", "error", "scale", "dend", "tracked"]
    AnalysPar   = namedtuple("AnalysPar", analys_keys)
    analyspar   = AnalysPar(*analys_pars)
    return analyspar


#############################################
def init_sesspar(sess_n=1, plane="soma", line="any", min_rois=1, pass_fail="P", 
                 incl="yes", runtype="prod", mouse_n="any"):
    """
    init_sesspar()

    Returns a SessPar namedtuple with the inputs arguments as named 
    attributes.

    Optional args:
        - sess_n (int)              : session number aimed for
                                      default: 1
        - plane (str)               : plane ("soma", "dend", "L23_soma",  
                                      "L5_soma", "L23_dend", "L5_dend", 
                                      "L23_all", "L5_all")
                                      default: "soma"
        - line (str)                : mouse line
                                      default: "any"
        - min_rois (int)            : min number of ROIs
                                      default: 1
        - pass_fail (str or list)   : pass/fail values of interest ("P", "F")
                                      default: "P"
        - incl (str or list)        : incl values of interest ("yes", "no", 
                                      "all")
                                      default: "yes"
        - runtype (str)             : runtype value ("pilot", "prod")
                                      default: "prod"
        - mouse_n (str, int or list): mouse number
                                      default: "any"
    
    Returns:
        - sesspar (SessPar namedtuple): SessPar with input arguments as 
                                        attributes
    """

    sess_pars = [sess_n, plane, line, min_rois, pass_fail, incl, 
        runtype, mouse_n]
    sess_keys = ["sess_n", "plane", "line", "min_rois", "pass_fail", 
        "incl", "runtype", "mouse_n"]
    SessPar   = namedtuple("SessPar", sess_keys)
    sesspar   = SessPar(*sess_pars)
    return sesspar


#############################################
def init_stimpar(stimtype="both", visflow_dir=["right", "left"], 
                 visflow_size=128, gabfr=0, gabk=16, 
                 gab_ori=[0, 45, 90, 135, 180, 225], pre=0, post=1.5):
    """
    init_stimpar()

    Returns a StimPar namedtuple with the inputs arguments as named 
    attributes.

    Optional args:
        - stimtype (str)            : stimulus to analyse ("visflow", "gabors" 
                                      of "both")
                                      default: "both"
        - visflow_dir (str or list) : visual flow direction values to include
                                      ("right", "left", ["right", "left"])
                                      default: ["right", "left"]
        - visflow_size (int or list): visual flow square size values to include
                                      (128, 256 or [128, 256])
                                      default: 128
        - gabfr (int)               : gabor frame at which segments start 
                                      (0, 1, 2, 3) (or to include, for GLM)
                                      default: 0
        - gabk (int or list)        : gabor kappa values to include 
                                      (4, 16 or [4, 16])
                                      default: 16
        - gab_ori (int or list)     : gabor orientation values to include
                                      default: [0, 45, 90, 135, 180, 225]
        - pre (num)                 : range of frames to include before each
                                      reference frame (in s)
                                      default: 0
        - post (num)                : range of frames to include after each 
                                      reference frame (in s)
                                      default: 1.5
    
    Returns:
        - stimpar (StimPar namedtuple): StimPar with input arguments as 
                                        attributes
    """

    stim_keys = [
        "stimtype", "visflow_dir", "visflow_size", "gabfr", "gabk", 
        "gab_ori", "pre", "post"
        ]
    stim_pars = [
        stimtype, visflow_dir, visflow_size, gabfr, gabk, gab_ori, pre, post
        ]
    StimPar   = namedtuple("StimPar", stim_keys)
    stimpar   = StimPar(*stim_pars)
    return stimpar


#############################################
def init_basepar(baseline=0):
    """
    init_basepar()

    Returns a baseline namedtuple with the inputs arguments as named attributes.

    Optional args:
        - baseline (float): baseline time
                            default: 0
    Returns:
        - basepar (BasePar namedtuple): BasePar with input arguments as 
                                        attributes
    """

    base_pars = [baseline]
    base_keys = ["baseline"]
    BasePar   = namedtuple("BasePar", base_keys)
    basepar   = BasePar(*base_pars)
    return basepar


######################################
def load_info_from_mouse_df(sessid, mouse_df="mouse_df.csv"):
    """
    load_info_from_mouse_df(sessid)

    Returns dictionary containing information from the mouse dataframe.

    Required args:
        - sessid (int): session ID

    Optional args:
        - mouse_df (Path): path name of dataframe containing information on each 
                           session. Dataframe should have the following columns:
                               sessid, mouse_n, depth, plane, line, sess_n, 
                               pass_fail, all_files, any_files, notes
                           default: "mouse_df.csv"

    Returns:
        - df_dict (dict): dictionary with following keys:
            - age_weeks (float): age (in weeks)
            - all_files (bool) : if True, all files have been acquired for
                                 the session
            - any_files (bool) : if True, some files have been acquired for
                                 the session
            - timestamp (str)  : session timestamp (in UTC)
            - date (str)       : session date (i.e., yyyymmdd)
            - depth (int)      : recording depth 
            - DOB (int)        : date of birth (i.e., yyyymmdd)
            - plane (str)      : recording plane ("soma" or "dend")
            - line (str)       : mouse line (e.g., "L5-Rbp4")
            - mouse_n (int)    : mouse number (e.g., 1)
            - mouseid (int)    : mouse ID (6 digits)
            - notes (str)      : notes from the dataframe on the session
            - pass_fail (str)  : whether session passed "P" or failed "F" 
                                 quality control
            - runtype (str)    : "prod" (production) or "pilot" data
            - sess_n (int)     : overall session number (e.g., 1)
            - sex (str)        : sex (e.g., "F" or "M")
            - stim_seed (int)  : random seed used to generated stimulus 
    """

    if isinstance(mouse_df, (str, Path)):
        if Path(mouse_df).is_file():
            mouse_df = pd.read_csv(mouse_df)
        else:
            raise OSError(f"{mouse_df} does not exist.")

    # retrieve the dataframe line for the session
    df_line = mouse_df.loc[mouse_df["sessid"] == sessid]
    if len(df_line) == 0:
        raise ValueError(
            f"Session ID {sessid} not found in the mouse dataframe."
            )
    elif len(df_line) > 1:
        raise ValueError(
            f"Found multiple matches for session ID {sessid} in the mouse "
            "dataframe."
            )

    df_dict = {
        "mouse_n"      : int(df_line["mouse_n"].tolist()[0]),
        "timestamp"    : df_line["full_timestamp"].tolist()[0],
        "sex"          : str(df_line["sex"].tolist()[0]),
        "DOB"          : int(df_line["DOB"].tolist()[0]),
        "date"         : int(df_line["date"].tolist()[0]),
        "age_weeks"    : float(df_line["age_weeks"].tolist()[0]),
        "depth"        : df_line["depth"].tolist()[0],
        "plane"        : df_line["plane"].tolist()[0],
        "line"         : df_line["line"].tolist()[0],
        "mouseid"      : int(df_line["mouseid"].tolist()[0]),
        "runtype"      : df_line["runtype"].tolist()[0],
        "sess_n"       : int(df_line["sess_n"].tolist()[0]),
        "stim_seed"    : int(df_line["stim_seed"].tolist()[0]),
        "pass_fail"    : df_line["pass_fail"].tolist()[0],
        "all_files"    : bool(int(df_line["all_files"].tolist()[0])),
        "any_files"    : bool(int(df_line["any_files"].tolist()[0])),
        "notes"        : df_line["notes"].tolist()[0],
    }

    return df_dict
    
    
#############################################
def get_sess_vals(mouse_df, mouse_n="any", sess_n="any", runtype="any", 
                  plane="any", line="any", pass_fail="P", incl="all", 
                  min_rois=1):
    """
    get_sess_vals(mouse_df, returnlab)

    Returns list of values under the specified label that fit the specified
    criteria.

    Required args:
        - mouse_df (Path)        : path name of dataframe containing 
                                   information on each session

    Optional args:
        - mouse_n (int, str or list)  : mouse number(s) of interest
                                        default: "any"
        - sess_n (int, str or list)   : session number(s) of interest
                                        default: "any"
        - runtype (str or list)       : runtype value(s) of interest
                                        ("pilot", "prod")
                                        default: "any"
        - plane (str or list)         : plane value(s) of interest
                                        ("soma", "dend", "any")
                                        default: "any"
        - line (str or list)          : line value(s) of interest
                                        ("L5", "L23", "any")
                                        default: "any"
        - pass_fail (str or list)     : pass/fail values of interest 
                                        ("P", "F", "any")
                                        default: "P"
        - incl (str)                  : which sessions to include ("yes", "no", 
                                        "any")
                                        default: "yes"
        - min_rois (int)              : min number of ROIs
                                        default: 1
     
    Returns:
        - sessids (list): session IDs, based on criteria
    """
    
    if runtype != "prod":
        raise ValueError("runtype can only be set to 'prod'.")

    if isinstance(mouse_df, (str, Path)):
        if Path(mouse_df).is_file():
            mouse_df = pd.read_csv(mouse_df)
        else:
            raise OSError(f"{mouse_df} does not exist.")

    # get depth values corresponding to the plane
    depth = depth_vals(plane, line)

    params      = [mouse_n, sess_n, runtype, depth, pass_fail, incl]
    param_names = ["mouse_n", "sess_n", "runtype", "depth", "pass_fail", "incl"]
    
    # for each label, collect values in a list
    for i in range(len(params)):
        params[i] = gen_util.get_df_label_vals(
            mouse_df, param_names[i], params[i])   
        col_dtype = mouse_df[param_names[i]].dtype        
        if col_dtype == bool:
            params[i] = [bool(param) for param in params[i]]
        if col_dtype == float:
            params[i] = [float(param) for param in params[i]]
        if col_dtype == int:
            params[i] = [int(param) for param in params[i]]
    [mouse_n, sess_n, runtype, depth, pass_fail, incl] = params

    sessids = mouse_df.loc[
        (mouse_df["mouse_n"].isin(mouse_n)) &
        (mouse_df["sess_n"].isin(sess_n)) &
        (mouse_df["runtype"].isin(runtype)) &
        (mouse_df["depth"].isin(depth)) &
        (mouse_df["pass_fail"].isin(pass_fail)) &
        (mouse_df["incl"].isin(incl)) &
        (mouse_df["nrois"].astype(int) >= min_rois)]["sessid"].tolist()
   
    return sessids
    
    
#############################################
def init_sessions(sessids, datadir, mouse_df, runtype="prod", full_table=True, 
                  roi=True, run=False, pupil=False, temp_log=None):
    """
    init_sessions(sessids, datadir)

    Creates list of Session objects for each session ID passed.

    Required args:
        - sessids (int or list): ID or list of IDs of sessions
        - datadir (Path)       : directory where sessions are stored
        - mouse_df (Path)      : path name of dataframe containing information 
                                 on each session

    Optional args:
        - runtype (str)    : the type of run, either "pilot" or "prod"
                             default: "prod"
        - full_table (bool): if True, the full stimulus dataframe is loaded 
                             (with all the visual flow square positions and 
                             individual Gabor patch orientations).
                             default: True
        - roi (bool)       : if True, ROI data is loaded into sessions
                             default: True
        - run (bool)       : if True, running data is loaded into sessions
                             default: False
        - pupil (bool)     : if True, pupil data is loaded into session and 
                             only sessions with pupil data are included
                             default: False
        - temp_log (bool)  : temporary log level to set logger to. If None, 
                             logger is left at current level.
                             default: None

    Returns:
        - sessions (list): list of Session objects
    """

    from analysis import session
    
    with logger_util.TempChangeLogLevel(level=temp_log):
        sessions = []
        if not isinstance(sessids, list):
            sessids = [sessids]
        for sessid in sessids:
            logger.info(
                f"Creating session {sessid}...", extra={"spacing": "\n"}
                )
            # creates a session object to work with
            sess = session.Session(
                sessid, datadir, runtype=runtype, mouse_df=mouse_df) 
            # extracts necessary info for analysis
            sess.extract_info(full_table=full_table, roi=roi, run=run)
            if pupil:
                if sess.pup_data_available:
                    sess.load_pup_data()
                else:
                    logger.info(
                        f"Omitting session {sessid} as no pupil data was found."
                    )
                    continue
                    
            logger.info(f"Finished creating session {sessid}.")
            sessions.append(sess)

    return sessions


#############################################
def check_session(sess, roi=True, run=False, pupil=False):
    """
    check_session(session, analyspar)

    Checks whether required data is loaded, and returns session with required data loaded.

    Required args:
        - sess (Session): session object

    Optional args:
        - roi (bool)  : whether ROI information should be loaded
                        default: True
        - run (bool)  : whether running data should be loaded
                        default: False
        - pupil (bool): whether pupil data should be loaded
                        default: False
    
    Returns:
        - session (Session): session object, with required data loaded
    """

    sess = copy.deepcopy(sess)

    roi_loaded, run_loaded, pupil_loaded = sess.data_loaded()
    
    if roi and not(roi_loaded):
        sess.load_roi_info()
    
    if run and not(run_loaded):
        sess.load_run_data()
    
    if pupil and not(pupil_loaded):
        sess.load_pup_data()    
            
    return sess


############################################
def get_sess_info(sessions, add_none=False, incl_roi=True, return_df=False):
    """
    get_sess_info(sessions)

    Puts information from all sessions into a dictionary. Optionally allows 
    None sessions.

    Required args:
        - sessions (list): ordered list of Session objects
    
    Optional args:
        - add_none (bool): if True, None sessions are allowed and all values 
                           are filled with None
                           default: False
        - incl_roi (bool): if True, ROI information is included
                           default: True

    Returns:
        - sess_info (dict or df): dictionary or dataframe containing 
                                  information from each session, as lists or 
                                  in dataframe rows, under the following keys 
                                  or columns
            "mouse_ns", "mouseids", "sess_ns", "sessids", "lines", "planes"
            if datatype == "roi":
                "nrois", "twop_fps"
    """

    if return_df and add_none:
        raise ValueError("'add_none' cannot be True if 'return_df' is True.")

    if add_none and set(sessions) == {None}:
        logger.info("All None value sessions.")

    sess_info = dict()
    keys = ["mouse_ns", "mouseids", "sess_ns", "sessids", "lines", "planes"]
    if incl_roi:
        keys.extend(["nrois", "twop_fps"])
    
    for key in keys:
        sess_info[key] = []

    if not isinstance(sessions, list):
        sessions = [sessions]

    for _, sess in enumerate(sessions):
        if sess is None:
            if add_none:
                 for key in keys:
                     sess_info[key].append(None)
            else:
                raise RuntimeError("None sessions not allowed.")
        else:
            sess_info["mouse_ns"].append(sess.mouse_n)
            sess_info["mouseids"].append(sess.mouseid)
            sess_info["sess_ns"].append(sess.sess_n)
            sess_info["sessids"].append(sess.sessid)
            sess_info["lines"].append(sess.line)
            sess_info["planes"].append(sess.plane)
            if not incl_roi:
                continue
            
            nrois = sess.get_nrois()
            sess_info["nrois"].append(nrois)
            sess_info["twop_fps"].append(sess.twop_fps)             

    if return_df:
        sess_info = pd.DataFrame.from_dict(sess_info)

    return sess_info


######################################
def get_sessid_from_mouse_df(mouse_n=1, sess_n=1, runtype="prod", 
                             mouse_df="mouse_df.csv"):
    """
    get_sessid_from_mouse_df(sessid)

    Returns session ID, based on the mouse number, session number, and runtype,
    based on the mouse dataframe.

    Optional args:
        - mouse_n (int)  : mouse number
                           default: 1
        - sess_n (int)   : session number
                           default: 1
        - runtype (str)  : type of data
                           default: 1
        - mouse_df (Path): path name of dataframe containing information on each 
                           session. Dataframe should have the following columns:
                               mouse_n, sess_n, runtype
                           default: "mouse_df.csv"

    Returns:
        - sessid (int): session ID
    """

    if isinstance(mouse_df, (str, Path)):
        if Path(mouse_df).is_file():
            mouse_df = pd.read_csv(mouse_df)
        else:
            raise OSError(f"{mouse_df} does not exist.")

    # retrieve the dataframe line for the mouse number, session and runtype
    df_line = mouse_df.loc[
        (mouse_df["mouse_n"] == int(mouse_n)) &
        (mouse_df["sess_n"] == int(sess_n)) &
        (mouse_df["runtype"] == runtype)
        ]

    if len(df_line) == 0:
        raise ValueError(
            f"No matches for mouse {mouse_n}, session {sess_n} ({runtype} run) "
            "in the mouse dataframe."
            )
    elif len(df_line) > 1:
        raise ValueError(
            f"Multiple matches for mouse {mouse_n}, session {sess_n} ({runtype} "
            "run) in the mouse dataframe."
            )

    sessid = int(df_line["sessid"].tolist()[0])

    return sessid


#############################################
def scale_data_df(data_df, datatype, interpolated="no", other_vals=[]):
    """
    scale_data_df(data_df, datatype)

    Returns data frame with specific column scaled using the factors
    found in the dataframe.

    Required args:
        - data_df (pd DataFrame): dataframe containing data for each frame, 
                                  organized by:
            hierarchical columns:
                - datatype    : type of data
                - interpolated: whether data is interpolated ("yes", "no")
                - scaled      : whether data is scaled ("yes", "no")
            hierarchical rows:
                - "info"      : type of information contained 
                                ("frames": values for each frame, 
                                "factors": scaling values for 
                                each factor)
                - "specific"  : specific type of information contained 
                                (frame number, scaling factor name)
        - datatype (str)        : datatype to be scaled

    Optional args:
        - interpolated (str): whether to scale interpolated data ("yes") 
                              or not ("no")
                              default: "no"
        - other_vals (list) : other values in the hierarchical dataframe to
                              copy to new column
                              default: []
    
    Returns:
        - data_df (pd DataFrame): same dataframe, but with scaled 
                                  data added
    """

    datatypes = data_df.columns.unique(level="datatype").tolist()

    if datatype not in datatypes:
        gen_util.accepted_values_error("datatype", datatype, datatypes)

    if not isinstance(other_vals, list):
        other_vals = [other_vals]

    if "yes" in data_df[datatype].columns.get_level_values(
        level="scaled"):
        logger.info("Data already scaled.")

    factor_names = data_df.loc["factors"].index.unique(
        level="specific").tolist()
    sub_names =  list(filter(lambda x: "sub" in x, factor_names))
    if len(sub_names) != 1:
        raise RuntimeError("Only one factor should contain 'sub'.")
    div_names =  list(filter(lambda x: "div" in x, factor_names))
    if len(div_names) != 1:
        raise RuntimeError("Only one row should contain 'div'.")

    sub = data_df.loc[("factors", sub_names[0])].values[0]
    div = data_df.loc[("factors", div_names[0])].values[0]

    data_df = data_df.copy(deep=True)

    data_df.loc[("frames",), (datatype, interpolated, "yes", *other_vals)] = \
        (data_df.loc[("frames",), 
        (datatype, interpolated, "no", *other_vals)].values - sub)/div

    return data_df


#############################################
def format_stim_criteria(stim_df, stimtype="gabors", unexp="any", 
                         stim_seg="any", gabfr="any", gabk=None, gab_ori=None, 
                         visflow_size=None, visflow_dir=None, start2pfr="any", 
                         end2pfr="any", num2pfr="any"):
    """
    format_stim_criteria()

    Returns a list of stimulus parameters formatted correctly to use
    as criteria when searching through the stim dataframe. 

    Will strip any criteria not related to the relevant stimulus type.

    Required args:
        - stim_df (pd DataFrame)       : stimulus dataframe

    Optional args:
        - stimtype (str)               : stimulus type
                                            default: "gabors"
        - unexp (str, int or list)     : unexpected value(s) of interest (0, 1)
                                            default: "any"
        - stim_seg (str, int or list)  : stimulus segment value(s) of interest
                                            default: "any"
        - gabfr (str, int or list)     : gaborframe value(s) of interest 
                                            (0, 1, 2, 3, 4 or letters)
                                            default: "any"
        - gabk (int or list)           : if not None, will overwrite 
                                            stimPar2 (4, 16, or "any")
                                            default: None
        - gab_ori (int or list)        : if not None, will overwrite 
                                            stimPar1 (0, 45, 90, 135, 180, 225, 
                                            or "any")
                                            default: None
        - visflow_size (int or list)   : if not None, will overwrite 
                                            stimPar1 (128, 256, or "any")
                                            default: None
        - visflow_dir (str or list)    : if not None, will overwrite 
                                            stimPar2 ("right", "left", "temp", 
                                             "nasal", or "any")
                                            default: None
        - start2pfr (str or list)      : 2p start frames range of interest
                                            [min, max (excl)] 
                                            default: "any"
        - end2pfr (str or list)        : 2p excluded end frames range of 
                                            interest [min, max (excl)]
                                            default: "any"
        - num2pfr (str or list)        : 2p num frames range of interest
                                            [min, max (excl)]
                                            default: "any"
    
    Returns:
        - unexp (list)       : unexpected value(s) of interest (0, 1)
        - stim_seg (list)    : stim_seg value(s) of interest
        - gabfr (list)       : gaborframe value(s) of interest 
        - gabk (list)        : gabor kappa value(s) of interest 
        - gab_ori (list)     : gabor mean orientation value(s) of interest 
        - visflow_size (list): visual flow square size value(s) of interest 
        - visflow_dir (list) : visual flow direction value(s) of interest 
        - start2pfr_min (int): minimum of 2p start2pfr range of interest 
        - start2pfr_max (int): maximum of 2p start2pfr range of interest 
                                (excl)
        - end2pfr_min (int)  : minimum of 2p end2pfr range of interest
        - end2pfr_max (int)  : maximum of 2p end2pfr range of interest 
                                (excl)
        - num2pfr_min (int)  : minimum of num2pfr range of interest
        - num2pfr_max (int)  : maximum of num2pfr range of interest 
                                (excl)
    """

    # remove visual flow criteria for gabors and vv
    if stimtype == "gabors":
        visflow_size = None
        visflow_dir = None
    elif stimtype == "visflow":
        gabfr = None
        gabk = None
        gab_ori = None
    else:
        gen_util.accepted_values_error(
            "stimtype", stimtype, ["gabors", "visflow"])

    # converts values to lists or gets all possible values, if "any"
    unexp    = gen_util.get_df_label_vals(stim_df, "unexpected", unexp)    
    gabk     = gen_util.get_df_label_vals(stim_df, "gabor_kappa", gabk)
    gabfr    = gen_util.get_df_label_vals(stim_df, "gabor_frame", gabfr)
    gab_ori  = gen_util.get_df_label_vals(
        stim_df, "gabor_mean_orientation", gab_ori
        )
    visflow_dir = gen_util.get_df_label_vals(
        stim_df, "main_flow_direction", visflow_dir
        )
    visflow_size = gen_util.get_df_label_vals(
        stim_df, "square_size", visflow_size
        )

    if stim_seg in ["any", "all"]:
        stim_seg = stim_df.index
    elif not isinstance(stim_seg, list):
        stim_seg = [stim_seg]

    gabfr = copy.deepcopy(gabfr)
    for fr in gabfr:
        if str(fr) == "0":
            gabfr.append("A")
        elif str(fr) == "1":
            gabfr.append("B")
        elif str(fr) == "2":
            gabfr.append("C")
        elif str(fr) == "3":
            gabfr.extend(["D", "U"])
        elif str(fr) == "4":
            gabfr.append("G")

    for i in range(len(visflow_dir)):
        if visflow_dir[i] in ["right", "left", "temp", "nasal"]:
            visflow_dir[i] = \
                get_visflow_screen_mouse_direc(visflow_dir[i])

    if start2pfr in ["any", None]:
        start2pfr_min = int(stim_df["start_frame_twop"].min())
        start2pfr_max = int(stim_df["start_frame_twop"].max()+1)

    elif len(start2pfr) == 2:
        start2pfr_min, start2pfr_max = start2pfr
    else:
        raise ValueError("'start2pfr' must be of length 2 if passed.")

    if end2pfr in ["any", None]:
        end2pfr_min = int(stim_df["stop_frame_twop"].min())
        end2pfr_max = int(stim_df["stop_frame_twop"].max() + 1)
    elif len(start2pfr) == 2:
        end2pfr_min, end2pfr_max = end2pfr
    else:
        raise ValueError("'end2pfr' must be of length 2 if passed.")

    if num2pfr in ["any", None]:
        num2pfr_min = int(stim_df["num_frames_twop"].min())
        num2pfr_max = int(stim_df["num_frames_twop"].max() + 1)
    elif len(start2pfr) == 2:
        num2pfr_min, num2pfr_max = num2pfr
    else:
        raise ValueError("'num2pfr' must be of length 2 if passed.")

    return [unexp, stim_seg, gabfr, gabk, gab_ori, visflow_size, visflow_dir, 
        start2pfr_min, start2pfr_max, end2pfr_min, end2pfr_max, num2pfr_min, 
        num2pfr_max] 

