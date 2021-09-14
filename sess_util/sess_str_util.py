"""
sess_str_util.py

This module contains basic math functions for getting strings to print or save
files for Allen Institute OpenScope experiments for the Credit Assignment 
Project.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

from sess_util import sess_gen_util
from util import gen_util, logger_util


#############################################
def base_par_str(baseline=None, str_type="file"):
    """
    base_par_str()

    Returns string from baseline parameter to print or for a filename.

    Optional args:
        - baseline (num)  : baseline value, in seconds
                            default: None
        - str_type (str)  : "print" for a printable string and "file" for a
                            string usable in a filename.
                            default: "file"
    
    Returns:
        - base_str (str): baseline parameter string
    """

    if baseline is not None:
        if str_type == "print":
            baseline = gen_util.num_to_str(baseline, n_dec=2, dec_sep=".")
            base_str = f" ({baseline}s baseline)"    
        elif str_type == "file":
            baseline = gen_util.num_to_str(baseline, n_dec=2, dec_sep="-")
            base_str = f"_b{baseline}"
        else:
            gen_util.accepted_values_error(
                "str_type", str_type, ["print", "file"])        
    else:
        base_str = ""

    return base_str
    
    
#############################################
def shuff_par_str(shuffle=True, str_type="file"):
    """
    shuff_par_str()

    Returns string from shuffle parameter to print or for a filename.

    Optional args:
        - shuffle (bool): default: True
        - str_type (str): "print" for a printable string and "file" for a
                          string usable in a filename, "label" for a label.
                          default: "file"
    
    Returns:
        - shuff_str (str): shuffle parameter string
    """

    if shuffle:
        if str_type == "print":
            shuff_str = ", shuffled"
        elif str_type == "file":
            shuff_str = "_shuffled"
        elif str_type == "labels":
            shuff_str = " (shuffled labels)"
        else:
            gen_util.accepted_values_error(
                "str_type", str_type, ["labels", "print", "file"])
    else:
        shuff_str = ""

    return shuff_str
    
    
#############################################
def ctrl_par_str(ctrl=False, str_type="file"):
    """
    ctrl_par_str()

    Returns string from control parameter to print or for a filename.

    Optional args:
        - ctrl (bool)   : default: False
        - str_type (str): "print" for a printable string and "file" for a
                          string usable in a filename, "label" for a label.
                          default: "file"
    
    Returns:
        - ctrl_str (str): shuffle parameter string
    """

    if ctrl:
        if str_type == "print":
            ctrl_str = " (control)"
        elif str_type == "file":
            ctrl_str = "_ctrl"
        else:
            gen_util.accepted_values_error(
                "str_type", str_type, ["print", "file"])
    else:
        ctrl_str = ""

    return ctrl_str
    
    
#############################################
def scale_par_str(scale=True, str_type="file"):
    """
    scale_par_str()

    Returns a string from scaling parameter to print or for a filename.

    Optional args:
        - scale (str or bool): if scaling is used or type of scaling used 
                               (e.g., "roi", "all", "none", True, False)
                               default: None
        - str_type (str)     : "print" for a printable string and "file" for a
                               string usable in a filename.
                               default: "file"
    
    Returns:
        - scale_str (str): scale parameter string
    """

    if scale not in ["None", "none"] and scale:
        if str_type == "print":
            scale_str = " (scaled)"
        elif str_type == "file":
            scale_str = "_scaled"
        else:
            gen_util.accepted_values_error(
                "str_type", str_type, ["print", "file"])
    else:
        scale_str = ""

    return scale_str
    

#############################################
def fluor_par_str(fluor="dff", str_type="file"):
    """
    fluor_par_str()

    Returns a string from the fluorescence parameter to print or for a 
    filename.

    Optional args:
        - fluor (str)   : whether "raw" or processed fluorescence traces "dff"  
                          are used  
                          default: "dff"
        - str_type (str): "print" for a printable string and "file" for a
                          string usable in a filename.
                          default: "file"
    
    Returns:
        - fluor_str (str): fluorescence parameter string
    """

    if fluor == "raw":
        if str_type == "print":
            fluor_str = "raw fluorescence intensity"
        elif str_type == "file":
            fluor_str = "raw"
        else:
            gen_util.accepted_values_error(
                "str_type", str_type, ["print", "file"])
    elif fluor == "dff":
        if str_type == "print":
            delta = u"\u0394"
            fluor_str = u"{}F/F".format(delta)
        elif str_type == "file":
            fluor_str = "dff"
        else:
            gen_util.accepted_values_error(
                "str_type", str_type, ["print", "file"])
    else:
        gen_util.accepted_values_error("fluor", fluor, ["raw", "dff"])

    return fluor_str
    

#############################################
def stat_par_str(stats="mean", error="sem", str_type="file"):
    """
    stat_par_str()

    Returns a string from statistical analysis parameters to print or for a 
    filename.

    Optional args:
        - stats (str)   : "mean" or "median"
                          default: "mean"
        - error (str)   : "std" (for std or quartiles) or "sem" (for SEM or MAD)
                          or "None" is no error
                          default: "sem"
        - str_type (str): use of output str, i.e., for a filename ("file") or
                          to print the info to console or for title ("print")
                          default: "file"
    
    Returns:
        - stat_str (str): statistics combo string
    """
    if error in ["None", "none"]:
        stat_str = stats
    else:
        if str_type == "print":
            sep = u" \u00B1 " # +- symbol
        elif str_type == "file":
            sep = "_"
        else:
            gen_util.accepted_values_error(
                "str_type", str_type, ["print", "file"])

        if stats == "mean":
            if error == "sem":
                error = "SEM"
            stat_str = u"{}{}{}".format(stats, sep, error)
        elif stats == "median":
            if error == "std":
                stat_str = u"{}{}qu".format(stats, sep)
            elif error == "sem":
                stat_str = u"{}{}MAD".format(stats, sep)
            else:
                gen_util.accepted_values_error(
                    "error", error, ["std", "sem", "None", "none"])
        else:
            gen_util.accepted_values_error("stats", stats, ["mean", "median"])
    return stat_str


#############################################
def op_par_str(plot_vals="both", op="diff", str_type="file"):
    """
    op_par_str()

    Returns a string from plot values and operation parameters to print or  
    for a filename.

    Optional args:
        - plot_vals (str): "both", "surp" or "reg"
        - op (str)       : "diff", "ratio"
                           default: "diff"
        - str_type (str) : use of output str, i.e., for a filename ("file") or
                           to print the info to console ("print")
                           default: "file"
    
    Returns:

        - op_str (str): operation type string
    """
    
    if op not in ["diff", "ratio"]:
        gen_util.accepted_values_error("op", op, ["diff", "ratio"])
    
    if plot_vals not in ["both", "reg", "surp"]:
        gen_util.accepted_values_error(
            "plot_vals", plot_vals, ["both", "reg", "surp"])
    
    if plot_vals == "both":
        if str_type == "print":
            op_str = "for surp v reg"
        elif str_type == "file":
            op_str = op
        else:
            gen_util.accepted_values_error(
                "str_type", str_type, ["print", "file"])
    else:
        if str_type == "print":
            op_str = f"for {plot_vals}"
        elif str_type == "file":
            op_str = plot_vals
        else:
            gen_util.accepted_values_error(
                "str_type", str_type, ["print", "file"])
    
    return op_str


#############################################
def lat_par_str(method="ttest", p_val_thr=0.005, rel_std=0.5, str_type="file"):
    """
    lat_par_str()

    Returns a string for the latency calculation info.

    Optional args:
        - method (str)     : latency calculating method ("ratio" or "ttest")
                             default: "ttest"
        - p_val_thr (float): p-value threshold for t-test method
                             default: 0.005
        - rel_std (flot)   : relative standard deviation threshold for ratio 
                             method
                             default: 0.5
        - str_type (str)   : use of output str, i.e., for a filename 
                             ("file") or to print the info to console 
                             ("print")
                             default: "file"
    Return:
        - lat_str (str): string containing latency info
    """

    if method == "ttest":
        ext_str = "pval"
        val = p_val_thr
    elif method == "ratio":
        ext_str = "std"
        val = rel_std
    else:
        gen_util.accepted_values_error("method", method, ["ttest", "ratio"])

    if str_type == "print":
        val = gen_util.num_to_str(val, n_dec=5, dec_sep=".")
        lat_str = f"{val} {ext_str}"
    elif str_type == "file":
        val = gen_util.num_to_str(val, n_dec=5, dec_sep="-")
        lat_str = f"{val}{ext_str}"
    else:
        gen_util.accepted_values_error("str_type", str_type, ["print", "file"])

    return lat_str


#############################################
def prepost_par_str(pre, post, str_type="file"):
    """
    prepost_par_str(pre, post)

    Returns a string for the pre and post values.

    Required args:
        - pre (num) : pre value (in seconds)
        - post (num): post value (in seconds) 

    Optional args:
        - str_type (str): use of output str, i.e., for a filename 
                          ("file") or to print the info to console 
                          ("print")
                          default: "file"
    Return:
        - prepost_str (str): string containing pre-post info
    """

    vals = [pre, post]

    # convert to int if equivalent
    for i in range(len(vals)):
        if int(vals[i]) == float(vals[i]):
            vals[i] = int(vals[i])
    
    if str_type == "file":
        # replace . by -
        prepost_str = "{}pre-{}post".format(*vals).replace(".", "")
    elif str_type == "print":
        prepost_str = "{}-{}s".format(*vals)
    else:
        gen_util.accepted_values_error("str_type", str_type, ["print", "file"])

    return prepost_str


#############################################
def dend_par_str(dend="extr", plane="dend", datatype="roi", str_type="file"):
    """
    dend_par_str()

    Returns a string from dendrite parameter to print or for a filename.

    Optional args:
        - dend (str)     : type of dendrite ("aibs" or "extr")
                           default: "extr"
        - plane (str)    : plane ("dend" or "soma")
                           default: "dend"
        - datatype (str) : type of data ("roi", "run")
                           default: "roi"
        - str_type (str) : use of output str, i.e., for a filename ("file") or
                           to print the info to console ("print")
                           default: "file"
    
    Returns:
        - dend_str (str): dendrite type string
    """

    planes = ["dend", "soma", "any"]
    if plane not in planes:
        gen_util.accepted_values_error("plane", plane, planes)
    
    datatypes = ["roi", "run"]
    if datatype not in datatypes:
        gen_util.accepted_values_error("datatype", datatype, datatypes)
    
    dend_str = ""
    if plane in ["dend", "any"] and datatype == "roi" and dend == "aibs":
        if str_type == "file":
            dend_str = "_aibs"
        elif str_type == "print":
            dend_str = " (aibs)"
        else:
            gen_util.accepted_values_error(
                "str_type", str_type, ["print", "file"])
    
    return dend_str


#############################################
def gabfr_nbrs(gabfr):
    """
    gabfr_nbrs(gabfr)

    Returns the numbers corresponding to the Gabor frame letters (A, B, C, D/U).

    Required args:
        - gabfr (str or list): gabor frame letter(s)

    Returns:
        - gab_nbr (int or list): gabor frame number(s)
    """

    if not isinstance(gabfr, list):
        gabfr_list = False
        gabfr = [gabfr]
    else:
        gabfr_list = True

    all_gabfr  = ["A", "B", "C", "D", "U", "D/U"]
    all_gabnbr = [0, 1, 2, 3, 3, 3]


    if sum([g not in all_gabfr for g in gabfr]):
        raise ValueError("Gabor frames letters include A, B, C, D and U only.")
    
    
    if gabfr_list:
        gab_nbr = [all_gabnbr[all_gabfr.index(g)] for g in gabfr]
    
    else:
        gab_nbr = all_gabnbr[all_gabfr.index(gabfr[0])]
    
    return gab_nbr


#############################################
def gabfr_letters(gabfr, surp="any"):
    """
    gabfr_letters(gabfr)

    Returns the letters corresponding to the Gabor frame numbers (0, 1, 2, 3).

    Required args:
        - gabfr (int or list): gabor frame number(s)

    Optional args:
        - surp (str, int or list): surprise values for all or each gabor frame 
                                   number. If only value, applies to all.
                                   (0, 1 or "any")
                                   default: "any"

    Returns:
        - gab_letts (str or list): gabor frame letter(s)
    """

    if not isinstance(gabfr, list):
        gabfr_list = False
        gabfr = [gabfr]
    else:
        gabfr_list = True

    surp = gen_util.list_if_not(surp)
    if len(surp) == 1:
        surp = surp * len(gabfr)    
    else:
        if len(gabfr) != len(surp):
            raise ValueError("If passing more than one surp value, must "
                "pass as many as gabfr.")

    if min(gabfr) < 0 or max(gabfr) > 3:
        raise ValueError("Gabor frames are only between 0 and 3, inclusively.")

    all_gabfr = ["A", "B", "C", "D/U"]

    gab_letts = []
    for i, gf in enumerate(gabfr):
        if gf == 3 and surp[i] != "any":
            gab_letts.append(all_gabfr[gf][-surp[i]]) # D or U is retained
        else:
            gab_letts.append(all_gabfr[gf])

    if not gabfr_list:
        gab_letts = gab_letts[0]
    
    return gab_letts


#############################################
def gabk_par_str(gabk, str_type="file"):
    """
    gabk_par_str(gabk)

    Returns a string with stim type, as well as kappa parameters
    (e.g., 4, 16), unless only 16 is passed.

    Required args:
        - gabk (int or list): gabor kappa parameter

    Optional args:
        - str_type (str) : use of output str, i.e., for a filename ("file") or
                           to print the info to console ("print")
                           default: "file"

    Returns:
        - pars (str): string containing stim type (gabors) and kappa, 
                      unless only 16 is passed.
    """

    gabk = gen_util.list_if_not(gabk)
    gabk = [int(g) for g in gabk]

    if str_type == "file":
        pars = "gab"
    elif str_type == "print":
        pars = "gabors"
    else:
        gen_util.accepted_values_error("str_type", str_type, ["print", "file"])

    if 4 in gabk:
        if len(gabk) > 1:
            if str_type == "file":
                pars = f"{pars}_both"
            elif str_type == "print":
                pars = f"{pars} (both)"
        else:
            if str_type == "file":
                pars = f"{pars}{gabk[0]}"
            elif str_type == "print":
                pars = f"{pars} ({gabk[0]})"

    return pars


#############################################
def size_par_str(size, str_type="file"):
    """
    size_par_str(size)

    Returns a string with stim type, as well as size parameters
    (e.g., 128, 256), unless only 128 is passed.

    Required args:
        - size (int or list): brick size parameter

    Optional args:
        - str_type (str) : use of output str, i.e., for a filename ("file") or
                           to print the info to console ("print")
                           default: "file"

    Returns:
        - pars (str): string containing stim type (bricks) and size, 
                      unless only 128 is passed.
    """

    size = gen_util.list_if_not(size)
    size = [int(s) for s in size]

    if str_type == "file":
        pars = "bri"
    elif str_type == "print":
        pars = "bricks"
    else:
        gen_util.accepted_values_error("str_type", str_type, ["print", "file"])

    if 256 in size:
        if len(size) > 1:
            if str_type == "file":
                pars = f"{pars}_both_siz"
            elif str_type == "print":
                pars = f"{pars} (both sizes)"
        else:
            if str_type == "file":
                pars = f"{pars}{size[0]}"
            elif str_type == "print":
                pars = f"{pars} ({size[0]})"

    return pars


#############################################
def dir_par_str(direc, str_type="file"):
    """
    dir_par_str(direc)

    Returns a string with stim type, as well as direction parameters
    (e.g., "right", "left"), unless both possible values are passed.

    Required args:
        - direc (str or list): brick direction parameter

    Optional args:
        - str_type (str) : use of output str, i.e., for a filename ("file") or
                           to print the info to console ("print")
                           default: "file"

    Returns:
        - pars (str): string containing stim type (bricks) and direction, 
                      unless both possible values are passed.
    """

    direc = gen_util.list_if_not(direc)
    if str_type == "file":
        pars = "bri"
    elif str_type == "print":
        pars = "bricks"
    else:
        gen_util.accepted_values_error("str_type", str_type, ["print", "file"])
    
    if len(direc) == 1 and direc[0] != "both":
        direc = direc[0]
        direc_detailed = sess_gen_util.get_bri_screen_mouse_direc(direc)
        if str_type == "file":
            direc = direc_detailed[:5].strip(" ") # get left/right
            pars = f"{pars}_{direc}"
        elif str_type == "print":
            direc_detailed = direc_detailed.replace(" (", ", ").replace(")", "")
            pars = f"{pars} ({direc_detailed})"

    return pars


#############################################
def bri_par_str(direc, size, str_type="file"):
    """
    bri_par_str()

    Returns a string with stim type, as well as size (e.g., 128, 256) and 
    direction (e.g., "right", "left") parameters, unless all possible bricks 
    parameters values are passed.

    Required args:
        - direc (str or list) : brick direction parameter values
        - size (int or list): brick size parameter values

    Optional args:
        - str_type (str) : use of output str, i.e., for a filename ("file") or
                           to print the info to console ("print")
                           default: "file"

    Returns:
        - pars (str): string containing stim type (bricks) and parameter values, 
                      unless all parameter values for bricks are passed.
    """
    
    if size is None or direc is None:
        raise ValueError("Must pass value for brick size or direction "
                         "parameter.")

    dirstr = dir_par_str(direc, str_type=str_type)
    sizestr = size_par_str(size, str_type=str_type)
    if str_type == "print":
        if len(dirstr) > 6: # specified direction
            if len(sizestr) > 6: # specified size
                pars = (f"{sizestr.replace(')', '')}, "
                    f"{dirstr.replace('bricks (', '')}")
            else:
                pars = dirstr
        else:
            pars = sizestr
    elif str_type == "file":
        if len(dirstr) > 3: # specified direction
            if len(sizestr) > 3:
                pars = f"{sizestr}_{dirstr[4:]}"
            else:
                pars = dirstr
        else:
            pars = sizestr
    else:
        gen_util.accepted_values_error("str_type", str_type, ["print", "file"])
        
    return pars


#############################################
def stim_par_str(stimtype="gabors", bri_dir=None, bri_size=None, gabk=None,  
                 str_type="file"):
    """
    stim_par_str(par)

    Returns a string with stim type, as well as gabor kappa or brick size and 
    direction parameters, unless all possible parameters values for the stim 
    type are passed.

    Optional args:
        - stimtype (str)        : type of stimulus
                                  default: "gabors"
        - bri_dir (str or list) : brick direction parameter
                                  default: None
        - bri_size (int or list): brick size parameter
                                  default: None
        - gabk (int or list)    : gabor kappa parameter
                                  default: None
        - str_type (str)        : use of output str, i.e., for a filename 
                                  ("file") or to print the info to console 
                                  ("print")
                                  default: "file"

    Returns:
        - pars (str): string containing stim type and parameter values, unless
                      all parameter values for the stim type are passed.
    """
    
    all_pars = []
    if stimtype in ["gabors", "both"]:
        if gabk is None:
            raise ValueError("If stimulus is gabors, must pass gabk "
                "parameters.")
        pars = gabk_par_str(gabk, str_type)
        if stimtype == "both":
            all_pars.append(pars)
    elif stimtype in ["bricks", "both"]:
        if bri_size is None or bri_dir is None:
            raise ValueError("If stimulus is bricks, must pass direction and "
                "size parameters.")
        pars = bri_par_str(bri_dir, bri_size, str_type=str_type)
        if stimtype == "both":
            all_pars.append(pars)
    else:
        gen_util.accepted_values_error(
            "stimtype", stimtype, ["gabors", "bricks", "both"])

    if stimtype == "both":
        if str_type == "file":
            pars = "_".join(all_pars)
        elif str_type == "print":
            pars = ", ".join(all_pars)
        else:
            gen_util.accepted_values_error(
                "str_type", str_type, ["file", "print"])

    return pars


#############################################
def sess_par_str(sess_n, stimtype="gabors", layer="soma", bri_dir=None, 
                 bri_size=None, gabk=None, str_type="file"):
    """
    sess_par_str(sess_n)

    Returns a string from session and stimulus parameters for a filename, 
    or to print or use in a title.

    Required args:
        - sess_n (int or list)  : session number aimed for

    Optional args:
        - stimtype (str)        : type of stimulus
                                  default: "gabors"
        - layer (str)           : layer ("soma", "dend", "L23_soma", "L5_soma", 
                                         "L23_dend", "L5_dend", "L23_all", 
                                         "L5_all")
                                  default: "soma"
        - bri_dir (str or list) : brick direction parameter
                                  default: None
        - bri_size (int or list): brick size parameter
                                  default: None
        - gabk (int or list)    : gabor kappa parameter
                                  default: None
        - str_type (str)        : use of output str, i.e., for a filename 
                                  ("file") or to print the info to console 
                                  ("print")
                                  default: "file"
    Returns:
        - sess_str (list): string containing info on session and stimulus  
                           parameters
    """
    if gabk is None and (bri_size is None or bri_dir is None):
        raise ValueError("Must pass value for gabor k parameter or brick "
            "size and direction.")
    elif gabk is None:
        stimtype = "bricks"
    elif bri_size is None or bri_dir is None:
        stimtype = "gabors"

    stim_str = stim_par_str(stimtype, bri_dir, bri_size, gabk, str_type)

    if isinstance(sess_n, list):
        sess_n = gen_util.intlist_to_str(sess_n)

    if str_type == "file":
        sess_str = f"sess{sess_n}_{stim_str}_{layer}"
    elif str_type == "print":
        stim_str = stim_str.replace(" (", ": ").replace(")", "")
        sess_str = f"{stim_str}, session: {sess_n}, layer: {layer}"
    else:
        gen_util.accepted_values_error("str_type", str_type, ["file", "print"])
    
    return sess_str
    

#############################################
def datatype_par_str(datatype="roi"):
    """
    datatype_par_str()

    Returns a string for the datatype.

    Optional args:
        - datatype (str): type of data, i.e. "run" or "roi"
                          default: "roi"
    Returns:
        - data_str (list): string containing dimension
    """

    if datatype == "run":
        data_str = "running"
    elif datatype == "roi":
        data_str = "ROI"
    else:
        gen_util.accepted_values_error("datatype", datatype, ["roi", "run"])
    
    return data_str
    

#############################################
def datatype_dim_str(datatype="roi"):
    """
    datatype_dim_str()

    Returns a string for the dimension along which error is calculated for 
    the specified datatype.

    Optional args:
        - datatype (str): type of data, i.e. "run" or "roi"
                          default: "roi"
    Returns:
        - dim_str (list): string containing dimension
    """

    if datatype == "run":
        dim_str = "seqs"
    elif datatype == "roi":
        dim_str = "ROIs"
    else:
        gen_util.accepted_values_error("datatype", datatype, ["roi", "run"])
    
    return dim_str
    

#############################################
def pars_to_descr(par_str):
    """
    pars_to_descr()

    Converts numeric parameters in a string to parameter descriptions.

    Required args:
        - par_str (str): string with numeric parameters

    Returns:
        - par_str (str): string with parameter descriptions
    """

    vals  = [128.0, 256.0, 4.0, 16.0]
    descs = ["small", "big", "high disp", "low disp"]

    for val, desc in zip(vals, descs):
        par_str = par_str.replace(str(val), desc)
        par_str = par_str.replace(str(int(val)), desc)
    
    return par_str
    

#############################################
def get_split_oris(comp="DoriA"):
    """
    get_split_oris()

    Returns Gabor frames values split for orientation comparisons, if 
    applicable.

    Optional args:
        - comp (str): type of comparison
                      default: "DoriA"
    
    Returns
        - split_oris (bool or list): List of Gabor frames for each split, or 
                                     False if splitting orientation comparison 
                                     is not applicable.
    """

    split_oris = False
    if "ori" in comp:
        gab_letts = [lett.upper() for lett in comp.split("ori") 
                    if len(lett) > 0]
        if len(gab_letts) == 2:
            split_oris = gab_letts

    return split_oris


#############################################
def ext_test_str(q1v4=False, rvs=False, comp="surp", str_type="file"):
    """
    ext_test_str()
    
    Returns the string for the extra test set for logistic regressions, based 
    on the parameters. Returns "" if neither q1v4 nor regvsurp is True.

    Optional args:
        - q1v4 (bool)    : if True, analysis is separated across first and last 
                           quintiles
                           default: False
        - rvs (bool)     : if True, analysis is separated across regular and 
                           surprise sequences 
                           default: False
        - comp (str)     : type of comparison
                           default: "surp"
        - str_type (str) : use of output str, i.e., for a filename 
                           ("file") or to print the info to console ("print")
                           or for a label ("label")
                           default: "file"

    Returns:
        - ext_str (str): string for the extra dataset ("" if neither 
                         q1v4 nor rvs is True), and comparison details
    """
    if q1v4 + rvs > 1:
        raise ValueError("'q1v4' and 'rvs' cannot both be True.")

    if str_type not in ["file", "print", "label"]:
        gen_util.accepted_values_error(
            "str_type", str_type, ["file", "print", "label"])

    split_oris = get_split_oris(comp)

    if q1v4:
        if str_type == "file":
            ext_str = "test_Q4"
        elif str_type == "label":
            ext_str = " (only Q1)"
        else:
            ext_str = " (trained on Q1 and tested on Q4)"
    elif rvs:
        if str_type == "file":
            ext_str = "test_surp"
        elif str_type == "label":
            ext_str = " (only reg)"
        else:
            ext_str = " (trained on reg and tested on surp)"
    elif split_oris is not False:
        if str_type == "file":
            ext_str = f"test_{split_oris[1]}"
        elif str_type == "label":
            ext_str = f" ({split_oris[0]} Gabors)"
        else:
            ext_str = " (trained on {} and tested on {})".format(*split_oris)
    else:
        ext_str = ""

    return ext_str


#############################################
def get_nroi_strs(sess_info, remnans=True, fluor="dff", empty=False, 
                  style="comma"):
    """
    get_nroi_strs(sess_info)

    Returns strings with number of ROIs for each session.

    Required args:
        - sess_info (dict): dictionary containing information from each
                            session 
            ["mouse_ns"] (list)   : mouse numbers
            if not empty:
            ["nrois"] (list)      : number of ROIs in session
            if remnans:
            ["nanrois_{}"] (list) : list of ROIs with NaNs/Infs in traces 
                                    ("raw" or "dff")
        - remnans (bool)  : if True, the number of ROIs with NaN/Infs is  
                            removed from the total
                            default: True
        - fluor (str)     : if "raw", number of ROIs is calculated with 
                            n_nanrois. If "dff", it is calculated with 
                            n_nanrois_dff  
                            default: "dff"
        - empty (bool)    : if True, empty strings are returned for each session
                            default: False
        - style (str)     : style to use (following a comma ("comma") or in 
                            parentheses ("par"))

    Returns:
        - nroi_strs (list): list of strings containing number of ROIs for each 
                            session
    """

    if empty:
        nroi_strs = [""] * len(sess_info["mouse_ns"])

    else:
        nrois = sess_info["nrois"]
        if remnans:
            sub_rois = sess_info[f"nanrois_{fluor}"]
            sub_vals = [len(rois) for rois in sub_rois]
            nrois = [nrois[i] - sub_vals[i] for i in range(len(nrois))]
        
        if style == "comma":
            nroi_strs = [f", n={nroi}" for nroi in nrois]
        elif style == "par":
            nroi_strs = [f" (n={nroi})" for nroi in nrois]
        else:
            gen_util.accepted_values_error("style", style, ["comma", "par"])
    
    return nroi_strs


#############################################
def get_stimdir(stimtype="gabors", gabfr=0):
    """
    get_stimdir()

    Returns directory with stimulus parameter name and gabor frame, if 
    applicable.

    Optional args:
        - stimtype (str): stimulus type
                          default: "gabors"
        - gabfr (int)   : gabor frame number
                          default: 0
    
    Returns:
        - stimdir (str): stimulus directory
    """

    stimdir = stimtype[:3]
    if stimtype == "gabors":
        gab_lett = gabfr_letters(gabfr)
        if "/" in gab_lett:
            gab_lett = gab_lett.replace("/", "")
        stimdir = f"{stimdir}{gab_lett}"
    elif stimtype == "both":
        stimdir = stimtype

    return stimdir


#############################################
def get_position_name(position):
    """
    get_position_name()

    Returns position name based on the index (e.g., "first" for 0).

    Required args:
        - position (int): position number type
    
    Returns:
        - (str): position name
    """

    position_names = [
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh", 
        "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", 
        "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth", 
        "nineteenth", "twentieth"]

    if not int(position) == float(position):
        raise TypeError("position must be of type int.")
    if position >= len(position_names):
        raise ValueError(
            f"Only values smaller than {len(position_names)} allowed for "
            "'position'.")

    return position_names[position]

