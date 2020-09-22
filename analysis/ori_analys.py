"""
ori_analys.py

This module contains functions to run analyses of ROI orientation selectivity 
in the data generated by the AIBS experiments for the Credit Assignment Project

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

from joblib import Parallel, delayed
import numpy as np
import scipy.stats as st
from sklearn import linear_model

from util import gen_util, logger_util, math_util
from sess_util import sess_gen_util


#############################################
def collapse_dir(oris):
    """
    collapse_dir(oris)

    Collapses opposite orientations in the -180 to 180 degree range to the -90 
    to 90 range.

    Required args:
        - oris (list or nd array): array of orientations in the -180 to 180 
                                   range
    
    Return:
        - oris_nodir (nd array): array of orientations collapsed to within the 
                                 -90 to 90 range
    """

    oris_nodir = np.copy(oris)
    if (np.absolute(oris) > 180).any():
        raise ValueError("Only orientations between -180 and 180 are "
            "accepted.")

    ori_ch = np.where(np.absolute(oris) > 90)
    new_vals = oris[ori_ch] - np.sign(oris[ori_ch]) * 180.0
    oris_nodir[ori_ch] = new_vals

    return oris_nodir


#############################################
def estim_vm(x_cuml, data_sort, tc_oris, hist_n=1000):
    """
    estim_vm(x_cuml, data_sort, tc_oris)

    Returns estimates of von Mises distributions for an ROI based on the 
    input orientations and ROI activations.

    Required args:
        - x_cuml (1D array)   : mask array where values are incremented for 
                                each new orientation (sorted)
        - data_sort (1D array): ROI values corresponding to the mask
        - tc_oris (list)      : list of orientation values (sorted and unique)
   
    Optional args:
        - hist_n (int): number of values to build the histogram with
                        default: 1000 

   Returns:
        - av_roi_data (list): list of mean integrated fluorescence data per 
                              unique orientation
        - vm_pars (list)    : tuning curve (von Mises) parameter estimates
                              (kappa, mean, scale)
        - hist_pars (list)  : parameters to create histograms from (sub, mult) 
    """

    av_roi_data = np.bincount(
        x_cuml, data_sort)/np.bincount(x_cuml).astype(float)
    sub = np.min(av_roi_data)
    mult = hist_n/np.sum(av_roi_data - sub)
    counts = np.around((av_roi_data - sub) * mult).astype(int)
    freq_data = np.repeat(tc_oris, counts)
    # von mises fit
    pars = st.vonmises.fit(np.radians(freq_data), fkappa=0.5, fscale=0.5)
    vm_pars = st.vonmises.fit(np.radians(freq_data), floc=pars[1], fscale=0.5)
    hist_pars = [sub, mult]

    av_roi_data = av_roi_data.tolist()

    return av_roi_data, vm_pars, hist_pars


#############################################
def estim_vm_by_roi(oris, roi_data, hist_n=1000, parallel=False):
    """
    estim_vm_by_roi(oris, roi_data)

    Returns estimates of von Mises distributions for each ROI based on the 
    input orientations and ROI activations.

    Required args:
        - oris (array-like)  : array of orientations
        - roi_data (2D array): corresponding array of ROI values, 
                               structured as ROI x ori
    
    Optional args:
        - hist_n (int)   : number of values to build the histogram with
                           default: 1000 
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores

    Returns:
        - tc_oris (list)         : list of orientation values (sorted and 
                                   unique)
        - tc_data (list)         : list of mean integrated fluorescence data 
                                   per unique orientation, structured 
                                   as ROI x orientation
        - tc_vm_pars (nd array)  : tuning curve (von Mises) parameter estimates, 
                                   structured as ROI x par (kappa, mean, scale)
        - tc_hist_pars (nd array): paremeters to create histograms, structured 
                                   as ROI x par (sub, mult)
    """

    # sort by gab orientation
    x_sort_idx    = np.argsort(oris)
    xsort         = oris[x_sort_idx]
    roi_data_sort = roi_data[:, x_sort_idx]

    # get mask of int values for each unique x value
    x_cuml = np.insert((np.diff(xsort) != 0).cumsum(), 0, 0)
    tc_oris = np.unique(xsort).tolist()

    if parallel:
        n_jobs = gen_util.get_n_jobs(len(roi_data_sort))
        returns = Parallel(n_jobs=n_jobs)(delayed(estim_vm)
            (x_cuml, data_sort, tc_oris, hist_n) for data_sort in roi_data_sort)       
    else:
        returns = []
        for data_sort in roi_data_sort:
            returns.append(estim_vm(x_cuml, data_sort, tc_oris, hist_n))        

    returns = list(zip(*returns))
    tc_data = [list(ret) for ret in returns[0]]
    tc_vm_pars = np.asarray([list(ret) for ret in returns[1]])
    tc_hist_pars = np.asarray([list(ret) for ret in returns[2]])
    
    return tc_oris, tc_data, tc_vm_pars, tc_hist_pars


#############################################
def tune_curv_estims(gab_oris, roi_data, ngabs_tot, nrois="all", ngabs="all",
                     comb_gabs=False, hist_n=1000, collapse=True, 
                     parallel=False):
    """
    tune_curv_estims(gab_oris, roi_data, ngabs_tot)

    Returns estimates of von Mises distributions for ROIs based on the 
    input orientations and ROI activations.

    Required args:
        - gab_oris (2D array): gabor orientation values (deg), 
                               structured as gab x seq
        - roi_data (2D array): ROI fluorescence data, structured as ROI x seq
        - ngabs_tot (int)    : total number of ROIs

    Optional args:
        - nrois (int)     : number of ROIs to include in analysis
                            default: "all"
        - ngabs (int)     : number of gabors to include in analysis (set to 1 
                            if comb_gabs, as all gabors are combined)
                            default: "all"
        - comb_gabs (bool): if True, all gabors have been combined for 
                            gab_oris and roi_data 
                            default: False
        - hist_n (int)    : value by which to multiply fluorescence data to 
                            obtain histogram values
                            default: 1000
        - collapse (bool) : if True, opposite orientations in the -180 to 180 
                            range are collapsed to the -90 to 90 range 
        - parallel (bool) : if True, some of the analysis is parallelized 
                            across CPU cores

    Returns:
        - gab_tc_oris (list)      : list of orientation values (deg) 
                                    corresponding to the gab_tc_data, 
                                    structured as 
                                       gabor (1 if comb_gabs) x oris 
        - gab_tc_data (list)      : list of mean integrated fluorescence data 
                                    per orientation, for each ROI, structured 
                                    as: 
                                       ROI x gabor (1 if comb_gabs) x oris
        - gab_vm_pars (3D array)  : array of Von Mises parameters for each ROI: 
                                       ROI x gabor (1 if comb_gabs) x par
        - gab_vm_mean (2D array)  : array of mean Von Mises means for each ROI, 
                                    not weighted by kappa value or weighted 
                                    (if not comb_gabs) (in rad): 
                                        ROI x kappa weighted (False, (True))
        - gab_hist_pars (3D array): parameters used to convert tc_data to 
                                    histogram values (sub, mult) used in Von 
                                    Mises parameter estimation, structured as:
                                       ROI x gabor (1 if comb_gabs) x 
                                             param (sub, mult)
    """

    gab_oris = collapse_dir(gab_oris)
    
    kapw_bool = [0, 1]
    if comb_gabs:
        hist_n *= ngabs_tot           
        kapw_bool = [0]
        ngabs = 1

    if ngabs == "all":
        ngabs = ngabs_tot
    if nrois == "all":
        roi_data.shape[0]

    if parallel and ngabs > nrois:
        n_jobs = gen_util.get_n_jobs(ngabs)
        returns = Parallel(n_jobs=n_jobs)(delayed(estim_vm_by_roi)
            (gab_oris[g], roi_data, hist_n, False) for g in range(ngabs))       
    else:
        returns = []
        for g in range(ngabs):
            returns.append(estim_vm_by_roi(
                gab_oris[g], roi_data, hist_n, parallel))
    
    returns = list(zip(*returns))
    gab_tc_oris = [list(ret) for ret in returns[0]]
    gab_tc_data = [list(ret) for ret in zip(*returns[1])] # move ROIs to first
    gab_vm_pars = np.transpose(
        np.asarray([list(ret) for ret in returns[2]]), [1, 0, 2])
    gab_hist_pars = np.transpose(
        np.asarray([list(ret) for ret in returns[3]]), [1, 0, 2])
    means = gab_vm_pars[:, :, 1] 
    kaps  = gab_vm_pars[:, :, 0]

    gab_vm_mean = np.empty([nrois, len(kapw_bool)])
    gab_vm_mean[:, 0] = st.circmean(means, np.pi/2., -np.pi/2, axis=1)
    if not comb_gabs:
        import astropy.stats as astrost
        # astropy only implemented with -pi to pi range
        gab_vm_mean[:, 1] = astrost.circmean(
            means * 2., axis=1, weights=kaps)/2.
    
    return gab_tc_oris, gab_tc_data, gab_vm_pars, gab_vm_mean, gab_hist_pars


#############################################
def calc_tune_curvs(sess, analyspar, stimpar, nrois="all", ngabs="all", 
                    grp2="surp", comb_gabs=True, prev=False, collapse=True, 
                    parallel=True):
    """
    calc_tune_curvs(sess, analyspar, stimpar)

    Returns orientations and corresponding fluorescence levels for the sessions
    of interest.

    Required args:
        - sess (Session)       : session
        - analyspar (AnalysPar): named tuple containing analysis parameters
        - stimpar (StimPar)    : named tuple containing stimulus parameters

    Optional args:
        - nrois (int)     : number of ROIs to include in analysis
                            default: "all"
        - ngabs (int)     : number of gabors to include in analysis (set to 1 
                            if comb_gabs, as all gabors are combined)
                            default: "all"
        - grp2 (str)      : second group: either surp, reg or rand (random 
                            subsample of reg, the size of surp)
                            default: "surp"
        - comb_gabs (bool): if True, all gabors have been combined for 
                            gab_oris and roi_data 
                            default: False
        - prev (bool)     : if True, analysis is run using previous tuning 
                            estimation method
        - collapse (bool) : if True, opposite orientations in the -180 to 180 
                            range are collapsed to the -90 to 90 range 
        - parallel (bool) : if True, some of the analysis is parallelized 
                            across CPU cores

    Returns:
        - tc_oris (list)     : list of orientation values corresponding to the 
                               tc_data:
                                   surp x gabor (1 if comb_gabs) x oris
        - tc_data (list)     : list of mean integrated fluorescence data per 
                               orientation, for each ROI, structured as 
                                  ROI x surp x gabor (1 if comb_gabs) 
                                      x oris
        - tc_nseqs (list)    : number of sequences per surp

        if prev, also:
        - tc_vm_pars (list)  : nested list of Von Mises parameters for each ROI: 
                                  ROI x surp x gabor (1 if comb_gabs) x par
        - tc_vm_mean (list)  : nested list of mean Von Mises means for each ROI, 
                               not weighted by kappa value or weighted (if not 
                               comb_gabs) (in rad): 
                                   ROI x surp x kappa weighted (False, (True))
        - tc_hist_pars (list): parameters used to convert tc_data to histogram 
                               values (sub, mult) used in Von Mises parameter 
                               estimation, structured as:
                                   ROI x surp x gabor (1 if comb_gabs) x 
                                   param (sub, mult)
    """
    
    gabfrs = gen_util.list_if_not(stimpar.gabfr)
    if len(gabfrs) == 1:
        gabfrs = gabfrs * 2
    if grp2 == "surp":
        surps = [0, 1]
    elif grp2 in ["reg", "rand"]:
        surps = [0, 0]
    else:
        gen_util.accepted_values_error("grp2", grp2, ["surp", "reg", "rand"])
    
    stim = sess.get_stim(stimpar.stimtype)
    nrois_tot = sess.get_nrois(analyspar.remnans, analyspar.fluor)
    ngabs_tot = stim.n_patches
    if nrois == "all":
        sess_nrois = nrois_tot
    else:
        sess_nrois = np.min([nrois_tot, nrois])
    
    if ngabs == "all":
        sess_ngabs = stim.n_patches
    else:
        sess_ngabs = np.min([stim.n_patches, ngabs])                

    tc_data, tc_oris, tc_nseqs = [], [], []
    if prev: # PREVIOUS ESTIMATION METHOD
        tc_vm_pars, tc_vm_mean, tc_hist_pars = [], [], []

    for i, (gf, s) in enumerate(zip(gabfrs, surps)):
        # get segments
        segs = stim.get_segs_by_criteria(
            gabfr=gf, bri_dir=stimpar.bri_dir, bri_size=stimpar.bri_size, 
            gabk=stimpar.gabk, surp=s, by="seg")
        
        if grp2 == "rand" and i == 1:
            n_segs = len(stim.get_segs_by_criteria(
                gabfr=gf, bri_dir=stimpar.bri_dir, bri_size=stimpar.bri_size, 
                gabk=stimpar.gabk, surp=1, by="seg"))
            np.random.shuffle(segs)
            segs = sorted(segs[: n_segs])
        tc_nseqs.append(len(segs))
        twopfr = stim.get_twop_fr_by_seg(segs, first=True)["first_twop_fr"]
        # ROI x seq
        roi_data = stim.get_roi_data(
            twopfr, stimpar.pre, stimpar.post, analyspar.fluor, integ=True, 
            remnans=analyspar.remnans, scale=analyspar.scale
            )["roi_traces"].unstack().to_numpy()[:sess_nrois]

        # gab x seq 
        gab_oris = gen_util.reshape_df_data(stim.get_stim_par_by_seg(
            segs, pos=False, ori=True, size=False), squeeze_cols=True).T
        if collapse:
            gab_oris = collapse_dir(gab_oris)

        if comb_gabs:
            ngabs = 1
            gab_oris = gab_oris.reshape([1, -1])
            roi_data = np.tile(roi_data, [1, ngabs_tot])
            
        tc_oris.append(gab_oris.tolist())
        tc_data.append(roi_data.tolist())

        if prev: # PREVIOUS ESTIMATION METHOD
            [gab_tc_oris, gab_tc_data, gab_vm_pars, 
                gab_vm_mean, gab_hist_pars] = tune_curv_estims(
                    gab_oris, roi_data, ngabs_tot, sess_nrois, sess_ngabs, 
                    comb_gabs, collapse=False, parallel=parallel)
            tc_oris[i] = gab_tc_oris
            tc_data[i] = gab_tc_data
            tc_vm_pars.append(gab_vm_pars)
            tc_vm_mean.append(gab_vm_mean)
            tc_hist_pars.append(gab_hist_pars)

    if prev:
        return tc_oris, tc_data, tc_nseqs, tc_vm_pars, tc_vm_mean, tc_hist_pars
    else:
        return tc_oris, tc_data, tc_nseqs


#############################################
def ori_pref_regr(ori_prefs):
    """
    ori_pref_regr(ori_prefs)

    Calculates the correlation between 2 sets of orientation preferences.

    Required args:
        - ori_prefs (3D array): array of mean orientation preferences (in rad), 
                                not weighted by kappa value or weighted 
                                , structured as: 
                                      vals x set
                                           x kappa weighted (False, (True))

    Returns:
        - ori_regr (2D array): array of regression results, structured as
                               regr values (R2, slope, intercept) x
                                   kappa weighted (False, (True))
    """

    ori_prefs = np.asarray(ori_prefs)
    kapw_bool = list(range(ori_prefs.shape[-1]))

    # R squared, slope, intercept x non kappa weighted/kappa weighted
    ori_regr = np.empty([3, len(kapw_bool)]) 
    
    for i in kapw_bool: # mean value idx (0: not weighted, 1: weighted)
        regr = linear_model.LinearRegression(fit_intercept=True)
        regr.fit(ori_prefs[:, 0:1, i], ori_prefs[:, 1:2, i])
        r_sqr = regr.score(ori_prefs[:, 0:1, i], ori_prefs[:, 1:2, i])
        slope = regr.coef_.squeeze()
        intercept = regr.intercept_.squeeze()
        ori_regr[:, i] = r_sqr, slope, intercept

    return ori_regr

