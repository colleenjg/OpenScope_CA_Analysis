"""
plot_from_dicts_tool.py

This script contains functions to plot from dictionaries.

Authors: Colleen Gillon

Date: October, 2019

Note: this code uses python 3.7.

"""

import glob
import os

from joblib import Parallel, delayed

from util import gen_util
from plot_fcts import roi_analysis_plots as roi_plots
from plot_fcts import gen_analysis_plots as gen_plots
from plot_fcts import pup_analysis_plots as pup_plots
from plot_fcts import modif_analysis_plots as mod_plots
from plot_fcts import acr_sess_analysis_plots as acr_sess_plots
from plot_fcts import logreg_plots, glm_plots



#############################################
def plot_from_dicts(direc, source='roi', plt_bkend=None, fontdir=None, 
                    plot_tc=True, parallel=False):
    """
    plot_from_dicts(direc)

    Plots data from dictionaries containing analysis parameters and results, or 
    path to results.

    Required args:
        - direc (str): path to directory in which dictionaries to plot data 
                       from are located
    
    Optional_args:
        - source (str)   : plotting source ('roi', 'run', 'gen', 'pup', 
                           'modif', 'logreg', 'glm')
        - plt_bkend (str): mpl backend to use for plotting (e.g., 'agg')
                           default: None
        - fontdir (str)  : directory in which additional fonts are stored
                           default: None
        - plot_tc (bool) : if True, tuning curves are plotted for each ROI 
                           default: True
        - parallel (bool): if True, some of the analysis is parallelized across 
                           CPU cores
                           default: False
    """

    if os.path.isdir(direc):
        if source == 'logreg': 
            fn = 'hyperparameters.json'
            all_paths = glob.glob(os.path.join(direc, fn)) + \
                        glob.glob(os.path.join(direc, '*', fn))
            dict_paths = [os.path.dirname(dp) for dp in all_paths]
        else:
            dict_paths = glob.glob(os.path.join(direc, '*.json'))
        
        if len(dict_paths) == 0:
            raise ValueError(f'No jsons found in directory: {direc}.')
    else:
        dict_paths = [direc]

    sub_parallel = parallel * (len(dict_paths) == 1)

    sources = ['roi', 'run', 'gen', 'modif', 'pup', 'logreg', 'glm', 'acr_sess']
    args = [plt_bkend, fontdir]
    if source == 'roi':
        fct = roi_plots.plot_from_dict
        args.extend([plot_tc, sub_parallel])
    elif source in ['run', 'gen']:
        fct = gen_plots.plot_from_dict
        args.extend([sub_parallel])
    elif source in ['pup', 'pupil']:
        fct = pup_plots.plot_from_dict
        args.extend([sub_parallel])
    elif source == 'modif':
        fct = mod_plots.plot_from_dict
        args.extend([plot_tc, sub_parallel])
    elif source == 'logreg':
        fct = logreg_plots.plot_from_dict
    elif source == 'glm':
        fct = glm_plots.plot_from_dict
    elif source == 'acr_sess':
        fct = acr_sess_plots.plot_from_dict
        args.extend([sub_parallel])
    else:
        gen_util.accepted_values_error('source', source, sources)

    if parallel and len(dict_paths) > 1:
        n_jobs = gen_util.get_n_jobs(len(dict_paths))
        Parallel(n_jobs=n_jobs)(delayed(fct)(dict_path, *args) 
                                for dict_path in dict_paths)
    else:
        for dict_path in dict_paths:
            fct(dict_path, *args)


