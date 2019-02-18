import yaml
import os
import argparse
import random
import glob
import multiprocessing

from matplotlib import pyplot as plt
import scipy.stats as st
import torch
import torchvision
import h5py
import pickle
import pandas as pd
from joblib import Parallel, delayed
import re

np = torch._np

from util import data_util, file_util, gen_util, logreg_util, math_util, plot_util, str_util


#############################################
def get_compdir(mouse_n, sess_n, layer, fluor='dff', norm='roi', comp='surp', 
                shuffle=False):
    """
    get_compdir(mouse_n, sess_n, layer)

    Generates the name of the general directory in which an analysis type is
    saved, based on analysis parameters.
    
    Required arguments:
        - mouse_n (int): mouse number
        - sess_n (int) : session number
        - layer (str)  : layer name
    
    Optional arguments:
        - fluor (str)   : fluorescence trace type
                          default: 'dff'
        - norm (str)    : type of normalization 
                          default: 'roi'
        - comp (str)    : type of comparison
                          default: 'surp'
        - shuffle (bool): whether analysis is on shuffled data
                          default: False

    Returns:
        - compdir (str): name of directory to save analysis in
    """

    fluor_str = fluor
    if norm == 'none':
        norm_bool = False
    elif norm in ['roi', 'all']:
        norm_bool = True

    norm_str = str_util.norm_par_str(norm_bool, type_str='file')
    shuff_str = str_util.shuff_par_str(shuffle, type_str='file')

    compdir = 'm{}_s{}_{}_{}{}_{}{}'.format(mouse_n, sess_n, layer, fluor_str, 
                                            norm_str, comp, shuff_str)

    return compdir


#############################################
def get_rundir(run, uniqueid=None):
    """
    get_rundir(run)

    Generates the name of the specific subdirectory in which an analysis is
    saved, based on a run number and unique ID.
    
    Required arguments:
        - run (int): run number
    
    Optional arguments:
        - uniqueid (str or int): unique ID for analysis
                                 default: None

    Returns:
        - rundir (str): name of subdirectory to save analysis in
    """

    if uniqueid is None:
        rundir = 'run_{}'.format(run)
    else:
        rundir = '{}_{}'.format(uniqueid, run)

    return rundir


#############################################
def get_compdir_dict(rundir):
    """
    get_compdir_dict(rundir)

    Generates a dictionary with analysis parameters based on the full analysis 
    path.
    
    Required arguments:
        - rundir (str): path of subdirectory in which analysis is saved,
                        structured as 
                        '.../mouse_sess_layer_fluor_norm_comp_shuffled/uniqueid_run'
    
    Returns:
        - compdir_dict (dict): analysis parameters dictionary
    """

    parts = rundir.split(os.sep)
    first = parts[-2].split('_')
    second = parts[-1].split('_')

    if len(second) == 3:
        # rejoin if uniqueid is datetime
        second = ['{}_{}'.format(second[0], second[1]), int(second[2])]
    else:
        second = [int(x) for x in second] 

    if first[4] == 'norm':
        comp = first[5]
    else:
        comp = first[4]
    
    if first[-1] == 'shuffled':
        shuffle = True
    else:
        shuffle = False

    compdir_dict = {'mouse_n': int(first[0][1]),
                    'sess_n':int(first[1][1]),
                    'layer': first[2],
                    'fluor': first[3],
                    'comp': comp,
                    'shuffled': shuffle,
                    'uniqueid': second[0],
                    'run': int(second[1])
                    }
    
    return compdir_dict


#############################################
def load_sess_dict(mouse_n, sess_n, layer, runtype='prod'):
    """
    load_sess_dict(mouse_n, sess_n, layer)

    Loads a session dictionary for the mouse, session number and layer, as
    well as the runtype.
    
    Required arguments:
        - mouse_n (int): mouse number
        - sess_n (int) : session number
        - layer (str)  : layer name
    
    Optional arguments:
        - runtype (str): type of run ('prod' or 'pilot')
                         default: 'prod'
    
    Returns:
        - sess_dict (dict): session dictionary
    """

    sess_dict_name = 'sess_dict_mouse{}_sess{}_{}.json'.format(mouse_n, 
                    sess_n, layer)
    sess_dict_dir = os.path.join('session_dicts', runtype)

    sess_dict_file = os.path.join(sess_dict_dir, sess_dict_name)
    if os.path.exists(sess_dict_file):
        sess_dict = file_util.load_file(sess_dict_file, file_type='json')
    else:
        print(('No session dictionary found for\n\tMouse: {}\n\tSess: {}\n\t'
               'Layer: {}\n\tRuntype: {}\n'.format(mouse_n, sess_n, layer, 
               runtype)))
        sess_dict = None

    return sess_dict


#############################################
def info_dict(args, epoch=None):
    """
    info_dict(args)

    Creates an info dictionary for an analysis from the args. Includes epoch
    number is it is passed. If args is None, the list of dictionary keys
    is returned instead. 
    
    Required arguments:
        - args (Argument parser): parser containing analysis parameters:
                comp (str)           : type of comparison
                fluor (str)          : fluorescence trace type
                layer (str)          : layer name
                line (str)           : transgenic line name
                mouse_n (int)        : mouse number
                n_roi (int)          : nuber of ROIs in analysis 
                norm (str)           : type of normalization
                run (int)            : run number
                runtype (str)        : type of run ('prod' or 'pilot')
                sess_n (int)         : session number
                shuffle (bool)       : whether analysis is on shuffled data
                uniqueid (str or int): unique ID for analysis
    
    Optional arguments:
        - epoch (int): epoch number
                       default: None
    
    Returns:
        args is an Argument parser:
            - info (dict): analysis dictionary
        args is None:
            - info (list): list of dictionary keys
    """

    if args is not None:
        info = {'mouse_n': args.mouse_n,   'sess_n': args.sess_n,
                'layer': args.layer,       'line': args.line,
                'fluor': args.fluor,       'norm': args.norm,
                'shuffled': args.shuffle,  'comp': args.comp,
                'uniqueid': args.uniqueid, 'run': args.run,
                'runtype': args.runtype,   'n_roi': args.n_roi}
        
        if epoch is not None:
            info['epoch'] = epoch

    # if no args are passed, just returns keys
    else:
        info = ['mouse_n', 'sess_n', 'layer', 'line', 'fluor', 'norm', 
                'shuffled', 'comp', 'uniqueid', 'run', 'runtype', 'n_roi', 
                'epoch']

    return info


#############################################
def save_hyperpar(args, task='run_regr'): 
    """
    save_hyperpar(args)

    Saves the hyperparameters for an analysis.
    
    Required arguments:
        - args (Argument parser): parser containing analysis parameters:
                dirname (str): name of directory in which to save scores
    
    Optional arguments:
        - task (str): task name
                      default: 'run_regr'
    """

    args_dict = args.__dict__.copy()

    if task == 'run_regr':
        # removing non task (CI and fig params) and redundant params 
        # (output and compdir included in dirname and cuda overriden by device)
        for key in ['output', 'compdir', 'cuda', 'CI', 'ncols', 'no_sharey', 
                    'subplot_wid', 'subplot_hei']:
            args_dict.pop(key)

    file_util.save_info(args_dict, 'hyperparameters', args.dirname, 'json')


#############################################
def get_roi_traces(data_dir, roi_tr_file):
    """
    get_roi_traces(data_dir, roi_tr_file)

    Returns ROI traces, along with number of ROIs and frames.
    
    Required arguments:
        - data_dir (str)   : path to the data directory
        - roi_tr_file (str): relative path to the ROI traces
    
    Returns:
        - roi_traces (2D array): array of ROI traces, nroi x nframes
        - nroi (int)           : nbr of ROIs
        - nframes (int)        : nbr of frames
    """

    roi_tr_file = os.path.join(data_dir, roi_tr_file)

    with h5py.File(roi_tr_file,'r') as f:
        # get traces
        roi_traces = np.asarray(f['data'].value)
        nroi = roi_traces.shape[0]
        nframes = roi_traces.shape[1]
    
    return roi_traces, nroi, nframes


#############################################
def gab_classes(comp='surp', gab_fr=0):
    """
    gab_classes()

    Returns information on the Gabor segments being compared: 
    segment length, class names, Gabor frame names, Gabor frame numbers
    
    Optional arguments:
        - comp (str)  : type of comparison
                        default: 'surp'
        - gab_fr (int): number of the Gabor frame with which segment should
                        start (for full segment comparisons)
                        default: 0
    
    Returns:
        - len_s (float)       : length of segments being compared
        - classes (list)      : list of class names
        - gab_fr (int or list): Gabor frame nbr or list of Gabor frame nbrs 
        - gabs (str or list)  : Gabor frame name or list of Gabor frame names
    """

    frame_names = ['A', 'B', 'C', 'D/E']
    if comp == 'surp':
        len_s = 1.5
        classes = ['Regular', 'Surprise']
        gabs = frame_names[gab_fr]

    elif comp == 'DvE':
        len_s = 0.45
        classes = ['Gabor D', 'Gabor E']
        gab_fr = frame_names.index('D/E')
        gabs = frame_names[gab_fr]

    elif comp in ['AvB', 'AvC', 'BvC']:
        len_s = 0.45
        classes = ['Gabor {}'.format(fr) for fr in [comp[0], comp[2]]]
        gab_fr = [frame_names.index(fr) for fr in [comp[0], comp[2]]]
        gabs = [frame_names[gf] for gf in gab_fr]
    
    else:
        gen_util.accepted_values_error('comp', comp, ['surp', 'AvB', 'AvC', 
                                       'BvC', 'DvE'])

    return len_s, classes, gab_fr, gabs



#############################################
def comp_segs(sess_dict, gab_fr=0, comp='surp', fluor='dff'):
    """
    comp_segs(sess_dict)

    Returns information on the Gabor segments being compared: 
    segment length, class names, Gabor frame names, Gabor frame numbers
    
    Required arguments:
        - sess_dict (dict): session dictionary
                ['frames'] (list)     : session number
                ['gab_fr'] (list)     : list where the first element is the
                                        number of the Gabor frame for which 
                                        frame numbers were recorded
                ['nanrois_dff'] (list): list of ROIs with NaNs in dff traces
                ['nanrois'] (list)    : list of ROIs with NaNs in traces
                ['surp_idx'] (list)   : list of frames in surprise segment
                ['twop_fps'] (float)  : recording frames per second

    Optional arguments:
        - comp (str)  : type of comparison
                        default: 'surp'
        - gab_fr (int): number of the Gabor frame with which segment should
                        start (for full segment comparisons)
                        default: 0
        - fluor (str) : fluorescence trace type
                        default: 'dff'
    
    Returns:
        - gabs (str or list)    : Gabor frame name or list of Gabor frame names
        - classes (list)        : list of class names
        - seg_fr (2D array)     : array of frame numbers, structured as 
                                  segments x frames 
        - seg_classes (2D array): array of segment classes, structures as 
                                  classes x 1
        - n_surp (int)          : number of surprise segments
        - nan_rois (list)       : list of ROIs with NaNs in traces
    """


    fps = sess_dict['twop_fps']
    frames = sess_dict['frames']
    n_surp = len(sess_dict['surp_idx'])

    if fluor == 'dff':
        nan_rois = sess_dict['nanrois_dff']
    else:
        nan_rois = sess_dict['nanrois']

    len_s, classes, gab_fr, gabs = gab_classes(comp, gab_fr)

    if comp in ['surp', 'DvE']:
        pre_s = (sess_dict['gab_fr'][0] - gab_fr) * 0.3 # in sec
        seg_fr = gen_util.idx_segs(frames, pre=pre_s*fps, leng=len_s*fps)

        seg_classes = np.zeros([len(frames), 1])
        seg_classes[sess_dict['surp_idx']] = 1

    elif comp in['AvB', 'AvC', 'BvC']:
        pre_s = [(sess_dict['gab_fr'][0] - gf)*0.3 for gf in gab_fr] # in sec
        seg_fr = [gen_util.idx_segs(frames, pre=pr*fps, leng=0.45*fps) 
                  for pr in pre_s]

        # trim segments if longer from one class than the other
        seg_len = min(seg_fr[0].shape[1], seg_fr[1].shape[1])

        seg_classes = np.concatenate((np.zeros([len(seg_fr[0]), 1]), 
                                      np.ones([len(seg_fr[1]), 1])), axis=0)
        seg_fr = np.concatenate([seg_fr[0][:, :seg_len], 
                                 seg_fr[1][:, :seg_len]], axis=0)

    return gabs, classes, seg_fr, seg_classes, n_surp, nan_rois


#############################################
def get_sess_data(data_dir, mouse_n, sess_n, layer, comp='surp', gab_fr=0, 
                  fluor='dff', runtype='prod'):
    """
    get_sess_data(data_dir, mouse_n, sess_n, layer)

    Print session information and returns ROI trace segments, target classes 
    and class information and number of surprise segments in the dataset.
    
    Required arguments:
        - data_dir (str): path to the data directory
        - mouse_n (int) : mouse number
        - sess_n (int)  : session number
        - layer (str)   : layer name

    Optional arguments:
        - comp (str)   : type of comparison
                         default: 'surp'
        - gab_fr (int) : number of the Gabor frame with which segment should
                         start (for full segment comparisons)
                         default: 0
        - fluor (str)  : fluorescence trace type
                         default: 'dff'
        - runtype (str): type of run ('prod' or 'pilot')
                         default: 'prod'
 
    Returns:
        - roi_tr_segs (3D array): array of all ROI trace segments, structured as 
                                  segments x frames x ROIs
        - classes (list)        : list of class names
        - seg_classes (2D array): array of all segment classes, structured as 
                                  classes x 1
        - n_surp (int)          : number of surprise segments
    """

    sess_dict = load_sess_dict(mouse_n, sess_n, layer, runtype)

    if sess_dict is not None:
        if fluor == 'raw':
            roi_tr_file = sess_dict['traces_dir']
        elif fluor == 'dff':
            roi_tr_file = sess_dict['dff_traces_dir']
        else:
            gen_util.accepted_values_error('fluor', fluor, ['raw', 'dff'])

        roi_traces, nroi, _ = get_roi_traces(data_dir, roi_tr_file)
    
        [gabs, classes, seg_fr, seg_classes, 
            n_surp, nan_rois] = comp_segs(sess_dict, gab_fr, comp, fluor)

        roi_tr_segs = roi_traces[:, seg_fr].transpose(1, 2, 0)

        roi_tr_segs = gen_util.remove_idx(roi_tr_segs, nan_rois, axis=2)

        log_var = np.log(np.var(roi_tr_segs))

        print('Runtype: {}\nMouse: {}\nSess: {}\nLayer: {}\nLine: {}\nFluor: {}\n'
            'ROIs: {}\nGab fr: {}\nGab K: {}\nFrames per seg: {}'
            '\nLogvar: {:.2f}'.format(runtype, mouse_n, sess_n, layer, 
                    sess_dict['line'], fluor, nroi-len(nan_rois), gabs, 
                    sess_dict['gab_k'], roi_tr_segs.shape[1], log_var))
    
        return roi_tr_segs, classes, seg_classes, n_surp
    
    else:
        return None


#############################################
def sample_segs(roi_tr_segs, seg_classes, n_surp):
    """
    sample_segs(roi_tr_segs, seg_classes, n_surp)

    Samples segments to correspond to the ratio of surprise to regular segments.
    
    Required arguments:
        - roi_tr_segs (3D array): array of all ROI trace segments, structured as 
                                  segments x frames x ROIs
        - seg_classes (2D array): array of all segment classes, structured as 
                                  classes x 1
        - n_surp (int)          : number of surprise segments

    Returns:
        - roi_tr_segs (3D array): array of selected ROI trace segments, 
                                  structured as segments x frames x ROIs
        - seg_classes (2D array): array of segment classes, structured as 
                                  classes x 1
    """

    class0_all = np.where(seg_classes == 0)[0]
    class1_all = np.where(seg_classes == 1)[0]
    n_reg = (len(class0_all) + len(class1_all))/2 - n_surp

    class0_idx = np.random.choice(class0_all, n_reg, replace=False)
    class1_idx = np.random.choice(class1_all, n_surp, replace=False)
    
    roi_tr_segs = np.concatenate([roi_tr_segs[class0_idx], 
                                  roi_tr_segs[class1_idx]], axis=0)

    seg_classes = np.concatenate([seg_classes[class0_idx], 
                                  seg_classes[class1_idx]], axis=0)
    return roi_tr_segs, seg_classes


#############################################
def init_comp_model(roi_tr_segs, seg_classes, args, test=True):
    """
    init_comp_model(roi_tr_segs, seg_classes, args)

    Initializes and returns the comparison model and dataloaders.
    
    Required arguments:
        - roi_tr_segs (3D array): array of selected ROI trace segments, 
                                  structured as segments x frames x ROIs
        - seg_classes (2D array): array of segment classes, structured as 
                                  classes x 1
        - args (Argument parser): parser containing analysis parameters:
                batch_size (int): nbr of samples dataloader will load per 
                                  batch
                device (str)    : device name (i.e., 'cuda' or 'cpu')
                lr (float)      : model learning rate
                norm (str)      : normalization type
                shuffle (bool)  : whether analysis is on shuffled data
                train_p (list)  : proportion of dataset to allocate to 
                                  training

    Optional arguments:
        - test (bool): whether a test set should be included

    Returns:
        - model (torch.nn.Module)        : Neural network module with optimizer 
                                           and loss function as attributes
        - dls (list of torch DataLoaders): list of torch DataLoaders for 
                                           each set. If a set is empty, the 
                                           corresponding dls value is None.
    """

    dim = args.norm
    if args.norm == 'roi':
        dim = 'last' # by last dimension

    if not test:
        test_p = 0
    else:
        test_p = None

    dl_info = data_util.create_dls(roi_tr_segs, seg_classes, 
                                   train_p=args.train_p, test_p=test_p,
                                   norm_dim=dim, shuffle=args.shuffle, 
                                   batch_size=args.batch_size)
    
    dls = dl_info[0]

    if args.norm != 'none':
       args.train_means = dl_info[1][0]
       args.train_stds = dl_info[1][1]

    if args.shuffle:
        args.shuff_reidx = dl_info[-1]
 
    args.cl_wei = logreg_util.class_weights(dls[0].dataset.target) # from train targets

    model          = logreg_util.LogReg(roi_tr_segs.shape[2], roi_tr_segs.shape[1]).to(args.device)
    model.opt      = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.loss_fct = logreg_util.weighted_BCE(args.cl_wei)
    
    return model, dls


#############################################
def save_scores(args, scores, saved_eps):
    """
    save_scores(args, scores, saved_eps)

    Saves run information and scores per epoch as a dataframe.
    
    Required arguments:
        - args (Argument parser): parser with analysis parameters as attributes:
                comp (str)           : type of comparison
                dirname (str)        : name of directory in which to save scores
                epochs (int)         : number of epochs
                fluor (str)          : fluorescence trace type
                layer (str)          : layer name
                line (str)           : transgenic line name
                mouse_n (int)        : mouse number
                n_roi (int)          : nuber of ROIs in analysis 
                norm (str)           : type of normalization
                run (int)            : run number
                runtype (str)        : type of run ('prod' or 'pilot')
                sess_n (int)         : session number
                shuffle (bool)       : whether analysis is on shuffled data
                uniqueid (str or int): unique ID for analysis

        - scores (3D array)   : array in which scores are recorded, structured
                                as epochs x nbr sets x nbr score types
        - saved_eps (2D array): array recording which epochs models are 
                                saved for, structured as epochs x 1
    """

    df_labs = gen_util.remove_if(info_dict(None), 'epoch') # to get order
    df_info = info_dict(args, None)
    
    logreg_util.save_scores(df_labs, df_info, args.epochs, saved_eps, scores, 
                            test=True, dirname=args.dirname)


#############################################
def plot_title(mouse_n, sess_n, line, layer, comp):
    """
    plot_title(mouse_n, sess_n, line, layer)

    Creates plot title from session information.
    
    Required arguments:
        - mouse_n (int): mouse number
        - sess_n (int) : session number
        - line (str)   : transgenic line name
        - layer (str)  : layer name
        - comp (str)   : comparison name
    
    Returns:
        - (str): plot title 
    """
    if comp == 'surp':
        comp_str = 'Surp v Reg'
    else:
        comp_str = comp

    return 'Mouse {}, sess {}, {} {}\n{}'.format(mouse_n, sess_n, line, layer, 
                                                comp_str)


#############################################
def plot_tr_traces(args, tr_traces, seg_classes, plot_wei=True):
    """
    plot_tr_traces(args, tr_traces, seg_classes)

    Plots training traces by class, and optionally weights, and saves figures. 

    Required arguments:
        - args (Argument parser): parser with analysis parameters as attributes:
                comp (str)           : type of comparison 
                dirname (str)        : name of directory in which to save 
                                       scores
                error (str)          : error to take, i.e., 'std' (for std 
                                       or quintiles) or 'sem' (for SEM or MAD)
                fig_ext (str)        : extension for saving figure
                fluor (str)          : fluorescence trace type
                gab_fr (int)         : number of the Gabor frame with which 
                                       segment should start (for full segment 
                                       comparisons)
                layer (str)          : layer name
                line (str)           : transgenic line name
                mouse_n (int)        : mouse number
                norm (str)           : type of normalization
                sess_n (int)         : session number
                shuffle (bool)       : whether analysis is on shuffled data 
                stats (str)          : stats to take, i.e., 'mean' or 'median'

        - roi_tr_segs (3D array): array of training ROI trace segments, 
                                  structured as segments x frames x ROIs
        - seg_classes (2D array): array of segment classes, structured as 
                                  classes x 1

    Optional arguments:
        - plot_wei (bool): if True, weights are plotted in a subplot
    """

    # train traces: segments x steps x cells
    tr_traces   = np.asarray(tr_traces)
    seg_classes = np.asarray(seg_classes).squeeze()

    len_s, classes, _, _ = gab_classes(args.comp, args.gab_fr)

    fig, ax_tr, cols = logreg_util.plot_tr_data(tr_traces, seg_classes, classes, 
                                                len_s, plot_wei, args.dirname, 
                                                args.stats, args.error)

    # add plot details
    if args.comp == 'surp':
        gab_par = {'gab_fr': args.gab_fr, 'pre': 0, 'post': len_s}
        [xpos, labels_reg, h_bars, 
        seg_bars] = plot_util.plot_seg_comp(gab_par, 'reg')
        _, labels_surp, _, _ = plot_util.plot_seg_comp(gab_par, 'surp')
        t_heis = [0.85, 0.95]
        labels = [labels_reg, labels_surp]
        plot_util.add_bars(ax_tr, hbars=h_bars, bars=seg_bars)

        for (lab, t_hei, col) in zip(labels, t_heis, cols):
            plot_util.add_labels(ax_tr, lab, xpos, t_hei, col=col)
        
    fluor_str = str_util.fluor_par_str(args.fluor)
    norm_str = str_util.norm_par_str(args.norm)
    shuff_str = str_util.shuff_par_str(args.shuffle, 'labels')
    stat_str = str_util.stat_par_str(args.stats, args.error)
    
    ax_tr.set_ylabel('{}{}'.format(fluor_str, norm_str))

    fig_title = plot_title(args.mouse_n, args.sess_n, args.line, args.layer, 
                           args.comp)
    ax_tr.set_title(u'{}, {} across ROIs{}'.format(fig_title, stat_str, 
                                                   shuff_str))
    ax_tr.legend()

    save_name = os.path.join(args.dirname, 'train_traces{}'.format(args.fig_ext))
    fig.savefig(save_name, bbox_inches='tight')


#############################################
def plot_scores(args, scores, classes, loss_name='loss'):
    """
    plot_scores(args, scores, classes)

    Plots each score type in a figure and saves each figure.
    
    Required arguments:
        - args (Argument parser): parser with analysis parameters as attributes:
                comp (str)           : type of comparison 
                dirname (str)        : name of directory in which to save 
                                       scores
                epochs (int)         : number of epochs
                fluor (str)          : fluorescence trace type
                layer (str)          : layer name
                line (str)           : transgenic line name
                mouse_n (int)        : mouse number
                norm (str)           : type of normalization
                sess_n (int)         : session number
                shuffle (bool)       : whether analysis is on shuffled data

        - scores (3D array): array in which scores are recorded, structured
                             as epochs x nbr sets x nbr score types
        - classes (list)   : list of class names
    
    Optional arguments:
        - loss_name (str): name of type of loss
                           default: 'loss'
    """

    fluor_str = str_util.fluor_par_str(args.fluor, 'print')
    norm_str = str_util.norm_par_str(args.norm, 'print')
    shuff_str = str_util.shuff_par_str(args.shuffle, 'labels')
    fig_title = plot_title(args.mouse_n, args.sess_n, args.line, args.layer,
                           args.comp)


    gen_title = '{}, {}{}{}'.format(fig_title, fluor_str, norm_str, shuff_str)
    logreg_util.plot_scores(args.epochs, scores, classes, args.dirname, 
                            loss_name=loss_name, test=True, gen_title=gen_title)


#############################################
def single_run(roi_tr_segs, seg_classes, classes, n_surp, args, run):
    """
    single_run(roi_tr_segs, seg_classes, classes, n_surp, args, run)

    Does a single run of a logistic regression on the specified comparison
    and session data.
    
    Required arguments:
        - roi_tr_segs (3D array): array of all ROI trace segments, structured as 
                                  segments x frames x ROIs
        - seg_classes (2D array): array of all segment classes, structured as 
                                  classes x 1
        - classes (list)        : list of class names
        - n_surp (int)          : number of surprise segments
        - args (Argument parser): parser with analysis parameters as attributes:
                batch_size (int)     : nbr of samples dataloader will load per 
                                       batch
                classes (lists)      : list of class names
                comp (str)           : type of comparison
                compdir (str)        : name of directory to save analysis in
                device (str)         : device name (i.e., 'cuda' or 'cpu')
                ep_freq (int)        : frequency at which to print loss to 
                                       console
                epochs (int)         : number of epochs
                error (str)          : error to take, i.e., 'std' (for std 
                                       or quintiles) or 'sem' (for SEM or MAD)
                fig_ext (str)        : extension for saving figure
                fluor (str)          : fluorescence trace type
                gab_fr (int)         : number of the Gabor frame with which 
                                       segment should start (for full segment 
                                       comparisons)
                keep (str)           : models to save ('best' or 'all')
                layer (str)          : layer name
                line (str)           : transgenic line name
                lr (float)           : model learning rate
                mouse_n (int)        : mouse number
                n_roi (int)          : nuber of ROIs in analysis 
                norm (str)           : type of normalization
                parallel (bool)      : if True, runs are done in parallel
                plt_bkend (str)      : pyplot backend to use
                output (str)         : general directory in which to save output
                reseed (bool)        : whether to reseed each run
                runtype (str)        : type of run ('prod' or 'pilot')
                seed (int)           : seed to seed random processes with
                sess_n (int)         : session number
                shuffle (bool)       : whether analysis is on shuffled data
                stats (str)          : stats to take, i.e., 'mean' or 'median'
                train_p (list)       : proportion of dataset to allocate to 
                                       training
                uniqueid (str or int): unique ID for analysis 
        
        - run (int)             : run number
    """
    
    if args.parallel and args.plt_bkend is not None:
        plt.switch_backend(args.plt_bkend) # needs to be repeated within joblib

    rundir = get_rundir(run, args.uniqueid)

    if args.reseed:
        # reseed every run 
        args.seed = gen_util.seed_all(None, args.device)
    else:
        # all run with same seed
        args.seed = gen_util.seed_all(args.seed, args.device)
    
    args.run = run
    args.dirname = file_util.create_dir([args.output, args.compdir, rundir])
    save_hyperpar(args)

    # select a random subsample
    if args.comp in ['AvB', 'AvC', 'BvC']:
        roi_tr_segs, seg_classes = sample_segs(roi_tr_segs, seg_classes, n_surp)

    mod, dls = init_comp_model(roi_tr_segs, seg_classes, args, test=True)

    norm_str = str_util.norm_par_str(args.norm, 'print')
    shuff_str = str_util.shuff_par_str(args.shuffle, 'labels')
    print('\nRun: {}{}{}'.format(args.run, norm_str, shuff_str))

    # scores: ep x set (train, val, test) x sc (loss, acc, acc0, acc1)
    info = info_dict(args)
    scores, saved_eps = logreg_util.fit_model(info, args.epochs, mod, dls, 
                                              args.device, args.dirname, 
                                              ep_freq=args.ep_freq, 
                                              keep=args.keep)

    print('Run {}: training done.\n'.format(run))

    # plot traces and scores
    plot_tr_traces(args, dls[0].dataset.data.numpy(), dls[0].dataset.target.numpy())
    plot_scores(args, scores, classes, mod.loss_fct.name)
    
    # save scores in dataframe
    save_scores(args, scores, saved_eps)

    plt.close('all')


#############################################
def run_regr(args):
    """
    run_regr(args)

    Does runs of a logistic regressions on the specified comparison and range
    of sessions.
    
    Required arguments:
        - args (Argument parser): parser with analysis parameters as attributes:
                batch_size (int)     : nbr of samples dataloader will load per 
                                       batch
                comp (str)           : type of comparison
                datadir (str)        : data directory
                device (str)         : device name (i.e., 'cuda' or 'cpu')
                ep_freq (int)        : frequency at which to print loss to 
                                       console
                epochs (int)         : number of epochs
                error (str)          : error to take, i.e., 'std' (for std 
                                       or quintiles) or 'sem' (for SEM or MAD)
                fig_ext (str)        : extension for saving figure
                fluor (str)          : fluorescence trace type
                gab_fr (int)         : number of the Gabor frame with which 
                                       segment should start (for full segment 
                                       comparisons)
                keep (str)           : models to save ('best' or 'all')
                layer (str)          : layer name
                line (str)           : transgenic line name
                lr (float)           : model learning rate
                mouse_n (int)        : mouse number
                norm (str)           : type of normalization
                parallel (bool)      : if True, runs are done in parallel
                plt_bkend (str)      : pyplot backend to use
                output (str)         : general directory in which to save output
                reseed (bool)        : whether to reseed each run
                runtype (str)        : type of run ('prod' or 'pilot')
                seed (int)           : seed to seed random processes with
                sess_n (int)         : session number
                stats (str)          : stats to take, i.e., 'mean' or 'median'
                train_p (list)       : proportion of dataset to allocate to 
                                       training
                uniqueid (str or int): unique ID for analysis
    """

    if args.datadir is None:
        # previously: '/media/colleen/LaCie/CredAssign/pilot_data'
        args.datadir = '../data/AIBS/{}'.format(args.runtype) 

    if args.uniqueid == 'datetime':
        args.uniqueid = str_util.create_time_str()

    if args.seed in [None, 'None']:
        args.reseed = True
    else:
        args.reseed = False

    mouse_df = file_util.load_file('mouse_df_{}.csv'.format(args.runtype), 
                                    file_type='csv')
    
    atts       = ['mouseid', 'pass_fail', 'all_files']
    cri        = [args.mouse_n, 'P', 1]
    curr_lines = gen_util.get_df_vals(mouse_df, atts, cri)
    sesses     = gen_util.get_df_vals(curr_lines, label='overall_sess_n')
    
    if args.sess_n != 'all':
        sesses = sorted(set(sesses) + set(gen_util.list_if_not(args.sess_n)))

    for sess_n in sesses:
        args.sess_n = sess_n
        curr_line   = gen_util.get_df_vals(curr_lines, 'overall_sess_n', args.sess_n)
        args.layer  = curr_line['layer'].item()
        args.line   = curr_line['line'].item()

        sess_data = get_sess_data(args.datadir, args.mouse_n, args.sess_n, 
                                  args.layer, comp=args.comp, 
                                  gab_fr=args.gab_fr, fluor=args.fluor,
                                  runtype=args.runtype)
        
        if sess_data is None:
            continue
        else:
            [roi_tr_segs, classes, seg_classes, n_surp] = sess_data

        args.n_roi = roi_tr_segs.shape[2]
        args.classes = classes

        for runs, shuffle in zip([args.n_reg, args.n_shuff], [False, True]):

            args.shuffle = shuffle
            args.compdir = get_compdir(args.mouse_n, args.sess_n, args.layer, 
                                        args.fluor, args.norm, args.comp, 
                                        args.shuffle)

            if args.parallel:
                num_cores = multiprocessing.cpu_count()
                Parallel(n_jobs=num_cores)(delayed(single_run)
                        (roi_tr_segs, seg_classes, classes, n_surp, args, run) 
                         for run in range(runs))
            else:
                for run in range(runs):
                    single_run(roi_tr_segs, seg_classes, classes, n_surp, args, run)


#############################################
def collate_scores(direc, all_labels):
    """
    collate_scores(direc, all_labels)

    Collects the analysis information and scores from the last epoch recorded 
    for a run and returns in dataframe.
    
    Required arguments:
        - direc (str)      : path to the specific comparison run folder
        - all_labels (list): ordered list of columns to save to dataframe
    
    Return:
        - scores (pd DataFrame): Dataframe containing run analysis information
                                 and scores from the last epoch recorded.
    """

    print(direc)
 
    scores = pd.DataFrame()

    ep_info, hyperpars = logreg_util.get_scores(direc)

    if ep_info is None:
        comp_dict = get_compdir_dict(direc)
        comp_dict['norm'] = hyperpars['norm']
        comp_dict['runtype'] = hyperpars['runtype']
        comp_dict['line'] = hyperpars['line']
        comp_dict['n_roi'] = hyperpars['n_roi']
        for col in all_labels: # ensures correct order
            if col in comp_dict.keys():
                scores.loc[0, col] = comp_dict[col]
    else:
        for col in all_labels:
            scores.loc[0, col] = ep_info[col].item()

    return scores


#############################################
def run_collate(args):
    """
    run_collate(args)

    Collects the analysis information and scores from the last epochs recorded 
    for all runs for a comparison type, and saves to a dataframe.
    
    Required arguments:
        - args (Argument parser): parser with analysis parameters as attributes:
                comp (str)           : type of comparison
                parallel (bool)      : if True, run information is collected 
                                       in parallel
                output (str)         : general directory in which run information
                                       is located output
    """

    comp_dirs = file_util.get_files(args.output, 'subdirs', args.comp)
    run_dirs = [run_dir for comp_dir in comp_dirs 
                for run_dir in file_util.get_files(comp_dir, 'subdirs')]
    all_labels = info_dict(None) + ['saved'] + logreg_util.get_sc_labs(test=True)

    if args.parallel:
        num_cores = multiprocessing.cpu_count()
        scores_list = Parallel(n_jobs=num_cores)(delayed(collate_scores)
                        (run_dir, all_labels) for run_dir in run_dirs)
        all_scores = pd.concat(scores_list)
        all_scores = all_scores[all_labels] # reorder
    else:
        all_scores = pd.DataFrame(columns=all_labels)
        for run_dir in run_dirs:
            scores = collate_scores(run_dir, all_labels)
            all_scores = all_scores.append(scores)

    # sort df by mouse, session, layer, line, fluor, norm, shuffled, comp, 
    # uniqueid, run, runtype
    sorter = info_dict(None)[0:11]
    all_scores = all_scores.sort_values(by=sorter).reset_index(drop=True)

    file_util.save_info(all_scores, '{}_all_scores_df'.format(args.comp), 
                        args.output, 'csv')


#############################################
def calc_stats(scores_summ, curr_lines, curr_idx, CI=0.95):
    """
    calc_stats(scores_summ, curr_lines, curr_idx)

    Calculates statistics on scores from runs with specific analysis criteria
    and records them in the summary scores dataframe.  
    
    Required arguments:
        - scores_summ (pd DataFrame): DataFrame containing scores summary
        - curr_lines (pd DataFrame) : DataFrame lines corresponding to specific
                                      analysis criteria
        - curr_idx (int)            : Current row in the scores summary 
                                      DataFrame 
    
    Optional arguments:
        - CI (float): Confidence interval around which to collect percentile 
                      values
                      default: 0.95

    Returns:
        - scores_summ (pd DataFrame): Updated DataFrame containing scores 
                                      summary
    """

    # score labels to perform statistics on
    sc_labs = ['epoch'] + logreg_util.get_sc_labs(test=True)

    # percentiles to record
    ps, p_names = math_util.get_percentiles(CI)

    for sc_lab in sc_labs:
        if sc_lab in curr_lines.keys():
            cols = []
            vals = []
            for stat in ['mean', 'median']:
                cols.extend([stat])
                vals.extend([math_util.mean_med(curr_lines[sc_lab], stats=stat, 
                                                nanpol='omit')])
            for error in ['std', 'sem']:
                cols.extend([error])
                vals.extend([math_util.error_stat(curr_lines[sc_lab], stats='mean', 
                                                  error=error, nanpol='omit')])
            # get 25th and 75th quartiles
            cols.extend(['q25', 'q75'])
            vals.extend(math_util.error_stat(curr_lines[sc_lab], stats='median', 
                                                error='std', nanpol='omit'))                                            
            # get other percentiles (for CI)
            cols.extend(p_names)
            vals.extend(math_util.error_stat(curr_lines[sc_lab], stats='median', 
                                             error='std', nanpol='omit', qu=ps))
            
            # get MAD
            cols.extend(['mad'])
            vals.extend([math_util.error_stat(curr_lines[sc_lab], stats='median', 
                                              error='sem', nanpol='omit')])

            # plug in values
            cols = ['{}_{}'.format(sc_lab, name) for name in cols]
            gen_util.set_df_vals(scores_summ, curr_idx, cols, vals)
    
    return scores_summ


#############################################
def run_analysis(args):  
    """
    run_analysis(args)

    Calculates statistics on scores from runs for each specific analysis 
    criteria and saves them in the summary scores dataframe.  
    
    Required arguments:
        - args (Argument parser): parser with analysis parameters as attributes:
                CI (float)  : confidence interval around which to collect 
                              percentile values
                comp (str)  : type of comparison
                output (str): general directory in which run information
                              is located output
    """

    all_scores_df = file_util.load_file('{}_all_scores_df.csv'.format(args.comp),
                                        args.output, 'csv')
    scores_summ = pd.DataFrame()

    # common labels
    comm_labs = gen_util.remove_if(info_dict(None), ['uniqueid', 'run', 'epoch'])

    # get all unique comb of labels
    df_unique = all_scores_df[comm_labs].drop_duplicates()
    for _, df_row in df_unique.iterrows():
        vals = [df_row[x] for x in comm_labs]
        curr_lines = gen_util.get_df_vals(all_scores_df, comm_labs, vals)
        # assign values to current line in summary df
        curr_idx = len(scores_summ)
        gen_util.set_df_vals(scores_summ, curr_idx, comm_labs, vals)
        # calculate n_runs (without nans and with)
        scores_summ.loc[curr_idx, 'runs_total'] = len(curr_lines)
        scores_summ.loc[curr_idx, 'runs_nan'] = curr_lines['epoch'].isna().sum()
        # calculate stats
        scores_summ = calc_stats(scores_summ, curr_lines, curr_idx, args.CI)

    file_util.save_info(scores_summ, '{}_score_stats_df'.format(args.comp), 
                        args.output, 'csv')


#############################################    
def init_res_fig(args, n_subplots):
    """
    init_res_fig(args, n_subplots)

    Initializes a figure in which to plot summary results.

    Required arguments:
        - args (Argument parser): parser with analysis parameters as attributes:
                ncols (int)        : number of columns in the figure
                sharey (bool)      : if True, y axis lims are shared across 
                                     subplots
                subplot_hei (float): height of each subplot (inches)
                subplot_wid (float): width of each subplot (inches)

        - n_subplots (int)      : number of subplots
    
    Returns:
        - fig (plt Fig): figure
        - ax (plt Axis): axis
    """

    fig_par = {'ncols'      : args.ncols,
               'sharey'     : not(args.no_sharey),
               'subplot_wid': args.subplot_wid,
               'subplot_hei': args.subplot_hei
                }
    
    fig, ax = plot_util.init_fig(n_subplots, fig_par)

    return fig, ax


#############################################
def rois_x_label(sess_ns, arr):
    """
    rois_x_label(sess_ns, arr)

    Creates x axis labels with the number of ROIs per mouse for each session.
    
    For each session, formatted as: Session # (n/n rois)
    
    Required arguments:
        - sess_ns (list): list of session numbers
        - arr (3D array): array of number of ROIs, structured as 
                          mouse x session x shuffle

    Returns:
        - x_label (list): list of x_labels for each session.
    """

    arr = np.nan_to_num(arr) # convert NaNs to 0s
    
    # check that shuff and non shuff are the same
    if not (arr[:, :, 0] == arr[:, :, 1]).all():
        raise ValueError('Shuffle and non shuffle n_rois are not the same.')

    x_label = []
    for s, sess_n in enumerate(sess_ns):
        for m in range(arr.shape[0]):
            if m == 0:
                n_rois_str = '{}'.format(int(arr[m, s, 0]))
            if m > 0:
                n_rois_str = '{}/{}'.format(n_rois_str, int(arr[m, s, 0]))
        x_label.append('Session {}\n({} rois)'.format(int(sess_n), n_rois_str))
    return x_label


#############################################
def mouse_runs_leg(arr, mouse_n=None, shuffle=False, CI=0.95):
    """
    mouse_runs_leg(arr)

    Creates legend labels for a mouse or shuffle set.  
    
    For each mouse or shuffle set, formatted as: 
    Mouse # (n/n runs) or Shuff (n/n runs)

    Required arguments:
        - arr (3D array): array of number of ROIs, structured as 
                          mouse (or mice to sum) x session x shuffle

    Optional arguments:
        - mouse_n (int) : mouse number (only needed if shuffle is False)
                          default: None
        - shuffle (bool): if True, shuffle legend is created. Otherwise, 
                          mouse legend is created.
                          default: False
        - CI (float)    : CI for shuffled data
                          default: 0.95 
    Returns:
        - leg (str): legend for the mouse or shuffle set
    """

    # create legend: Mouse # (n/n runs) or Shuff (n/n runs)
    if len(arr.shape) == 1:
        arr = arr[np.newaxis, :]
    arr = np.nan_to_num(arr) # convert NaNs to 0s

    for s in range(arr.shape[1]):
        if s == 0:
            n_runs_str = '{}'.format(int(np.sum(arr[:, s])))
        if s > 0:
            n_runs_str = '{}/{}'.format(n_runs_str, int(np.sum(arr[:, s])))
    
    if shuffle:
        if CI is not None:
            CI_pr = CI*100
            if CI_pr%1 == 0:
                CI_pr = int(CI_pr)
            leg = 'shuffled ({}% CI)\n({} runs)'.format(CI_pr, n_runs_str)
        else:
            leg = 'shuffled\n({} runs)'.format(n_runs_str)

    else:
        if mouse_n is None:
            raise IOError('If \'shuffle\' is False, Must specify \'mouse_n\'.')
        
        leg = 'mouse {}\n({} runs)'.format(int(mouse_n), n_runs_str)
    
    return leg


#############################################
def plot_CI(ax, x_label, arr, sess_ns, CI=0.95):
    """
    plot_CI(ax, x_label, arr, sess_ns)

    Plots confidence intervals for each session.

    Required arguments:
        - ax (plt Axis subplot): subplot
        - x_label (list)       : list of x_labels for each session
        - arr (3D array)       : array of number of ROIs, structured as 
                                 mouse (or mice to sum) x session x shuffle
        - sess_ns (list)       : list of session numbers
    
        Optional arguments:
        - CI (float)           : CI for shuffled data
                                 default: 0.95 
    """

    # shuffle (combine across mice)
    med = np.nanmedian(arr[:, :, 0], axis=0)
    p_lo = np.nanmedian(arr[:, :, 1], axis=0)
    p_hi = np.nanmedian(arr[:, :, 2], axis=0)

    leg = mouse_runs_leg(arr[:,:,4], shuffle=True, CI=CI)

    # plot CI
    ax.bar(x_label, height=p_hi-p_lo, bottom=p_lo, color='lightgray', width=0.2, 
           label=leg)
    
    # plot median (with some thickness based on ylim)
    y_lim = ax.get_ylim()
    med_th = 0.005*(y_lim[1]-y_lim[0])
    ax.bar(x_label, height=med_th, bottom=med-med_th/2.0, color='grey', 
           width=0.2)

#############################################
def summ_subplot(ax, arr, data_title, mouse_ns, sess_ns, line, layer, fluor, 
                 norm, comp='surp', runtype='prod', stat='mean', CI=0.95):
    """
    summ_subplot(ax, arr, datatype, mouse_ns, sess_ns, line, layer, fluor, norm,
                 comp, runtype)

    Plots summary data in the specific subplot for a line and layer.

    Required arguments:
        - ax (plt Axis subplot): subplot
        - arr (3D array)       : array of session information, structured as 
                                 mice x sessions x shuffle x vals, where vals
                                 are: mean/med, sem/low_perc, sem/hi_perc, 
                                      n_rois, n_runs
        - data_title (str)     : name of type of data plotted, 
                                 i.e. for epochs or test accuracy
        - mouse_ns (int)       : mouse numbers
        - sess_ns (int)        : session numbers
        - line (str)           : transgenic line name
        - layer (str)          : layer name
        - fluor (str)          : fluorescence trace type
        - norm (str)           : type of normalization
    
    Optional arguments:
        - comp (str)           : type of comparison
                                 default: 'surp'
        - runtype (str)        : type of run ('prod' or 'pilot')
                                 default: 'prod'
        - stat (str)           : stats to take for non shuffled data, 
                                 i.e., 'mean' or 'median' 
                                 default: 'mean'
        - CI (float)           : CI for shuffled data
                                 default: 0.95 
    """

    col=['steelblue', 'coral', 'forestgreen']
    
    x_label = rois_x_label(sess_ns, arr[:,:,:,3])

    plot_CI(ax, x_label, arr[:,:,1], sess_ns, CI)

    # plot non shuffle data
    for m, mouse_n in enumerate(mouse_ns):
        leg = mouse_runs_leg(arr[m,:,0,4], mouse_n, False)
        ax.errorbar(x_label, arr[m,:,0,0], yerr=arr[m,:,0,1], fmt='-o', 
                    capsize=6, capthick=2, color=col[m], label=leg, alpha=0.5)     

    # add a mean line
    for i in range(len(x_label)):
        if not np.isnan(arr[:,i,0,0]).all():
            med = math_util.mean_med(arr[:,i,0,0], axis=0, stats=stat, 
                                     nanpol='omit')
            y_lim = ax.get_ylim()
            med_th = 0.005*(y_lim[1]-y_lim[0])

            ax.bar(x_label[i], height=med_th, bottom=med-med_th/2.0, color='black', 
                width=0.3)

    if line == 'L23':
        line = 'L2/3'
    if comp == 'surp':
        comp = 'Surp'
    norm_str = str_util.norm_par_str(norm, type_str='file')[1:]

    title = ('{} - {} for log regr on'
             '\n{} {} {} {} data ({})').format(comp, data_title, norm_str, fluor, 
                                               line, layer, runtype)
    
    ax.set_title(title)
    ax.legend()

    if 'acc' in data_title:
        ax.set_ylabel('Accuracy (%)')
    elif 'epoch' in data_title:
        ax.set_ylabel('Nbr epochs')


#############################################    
def plot_data_summ(args, summ_scores, data, title, stats, shuff_stats):
    """
    plot_data_summ(args, summ_scores, data, title, stats, shuff_stats)

    Plots summary data for a specific comparison, for each line and layer and 
    saves figure.

    Required arguments:
         - args (Argument parser): parser with analysis parameters as attributes:
                comp (str)         : type of comparison
                fig_ext (str)      : extension for saving figure
                fluor (str)        : fluorescence trace type
                ncols (int)        : number of columns in the figure
                norm (str)         : type of normalization
                output (str)       : general directory in which summary 
                                     dataframe is saved
                runtype (str)      : type of run ('prod' or 'pilot')
                sharey (bool)      : if True, y axis lims are shared across 
                                     subplots
                subplot_hei (float): height of each subplot (inches)
                subplot_wid (float): width of each subplot (inches)

        - summ_scores (pd DataFrame): DataFrame containing scores summary
                                      for specific comparison
        - data (str)                : label of type of data to plot,
                                      e.g., 'epochs' or 'test_acc' 
        - title (str)               : name of type of data plotted, 
                                      e.g. for epochs or test accuracy
        - stats (list)              : list of stats to use for non shuffled 
                                      data, e.g., ['mean', 'sem', 'sem']
        - shuff_stats (list)        : list of stats to use for shuffled 
                                      data, e.g., ['median', 'p2p5', 'p97p5']
    """
    
    celltypes = [[x, y] for x in ['L23', 'L5'] for y in ['soma', 'dend']]

    fig, ax = init_res_fig(args, len(celltypes))

    for i, [line, layer] in enumerate(celltypes):
        sub_ax = plot_util.get_subax(ax, i)
        # get the right rows in dataframe
        cols       = ['layer', 'fluor', 'norm', 'runtype']
        cri        = [layer, args.fluor, args.norm, args.runtype]
        curr_lines = gen_util.get_df_vals(summ_scores.loc[summ_scores['line'].str.contains(line)], cols, cri)
        if len(curr_lines) == 0:
            cri_str = ['{}: {}'.format(col, crit) for col, crit in zip(cols, cri)]
            print('No data found for line: {}, {}'.format(line, ', '.join(cri_str)))
            continue
        sess_ns    = gen_util.get_df_vals(curr_lines, label='sess_n', dtype=int)
        mouse_ns   = gen_util.get_df_vals(curr_lines, label='mouse_n', dtype=int)
        # mouse x sess x shuffle x vals (mean/med, sem/2.5p, sem/97.5p, n_rois, n_runs)
        data_arr = np.empty((len(mouse_ns), len(sess_ns), 2, 5)) * np.nan
        
        for s, sess_n in enumerate(sess_ns):
            sess_mice = gen_util.get_df_vals(curr_lines, 'sess_n', sess_n, 'mouse_n', dtype=int)
            for m, mouse_n in enumerate(mouse_ns):
                if mouse_n in sess_mice:
                    for sh, stat_types in enumerate([stats, shuff_stats]):
                        curr_line = gen_util.get_df_vals(curr_lines, ['sess_n', 'mouse_n', 'shuffled'], [sess_n, mouse_n, sh])
                        for st, stat in enumerate(stat_types):
                            data_arr[m, s, sh, st] = curr_line['{}_{}'.format(data, stat)]
                        data_arr[m, s, sh, 3] = curr_line['n_roi']
                        data_arr[m, s, sh, 4] = curr_line['runs_total'] - curr_line['runs_nan']
        
        summ_subplot(sub_ax, data_arr, title, mouse_ns, sess_ns, line, layer, 
                     args.fluor, args.norm, args.comp, args.runtype, stats[0],
                     shuff_stats)

    norm_str = str_util.norm_par_str(args.norm, type_str='file')
    save_dir = os.path.join(args.output, 'figures_{}'.format(args.fluor))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    save_name = os.path.join(save_dir, '{}_{}{}{}'.format(data, args.comp, 
                             norm_str, args.fig_ext))
    fig.savefig(save_name, bbox_inches='tight')


#############################################    
def run_plot(args):
    """
    run_plot(args)

    Plots summary data for a specific comparison, for each datatype in a 
    separate figure and saves figures. 

    Required arguments:
         - args (Argument parser): parser with analysis parameters as attributes:
                CI (float)         : CI for shuffled data
                comp (str)         : type of comparison
                fig_ext (str)      : extension for saving figure
                fluor (str)        : fluorescence trace type
                ncols (int)        : number of columns in the figure
                norm (str)         : type of normalization
                output (str)       : general directory in which summary 
                                     dataframe is saved
                runtype (str)      : type of run ('prod' or 'pilot')
                sharey (bool)      : if True, y axis lims are shared across 
                                     subplots
                subplot_hei (float): height of each subplot (inches)
                subplot_wid (float): width of each subplot (inches)
    """

    summ_scores_file = os.path.join(args.output, '{}_score_stats_df.csv'.format(args.comp))
    
    if os.path.exists(summ_scores_file):
        summ_scores = file_util.load_file(summ_scores_file, file_type='csv')
    else:
        print('{} not found.'.format(summ_scores_file))
        return

    data_types  = ['epoch', 'test_acc']
    data_titles = ['epoch nbr', 'test accuracy']

    stats = ['mean', 'sem', 'sem']
    shuff_stats = ['median'] + math_util.get_percentiles(args.CI)[1]

    for data, title in zip(data_types, data_titles):
        plot_data_summ(args, summ_scores, data, title, stats, shuff_stats)

    plt.close('all')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='logreg_models', help='where to store output')
    parser.add_argument('--task', default='run_regr', help='run_regr or collate')
    parser.add_argument('--comp', default='surp', help='surp, AvB, AvC, BvC, DvE, all')
    parser.add_argument('--fig_ext', default='.svg')
    parser.add_argument('--plt_bkend', default=None, 
                        help='switch matplotlib backend when running on server')

        # run_regr general
    parser.add_argument('--runtype', default='prod', help='prod or pilot')
    parser.add_argument('--mouse_n', default=1, type=int)
    parser.add_argument('--sess_n', default='all')
 
    parser.add_argument('--parallel', action='store_true', 
                        help='do runs in parallel.')
    parser.add_argument('--cuda', action='store_true', 
                        help='run on cuda.')
    parser.add_argument('--datadir', default=None, 
                        help=('data directory (if None, uses a directory '
                              'defined below'))
    parser.add_argument('--keep', default='best', 
                        help=('record only best or all models.'))
        
        # run_regr hyperparameters
    parser.add_argument('--n_reg', default=50, type=int, help='n regular runs')
    parser.add_argument('--n_shuff', default=50, type=int, help='n shuffled runs')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--train_p', default=0.75, type=float, 
                        help='proportion of dataset used in training set')
    parser.add_argument('--stats', default='mean', help='mean or median')
    parser.add_argument('--error', default='sem', help='std or sem')
    parser.add_argument('--gab_fr', default=0, type=int, 
                        help='starting gab frame in comp is surp')
    parser.add_argument('--norm', default='none', 
                        help='normalize data: none, all or roi (by roi)')
    parser.add_argument('--fluor', default='dff', 
                        help='raw or dff')
    parser.add_argument('--ep_freq', default=50, type=int,  
                        help='epoch frequency at which to print loss')
    parser.add_argument('--seed', default=None, type=int,  
                        help='manual seed')
    parser.add_argument('--uniqueid', default='datetime', 
                        help=('passed string, \'datetime\' for date and time '
                              'or None for no uniqueid'))

        # analysis parameters
    parser.add_argument('--CI', default=0.95, type=float, help='shuffled CI')

        # plot parameters
    parser.add_argument('--ncols', default=2, type=int, help='nbr of cols per figure')
    parser.add_argument('--no_sharey', action='store_true', help='subplots do not share y axis')
    parser.add_argument('--subplot_wid', default=7.5, type=float, help='figure width')
    parser.add_argument('--subplot_hei', default=7.5, type=float, help='figure height')
        
        # NOTE: CI, norm, fluor, runtype also used for plots

    args = parser.parse_args()

    args.device = gen_util.get_device(args.cuda)

    if args.plt_bkend not in [None, 'None']:
        plt.switch_backend(args.plt_bkend) 

    if args.comp == 'all':
        comps = ['surp', 'AvB', 'AvC', 'BvC', 'DvE']
    else:
        comps = gen_util.list_if_not(args.comp)

    for comp in comps:
        args.comp = comp
        print('Task: {}'.format(args.task))
        print('Comparison: {}\n'.format(args.comp))

        if args.task == 'run_regr':
            run_regr(args)

        elif args.task == 'collate':
            run_collate(args)

        # analyses accuracy
        elif args.task == 'analyse':
            run_analysis(args)

        elif args.task == 'plot':
            run_plot(args)




    ############## QUICK ADD ##################
    if args.task == 'mags':
        sess_labels = ['Session 1', 'Session 2']
        mag_labels = ['Regular', 'Surprise']
        x_labels = ['{}\n{}'.format(m, s) for s in sess_labels for m in mag_labels]

        direc = os.path.join(args.output, 'figures', 'summ_mags')
        summ_mag_dict = file_util.load_file('summ_mag_dict.pkl', direc, 'pickle')

        celltypes = [[x, y] for x in ['L23-Cux2', 'L5-Rbp4'] for y in ['soma', 'dend']]

        fig, ax = init_res_fig(args, len(celltypes))
        fig_norm, ax_norm = init_res_fig(args, len(celltypes))

        for i, [line, layer] in enumerate(celltypes):
            for name, axis in zip(['reg', 'norm'], [ax, ax_norm]):
                sub_ax = plot_util.get_subax(axis, i)

                subdict = '{}_{}'.format(line, layer)
                sub_ax.set_title(subdict)
                mouse_leg = summ_mag_dict[subdict]['mouse_leg']
                data = summ_mag_dict[subdict]['{}_data'.format(name)]

                data = data.reshape([data.shape[0], -1]).T

                sub_ax.plot(x_labels, data, marker='o', alpha=0.5)
                sub_ax.legend(mouse_leg)

        for i, [line, layer] in enumerate(celltypes):
            for name, axis in zip(['reg', 'norm'], [ax, ax_norm]):
                sub_ax = plot_util.get_subax(axis, i)
                subdict = '{}_{}'.format(line, layer)
                data = summ_mag_dict[subdict]['{}_data'.format(name)]
                data = data.reshape([data.shape[0], -1]).T

                if not np.isnan(data).all():
                    med = math_util.mean_med(data, axis=1, stats=args.stats, 
                                             nanpol='omit')

                    y_lim = sub_ax.get_ylim()
                    med_th = 0.005*(y_lim[1]-y_lim[0])

                    sub_ax.bar(x_labels, height=med_th, bottom=med-med_th/2.0, 
                            color='black', width=0.3)
            
        fig.savefig(os.path.join(direc, 'reg_mags{}'.format(args.fig_ext)), 
                    bbox_inches='tight')
        fig_norm.savefig(os.path.join(direc, 'norm_mags{}'.format(args.fig_ext)), 
                         bbox_inches='tight')

