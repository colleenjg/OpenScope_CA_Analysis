"""
run_logreg.py

This script runs and analyses logistic regressions predicting stimulus 
information based on ROI activity for data generated by the AIBS experiments 
for the Credit Assignment Project

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import os
import copy
import argparse

from analysis import logreg
from util import gen_util
from sess_util import sess_gen_util, sess_ntuple_util, sess_str_util
from plot_fcts import plot_from_dicts_tool as plot_dicts


DEFAULT_DATADIR = os.path.join('..', 'data', 'AIBS')
DEFAULT_MOUSE_DF_PATH = 'mouse_df.csv'
DEFAULT_FONTDIR = os.path.join('..', 'tools', 'fonts')


#############################################
def get_comps(stimtype='gabors', q1v4=False, regvsurp=False):
    """
    get_comps()

    Returns comparisons that fit the criteria.

    Optional args:
        - stimtype (str) : stimtype
                           default: 'gabors'
        - q1v4 (bool)    : if True, analysis is trained on first and tested on 
                           last quintiles
                           default: False
        - regvsurp (bool): if True, analysis is trained on regular and tested 
                           on regular sequences
                           default: False
    
    Returns:
        - comps (list): list of comparisons that fit the criteria
    """

    if stimtype == 'gabors':
        if regvsurp:
            raise ValueError('regvsurp can only be used with bricks.')
        comps = ['surp', 'AvB', 'AvC', 'BvC', 'DvE', 'Aori', 'Bori', 'Cori', 
            'Dori', 'Eori']
    elif stimtype == 'bricks':
        comps = ['surp', 'dir_all', 'dir_surp', 'dir_reg', 'half_right', 
            'half_left', 'half_diff'] 
        if regvsurp:
            comps = gen_util.remove_if(
                comps, ['surp', 'dir_surp', 'dir_all', 'half_right', 
                'half_left', 'half_diff'])
        if q1v4:
            comps = gen_util.remove_if(
                comps, ['half_left', 'half_right', 'half_diff'])
    else:
        gen_util.accepted_values_error(
            'stimtype', stimtype, ['gabors', 'bricks'])

    return comps


#############################################
def check_args(comp='surp', stimtype='gabors', q1v4=False, regvsurp=False):
    """
    check_args()

    Verifies whether the comparison type is compatible with the stimulus type, 
    q1v4 and regvsurp.

    Optional args:
        - comp (str)     : comparison type
                           default: 'surp'
        - stimtype (str) : stimtype
                           default: 'gabors'
        - q1v4 (bool)    : if True, analysis is trained on first and tested on 
                           last quintiles
                           default: False
        - regvsurp (bool): if True, analysis is trained on regular and tested 
                           on regular sequences
                           default: False
    """


    poss_comps = get_comps(stimtype, q1v4, regvsurp)

    if q1v4 and regvsurp:
        raise ValueError('q1v4 and regvsurp cannot both be set to True.')

    if comp not in poss_comps:
        comps_str = ', '.join(poss_comps)
        raise ValueError(f'With stimtype={stimtype}, q1v4={q1v4}, '
            f'regvsurp={regvsurp}, can only use the following '
            f'comps: {comps_str}')
    return


#############################################
def set_ctrl(ctrl=False, comp='surp'):
    """
    set_ctrl()

    Sets the control value (only modifies if it is True).

    Optional args:
        - ctrl (bool): whether the run is a control
                       default: False
        - comp (str) : comparison type
                       default: 'surp'
    
    Returns:
        - ctrl (bool): modified control value
    """    

    if comp in ['surp', 'DvE', 'Eori', 'dir_surp']:
        ctrl = False
    
    if comp == 'all':
        raise ValueError('Should not be used if comp is \'all\'.')
    
    return ctrl


#############################################
def format_output(output, runtype='prod', q1v4=False, bal=False, 
                  regvsurp=False):
    """
    format_output(output)

    Returns output modified based on the arguments.

    Required args:
        - output (str): base output path

    Optional args:
        - runtype (str)  : runtype
                           default: 'prod'
        - q1v4 (bool)    : if True, analysis is trained on first and tested on 
                           last quintiles
                           default: False
        - bal (bool)     : if True, all classes are balanced
                           default: False
        - regvsurp (bool): if True, analysis is trained on regular and tested 
                           on regular sequences
                           default: False

    Returns:
        - output (str): modified output path
    """

    if runtype == 'pilot':
       output = f'{output}_pilot'

    if q1v4:
        output = f'{output}_q1v4'

    if bal:
        output = f'{output}_bal'

    if regvsurp:
        output = f'{output}_rvs'

    return output


#############################################
def run_regr(args):
    """
    run_regr(args)

    Does runs of a logistic regressions on the specified comparison and range
    of sessions.
    
    Required args:
        - args (Argument parser): parser with analysis parameters as attributes:
            alg (str)             : algorithm to use ('sklearn' or 'pytorch')
            bal (bool)            : if True, classes are balanced
            batchsize (int)       : nbr of samples dataloader will load per 
                                    batch (for 'pytorch' alg)
            bri_dir (str)         : brick direction to analyse
            bri_per (float)       : number of seconds to include before Bricks 
                                    segments
            bri_size (int or list): brick sizes to include
            comp (str)            : type of comparison
            datadir (str)         : data directory
            dend (str)            : type of dendrites to use ('aibs' or 'dend')
            device (str)          : device name (i.e., 'cuda' or 'cpu')
            ep_freq (int)         : frequency at which to print loss to 
                                    console
            error (str)           : error to take, i.e., 'std' (for std 
                                    or quintiles) or 'sem' (for SEM or MAD)
            fluor (str)           : fluorescence trace type
            fontdir (str)         : directory in which additional fonts are 
                                    located
            gabfr (int)           : gabor frame of reference if comparison 
                                    is 'surp'
            gabk (int or list)    : gabor kappas to include
            incl (str or list)    : sessions to include ('yes', 'no', 'all')
            lr (num)              : model learning rate (for 'pytorch' alg)
            mouse_n (int)         : mouse number
            n_epochs (int)        : number of epochs
            n_reg (int)           : number of regular runs
            n_shuff (int)         : number of shuffled runs
            scale (bool)          : if True, each ROI is scaled
            output (str)          : general directory in which to save 
                                    output
            parallel (bool)       : if True, runs are done in parallel
            plt_bkend (str)       : pyplot backend to use
            q1v4 (bool)           : if True, analysis is trained on first and 
                                    tested on last quintiles
            regvsurp (bool)       : if True, analysis is trained on 
                                    regular and tested on surprise sequences
            runtype (str)         : type of run ('prod' or 'pilot')
            seed (int)            : seed to seed random processes with
            sess_n (int)          : session number
            stats (str)           : stats to take, i.e., 'mean' or 'median'
            stimtype (str)        : stim to analyse ('gabors' or 'bricks')
            train_p (list)        : proportion of dataset to allocate to 
                                    training
            uniqueid (str or int) : unique ID for analysis
            wd (float)            : weight decay value (for 'pytorch' arg)
    """

    args = copy.deepcopy(args)

    if args.datadir is None: args.datadir = DEFAULT_DATADIR

    if args.uniqueid == 'datetime':
        args.uniqueid = gen_util.create_time_str()
    elif args.uniqueid in ['None', 'none']:
        args.uniqueid = None

    reseed = False
    if args.seed in [None, 'None']:
        reseed = True

    # deal with parameters
    extrapar = {'uniqueid' : args.uniqueid,
                'seed'     : args.seed
               }
    
    techpar = {'reseed'   : reseed,
               'device'   : args.device,
               'alg'      : args.alg,
               'parallel' : args.parallel,
               'plt_bkend': args.plt_bkend,
               'fontdir'  : args.fontdir,
               'output'   : args.output,
               'ep_freq'  : args.ep_freq,
               'n_reg'    : args.n_reg,
               'n_shuff'  : args.n_shuff,
               }

    mouse_df = DEFAULT_MOUSE_DF_PATH

    stimpar = logreg.get_stimpar(args.comp, args.stimtype, args.bri_dir, 
        args.bri_size, args.gabfr, args.gabk, bri_pre=args.bri_pre)
    
    analyspar = sess_ntuple_util.init_analyspar(args.fluor, stats=args.stats, 
        error=args.error, scale=not(args.no_scale), dend=args.dend)  
    
    if args.q1v4:
        quintpar = sess_ntuple_util.init_quintpar(4, [0, -1])
    else:
        quintpar = sess_ntuple_util.init_quintpar(1)
    
    logregpar = sess_ntuple_util.init_logregpar(args.comp, not(args.not_ctrl), 
        args.q1v4, args.regvsurp, args.n_epochs, args.batchsize, args.lr, 
        args.train_p, args.wd, args.bal, args.alg)
    
    omit_sess, omit_mice = sess_gen_util.all_omit(stimpar.stimtype, 
        args.runtype, stimpar.bri_dir, stimpar.bri_size, stimpar.gabk)

    sessids = sess_gen_util.get_sess_vals(mouse_df, 'sessid', args.mouse_n, 
        args.sess_n, args.runtype, incl=args.incl, omit_sess=omit_sess, 
        omit_mice=omit_mice)

    if len(sessids) == 0:
        print(f'No sessions found (mouse: {args.mouse_n}, sess: {args.sess_n}, '
            f'runtype: {args.runtype})')

    for sessid in sessids:
        sess = sess_gen_util.init_sessions(sessid, args.datadir, mouse_df, 
            args.runtype, fulldict=False, dend=analyspar.dend)[0]
        logreg.run_regr(sess, analyspar, stimpar, logregpar, quintpar, 
            extrapar, techpar)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', 
        default=os.path.join('results', 'logreg_models'),
        help='where to store output')
    parser.add_argument('--datadir', default=None, 
        help=('data directory (if None, uses a directory defined below'))
    parser.add_argument('--task', default='run_regr', 
        help='run_regr, analyse or plot')

        # technical parameters
    parser.add_argument('--plt_bkend', default=None, 
        help='switch mpl backend when running on server')
    parser.add_argument('--parallel', action='store_true', 
        help='do runs in parallel.')
    parser.add_argument('--cuda', action='store_true', 
        help='run on cuda.')
    parser.add_argument('--ep_freq', default=50, type=int,  
        help='epoch frequency at which to print loss')
    parser.add_argument('--n_reg', default=50, type=int, help='n regular runs')
    parser.add_argument('--n_shuff', default=50, type=int, 
        help='n shuffled runs')

        # logregpar
    parser.add_argument('--comp', default='surp', 
        help='surp, AvB, AvC, BvC, DvE, Eori, dir_all, dir_reg, dir_surp, '
            'half_right, half_left, half_diff')
    parser.add_argument('--not_ctrl', action='store_true', 
        help=('run comparisons not as controls for surp (ignored for surp)'))
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--batchsize', default=200, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, 
        help='learning rate')
    parser.add_argument('--train_p', default=0.75, type=float, 
        help='proportion of dataset used in training set')
    parser.add_argument('--wd', default=0, type=float, 
        help='weight decay to use')
    parser.add_argument('--q1v4', action='store_true', 
        help='run on 1st quintile and test on last')
    parser.add_argument('--regvsurp', action='store_true', 
        help='use with dir_reg to run on reg and test on surp')
    parser.add_argument('--bal', action='store_true', 
        help='if True, classes are balanced')
    parser.add_argument('--alg', default='sklearn', 
        help='use sklearn or pytorch log reg.')

        # sesspar
    parser.add_argument('--mouse_n', default=1, type=int)
    parser.add_argument('--runtype', default='prod', help='prod or pilot')
    parser.add_argument('--sess_n', default='all')
    parser.add_argument('--incl', default='any',
        help='include only `yes`, `no` or `any`')
        # stimpar
    parser.add_argument('--stimtype', default='gabors', help='gabors or bricks')
    parser.add_argument('--gabk', default=16, type=int, 
        help='gabor kappa parameter')
    parser.add_argument('--gabfr', default=0, type=int, 
        help='starting gab frame if comp is surp')
    parser.add_argument('--bri_dir', default='both', help='brick direction')
    parser.add_argument('--bri_size', default=128, help='brick size')
    parser.add_argument('--bri_pre', default=0.0, type=float, help='brick pre')

        # analyspar
    parser.add_argument('--no_scale', action='store_true', 
        help='do not scale each roi')
    parser.add_argument('--fluor', default='dff', help='raw or dff')
    parser.add_argument('--stats', default='mean', help='mean or median')
    parser.add_argument('--error', default='sem', help='std or sem')
    parser.add_argument('--dend', default='extr', help='aibs, extr')

        # extra parameters
    parser.add_argument('--seed', default=-1, type=int, 
        help='manual seed (-1 for None)')
    parser.add_argument('--uniqueid', default='datetime', 
        help=('passed string, `datetime` for date and time '
            'or `none` for no uniqueid'))

        # CI parameter for analyse and plot tasks
    parser.add_argument('--CI', default=0.95, type=float, help='shuffled CI')

        # from dict
    parser.add_argument('--dict_path', default='', 
        help=('path to info dictionary directories from which '
            'to plot data.'))
        # plot modif
    parser.add_argument('--modif', action='store_true', 
        help=('run plot task using modified plots.'))

    args = parser.parse_args()

    args.device = gen_util.get_device(args.cuda)
    args.fontdir = DEFAULT_FONTDIR


    if args.comp == 'all':
        comps = get_comps(args.stimtype, args.q1v4, args.regvsurp)
    else:
        check_args(args.comp, args.stimtype, args.q1v4, args.regvsurp)
        comps = gen_util.list_if_not(args.comp)

    args.output = format_output(
        args.output, args.runtype, args.q1v4, args.bal, args.regvsurp)

    args_orig = copy.deepcopy(args)

    if args.dict_path != '':
        plot_dicts.plot_from_dicts(
            args.dict_path, source='logreg', plt_bkend=args.plt_bkend, 
            fontdir=args.fontdir, parallel=args.parallel)

    else:
        for comp in comps:
            args = copy.deepcopy(args_orig)
            args.comp = comp
            args.not_ctrl = not(set_ctrl(not(args.not_ctrl), comp=args.comp))

            print(f'\nTask: {args.task}\nStim: {args.stimtype} '
                f'\nComparison: {args.comp}\n')

            if args.task == 'run_regr':
                run_regr(args)

            # collates regression runs and analyses accuracy
            elif args.task == 'analyse':
                print(f'Folder: {args.output}')
                logreg.run_analysis(
                    args.output, args.stimtype, args.comp, not(args.not_ctrl), 
                    args.CI, args.alg, args.parallel)

            elif args.task == 'plot':
                logreg.run_plot(
                    args.output, args.stimtype, args.comp, not(args.not_ctrl), 
                    args.bri_dir, args.fluor, not(args.no_scale), args.CI, 
                    args.alg, args.plt_bkend, args.fontdir, args.modif)

            else:
                gen_util.accepted_values_error('args.task', args.task, 
                    ['run_regr', 'analyse', 'plot'])

