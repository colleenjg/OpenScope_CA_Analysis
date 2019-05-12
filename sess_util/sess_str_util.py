"""
str_util.py

This module contains basic math functions for getting strings to print or save
files for AIBS experiments for the Credit Assignment Project.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 2.7.

"""

import datetime

from util import gen_util


#############################################
def shuff_par_str(shuffle=True, str_type='file'):
    """
    shuff_par_str()

    Returns string from shuffle parameter to print or for a filename.

    Optional args:
        - shuffle (bool): default: True
        - str_type (str): 'print' for a printable string and 'file' for a
                          string usable in a filename.
                          default: 'file'
    
    Returns:
        - shuff_str (str): shuffle parameter string
    """

    if shuffle:
        if str_type == 'print':
            shuff_str = ', shuffled'
        elif str_type == 'file':
            shuff_str = '_shuffled'
        elif str_type == 'labels':
            shuff_str = ' (shuffled labels)'
        else:
            gen_util.accepted_values_error('str_type', str_type, 
                                           ['print', 'file'])
    else:
        shuff_str = ''

    return shuff_str
    
    
#############################################
def scale_par_str(scale=True, str_type='file'):
    """
    scale_par_str()

    Returns a string from scaling parameter to print or for a filename.

    Optional args:
        - scale (str or bool): if scaling is used or type of scaling used 
                               (e.g., 'roi', 'all', 'none')
                               default: None
        - str_type (str)     : 'print' for a printable string and 'file' for a
                               string usable in a filename.
                               default: 'file'
    
    Returns:
        - scale_str (str): scale parameter string
    """

    if scale not in ['None', 'none'] and scale:
        if str_type == 'print':
            scale_str = ' (scaled)'
        elif str_type == 'file':
            scale_str = '_scaled'
        else:
            gen_util.accepted_values_error('str_type', str_type, 
                                           ['print', 'file'])
    else:
        scale_str = ''

    return scale_str
    

#############################################
def fluor_par_str(fluor='dff', str_type='file'):
    """
    fluor_par_str()

    Returns a string from the fluorescence parameter to print or for a 
    filename.

    Optional args:
        - fluor (str)   : whether 'raw' or processed fluorescence traces 'dff'  
                          are used  
                          default: 'dff'
        - str_type (str): 'print' for a printable string and 'file' for a
                          string usable in a filename.
                          default: 'file'
    
    Returns:
        - fluor_str (str): fluorescence parameter string
    """

    if fluor == 'raw':
        if str_type == 'print':
            fluor_str = 'raw fluorescence intensity'
        elif str_type == 'file':
            fluor_str = 'raw'
        else:
            gen_util.accepted_values_error('str_type', str_type, 
                                           ['print', 'file'])
    elif fluor == 'dff':
        if str_type == 'print':
            delta = u'\u0394'
            fluor_str = u'{}F/F'.format(delta)
        elif str_type == 'file':
            fluor_str = 'dff'
        else:
            gen_util.accepted_values_error('str_type', str_type, 
                                           ['print', 'file'])
    else:
        gen_util.accepted_values_error('fluor', fluor, ['raw', 'dff'])

    return fluor_str
    

#############################################
def stat_par_str(stats='mean', error='sem', str_type='file'):
    """
    stat_par_str()

    Returns a string from statistical analysis parameters to print or for a 
    title.

    Optional args:
        - stats (str)   : 'mean' or 'median'
                          default: 'mean'
        - error (str)   : 'std' (for std or quartiles) or 'sem' (for SEM or MAD)
                          or 'None' is no error
                          default: 'sem'
        - str_type (str): use of output str, i.e., for a filename ('file') or
                          to print the info to console or for title ('print')
                          default: 'file'
    
    Returns:
        - stat_str (str): statistics combo string
    """
    if error in ['None', 'none']:
        stat_str = stats
    else:
        if str_type == 'print':
            sep = u'\u00B1' # +- symbol
        elif str_type == 'file':
            sep = '_'
        else:
            gen_util.accepted_values_error('str_type', str_type, 
                                           ['print', 'file'])

        if stats == 'mean':
            stat_str = u'{}{}{}'.format(stats, sep, error)
        elif stats == 'median':
            if error == 'std':
                stat_str = u'{}{}qu'.format(stats, sep)
            elif error == 'sem':
                stat_str = u'{}{}mad'.format(stats, sep)
            else:
                gen_util.accepted_values_error('error', error, 
                                            ['std', 'sem', 'None', 'none'])
        else:
            gen_util.accepted_values_error('stats', stats, ['mean', 'median'])
    return stat_str


#############################################
def op_par_str(plot_vals='both', op='diff', area=False, str_type='file'):
    """
    op_par_str()

    Returns a string from plot values and operation parameters to print or  
    for a title.

    Optional args:
        - plot_vals (str): 'both', 'surp' or 'reg'
        - op (str)       : 'diff', 'ratio'
                           default: 'diff'
        - area (bool)    : if True, print string indicates that 'area' is 
                           plotted, not trace
                           default: False 
        - str_type (str) : use of output str, i.e., for a filename ('file') or
                           to print the info to console ('print')
                           default: 'file'
    
    Returns:

        - op_str (str): operation type string
    """
    
    if op not in ['diff', 'ratio']:
        gen_util.accepted_values_error('op', op, ['diff', 'ratio'])
    
    if plot_vals not in ['both', 'reg', 'surp']:
        gen_util.accepted_values_error('plot_vals', plot_vals, 
                                      ['both', 'reg', 'surp'])
    
    if area:
        area_str = ' area'
    else:
        area_str = ''

    if plot_vals == 'both':
        if str_type == 'print':
            op_str = '{} in dF/F{} for surp v reg'.format(op, area_str)
        elif str_type == 'file':
            op_str = op
        else:
            gen_util.accepted_values_error('str_type', str_type, 
                                           ['print', 'file'])
    else:
        if str_type == 'print':
            op_str = 'dF/F{} for {}'.format(area_str, plot_vals)
        elif str_type == 'file':
            op_str = plot_vals
        else:
            gen_util.accepted_values_error('str_type', str_type, 
                                           ['print', 'file'])
    
    return op_str


#############################################
def gabfr_nbrs(gabfr):
    """
    gabfr_nbrs(gabfr)

    Returns the numbers corresponding to the Gabor frame letters (A, B, C, D/E).

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

    all_gabfr  = ['A', 'B', 'C', 'D', 'E', 'D/E']
    all_gabnbr = [0, 1, 2, 3, 3, 3]


    if sum([g not in all_gabfr for g in gabfr]):
        raise ValueError('Gabor frames letters include A, B, C, D and E only.')
    
    
    if gabfr_list:
        gab_nbr = [all_gabnbr[all_gabfr.index(g)] for g in gabfr]
    
    else:
        gab_nbr = all_gabnbr[all_gabfr.index(gabfr[0])]
    
    return gab_nbr


#############################################
def gabfr_letters(gabfr, surp='any'):
    """
    gabfr_letters(gabfr)

    Returns the letters corresponding to the Gabor frame numbers (0, 1, 2, 3).

    Required args:
        - gabfr (int or list): gabor frame number(s)

    Optional args:
        - surp (str, int or list): surprise values for all or each gabor frame 
                                   number. If only value, applies to all.
                                   (0, 1 or 'any')
                                   default: 'any'

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
            raise ValueError(('If passing more than one surp value, must '
                              'pass as many as gabfr.'))

    if min(gabfr) < 0 or max(gabfr) > 3:
        raise ValueError('Gabor frames are only between 0 and 3, inclusively.')

    all_gabfr = ['A', 'B', 'C', 'D/E']

    gab_letts = []
    for i, gf in enumerate(gabfr):
        if gf == 3 and surp[i] != 'any':
            gab_letts.append(all_gabfr[gf][-surp[i]]) # D or E is retained
        else:
            gab_letts.append(all_gabfr[gf])

    if not gabfr_list:
        gab_letts = gab_letts[0]
    
    return gab_letts


#############################################
def gabk_par_str(gabk, str_type='file'):
    """
    gabk_par_str(gabk)

    Returns a string with stim type, as well as kappa parameters
    (e.g., 4, 16), unless only 16 is passed.

    Required args:
        - gabk (int or list): gabor kappa parameter

    Optional args:
        - str_type (str) : use of output str, i.e., for a filename ('file') or
                           to print the info to console ('print')
                           default: 'file'

    Returns:
        - pars (str): string containing stim type (gabors) and kappa, 
                      unless only 16 is passed.
    """

    gabk = gen_util.list_if_not(gabk)
    gabk = [int(g) for g in gabk]

    if str_type == 'file':
        pars = 'gab'
    elif str_type == 'print':
        pars = 'gabors'
    else:
        gen_util.accepted_values_error('str_type', str_type, ['print', 'file'])

    if 4 in gabk:
        if len(gabk) > 1:
            if str_type == 'file':
                pars = '{}_both'.format(pars)
            elif str_type == 'print':
                pars = '{} (both)'.format(pars)
        else:
            if str_type == 'file':
                pars = '{}{}'.format(pars, gabk[0])
            elif str_type == 'print':
                pars = '{} ({})'.format(pars, gabk[0])

    return pars


#############################################
def size_par_str(size, str_type='file'):
    """
    size_par_str(size)

    Returns a string with stim type, as well as size parameters
    (e.g., 128, 256), unless only 128 is passed.

    Required args:
        - size (int or list): brick size parameter

    Optional args:
        - str_type (str) : use of output str, i.e., for a filename ('file') or
                           to print the info to console ('print')
                           default: 'file'

    Returns:
        - pars (str): string containing stim type (bricks) and size, 
                      unless only 128 is passed.
    """

    size = gen_util.list_if_not(size)
    size = [int(s) for s in size]

    if str_type == 'file':
        pars = 'bri'
    elif str_type == 'print':
        pars = 'bricks'
    else:
        gen_util.accepted_values_error('str_type', str_type, ['print', 'file'])

    if 256 in size:
        if len(size) > 1:
            if str_type == 'file':
                pars = '{}_both_siz'.format(pars)
            elif str_type == 'print':
                pars = '{} (both sizes)'.format(pars)
        else:
            if str_type == 'file':
                pars = '{}{}'.format(pars, size[0])
            elif str_type == 'print':
                pars = '{} ({})'.format(pars, size[0])

    return pars


#############################################
def dir_par_str(direc, str_type='file'):
    """
    dir_par_str(direc)

    Returns a string with stim type, as well as direction parameters
    (e.g., 'right', 'left'), unless both possible values are passed.

    Required args:
        - direc (str or list): brick direction parameter

    Optional args:
        - str_type (str) : use of output str, i.e., for a filename ('file') or
                           to print the info to console ('print')
                           default: 'file'

    Returns:
        - pars (str): string containing stim type (bricks) and direction, 
                      unless both possible values are passed.
    """

    direc = gen_util.list_if_not(direc)
    if str_type == 'file':
        pars = 'bri'
    elif str_type == 'print':
        pars = 'bricks'
    else:
        gen_util.accepted_values_error('str_type', str_type, ['print', 'file'])
    
    if len(direc) == 1:
        if str_type == 'file':
            pars = '{}_{}'.format(pars, direc[0])
        elif str_type == 'print':
            pars = '{} ({})'.format(pars, direc[0])
        
    return pars


#############################################
def bri_par_str(direc, size, str_type='file'):
    """
    bri_par_str()

    Returns a string with stim type, as well as size (e.g., 128, 256) and 
    direction (e.g., 'right', 'left') parameters, unless all possible bricks 
    parameters values are passed.

    Required args:
        - direc (str or list) : brick direction parameter values
        - size (int or list): brick size parameter values

    Optional args:
        - str_type (str) : use of output str, i.e., for a filename ('file') or
                           to print the info to console ('print')
                           default: 'file'

    Returns:
        - pars (str): string containing stim type (bricks) and parameter values, 
                      unless all parameter values for bricks are passed.
    """
    
    if size is None or direc is None:
        raise ValueError(('Must pass value for brick size or direction '
                          'parameter.'))

    dirstr = dir_par_str(direc, str_type=str_type)
    sizestr = size_par_str(size, str_type=str_type)
    if str_type == 'print':
        if len(dirstr) > 6: # specified direction
            if len(sizestr) > 6: # specified size
                pars = '{}, {}'.format(sizestr.replace(')', ''), 
                                    dirstr.replace('bricks (', '')) 
            else:
                pars = dirstr
        else:
            pars = sizestr
    elif str_type == 'file':
        if len(dirstr) > 3: # specified direction
            if len(sizestr) > 3:
                pars = '{}_{}'.format(sizestr, dirstr[4:])
            else:
                pars = dirstr
        else:
            pars = sizestr
    else:
        gen_util.accepted_values_error('str_type', str_type, ['print', 'file'])
    
    return pars


#############################################
def stim_par_str(stimtype='gabors', bri_dir=None, bri_size=None, gabk=None,  
                 str_type='file'):
    """
    stim_par_str(par)

    Returns a string with stim type, as well as gabor kappa or brick size and 
    direction parameters, unless all possible parameters values for the stim 
    type are passed.

    Optional args:
        - stimtype (str)        : type of stimulus
                                  default: 'gabors'
        - bri_dir (str or list) : brick direction parameter
                                  default: None
        - bri_size (int or list): brick size parameter
                                  default: None
        - gabk (int or list)    : gabor kappa parameter
                                  default: None
        - str_type (str)        : use of output str, i.e., for a filename 
                                  ('file') or to print the info to console 
                                  ('print')
                                  default: 'file'

    Returns:
        - pars (str): string containing stim type and parameter values, unless
                      all parameter values for the stim type are passed.
    """
    
    if stimtype == 'gabors':
        if gabk is None:
            raise ValueError(('If stimulus is gabors, must pass gabk '
                              'parameters.'))
        pars = gabk_par_str(gabk, str_type)
    elif stimtype == 'bricks':
        if bri_size is None or bri_dir is None:
            raise ValueError(('If stimulus is bricks, must pass direction and '
                              'size parameters.'))
        pars = bri_par_str(bri_dir, bri_size, str_type=str_type)
    else:
        gen_util.accepted_values_error('stimtype', stimtype, 
                                       ['gabors', 'bricks'])

    return pars


#############################################
def sess_par_str(sess_n, stimtype='gabors', layer='soma', bri_dir=None, 
                 bri_size=None, gabk=None, str_type='file'):
    """
    sess_par_str(sess_n)

    Returns a string from session and stimulus parameters for a filename, 
    or to print or use in a title.

    Required args:
        - sess_n (int)          : session number aimed for

    Optional args:
        - stimtype (str)        : type of stimulus
                                  default: 'gabors'
        - layer (str)           : layer ('soma', 'dend', 'L23_soma', 'L5_soma', 
                                         'L23_dend', 'L5_dend', 'L23_all', 
                                         'L5_all')
                                  default: 'soma'
        - bri_dir (str or list) : brick direction parameter
                                  default: None
        - bri_size (int or list): brick size parameter
                                  default: None
        - gabk (int or list)    : gabor kappa parameter
                                  default: None
        - str_type (str)        : use of output str, i.e., for a filename 
                                  ('file') or to print the info to console 
                                  ('print')
                                  default: 'file'
    Returns:
        - sess_str (list): string containing info on session and gabor kappa 
                           parameters
    """
    if gabk is None and (bri_size is None or bri_dir is None):
        raise ValueError(('Must pass value for gabor k parameter or brick '
                          'size and direction.'))
    elif gabk is None:
        stimtype = 'bricks'
    elif bri_size is None or bri_dir is None:
        stimtype = 'gabors'
    
    stim_str = stim_par_str(stimtype, bri_dir, bri_size, gabk, str_type)

    if str_type == 'file':
        sess_str = 'sess{}_{}_{}'.format(sess_n, stim_str, layer)
    elif str_type == 'print':
        stim_str = stim_str.replace(' (', ': ').replace(')', '')
        sess_str = '{}, session: {}, layer: {}'.format(stim_str, sess_n, layer)
    else:
        gen_util.accepted_values_error('str_type', str_type, ['file', 'print'])
    
    return sess_str
    

