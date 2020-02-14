"""
file_util.py

This module contains functions for dealing with reading and writing files.

Authors: Colleen Gillon

Date: October, 2018

Note: this code uses python 3.7.

"""

import os.path
import sys

import glob
import json
import pandas as pd
import pickle

from util import gen_util


#############################################
def add_ext(filename, filetype='pickle'):
    """
    add_ext(filename)

    Returns a file name with extension added if there wasn't already an
    extension. Only adds pickle, json or csv extensions.
 
    Required args:
        - filename (str): name of file, can include the whole directory name
                          and extension
    
    Optional args:
        - filetype (str): type of file (pickle, pkl, json, png, csv, svg, jpg).
                          Overridden if extension already in filename.
                          Can include ''
                          default: 'pickle'

    Returns:
        - filename (str): file name, including extension
        - ext (str)     : extension, including ''
    """

    _, ext = os.path.splitext(filename)

    filetype = filetype.replace('', '')

    if ext == '':
        filetypes = ['pkl', 'pickle', 'json', 'csv', 'png', 'svg', 'jpg']
        file_exts  = ['.pkl', '.pkl', '.json', '.csv', '.png', '.svg', '.jpg']
        if filetype not in filetypes:
            gen_util.accepted_values_error('filetype', filetype, filetypes)
        ext = file_exts[filetypes.index(filetype)]
        filename = f'{filename}{ext}'
    
    return filename, ext


#############################################
def loadfile(filename, fulldir='', filetype='pickle', dtype=None):
    """
    loadfile(filename)

    Returns safely opened and loaded pickle, json or csv. If the file 
    name includes the extension, it will override the filetype argument. 
 
    Required args:
        - filename (str): name of file, can include the whole directory name
                          and extension
    
    Optional args:
        - fulldir (str) : directory in which file is saed
                          default: ''
        - filetype (str): type of file (pickle, pkl, json, csv)
                          default: 'pickle'
        - dtype (str)   : datatype for csv
                          default: None

    Returns:
        - datafile (dict or pd df): loaded file
    """

    filename, ext = add_ext(filename, filetype)
    fullname = os.path.join(fulldir, filename)
    
    if os.path.exists(fullname):
        if ext == '.pkl':
            try:
                with open(fullname, 'rb') as f:
                    datafile = pickle.load(f)
            except:
                # load a python 2 pkl in python 3
                with open(fullname, 'rb') as f: 
                    datafile = pickle.load(f, encoding='latin1')
        elif ext == '.json':
            with open(fullname, 'rb') as f:
                datafile = json.load(f)
        elif ext == '.csv':
            datafile = pd.read_csv(fullname, dtype=dtype)
    else:
        raise ValueError(f'{fullname} does not exist.')

    return datafile


#############################################
def rename_files(direc, pattern, replace='', depth=0, verbose=True, 
                 dry_run=False):
    """
    rename_files(direc, pattern)

    Renames all files in a directory containing the specified pattern by 
    replacing the pattern. 
 
    Required args:
        - direc (str)  : name of the directory in which to search
        - pattern (str): pattern to replace
    
    Optional args:
        - replace (str) : string with which to replace pattern
                          default: ''
        - depth (int)   : depth at which to search for pattern
                          default: 0
        - verbose (bool): if True, print old and new names of each renamed file 
                          default: True
        - dry_run (bool : if True, runs a dry run printing old and new names
                          default: False
    """

    direc_path = os.path.join(os.path.normpath(direc), *(['*'] * depth))
    change_paths = glob.glob(f'{direc_path}*{pattern}*')


    if len(change_paths) == 0:
        print('No pattern matches found.')
        return

    if dry_run:
        print('DRY RUN ONLY')

    for change_path in change_paths:
        new_path_name = change_path.replace(pattern, replace)
        if verbose or dry_run:
            print(f'\n{change_path} -> {new_path_name}')
        if not dry_run:
            os.rename(change_path, new_path_name)

    return


#############################################
def get_unique_path(savename, fulldir='', ext=None):
    """
    get_unique_path(savename)

    Returns a unique version of savename by adding numbers if a file by the 
    same name already exists. 

    Required args:
        - savename (str): name under which to save info, can include the 
                          whole directory name and extension
   
    Optional args:
        - fulldir (str): directory to append savename to
                         default: ''
        - ext (str)    : extension to use which, if provided, overrides any
                         extension in savename
                         default: None
    
    Returns:
        - fullname (str): savename with full directory and extension, modified 
                          with a number if needed
    """

    if ext is None:
        savename, ext = os.path.splitext(savename)
    elif '.' not in ext:
        ext = f'.{ext}'

    fullname = os.path.join(fulldir, f'{savename}{ext}')
    if os.path.exists(fullname):
        savename, _ = os.path.splitext(fullname) # get only savename
        count = 1
        fullname = f'{savename}_{count}{ext}' 
        while os.path.exists(os.path.join(fulldir, fullname)):
            count += 1 
            fullname = f'{savename}_{count}{ext}'

    return fullname

#############################################
def saveinfo(saveobj, savename='info', fulldir='', save_as='pickle', 
             sort=True, overwrite=False):
    """
    saveinfo(saveobj)

    Saves dictionary or csv as a pickle, json or csv, under a specific 
    directory and optional name. If savename includes the extension, it will 
    override the save_as argument.

    Required args:
        - saveobj (dict): object to save
    
    Optional args:
        - fulldir (str)   : directory in which to save file
                            default: ''
        - savename (str)  : name under which to save info, can include the 
                            whole directory name and extension
                            default: 'info'
        - save_as (str)   : type of file to save as (pickle, pkl, json, csv).
                            Overridden if extension included in savename.
                            default: 'pickle'
        - sort (bool)     : whether to sort keys alphabetically, if saving a 
                            dictionary as .json
                            default: True
        - overwrite (bool): if False, file name is modified if needed to prevent 
                            overwriting
    """


    # create directory if it doesn't exist
    createdir(fulldir, print_dir=False)
    
    # get extension and savename
    savename, ext = add_ext(savename, save_as) 
    fullname      = os.path.join(fulldir, savename)

    # check if file aready exists, and if so, add number at end
    if not overwrite:
        fullname = get_unique_path(fullname)

    if ext == '.pkl':
        with open(fullname, 'wb') as f:
            pickle.dump(saveobj, f)
    elif ext == '.json':
        with open(fullname, 'w') as f:
            json.dump(saveobj, f, sort_keys=sort)
    elif ext == '.csv':
        saveobj.to_csv(fullname)
    
    return fullname


###############################################################################
def checkdir(dirname):
    """
    checkdir(dirname)

    Checks whether the specified directory exists and throws an error if it
    does not.
 
    Required args:
        - dirname (str): directory path
    """

    # check that the directory exists
    if not os.path.isdir(dirname):
        raise OSError(f'{dirname} either does not exist or is not a '
                       'directory')


#############################################
def createdir(dirname, unique=False, print_dir=True):
    """
    createdir(dirname)

    Creates specified directory if it does not exist, and returns final
    directory name.
 
    Required args:
        - dirname (str or list): path or hierarchical list of directories, 
                                 e.g. ['dir', 'subdir', 'subsubdir']

    Optional args:
        - unique (bool)   : if True, ensures that a new directory is created by  
                            adding a suffix, e.g. '_1' if necessary
                            default: False
        - print_dir (bool): if True, the name of the created directory is 
                            printed

    Returns:
        - dirname (str): name of new directory
    """

    # convert directory list to full path
    dirname = os.path.join(*gen_util.list_if_not(dirname))

    if unique and os.path.exists(dirname):
        i=1
        while os.path.exists(f'{dirname}_{i}'):
            i += 1
        dirname = f'{dirname}_{i}'
        os.makedirs(dirname)
    else:
        # included due to problems when running parallel scripts 
        try:
            os.makedirs(dirname)
        except OSError:
            pass

    if print_dir:
        print(f'Directory: {dirname}')

    return dirname


###############################################################################
def getfiles(dirname='', filetype='all', criteria=None):
    """
    getfiles()

    Returns a list of all files in given directory.

    Optional args:
        - dirname (str)         : directory
                                  default: ''
        - filetype (str)        : type of file to return: 'all', 'subdirs' or 
                                  'files'
                                  default: 'all'
        - criteria (list or str): criteria for including files, i.e., contains 
                                  specified strings
                                  default: None

    Returns:
        - files (list): list of files in directory
    """

    allfiles = os.listdir(dirname)

    if criteria is not None:
        criteria = gen_util.list_if_not(criteria)
        for cri in criteria:
            allfiles = [x for x in allfiles if cri in x]
    
    allfiles = [os.path.join(dirname, x) for x in allfiles]

    if filetype == 'subdirs':
        allfiles = [x for x in allfiles if os.path.isdir(x)]

    elif filetype == 'files':
        allfiles = [x for x in allfiles if not os.path.isdir(x)]

    elif filetype != 'all':
        gen_util.accepted_values_error('filetype', filetype, 
                                       ['all', 'subdirs', 'files'])
    
    return allfiles

    